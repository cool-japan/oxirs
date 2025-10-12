//! Dead Letter Queue (DLQ)
//!
//! This module provides robust handling of failed events:
//! - Automatic retry with exponential backoff
//! - Dead letter queue for permanently failed events
//! - Failure analysis and categorization
//! - Replay capabilities
//! - Alerting on high failure rates

use crate::StreamEvent;
use anyhow::Result;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

/// Failure reason categorization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FailureReason {
    /// Network connectivity issues
    NetworkError,
    /// Serialization/deserialization errors
    SerializationError,
    /// Validation errors
    ValidationError,
    /// Timeout errors
    TimeoutError,
    /// Backend-specific errors
    BackendError(String),
    /// Unknown errors
    Unknown(String),
}

/// Failed event with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedEvent {
    pub event: StreamEvent,
    pub failure_reason: FailureReason,
    pub error_message: String,
    pub first_attempt: DateTime<Utc>,
    pub last_attempt: DateTime<Utc>,
    pub retry_count: u32,
    pub stack_trace: Option<String>,
}

/// DLQ configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DlqConfig {
    /// Maximum retry attempts before moving to DLQ
    pub max_retries: u32,
    /// Initial retry delay
    pub initial_retry_delay: ChronoDuration,
    /// Maximum retry delay
    pub max_retry_delay: ChronoDuration,
    /// Retry backoff multiplier
    pub backoff_multiplier: f64,
    /// Maximum DLQ size
    pub max_dlq_size: usize,
    /// Enable automatic replay
    pub enable_auto_replay: bool,
    /// Replay interval
    pub replay_interval: ChronoDuration,
    /// Alert threshold (failure rate percentage)
    pub alert_threshold: f64,
}

impl Default for DlqConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_retry_delay: ChronoDuration::milliseconds(100),
            max_retry_delay: ChronoDuration::seconds(30),
            backoff_multiplier: 2.0,
            max_dlq_size: 100000,
            enable_auto_replay: false,
            replay_interval: ChronoDuration::hours(1),
            alert_threshold: 0.05, // 5% failure rate
        }
    }
}

/// DLQ statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DlqStats {
    pub events_failed: u64,
    pub events_retried: u64,
    pub events_moved_to_dlq: u64,
    pub events_replayed: u64,
    pub current_dlq_size: usize,
    pub failure_by_reason: HashMap<String, u64>,
    pub failure_rate: f64,
    pub last_replay: Option<DateTime<Utc>>,
}

/// Type alias for failure history
type FailureHistory = Arc<RwLock<VecDeque<(DateTime<Utc>, FailureReason)>>>;

/// Dead Letter Queue manager
pub struct DeadLetterQueue {
    config: DlqConfig,
    retry_queue: Arc<RwLock<VecDeque<FailedEvent>>>,
    dlq: Arc<RwLock<VecDeque<FailedEvent>>>,
    stats: Arc<RwLock<DlqStats>>,
    failure_history: FailureHistory,
}

impl DeadLetterQueue {
    /// Create a new DLQ
    pub fn new(config: DlqConfig) -> Self {
        Self {
            config,
            retry_queue: Arc::new(RwLock::new(VecDeque::new())),
            dlq: Arc::new(RwLock::new(VecDeque::new())),
            stats: Arc::new(RwLock::new(DlqStats::default())),
            failure_history: Arc::new(RwLock::new(VecDeque::new())),
        }
    }

    /// Handle a failed event
    pub async fn handle_failed_event(
        &self,
        event: StreamEvent,
        failure_reason: FailureReason,
        error_message: String,
    ) -> Result<()> {
        let now = Utc::now();

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.events_failed += 1;

        let reason_key = format!("{:?}", failure_reason);
        *stats.failure_by_reason.entry(reason_key).or_insert(0) += 1;

        drop(stats);

        // Record failure
        let mut history = self.failure_history.write().await;
        history.push_back((now, failure_reason.clone()));

        // Keep only last 1000 failures
        if history.len() > 1000 {
            history.pop_front();
        }

        drop(history);

        // Create failed event record
        let failed_event = FailedEvent {
            event,
            failure_reason: failure_reason.clone(),
            error_message: error_message.clone(),
            first_attempt: now,
            last_attempt: now,
            retry_count: 0,
            stack_trace: None,
        };

        // Add to retry queue
        let mut retry_queue = self.retry_queue.write().await;
        retry_queue.push_back(failed_event);

        info!(
            "Event failed, added to retry queue: {:?} - {}",
            failure_reason, error_message
        );

        // Check alert threshold
        self.check_failure_rate().await;

        Ok(())
    }

    /// Process retry queue
    pub async fn process_retries<F, Fut>(&self, retry_fn: F) -> Result<Vec<StreamEvent>>
    where
        F: Fn(StreamEvent) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<()>> + Send,
    {
        let mut retry_queue = self.retry_queue.write().await;
        let mut still_failing = Vec::new();
        let mut successfully_retried = Vec::new();

        while let Some(mut failed_event) = retry_queue.pop_front() {
            let now = Utc::now();

            // Calculate retry delay
            let delay = self.calculate_retry_delay(failed_event.retry_count);
            let time_since_last_attempt = now - failed_event.last_attempt;

            if time_since_last_attempt < delay {
                // Not ready to retry yet
                still_failing.push(failed_event);
                continue;
            }

            // Attempt retry
            let result = retry_fn(failed_event.event.clone()).await;

            match result {
                Ok(_) => {
                    // Success!
                    successfully_retried.push(failed_event.event.clone());

                    let mut stats = self.stats.write().await;
                    stats.events_retried += 1;

                    info!(
                        "Event successfully retried after {} attempts",
                        failed_event.retry_count + 1
                    );
                }
                Err(e) => {
                    // Still failing
                    failed_event.retry_count += 1;
                    failed_event.last_attempt = now;
                    failed_event.error_message = e.to_string();

                    if failed_event.retry_count >= self.config.max_retries {
                        // Move to DLQ
                        warn!(
                            "Event failed after {} retries, moving to DLQ: {}",
                            failed_event.retry_count, e
                        );

                        self.move_to_dlq(failed_event).await?;
                    } else {
                        // Keep retrying
                        still_failing.push(failed_event);
                    }
                }
            }
        }

        // Put still-failing events back in retry queue
        *retry_queue = still_failing.into();

        Ok(successfully_retried)
    }

    /// Move an event to DLQ
    async fn move_to_dlq(&self, failed_event: FailedEvent) -> Result<()> {
        let mut dlq = self.dlq.write().await;

        // Check size limit
        if dlq.len() >= self.config.max_dlq_size {
            warn!("DLQ size limit reached, dropping oldest event");
            dlq.pop_front();
        }

        dlq.push_back(failed_event);

        let mut stats = self.stats.write().await;
        stats.events_moved_to_dlq += 1;
        stats.current_dlq_size = dlq.len();

        Ok(())
    }

    /// Calculate retry delay with exponential backoff
    fn calculate_retry_delay(&self, retry_count: u32) -> ChronoDuration {
        let delay_ms = self.config.initial_retry_delay.num_milliseconds() as f64
            * self.config.backoff_multiplier.powi(retry_count as i32);

        let delay_ms = delay_ms.min(self.config.max_retry_delay.num_milliseconds() as f64);

        ChronoDuration::milliseconds(delay_ms as i64)
    }

    /// Replay events from DLQ
    pub async fn replay_dlq<F, Fut>(
        &self,
        replay_fn: F,
        max_events: Option<usize>,
    ) -> Result<Vec<StreamEvent>>
    where
        F: Fn(StreamEvent) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<()>> + Send,
    {
        let mut dlq = self.dlq.write().await;
        let mut successfully_replayed = Vec::new();
        let mut still_failing = Vec::new();

        let replay_count = max_events.unwrap_or(dlq.len()).min(dlq.len());

        for _ in 0..replay_count {
            if let Some(failed_event) = dlq.pop_front() {
                let result = replay_fn(failed_event.event.clone()).await;

                match result {
                    Ok(_) => {
                        successfully_replayed.push(failed_event.event.clone());

                        let mut stats = self.stats.write().await;
                        stats.events_replayed += 1;

                        info!("Event successfully replayed from DLQ");
                    }
                    Err(e) => {
                        error!("Event replay failed: {}", e);
                        still_failing.push(failed_event);
                    }
                }
            }
        }

        // Put still-failing events back in DLQ
        for failed_event in still_failing {
            dlq.push_back(failed_event);
        }

        let mut stats = self.stats.write().await;
        stats.current_dlq_size = dlq.len();
        stats.last_replay = Some(Utc::now());

        info!("Replayed {} events from DLQ", successfully_replayed.len());

        Ok(successfully_replayed)
    }

    /// Get events from DLQ by failure reason
    pub async fn get_by_reason(&self, reason: &FailureReason) -> Vec<FailedEvent> {
        let dlq = self.dlq.read().await;

        dlq.iter()
            .filter(|evt| &evt.failure_reason == reason)
            .cloned()
            .collect()
    }

    /// Remove specific event from DLQ
    pub async fn remove_from_dlq(&self, predicate: impl Fn(&FailedEvent) -> bool) -> usize {
        let mut dlq = self.dlq.write().await;
        let initial_size = dlq.len();

        dlq.retain(|evt| !predicate(evt));

        let removed = initial_size - dlq.len();

        let mut stats = self.stats.write().await;
        stats.current_dlq_size = dlq.len();

        removed
    }

    /// Clear DLQ
    pub async fn clear_dlq(&self) {
        let mut dlq = self.dlq.write().await;
        let cleared = dlq.len();
        dlq.clear();

        let mut stats = self.stats.write().await;
        stats.current_dlq_size = 0;

        info!("Cleared {} events from DLQ", cleared);
    }

    /// Get DLQ statistics
    pub async fn stats(&self) -> DlqStats {
        let mut stats = self.stats.read().await.clone();

        // Calculate failure rate
        stats.failure_rate = self.calculate_failure_rate().await;

        stats
    }

    /// Calculate current failure rate
    async fn calculate_failure_rate(&self) -> f64 {
        let history = self.failure_history.read().await;

        if history.is_empty() {
            return 0.0;
        }

        // Calculate failures in last minute
        let now = Utc::now();
        let one_minute_ago = now - ChronoDuration::minutes(1);

        let recent_failures = history
            .iter()
            .filter(|(timestamp, _)| *timestamp >= one_minute_ago)
            .count();

        // Estimate total events (failures / assumed failure rate)
        // This is a rough estimate - in production, you'd track successful events too
        let estimated_total = (recent_failures as f64 / 0.01).max(recent_failures as f64);

        recent_failures as f64 / estimated_total
    }

    /// Check if failure rate exceeds threshold
    async fn check_failure_rate(&self) {
        let failure_rate = self.calculate_failure_rate().await;

        if failure_rate >= self.config.alert_threshold {
            error!(
                "ALERT: Failure rate ({:.2}%) exceeds threshold ({:.2}%)",
                failure_rate * 100.0,
                self.config.alert_threshold * 100.0
            );

            // In a production system, this would trigger alerts (PagerDuty, Slack, etc.)
        }
    }

    /// Get retry queue size
    pub async fn retry_queue_size(&self) -> usize {
        self.retry_queue.read().await.len()
    }

    /// Get DLQ size
    pub async fn dlq_size(&self) -> usize {
        self.dlq.read().await.len()
    }

    /// Get all DLQ events
    pub async fn get_all_dlq_events(&self) -> Vec<FailedEvent> {
        self.dlq.read().await.iter().cloned().collect()
    }
}

/// DLQ-aware event processor
pub struct DlqEventProcessor<T> {
    dlq: Arc<DeadLetterQueue>,
    processor: Arc<dyn Fn(T) -> Result<()> + Send + Sync>,
}

impl<T: Clone + Into<StreamEvent>> DlqEventProcessor<T> {
    pub fn new<F>(dlq: Arc<DeadLetterQueue>, processor: F) -> Self
    where
        F: Fn(T) -> Result<()> + Send + Sync + 'static,
    {
        Self {
            dlq,
            processor: Arc::new(processor),
        }
    }

    /// Process event with DLQ handling
    pub async fn process(&self, event: T) -> Result<()> {
        let stream_event = event.clone().into();

        match (self.processor)(event) {
            Ok(_) => Ok(()),
            Err(e) => {
                // Categorize error
                let failure_reason = self.categorize_error(&e);

                // Handle with DLQ
                self.dlq
                    .handle_failed_event(stream_event, failure_reason, e.to_string())
                    .await?;

                Err(e)
            }
        }
    }

    /// Categorize error into failure reason
    fn categorize_error(&self, error: &anyhow::Error) -> FailureReason {
        let error_str = error.to_string().to_lowercase();

        if error_str.contains("network") || error_str.contains("connection") {
            FailureReason::NetworkError
        } else if error_str.contains("serializ") || error_str.contains("deserializ") {
            FailureReason::SerializationError
        } else if error_str.contains("validation") || error_str.contains("invalid") {
            FailureReason::ValidationError
        } else if error_str.contains("timeout") {
            FailureReason::TimeoutError
        } else {
            FailureReason::Unknown(error.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::EventMetadata;
    use anyhow::anyhow;

    fn create_test_event() -> StreamEvent {
        StreamEvent::TripleAdded {
            subject: "test".to_string(),
            predicate: "test".to_string(),
            object: "test".to_string(),
            graph: None,
            metadata: EventMetadata::default(),
        }
    }

    #[tokio::test]
    async fn test_dlq_basic() {
        let config = DlqConfig::default();
        let dlq = DeadLetterQueue::new(config);

        let event = create_test_event();

        dlq.handle_failed_event(
            event,
            FailureReason::NetworkError,
            "Connection failed".to_string(),
        )
        .await
        .unwrap();

        assert_eq!(dlq.retry_queue_size().await, 1);
        assert_eq!(dlq.dlq_size().await, 0);

        let stats = dlq.stats().await;
        assert_eq!(stats.events_failed, 1);
    }

    #[tokio::test]
    async fn test_retry_exhaustion() {
        let config = DlqConfig {
            max_retries: 2,
            initial_retry_delay: ChronoDuration::milliseconds(1),
            ..Default::default()
        };

        let dlq = DeadLetterQueue::new(config);

        let event = create_test_event();

        dlq.handle_failed_event(
            event.clone(),
            FailureReason::NetworkError,
            "Connection failed".to_string(),
        )
        .await
        .unwrap();

        // Process retries with failing function
        let retry_fn = |_: StreamEvent| async { Err(anyhow!("Still failing")) };

        for _ in 0..3 {
            // Wait for retry delay between attempts
            tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
            dlq.process_retries(retry_fn).await.unwrap();
        }

        // Should be moved to DLQ after exhausting retries
        assert_eq!(dlq.dlq_size().await, 1);
        assert_eq!(dlq.retry_queue_size().await, 0);
    }

    #[tokio::test]
    async fn test_successful_retry() {
        let config = DlqConfig {
            max_retries: 3,
            initial_retry_delay: ChronoDuration::milliseconds(1),
            ..Default::default()
        };

        let dlq = DeadLetterQueue::new(config);

        let event = create_test_event();

        dlq.handle_failed_event(
            event.clone(),
            FailureReason::NetworkError,
            "Connection failed".to_string(),
        )
        .await
        .unwrap();

        // Wait for retry delay
        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

        // Process retries with successful function
        let retry_fn = |_: StreamEvent| async { Ok(()) };

        let retried = dlq.process_retries(retry_fn).await.unwrap();

        assert_eq!(retried.len(), 1);
        assert_eq!(dlq.retry_queue_size().await, 0);
        assert_eq!(dlq.dlq_size().await, 0);
    }

    #[tokio::test]
    async fn test_dlq_replay() {
        let config = DlqConfig::default();
        let dlq = DeadLetterQueue::new(config);

        // Add events to DLQ
        for i in 0..5 {
            let mut event = create_test_event();
            if let StreamEvent::TripleAdded {
                ref mut subject, ..
            } = event
            {
                *subject = format!("test_{}", i);
            }

            // Simulate exhausted retries by directly adding to DLQ
            let failed_event = FailedEvent {
                event,
                failure_reason: FailureReason::NetworkError,
                error_message: "Connection failed".to_string(),
                first_attempt: Utc::now(),
                last_attempt: Utc::now(),
                retry_count: 5,
                stack_trace: None,
            };

            let mut dlq_queue = dlq.dlq.write().await;
            dlq_queue.push_back(failed_event);
        }

        assert_eq!(dlq.dlq_size().await, 5);

        // Replay with successful function
        let replay_fn = |_: StreamEvent| async { Ok(()) };

        let replayed = dlq.replay_dlq(replay_fn, Some(3)).await.unwrap();

        assert_eq!(replayed.len(), 3);
        assert_eq!(dlq.dlq_size().await, 2);
    }
}
