//! # Reliability Module
//!
//! Provides reliability guarantees for message producers including:
//! - At-least-once delivery guarantee
//! - Exactly-once semantics via idempotent publishing
//! - Message deduplication
//! - Retry mechanisms with exponential backoff
//! - Dead letter queue (DLQ) support
//! - Delivery confirmations and acknowledgments

use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Mutex, RwLock, Semaphore};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::StreamEvent;

/// Delivery guarantee levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeliveryGuarantee {
    /// Messages may be lost but will not be duplicated
    AtMostOnce,
    /// Messages will not be lost but may be duplicated
    AtLeastOnce,
    /// Messages will be delivered exactly once
    ExactlyOnce,
}

/// Reliability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityConfig {
    /// Delivery guarantee level
    pub delivery_guarantee: DeliveryGuarantee,
    /// Enable message deduplication
    pub enable_deduplication: bool,
    /// Deduplication window duration
    pub deduplication_window: Duration,
    /// Maximum message retries
    pub max_retries: u32,
    /// Initial retry backoff
    pub initial_backoff: Duration,
    /// Maximum retry backoff
    pub max_backoff: Duration,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Enable exponential backoff jitter
    pub backoff_jitter: bool,
    /// Dead letter queue configuration
    pub dlq_config: Option<DlqConfig>,
    /// Message timeout for acknowledgment
    pub ack_timeout: Duration,
    /// Enable persistence for reliability state
    pub enable_persistence: bool,
    /// Maximum in-flight messages
    pub max_in_flight: usize,
    /// Enable message ordering guarantees
    pub preserve_ordering: bool,
}

impl Default for ReliabilityConfig {
    fn default() -> Self {
        Self {
            delivery_guarantee: DeliveryGuarantee::AtLeastOnce,
            enable_deduplication: true,
            deduplication_window: Duration::from_secs(300), // 5 minutes
            max_retries: 3,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            backoff_jitter: true,
            dlq_config: Some(DlqConfig::default()),
            ack_timeout: Duration::from_secs(30),
            enable_persistence: false,
            max_in_flight: 1000,
            preserve_ordering: false,
        }
    }
}

/// Dead Letter Queue configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DlqConfig {
    /// Enable DLQ
    pub enabled: bool,
    /// DLQ topic/queue name
    pub topic: String,
    /// Maximum DLQ size
    pub max_size: usize,
    /// DLQ retention duration
    pub retention: Duration,
    /// Include error details in DLQ messages
    pub include_error_details: bool,
    /// Enable message replay from DLQ
    pub enable_replay: bool,
    /// Maximum replay attempts per message
    pub max_replay_attempts: u32,
    /// Replay backoff duration
    pub replay_backoff: Duration,
    /// Replay batch size for bulk operations
    pub replay_batch_size: usize,
}

impl Default for DlqConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            topic: "oxirs-dlq".to_string(),
            max_size: 10000,
            retention: Duration::from_secs(86400 * 7), // 7 days
            include_error_details: true,
            enable_replay: true,
            max_replay_attempts: 3,
            replay_backoff: Duration::from_secs(60), // 1 minute
            replay_batch_size: 100,
        }
    }
}

/// Replay status for DLQ messages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ReplayStatus {
    /// Message is available for replay
    #[default]
    Available,
    /// Message is currently being replayed
    InProgress,
    /// Message replay completed successfully
    Succeeded,
    /// Message replay failed and cannot be retried
    Failed,
    /// Message replay is temporarily paused
    Paused,
}

/// Message wrapper with reliability metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliableMessage {
    /// Unique message ID for deduplication
    pub message_id: String,
    /// Original event
    pub event: StreamEvent,
    /// Number of retry attempts
    pub retry_count: u32,
    /// First attempt timestamp
    pub first_attempt: DateTime<Utc>,
    /// Last attempt timestamp
    pub last_attempt: DateTime<Utc>,
    /// Error history
    pub errors: Vec<String>,
    /// Message checksum for integrity
    pub checksum: Option<String>,
    /// Sequence number for ordering
    pub sequence_number: Option<u64>,
    /// Partition key for ordering within partition
    pub partition_key: Option<String>,
    /// Number of replay attempts from DLQ
    pub replay_count: u32,
    /// Last replay attempt timestamp
    pub last_replay_attempt: Option<DateTime<Utc>>,
    /// Replay status
    pub replay_status: ReplayStatus,
}

impl ReliableMessage {
    /// Create a new reliable message
    pub fn new(event: StreamEvent) -> Self {
        let now = Utc::now();
        Self {
            message_id: Uuid::new_v4().to_string(),
            event,
            retry_count: 0,
            first_attempt: now,
            last_attempt: now,
            errors: Vec::new(),
            checksum: None,
            sequence_number: None,
            partition_key: None,
            replay_count: 0,
            last_replay_attempt: None,
            replay_status: ReplayStatus::default(),
        }
    }

    /// Add error to message history
    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
        self.retry_count += 1;
        self.last_attempt = Utc::now();
    }

    /// Check if message should be retried
    pub fn should_retry(&self, max_retries: u32) -> bool {
        self.retry_count < max_retries
    }

    /// Calculate next retry delay
    pub fn next_retry_delay(&self, config: &ReliabilityConfig) -> Duration {
        let base_delay = config.initial_backoff.as_millis() as f64
            * config.backoff_multiplier.powi(self.retry_count as i32);

        let mut delay = Duration::from_millis(base_delay as u64).min(config.max_backoff);

        // Add jitter if enabled
        if config.backoff_jitter {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let jitter = rng.gen_range(0.8..1.2);
            delay = Duration::from_millis((delay.as_millis() as f64 * jitter) as u64);
        }

        delay
    }
}

/// Delivery confirmation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryConfirmation {
    /// Message ID
    pub message_id: String,
    /// Delivery status
    pub status: DeliveryStatus,
    /// Delivery timestamp
    pub timestamp: DateTime<Utc>,
    /// Backend-specific metadata
    pub metadata: HashMap<String, String>,
}

/// Delivery status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeliveryStatus {
    /// Successfully delivered
    Delivered,
    /// Failed to deliver
    Failed(String),
    /// Sent to DLQ
    DeadLettered(String),
    /// Delivery pending
    Pending,
}

/// Reliability manager for message producers
pub struct ReliabilityManager {
    config: ReliabilityConfig,
    /// Deduplication cache: message_id -> expiry time
    dedup_cache: Arc<RwLock<HashMap<String, DateTime<Utc>>>>,
    /// In-flight messages: message_id -> ReliableMessage
    in_flight: Arc<RwLock<HashMap<String, ReliableMessage>>>,
    /// Retry queue
    retry_queue: Arc<Mutex<VecDeque<ReliableMessage>>>,
    /// Dead letter queue
    dlq: Arc<Mutex<VecDeque<ReliableMessage>>>,
    /// Sequence counter for ordering
    sequence_counter: Arc<RwLock<u64>>,
    /// Acknowledgment tracking
    ack_tracker: Arc<RwLock<HashMap<String, Instant>>>,
    /// Semaphore for in-flight message limiting
    in_flight_semaphore: Arc<Semaphore>,
    /// Shutdown signal
    shutdown_tx: Option<mpsc::Sender<()>>,
    shutdown_rx: Arc<Mutex<Option<mpsc::Receiver<()>>>>,
}

impl ReliabilityManager {
    /// Create a new reliability manager
    pub fn new(config: ReliabilityConfig) -> Self {
        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);
        let in_flight_semaphore = Arc::new(Semaphore::new(config.max_in_flight));

        Self {
            config,
            dedup_cache: Arc::new(RwLock::new(HashMap::new())),
            in_flight: Arc::new(RwLock::new(HashMap::new())),
            retry_queue: Arc::new(Mutex::new(VecDeque::new())),
            dlq: Arc::new(Mutex::new(VecDeque::new())),
            sequence_counter: Arc::new(RwLock::new(0)),
            ack_tracker: Arc::new(RwLock::new(HashMap::new())),
            in_flight_semaphore,
            shutdown_tx: Some(shutdown_tx),
            shutdown_rx: Arc::new(Mutex::new(Some(shutdown_rx))),
        }
    }

    /// Start background tasks for reliability management
    pub async fn start(&self) -> Result<()> {
        // Start deduplication cache cleanup task
        self.start_dedup_cleanup_task().await;

        // Start acknowledgment timeout checker
        self.start_ack_timeout_checker().await;

        // Start retry processor
        self.start_retry_processor().await;

        info!("Reliability manager started");
        Ok(())
    }

    /// Prepare message for reliable delivery
    pub async fn prepare_message(&self, event: StreamEvent) -> Result<ReliableMessage> {
        let mut message = ReliableMessage::new(event);

        // Add sequence number if ordering is enabled
        if self.config.preserve_ordering {
            let mut counter = self.sequence_counter.write().await;
            *counter += 1;
            message.sequence_number = Some(*counter);
        }

        // Check deduplication if enabled
        if self.config.enable_deduplication {
            if self.is_duplicate(&message.message_id).await? {
                return Err(anyhow!(
                    "Duplicate message detected: {}",
                    message.message_id
                ));
            }
            self.record_message_id(&message.message_id).await?;
        }

        // Acquire in-flight permit
        let _permit = self
            .in_flight_semaphore
            .acquire()
            .await
            .map_err(|_| anyhow!("Failed to acquire in-flight permit"))?;

        // Track in-flight message
        self.in_flight
            .write()
            .await
            .insert(message.message_id.clone(), message.clone());

        // Track for acknowledgment timeout
        self.ack_tracker
            .write()
            .await
            .insert(message.message_id.clone(), Instant::now());

        Ok(message)
    }

    /// Record successful delivery
    pub async fn record_delivery(&self, message_id: &str) -> Result<()> {
        // Remove from in-flight tracking
        self.in_flight.write().await.remove(message_id);

        // Remove from ack tracker
        self.ack_tracker.write().await.remove(message_id);

        // Release in-flight permit (implicitly done when permit is dropped)

        debug!("Recorded successful delivery for message: {}", message_id);
        Ok(())
    }

    /// Record delivery failure
    pub async fn record_failure(&self, message_id: &str, error: String) -> Result<DeliveryStatus> {
        let mut in_flight = self.in_flight.write().await;

        if let Some(mut message) = in_flight.remove(message_id) {
            message.add_error(error.clone());

            if message.should_retry(self.config.max_retries) {
                // Add to retry queue
                self.retry_queue.lock().await.push_back(message);
                Ok(DeliveryStatus::Pending)
            } else {
                // Max retries exceeded, send to DLQ
                if let Some(dlq_config) = &self.config.dlq_config {
                    if dlq_config.enabled {
                        self.send_to_dlq(message).await?;
                        Ok(DeliveryStatus::DeadLettered(error))
                    } else {
                        Ok(DeliveryStatus::Failed(error))
                    }
                } else {
                    Ok(DeliveryStatus::Failed(error))
                }
            }
        } else {
            Err(anyhow!(
                "Message not found in in-flight tracking: {}",
                message_id
            ))
        }
    }

    /// Check if message is duplicate
    async fn is_duplicate(&self, message_id: &str) -> Result<bool> {
        let cache = self.dedup_cache.read().await;
        Ok(cache.contains_key(message_id))
    }

    /// Record message ID for deduplication
    async fn record_message_id(&self, message_id: &str) -> Result<()> {
        let expiry = Utc::now()
            + ChronoDuration::from_std(self.config.deduplication_window)
                .map_err(|e| anyhow!("Invalid deduplication window: {}", e))?;

        self.dedup_cache
            .write()
            .await
            .insert(message_id.to_string(), expiry);

        Ok(())
    }

    /// Send message to DLQ
    async fn send_to_dlq(&self, message: ReliableMessage) -> Result<()> {
        let mut dlq = self.dlq.lock().await;

        // Check DLQ size limit
        if let Some(dlq_config) = &self.config.dlq_config {
            if dlq.len() >= dlq_config.max_size {
                warn!("DLQ is full, dropping oldest message");
                dlq.pop_front();
            }
        }

        dlq.push_back(message.clone());
        info!(
            "Message {} sent to DLQ after {} retries",
            message.message_id, message.retry_count
        );

        Ok(())
    }

    /// Get next message from retry queue
    pub async fn get_retry_message(&self) -> Option<ReliableMessage> {
        self.retry_queue.lock().await.pop_front()
    }

    /// Get DLQ messages
    pub async fn get_dlq_messages(&self, limit: usize) -> Vec<ReliableMessage> {
        let dlq = self.dlq.lock().await;
        dlq.iter().take(limit).cloned().collect()
    }

    /// Clear DLQ
    pub async fn clear_dlq(&self) -> Result<()> {
        self.dlq.lock().await.clear();
        info!("Dead letter queue cleared");
        Ok(())
    }

    /// Get reliability statistics
    pub async fn get_stats(&self) -> ReliabilityStats {
        ReliabilityStats {
            in_flight_count: self.in_flight.read().await.len(),
            retry_queue_size: self.retry_queue.lock().await.len(),
            dlq_size: self.dlq.lock().await.len(),
            dedup_cache_size: self.dedup_cache.read().await.len(),
            total_sequences: *self.sequence_counter.read().await,
        }
    }

    /// Start deduplication cache cleanup task
    async fn start_dedup_cleanup_task(&self) {
        let cache = Arc::clone(&self.dedup_cache);
        let interval = Duration::from_secs(60); // Cleanup every minute
        let shutdown_rx = Arc::clone(&self.shutdown_rx);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                // Check for shutdown
                if let Ok(mut rx) = shutdown_rx.try_lock() {
                    if let Some(rx) = rx.as_mut() {
                        if rx.try_recv().is_ok() {
                            break;
                        }
                    }
                }

                interval_timer.tick().await;

                // Clean expired entries
                let now = Utc::now();
                let mut cache_write = cache.write().await;
                cache_write.retain(|_, expiry| *expiry > now);

                debug!(
                    "Dedup cache cleanup: {} entries remaining",
                    cache_write.len()
                );
            }
        });
    }

    /// Start acknowledgment timeout checker
    async fn start_ack_timeout_checker(&self) {
        let ack_tracker = Arc::clone(&self.ack_tracker);
        let in_flight = Arc::clone(&self.in_flight);
        let retry_queue = Arc::clone(&self.retry_queue);
        let timeout = self.config.ack_timeout;
        let shutdown_rx = Arc::clone(&self.shutdown_rx);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(Duration::from_secs(5));

            loop {
                // Check for shutdown
                if let Ok(mut rx) = shutdown_rx.try_lock() {
                    if let Some(rx) = rx.as_mut() {
                        if rx.try_recv().is_ok() {
                            break;
                        }
                    }
                }

                interval_timer.tick().await;

                let now = Instant::now();
                let mut expired_messages = Vec::new();

                // Find expired messages
                {
                    let tracker = ack_tracker.read().await;
                    for (message_id, start_time) in tracker.iter() {
                        if now.duration_since(*start_time) > timeout {
                            expired_messages.push(message_id.clone());
                        }
                    }
                }

                // Handle expired messages
                for message_id in expired_messages {
                    warn!("Message {} timed out, adding to retry queue", message_id);

                    // Remove from trackers
                    ack_tracker.write().await.remove(&message_id);

                    // Move to retry queue
                    if let Some(message) = in_flight.write().await.remove(&message_id) {
                        retry_queue.lock().await.push_back(message);
                    }
                }
            }
        });
    }

    /// Start retry processor
    async fn start_retry_processor(&self) {
        let retry_queue = Arc::clone(&self.retry_queue);
        let in_flight = Arc::clone(&self.in_flight);
        let ack_tracker = Arc::clone(&self.ack_tracker);
        let config = self.config.clone();
        let shutdown_rx = Arc::clone(&self.shutdown_rx);

        tokio::spawn(async move {
            loop {
                // Check for shutdown
                if let Ok(mut rx) = shutdown_rx.try_lock() {
                    if let Some(rx) = rx.as_mut() {
                        if rx.try_recv().is_ok() {
                            break;
                        }
                    }
                }

                // Process retry queue
                let message = retry_queue.lock().await.pop_front();

                if let Some(msg) = message {
                    // Calculate retry delay
                    let delay = msg.next_retry_delay(&config);

                    info!(
                        "Retrying message {} after {:?} (attempt {})",
                        msg.message_id,
                        delay,
                        msg.retry_count + 1
                    );

                    // Wait for retry delay
                    tokio::time::sleep(delay).await;

                    // Re-add to in-flight tracking
                    in_flight
                        .write()
                        .await
                        .insert(msg.message_id.clone(), msg.clone());

                    // Update ack tracker
                    ack_tracker
                        .write()
                        .await
                        .insert(msg.message_id.clone(), Instant::now());
                } else {
                    // No messages to retry, wait a bit
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        });
    }

    /// Shutdown reliability manager
    pub async fn shutdown(&mut self) -> Result<()> {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(()).await;
        }

        info!("Reliability manager shutdown");
        Ok(())
    }

    /// Replay a single message from DLQ by message ID
    pub async fn replay_message(&self, message_id: &str) -> Result<ReliableMessage> {
        if !self
            .config
            .dlq_config
            .as_ref()
            .map(|dlq| dlq.enable_replay)
            .unwrap_or(false)
        {
            return Err(anyhow!("Message replay is disabled"));
        }

        let mut dlq = self.dlq.lock().await;
        let position = dlq.iter().position(|msg| msg.message_id == message_id);

        if let Some(pos) = position {
            let mut message = dlq[pos].clone();

            // Check if message can be replayed
            if message.replay_status == ReplayStatus::Failed
                || message.replay_count
                    >= self
                        .config
                        .dlq_config
                        .as_ref()
                        .map(|dlq| dlq.max_replay_attempts)
                        .unwrap_or(0)
            {
                return Err(anyhow!("Message has exceeded maximum replay attempts"));
            }

            // Update replay status and metadata
            message.replay_status = ReplayStatus::InProgress;
            message.replay_count += 1;
            message.last_replay_attempt = Some(Utc::now());
            dlq[pos] = message.clone();

            info!(
                "Replaying message {} (attempt {})",
                message_id, message.replay_count
            );
            Ok(message)
        } else {
            Err(anyhow!("Message not found in DLQ: {}", message_id))
        }
    }

    /// Replay multiple messages from DLQ with optional filter
    pub async fn replay_messages_with_filter<F>(
        &self,
        filter: F,
        limit: usize,
    ) -> Result<Vec<ReliableMessage>>
    where
        F: Fn(&ReliableMessage) -> bool,
    {
        if !self
            .config
            .dlq_config
            .as_ref()
            .map(|dlq| dlq.enable_replay)
            .unwrap_or(false)
        {
            return Err(anyhow!("Message replay is disabled"));
        }

        let mut dlq = self.dlq.lock().await;
        let mut replayed_messages = Vec::new();
        let mut updated_indices = Vec::new();

        for (index, message) in dlq.iter().enumerate() {
            if replayed_messages.len() >= limit {
                break;
            }

            if filter(message)
                && message.replay_status != ReplayStatus::Failed
                && message.replay_count
                    < self
                        .config
                        .dlq_config
                        .as_ref()
                        .map(|dlq| dlq.max_replay_attempts)
                        .unwrap_or(0)
            {
                let mut updated_message = message.clone();
                updated_message.replay_status = ReplayStatus::InProgress;
                updated_message.replay_count += 1;
                updated_message.last_replay_attempt = Some(Utc::now());

                replayed_messages.push(updated_message.clone());
                updated_indices.push((index, updated_message));
            }
        }

        // Update the messages in the DLQ
        for (index, updated_message) in updated_indices {
            dlq[index] = updated_message;
        }

        info!("Replaying {} messages from DLQ", replayed_messages.len());
        Ok(replayed_messages)
    }

    /// Remove successfully replayed message from DLQ
    pub async fn remove_from_dlq(&self, message_id: &str) -> Result<()> {
        let mut dlq = self.dlq.lock().await;
        let initial_len = dlq.len();
        dlq.retain(|msg| msg.message_id != message_id);

        if dlq.len() < initial_len {
            info!(
                "Removed successfully replayed message {} from DLQ",
                message_id
            );
            Ok(())
        } else {
            Err(anyhow!("Message not found in DLQ: {}", message_id))
        }
    }

    /// Update replay status for a message in DLQ
    pub async fn update_replay_status(&self, message_id: &str, status: ReplayStatus) -> Result<()> {
        let mut dlq = self.dlq.lock().await;
        if let Some(message) = dlq.iter_mut().find(|msg| msg.message_id == message_id) {
            message.replay_status = status;
            debug!(
                "Updated replay status for message {} to {:?}",
                message_id, status
            );
            Ok(())
        } else {
            Err(anyhow!("Message not found in DLQ: {}", message_id))
        }
    }

    /// Get DLQ statistics for monitoring
    pub async fn get_dlq_stats(&self) -> DlqStats {
        let dlq = self.dlq.lock().await;

        let mut error_categories = HashMap::new();
        let mut status_counts = HashMap::new();
        let mut oldest_message_age = 0u64;
        let mut total_replay_attempts = 0u32;
        let mut size_bytes = 0u64;

        let now = Utc::now();

        for message in dlq.iter() {
            // Count error categories
            for error in &message.errors {
                *error_categories.entry(error.clone()).or_insert(0) += 1;
            }

            // Count replay statuses
            let status_key = format!("{:?}", message.replay_status);
            *status_counts.entry(status_key).or_insert(0) += 1;

            // Calculate oldest message age
            let age_ms = (now - message.first_attempt).num_milliseconds().max(0) as u64;
            oldest_message_age = oldest_message_age.max(age_ms);

            // Sum replay attempts
            total_replay_attempts += message.replay_count;

            // Estimate size (rough calculation)
            size_bytes += 1024; // Average estimate per message
        }

        let messages_count = dlq.len() as u64;
        let replay_success_rate = if total_replay_attempts > 0 {
            (status_counts.get("Succeeded").unwrap_or(&0) * 100) as f64
                / total_replay_attempts as f64
        } else {
            100.0
        };

        DlqStats {
            messages_count,
            oldest_message_age_ms: oldest_message_age,
            error_categories,
            status_counts,
            total_replay_attempts,
            replay_success_rate,
            size_bytes,
            retention_period_ms: self
                .config
                .dlq_config
                .as_ref()
                .map(|dlq| dlq.retention.as_millis() as u64)
                .unwrap_or(0),
        }
    }

    /// Bulk replay messages with batching
    pub async fn bulk_replay_messages(&self, message_ids: Vec<String>) -> Result<BulkReplayResult> {
        if !self
            .config
            .dlq_config
            .as_ref()
            .map(|dlq| dlq.enable_replay)
            .unwrap_or(false)
        {
            return Err(anyhow!("Message replay is disabled"));
        }

        let batch_size = self
            .config
            .dlq_config
            .as_ref()
            .map(|dlq| dlq.replay_batch_size)
            .unwrap_or(100);
        let mut successful = Vec::new();
        let mut failed = Vec::new();

        for chunk in message_ids.chunks(batch_size) {
            for message_id in chunk {
                match self.replay_message(message_id).await {
                    Ok(message) => successful.push(message),
                    Err(e) => failed.push((message_id.clone(), e.to_string())),
                }

                // Add backoff between replays
                tokio::time::sleep(
                    self.config
                        .dlq_config
                        .as_ref()
                        .map(|dlq| dlq.replay_backoff)
                        .unwrap_or(std::time::Duration::from_secs(1)),
                )
                .await;
            }
        }

        Ok(BulkReplayResult {
            successful_count: successful.len(),
            failed_count: failed.len(),
            successful_messages: successful,
            failed_messages: failed,
        })
    }
}

/// Dead Letter Queue statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DlqStats {
    /// Total number of messages in DLQ
    pub messages_count: u64,
    /// Age of oldest message in milliseconds
    pub oldest_message_age_ms: u64,
    /// Error categories and their counts
    pub error_categories: HashMap<String, u64>,
    /// Replay status counts
    pub status_counts: HashMap<String, u64>,
    /// Total replay attempts across all messages
    pub total_replay_attempts: u32,
    /// Success rate of replay operations (percentage)
    pub replay_success_rate: f64,
    /// Estimated DLQ size in bytes
    pub size_bytes: u64,
    /// DLQ retention period in milliseconds
    pub retention_period_ms: u64,
}

/// Result of bulk replay operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulkReplayResult {
    /// Number of successfully replayed messages
    pub successful_count: usize,
    /// Number of failed replay attempts
    pub failed_count: usize,
    /// Successfully replayed messages
    pub successful_messages: Vec<ReliableMessage>,
    /// Failed messages with error details
    pub failed_messages: Vec<(String, String)>, // (message_id, error)
}

/// Reliability statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityStats {
    pub in_flight_count: usize,
    pub retry_queue_size: usize,
    pub dlq_size: usize,
    pub dedup_cache_size: usize,
    pub total_sequences: u64,
}

/// Message publisher interface for reliable delivery
#[async_trait::async_trait]
pub trait ReliablePublisher: Send + Sync {
    /// Publish message with reliability guarantees
    async fn publish_reliable(&self, message: ReliableMessage) -> Result<DeliveryConfirmation>;

    /// Check if publisher supports idempotency
    fn supports_idempotency(&self) -> bool;

    /// Get publisher reliability capabilities
    fn reliability_capabilities(&self) -> PublisherCapabilities;
}

/// Publisher reliability capabilities
#[derive(Debug, Clone)]
pub struct PublisherCapabilities {
    pub supports_transactions: bool,
    pub supports_idempotency: bool,
    pub supports_ordering: bool,
    pub supports_partitioning: bool,
    pub max_message_size: usize,
    pub max_batch_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_reliability_manager_deduplication() {
        let config = ReliabilityConfig {
            enable_deduplication: true,
            deduplication_window: Duration::from_secs(60),
            ..Default::default()
        };

        let manager = ReliabilityManager::new(config);

        // Create test event
        let event = StreamEvent::Heartbeat {
            timestamp: Utc::now(),
            source: "test".to_string(),
            metadata: crate::event::EventMetadata::default(),
        };

        // First message should succeed
        let msg1 = manager.prepare_message(event.clone()).await.unwrap();

        // Duplicate with same ID should fail
        manager.record_message_id(&msg1.message_id).await.unwrap();
        assert!(manager.is_duplicate(&msg1.message_id).await.unwrap());
    }

    #[tokio::test]
    async fn test_retry_delay_calculation() {
        let config = ReliabilityConfig {
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(10),
            backoff_multiplier: 2.0,
            backoff_jitter: false,
            ..Default::default()
        };

        let event = StreamEvent::Heartbeat {
            timestamp: Utc::now(),
            source: "test".to_string(),
            metadata: crate::event::EventMetadata::default(),
        };

        let mut message = ReliableMessage::new(event);

        // Test exponential backoff
        assert_eq!(
            message.next_retry_delay(&config),
            Duration::from_millis(100)
        );

        message.retry_count = 1;
        assert_eq!(
            message.next_retry_delay(&config),
            Duration::from_millis(200)
        );

        message.retry_count = 2;
        assert_eq!(
            message.next_retry_delay(&config),
            Duration::from_millis(400)
        );

        // Test max backoff cap
        message.retry_count = 10;
        assert_eq!(message.next_retry_delay(&config), Duration::from_secs(10));
    }

    #[tokio::test]
    async fn test_dlq_management() {
        let config = ReliabilityConfig {
            max_retries: 1,
            dlq_config: Some(DlqConfig {
                enabled: true,
                max_size: 2,
                ..Default::default()
            }),
            ..Default::default()
        };

        let manager = ReliabilityManager::new(config);

        // Create test messages
        for i in 0..3 {
            let event = StreamEvent::Heartbeat {
                timestamp: Utc::now(),
                source: format!("test-{i}"),
                metadata: crate::event::EventMetadata::default(),
            };

            let message = ReliableMessage::new(event);
            manager.send_to_dlq(message).await.unwrap();
        }

        // Check DLQ size limit
        let dlq_messages = manager.get_dlq_messages(10).await;
        assert_eq!(dlq_messages.len(), 2);
    }
}
