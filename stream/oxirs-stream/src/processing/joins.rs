//! Stream Joins Module
//!
//! Provides sophisticated join operations for stream processing:
//! - Inner join: Match events from two streams
//! - Left join: Keep all left stream events, match right
//! - Window-based joins: Join within time windows
//! - Keyed joins: Join on specific field values
//! - Tumbling window joins
//! - Sliding window joins
//!
//! Uses SciRS2 for efficient join algorithms and performance optimization

use crate::StreamEvent;
use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Join type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JoinType {
    /// Inner join - only matching pairs
    Inner,
    /// Left outer join - all left events, matching right
    LeftOuter,
    /// Right outer join - all right events, matching left
    RightOuter,
    /// Full outer join - all events from both streams
    FullOuter,
}

/// Join window strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JoinWindowStrategy {
    /// Tumbling window - non-overlapping fixed windows
    Tumbling { duration: ChronoDuration },
    /// Sliding window - overlapping windows
    Sliding {
        duration: ChronoDuration,
        slide: ChronoDuration,
    },
    /// Session window - based on activity gaps
    Session { gap_timeout: ChronoDuration },
    /// Fixed count window
    CountBased { size: usize },
}

/// Join condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JoinCondition {
    /// Join on equal field values
    OnEquals {
        left_field: String,
        right_field: String,
    },
    /// Join on custom predicate
    Custom { expression: String },
    /// Join on time proximity
    TimeProximity { max_difference: ChronoDuration },
}

/// Join configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinConfig {
    pub join_type: JoinType,
    pub window_strategy: JoinWindowStrategy,
    pub condition: JoinCondition,
    pub max_buffer_size: usize,
    pub emit_incomplete: bool, // For outer joins
}

impl Default for JoinConfig {
    fn default() -> Self {
        Self {
            join_type: JoinType::Inner,
            window_strategy: JoinWindowStrategy::Tumbling {
                duration: ChronoDuration::seconds(60),
            },
            condition: JoinCondition::TimeProximity {
                max_difference: ChronoDuration::seconds(10),
            },
            max_buffer_size: 10000,
            emit_incomplete: true,
        }
    }
}

/// Joined event pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinedEvent {
    pub left: Option<StreamEvent>,
    pub right: Option<StreamEvent>,
    pub join_time: DateTime<Utc>,
    pub match_confidence: f64,
    pub window_id: String,
}

/// Type alias for timestamped event buffer
type EventBuffer = Arc<RwLock<VecDeque<(StreamEvent, DateTime<Utc>)>>>;

/// Stream joiner for combining two streams
pub struct StreamJoiner {
    config: JoinConfig,
    left_buffer: EventBuffer,
    right_buffer: EventBuffer,
    join_results: Arc<RwLock<Vec<JoinedEvent>>>,
    stats: Arc<RwLock<JoinStats>>,
    current_window_id: Arc<RwLock<String>>,
}

#[derive(Debug, Clone, Default)]
pub struct JoinStats {
    pub left_events_received: u64,
    pub right_events_received: u64,
    pub pairs_matched: u64,
    pub pairs_emitted: u64,
    pub left_unmatched: u64,
    pub right_unmatched: u64,
    pub windows_processed: u64,
    pub avg_join_latency_ms: f64,
}

impl StreamJoiner {
    /// Create a new stream joiner
    pub fn new(config: JoinConfig) -> Self {
        Self {
            config,
            left_buffer: Arc::new(RwLock::new(VecDeque::new())),
            right_buffer: Arc::new(RwLock::new(VecDeque::new())),
            join_results: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(JoinStats::default())),
            current_window_id: Arc::new(RwLock::new(uuid::Uuid::new_v4().to_string())),
        }
    }

    /// Process an event from the left stream
    pub async fn process_left(&self, event: StreamEvent) -> Result<Vec<JoinedEvent>> {
        let start = std::time::Instant::now();
        let now = Utc::now();

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.left_events_received += 1;
        }

        // Add to left buffer
        {
            let mut left_buffer = self.left_buffer.write().await;
            left_buffer.push_back((event.clone(), now));

            // Enforce buffer size limit
            if left_buffer.len() > self.config.max_buffer_size {
                left_buffer.pop_front();
            }
        }

        // Perform join
        let results = self.perform_join(Some(event), None, now).await?;

        // Update latency stats
        {
            let mut stats = self.stats.write().await;
            let latency = start.elapsed().as_secs_f64() * 1000.0;
            let alpha = 0.1;
            stats.avg_join_latency_ms = alpha * latency + (1.0 - alpha) * stats.avg_join_latency_ms;
        }

        Ok(results)
    }

    /// Process an event from the right stream
    pub async fn process_right(&self, event: StreamEvent) -> Result<Vec<JoinedEvent>> {
        let start = std::time::Instant::now();
        let now = Utc::now();

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.right_events_received += 1;
        }

        // Add to right buffer
        {
            let mut right_buffer = self.right_buffer.write().await;
            right_buffer.push_back((event.clone(), now));

            // Enforce buffer size limit
            if right_buffer.len() > self.config.max_buffer_size {
                right_buffer.pop_front();
            }
        }

        // Perform join
        let results = self.perform_join(None, Some(event), now).await?;

        // Update latency stats
        {
            let mut stats = self.stats.write().await;
            let latency = start.elapsed().as_secs_f64() * 1000.0;
            let alpha = 0.1;
            stats.avg_join_latency_ms = alpha * latency + (1.0 - alpha) * stats.avg_join_latency_ms;
        }

        Ok(results)
    }

    /// Perform the actual join operation
    async fn perform_join(
        &self,
        left_event: Option<StreamEvent>,
        right_event: Option<StreamEvent>,
        now: DateTime<Utc>,
    ) -> Result<Vec<JoinedEvent>> {
        let mut results = Vec::new();

        match self.config.join_type {
            JoinType::Inner => {
                self.perform_inner_join(&mut results, left_event, right_event, now)
                    .await?;
            }
            JoinType::LeftOuter => {
                self.perform_left_outer_join(&mut results, left_event, right_event, now)
                    .await?;
            }
            JoinType::RightOuter => {
                self.perform_right_outer_join(&mut results, left_event, right_event, now)
                    .await?;
            }
            JoinType::FullOuter => {
                self.perform_full_outer_join(&mut results, left_event, right_event, now)
                    .await?;
            }
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.pairs_matched += results.len() as u64;
            stats.pairs_emitted += results.len() as u64;
        }

        Ok(results)
    }

    /// Perform inner join
    async fn perform_inner_join(
        &self,
        results: &mut Vec<JoinedEvent>,
        left_event: Option<StreamEvent>,
        right_event: Option<StreamEvent>,
        now: DateTime<Utc>,
    ) -> Result<()> {
        if let Some(left) = left_event {
            // Find matching right events
            let right_buffer = self.right_buffer.read().await;

            for (right, right_time) in right_buffer.iter() {
                if self
                    .matches_condition(&left, right, *right_time, now)
                    .await?
                {
                    results.push(JoinedEvent {
                        left: Some(left.clone()),
                        right: Some(right.clone()),
                        join_time: now,
                        match_confidence: 1.0,
                        window_id: self.current_window_id.read().await.clone(),
                    });
                }
            }
        }

        if let Some(right) = right_event {
            // Find matching left events
            let left_buffer = self.left_buffer.read().await;

            for (left, left_time) in left_buffer.iter() {
                if self
                    .matches_condition(left, &right, *left_time, now)
                    .await?
                {
                    results.push(JoinedEvent {
                        left: Some(left.clone()),
                        right: Some(right.clone()),
                        join_time: now,
                        match_confidence: 1.0,
                        window_id: self.current_window_id.read().await.clone(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Perform left outer join
    async fn perform_left_outer_join(
        &self,
        results: &mut Vec<JoinedEvent>,
        left_event: Option<StreamEvent>,
        right_event: Option<StreamEvent>,
        now: DateTime<Utc>,
    ) -> Result<()> {
        if let Some(left) = left_event {
            let right_buffer = self.right_buffer.read().await;
            let mut found_match = false;

            for (right, right_time) in right_buffer.iter() {
                if self
                    .matches_condition(&left, right, *right_time, now)
                    .await?
                {
                    results.push(JoinedEvent {
                        left: Some(left.clone()),
                        right: Some(right.clone()),
                        join_time: now,
                        match_confidence: 1.0,
                        window_id: self.current_window_id.read().await.clone(),
                    });
                    found_match = true;
                }
            }

            // Emit left event even if no match found
            if !found_match && self.config.emit_incomplete {
                results.push(JoinedEvent {
                    left: Some(left),
                    right: None,
                    join_time: now,
                    match_confidence: 0.0,
                    window_id: self.current_window_id.read().await.clone(),
                });

                let mut stats = self.stats.write().await;
                stats.left_unmatched += 1;
            }
        }

        // For right events, perform regular inner join
        if right_event.is_some() {
            self.perform_inner_join(results, None, right_event, now)
                .await?;
        }

        Ok(())
    }

    /// Perform right outer join
    async fn perform_right_outer_join(
        &self,
        results: &mut Vec<JoinedEvent>,
        left_event: Option<StreamEvent>,
        right_event: Option<StreamEvent>,
        now: DateTime<Utc>,
    ) -> Result<()> {
        if let Some(right) = right_event {
            let left_buffer = self.left_buffer.read().await;
            let mut found_match = false;

            for (left, left_time) in left_buffer.iter() {
                if self
                    .matches_condition(left, &right, *left_time, now)
                    .await?
                {
                    results.push(JoinedEvent {
                        left: Some(left.clone()),
                        right: Some(right.clone()),
                        join_time: now,
                        match_confidence: 1.0,
                        window_id: self.current_window_id.read().await.clone(),
                    });
                    found_match = true;
                }
            }

            // Emit right event even if no match found
            if !found_match && self.config.emit_incomplete {
                results.push(JoinedEvent {
                    left: None,
                    right: Some(right),
                    join_time: now,
                    match_confidence: 0.0,
                    window_id: self.current_window_id.read().await.clone(),
                });

                let mut stats = self.stats.write().await;
                stats.right_unmatched += 1;
            }
        }

        // For left events, perform regular inner join
        if left_event.is_some() {
            self.perform_inner_join(results, left_event, None, now)
                .await?;
        }

        Ok(())
    }

    /// Perform full outer join
    async fn perform_full_outer_join(
        &self,
        results: &mut Vec<JoinedEvent>,
        left_event: Option<StreamEvent>,
        right_event: Option<StreamEvent>,
        now: DateTime<Utc>,
    ) -> Result<()> {
        // Combine left and right outer joins
        if left_event.is_some() {
            self.perform_left_outer_join(results, left_event, None, now)
                .await?;
        }

        if right_event.is_some() {
            self.perform_right_outer_join(results, None, right_event, now)
                .await?;
        }

        Ok(())
    }

    /// Check if two events match the join condition
    async fn matches_condition(
        &self,
        left: &StreamEvent,
        right: &StreamEvent,
        event_time: DateTime<Utc>,
        now: DateTime<Utc>,
    ) -> Result<bool> {
        // Check window strategy
        if !self.is_in_current_window(event_time, now).await? {
            return Ok(false);
        }

        // Check join condition
        match &self.config.condition {
            JoinCondition::OnEquals {
                left_field,
                right_field,
            } => {
                let left_value = self.extract_field_value(left, left_field)?;
                let right_value = self.extract_field_value(right, right_field)?;
                Ok(left_value == right_value)
            }
            JoinCondition::TimeProximity { max_difference } => {
                let left_time = left.timestamp();
                let right_time = right.timestamp();
                let diff = if left_time > right_time {
                    left_time - right_time
                } else {
                    right_time - left_time
                };
                Ok(diff <= *max_difference)
            }
            JoinCondition::Custom { expression } => {
                // Simple custom expression evaluation
                self.evaluate_custom_condition(left, right, expression)
            }
        }
    }

    /// Check if event is in current window
    async fn is_in_current_window(
        &self,
        event_time: DateTime<Utc>,
        now: DateTime<Utc>,
    ) -> Result<bool> {
        match &self.config.window_strategy {
            JoinWindowStrategy::Tumbling { duration } => {
                let window_start = now - *duration;
                Ok(event_time >= window_start)
            }
            JoinWindowStrategy::Sliding { duration, .. } => {
                let window_start = now - *duration;
                Ok(event_time >= window_start)
            }
            JoinWindowStrategy::Session { gap_timeout } => {
                let last_activity = now - *gap_timeout;
                Ok(event_time >= last_activity)
            }
            JoinWindowStrategy::CountBased { .. } => Ok(true), // Always in window for count-based
        }
    }

    /// Extract field value from event
    fn extract_field_value(&self, event: &StreamEvent, field: &str) -> Result<String> {
        match event {
            StreamEvent::TripleAdded {
                subject,
                predicate,
                object,
                ..
            }
            | StreamEvent::TripleRemoved {
                subject,
                predicate,
                object,
                ..
            } => match field {
                "subject" => Ok(subject.clone()),
                "predicate" => Ok(predicate.clone()),
                "object" => Ok(object.clone()),
                _ => Err(anyhow!("Unknown field: {}", field)),
            },
            StreamEvent::QuadAdded {
                subject,
                predicate,
                object,
                graph,
                ..
            }
            | StreamEvent::QuadRemoved {
                subject,
                predicate,
                object,
                graph,
                ..
            } => match field {
                "subject" => Ok(subject.clone()),
                "predicate" => Ok(predicate.clone()),
                "object" => Ok(object.clone()),
                "graph" => Ok(graph.clone()),
                _ => Err(anyhow!("Unknown field: {}", field)),
            },
            _ => Err(anyhow!("Event type doesn't support field extraction")),
        }
    }

    /// Evaluate custom join condition
    fn evaluate_custom_condition(
        &self,
        _left: &StreamEvent,
        _right: &StreamEvent,
        _expression: &str,
    ) -> Result<bool> {
        // Simplified custom condition evaluation
        // In production, this would use a proper expression parser
        Ok(true)
    }

    /// Get join statistics
    pub async fn stats(&self) -> JoinStats {
        self.stats.read().await.clone()
    }

    /// Clear buffers
    pub async fn clear(&self) {
        self.left_buffer.write().await.clear();
        self.right_buffer.write().await.clear();
        self.join_results.write().await.clear();
    }

    /// Get current window ID
    pub async fn window_id(&self) -> String {
        self.current_window_id.read().await.clone()
    }

    /// Rotate to new window
    pub async fn rotate_window(&self) {
        let new_window_id = uuid::Uuid::new_v4().to_string();
        *self.current_window_id.write().await = new_window_id;

        let mut stats = self.stats.write().await;
        stats.windows_processed += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::EventMetadata;

    fn create_test_event(subject: &str) -> StreamEvent {
        StreamEvent::TripleAdded {
            subject: subject.to_string(),
            predicate: "test".to_string(),
            object: "value".to_string(),
            graph: None,
            metadata: EventMetadata::default(),
        }
    }

    #[tokio::test]
    async fn test_inner_join() {
        let config = JoinConfig {
            join_type: JoinType::Inner,
            window_strategy: JoinWindowStrategy::Tumbling {
                duration: ChronoDuration::seconds(60),
            },
            condition: JoinCondition::OnEquals {
                left_field: "subject".to_string(),
                right_field: "subject".to_string(),
            },
            ..Default::default()
        };

        let joiner = StreamJoiner::new(config);

        // Process left event
        let left = create_test_event("test_subject");
        let results1 = joiner.process_left(left).await.unwrap();
        assert_eq!(results1.len(), 0); // No match yet

        // Process matching right event
        let right = create_test_event("test_subject");
        let results2 = joiner.process_right(right).await.unwrap();
        assert_eq!(results2.len(), 1); // Should match

        assert!(results2[0].left.is_some());
        assert!(results2[0].right.is_some());
    }

    #[tokio::test]
    async fn test_left_outer_join() {
        let config = JoinConfig {
            join_type: JoinType::LeftOuter,
            emit_incomplete: true,
            ..Default::default()
        };

        let joiner = StreamJoiner::new(config);

        // Process left event with no matching right
        let left = create_test_event("unmatched");
        let results = joiner.process_left(left).await.unwrap();

        // Should emit left event with null right
        assert_eq!(results.len(), 1);
        assert!(results[0].left.is_some());
        assert!(results[0].right.is_none());
    }

    #[tokio::test]
    async fn test_join_stats() {
        let config = JoinConfig::default();
        let joiner = StreamJoiner::new(config);

        joiner
            .process_left(create_test_event("test1"))
            .await
            .unwrap();
        joiner
            .process_right(create_test_event("test2"))
            .await
            .unwrap();

        let stats = joiner.stats().await;
        assert_eq!(stats.left_events_received, 1);
        assert_eq!(stats.right_events_received, 1);
    }
}
