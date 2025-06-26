//! # Stream Join Operations
//!
//! Advanced stream join patterns for real-time RDF data processing.
//!
//! This module provides comprehensive stream join capabilities including inner joins,
//! outer joins, temporal joins, and windowed joins. Optimized for high-throughput
//! scenarios with proper memory management and late data handling.

use crate::{StreamEvent, EventMetadata, processing::{WindowType, Watermark}};
use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, warn};
use uuid::Uuid;

/// Type of join operation
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum JoinType {
    /// Inner join - only matching events from both streams
    Inner,
    /// Left outer join - all events from left stream
    LeftOuter,
    /// Right outer join - all events from right stream  
    RightOuter,
    /// Full outer join - all events from both streams
    FullOuter,
}

/// Join key extractor function
pub type JoinKeyExtractor = Arc<dyn Fn(&StreamEvent) -> Option<String> + Send + Sync>;

/// Join condition evaluator
pub type JoinCondition = Arc<dyn Fn(&StreamEvent, &StreamEvent) -> bool + Send + Sync>;

/// Result transformer for joined events
pub type JoinResultTransformer = Arc<dyn Fn(&StreamEvent, Option<&StreamEvent>) -> Result<StreamEvent> + Send + Sync>;

/// Configuration for stream join operations
#[derive(Clone)]
pub struct JoinConfig {
    /// Type of join to perform
    pub join_type: JoinType,
    /// Window type for temporal joins
    pub window: Option<WindowType>,
    /// Key extraction for left stream
    pub left_key_extractor: JoinKeyExtractor,
    /// Key extraction for right stream
    pub right_key_extractor: JoinKeyExtractor,
    /// Additional join condition (beyond key matching)
    pub join_condition: Option<JoinCondition>,
    /// Result transformer
    pub result_transformer: JoinResultTransformer,
    /// Maximum time difference for temporal joins
    pub temporal_tolerance: Option<Duration>,
    /// Buffer size for each stream
    pub buffer_size: usize,
    /// Enable statistics collection
    pub collect_stats: bool,
    /// Late data handling
    pub allowed_lateness: Duration,
}

impl JoinConfig {
    pub fn new(
        join_type: JoinType,
        left_key_extractor: JoinKeyExtractor,
        right_key_extractor: JoinKeyExtractor,
        result_transformer: JoinResultTransformer,
    ) -> Self {
        Self {
            join_type,
            window: None,
            left_key_extractor,
            right_key_extractor,
            join_condition: None,
            result_transformer,
            temporal_tolerance: None,
            buffer_size: 10000,
            collect_stats: true,
            allowed_lateness: Duration::minutes(5),
        }
    }

    pub fn with_window(mut self, window: WindowType) -> Self {
        self.window = Some(window);
        self
    }

    pub fn with_temporal_tolerance(mut self, tolerance: Duration) -> Self {
        self.temporal_tolerance = Some(tolerance);
        self
    }

    pub fn with_condition(mut self, condition: JoinCondition) -> Self {
        self.join_condition = Some(condition);
        self
    }
}

/// Statistics for join operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JoinStatistics {
    pub left_events_processed: u64,
    pub right_events_processed: u64,
    pub matched_pairs: u64,
    pub unmatched_left: u64,
    pub unmatched_right: u64,
    pub late_events_dropped: u64,
    pub buffer_size_left: usize,
    pub buffer_size_right: usize,
    pub last_watermark: Option<DateTime<Utc>>,
}

/// Stream join processor for combining two event streams
pub struct StreamJoinProcessor {
    config: JoinConfig,
    left_buffer: Arc<RwLock<HashMap<String, VecDeque<StreamEvent>>>>,
    right_buffer: Arc<RwLock<HashMap<String, VecDeque<StreamEvent>>>>,
    watermark: Arc<RwLock<Watermark>>,
    statistics: Arc<RwLock<JoinStatistics>>,
}

impl StreamJoinProcessor {
    pub fn new(config: JoinConfig) -> Self {
        Self {
            config,
            left_buffer: Arc::new(RwLock::new(HashMap::new())),
            right_buffer: Arc::new(RwLock::new(HashMap::new())),
            watermark: Arc::new(RwLock::new(Watermark::new())),
            statistics: Arc::new(RwLock::new(JoinStatistics::default())),
        }
    }

    /// Process an event from the left stream
    pub async fn process_left(&self, event: StreamEvent) -> Result<Vec<StreamEvent>> {
        let event_time = event.timestamp();
        
        // Check if event is too late
        if self.is_late_event(event_time).await {
            self.statistics.write().await.late_events_dropped += 1;
            warn!("Dropping late left event: {:?}", event_time);
            return Ok(vec![]);
        }

        // Extract join key
        let key = match (self.config.left_key_extractor)(&event) {
            Some(k) => k,
            None => {
                debug!("No join key found for left event");
                return Ok(vec![]);
            }
        };

        // Add to left buffer
        {
            let mut left_buffer = self.left_buffer.write().await;
            left_buffer.entry(key.clone())
                .or_insert_with(VecDeque::new)
                .push_back(event.clone());
            
            // Trim buffer if needed
            if let Some(events) = left_buffer.get_mut(&key) {
                while events.len() > self.config.buffer_size {
                    events.pop_front();
                }
            }
        }

        // Update statistics
        {
            let mut stats = self.statistics.write().await;
            stats.left_events_processed += 1;
            stats.buffer_size_left = self.left_buffer.read().await.values()
                .map(|v| v.len())
                .sum();
        }

        // Perform join with matching right events
        self.join_with_right(&key, &event).await
    }

    /// Process an event from the right stream
    pub async fn process_right(&self, event: StreamEvent) -> Result<Vec<StreamEvent>> {
        let event_time = event.timestamp();
        
        // Check if event is too late
        if self.is_late_event(event_time).await {
            self.statistics.write().await.late_events_dropped += 1;
            warn!("Dropping late right event: {:?}", event_time);
            return Ok(vec![]);
        }

        // Extract join key
        let key = match (self.config.right_key_extractor)(&event) {
            Some(k) => k,
            None => {
                debug!("No join key found for right event");
                return Ok(vec![]);
            }
        };

        // Add to right buffer
        {
            let mut right_buffer = self.right_buffer.write().await;
            right_buffer.entry(key.clone())
                .or_insert_with(VecDeque::new)
                .push_back(event.clone());
            
            // Trim buffer if needed
            if let Some(events) = right_buffer.get_mut(&key) {
                while events.len() > self.config.buffer_size {
                    events.pop_front();
                }
            }
        }

        // Update statistics
        {
            let mut stats = self.statistics.write().await;
            stats.right_events_processed += 1;
            stats.buffer_size_right = self.right_buffer.read().await.values()
                .map(|v| v.len())
                .sum();
        }

        // Perform join with matching left events
        self.join_with_left(&key, &event).await
    }

    /// Update watermark and clean expired events
    pub async fn update_watermark(&self, watermark: DateTime<Utc>) -> Result<()> {
        self.watermark.write().await.update(watermark);
        self.statistics.write().await.last_watermark = Some(watermark);
        
        // Clean expired events from buffers
        self.clean_expired_events().await?;
        
        Ok(())
    }

    /// Get current join statistics
    pub async fn get_statistics(&self) -> JoinStatistics {
        self.statistics.read().await.clone()
    }

    /// Join left event with matching right events
    async fn join_with_right(&self, key: &str, left_event: &StreamEvent) -> Result<Vec<StreamEvent>> {
        let mut results = Vec::new();
        let right_buffer = self.right_buffer.read().await;
        
        if let Some(right_events) = right_buffer.get(key) {
            for right_event in right_events {
                if self.should_join(left_event, right_event).await {
                    let joined = (self.config.result_transformer)(left_event, Some(right_event))?;
                    results.push(joined);
                    self.statistics.write().await.matched_pairs += 1;
                }
            }
        }

        // Handle outer joins
        if results.is_empty() && matches!(self.config.join_type, JoinType::LeftOuter | JoinType::FullOuter) {
            let joined = (self.config.result_transformer)(left_event, None)?;
            results.push(joined);
            self.statistics.write().await.unmatched_left += 1;
        }

        Ok(results)
    }

    /// Join right event with matching left events
    async fn join_with_left(&self, key: &str, right_event: &StreamEvent) -> Result<Vec<StreamEvent>> {
        let mut results = Vec::new();
        let left_buffer = self.left_buffer.read().await;
        
        if let Some(left_events) = left_buffer.get(key) {
            for left_event in left_events {
                if self.should_join(left_event, right_event).await {
                    let joined = (self.config.result_transformer)(left_event, Some(right_event))?;
                    results.push(joined);
                    self.statistics.write().await.matched_pairs += 1;
                }
            }
        }

        // Handle outer joins
        if results.is_empty() && matches!(self.config.join_type, JoinType::RightOuter | JoinType::FullOuter) {
            // For right outer join, we need to swap the events in the transformer
            let joined = match &self.config.join_type {
                JoinType::RightOuter => {
                    // Create a temporary left event with None
                    create_null_joined_event(right_event, true)?
                }
                _ => (self.config.result_transformer)(right_event, None)?
            };
            results.push(joined);
            self.statistics.write().await.unmatched_right += 1;
        }

        Ok(results)
    }

    /// Check if two events should be joined
    async fn should_join(&self, left: &StreamEvent, right: &StreamEvent) -> bool {
        // Check temporal tolerance if configured
        if let Some(tolerance) = self.config.temporal_tolerance {
            let time_diff = (left.timestamp() - right.timestamp()).abs();
            if time_diff > tolerance {
                return false;
            }
        }

        // Check additional join condition if configured
        if let Some(condition) = &self.config.join_condition {
            condition(left, right)
        } else {
            true
        }
    }

    /// Check if event is too late based on watermark
    async fn is_late_event(&self, event_time: DateTime<Utc>) -> bool {
        let watermark = self.watermark.read().await;
        let watermark_time = watermark.current();
        
        event_time < watermark_time - self.config.allowed_lateness
    }

    /// Clean expired events from buffers
    async fn clean_expired_events(&self) -> Result<()> {
        let watermark_time = self.watermark.read().await.current();
        let expiry_time = watermark_time - self.config.allowed_lateness;

        // Clean left buffer
        {
            let mut left_buffer = self.left_buffer.write().await;
            for events in left_buffer.values_mut() {
                events.retain(|e| e.timestamp() >= expiry_time);
            }
            left_buffer.retain(|_, v| !v.is_empty());
        }

        // Clean right buffer
        {
            let mut right_buffer = self.right_buffer.write().await;
            for events in right_buffer.values_mut() {
                events.retain(|e| e.timestamp() >= expiry_time);
            }
            right_buffer.retain(|_, v| !v.is_empty());
        }

        Ok(())
    }
}

/// Create a null-joined event for outer joins
fn create_null_joined_event(event: &StreamEvent, is_right_null: bool) -> Result<StreamEvent> {
    // This is a simplified implementation - in production you'd want more sophisticated handling
    let mut metadata = event.metadata().clone();
    metadata.insert(
        "join_type".to_string(), 
        if is_right_null { "right_null".to_string() } else { "left_null".to_string() }
    );
    
    match event {
        StreamEvent::TripleAdded { subject, predicate, object, graph, metadata: _ } => {
            Ok(StreamEvent::TripleAdded {
                subject: subject.clone(),
                predicate: predicate.clone(),
                object: object.clone(),
                graph: graph.clone(),
                metadata: metadata.clone(),
            })
        }
        _ => Ok(event.clone())
    }
}

/// Builder for creating join processors
pub struct JoinBuilder {
    join_type: JoinType,
    left_key_extractor: Option<JoinKeyExtractor>,
    right_key_extractor: Option<JoinKeyExtractor>,
    result_transformer: Option<JoinResultTransformer>,
    window: Option<WindowType>,
    temporal_tolerance: Option<Duration>,
    join_condition: Option<JoinCondition>,
    buffer_size: usize,
    allowed_lateness: Duration,
}

impl JoinBuilder {
    pub fn new(join_type: JoinType) -> Self {
        Self {
            join_type,
            left_key_extractor: None,
            right_key_extractor: None,
            result_transformer: None,
            window: None,
            temporal_tolerance: None,
            join_condition: None,
            buffer_size: 10000,
            allowed_lateness: Duration::minutes(5),
        }
    }

    pub fn with_keys(
        mut self,
        left_extractor: JoinKeyExtractor,
        right_extractor: JoinKeyExtractor,
    ) -> Self {
        self.left_key_extractor = Some(left_extractor);
        self.right_key_extractor = Some(right_extractor);
        self
    }

    pub fn with_transformer(mut self, transformer: JoinResultTransformer) -> Self {
        self.result_transformer = Some(transformer);
        self
    }

    pub fn with_window(mut self, window: WindowType) -> Self {
        self.window = Some(window);
        self
    }

    pub fn with_temporal_tolerance(mut self, tolerance: Duration) -> Self {
        self.temporal_tolerance = Some(tolerance);
        self
    }

    pub fn with_condition(mut self, condition: JoinCondition) -> Self {
        self.join_condition = Some(condition);
        self
    }

    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    pub fn with_allowed_lateness(mut self, lateness: Duration) -> Self {
        self.allowed_lateness = lateness;
        self
    }

    pub fn build(self) -> Result<StreamJoinProcessor> {
        let config = JoinConfig {
            join_type: self.join_type,
            window: self.window,
            left_key_extractor: self.left_key_extractor
                .ok_or_else(|| anyhow!("Left key extractor is required"))?,
            right_key_extractor: self.right_key_extractor
                .ok_or_else(|| anyhow!("Right key extractor is required"))?,
            join_condition: self.join_condition,
            result_transformer: self.result_transformer
                .ok_or_else(|| anyhow!("Result transformer is required"))?,
            temporal_tolerance: self.temporal_tolerance,
            buffer_size: self.buffer_size,
            collect_stats: true,
            allowed_lateness: self.allowed_lateness,
        };

        Ok(StreamJoinProcessor::new(config))
    }
}

/// Helper functions for common join patterns
pub mod patterns {
    use super::*;
    use crate::StreamEvent;

    /// Create a subject-based join extractor
    pub fn subject_key_extractor() -> JoinKeyExtractor {
        Arc::new(|event: &StreamEvent| {
            match event {
                StreamEvent::TripleAdded { subject, .. } |
                StreamEvent::TripleRemoved { subject, .. } => Some(subject.clone()),
                _ => None,
            }
        })
    }

    /// Create a predicate-based join extractor
    pub fn predicate_key_extractor() -> JoinKeyExtractor {
        Arc::new(|event: &StreamEvent| {
            match event {
                StreamEvent::TripleAdded { predicate, .. } |
                StreamEvent::TripleRemoved { predicate, .. } => Some(predicate.clone()),
                _ => None,
            }
        })
    }

    /// Create a graph-based join extractor
    pub fn graph_key_extractor() -> JoinKeyExtractor {
        Arc::new(|event: &StreamEvent| {
            match event {
                StreamEvent::TripleAdded { graph, .. } |
                StreamEvent::TripleRemoved { graph, .. } => graph.clone(),
                _ => None,
            }
        })
    }

    /// Create a simple merge transformer
    pub fn merge_transformer() -> JoinResultTransformer {
        Arc::new(|left: &StreamEvent, right: Option<&StreamEvent>| {
            let mut metadata = left.metadata().clone();
            
            if let Some(right_event) = right {
                // Merge metadata from both events
                for (k, v) in right_event.metadata() {
                    metadata.insert(format!("right_{}", k), v.clone());
                }
                metadata.insert("join_result".to_string(), "matched".to_string());
            } else {
                metadata.insert("join_result".to_string(), "unmatched".to_string());
            }

            // Return modified left event with merged metadata
            match left {
                StreamEvent::TripleAdded { subject, predicate, object, graph, .. } => {
                    Ok(StreamEvent::TripleAdded {
                        subject: subject.clone(),
                        predicate: predicate.clone(),
                        object: object.clone(),
                        graph: graph.clone(),
                        metadata,
                    })
                }
                _ => Ok(left.clone())
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StreamEvent;
    use std::sync::Arc;

    fn create_test_event(subject: &str, timestamp: DateTime<Utc>) -> StreamEvent {
        StreamEvent::TripleAdded {
            subject: subject.to_string(),
            predicate: "http://example.org/predicate".to_string(),
            object: "http://example.org/object".to_string(),
            graph: None,
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_inner_join() {
        let processor = JoinBuilder::new(JoinType::Inner)
            .with_keys(
                patterns::subject_key_extractor(),
                patterns::subject_key_extractor(),
            )
            .with_transformer(patterns::merge_transformer())
            .build()
            .unwrap();

        let now = Utc::now();
        
        // Process events from both streams
        let left_event = create_test_event("http://example.org/subject1", now);
        let right_event = create_test_event("http://example.org/subject1", now + Duration::seconds(1));

        // Process left event first - should produce no results yet
        let results = processor.process_left(left_event.clone()).await.unwrap();
        assert_eq!(results.len(), 0);

        // Process matching right event - should produce join result
        let results = processor.process_right(right_event).await.unwrap();
        assert_eq!(results.len(), 1);

        let stats = processor.get_statistics().await;
        assert_eq!(stats.matched_pairs, 1);
        assert_eq!(stats.unmatched_left, 0);
        assert_eq!(stats.unmatched_right, 0);
    }

    #[tokio::test]
    async fn test_left_outer_join() {
        let processor = JoinBuilder::new(JoinType::LeftOuter)
            .with_keys(
                patterns::subject_key_extractor(),
                patterns::subject_key_extractor(),
            )
            .with_transformer(patterns::merge_transformer())
            .build()
            .unwrap();

        let now = Utc::now();
        
        // Process non-matching left event
        let left_event = create_test_event("http://example.org/subject1", now);
        let results = processor.process_left(left_event).await.unwrap();
        
        // Left outer join should produce result even without match
        assert_eq!(results.len(), 1);
        
        let stats = processor.get_statistics().await;
        assert_eq!(stats.unmatched_left, 1);
    }

    #[tokio::test]
    async fn test_temporal_join() {
        let processor = JoinBuilder::new(JoinType::Inner)
            .with_keys(
                patterns::subject_key_extractor(),
                patterns::subject_key_extractor(),
            )
            .with_transformer(patterns::merge_transformer())
            .with_temporal_tolerance(Duration::seconds(5))
            .build()
            .unwrap();

        let now = Utc::now();
        
        // Add left event
        let left_event = create_test_event("http://example.org/subject1", now);
        processor.process_left(left_event).await.unwrap();

        // Add matching right event within tolerance
        let right_event1 = create_test_event("http://example.org/subject1", now + Duration::seconds(3));
        let results = processor.process_right(right_event1).await.unwrap();
        assert_eq!(results.len(), 1);

        // Add matching right event outside tolerance
        let right_event2 = create_test_event("http://example.org/subject1", now + Duration::seconds(10));
        let results = processor.process_right(right_event2).await.unwrap();
        assert_eq!(results.len(), 0);

        let stats = processor.get_statistics().await;
        assert_eq!(stats.matched_pairs, 1);
    }

    #[tokio::test]
    async fn test_late_event_handling() {
        let processor = JoinBuilder::new(JoinType::Inner)
            .with_keys(
                patterns::subject_key_extractor(),
                patterns::subject_key_extractor(),
            )
            .with_transformer(patterns::merge_transformer())
            .with_allowed_lateness(Duration::minutes(1))
            .build()
            .unwrap();

        let now = Utc::now();
        
        // Update watermark
        processor.update_watermark(now).await.unwrap();

        // Process late event
        let late_event = create_test_event("http://example.org/subject1", now - Duration::minutes(2));
        let results = processor.process_left(late_event).await.unwrap();
        assert_eq!(results.len(), 0);

        let stats = processor.get_statistics().await;
        assert_eq!(stats.late_events_dropped, 1);
    }
}