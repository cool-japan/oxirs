//! # Stream Backend Abstraction
//!
//! This module provides a unified interface for different streaming backends
//! including Kafka, NATS, Redis, Kinesis, and Pulsar.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{StreamError, StreamResult};
use crate::event::{StreamEvent, StreamEventType};
use crate::types::{EventMetadata, Offset, PartitionId, StreamPosition, TopicName};
use crate::consumer::{ConsumerConfig, ConsumerGroup};
use crate::producer::ProducerConfig;

/// Common trait for all streaming backends
#[async_trait]
pub trait StreamBackend: Send + Sync {
    /// Get the name of this backend
    fn name(&self) -> &'static str;

    /// Connect to the backend
    async fn connect(&mut self) -> StreamResult<()>;

    /// Disconnect from the backend
    async fn disconnect(&mut self) -> StreamResult<()>;

    /// Create a new topic/stream
    async fn create_topic(&self, topic: &TopicName, partitions: u32) -> StreamResult<()>;

    /// Delete a topic/stream
    async fn delete_topic(&self, topic: &TopicName) -> StreamResult<()>;

    /// List all topics/streams
    async fn list_topics(&self) -> StreamResult<Vec<TopicName>>;

    /// Send a single event
    async fn send_event(&self, topic: &TopicName, event: StreamEvent) -> StreamResult<Offset>;

    /// Send multiple events as a batch
    async fn send_batch(&self, topic: &TopicName, events: Vec<StreamEvent>) -> StreamResult<Vec<Offset>>;

    /// Receive events from a topic
    async fn receive_events(
        &self,
        topic: &TopicName,
        consumer_group: Option<&ConsumerGroup>,
        position: StreamPosition,
        max_events: usize,
    ) -> StreamResult<Vec<(StreamEvent, Offset)>>;

    /// Commit consumer offset
    async fn commit_offset(
        &self,
        topic: &TopicName,
        consumer_group: &ConsumerGroup,
        partition: PartitionId,
        offset: Offset,
    ) -> StreamResult<()>;

    /// Seek to a specific position
    async fn seek(
        &self,
        topic: &TopicName,
        consumer_group: &ConsumerGroup,
        partition: PartitionId,
        position: StreamPosition,
    ) -> StreamResult<()>;

    /// Get consumer lag information
    async fn get_consumer_lag(
        &self,
        topic: &TopicName,
        consumer_group: &ConsumerGroup,
    ) -> StreamResult<HashMap<PartitionId, u64>>;

    /// Get topic metadata
    async fn get_topic_metadata(&self, topic: &TopicName) -> StreamResult<HashMap<String, String>>;
}

/// Backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamBackendConfig {
    pub backend_type: BackendType,
    pub connection_timeout_ms: u64,
    pub retry_attempts: u32,
    pub retry_delay_ms: u64,
    pub health_check_interval_ms: u64,
}

/// Backend types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackendType {
    Kafka,
    Nats,
    Redis,
    Kinesis,
    Pulsar,
    Memory,
}

impl Default for StreamBackendConfig {
    fn default() -> Self {
        Self {
            backend_type: BackendType::Memory,
            connection_timeout_ms: 5000,
            retry_attempts: 3,
            retry_delay_ms: 100,
            health_check_interval_ms: 30000,
        }
    }
}

// Re-export backend implementations
pub mod memory;

#[cfg(feature = "redis")]
pub mod redis;

#[cfg(feature = "kafka")]
pub mod kafka;

#[cfg(feature = "nats")]
pub mod nats;

#[cfg(feature = "kinesis")]
pub mod kinesis;

#[cfg(feature = "pulsar")]
pub mod pulsar;

// TODO: Add reqwest dependency to enable this
// #[cfg(feature = "kafka")]
// pub mod kafka_schema_registry;