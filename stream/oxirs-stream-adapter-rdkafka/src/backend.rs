//! # Apache Kafka Backend - Modular Architecture
//!
//! Complete Apache Kafka integration for enterprise-scale RDF streaming.
//!
//! This module provides comprehensive Kafka integration with transactional producers,
//! exactly-once semantics, schema registry, consumer groups, and advanced performance
//! optimizations for mission-critical RDF streaming applications.
//!
//! ## Architecture
//!
//! This module is a thin facade. The implementation is split across sibling modules:
//! - [`super::backend_types`]    — consumer identifiers, metrics, partition
//!   assignments, and the rdkafka rebalance callback context.
//! - [`super::backend_producer`] — the [`KafkaBackend`] struct, connection setup,
//!   topic creation, and producer-side publishing / statistics.
//! - [`super::backend_consumer`] — persistent/streaming consumer management,
//!   lifecycle control, offset seeking, and consumer metrics.
//! - [`super::backend_trait`]    — the [`StreamBackend`](oxirs_stream::backend::StreamBackend)
//!   trait implementation for [`KafkaBackend`].

pub use super::backend_producer::KafkaBackend;
pub use super::backend_types::{ConsumerId, ConsumerMetrics, MessageCallback, PartitionAssignment};
