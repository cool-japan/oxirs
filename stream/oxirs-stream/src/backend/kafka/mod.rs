//! Apache Kafka backend module
//!
//! This module provides comprehensive Kafka integration for OxiRS streaming,
//! organized into specialized sub-modules for better maintainability.

pub mod backend;
pub mod backend_consumer;
pub mod backend_producer;
pub mod backend_trait;
pub mod backend_types;
pub mod config;
pub mod message;

#[cfg(test)]
mod backend_tests;

// Re-export commonly used types
pub use backend::{
    ConsumerId, ConsumerMetrics, KafkaBackend, MessageCallback, PartitionAssignment,
};
pub use config::{
    KafkaAcks, KafkaCompressionType, KafkaProducerConfig, KafkaProducerStats, KafkaSecurityConfig,
    PartitionStrategy, SaslConfig, SaslMechanism, SchemaRegistryConfig, SecurityProtocol,
    SslConfig,
};
pub use message::KafkaEvent;

// Type alias for producer (KafkaBackend acts as the producer)
pub type KafkaProducer = KafkaBackend;

// Type alias for consumer (KafkaBackend acts as the consumer manager)
pub type KafkaConsumer = KafkaBackend;
