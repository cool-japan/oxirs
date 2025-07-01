//! Apache Kafka backend module
//!
//! This module provides comprehensive Kafka integration for OxiRS streaming,
//! organized into specialized sub-modules for better maintainability.

pub mod backend;
pub mod config;
pub mod message;

// Re-export commonly used types
pub use backend::KafkaBackend;
pub use config::{
    KafkaAcks, KafkaCompressionType, KafkaProducerConfig, KafkaProducerStats, KafkaSecurityConfig,
    PartitionStrategy, SaslConfig, SaslMechanism, SchemaRegistryConfig, SecurityProtocol,
    SslConfig,
};
pub use message::KafkaEvent;
