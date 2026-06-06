//! Type definitions extracted from lib.rs for 2000-line policy compliance.
//!
//! This module is the root that re-exports all types from its sibling sub-modules:
//! - `lib_types_config`   — configuration structs & StreamBackendType
//! - `lib_types_producer` — StreamProducer, ProducerStats, memory backend helpers
//! - `lib_types_consumer` — StreamConsumer, ConsumerStats
//! - `lib_types_patch`    — PatchOperation, RdfPatch, Stream (unified)

pub mod lib_types_config;
pub mod lib_types_consumer;
pub mod lib_types_patch;
pub mod lib_types_producer;

// Re-export config types
pub use lib_types_config::{
    AwsCredentials, CircuitBreakerConfig, CompressionType, MonitoringConfig, NatsJetStreamConfig,
    PulsarAuthConfig, PulsarAuthMethod, RetryConfig, SaslConfig, SaslMechanism, SecurityConfig,
    StreamBackendType, StreamConfig, StreamPerformanceConfig,
};

// Re-export producer types
pub use lib_types_producer::{
    clear_memory_events, get_memory_events, ProducerStats, StreamProducer,
};

// Re-export consumer types
pub use lib_types_consumer::{ConsumerStats, StreamConsumer};

// Re-export patch / stream types
pub use lib_types_patch::{publish_patch, PatchOperation, RdfPatch, Stream};
