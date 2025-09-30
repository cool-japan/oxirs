//! # NATS Streaming Backend
//!
//! NATS support for streaming RDF data.
//!
//! This module provides lightweight NATS integration for streaming
//! RDF updates with JetStream for persistence and delivery guarantees.

pub mod circuit_breaker;
pub mod compression;
pub mod config;
pub mod connection_pool;
pub mod health_monitor;
pub mod message;
pub mod producer;
pub mod types;

// Re-export main types for backward compatibility
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState};
pub use compression::{
    CompressionAlgorithm, CompressionConfig, CompressionManager, CompressionStats,
};
pub use config::{NatsConfig, NatsConsumerConfig};
pub use connection_pool::{ConnectionPool, ConnectionWrapper};
pub use health_monitor::{HealthMonitor, HealthMonitorConfig, HealthStatus};
pub use producer::NatsProducer;
pub use types::*;

#[cfg(feature = "nats")]
pub use async_nats;
