//! # NATS Streaming Backend
//!
//! NATS support for streaming RDF data.
//!
//! This module provides lightweight NATS integration for streaming
//! RDF updates with JetStream for persistence and delivery guarantees.

pub mod admin;
pub mod backend;
pub mod circuit_breaker;
pub mod compression;
pub mod config;
pub mod connection_pool;
pub mod consumer;
pub mod health_monitor;
pub mod producer;
pub mod types;
pub mod utils;

// Re-export main types for backward compatibility
pub use backend::NatsBackend;
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState};
pub use compression::{CompressionAlgorithm, CompressionConfig, CompressionManager, CompressionStats};
pub use config::{NatsConfig, NatsConsumerConfig};
pub use connection_pool::{ConnectionPool, ConnectionWrapper};
pub use consumer::NatsConsumer;
pub use health_monitor::{HealthMonitor, HealthMonitorConfig, HealthStatus};
pub use producer::NatsProducer;
pub use types::*;

#[cfg(feature = "nats")]
pub use async_nats;