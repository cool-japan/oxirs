//! Quarantined Apache Kafka (`rdkafka`) backend adapter for `oxirs-stream`.
//!
//! # Why this crate exists
//!
//! Under the COOLJAPAN **Pure Rust Policy v2** (purity is measured on the full
//! `--all-features` dependency closure), a *published* crate must not drag in
//! C FFI. The real Kafka backend uses [`rdkafka`], which pulls in `rdkafka-sys`
//! (a C FFI binding to `librdkafka`) and, transitively, `libz-sys`. To keep the
//! published `oxirs-stream` surface 100% Pure Rust while *preserving* the real
//! Kafka capability, the live `rdkafka`-backed backend has been **quarantined**
//! into this crate.
//!
//! This crate is `publish = false`: it never ships to crates.io, so its C FFI
//! dependency never appears in the published Pure-Rust surface. Binaries that
//! actually want a Kafka backend depend on this crate directly.
//!
//! # Relationship to `oxirs-stream`
//!
//! [`KafkaBackend`] implements [`oxirs_stream::backend::StreamBackend`] — the very
//! same trait the in-crate Memory/NATS/Redis backends implement — and is
//! constructed from [`oxirs_stream::StreamConfig`] with an
//! [`oxirs_stream::StreamBackendType::Kafka`] backend selector. Callers therefore
//! drive it through the identical trait surface, just from this adapter crate.
//!
//! [`rdkafka`]: https://docs.rs/rdkafka

pub mod backend;
pub mod backend_consumer;
pub mod backend_producer;
pub mod backend_trait;
pub mod backend_types;
pub mod config;
pub mod kafka_schema_registry;
pub mod message;

#[cfg(test)]
mod backend_tests;

// Re-export commonly used types (mirrors the former `backend::kafka` facade).
pub use backend::{
    ConsumerId, ConsumerMetrics, KafkaBackend, MessageCallback, PartitionAssignment,
};
pub use config::{
    KafkaAcks, KafkaCompressionType, KafkaProducerConfig, KafkaProducerStats, KafkaSecurityConfig,
    PartitionStrategy, SaslConfig, SaslMechanism, SchemaRegistryConfig, SecurityProtocol,
    SslConfig,
};
pub use message::KafkaEvent;

/// Type alias for producer (`KafkaBackend` acts as the producer).
pub type KafkaProducer = KafkaBackend;

/// Type alias for consumer (`KafkaBackend` acts as the consumer manager).
pub type KafkaConsumer = KafkaBackend;
