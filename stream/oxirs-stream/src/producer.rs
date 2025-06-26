//! # Stream Producer
//!
//! Producer types and configuration for streaming backends.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Producer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProducerConfig {
    pub client_id: String,
    pub acks: AckLevel,
    pub retries: u32,
    pub batch_size: usize,
    pub linger_ms: u64,
    pub buffer_memory: usize,
    pub compression_type: CompressionType,
    pub request_timeout_ms: u64,
    pub delivery_timeout_ms: u64,
    pub enable_idempotence: bool,
    pub transactional_id: Option<String>,
}

impl Default for ProducerConfig {
    fn default() -> Self {
        Self {
            client_id: "oxirs-producer".to_string(),
            acks: AckLevel::All,
            retries: 3,
            batch_size: 16384,
            linger_ms: 10,
            buffer_memory: 33554432, // 32MB
            compression_type: CompressionType::None,
            request_timeout_ms: 30000,
            delivery_timeout_ms: 120000,
            enable_idempotence: true,
            transactional_id: None,
        }
    }
}

/// Acknowledgment level for producers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AckLevel {
    /// No acknowledgment
    None,
    /// Leader acknowledgment only
    Leader,
    /// All in-sync replicas acknowledgment
    All,
}

/// Compression types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionType {
    None,
    Gzip,
    Snappy,
    Lz4,
    Zstd,
}

/// Producer record with key and headers
#[derive(Debug, Clone)]
pub struct ProducerRecord {
    pub topic: String,
    pub partition: Option<u32>,
    pub key: Option<Vec<u8>>,
    pub value: Vec<u8>,
    pub headers: Vec<(String, Vec<u8>)>,
    pub timestamp: Option<u64>,
}