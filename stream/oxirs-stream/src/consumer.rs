//! # Stream Consumer
//!
//! Consumer types and configuration for streaming backends.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Consumer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumerConfig {
    pub group_id: String,
    pub consumer_id: Option<String>,
    pub auto_commit: bool,
    pub commit_interval_ms: u64,
    pub max_poll_records: usize,
    pub session_timeout_ms: u64,
    pub heartbeat_interval_ms: u64,
    pub enable_auto_offset_store: bool,
    pub isolation_level: IsolationLevel,
}

impl Default for ConsumerConfig {
    fn default() -> Self {
        Self {
            group_id: "oxirs-consumer-group".to_string(),
            consumer_id: None,
            auto_commit: true,
            commit_interval_ms: 5000,
            max_poll_records: 500,
            session_timeout_ms: 30000,
            heartbeat_interval_ms: 3000,
            enable_auto_offset_store: true,
            isolation_level: IsolationLevel::ReadCommitted,
        }
    }
}

/// Consumer group representation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConsumerGroup {
    name: String,
    consumer_id: Option<String>,
}

impl ConsumerGroup {
    pub fn new(name: String) -> Self {
        Self {
            name,
            consumer_id: None,
        }
    }

    pub fn with_consumer_id(mut self, consumer_id: String) -> Self {
        self.consumer_id = Some(consumer_id);
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn consumer_id(&self) -> Option<&str> {
        self.consumer_id.as_deref()
    }
}

/// Transaction isolation levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IsolationLevel {
    ReadUncommitted,
    ReadCommitted,
}

/// Consumer position reset strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OffsetReset {
    Earliest,
    Latest,
    None,
}
