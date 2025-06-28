//! # Stream Types
//!
//! Common types used throughout the streaming module.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Topic name wrapper
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TopicName(String);

impl TopicName {
    pub fn new(name: String) -> Self {
        Self(name)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TopicName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for TopicName {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Partition identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PartitionId(u32);

impl PartitionId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn value(&self) -> u32 {
        self.0
    }
}

impl fmt::Display for PartitionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Message offset
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Offset(u64);

impl Offset {
    pub fn new(offset: u64) -> Self {
        Self(offset)
    }

    pub fn value(&self) -> u64 {
        self.0
    }
}

impl fmt::Display for Offset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Stream position for seeking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamPosition {
    /// Start from the beginning
    Beginning,
    /// Start from the end
    End,
    /// Start from a specific offset
    Offset(u64),
}

/// Event metadata for tracking and provenance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMetadata {
    /// Source system or component
    pub source: String,
    /// User who triggered the event
    pub user: Option<String>,
    /// Session identifier
    pub session_id: Option<String>,
    /// Trace identifier for distributed tracing
    pub trace_id: Option<String>,
    /// Causality token for event ordering
    pub causality_token: Option<String>,
    /// Event version
    pub version: Option<String>,
}

impl Default for EventMetadata {
    fn default() -> Self {
        Self {
            source: "oxirs-stream".to_string(),
            user: None,
            session_id: None,
            trace_id: None,
            causality_token: None,
            version: Some("1.0".to_string()),
        }
    }
}
