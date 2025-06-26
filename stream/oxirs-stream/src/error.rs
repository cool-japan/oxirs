//! # Stream Error Types
//!
//! Error types for the streaming module.

use std::fmt;
use thiserror::Error;

/// Result type for streaming operations
pub type StreamResult<T> = Result<T, StreamError>;

/// Errors that can occur in streaming operations
#[derive(Error, Debug)]
pub enum StreamError {
    #[error("Backend error: {0}")]
    Backend(String),

    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Deserialization error: {0}")]
    Deserialization(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Topic not found: {0}")]
    TopicNotFound(String),

    #[error("Consumer group error: {0}")]
    ConsumerGroup(String),

    #[error("Offset error: {0}")]
    Offset(String),

    #[error("Timeout error: {0}")]
    Timeout(String),

    #[error("Circuit breaker open")]
    CircuitBreakerOpen,

    #[error("Feature not supported: {0}")]
    NotSupported(String),

    #[error("Other error: {0}")]
    Other(String),
}

impl From<std::io::Error> for StreamError {
    fn from(err: std::io::Error) -> Self {
        StreamError::Backend(err.to_string())
    }
}

impl From<serde_json::Error> for StreamError {
    fn from(err: serde_json::Error) -> Self {
        StreamError::Serialization(err.to_string())
    }
}

impl From<anyhow::Error> for StreamError {
    fn from(err: anyhow::Error) -> Self {
        StreamError::Other(err.to_string())
    }
}