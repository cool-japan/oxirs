//! Error types for the OxiRS cluster module

use thiserror::Error;

/// Result type for cluster operations
pub type Result<T> = std::result::Result<T, ClusterError>;

/// Errors that can occur in the cluster module
#[derive(Debug, Error)]
pub enum ClusterError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Network communication error
    #[error("Network error: {0}")]
    Network(String),

    /// Storage backend error
    #[error("Storage error: {0}")]
    Storage(String),

    /// Consensus error
    #[error("Consensus error: {0}")]
    Consensus(String),

    /// Not the leader node
    #[error("Not the leader node")]
    NotLeader,

    /// Lock acquisition error
    #[error("Lock error: {0}")]
    Lock(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialize(String),

    /// Parse error
    #[error("Parse error: {0}")]
    Parse(String),

    /// Runtime error
    #[error("Runtime error: {0}")]
    Runtime(String),

    /// Byzantine fault detected
    #[error("Byzantine fault detected: {0}")]
    Byzantine(String),

    /// Shard not found
    #[error("Shard not found: {0}")]
    ShardNotFound(crate::shard::ShardId),

    /// Circuit breaker is open
    #[error("Circuit breaker is open - too many failures")]
    CircuitOpen,

    /// Compression error
    #[error("Compression error: {0}")]
    Compression(String),

    /// Encryption error
    #[error("Encryption error: {0}")]
    Encryption(String),

    /// Generic error
    #[error("{0}")]
    Other(String),
}

impl From<std::io::Error> for ClusterError {
    fn from(err: std::io::Error) -> Self {
        ClusterError::Network(err.to_string())
    }
}

impl From<serde_json::Error> for ClusterError {
    fn from(err: serde_json::Error) -> Self {
        ClusterError::Serialize(err.to_string())
    }
}

impl From<anyhow::Error> for ClusterError {
    fn from(err: anyhow::Error) -> Self {
        ClusterError::Other(err.to_string())
    }
}
