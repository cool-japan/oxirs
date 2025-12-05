//! Error types for OxiRS TDB storage engine

use thiserror::Error;

/// TDB error type
#[derive(Error, Debug)]
pub enum TdbError {
    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Page not found
    #[error("Page not found: {0}")]
    PageNotFound(u64),

    /// Buffer pool full
    #[error("Buffer pool full, cannot evict any page")]
    BufferPoolFull,

    /// Invalid page size
    #[error("Invalid page size: expected {expected}, got {got}")]
    InvalidPageSize {
        /// Expected page size
        expected: usize,
        /// Actual page size received
        got: usize,
    },

    /// Transaction error
    #[error("Transaction error: {0}")]
    Transaction(String),

    /// Deadlock detected
    #[error("Deadlock detected for transaction {txn_id}")]
    Deadlock {
        /// Transaction ID that encountered deadlock
        txn_id: u64,
    },

    /// Transaction not active
    #[error("Transaction {txn_id} is not active")]
    TransactionNotActive {
        /// Transaction ID that is not active
        txn_id: u64,
    },

    /// Invalid node ID
    #[error("Invalid node ID: {0}")]
    InvalidNodeId(u64),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Deserialization error
    #[error("Deserialization error: {0}")]
    Deserialization(String),

    /// Node too large to fit in page
    #[error("Node too large: {size} bytes exceeds maximum {max} bytes")]
    NodeTooLarge {
        /// Actual node size in bytes
        size: usize,
        /// Maximum allowed size in bytes
        max: usize,
    },

    /// Index error
    #[error("Index error: {0}")]
    Index(String),

    /// WAL error
    #[error("WAL error: {0}")]
    Wal(String),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// Unsupported feature or operation
    #[error("Unsupported: {0}")]
    Unsupported(String),

    /// Generic error
    #[error("{0}")]
    Other(String),
}

/// Result type for TDB operations
pub type Result<T> = std::result::Result<T, TdbError>;
