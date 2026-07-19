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

    /// A distributed commit/replication protocol was invoked without a
    /// real `NetworkTransport` configured, so no genuine network
    /// acknowledgement can be obtained. Returned instead of fabricating
    /// success (see `consensus::transport::NetworkTransport`).
    #[error(
        "Distributed transport not configured for node '{node_id}': \
         supply a NetworkTransport implementation (see consensus::transport) \
         before using {protocol} across real nodes"
    )]
    DistributedTransportNotConfigured {
        /// The node that attempted a distributed operation.
        node_id: String,
        /// The protocol/component that required a transport (e.g. "Paxos", "TwoPhaseCommit", "Replication").
        protocol: String,
    },

    /// A page failed CRC32 integrity verification during a corruption scan.
    #[error("Page {page_id} failed integrity checksum verification (corrupt)")]
    CorruptPage {
        /// The page ID that failed checksum verification.
        page_id: u64,
    },

    /// A [`PageGuard`](crate::storage::PageGuard) was used after its buffer
    /// frame had been re-assigned to a different page. Returning this instead
    /// of silently reading the wrong page's bytes turns a latent buffer-pool
    /// bookkeeping bug into a detectable, loud error.
    #[error(
        "Page guard identity mismatch: guard expected page {expected} but the buffer frame \
         now holds page {actual}"
    )]
    PageIdMismatch {
        /// The page id the guard was created for.
        expected: u64,
        /// The page id currently resident in the frame.
        actual: u64,
    },
}

/// Result type for TDB operations
pub type Result<T> = std::result::Result<T, TdbError>;
