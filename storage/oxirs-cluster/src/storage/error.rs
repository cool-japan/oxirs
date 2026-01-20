//! Storage error types

use thiserror::Error;

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Corruption detected in {file}: {message}")]
    Corruption { file: String, message: String },

    #[error("Snapshot not found")]
    SnapshotNotFound,

    #[error("Log entry not found at index {index}")]
    LogEntryNotFound { index: u64 },

    #[error("Invalid log range: {start} to {end}")]
    InvalidRange { start: u64, end: u64 },
}
