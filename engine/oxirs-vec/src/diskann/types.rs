//! Core types for DiskANN

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Result type for DiskANN operations
pub type DiskAnnResult<T> = Result<T, DiskAnnError>;

/// Node ID in the Vamana graph
pub type NodeId = u32;

/// Vector ID (external identifier)
pub type VectorId = String;

/// Errors that can occur in DiskANN operations
#[derive(Debug, Error, Clone, Serialize, Deserialize)]
pub enum DiskAnnError {
    #[error("Vector not found: {id}")]
    VectorNotFound { id: VectorId },

    #[error("Invalid dimension: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Index not built")]
    IndexNotBuilt,

    #[error("IO error: {message}")]
    IoError { message: String },

    #[error("Serialization error: {message}")]
    SerializationError { message: String },

    #[error("Invalid configuration: {message}")]
    InvalidConfiguration { message: String },

    #[error("Graph error: {message}")]
    GraphError { message: String },

    #[error("Storage error: {message}")]
    StorageError { message: String },

    #[error("Memory limit exceeded: {message}")]
    MemoryLimitExceeded { message: String },

    #[error("Concurrent modification detected")]
    ConcurrentModification,

    #[error("Internal error: {message}")]
    InternalError { message: String },
}

impl From<std::io::Error> for DiskAnnError {
    fn from(err: std::io::Error) -> Self {
        DiskAnnError::IoError {
            message: err.to_string(),
        }
    }
}

impl From<oxicode::Error> for DiskAnnError {
    fn from(err: oxicode::Error) -> Self {
        DiskAnnError::SerializationError {
            message: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = DiskAnnError::VectorNotFound {
            id: "vec123".to_string(),
        };
        assert!(err.to_string().contains("vec123"));

        let err = DiskAnnError::DimensionMismatch {
            expected: 128,
            actual: 256,
        };
        assert!(err.to_string().contains("128"));
        assert!(err.to_string().contains("256"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let disk_err: DiskAnnError = io_err.into();
        assert!(matches!(disk_err, DiskAnnError::IoError { .. }));
    }
}
