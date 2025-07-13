//! Core error types for OxiRS
//!
//! This module provides the base error types that all OxiRS modules should use.
//! Module-specific errors should include this as a variant.

// Removed unused std::fmt import

/// Core error type for OxiRS operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum CoreError {
    /// Invalid parameter provided
    #[error("Invalid parameter '{name}': {message}")]
    InvalidParameter { name: String, message: String },

    /// Resource not found
    #[error("Resource not found: {0}")]
    NotFound(String),

    /// Operation not supported
    #[error("Operation not supported: {0}")]
    NotSupported(String),

    /// Dimension mismatch in vector operations
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Platform capability not available
    #[error("Platform capability not available: {0}")]
    CapabilityNotAvailable(String),

    /// Memory allocation failure
    #[error("Memory allocation failed: {0}")]
    MemoryError(String),

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Timeout error
    #[error("Operation timed out: {0}")]
    Timeout(String),

    /// Generic internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<std::io::Error> for CoreError {
    fn from(err: std::io::Error) -> Self {
        CoreError::IoError(err.to_string())
    }
}

impl From<serde_json::Error> for CoreError {
    fn from(err: serde_json::Error) -> Self {
        CoreError::SerializationError(err.to_string())
    }
}

/// Result type alias using CoreError
pub type CoreResult<T> = Result<T, CoreError>;

// Re-export the main OxiRS error types for compatibility
pub use crate::{OxirsError, Result as OxirsResult};

/// Validation functions for common parameter checks
pub mod validation {
    use super::{CoreError, CoreResult};
    use std::fmt;

    /// Check that a value is positive
    pub fn check_positive<T>(value: T, name: &str) -> CoreResult<T>
    where
        T: PartialOrd + Default + fmt::Display + Copy,
    {
        if value <= T::default() {
            Err(CoreError::InvalidParameter {
                name: name.to_string(),
                message: format!("Value must be positive, got {value}"),
            })
        } else {
            Ok(value)
        }
    }

    /// Check that a value is finite (for floating point)
    pub fn check_finite_f32(value: f32, name: &str) -> CoreResult<f32> {
        if !value.is_finite() {
            Err(CoreError::InvalidParameter {
                name: name.to_string(),
                message: format!("Value must be finite, got {value}"),
            })
        } else {
            Ok(value)
        }
    }

    /// Check that a value is finite (for floating point)
    pub fn check_finite_f64(value: f64, name: &str) -> CoreResult<f64> {
        if !value.is_finite() {
            Err(CoreError::InvalidParameter {
                name: name.to_string(),
                message: format!("Value must be finite, got {value}"),
            })
        } else {
            Ok(value)
        }
    }

    /// Check that an array contains only finite values
    pub fn check_finite_array(values: &[f32]) -> CoreResult<()> {
        for (i, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(CoreError::InvalidParameter {
                    name: format!("array[{i}]"),
                    message: format!("Value must be finite, got {value}"),
                });
            }
        }
        Ok(())
    }

    /// Check that dimensions match
    pub fn check_dimensions(expected: usize, actual: usize, _context: &str) -> CoreResult<()> {
        if expected != actual {
            Err(CoreError::DimensionMismatch { expected, actual })
        } else {
            Ok(())
        }
    }

    /// Check that a slice is not empty
    pub fn check_non_empty<T>(slice: &[T], name: &str) -> CoreResult<()> {
        if slice.is_empty() {
            Err(CoreError::InvalidParameter {
                name: name.to_string(),
                message: "Value must not be empty".to_string(),
            })
        } else {
            Ok(())
        }
    }

    /// Check that a string is not empty
    pub fn check_non_empty_str(value: &str, name: &str) -> CoreResult<()> {
        if value.is_empty() {
            Err(CoreError::InvalidParameter {
                name: name.to_string(),
                message: "Value must not be empty".to_string(),
            })
        } else {
            Ok(())
        }
    }

    /// Check that a value is within a range
    pub fn check_range<T>(value: T, min: T, max: T, name: &str) -> CoreResult<T>
    where
        T: PartialOrd + fmt::Display + Copy,
    {
        if value < min || value > max {
            Err(CoreError::InvalidParameter {
                name: name.to_string(),
                message: format!("Value must be between {min} and {max}, got {value}"),
            })
        } else {
            Ok(value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::validation::*;
    use super::*;

    #[test]
    fn test_core_error_display() {
        let error = CoreError::InvalidParameter {
            name: "test".to_string(),
            message: "test message".to_string(),
        };
        assert_eq!(error.to_string(), "Invalid parameter 'test': test message");

        let error = CoreError::NotFound("resource".to_string());
        assert_eq!(error.to_string(), "Resource not found: resource");

        let error = CoreError::DimensionMismatch {
            expected: 3,
            actual: 5,
        };
        assert_eq!(error.to_string(), "Dimension mismatch: expected 3, got 5");
    }

    #[test]
    fn test_from_io_error() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let core_error = CoreError::from(io_error);
        match core_error {
            CoreError::IoError(msg) => assert!(msg.contains("file not found")),
            _ => panic!("Expected IoError"),
        }
    }

    #[test]
    fn test_from_serde_error() {
        let serde_error = serde_json::from_str::<i32>("invalid").unwrap_err();
        let core_error = CoreError::from(serde_error);
        match core_error {
            CoreError::SerializationError(_) => {} // Expected
            _ => panic!("Expected SerializationError"),
        }
    }

    #[test]
    fn test_check_positive() {
        // Test positive values
        assert_eq!(check_positive(5, "test").unwrap(), 5);
        assert_eq!(check_positive(1.5f32, "test").unwrap(), 1.5f32);

        // Test zero and negative values
        assert!(check_positive(0, "test").is_err());
        assert!(check_positive(-1, "test").is_err());
        assert!(check_positive(-1.5f32, "test").is_err());

        // Check error message
        let err = check_positive(0, "test_param").unwrap_err();
        match err {
            CoreError::InvalidParameter { name, message } => {
                assert_eq!(name, "test_param");
                assert!(message.contains("must be positive"));
            }
            _ => panic!("Expected InvalidParameter"),
        }
    }

    #[test]
    fn test_check_finite_f32() {
        // Test finite values
        assert_eq!(check_finite_f32(1.5, "test").unwrap(), 1.5);
        assert_eq!(check_finite_f32(0.0, "test").unwrap(), 0.0);
        assert_eq!(check_finite_f32(-1.5, "test").unwrap(), -1.5);

        // Test non-finite values
        assert!(check_finite_f32(f32::INFINITY, "test").is_err());
        assert!(check_finite_f32(f32::NEG_INFINITY, "test").is_err());
        assert!(check_finite_f32(f32::NAN, "test").is_err());

        // Check error message
        let err = check_finite_f32(f32::INFINITY, "test_param").unwrap_err();
        match err {
            CoreError::InvalidParameter { name, message } => {
                assert_eq!(name, "test_param");
                assert!(message.contains("must be finite"));
            }
            _ => panic!("Expected InvalidParameter"),
        }
    }

    #[test]
    fn test_check_finite_f64() {
        // Test finite values
        assert_eq!(check_finite_f64(1.5, "test").unwrap(), 1.5);
        assert_eq!(check_finite_f64(0.0, "test").unwrap(), 0.0);
        assert_eq!(check_finite_f64(-1.5, "test").unwrap(), -1.5);

        // Test non-finite values
        assert!(check_finite_f64(f64::INFINITY, "test").is_err());
        assert!(check_finite_f64(f64::NEG_INFINITY, "test").is_err());
        assert!(check_finite_f64(f64::NAN, "test").is_err());
    }

    #[test]
    fn test_check_finite_array() {
        // Test array with all finite values
        let finite_array = [1.0, 2.5, -3.0, 0.0];
        assert!(check_finite_array(&finite_array).is_ok());

        // Test empty array
        assert!(check_finite_array(&[]).is_ok());

        // Test array with infinity
        let inf_array = [1.0, f32::INFINITY, 3.0];
        let err = check_finite_array(&inf_array).unwrap_err();
        match err {
            CoreError::InvalidParameter { name, message } => {
                assert_eq!(name, "array[1]");
                assert!(message.contains("must be finite"));
            }
            _ => panic!("Expected InvalidParameter"),
        }

        // Test array with NaN
        let nan_array = [1.0, f32::NAN, 3.0];
        assert!(check_finite_array(&nan_array).is_err());
    }

    #[test]
    fn test_check_dimensions() {
        // Test matching dimensions
        assert!(check_dimensions(5, 5, "test").is_ok());
        assert!(check_dimensions(0, 0, "test").is_ok());

        // Test mismatched dimensions
        let err = check_dimensions(5, 3, "test").unwrap_err();
        match err {
            CoreError::DimensionMismatch { expected, actual } => {
                assert_eq!(expected, 5);
                assert_eq!(actual, 3);
            }
            _ => panic!("Expected DimensionMismatch"),
        }
    }

    #[test]
    fn test_check_non_empty() {
        // Test non-empty slice
        let data = [1, 2, 3];
        assert!(check_non_empty(&data, "test").is_ok());

        // Test empty slice
        let empty: &[i32] = &[];
        let err = check_non_empty(empty, "test_param").unwrap_err();
        match err {
            CoreError::InvalidParameter { name, message } => {
                assert_eq!(name, "test_param");
                assert!(message.contains("must not be empty"));
            }
            _ => panic!("Expected InvalidParameter"),
        }
    }

    #[test]
    fn test_check_non_empty_str() {
        // Test non-empty string
        assert!(check_non_empty_str("hello", "test").is_ok());

        // Test empty string
        let err = check_non_empty_str("", "test_param").unwrap_err();
        match err {
            CoreError::InvalidParameter { name, message } => {
                assert_eq!(name, "test_param");
                assert!(message.contains("must not be empty"));
            }
            _ => panic!("Expected InvalidParameter"),
        }
    }

    #[test]
    fn test_check_range() {
        // Test value within range
        assert_eq!(check_range(5, 1, 10, "test").unwrap(), 5);
        assert_eq!(check_range(1, 1, 10, "test").unwrap(), 1);
        assert_eq!(check_range(10, 1, 10, "test").unwrap(), 10);
        assert_eq!(check_range(2.5f32, 1.0, 5.0, "test").unwrap(), 2.5f32);

        // Test value below range
        let err = check_range(0, 1, 10, "test_param").unwrap_err();
        match err {
            CoreError::InvalidParameter { name, message } => {
                assert_eq!(name, "test_param");
                assert!(message.contains("between 1 and 10"));
                assert!(message.contains("got 0"));
            }
            _ => panic!("Expected InvalidParameter"),
        }

        // Test value above range
        let err = check_range(11, 1, 10, "test_param").unwrap_err();
        match err {
            CoreError::InvalidParameter { name, message } => {
                assert_eq!(name, "test_param");
                assert!(message.contains("between 1 and 10"));
                assert!(message.contains("got 11"));
            }
            _ => panic!("Expected InvalidParameter"),
        }
    }

    #[test]
    fn test_error_variants() {
        let error = CoreError::NotSupported("test operation".to_string());
        assert_eq!(error.to_string(), "Operation not supported: test operation");

        let error = CoreError::CapabilityNotAvailable("SIMD".to_string());
        assert_eq!(error.to_string(), "Platform capability not available: SIMD");

        let error = CoreError::MemoryError("allocation failed".to_string());
        assert_eq!(
            error.to_string(),
            "Memory allocation failed: allocation failed"
        );

        let error = CoreError::ConfigError("invalid setting".to_string());
        assert_eq!(error.to_string(), "Configuration error: invalid setting");

        let error = CoreError::Timeout("5 seconds".to_string());
        assert_eq!(error.to_string(), "Operation timed out: 5 seconds");

        let error = CoreError::Internal("unexpected state".to_string());
        assert_eq!(error.to_string(), "Internal error: unexpected state");
    }
}
