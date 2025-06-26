//! Core error types for OxiRS
//!
//! This module provides the base error types that all OxiRS modules should use.
//! Module-specific errors should include this as a variant.

use std::fmt;

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

/// Validation functions for common parameter checks
pub mod validation {
    use super::{CoreError, CoreResult};
    
    /// Check that a value is positive
    pub fn check_positive<T>(value: T, name: &str) -> CoreResult<T>
    where
        T: PartialOrd + Default + fmt::Display + Copy,
    {
        if value <= T::default() {
            Err(CoreError::InvalidParameter {
                name: name.to_string(),
                message: format!("Value must be positive, got {}", value),
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
                message: format!("Value must be finite, got {}", value),
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
                message: format!("Value must be finite, got {}", value),
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
                    name: format!("array[{}]", i),
                    message: format!("Value must be finite, got {}", value),
                });
            }
        }
        Ok(())
    }
    
    /// Check that dimensions match
    pub fn check_dimensions(expected: usize, actual: usize, context: &str) -> CoreResult<()> {
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
                message: format!("Value must be between {} and {}, got {}", min, max, value),
            })
        } else {
            Ok(value)
        }
    }
}

use std::fmt;