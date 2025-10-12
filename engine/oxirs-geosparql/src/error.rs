//! Error types for GeoSPARQL operations

use thiserror::Error;

/// Result type for GeoSPARQL operations
pub type Result<T> = std::result::Result<T, GeoSparqlError>;

/// Errors that can occur during GeoSPARQL operations
#[derive(Error, Debug)]
pub enum GeoSparqlError {
    /// Invalid WKT (Well-Known Text) format
    #[error("Invalid WKT format: {0}")]
    InvalidWkt(String),

    /// Invalid GML (Geography Markup Language) format
    #[error("Invalid GML format: {0}")]
    InvalidGml(String),

    /// Invalid geometry type
    #[error("Invalid geometry type: {0}")]
    InvalidGeometryType(String),

    /// Unsupported geometry operation
    #[error("Unsupported geometry operation: {0}")]
    UnsupportedOperation(String),

    /// Invalid coordinate reference system
    #[error("Invalid CRS: {0}")]
    InvalidCrs(String),

    /// CRS mismatch between geometries
    #[error("CRS mismatch: expected {expected}, found {found}")]
    CrsMismatch {
        /// Expected CRS URI
        expected: String,
        /// Found CRS URI
        found: String,
    },

    /// CRS transformation failed
    #[error("CRS transformation failed: {0}")]
    CrsTransformationFailed(String),

    /// Geometry operation failed
    #[error("Geometry operation failed: {0}")]
    GeometryOperationFailed(String),

    /// Invalid spatial relation
    #[error("Invalid spatial relation: {0}")]
    InvalidSpatialRelation(String),

    /// Invalid dimension
    #[error("Invalid dimension: {0}")]
    InvalidDimension(String),

    /// Parse error
    #[error("Parse error: {0}")]
    ParseError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Index error
    #[error("Index error: {0}")]
    IndexError(String),

    /// Other error
    #[error("GeoSPARQL error: {0}")]
    Other(String),
}

// WKT parse errors are handled as strings
// impl From<wkt::WktParseError> for GeoSparqlError {
//     fn from(err: wkt::WktParseError) -> Self {
//         GeoSparqlError::InvalidWkt(err.to_string())
//     }
// }

impl From<String> for GeoSparqlError {
    fn from(s: String) -> Self {
        GeoSparqlError::Other(s)
    }
}

impl From<&str> for GeoSparqlError {
    fn from(s: &str) -> Self {
        GeoSparqlError::Other(s.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = GeoSparqlError::InvalidWkt("test error".to_string());
        assert_eq!(err.to_string(), "Invalid WKT format: test error");

        let err = GeoSparqlError::CrsMismatch {
            expected: "EPSG:4326".to_string(),
            found: "EPSG:3857".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "CRS mismatch: expected EPSG:4326, found EPSG:3857"
        );
    }
}
