//! Comprehensive error handling for OxiRS Fuseki

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;
use tracing::{error, warn};

/// Fuseki result type
pub type FusekiResult<T> = std::result::Result<T, FusekiError>;

/// Short alias for FusekiResult (for backward compatibility)
pub type Result<T> = FusekiResult<T>;

/// Main error type for OxiRS Fuseki
#[derive(Error, Debug)]
pub enum FusekiError {
    #[error("Invalid SPARQL query: {message}")]
    InvalidQuery { message: String },

    #[error("Invalid SPARQL update: {message}")]
    InvalidUpdate { message: String },

    #[error("Query execution failed: {message}")]
    QueryExecution { message: String },

    #[error("Update execution failed: {message}")]
    UpdateExecution { message: String },

    #[error("Authentication failed: {message}")]
    Authentication { message: String },

    #[error("Authorization failed: {message}")]
    Authorization { message: String },

    #[error("Configuration error: {message}")]
    Configuration { message: String },

    #[error("Store error: {message}")]
    Store { message: String },

    #[error("Parse error: {message}")]
    Parse { message: String },

    #[error("Validation error: {message}")]
    Validation { message: String },

    #[error("Rate limit exceeded")]
    RateLimit,

    #[error("Resource not found: {resource}")]
    NotFound { resource: String },

    #[error("Method not allowed")]
    MethodNotAllowed,

    #[error("Unsupported media type: {media_type}")]
    UnsupportedMediaType { media_type: String },

    #[error("Request timeout")]
    Timeout,

    #[error("Timeout: {0}")]
    TimeoutWithMessage(String),

    #[error("Service unavailable: {message}")]
    ServiceUnavailable { message: String },

    #[error("Internal server error: {message}")]
    Internal { message: String },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("TOML error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] axum::http::Error),

    #[error("JWT error: {0}")]
    Jwt(#[from] jsonwebtoken::errors::Error),

    #[error("Validation error: {0}")]
    ValidatorError(#[from] validator::ValidationErrors),
}

/// HTTP error response format
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl ErrorResponse {
    pub fn new(error: &str, message: String) -> Self {
        Self {
            error: error.to_string(),
            message,
            details: None,
            request_id: None,
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(details);
        self
    }

    pub fn with_request_id(mut self, request_id: String) -> Self {
        self.request_id = Some(request_id);
        self
    }
}

impl FusekiError {
    /// Get the HTTP status code for this error
    pub fn status_code(&self) -> StatusCode {
        match self {
            FusekiError::InvalidQuery { .. }
            | FusekiError::InvalidUpdate { .. }
            | FusekiError::Parse { .. }
            | FusekiError::Validation { .. }
            | FusekiError::ValidatorError(..) => StatusCode::BAD_REQUEST,

            FusekiError::Authentication { .. } => StatusCode::UNAUTHORIZED,

            FusekiError::Authorization { .. } => StatusCode::FORBIDDEN,

            FusekiError::NotFound { .. } => StatusCode::NOT_FOUND,

            FusekiError::MethodNotAllowed => StatusCode::METHOD_NOT_ALLOWED,

            FusekiError::Timeout | FusekiError::TimeoutWithMessage(..) => {
                StatusCode::REQUEST_TIMEOUT
            }

            FusekiError::UnsupportedMediaType { .. } => StatusCode::UNSUPPORTED_MEDIA_TYPE,

            FusekiError::RateLimit => StatusCode::TOO_MANY_REQUESTS,

            FusekiError::ServiceUnavailable { .. } => StatusCode::SERVICE_UNAVAILABLE,

            FusekiError::QueryExecution { .. }
            | FusekiError::UpdateExecution { .. }
            | FusekiError::Store { .. }
            | FusekiError::Configuration { .. }
            | FusekiError::Internal { .. }
            | FusekiError::Io(..)
            | FusekiError::Json(..)
            | FusekiError::Yaml(..)
            | FusekiError::Toml(..)
            | FusekiError::Http(..)
            | FusekiError::Jwt(..) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// Get the error type string for API responses
    pub fn error_type(&self) -> &'static str {
        match self {
            FusekiError::InvalidQuery { .. } => "invalid_query",
            FusekiError::InvalidUpdate { .. } => "invalid_update",
            FusekiError::QueryExecution { .. } => "query_execution_failed",
            FusekiError::UpdateExecution { .. } => "update_execution_failed",
            FusekiError::Authentication { .. } => "authentication_failed",
            FusekiError::Authorization { .. } => "authorization_failed",
            FusekiError::Configuration { .. } => "configuration_error",
            FusekiError::Store { .. } => "store_error",
            FusekiError::Parse { .. } => "parse_error",
            FusekiError::Validation { .. } => "validation_error",
            FusekiError::ValidatorError(..) => "validation_error",
            FusekiError::RateLimit => "rate_limit_exceeded",
            FusekiError::NotFound { .. } => "not_found",
            FusekiError::MethodNotAllowed => "method_not_allowed",
            FusekiError::UnsupportedMediaType { .. } => "unsupported_media_type",
            FusekiError::Timeout | FusekiError::TimeoutWithMessage(..) => "timeout",
            FusekiError::ServiceUnavailable { .. } => "service_unavailable",
            FusekiError::Internal { .. } => "internal_error",
            FusekiError::Io(..) => "io_error",
            FusekiError::Json(..) => "json_error",
            FusekiError::Yaml(..) => "yaml_error",
            FusekiError::Toml(..) => "toml_error",
            FusekiError::Http(..) => "http_error",
            FusekiError::Jwt(..) => "jwt_error",
        }
    }

    /// Create an error response for this error
    pub fn to_error_response(&self, request_id: Option<String>) -> ErrorResponse {
        let mut response = ErrorResponse::new(self.error_type(), self.to_string());

        if let Some(id) = request_id {
            response = response.with_request_id(id);
        }

        // Add validation details for validation errors
        if let FusekiError::ValidatorError(validation_errors) = self {
            let details = serde_json::to_value(validation_errors).unwrap_or_default();
            response = response.with_details(details);
        }

        response
    }

    /// Log this error at the appropriate level
    pub fn log(&self, request_id: Option<&str>) {
        let id = request_id.unwrap_or("unknown");

        match self.status_code() {
            StatusCode::INTERNAL_SERVER_ERROR => {
                error!(
                    request_id = id,
                    error = %self,
                    error_type = self.error_type(),
                    "Internal server error"
                );
            }
            StatusCode::BAD_REQUEST
            | StatusCode::UNAUTHORIZED
            | StatusCode::FORBIDDEN
            | StatusCode::NOT_FOUND
            | StatusCode::METHOD_NOT_ALLOWED
            | StatusCode::UNSUPPORTED_MEDIA_TYPE
            | StatusCode::TOO_MANY_REQUESTS => {
                warn!(
                    request_id = id,
                    error = %self,
                    error_type = self.error_type(),
                    "Client error"
                );
            }
            _ => {
                error!(
                    request_id = id,
                    error = %self,
                    error_type = self.error_type(),
                    "Server error"
                );
            }
        }
    }
}

impl IntoResponse for FusekiError {
    fn into_response(self) -> Response {
        let status = self.status_code();

        // Extract request ID from tracing context if available
        let request_id = tracing::Span::current()
            .field("request_id")
            .and_then(|field| {
                // This is a simplified extraction - in practice you'd use a proper context
                None::<String>
            });

        // Log the error
        self.log(request_id.as_deref());

        // Create error response
        let error_response = self.to_error_response(request_id);

        (status, Json(error_response)).into_response()
    }
}

/// Convenience functions for creating common errors
impl FusekiError {
    pub fn invalid_query(message: impl Into<String>) -> Self {
        Self::InvalidQuery {
            message: message.into(),
        }
    }

    pub fn invalid_update(message: impl Into<String>) -> Self {
        Self::InvalidUpdate {
            message: message.into(),
        }
    }

    pub fn query_execution(message: impl Into<String>) -> Self {
        Self::QueryExecution {
            message: message.into(),
        }
    }

    pub fn update_execution(message: impl Into<String>) -> Self {
        Self::UpdateExecution {
            message: message.into(),
        }
    }

    pub fn authentication(message: impl Into<String>) -> Self {
        Self::Authentication {
            message: message.into(),
        }
    }

    pub fn authorization(message: impl Into<String>) -> Self {
        Self::Authorization {
            message: message.into(),
        }
    }

    pub fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration {
            message: message.into(),
        }
    }

    pub fn store(message: impl Into<String>) -> Self {
        Self::Store {
            message: message.into(),
        }
    }

    pub fn parse(message: impl Into<String>) -> Self {
        Self::Parse {
            message: message.into(),
        }
    }

    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation {
            message: message.into(),
        }
    }

    pub fn not_found(resource: impl Into<String>) -> Self {
        Self::NotFound {
            resource: resource.into(),
        }
    }

    pub fn unsupported_media_type(media_type: impl Into<String>) -> Self {
        Self::UnsupportedMediaType {
            media_type: media_type.into(),
        }
    }

    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self::ServiceUnavailable {
            message: message.into(),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::Parse {
            message: message.into(),
        }
    }

    pub fn conflict(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    pub fn method_not_allowed(message: impl Into<String>) -> Self {
        Self::MethodNotAllowed
    }

    pub fn forbidden(message: impl Into<String>) -> Self {
        Self::Authorization {
            message: message.into(),
        }
    }
}

/// Extension trait for converting Results to FusekiError
pub trait IntoFusekiError<T> {
    fn into_fuseki_error(self) -> FusekiResult<T>;
    fn with_context(self, context: &str) -> FusekiResult<T>;
}

impl<T, E> IntoFusekiError<T> for Result<T, E>
where
    E: fmt::Display,
{
    fn into_fuseki_error(self) -> FusekiResult<T> {
        self.map_err(|e| FusekiError::internal(e.to_string()))
    }

    fn with_context(self, context: &str) -> FusekiResult<T> {
        self.map_err(|e| FusekiError::internal(format!("{}: {}", context, e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_status_codes() {
        assert_eq!(
            FusekiError::invalid_query("test").status_code(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            FusekiError::authentication("test").status_code(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            FusekiError::authorization("test").status_code(),
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            FusekiError::not_found("test").status_code(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            FusekiError::RateLimit.status_code(),
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(
            FusekiError::internal("test").status_code(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[test]
    fn test_error_types() {
        assert_eq!(
            FusekiError::invalid_query("test").error_type(),
            "invalid_query"
        );
        assert_eq!(
            FusekiError::authentication("test").error_type(),
            "authentication_failed"
        );
        assert_eq!(FusekiError::RateLimit.error_type(), "rate_limit_exceeded");
    }

    #[test]
    fn test_error_response_creation() {
        let error = FusekiError::invalid_query("test query error");
        let response = error.to_error_response(Some("req-123".to_string()));

        assert_eq!(response.error, "invalid_query");
        assert_eq!(response.message, "Invalid SPARQL query: test query error");
        assert_eq!(response.request_id, Some("req-123".to_string()));
    }

    #[test]
    fn test_convenience_constructors() {
        let query_error = FusekiError::invalid_query("syntax error");
        assert!(matches!(query_error, FusekiError::InvalidQuery { .. }));

        let auth_error = FusekiError::authentication("invalid token");
        assert!(matches!(auth_error, FusekiError::Authentication { .. }));

        let not_found = FusekiError::not_found("dataset");
        assert!(matches!(not_found, FusekiError::NotFound { .. }));
    }

    #[test]
    fn test_into_fuseki_error_trait() {
        let result: Result<i32, &str> = Err("test error");
        let fuseki_result = result.into_fuseki_error();

        assert!(fuseki_result.is_err());
        assert!(matches!(
            fuseki_result.unwrap_err(),
            FusekiError::Internal { .. }
        ));
    }

    #[test]
    fn test_with_context_trait() {
        let result: Result<i32, &str> = Err("original error");
        let fuseki_result = result.with_context("processing request");

        assert!(fuseki_result.is_err());
        let error_message = fuseki_result.unwrap_err().to_string();
        assert!(error_message.contains("processing request"));
        assert!(error_message.contains("original error"));
    }
}
