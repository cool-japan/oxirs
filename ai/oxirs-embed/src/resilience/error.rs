//! Comprehensive error types for AI operations

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// Comprehensive error type for AI operations
#[derive(Debug, Error, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum AIError {
    /// LLM API errors (OpenAI, Anthropic, etc.)
    #[error("LLM API error: {message}")]
    LLMError {
        provider: String,
        message: String,
        status_code: Option<u16>,
        retry_after: Option<u64>,
    },

    /// Rate limit exceeded
    #[error("Rate limit exceeded: {message}")]
    RateLimitError {
        limit_type: String, // "requests" or "tokens"
        current: usize,
        limit: usize,
        reset_at: Option<chrono::DateTime<chrono::Utc>>,
        message: String,
    },

    /// Timeout errors
    #[error("Timeout: {operation} took longer than {timeout_secs}s")]
    TimeoutError {
        operation: String,
        timeout_secs: u64,
        elapsed_secs: f64,
    },

    /// Authentication/Authorization errors
    #[error("Authentication error: {message}")]
    AuthError {
        provider: String,
        message: String,
    },

    /// Invalid input/configuration
    #[error("Validation error: {message}")]
    ValidationError {
        field: String,
        message: String,
        expected: Option<String>,
        actual: Option<String>,
    },

    /// Network errors
    #[error("Network error: {message}")]
    NetworkError {
        url: Option<String>,
        message: String,
        retryable: bool,
    },

    /// Circuit breaker open
    #[error("Circuit breaker is open: {reason}")]
    CircuitBreakerError {
        service: String,
        reason: String,
        retry_after_secs: u64,
    },

    /// Resource exhaustion
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted {
        resource: String, // "memory", "tokens", "connections"
        current: usize,
        limit: usize,
    },

    /// Model errors
    #[error("Model error: {message}")]
    ModelError {
        model_name: String,
        message: String,
        error_type: String, // "not_found", "unsupported", "loading_failed"
    },

    /// Data parsing/serialization errors
    #[error("Parse error: {message}")]
    ParseError {
        format: String, // "json", "yaml", "sparql"
        message: String,
        line: Option<usize>,
        column: Option<usize>,
    },

    /// Internal errors
    #[error("Internal error: {message}")]
    InternalError {
        component: String,
        message: String,
        stacktrace: Option<String>,
    },
}

/// Error kind for categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AIErrorKind {
    /// Temporary error, retry may succeed
    Transient,

    /// Permanent error, retry will not succeed
    Permanent,

    /// Rate limit, should wait before retry
    RateLimit,

    /// Client error (bad request, invalid input)
    Client,

    /// Server error (service unavailable, internal error)
    Server,

    /// Authentication/authorization error
    Auth,

    /// Circuit breaker triggered
    CircuitBreaker,
}

impl AIError {
    /// Get the error kind for retry logic
    pub fn kind(&self) -> AIErrorKind {
        match self {
            Self::LLMError { status_code, .. } => {
                match status_code {
                    Some(429) => AIErrorKind::RateLimit,
                    Some(code) if *code >= 500 => AIErrorKind::Server,
                    Some(code) if *code >= 400 => AIErrorKind::Client,
                    _ => AIErrorKind::Transient,
                }
            }
            Self::RateLimitError { .. } => AIErrorKind::RateLimit,
            Self::TimeoutError { .. } => AIErrorKind::Transient,
            Self::AuthError { .. } => AIErrorKind::Auth,
            Self::ValidationError { .. } => AIErrorKind::Client,
            Self::NetworkError { retryable, .. } => {
                if *retryable {
                    AIErrorKind::Transient
                } else {
                    AIErrorKind::Permanent
                }
            }
            Self::CircuitBreakerError { .. } => AIErrorKind::CircuitBreaker,
            Self::ResourceExhausted { .. } => AIErrorKind::Server,
            Self::ModelError { error_type, .. } => {
                match error_type.as_str() {
                    "not_found" | "unsupported" => AIErrorKind::Permanent,
                    "loading_failed" => AIErrorKind::Transient,
                    _ => AIErrorKind::Server,
                }
            }
            Self::ParseError { .. } => AIErrorKind::Client,
            Self::InternalError { .. } => AIErrorKind::Server,
        }
    }

    /// Check if the error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self.kind(),
            AIErrorKind::Transient | AIErrorKind::RateLimit | AIErrorKind::Server
        )
    }

    /// Get retry delay in seconds
    pub fn retry_delay_secs(&self) -> Option<u64> {
        match self {
            Self::RateLimitError { reset_at, .. } => {
                reset_at.map(|reset| {
                    let now = chrono::Utc::now();
                    (reset - now).num_seconds().max(0) as u64
                })
            }
            Self::LLMError { retry_after, .. } => *retry_after,
            Self::CircuitBreakerError { retry_after_secs, .. } => Some(*retry_after_secs),
            Self::TimeoutError { .. } => Some(5), // 5 second delay for timeouts
            Self::NetworkError { retryable, .. } if *retryable => Some(3),
            _ => None,
        }
    }

    /// Check if the error indicates service unavailability
    pub fn is_service_unavailable(&self) -> bool {
        match self {
            Self::LLMError { status_code, .. } => matches!(status_code, Some(503)),
            Self::CircuitBreakerError { .. } => true,
            Self::NetworkError { .. } => true,
            _ => false,
        }
    }

    /// Get HTTP status code if applicable
    pub fn status_code(&self) -> Option<u16> {
        match self {
            Self::LLMError { status_code, .. } => *status_code,
            Self::RateLimitError { .. } => Some(429),
            Self::AuthError { .. } => Some(401),
            Self::ValidationError { .. } => Some(400),
            Self::TimeoutError { .. } => Some(504),
            Self::CircuitBreakerError { .. } => Some(503),
            Self::ResourceExhausted { .. } => Some(507),
            _ => None,
        }
    }
}

/// Additional context for errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    pub request_id: Option<String>,
    pub user_id: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub component: String,
    pub operation: String,
    pub additional_info: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    pub fn new(component: &str, operation: &str) -> Self {
        Self {
            request_id: None,
            user_id: None,
            timestamp: chrono::Utc::now(),
            component: component.to_string(),
            operation: operation.to_string(),
            additional_info: std::collections::HashMap::new(),
        }
    }

    pub fn with_request_id(mut self, request_id: String) -> Self {
        self.request_id = Some(request_id);
        self
    }

    pub fn with_user_id(mut self, user_id: String) -> Self {
        self.user_id = Some(user_id);
        self
    }

    pub fn add_info(mut self, key: String, value: String) -> Self {
        self.additional_info.insert(key, value);
        self
    }
}

/// Wrapper combining AIError with context
#[derive(Debug, Clone)]
pub struct ContextualError {
    pub error: AIError,
    pub context: ErrorContext,
}

impl fmt::Display for ContextualError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (component: {}, operation: {}, timestamp: {})",
            self.error,
            self.context.component,
            self.context.operation,
            self.context.timestamp
        )
    }
}

impl std::error::Error for ContextualError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_kind_classification() {
        let error = AIError::RateLimitError {
            limit_type: "requests".to_string(),
            current: 100,
            limit: 60,
            reset_at: None,
            message: "Too many requests".to_string(),
        };
        assert_eq!(error.kind(), AIErrorKind::RateLimit);
        assert!(error.is_retryable());
    }

    #[test]
    fn test_llm_error_status_codes() {
        let server_error = AIError::LLMError {
            provider: "openai".to_string(),
            message: "Internal error".to_string(),
            status_code: Some(500),
            retry_after: None,
        };
        assert_eq!(server_error.kind(), AIErrorKind::Server);
        assert!(server_error.is_retryable());

        let client_error = AIError::LLMError {
            provider: "openai".to_string(),
            message: "Bad request".to_string(),
            status_code: Some(400),
            retry_after: None,
        };
        assert_eq!(client_error.kind(), AIErrorKind::Client);
        assert!(!client_error.is_retryable());
    }

    #[test]
    fn test_retry_delay() {
        let error = AIError::CircuitBreakerError {
            service: "openai".to_string(),
            reason: "Too many failures".to_string(),
            retry_after_secs: 60,
        };
        assert_eq!(error.retry_delay_secs(), Some(60));
    }

    #[test]
    fn test_error_context() {
        let context = ErrorContext::new("chat", "generate_response")
            .with_request_id("req-123".to_string())
            .with_user_id("user-456".to_string())
            .add_info("model".to_string(), "gpt-4".to_string());

        assert_eq!(context.component, "chat");
        assert_eq!(context.operation, "generate_response");
        assert_eq!(context.request_id, Some("req-123".to_string()));
        assert_eq!(context.user_id, Some("user-456".to_string()));
        assert_eq!(context.additional_info.get("model"), Some(&"gpt-4".to_string()));
    }
}
