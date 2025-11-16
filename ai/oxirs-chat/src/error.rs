//! Error types for oxirs-chat

use thiserror::Error;

/// Result type alias for oxirs-chat operations
pub type Result<T> = std::result::Result<T, ChatError>;

/// Main error type for the chat system
#[derive(Debug, Error)]
pub enum ChatError {
    /// RAG retrieval error
    #[error("RAG retrieval failed: {0}")]
    RagRetrievalError(String),

    /// LLM generation error
    #[error("LLM generation failed: {0}")]
    LlmGenerationError(String),

    /// SPARQL generation error
    #[error("SPARQL generation failed: {0}")]
    SparqlGenerationError(String),

    /// SPARQL execution error
    #[error("SPARQL execution failed: {0}")]
    SparqlExecutionError(String),

    /// Session not found
    #[error("Session not found: {0}")]
    SessionNotFound(String),

    /// Session error
    #[error("Session error: {0}")]
    SessionError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Storage error
    #[error("Storage error: {0}")]
    StorageError(String),

    /// Network error
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Timeout error
    #[error("Operation timed out: {0}")]
    TimeoutError(String),

    /// Validation error
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Deserialization error
    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    /// Internal error
    #[error("Internal error: {0}")]
    InternalError(String),

    /// Quantum processing error
    #[error("Quantum processing error: {0}")]
    QuantumProcessingError(String),

    /// Consciousness processing error
    #[error("Consciousness processing error: {0}")]
    ConsciousnessProcessingError(String),

    /// Reasoning error
    #[error("Reasoning error: {0}")]
    ReasoningError(String),

    /// Schema introspection error
    #[error("Schema introspection error: {0}")]
    SchemaIntrospectionError(String),

    /// Wrapped anyhow error
    #[error(transparent)]
    Other(#[from] anyhow::Error),

    /// Wrapped serde_json error
    #[error(transparent)]
    JsonError(#[from] serde_json::Error),

    /// Wrapped IO error
    #[error(transparent)]
    IoError(#[from] std::io::Error),

    /// Wrapped oxirs-core error
    #[error("OxiRS core error: {0}")]
    CoreError(String),
}

impl From<String> for ChatError {
    fn from(err: String) -> Self {
        ChatError::CoreError(err)
    }
}

impl ChatError {
    /// Check if the error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            ChatError::NetworkError(_)
                | ChatError::TimeoutError(_)
                | ChatError::LlmGenerationError(_)
        )
    }

    /// Get error category
    pub fn category(&self) -> ErrorCategory {
        match self {
            ChatError::RagRetrievalError(_) => ErrorCategory::Retrieval,
            ChatError::LlmGenerationError(_) => ErrorCategory::Generation,
            ChatError::SparqlGenerationError(_) | ChatError::SparqlExecutionError(_) => {
                ErrorCategory::Query
            }
            ChatError::SessionNotFound(_) | ChatError::SessionError(_) => ErrorCategory::Session,
            ChatError::NetworkError(_) | ChatError::TimeoutError(_) => ErrorCategory::Network,
            ChatError::ValidationError(_) => ErrorCategory::Validation,
            _ => ErrorCategory::Internal,
        }
    }
}

/// Error category for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    Retrieval,
    Generation,
    Query,
    Session,
    Network,
    Validation,
    Internal,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let error = ChatError::SessionNotFound("test-session".to_string());
        assert_eq!(error.to_string(), "Session not found: test-session");
    }

    #[test]
    fn test_error_recoverability() {
        let recoverable = ChatError::TimeoutError("timeout".to_string());
        assert!(recoverable.is_recoverable());

        let not_recoverable = ChatError::ValidationError("invalid".to_string());
        assert!(!not_recoverable.is_recoverable());
    }

    #[test]
    fn test_error_category() {
        let error = ChatError::LlmGenerationError("failed".to_string());
        assert_eq!(error.category(), ErrorCategory::Generation);
    }
}
