//! # Stream Error Types
//!
//! Error types for the streaming module.

use thiserror::Error;

/// Result type for streaming operations
pub type StreamResult<T> = Result<T, StreamError>;

/// Errors that can occur in streaming operations
#[derive(Error, Debug)]
pub enum StreamError {
    #[error("Backend error: {0}")]
    Backend(String),

    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Deserialization error: {0}")]
    Deserialization(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Topic not found: {0}")]
    TopicNotFound(String),

    #[error("Consumer group error: {0}")]
    ConsumerGroup(String),

    #[error("Offset error: {0}")]
    Offset(String),

    #[error("Timeout error: {0}")]
    Timeout(String),

    #[error("Circuit breaker open")]
    CircuitBreakerOpen,

    #[error("Feature not supported: {0}")]
    NotSupported(String),

    #[error("Not connected: {0}")]
    NotConnected(String),

    #[error("Topic creation error: {0}")]
    TopicCreation(String),

    #[error("Topic deletion error: {0}")]
    TopicDeletion(String),

    #[error("Topic list error: {0}")]
    TopicList(String),

    #[error("Send error: {0}")]
    Send(String),

    #[error("Receive error: {0}")]
    Receive(String),

    #[error("Offset commit error: {0}")]
    OffsetCommit(String),

    #[error("Commit error: {0}")]
    CommitError(String),

    #[error("Seek error: {0}")]
    SeekError(String),

    #[error("Topic metadata error: {0}")]
    TopicMetadata(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Quantum processing error: {0}")]
    QuantumProcessing(String),

    #[error("Quantum error correction failed: {code_type:?}, error: {error}")]
    QuantumErrorCorrection { code_type: String, error: String },

    #[error("Quantum decoherence detected: coherence_time: {coherence_time:?}")]
    QuantumDecoherence { coherence_time: std::time::Duration },

    #[error("Biological computation error: {0}")]
    BiologicalComputation(String),

    #[error("DNA encoding error: sequence_length: {length}, gc_content: {gc_content}")]
    DNAEncoding { length: usize, gc_content: f64 },

    #[error("Cellular automaton error: generation: {generation}, error: {error}")]
    CellularAutomaton { generation: usize, error: String },

    #[error("Consciousness processing error: level: {level:?}, error: {error}")]
    ConsciousnessProcessing { level: String, error: String },

    #[error("Neural network error: {0}")]
    NeuralNetwork(String),

    #[error("Time travel query error: {0}")]
    TimeTravelQuery(String),

    #[error("WASM edge computing error: {0}")]
    WasmEdgeComputing(String),

    #[error("Invalid WASM module: {0}")]
    InvalidModule(String),

    #[error("Invalid signature: {0}")]
    InvalidSignature(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Performance optimization error: {metric}, expected: {expected}, actual: {actual}")]
    PerformanceOptimization {
        metric: String,
        expected: f64,
        actual: f64,
    },

    #[error("Other error: {0}")]
    Other(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("I/O error: {0}")]
    Io(String),

    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
}

impl From<std::io::Error> for StreamError {
    fn from(err: std::io::Error) -> Self {
        StreamError::Backend(err.to_string())
    }
}

impl From<serde_json::Error> for StreamError {
    fn from(err: serde_json::Error) -> Self {
        StreamError::Serialization(err.to_string())
    }
}

impl From<anyhow::Error> for StreamError {
    fn from(err: anyhow::Error) -> Self {
        StreamError::Other(err.to_string())
    }
}
