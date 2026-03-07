//! Error types for the federated query rewrite module.
//!
//! All public functions in `query_rewrite` return [`FederationResult<T>`],
//! which is a type alias for `Result<T, FederationError>`.

use thiserror::Error;

/// Errors that can occur during federated query rewriting and optimization.
#[derive(Debug, Error)]
pub enum FederationError {
    /// Raised when [`super::decomposer::QueryDecomposer`] is called with an empty
    /// endpoint list — there is nowhere to route queries.
    #[error("No SPARQL endpoints configured for federation")]
    EmptyEndpointList,

    /// Raised when the input query string cannot be parsed into triple patterns.
    #[error("Failed to parse SPARQL query: {0}")]
    QueryParseError(String),

    /// Raised when the optimizer detects an illegal or unsatisfiable execution plan.
    #[error("Invalid execution plan: {0}")]
    InvalidPlan(String),

    /// Raised when a cost estimation operation fails (e.g. arithmetic overflow).
    #[error("Cost estimation failed: {0}")]
    CostEstimationError(String),

    /// Generic I/O or network-level error string.
    #[error("Federation I/O error: {0}")]
    IoError(String),
}

/// Convenience alias for `Result<T, FederationError>`.
pub type FederationResult<T> = Result<T, FederationError>;
