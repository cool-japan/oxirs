//! Advanced fault tolerance for federated SPARQL endpoints.
//!
//! This module provides:
//! - [`CircuitBreaker`] – three-state circuit breaker (Closed → Open → Half-Open).
//! - [`RetryPolicy`] – exponential back-off with full jitter.

pub mod circuit_breaker;
pub mod retry_policy;

pub use circuit_breaker::{
    CallError, CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError, CircuitBreakerStats,
    CircuitState,
};
pub use retry_policy::{RetryAttempt, RetryConfig, RetryPolicy, RetryStats};
