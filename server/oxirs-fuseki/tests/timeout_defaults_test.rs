//! Shipped-default timeout invariant (Finding 3).
//!
//! The axum `TimeoutLayer` (`server.request_timeout_secs`) is a coarse safety net
//! that must sit ABOVE the per-query `ExecutionBudget`
//! (`performance.query_optimization.max_query_time_secs`) plus the small grace
//! the blocking task is given before the outer `tokio::time::timeout` frees the
//! response. Only then does the precise per-query 408 path fire in normal
//! operation; otherwise the layer preempts the budget and every long query looks
//! like a generic 408 while a detached blocking task keeps burning CPU up to
//! `max_query_time_secs`.
//!
//! `Runtime::build_router` warns exactly when
//! `request_timeout_secs <= max_query_time_secs + QUERY_TIMEOUT_GRACE_SECS`.
//! This test pins that the SHIPPED DEFAULTS satisfy the strict inequality, i.e.
//! the default configuration is warning-free and budget-first.

use oxirs_fuseki::config::ServerConfig;

/// Mirror of the private `handlers::sparql::core::QUERY_TIMEOUT_GRACE_SECS`.
/// Kept in sync by construction; the assertion below fails loudly if the default
/// relationship ever regresses, which is what actually matters.
const QUERY_TIMEOUT_GRACE_SECS: u64 = 5;

#[test]
fn default_timeouts_let_the_query_budget_fire_before_the_http_layer() {
    let config = ServerConfig::default();

    let request_timeout_secs = config.server.request_timeout_secs;
    let max_query_time_secs = config.performance.query_optimization.max_query_time_secs;

    // The exact predicate `build_router` warns on, negated: the default config
    // must NOT trip the warning.
    let min_required = max_query_time_secs + QUERY_TIMEOUT_GRACE_SECS;
    assert!(
        request_timeout_secs > min_required,
        "default request_timeout_secs ({request_timeout_secs}s) must exceed \
         max_query_time_secs ({max_query_time_secs}s) + grace ({QUERY_TIMEOUT_GRACE_SECS}s) = \
         {min_required}s so the per-query budget fires before the coarse HTTP TimeoutLayer; \
         otherwise the shipped defaults emit the startup warning and cut long queries at the \
         layer instead of aborting them cleanly."
    );
}

/// A concrete anti-regression on the individual values so a future edit that
/// silently lowers `request_timeout_secs` back toward the query cap is caught
/// with a pointed message.
#[test]
fn default_request_timeout_is_above_the_query_cap() {
    let config = ServerConfig::default();
    assert!(
        config.server.request_timeout_secs
            > config.performance.query_optimization.max_query_time_secs,
        "request_timeout_secs must be strictly greater than max_query_time_secs"
    );
}
