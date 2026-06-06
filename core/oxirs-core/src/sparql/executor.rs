//! SPARQL query execution engine
//!
//! This module provides the core query execution logic for SPARQL queries.
//! It delegates to parser, pattern, filter, expression, aggregate, and modifier modules.
//!
//! ## Performance Monitoring
//!
//! The executor integrates SciRS2-core metrics for comprehensive query performance tracking:
//! - Query execution time (per query type: SELECT, ASK, CONSTRUCT, DESCRIBE)
//! - Pattern matching operations
//! - Result set sizes
//! - Query complexity metrics
//!
//! ## Module layout
//!
//! - `executor.rs` (this file): struct definition, `new()`, `get_stats()`, `execute()`/`query()` dispatcher
//! - `executor_query_handlers.rs`: all `execute_*` method implementations and helpers (declared in `mod.rs`)
//! - `executor_tests.rs`: unit tests (declared in `mod.rs`)

use super::*;
use crate::rdf_store::{OxirsQueryResults, StorageBackend};
use crate::{OxirsError, Result};
use scirs2_core::metrics::{Counter, Histogram, Timer};
use std::collections::HashMap;
use std::sync::Arc;

/// Executor performance statistics
#[derive(Debug, Clone)]
pub struct ExecutorStats {
    /// Total number of queries executed
    pub total_queries: u64,
    /// Number of SELECT queries
    pub select_queries: u64,
    /// Number of ASK queries
    pub ask_queries: u64,
    /// Number of CONSTRUCT queries
    pub construct_queries: u64,
    /// Number of DESCRIBE queries
    pub describe_queries: u64,
    /// Total pattern matching operations
    pub pattern_matches: u64,
    /// Average execution time in seconds
    pub avg_execution_time_secs: f64,
    /// Total number of observations
    pub total_observations: u64,
}

/// VALUES clause for inline data (SPARQL-specific)
#[derive(Debug, Clone)]
pub(crate) struct ValuesClause {
    pub(crate) variables: Vec<String>,
    pub(crate) rows: Vec<Vec<String>>,
}

/// SPARQL query executor with integrated performance monitoring
///
/// This executor tracks query performance metrics using SciRS2-core:
/// - Query execution time broken down by query type
/// - Pattern matching statistics
/// - Result set size distribution
/// - Query complexity indicators
pub struct QueryExecutor<'a> {
    pub(crate) backend: &'a StorageBackend,
    /// Query execution timer
    pub(crate) query_timer: Arc<Timer>,
    /// SELECT query counter
    pub(crate) select_counter: Arc<Counter>,
    /// ASK query counter
    pub(crate) ask_counter: Arc<Counter>,
    /// CONSTRUCT query counter
    pub(crate) construct_counter: Arc<Counter>,
    /// DESCRIBE query counter
    pub(crate) describe_counter: Arc<Counter>,
    /// Pattern matching counter
    pub(crate) pattern_counter: Arc<Counter>,
    /// Result set size histogram
    pub(crate) result_size_histogram: Arc<Histogram>,
}

impl<'a> QueryExecutor<'a> {
    /// Create a new query executor for the given storage backend with performance monitoring
    ///
    /// The executor automatically tracks:
    /// - Query execution times
    /// - Query type distribution
    /// - Pattern matching operations
    /// - Result set sizes
    pub fn new(backend: &'a StorageBackend) -> Self {
        Self {
            backend,
            query_timer: Arc::new(Timer::new("query_execution_time".to_string())),
            select_counter: Arc::new(Counter::new("select_queries".to_string())),
            ask_counter: Arc::new(Counter::new("ask_queries".to_string())),
            construct_counter: Arc::new(Counter::new("construct_queries".to_string())),
            describe_counter: Arc::new(Counter::new("describe_queries".to_string())),
            pattern_counter: Arc::new(Counter::new("pattern_matches".to_string())),
            result_size_histogram: Arc::new(Histogram::new("result_set_size".to_string())),
        }
    }

    /// Get query execution statistics
    ///
    /// Returns the current metrics including:
    /// - Total queries executed by type
    /// - Average execution time
    /// - Pattern matching statistics
    /// - Result set size distribution
    pub fn get_stats(&self) -> ExecutorStats {
        let select = self.select_counter.get();
        let ask = self.ask_counter.get();
        let construct = self.construct_counter.get();
        let describe = self.describe_counter.get();

        let timer_stats = self.query_timer.get_stats();

        ExecutorStats {
            total_queries: select + ask + construct + describe,
            select_queries: select,
            ask_queries: ask,
            construct_queries: construct,
            describe_queries: describe,
            pattern_matches: self.pattern_counter.get(),
            avg_execution_time_secs: timer_stats.mean,
            total_observations: timer_stats.count,
        }
    }

    /// Execute a SPARQL query
    pub fn execute(&self, sparql: &str) -> Result<OxirsQueryResults> {
        self.query(sparql)
    }

    /// Execute a SPARQL query (main dispatcher)
    pub fn query(&self, sparql: &str) -> Result<OxirsQueryResults> {
        let start = std::time::Instant::now();

        let sparql = sparql.trim();

        let (prefixes, expanded_query) = self.extract_and_expand_prefixes(sparql)?;

        let query_to_execute = if !prefixes.is_empty() {
            &expanded_query
        } else {
            sparql
        };

        let result = if query_to_execute.to_uppercase().contains("SELECT") {
            self.select_counter.inc();
            self.execute_select_query(query_to_execute)
        } else if query_to_execute.to_uppercase().starts_with("ASK") {
            self.ask_counter.inc();
            self.execute_ask_query(query_to_execute)
        } else if query_to_execute.to_uppercase().starts_with("CONSTRUCT") {
            self.construct_counter.inc();
            self.execute_construct_query(query_to_execute)
        } else if query_to_execute.to_uppercase().starts_with("DESCRIBE") {
            self.describe_counter.inc();
            self.execute_describe_query(query_to_execute)
        } else {
            return Err(OxirsError::Query(format!(
                "Unsupported SPARQL query type: {sparql}"
            )));
        };

        let duration = start.elapsed();
        self.query_timer.observe(duration);

        if let Ok(ref query_result) = result {
            self.result_size_histogram
                .observe(query_result.len() as f64);
        }

        result
    }

    /// Extract and expand PREFIX declarations
    fn extract_and_expand_prefixes(
        &self,
        sparql: &str,
    ) -> Result<(HashMap<String, String>, String)> {
        extract_and_expand_prefixes(sparql)
    }
}
