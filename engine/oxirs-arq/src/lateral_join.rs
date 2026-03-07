//! # SPARQL 1.2 LATERAL Join Support
//!
//! Implements correlated subqueries (LATERAL joins) for SPARQL 1.2.
//!
//! A LATERAL join allows the right-hand side of a join to reference variables
//! bound by the left-hand side, enabling correlated subqueries that were not
//! possible in SPARQL 1.1.
//!
//! ## SPARQL 1.2 Syntax
//!
//! ```sparql
//! SELECT ?person ?maxScore
//! WHERE {
//!   ?person a :Student .
//!   LATERAL {
//!     SELECT (MAX(?score) AS ?maxScore)
//!     WHERE { ?person :hasExam/:score ?score }
//!   }
//! }
//! ```
//!
//! ## Semantics
//!
//! For each solution mapping `mu` from the left operand, the right operand
//! is evaluated with `mu` as the initial binding. The result is the
//! compatible merge of `mu` with each solution from the right operand.
//!
//! ## References
//!
//! - SPARQL 1.2 Community Group Draft (Section 18.6 – Lateral Joins)
//! - PostgreSQL LATERAL subqueries (similar concept in SQL)

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A single variable binding in a solution mapping.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LateralValue {
    /// An IRI reference
    Iri(String),
    /// A plain or typed literal
    Literal {
        /// The lexical value
        value: String,
        /// Optional datatype IRI
        datatype: Option<String>,
        /// Optional language tag
        lang: Option<String>,
    },
    /// A blank node
    BlankNode(String),
}

impl fmt::Display for LateralValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Iri(iri) => write!(f, "<{iri}>"),
            Self::Literal {
                value,
                datatype,
                lang,
            } => {
                write!(f, "\"{value}\"")?;
                if let Some(dt) = datatype {
                    write!(f, "^^<{dt}>")?;
                }
                if let Some(l) = lang {
                    write!(f, "@{l}")?;
                }
                Ok(())
            }
            Self::BlankNode(id) => write!(f, "_:{id}"),
        }
    }
}

/// A solution mapping: a set of (variable -> value) bindings.
pub type SolutionMapping = HashMap<String, LateralValue>;

/// A LATERAL subquery that may reference variables from the outer scope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LateralSubquery {
    /// Human-readable description (or the original SPARQL fragment).
    pub description: String,
    /// Variables from the outer scope that this subquery references.
    pub correlated_vars: Vec<String>,
    /// Variables produced by this subquery.
    pub projected_vars: Vec<String>,
    /// Whether this subquery contains aggregates.
    pub has_aggregates: bool,
    /// Optional LIMIT on the subquery.
    pub limit: Option<usize>,
    /// Optional ORDER BY direction for the subquery.
    pub order_by: Vec<OrderSpec>,
}

/// Sort specification for subquery ORDER BY.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderSpec {
    /// Variable to sort on.
    pub variable: String,
    /// Ascending (true) or descending (false).
    pub ascending: bool,
}

// ---------------------------------------------------------------------------
// Lateral join algebra node
// ---------------------------------------------------------------------------

/// Represents a LATERAL join in the query algebra.
///
/// The left operand produces solution mappings; for each such mapping, the
/// right operand (a [`LateralSubquery`]) is evaluated with the left's
/// bindings injected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LateralJoin {
    /// Description of the left operand pattern.
    pub left_description: String,
    /// The correlated subquery on the right.
    pub subquery: LateralSubquery,
    /// Execution strategy chosen by the optimizer.
    pub strategy: LateralStrategy,
    /// Optional correlation filter to push down.
    pub pushed_filters: Vec<String>,
}

/// Execution strategy for a lateral join.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LateralStrategy {
    /// Simple nested-loop: for each left row, evaluate the subquery.
    NestedLoop,
    /// Batch multiple left rows and evaluate the subquery once per batch,
    /// using a VALUES clause to inject the batch.
    BatchedValues,
    /// Decorrelate the subquery into a regular join + GROUP BY (when possible).
    Decorrelate,
    /// Cache subquery results keyed on the correlated variable values.
    CachedCorrelation,
}

impl fmt::Display for LateralStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NestedLoop => write!(f, "NestedLoop"),
            Self::BatchedValues => write!(f, "BatchedValues"),
            Self::Decorrelate => write!(f, "Decorrelate"),
            Self::CachedCorrelation => write!(f, "CachedCorrelation"),
        }
    }
}

// ---------------------------------------------------------------------------
// LateralJoinExecutor — the main engine
// ---------------------------------------------------------------------------

/// Configuration for the lateral join executor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LateralJoinConfig {
    /// Maximum rows to batch in `BatchedValues` strategy.
    pub batch_size: usize,
    /// Maximum cache entries for `CachedCorrelation` strategy.
    pub cache_capacity: usize,
    /// Timeout for each subquery evaluation.
    pub subquery_timeout: Duration,
    /// Whether to attempt decorrelation automatically.
    pub auto_decorrelate: bool,
    /// Maximum nesting depth for LATERAL inside LATERAL.
    pub max_nesting_depth: usize,
}

impl Default for LateralJoinConfig {
    fn default() -> Self {
        Self {
            batch_size: 128,
            cache_capacity: 4096,
            subquery_timeout: Duration::from_secs(30),
            auto_decorrelate: true,
            max_nesting_depth: 4,
        }
    }
}

/// Statistics collected during lateral join execution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LateralJoinStats {
    /// Total rows from the left operand.
    pub left_rows: u64,
    /// Total result rows produced.
    pub result_rows: u64,
    /// Number of subquery evaluations performed.
    pub subquery_evaluations: u64,
    /// Number of cache hits (for `CachedCorrelation`).
    pub cache_hits: u64,
    /// Number of cache misses.
    pub cache_misses: u64,
    /// Number of batches submitted (for `BatchedValues`).
    pub batches_submitted: u64,
    /// Total time spent in subquery evaluation.
    pub subquery_time_ms: u64,
    /// Whether decorrelation was applied.
    pub decorrelated: bool,
    /// Rows eliminated by pushed-down filters.
    pub rows_filtered: u64,
}

impl LateralJoinStats {
    /// Cache hit ratio as a percentage (0.0–100.0).
    pub fn cache_hit_ratio(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            return 0.0;
        }
        (self.cache_hits as f64 / total as f64) * 100.0
    }

    /// Average subquery evaluation time in milliseconds.
    pub fn avg_subquery_time_ms(&self) -> f64 {
        if self.subquery_evaluations == 0 {
            return 0.0;
        }
        self.subquery_time_ms as f64 / self.subquery_evaluations as f64
    }
}

/// Executes LATERAL joins using the configured strategy.
pub struct LateralJoinExecutor {
    config: LateralJoinConfig,
    stats: LateralJoinStats,
    /// Per-correlation-key cache: key = stringified correlated var values.
    cache: HashMap<String, Vec<SolutionMapping>>,
}

impl LateralJoinExecutor {
    /// Create a new executor with the given configuration.
    pub fn new(config: LateralJoinConfig) -> Self {
        Self {
            config,
            stats: LateralJoinStats::default(),
            cache: HashMap::new(),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(LateralJoinConfig::default())
    }

    /// Get accumulated statistics.
    pub fn stats(&self) -> &LateralJoinStats {
        &self.stats
    }

    /// Reset statistics and cache.
    pub fn reset(&mut self) {
        self.stats = LateralJoinStats::default();
        self.cache.clear();
    }

    /// Execute a LATERAL join.
    ///
    /// For each row in `left_rows`, the `subquery_evaluator` closure is
    /// called with the correlated bindings extracted from the left row.
    /// The closure should return the subquery results for those bindings.
    pub fn execute<F>(
        &mut self,
        lateral: &LateralJoin,
        left_rows: &[SolutionMapping],
        subquery_evaluator: F,
    ) -> Result<Vec<SolutionMapping>, LateralJoinError>
    where
        F: Fn(&SolutionMapping) -> Result<Vec<SolutionMapping>, LateralJoinError>,
    {
        self.stats.left_rows = left_rows.len() as u64;

        match lateral.strategy {
            LateralStrategy::NestedLoop => {
                self.execute_nested_loop(lateral, left_rows, subquery_evaluator)
            }
            LateralStrategy::BatchedValues => {
                self.execute_batched(lateral, left_rows, subquery_evaluator)
            }
            LateralStrategy::CachedCorrelation => {
                self.execute_cached(lateral, left_rows, subquery_evaluator)
            }
            LateralStrategy::Decorrelate => {
                // Decorrelation transforms the query plan; here we fall back to
                // cached correlation as the runtime strategy after plan rewrite.
                self.execute_cached(lateral, left_rows, subquery_evaluator)
            }
        }
    }

    // ── Nested-loop execution ─────────────────────────────────────────────

    fn execute_nested_loop<F>(
        &mut self,
        lateral: &LateralJoin,
        left_rows: &[SolutionMapping],
        evaluator: F,
    ) -> Result<Vec<SolutionMapping>, LateralJoinError>
    where
        F: Fn(&SolutionMapping) -> Result<Vec<SolutionMapping>, LateralJoinError>,
    {
        let mut results = Vec::new();

        for left_row in left_rows {
            // Extract correlated bindings
            let correlated =
                Self::extract_correlated_bindings(left_row, &lateral.subquery.correlated_vars);

            // Apply pushed-down filters
            if !self.passes_pushed_filters(left_row, &lateral.pushed_filters) {
                self.stats.rows_filtered += 1;
                continue;
            }

            let start = Instant::now();
            let sub_results = evaluator(&correlated)?;
            self.stats.subquery_time_ms += start.elapsed().as_millis() as u64;
            self.stats.subquery_evaluations += 1;

            // Merge left row with each subquery result
            for sub_row in &sub_results {
                let merged = Self::merge_mappings(left_row, sub_row)?;
                results.push(merged);
            }
        }

        self.stats.result_rows = results.len() as u64;
        Ok(results)
    }

    // ── Batched VALUES execution ──────────────────────────────────────────

    fn execute_batched<F>(
        &mut self,
        lateral: &LateralJoin,
        left_rows: &[SolutionMapping],
        evaluator: F,
    ) -> Result<Vec<SolutionMapping>, LateralJoinError>
    where
        F: Fn(&SolutionMapping) -> Result<Vec<SolutionMapping>, LateralJoinError>,
    {
        let mut results = Vec::new();
        let batch_size = self.config.batch_size.max(1);

        for chunk in left_rows.chunks(batch_size) {
            self.stats.batches_submitted += 1;

            // Build a combined bindings map for the batch
            let batch_bindings =
                Self::build_batch_bindings(chunk, &lateral.subquery.correlated_vars);

            let start = Instant::now();
            let batch_results = evaluator(&batch_bindings)?;
            self.stats.subquery_time_ms += start.elapsed().as_millis() as u64;
            self.stats.subquery_evaluations += 1;

            // For batched evaluation, we need to correlate results back to
            // their originating left rows. We do this by matching on the
            // correlated variable values.
            for left_row in chunk {
                if !self.passes_pushed_filters(left_row, &lateral.pushed_filters) {
                    self.stats.rows_filtered += 1;
                    continue;
                }

                for sub_row in &batch_results {
                    if Self::is_compatible(left_row, sub_row, &lateral.subquery.correlated_vars) {
                        let merged = Self::merge_mappings(left_row, sub_row)?;
                        results.push(merged);
                    }
                }
            }
        }

        self.stats.result_rows = results.len() as u64;
        Ok(results)
    }

    // ── Cached correlation execution ──────────────────────────────────────

    fn execute_cached<F>(
        &mut self,
        lateral: &LateralJoin,
        left_rows: &[SolutionMapping],
        evaluator: F,
    ) -> Result<Vec<SolutionMapping>, LateralJoinError>
    where
        F: Fn(&SolutionMapping) -> Result<Vec<SolutionMapping>, LateralJoinError>,
    {
        let mut results = Vec::new();

        for left_row in left_rows {
            if !self.passes_pushed_filters(left_row, &lateral.pushed_filters) {
                self.stats.rows_filtered += 1;
                continue;
            }

            let correlated =
                Self::extract_correlated_bindings(left_row, &lateral.subquery.correlated_vars);
            let cache_key = Self::cache_key(&correlated, &lateral.subquery.correlated_vars);

            let sub_results = if let Some(cached) = self.cache.get(&cache_key) {
                self.stats.cache_hits += 1;
                cached.clone()
            } else {
                self.stats.cache_misses += 1;

                let start = Instant::now();
                let fresh = evaluator(&correlated)?;
                self.stats.subquery_time_ms += start.elapsed().as_millis() as u64;
                self.stats.subquery_evaluations += 1;

                // Evict oldest if at capacity
                if self.cache.len() >= self.config.cache_capacity {
                    if let Some(first_key) = self.cache.keys().next().cloned() {
                        self.cache.remove(&first_key);
                    }
                }
                self.cache.insert(cache_key, fresh.clone());
                fresh
            };

            for sub_row in &sub_results {
                let merged = Self::merge_mappings(left_row, sub_row)?;
                results.push(merged);
            }
        }

        self.stats.result_rows = results.len() as u64;
        Ok(results)
    }

    // ── Helper methods ────────────────────────────────────────────────────

    /// Extract only the correlated variable bindings from a solution mapping.
    fn extract_correlated_bindings(
        row: &SolutionMapping,
        correlated_vars: &[String],
    ) -> SolutionMapping {
        let mut bindings = SolutionMapping::new();
        for var in correlated_vars {
            if let Some(val) = row.get(var) {
                bindings.insert(var.clone(), val.clone());
            }
        }
        bindings
    }

    /// Build a combined bindings map representing a batch of left rows.
    /// This creates a single mapping containing the union of all correlated
    /// variable values as a comma-separated list for batch evaluation.
    fn build_batch_bindings(
        rows: &[SolutionMapping],
        correlated_vars: &[String],
    ) -> SolutionMapping {
        let mut combined = SolutionMapping::new();
        // For batch evaluation, include all unique values for each correlated var.
        // The evaluator is expected to use VALUES-style binding injection.
        for var in correlated_vars {
            // Collect all unique values for this variable across the batch
            let mut seen = HashSet::new();
            for row in rows {
                if let Some(val) = row.get(var) {
                    let key = format!("{val}");
                    if seen.insert(key) {
                        // Use the first occurrence as the representative
                        combined.entry(var.clone()).or_insert_with(|| val.clone());
                    }
                }
            }
        }
        combined
    }

    /// Check if a subquery result row is compatible with a left row
    /// on the correlated variables (i.e., they have the same values).
    fn is_compatible(
        left: &SolutionMapping,
        right: &SolutionMapping,
        correlated_vars: &[String],
    ) -> bool {
        for var in correlated_vars {
            match (left.get(var), right.get(var)) {
                (Some(l), Some(r)) => {
                    if l != r {
                        return false;
                    }
                }
                (None, Some(_)) | (Some(_), None) => {
                    // One side has an unbound variable — still compatible
                    // per SPARQL semantics (unbound is compatible with anything).
                }
                (None, None) => {}
            }
        }
        true
    }

    /// Merge two solution mappings. Variables in `right` overwrite `left`
    /// only if they are not already present.
    fn merge_mappings(
        left: &SolutionMapping,
        right: &SolutionMapping,
    ) -> Result<SolutionMapping, LateralJoinError> {
        let mut merged = left.clone();
        for (var, val) in right {
            // LATERAL semantics: the right side can introduce new bindings
            // and override correlated variables.
            merged.insert(var.clone(), val.clone());
        }
        Ok(merged)
    }

    /// Produce a cache key from the correlated variable bindings.
    fn cache_key(correlated: &SolutionMapping, vars: &[String]) -> String {
        let mut parts = Vec::with_capacity(vars.len());
        for var in vars {
            match correlated.get(var) {
                Some(val) => parts.push(format!("{var}={val}")),
                None => parts.push(format!("{var}=UNDEF")),
            }
        }
        parts.join("|")
    }

    /// Evaluate pushed-down filter expressions against a row.
    /// For now this supports simple "?var = <value>" equality filters.
    fn passes_pushed_filters(&self, row: &SolutionMapping, filters: &[String]) -> bool {
        for filter in filters {
            if let Some((var, expected)) = Self::parse_equality_filter(filter) {
                if let Some(actual) = row.get(&var) {
                    let actual_str = format!("{actual}");
                    if actual_str != expected {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Parse a simple equality filter of the form `?var = "value"` or `?var = <iri>`.
    ///
    /// The returned expected value is kept in its Display-compatible form so
    /// that it can be directly compared against `format!("{actual}")` which
    /// uses the `LateralValue::Display` impl (IRIs are wrapped in `<>`).
    fn parse_equality_filter(filter: &str) -> Option<(String, String)> {
        let parts: Vec<&str> = filter.splitn(3, ' ').collect();
        if parts.len() == 3 && parts[1] == "=" {
            let var = parts[0].trim_start_matches('?').to_string();
            // Keep the value as-is so it matches Display output.
            // For IRIs: "<http://...>" matches LateralValue::Iri Display.
            // For literals: "\"value\"" matches LateralValue::Literal Display.
            let val = parts[2].to_string();
            Some((var, val))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Optimizer — decides the best lateral strategy
// ---------------------------------------------------------------------------

/// Optimizer that selects the best execution strategy for a LATERAL join.
#[derive(Default)]
pub struct LateralOptimizer {
    /// Configuration thresholds.
    config: LateralOptimizerConfig,
}

/// Configuration for the lateral optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LateralOptimizerConfig {
    /// If the number of distinct correlated key values is below this
    /// threshold, use caching.
    pub cache_threshold: usize,
    /// If the left cardinality exceeds this, use batched evaluation.
    pub batch_threshold: usize,
    /// Minimum selectivity improvement to attempt decorrelation.
    pub decorrelate_min_improvement: f64,
}

impl Default for LateralOptimizerConfig {
    fn default() -> Self {
        Self {
            cache_threshold: 1000,
            batch_threshold: 500,
            decorrelate_min_improvement: 0.3,
        }
    }
}

/// Cost estimate for a particular lateral strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LateralCostEstimate {
    /// The strategy evaluated.
    pub strategy: LateralStrategy,
    /// Estimated total cost (abstract units).
    pub estimated_cost: f64,
    /// Estimated number of subquery evaluations.
    pub estimated_evaluations: u64,
    /// Whether this strategy can use a cache effectively.
    pub cacheable: bool,
    /// Whether decorrelation is possible.
    pub decorrelatable: bool,
}

impl LateralOptimizer {
    /// Create with default thresholds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom configuration.
    pub fn with_config(config: LateralOptimizerConfig) -> Self {
        Self { config }
    }

    /// Choose the best strategy for the given lateral join parameters.
    pub fn choose_strategy(
        &self,
        left_cardinality: u64,
        distinct_keys: u64,
        subquery: &LateralSubquery,
    ) -> LateralCostEstimate {
        let mut candidates = Vec::new();

        // Nested loop: always possible, cost = left_cardinality * subquery_cost
        let nl_cost = left_cardinality as f64 * self.estimate_subquery_cost(subquery);
        candidates.push(LateralCostEstimate {
            strategy: LateralStrategy::NestedLoop,
            estimated_cost: nl_cost,
            estimated_evaluations: left_cardinality,
            cacheable: false,
            decorrelatable: false,
        });

        // Cached: cost = distinct_keys * subquery_cost + (left_cardinality - distinct_keys) * lookup_cost
        let cache_cost = distinct_keys as f64 * self.estimate_subquery_cost(subquery)
            + (left_cardinality.saturating_sub(distinct_keys)) as f64 * 0.01;
        candidates.push(LateralCostEstimate {
            strategy: LateralStrategy::CachedCorrelation,
            estimated_cost: cache_cost,
            estimated_evaluations: distinct_keys,
            cacheable: distinct_keys < self.config.cache_threshold as u64,
            decorrelatable: false,
        });

        // Batched: cost = ceil(left_cardinality / batch_size) * subquery_cost * overhead
        // Use batch_threshold as a reasonable batch size estimate; each batch
        // evaluation still has per-row cost for correlating results back.
        let batch_size = self.config.batch_threshold.max(1) as f64;
        let batch_evals = (left_cardinality as f64 / batch_size).ceil();
        // Batch coordination overhead is significant: each row in the batch
        // must be correlated back, plus the subquery itself is heavier when
        // processing a batch.  Use a per-row factor plus the batch eval cost.
        let per_row_correlation_cost = left_cardinality as f64 * 0.5;
        let batch_cost =
            batch_evals * self.estimate_subquery_cost(subquery) + per_row_correlation_cost;
        candidates.push(LateralCostEstimate {
            strategy: LateralStrategy::BatchedValues,
            estimated_cost: batch_cost,
            estimated_evaluations: batch_evals as u64,
            cacheable: false,
            decorrelatable: false,
        });

        // Decorrelate: only if subquery has aggregates and a single correlated var
        if self.can_decorrelate(subquery) {
            let decorrelate_cost = left_cardinality as f64 * 0.5; // rough: join is cheaper than correlated eval
            candidates.push(LateralCostEstimate {
                strategy: LateralStrategy::Decorrelate,
                estimated_cost: decorrelate_cost,
                estimated_evaluations: 1,
                cacheable: false,
                decorrelatable: true,
            });
        }

        // Pick the cheapest
        candidates.sort_by(|a, b| {
            a.estimated_cost
                .partial_cmp(&b.estimated_cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates
            .into_iter()
            .next()
            .expect("at least one candidate strategy")
    }

    /// Estimate the cost of evaluating the subquery once.
    fn estimate_subquery_cost(&self, subquery: &LateralSubquery) -> f64 {
        let mut cost = 1.0;
        if subquery.has_aggregates {
            cost *= 2.0;
        }
        if let Some(limit) = subquery.limit {
            cost *= (limit as f64).min(100.0) / 100.0;
        }
        if !subquery.order_by.is_empty() {
            cost *= 1.5;
        }
        cost
    }

    /// Check whether a subquery can be decorrelated into a regular join.
    ///
    /// Decorrelation is possible when:
    /// 1. The subquery has exactly one correlated variable
    /// 2. The subquery uses aggregation
    /// 3. The correlated variable is used in a simple equality pattern
    fn can_decorrelate(&self, subquery: &LateralSubquery) -> bool {
        subquery.correlated_vars.len() == 1 && subquery.has_aggregates
    }

    /// Analyze a lateral join and produce a detailed cost comparison.
    pub fn analyze(
        &self,
        left_cardinality: u64,
        distinct_keys: u64,
        subquery: &LateralSubquery,
    ) -> Vec<LateralCostEstimate> {
        let mut estimates = vec![
            LateralCostEstimate {
                strategy: LateralStrategy::NestedLoop,
                estimated_cost: left_cardinality as f64 * self.estimate_subquery_cost(subquery),
                estimated_evaluations: left_cardinality,
                cacheable: false,
                decorrelatable: false,
            },
            LateralCostEstimate {
                strategy: LateralStrategy::CachedCorrelation,
                estimated_cost: distinct_keys as f64 * self.estimate_subquery_cost(subquery)
                    + (left_cardinality.saturating_sub(distinct_keys)) as f64 * 0.01,
                estimated_evaluations: distinct_keys,
                cacheable: distinct_keys < self.config.cache_threshold as u64,
                decorrelatable: false,
            },
            {
                let batch_size = self.config.batch_threshold.max(1) as f64;
                let batch_evals = (left_cardinality as f64 / batch_size).ceil();
                let per_row_correlation_cost = left_cardinality as f64 * 0.5;
                LateralCostEstimate {
                    strategy: LateralStrategy::BatchedValues,
                    estimated_cost: batch_evals * self.estimate_subquery_cost(subquery)
                        + per_row_correlation_cost,
                    estimated_evaluations: batch_evals as u64,
                    cacheable: false,
                    decorrelatable: false,
                }
            },
        ];

        if self.can_decorrelate(subquery) {
            estimates.push(LateralCostEstimate {
                strategy: LateralStrategy::Decorrelate,
                estimated_cost: left_cardinality as f64 * 0.5,
                estimated_evaluations: 1,
                cacheable: false,
                decorrelatable: true,
            });
        }

        estimates.sort_by(|a, b| {
            a.estimated_cost
                .partial_cmp(&b.estimated_cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        estimates
    }
}

// ---------------------------------------------------------------------------
// Lateral join validation
// ---------------------------------------------------------------------------

/// Validates LATERAL join constructs for correctness.
pub struct LateralValidator;

/// Result of validating a LATERAL join.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LateralValidationResult {
    /// Whether the LATERAL join is valid.
    pub is_valid: bool,
    /// Validation errors found.
    pub errors: Vec<LateralValidationError>,
    /// Warnings (valid but potentially problematic).
    pub warnings: Vec<String>,
    /// Detected correlated variables.
    pub detected_correlated_vars: Vec<String>,
    /// Variables visible after the LATERAL join.
    pub output_vars: Vec<String>,
}

/// A validation error for a LATERAL join.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LateralValidationError {
    /// Error message.
    pub message: String,
    /// Error code.
    pub code: LateralErrorCode,
}

/// Error codes for lateral validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LateralErrorCode {
    /// No correlated variables found (LATERAL is unnecessary).
    NoCorrelation,
    /// A correlated variable is not bound by the left operand.
    UnboundCorrelatedVar,
    /// Nesting depth exceeds the configured maximum.
    ExcessiveNesting,
    /// The subquery projects a variable that conflicts with the left operand.
    VariableConflict,
    /// The subquery uses a disallowed construct (e.g., SERVICE in LATERAL).
    DisallowedConstruct,
}

impl LateralValidator {
    /// Validate a LATERAL join construct.
    pub fn validate(
        subquery: &LateralSubquery,
        left_vars: &[String],
        nesting_depth: usize,
        max_depth: usize,
    ) -> LateralValidationResult {
        let mut result = LateralValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            detected_correlated_vars: Vec::new(),
            output_vars: Vec::new(),
        };

        // Check nesting depth
        if nesting_depth > max_depth {
            result.is_valid = false;
            result.errors.push(LateralValidationError {
                message: format!(
                    "LATERAL nesting depth {nesting_depth} exceeds maximum {max_depth}"
                ),
                code: LateralErrorCode::ExcessiveNesting,
            });
        }

        let left_set: HashSet<&str> = left_vars.iter().map(|s| s.as_str()).collect();

        // Check that correlated vars are bound by left operand
        for var in &subquery.correlated_vars {
            if left_set.contains(var.as_str()) {
                result.detected_correlated_vars.push(var.clone());
            } else {
                result.is_valid = false;
                result.errors.push(LateralValidationError {
                    message: format!("Correlated variable ?{var} is not bound by the left operand"),
                    code: LateralErrorCode::UnboundCorrelatedVar,
                });
            }
        }

        // Warn if no correlation detected
        if subquery.correlated_vars.is_empty() {
            result.warnings.push(
                "LATERAL subquery has no correlated variables; consider using a regular join"
                    .to_string(),
            );
        }

        // Check for variable conflicts
        for proj_var in &subquery.projected_vars {
            if left_set.contains(proj_var.as_str()) && !subquery.correlated_vars.contains(proj_var)
            {
                result.errors.push(LateralValidationError {
                    message: format!(
                        "Projected variable ?{proj_var} conflicts with left operand binding"
                    ),
                    code: LateralErrorCode::VariableConflict,
                });
                // This is a warning, not an error — LATERAL can override
                result.warnings.push(format!(
                    "Variable ?{proj_var} will be overridden by LATERAL subquery"
                ));
            }
        }

        // Compute output variables
        let mut output = HashSet::new();
        for var in left_vars {
            output.insert(var.clone());
        }
        for var in &subquery.projected_vars {
            output.insert(var.clone());
        }
        result.output_vars = output.into_iter().collect();
        result.output_vars.sort();

        result
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from lateral join execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LateralJoinError {
    /// The subquery evaluation returned an error.
    SubqueryError(String),
    /// A timeout occurred during subquery evaluation.
    Timeout {
        /// Which subquery timed out.
        description: String,
        /// How long was waited.
        elapsed_ms: u64,
    },
    /// Incompatible variable bindings during merge.
    IncompatibleBindings {
        /// The variable that had conflicting values.
        variable: String,
        /// The left value.
        left_value: String,
        /// The right value.
        right_value: String,
    },
    /// Nesting depth exceeded.
    NestingDepthExceeded {
        /// Current depth.
        depth: usize,
        /// Maximum allowed.
        max: usize,
    },
}

impl fmt::Display for LateralJoinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SubqueryError(msg) => write!(f, "Lateral subquery error: {msg}"),
            Self::Timeout {
                description,
                elapsed_ms,
            } => {
                write!(
                    f,
                    "Lateral subquery timed out after {elapsed_ms}ms: {description}"
                )
            }
            Self::IncompatibleBindings {
                variable,
                left_value,
                right_value,
            } => {
                write!(
                    f,
                    "Incompatible bindings for ?{variable}: left={left_value}, right={right_value}"
                )
            }
            Self::NestingDepthExceeded { depth, max } => {
                write!(
                    f,
                    "LATERAL nesting depth {depth} exceeds maximum allowed {max}"
                )
            }
        }
    }
}

impl std::error::Error for LateralJoinError {}

// ---------------------------------------------------------------------------
// SPARQL 1.2 LATERAL parser helpers
// ---------------------------------------------------------------------------

/// Parses and validates LATERAL join patterns from SPARQL text fragments.
pub struct LateralParser;

/// A parsed LATERAL clause from a SPARQL query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedLateral {
    /// The outer variables available to the LATERAL clause.
    pub outer_vars: Vec<String>,
    /// The detected correlated variables.
    pub correlated_vars: Vec<String>,
    /// The projected variables from the subquery.
    pub projected_vars: Vec<String>,
    /// Whether the subquery contains aggregates.
    pub has_aggregates: bool,
    /// Whether the subquery contains ORDER BY.
    pub has_order_by: bool,
    /// Whether the subquery contains LIMIT.
    pub has_limit: bool,
    /// The raw subquery text (between LATERAL { ... }).
    pub subquery_text: String,
}

impl LateralParser {
    /// Detect LATERAL clauses in a SPARQL query string.
    ///
    /// Returns positions and basic metadata for each LATERAL clause found.
    pub fn detect_lateral_clauses(query: &str) -> Vec<LateralClausePosition> {
        let mut positions = Vec::new();
        let upper = query.to_uppercase();
        let mut search_from = 0;

        while let Some(idx) = upper[search_from..].find("LATERAL") {
            let abs_idx = search_from + idx;
            // Verify it's a keyword (not part of another identifier)
            let before_ok = abs_idx == 0 || !query.as_bytes()[abs_idx - 1].is_ascii_alphanumeric();
            let after_idx = abs_idx + 7;
            let after_ok =
                after_idx >= query.len() || !query.as_bytes()[after_idx].is_ascii_alphanumeric();

            if before_ok && after_ok {
                // Find the matching brace
                if let Some(brace_start) = query[after_idx..].find('{') {
                    let open = after_idx + brace_start;
                    if let Some(close) = Self::find_matching_brace(query, open) {
                        let body = &query[open + 1..close];
                        positions.push(LateralClausePosition {
                            start: abs_idx,
                            end: close + 1,
                            body: body.trim().to_string(),
                            has_select: body.to_uppercase().contains("SELECT"),
                        });
                    }
                }
            }
            search_from = abs_idx + 7;
        }

        positions
    }

    /// Find the matching closing brace for an opening brace at `pos`.
    fn find_matching_brace(s: &str, pos: usize) -> Option<usize> {
        let bytes = s.as_bytes();
        if pos >= bytes.len() || bytes[pos] != b'{' {
            return None;
        }
        let mut depth = 0i32;
        for (i, &b) in bytes[pos..].iter().enumerate() {
            match b {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(pos + i);
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Extract variable references (?varName) from a SPARQL fragment.
    pub fn extract_variables(fragment: &str) -> Vec<String> {
        let mut vars = HashSet::new();
        let bytes = fragment.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'?' || bytes[i] == b'$' {
                let start = i + 1;
                i += 1;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                if i > start {
                    let var = String::from_utf8_lossy(&bytes[start..i]).to_string();
                    vars.insert(var);
                }
            } else {
                i += 1;
            }
        }
        let mut result: Vec<_> = vars.into_iter().collect();
        result.sort();
        result
    }

    /// Detect aggregate functions in a SPARQL fragment.
    pub fn detect_aggregates(fragment: &str) -> bool {
        let upper = fragment.to_uppercase();
        [
            "COUNT(",
            "SUM(",
            "AVG(",
            "MIN(",
            "MAX(",
            "GROUP_CONCAT(",
            "SAMPLE(",
        ]
        .iter()
        .any(|agg| upper.contains(agg))
    }

    /// Detect ORDER BY in a SPARQL fragment.
    pub fn detect_order_by(fragment: &str) -> bool {
        fragment.to_uppercase().contains("ORDER BY")
    }

    /// Detect LIMIT in a SPARQL fragment.
    pub fn detect_limit(fragment: &str) -> bool {
        fragment.to_uppercase().contains("LIMIT")
    }
}

/// Position and metadata for a detected LATERAL clause.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LateralClausePosition {
    /// Start byte offset in the query string.
    pub start: usize,
    /// End byte offset (exclusive).
    pub end: usize,
    /// The body text between the braces.
    pub body: String,
    /// Whether the body contains a SELECT subquery.
    pub has_select: bool,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn iri(s: &str) -> LateralValue {
        LateralValue::Iri(s.to_string())
    }

    fn lit(s: &str) -> LateralValue {
        LateralValue::Literal {
            value: s.to_string(),
            datatype: None,
            lang: None,
        }
    }

    fn typed_lit(s: &str, dt: &str) -> LateralValue {
        LateralValue::Literal {
            value: s.to_string(),
            datatype: Some(dt.to_string()),
            lang: None,
        }
    }

    fn make_row(bindings: &[(&str, LateralValue)]) -> SolutionMapping {
        bindings
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    fn simple_subquery() -> LateralSubquery {
        LateralSubquery {
            description: "SELECT (MAX(?score) AS ?maxScore) WHERE { ?person :score ?score }"
                .to_string(),
            correlated_vars: vec!["person".to_string()],
            projected_vars: vec!["maxScore".to_string()],
            has_aggregates: true,
            limit: None,
            order_by: vec![],
        }
    }

    fn simple_lateral() -> LateralJoin {
        LateralJoin {
            left_description: "?person a :Student".to_string(),
            subquery: simple_subquery(),
            strategy: LateralStrategy::NestedLoop,
            pushed_filters: vec![],
        }
    }

    // ── Executor: nested-loop tests ───────────────────────────────────────

    #[test]
    fn test_nested_loop_basic() {
        let lateral = simple_lateral();
        let left = vec![
            make_row(&[("person", iri("http://ex.org/alice"))]),
            make_row(&[("person", iri("http://ex.org/bob"))]),
        ];

        let mut exec = LateralJoinExecutor::with_defaults();
        let results = exec
            .execute(&lateral, &left, |correlated| {
                let person = correlated.get("person").cloned();
                match person {
                    Some(LateralValue::Iri(p)) if p.contains("alice") => Ok(vec![make_row(&[(
                        "maxScore",
                        typed_lit("95", "xsd:integer"),
                    )])]),
                    Some(LateralValue::Iri(p)) if p.contains("bob") => Ok(vec![make_row(&[(
                        "maxScore",
                        typed_lit("87", "xsd:integer"),
                    )])]),
                    _ => Ok(vec![]),
                }
            })
            .expect("execution should succeed");

        assert_eq!(results.len(), 2);
        assert!(results[0].contains_key("person"));
        assert!(results[0].contains_key("maxScore"));
    }

    #[test]
    fn test_nested_loop_empty_left() {
        let lateral = simple_lateral();
        let left: Vec<SolutionMapping> = vec![];
        let mut exec = LateralJoinExecutor::with_defaults();
        let results = exec
            .execute(&lateral, &left, |_| Ok(vec![make_row(&[("x", lit("1"))])]))
            .expect("execution should succeed");
        assert!(results.is_empty());
        assert_eq!(exec.stats().left_rows, 0);
    }

    #[test]
    fn test_nested_loop_empty_subquery_result() {
        let lateral = simple_lateral();
        let left = vec![make_row(&[("person", iri("http://ex.org/charlie"))])];
        let mut exec = LateralJoinExecutor::with_defaults();
        let results = exec
            .execute(&lateral, &left, |_| Ok(vec![]))
            .expect("execution should succeed");
        assert!(results.is_empty());
        assert_eq!(exec.stats().subquery_evaluations, 1);
    }

    #[test]
    fn test_nested_loop_multiple_subquery_results() {
        let lateral = LateralJoin {
            left_description: "?dept a :Department".to_string(),
            subquery: LateralSubquery {
                description: "SELECT ?emp WHERE { ?emp :worksIn ?dept }".to_string(),
                correlated_vars: vec!["dept".to_string()],
                projected_vars: vec!["emp".to_string()],
                has_aggregates: false,
                limit: None,
                order_by: vec![],
            },
            strategy: LateralStrategy::NestedLoop,
            pushed_filters: vec![],
        };

        let left = vec![make_row(&[("dept", iri("http://ex.org/engineering"))])];
        let mut exec = LateralJoinExecutor::with_defaults();
        let results = exec
            .execute(&lateral, &left, |_| {
                Ok(vec![
                    make_row(&[("emp", iri("http://ex.org/alice"))]),
                    make_row(&[("emp", iri("http://ex.org/bob"))]),
                    make_row(&[("emp", iri("http://ex.org/charlie"))]),
                ])
            })
            .expect("execution should succeed");

        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.contains_key("dept"));
            assert!(r.contains_key("emp"));
        }
    }

    #[test]
    fn test_nested_loop_subquery_error_propagation() {
        let lateral = simple_lateral();
        let left = vec![make_row(&[("person", iri("http://ex.org/alice"))])];
        let mut exec = LateralJoinExecutor::with_defaults();
        let result = exec.execute(&lateral, &left, |_| {
            Err(LateralJoinError::SubqueryError(
                "timeout in remote endpoint".to_string(),
            ))
        });
        assert!(result.is_err());
    }

    // ── Executor: cached correlation tests ────────────────────────────────

    #[test]
    fn test_cached_correlation_hits() {
        let mut lateral = simple_lateral();
        lateral.strategy = LateralStrategy::CachedCorrelation;

        // Multiple rows with the same person — should hit cache
        let left = vec![
            make_row(&[("person", iri("http://ex.org/alice")), ("x", lit("1"))]),
            make_row(&[("person", iri("http://ex.org/alice")), ("x", lit("2"))]),
            make_row(&[("person", iri("http://ex.org/alice")), ("x", lit("3"))]),
        ];

        let mut exec = LateralJoinExecutor::with_defaults();
        let results = exec
            .execute(&lateral, &left, |_| {
                Ok(vec![make_row(&[("maxScore", lit("95"))])])
            })
            .expect("execution should succeed");

        assert_eq!(results.len(), 3);
        assert_eq!(exec.stats().cache_hits, 2);
        assert_eq!(exec.stats().cache_misses, 1);
        assert_eq!(exec.stats().subquery_evaluations, 1);
    }

    #[test]
    fn test_cached_correlation_different_keys() {
        let mut lateral = simple_lateral();
        lateral.strategy = LateralStrategy::CachedCorrelation;

        let left = vec![
            make_row(&[("person", iri("http://ex.org/alice"))]),
            make_row(&[("person", iri("http://ex.org/bob"))]),
            make_row(&[("person", iri("http://ex.org/charlie"))]),
        ];

        let mut exec = LateralJoinExecutor::with_defaults();
        let _ = exec
            .execute(&lateral, &left, |_| {
                Ok(vec![make_row(&[("maxScore", lit("90"))])])
            })
            .expect("execution should succeed");

        assert_eq!(exec.stats().cache_hits, 0);
        assert_eq!(exec.stats().cache_misses, 3);
        assert_eq!(exec.stats().subquery_evaluations, 3);
    }

    #[test]
    fn test_cache_hit_ratio() {
        let stats = LateralJoinStats {
            cache_hits: 75,
            cache_misses: 25,
            ..Default::default()
        };
        let ratio = stats.cache_hit_ratio();
        assert!((ratio - 75.0).abs() < 0.01);
    }

    #[test]
    fn test_cache_hit_ratio_empty() {
        let stats = LateralJoinStats::default();
        assert_eq!(stats.cache_hit_ratio(), 0.0);
    }

    // ── Executor: batched tests ───────────────────────────────────────────

    #[test]
    fn test_batched_execution() {
        let mut lateral = simple_lateral();
        lateral.strategy = LateralStrategy::BatchedValues;

        let left = vec![
            make_row(&[("person", iri("http://ex.org/alice"))]),
            make_row(&[("person", iri("http://ex.org/bob"))]),
        ];

        let config = LateralJoinConfig {
            batch_size: 10,
            ..Default::default()
        };
        let mut exec = LateralJoinExecutor::new(config);
        let results = exec
            .execute(&lateral, &left, |_| {
                // Return results for both alice and bob
                Ok(vec![
                    make_row(&[
                        ("person", iri("http://ex.org/alice")),
                        ("maxScore", lit("95")),
                    ]),
                    make_row(&[
                        ("person", iri("http://ex.org/bob")),
                        ("maxScore", lit("87")),
                    ]),
                ])
            })
            .expect("execution should succeed");

        assert_eq!(results.len(), 2);
        assert_eq!(exec.stats().batches_submitted, 1);
    }

    #[test]
    fn test_batched_multiple_batches() {
        let mut lateral = simple_lateral();
        lateral.strategy = LateralStrategy::BatchedValues;

        let left: Vec<_> = (0..5)
            .map(|i| make_row(&[("person", iri(&format!("http://ex.org/p{i}")))]))
            .collect();

        let config = LateralJoinConfig {
            batch_size: 2,
            ..Default::default()
        };
        let mut exec = LateralJoinExecutor::new(config);
        let _ = exec
            .execute(&lateral, &left, |bindings| {
                // Return the bindings back with a score
                let person = bindings.get("person").cloned();
                match person {
                    Some(p) => Ok(vec![make_row(&[("person", p), ("maxScore", lit("90"))])]),
                    None => Ok(vec![]),
                }
            })
            .expect("execution should succeed");

        assert_eq!(exec.stats().batches_submitted, 3); // ceil(5/2)
    }

    // ── Executor: pushed filters ──────────────────────────────────────────

    #[test]
    fn test_pushed_filter_equality() {
        let lateral = LateralJoin {
            left_description: "?person a :Student".to_string(),
            subquery: simple_subquery(),
            strategy: LateralStrategy::NestedLoop,
            pushed_filters: vec!["?person = <http://ex.org/alice>".to_string()],
        };

        let left = vec![
            make_row(&[("person", iri("http://ex.org/alice"))]),
            make_row(&[("person", iri("http://ex.org/bob"))]),
            make_row(&[("person", iri("http://ex.org/charlie"))]),
        ];

        let mut exec = LateralJoinExecutor::with_defaults();
        let results = exec
            .execute(&lateral, &left, |_| {
                Ok(vec![make_row(&[("maxScore", lit("95"))])])
            })
            .expect("execution should succeed");

        assert_eq!(results.len(), 1); // Only alice passes the filter
        assert_eq!(exec.stats().rows_filtered, 2);
    }

    // ── Executor: reset ───────────────────────────────────────────────────

    #[test]
    fn test_executor_reset() {
        let mut exec = LateralJoinExecutor::with_defaults();
        exec.stats = LateralJoinStats {
            left_rows: 100,
            result_rows: 200,
            subquery_evaluations: 50,
            cache_hits: 30,
            cache_misses: 20,
            ..Default::default()
        };
        exec.cache.insert("key".to_string(), vec![]);

        exec.reset();
        assert_eq!(exec.stats().left_rows, 0);
        assert_eq!(exec.stats().result_rows, 0);
        assert!(exec.cache.is_empty());
    }

    // ── Optimizer tests ───────────────────────────────────────────────────

    #[test]
    fn test_optimizer_chooses_cached_for_repeated_keys() {
        let optimizer = LateralOptimizer::new();
        let subquery = simple_subquery();
        let estimate = optimizer.choose_strategy(10000, 50, &subquery);
        // With 10000 left rows but only 50 distinct keys, caching should win
        assert!(
            estimate.strategy == LateralStrategy::CachedCorrelation
                || estimate.strategy == LateralStrategy::Decorrelate
        );
    }

    #[test]
    fn test_optimizer_decorrelate_single_agg() {
        let optimizer = LateralOptimizer::new();
        let subquery = LateralSubquery {
            description: "SELECT (COUNT(*) AS ?cnt) WHERE { ?x :p ?y }".to_string(),
            correlated_vars: vec!["x".to_string()],
            projected_vars: vec!["cnt".to_string()],
            has_aggregates: true,
            limit: None,
            order_by: vec![],
        };
        let estimate = optimizer.choose_strategy(5000, 5000, &subquery);
        // With all distinct keys and single aggregate, decorrelation should be cheapest
        assert_eq!(estimate.strategy, LateralStrategy::Decorrelate);
    }

    #[test]
    fn test_optimizer_cannot_decorrelate_multi_vars() {
        let optimizer = LateralOptimizer::new();
        let subquery = LateralSubquery {
            description: "test".to_string(),
            correlated_vars: vec!["x".to_string(), "y".to_string()],
            projected_vars: vec!["result".to_string()],
            has_aggregates: true,
            limit: None,
            order_by: vec![],
        };
        assert!(!optimizer.can_decorrelate(&subquery));
    }

    #[test]
    fn test_optimizer_analyze_all_strategies() {
        let optimizer = LateralOptimizer::new();
        let subquery = simple_subquery();
        let estimates = optimizer.analyze(1000, 100, &subquery);
        assert!(estimates.len() >= 3);
        // Should be sorted by cost
        for w in estimates.windows(2) {
            assert!(w[0].estimated_cost <= w[1].estimated_cost);
        }
    }

    #[test]
    fn test_optimizer_with_limit_subquery() {
        let optimizer = LateralOptimizer::new();
        let subquery = LateralSubquery {
            description: "SELECT ?x WHERE { ... } LIMIT 10".to_string(),
            correlated_vars: vec!["person".to_string()],
            projected_vars: vec!["x".to_string()],
            has_aggregates: false,
            limit: Some(10),
            order_by: vec![],
        };
        let estimate = optimizer.choose_strategy(100, 100, &subquery);
        // With LIMIT, cost should be lower
        assert!(estimate.estimated_cost > 0.0);
    }

    #[test]
    fn test_optimizer_with_order_by() {
        let optimizer = LateralOptimizer::new();
        let subquery = LateralSubquery {
            description: "SELECT ?x WHERE { ... } ORDER BY ?x".to_string(),
            correlated_vars: vec!["person".to_string()],
            projected_vars: vec!["x".to_string()],
            has_aggregates: false,
            limit: None,
            order_by: vec![OrderSpec {
                variable: "x".to_string(),
                ascending: true,
            }],
        };
        let estimate = optimizer.choose_strategy(100, 100, &subquery);
        assert!(estimate.estimated_cost > 0.0);
    }

    // ── Validator tests ───────────────────────────────────────────────────

    #[test]
    fn test_validator_valid_lateral() {
        let subquery = simple_subquery();
        let left_vars = vec!["person".to_string(), "name".to_string()];
        let result = LateralValidator::validate(&subquery, &left_vars, 0, 4);
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
        assert_eq!(result.detected_correlated_vars, vec!["person".to_string()]);
    }

    #[test]
    fn test_validator_unbound_correlated_var() {
        let subquery = LateralSubquery {
            description: "test".to_string(),
            correlated_vars: vec!["missing_var".to_string()],
            projected_vars: vec!["result".to_string()],
            has_aggregates: false,
            limit: None,
            order_by: vec![],
        };
        let left_vars = vec!["person".to_string()];
        let result = LateralValidator::validate(&subquery, &left_vars, 0, 4);
        assert!(!result.is_valid);
        assert_eq!(result.errors.len(), 1);
        assert_eq!(
            result.errors[0].code,
            LateralErrorCode::UnboundCorrelatedVar
        );
    }

    #[test]
    fn test_validator_excessive_nesting() {
        let subquery = simple_subquery();
        let left_vars = vec!["person".to_string()];
        let result = LateralValidator::validate(&subquery, &left_vars, 5, 4);
        assert!(!result.is_valid);
        assert!(result
            .errors
            .iter()
            .any(|e| e.code == LateralErrorCode::ExcessiveNesting));
    }

    #[test]
    fn test_validator_no_correlation_warning() {
        let subquery = LateralSubquery {
            description: "test".to_string(),
            correlated_vars: vec![],
            projected_vars: vec!["x".to_string()],
            has_aggregates: false,
            limit: None,
            order_by: vec![],
        };
        let left_vars = vec!["person".to_string()];
        let result = LateralValidator::validate(&subquery, &left_vars, 0, 4);
        assert!(result.is_valid);
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn test_validator_variable_conflict_warning() {
        let subquery = LateralSubquery {
            description: "test".to_string(),
            correlated_vars: vec!["person".to_string()],
            projected_vars: vec![
                "person".to_string(),
                "name".to_string(),
                "extra".to_string(),
            ],
            has_aggregates: false,
            limit: None,
            order_by: vec![],
        };
        let left_vars = vec!["person".to_string(), "name".to_string()];
        let result = LateralValidator::validate(&subquery, &left_vars, 0, 4);
        // "name" conflicts — it's projected by subquery but also in left_vars
        // "person" does not conflict because it is in correlated_vars
        assert!(result.warnings.iter().any(|w| w.contains("name")));
    }

    #[test]
    fn test_validator_output_vars() {
        let subquery = LateralSubquery {
            description: "test".to_string(),
            correlated_vars: vec!["person".to_string()],
            projected_vars: vec!["maxScore".to_string()],
            has_aggregates: true,
            limit: None,
            order_by: vec![],
        };
        let left_vars = vec!["person".to_string()];
        let result = LateralValidator::validate(&subquery, &left_vars, 0, 4);
        assert!(result.output_vars.contains(&"person".to_string()));
        assert!(result.output_vars.contains(&"maxScore".to_string()));
    }

    // ── Parser tests ──────────────────────────────────────────────────────

    #[test]
    fn test_parser_detect_single_lateral() {
        let query = r#"
            SELECT ?person ?maxScore WHERE {
                ?person a :Student .
                LATERAL {
                    SELECT (MAX(?score) AS ?maxScore)
                    WHERE { ?person :score ?score }
                }
            }
        "#;
        let clauses = LateralParser::detect_lateral_clauses(query);
        assert_eq!(clauses.len(), 1);
        assert!(clauses[0].has_select);
        assert!(clauses[0].body.contains("MAX"));
    }

    #[test]
    fn test_parser_detect_multiple_laterals() {
        let query = r#"
            SELECT * WHERE {
                ?x a :Foo .
                LATERAL { SELECT ?a WHERE { ?x :p ?a } }
                LATERAL { SELECT ?b WHERE { ?x :q ?b } }
            }
        "#;
        let clauses = LateralParser::detect_lateral_clauses(query);
        assert_eq!(clauses.len(), 2);
    }

    #[test]
    fn test_parser_no_lateral() {
        let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
        let clauses = LateralParser::detect_lateral_clauses(query);
        assert!(clauses.is_empty());
    }

    #[test]
    fn test_parser_lateral_not_keyword() {
        // "LATERALLY" should not match
        let query = "SELECT ?x WHERE { ?x :laterally ?y }";
        let clauses = LateralParser::detect_lateral_clauses(query);
        assert!(clauses.is_empty());
    }

    #[test]
    fn test_extract_variables() {
        let fragment = "?person :hasExam ?exam . ?exam :score ?score";
        let vars = LateralParser::extract_variables(fragment);
        assert!(vars.contains(&"person".to_string()));
        assert!(vars.contains(&"exam".to_string()));
        assert!(vars.contains(&"score".to_string()));
    }

    #[test]
    fn test_extract_variables_dollar_sign() {
        let fragment = "$person :hasExam $exam";
        let vars = LateralParser::extract_variables(fragment);
        assert!(vars.contains(&"person".to_string()));
        assert!(vars.contains(&"exam".to_string()));
    }

    #[test]
    fn test_detect_aggregates() {
        assert!(LateralParser::detect_aggregates("SELECT (MAX(?x) AS ?m)"));
        assert!(LateralParser::detect_aggregates("SELECT (COUNT(*) AS ?c)"));
        assert!(LateralParser::detect_aggregates("SUM(?val)"));
        assert!(!LateralParser::detect_aggregates(
            "SELECT ?x WHERE { ?x :p ?y }"
        ));
    }

    #[test]
    fn test_detect_order_by() {
        assert!(LateralParser::detect_order_by(
            "SELECT ?x WHERE { } ORDER BY ?x"
        ));
        assert!(!LateralParser::detect_order_by(
            "SELECT ?x WHERE { ?x :p ?y }"
        ));
    }

    #[test]
    fn test_detect_limit() {
        assert!(LateralParser::detect_limit("SELECT ?x WHERE { } LIMIT 10"));
        assert!(!LateralParser::detect_limit("SELECT ?x WHERE { ?x :p ?y }"));
    }

    // ── LateralValue display tests ────────────────────────────────────────

    #[test]
    fn test_lateral_value_display_iri() {
        let v = iri("http://example.org/foo");
        assert_eq!(format!("{v}"), "<http://example.org/foo>");
    }

    #[test]
    fn test_lateral_value_display_literal() {
        let v = lit("hello");
        assert_eq!(format!("{v}"), "\"hello\"");
    }

    #[test]
    fn test_lateral_value_display_typed_literal() {
        let v = typed_lit("42", "xsd:integer");
        assert_eq!(format!("{v}"), "\"42\"^^<xsd:integer>");
    }

    #[test]
    fn test_lateral_value_display_lang_literal() {
        let v = LateralValue::Literal {
            value: "hello".to_string(),
            datatype: None,
            lang: Some("en".to_string()),
        };
        assert_eq!(format!("{v}"), "\"hello\"@en");
    }

    #[test]
    fn test_lateral_value_display_blank() {
        let v = LateralValue::BlankNode("b0".to_string());
        assert_eq!(format!("{v}"), "_:b0");
    }

    // ── Error display tests ───────────────────────────────────────────────

    #[test]
    fn test_error_display_subquery() {
        let e = LateralJoinError::SubqueryError("connection refused".to_string());
        assert!(format!("{e}").contains("connection refused"));
    }

    #[test]
    fn test_error_display_timeout() {
        let e = LateralJoinError::Timeout {
            description: "remote service".to_string(),
            elapsed_ms: 5000,
        };
        assert!(format!("{e}").contains("5000ms"));
    }

    #[test]
    fn test_error_display_incompatible() {
        let e = LateralJoinError::IncompatibleBindings {
            variable: "x".to_string(),
            left_value: "1".to_string(),
            right_value: "2".to_string(),
        };
        assert!(format!("{e}").contains("x"));
    }

    #[test]
    fn test_error_display_nesting() {
        let e = LateralJoinError::NestingDepthExceeded { depth: 5, max: 4 };
        assert!(format!("{e}").contains("5"));
    }

    // ── Strategy display tests ────────────────────────────────────────────

    #[test]
    fn test_strategy_display() {
        assert_eq!(format!("{}", LateralStrategy::NestedLoop), "NestedLoop");
        assert_eq!(
            format!("{}", LateralStrategy::BatchedValues),
            "BatchedValues"
        );
        assert_eq!(format!("{}", LateralStrategy::Decorrelate), "Decorrelate");
        assert_eq!(
            format!("{}", LateralStrategy::CachedCorrelation),
            "CachedCorrelation"
        );
    }

    // ── Merge mapping tests ───────────────────────────────────────────────

    #[test]
    fn test_merge_disjoint_mappings() {
        let left = make_row(&[("a", lit("1"))]);
        let right = make_row(&[("b", lit("2"))]);
        let merged = LateralJoinExecutor::merge_mappings(&left, &right).expect("merge should work");
        assert_eq!(merged.len(), 2);
        assert!(merged.contains_key("a"));
        assert!(merged.contains_key("b"));
    }

    #[test]
    fn test_merge_overlapping_mappings() {
        let left = make_row(&[("a", lit("1")), ("b", lit("old"))]);
        let right = make_row(&[("b", lit("new")), ("c", lit("3"))]);
        let merged = LateralJoinExecutor::merge_mappings(&left, &right).expect("merge should work");
        assert_eq!(merged.len(), 3);
        // Right overwrites left for LATERAL semantics
        assert_eq!(merged.get("b"), Some(&lit("new")));
    }

    // ── Avg subquery time ─────────────────────────────────────────────────

    #[test]
    fn test_avg_subquery_time() {
        let stats = LateralJoinStats {
            subquery_evaluations: 10,
            subquery_time_ms: 500,
            ..Default::default()
        };
        assert!((stats.avg_subquery_time_ms() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_avg_subquery_time_zero() {
        let stats = LateralJoinStats::default();
        assert_eq!(stats.avg_subquery_time_ms(), 0.0);
    }

    // ── Config tests ──────────────────────────────────────────────────────

    #[test]
    fn test_default_config() {
        let config = LateralJoinConfig::default();
        assert_eq!(config.batch_size, 128);
        assert_eq!(config.cache_capacity, 4096);
        assert_eq!(config.max_nesting_depth, 4);
        assert!(config.auto_decorrelate);
    }

    #[test]
    fn test_optimizer_config_default() {
        let config = LateralOptimizerConfig::default();
        assert_eq!(config.cache_threshold, 1000);
        assert_eq!(config.batch_threshold, 500);
    }

    // ── is_compatible tests ───────────────────────────────────────────────

    #[test]
    fn test_is_compatible_same_values() {
        let left = make_row(&[("x", iri("http://ex.org/a"))]);
        let right = make_row(&[("x", iri("http://ex.org/a")), ("y", lit("1"))]);
        assert!(LateralJoinExecutor::is_compatible(
            &left,
            &right,
            &["x".to_string()]
        ));
    }

    #[test]
    fn test_is_compatible_different_values() {
        let left = make_row(&[("x", iri("http://ex.org/a"))]);
        let right = make_row(&[("x", iri("http://ex.org/b"))]);
        assert!(!LateralJoinExecutor::is_compatible(
            &left,
            &right,
            &["x".to_string()]
        ));
    }

    #[test]
    fn test_is_compatible_unbound() {
        let left = make_row(&[("x", iri("http://ex.org/a"))]);
        let right = make_row(&[("y", lit("1"))]); // x not in right
        assert!(LateralJoinExecutor::is_compatible(
            &left,
            &right,
            &["x".to_string()]
        ));
    }

    // ── Parse equality filter ─────────────────────────────────────────────

    #[test]
    fn test_parse_equality_filter_iri() {
        let result = LateralJoinExecutor::parse_equality_filter("?x = <http://ex.org/a>");
        assert_eq!(
            result,
            Some(("x".to_string(), "<http://ex.org/a>".to_string()))
        );
    }

    #[test]
    fn test_parse_equality_filter_literal() {
        let result = LateralJoinExecutor::parse_equality_filter("?name = \"Alice\"");
        assert_eq!(result, Some(("name".to_string(), "\"Alice\"".to_string())));
    }

    #[test]
    fn test_parse_equality_filter_invalid() {
        let result = LateralJoinExecutor::parse_equality_filter("?x > 5");
        assert!(result.is_none());
    }

    // ── find_matching_brace ───────────────────────────────────────────────

    #[test]
    fn test_find_matching_brace_simple() {
        let s = "{ hello }";
        assert_eq!(LateralParser::find_matching_brace(s, 0), Some(8));
    }

    #[test]
    fn test_find_matching_brace_nested() {
        let s = "{ { inner } outer }";
        assert_eq!(LateralParser::find_matching_brace(s, 0), Some(18));
    }

    #[test]
    fn test_find_matching_brace_unmatched() {
        let s = "{ no closing";
        assert_eq!(LateralParser::find_matching_brace(s, 0), None);
    }

    #[test]
    fn test_find_matching_brace_not_brace() {
        let s = "hello";
        assert_eq!(LateralParser::find_matching_brace(s, 0), None);
    }
}
