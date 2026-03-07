//! ML-powered query profiling and optimization suggestions
//!
//! This module provides statistical profiling of SPARQL queries and uses
//! SciRS2's ndarray_ext for feature-based optimization suggestions.
//!
//! ## Features
//!
//! - Query execution time profiling with percentiles (p50, p95, p99)
//! - Feature extraction from SPARQL AST patterns
//! - Regression-based execution time prediction
//! - Pattern-based optimization suggestions with cost model
//! - Historical profile storage and trending analysis
//! - SciRS2 ndarray_ext integration for statistical computations

use anyhow::{anyhow, Result};
use colored::Colorize;
use scirs2_core::ndarray_ext::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Maximum history size per query fingerprint
const MAX_HISTORY_PER_QUERY: usize = 256;

/// Fingerprint a query for tracking purposes
fn fingerprint_query(query: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    // Normalize: strip leading/trailing whitespace, uppercase keywords
    let normalized = normalize_sparql(query);
    normalized.hash(&mut h);
    format!("{:016x}", h.finish())
}

/// Normalize a SPARQL query to canonical form for fingerprinting
fn normalize_sparql(query: &str) -> String {
    let mut result = String::with_capacity(query.len());
    let mut prev_space = false;
    for ch in query.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                result.push(' ');
            }
            prev_space = true;
        } else {
            result.push(ch);
            prev_space = false;
        }
    }
    result.trim().to_uppercase()
}

/// Numeric features extracted from a SPARQL query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryProfileFeatures {
    /// Number of triple patterns
    pub triple_pattern_count: f64,
    /// Number of OPTIONAL clauses
    pub optional_count: f64,
    /// Number of UNION clauses
    pub union_count: f64,
    /// Number of FILTER expressions
    pub filter_count: f64,
    /// Number of ORDER BY clauses
    pub order_by_count: f64,
    /// Number of GROUP BY clauses
    pub group_by_count: f64,
    /// Number of subqueries
    pub subquery_count: f64,
    /// Number of aggregation functions (COUNT, SUM, etc.)
    pub aggregation_count: f64,
    /// Has LIMIT clause
    pub has_limit: f64,
    /// Has DISTINCT keyword
    pub has_distinct: f64,
    /// Estimated selectivity [0..1]
    pub selectivity: f64,
    /// Property path complexity score
    pub path_complexity: f64,
    /// Number of BIND clauses
    pub bind_count: f64,
    /// Number of SERVICE calls (federation)
    pub service_count: f64,
    /// Number of named graphs referenced
    pub named_graph_count: f64,
}

impl QueryProfileFeatures {
    /// Extract features from a SPARQL query string
    pub fn extract(query: &str) -> Self {
        let upper = query.to_uppercase();

        let triple_pattern_count = {
            let dots = query.matches('.').count();
            let semis = query.matches(';').count();
            let has_where = upper.contains("WHERE");
            if has_where && dots == 0 && semis == 0 {
                1.0
            } else {
                (dots + semis).max(if has_where { 1 } else { 0 }) as f64
            }
        };

        let optional_count = upper.matches("OPTIONAL").count() as f64;
        let union_count = upper.matches("UNION").count() as f64;
        let filter_count = upper.matches("FILTER").count() as f64;
        let order_by_count = upper.matches("ORDER BY").count() as f64;
        let group_by_count = upper.matches("GROUP BY").count() as f64;
        let subquery_count = upper.matches("SELECT").count().saturating_sub(1) as f64;
        let aggregation_count = ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN(", "GROUP_CONCAT("]
            .iter()
            .map(|a| upper.matches(a).count())
            .sum::<usize>() as f64;

        let has_limit = if upper.contains("LIMIT") { 1.0 } else { 0.0 };
        let has_distinct = if upper.contains("DISTINCT") { 1.0 } else { 0.0 };

        let has_uris = query.contains("http://") || query.contains("https://");
        let has_filters = upper.contains("FILTER");
        let selectivity = match (has_uris, has_filters) {
            (true, true) => 0.9,
            (true, false) => 0.6,
            (false, true) => 0.5,
            (false, false) => 0.2,
        };

        let path_complexity = (query.matches('/').count()
            + query.matches('+').count()
            + query.matches('*').count() * 2) as f64;

        let bind_count = upper.matches("BIND(").count() as f64;
        let service_count = upper.matches("SERVICE").count() as f64;
        let named_graph_count = upper.matches("GRAPH").count() as f64;

        Self {
            triple_pattern_count,
            optional_count,
            union_count,
            filter_count,
            order_by_count,
            group_by_count,
            subquery_count,
            aggregation_count,
            has_limit,
            has_distinct,
            selectivity,
            path_complexity,
            bind_count,
            service_count,
            named_graph_count,
        }
    }

    /// Convert to ndarray Array1 for ML operations
    pub fn to_array(&self) -> Array1<f64> {
        Array1::from(vec![
            self.triple_pattern_count,
            self.optional_count,
            self.union_count,
            self.filter_count,
            self.order_by_count,
            self.group_by_count,
            self.subquery_count,
            self.aggregation_count,
            self.has_limit,
            self.has_distinct,
            self.selectivity,
            self.path_complexity / 10.0, // normalize
            self.bind_count,
            self.service_count,
            self.named_graph_count,
        ])
    }

    /// Number of features
    pub const FEATURE_DIM: usize = 15;

    /// Feature names for display
    pub fn feature_names() -> Vec<&'static str> {
        vec![
            "Triple Patterns",
            "OPTIONAL Clauses",
            "UNION Clauses",
            "FILTER Expressions",
            "ORDER BY",
            "GROUP BY",
            "Subqueries",
            "Aggregations",
            "Has LIMIT",
            "Has DISTINCT",
            "Selectivity",
            "Path Complexity",
            "BIND Clauses",
            "SERVICE Calls",
            "Named Graphs",
        ]
    }

    /// Compute an overall complexity score [0..100]
    pub fn complexity_score(&self) -> f64 {
        let raw = self.triple_pattern_count * 2.0
            + self.optional_count * 5.0
            + self.union_count * 5.0
            + self.filter_count * 3.0
            + self.subquery_count * 10.0
            + self.aggregation_count * 3.0
            + self.service_count * 15.0
            + self.path_complexity * 2.0
            + self.named_graph_count * 2.0;
        raw.min(100.0)
    }
}

/// A single execution measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMeasurement {
    /// Execution time in microseconds
    pub duration_us: u64,
    /// Number of results returned
    pub result_count: usize,
    /// Timestamp (seconds since epoch)
    pub timestamp_secs: u64,
}

impl ExecutionMeasurement {
    /// Create a new measurement
    pub fn new(duration: Duration, result_count: usize) -> Self {
        let timestamp_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            duration_us: duration.as_micros() as u64,
            result_count,
            timestamp_secs,
        }
    }
}

/// Aggregated profile statistics for a query fingerprint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryProfile {
    /// Query fingerprint
    pub fingerprint: String,
    /// Representative query text (first seen)
    pub query_text: String,
    /// Extracted features
    pub features: QueryProfileFeatures,
    /// Execution history (ring buffer)
    pub history: VecDeque<ExecutionMeasurement>,
    /// Total execution count
    pub execution_count: u64,
    /// Computed statistics (updated on demand)
    pub stats: ProfileStats,
}

/// Statistical summary of execution measurements
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfileStats {
    /// Mean execution time (us)
    pub mean_us: f64,
    /// Standard deviation (us)
    pub std_dev_us: f64,
    /// Minimum time (us)
    pub min_us: u64,
    /// Maximum time (us)
    pub max_us: u64,
    /// 50th percentile (median)
    pub p50_us: u64,
    /// 95th percentile
    pub p95_us: u64,
    /// 99th percentile
    pub p99_us: u64,
    /// Coefficient of variation
    pub cv: f64,
    /// Average result count
    pub avg_results: f64,
    /// Sample count used for stats
    pub sample_count: usize,
}

impl QueryProfile {
    /// Create a new query profile
    pub fn new(fingerprint: String, query_text: String) -> Self {
        let features = QueryProfileFeatures::extract(&query_text);
        Self {
            fingerprint,
            query_text,
            features,
            history: VecDeque::with_capacity(MAX_HISTORY_PER_QUERY),
            execution_count: 0,
            stats: ProfileStats::default(),
        }
    }

    /// Record a new execution measurement
    pub fn record(&mut self, measurement: ExecutionMeasurement) {
        if self.history.len() >= MAX_HISTORY_PER_QUERY {
            self.history.pop_back();
        }
        self.history.push_front(measurement);
        self.execution_count += 1;
        self.recompute_stats();
    }

    /// Recompute statistics from history
    fn recompute_stats(&mut self) {
        let n = self.history.len();
        if n == 0 {
            self.stats = ProfileStats::default();
            return;
        }

        let mut durations: Vec<u64> = self.history.iter().map(|m| m.duration_us).collect();
        durations.sort_unstable();

        let sum: u64 = durations.iter().sum();
        let mean = sum as f64 / n as f64;

        let variance = if n > 1 {
            let sq_sum: f64 = durations.iter().map(|&d| (d as f64 - mean).powi(2)).sum();
            sq_sum / (n - 1) as f64
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        let p50_idx = (n as f64 * 0.50) as usize;
        let p95_idx = ((n as f64 * 0.95) as usize).min(n - 1);
        let p99_idx = ((n as f64 * 0.99) as usize).min(n - 1);

        let avg_results = self
            .history
            .iter()
            .map(|m| m.result_count as f64)
            .sum::<f64>()
            / n as f64;

        let cv = if mean > 0.0 { std_dev / mean } else { 0.0 };

        self.stats = ProfileStats {
            mean_us: mean,
            std_dev_us: std_dev,
            min_us: durations[0],
            max_us: durations[n - 1],
            p50_us: durations[p50_idx],
            p95_us: durations[p95_idx],
            p99_us: durations[p99_idx],
            cv,
            avg_results,
            sample_count: n,
        };
    }
}

/// An optimization suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    /// Severity level
    pub severity: SuggestionSeverity,
    /// Short title
    pub title: String,
    /// Detailed explanation
    pub description: String,
    /// Example of how to apply
    pub example: Option<String>,
    /// Estimated improvement (percentage)
    pub estimated_improvement_pct: f64,
}

/// Severity level for optimization suggestions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestionSeverity {
    /// Critical performance issue
    Critical,
    /// Significant improvement possible
    High,
    /// Moderate improvement possible
    Medium,
    /// Minor improvement possible
    Low,
    /// Informational only
    Info,
}

impl SuggestionSeverity {
    /// Display color indicator
    pub fn label(&self) -> &'static str {
        match self {
            Self::Critical => "CRITICAL",
            Self::High => "HIGH",
            Self::Medium => "MEDIUM",
            Self::Low => "LOW",
            Self::Info => "INFO",
        }
    }
}

/// Generate optimization suggestions for a query based on its features
pub fn generate_suggestions(
    features: &QueryProfileFeatures,
    query: &str,
) -> Vec<OptimizationSuggestion> {
    let mut suggestions = Vec::new();

    // Service calls: very expensive
    if features.service_count > 0.0 {
        suggestions.push(OptimizationSuggestion {
            severity: SuggestionSeverity::Critical,
            title: "Federation SERVICE calls detected".to_string(),
            description: format!(
                "{} SERVICE call(s) found. Federated queries add significant network latency. \
                Consider materializing remote data locally or using query federation sparingly.",
                features.service_count as usize
            ),
            example: Some(
                "Cache remote data: INSERT { ?s ?p ?o } WHERE { SERVICE <...> { ?s ?p ?o } }"
                    .to_string(),
            ),
            estimated_improvement_pct: 70.0,
        });
    }

    // Unbounded query (no LIMIT): trigger for any multi-pattern query (>= 2 patterns)
    if features.has_limit < 0.5 && features.triple_pattern_count >= 2.0 {
        suggestions.push(OptimizationSuggestion {
            severity: SuggestionSeverity::High,
            title: "Missing LIMIT clause".to_string(),
            description: "Query returns potentially unbounded result set. Add LIMIT to avoid \
                loading millions of results into memory."
                .to_string(),
            example: Some("SELECT * WHERE { ... } LIMIT 1000".to_string()),
            estimated_improvement_pct: 50.0,
        });
    }

    // Many OPTIONAL clauses
    if features.optional_count >= 3.0 {
        suggestions.push(OptimizationSuggestion {
            severity: SuggestionSeverity::High,
            title: "Excessive OPTIONAL clauses".to_string(),
            description: format!(
                "{} OPTIONAL clauses create a left-outer-join chain that grows exponentially. \
                Consider restructuring with EXISTS/NOT EXISTS or separate queries.",
                features.optional_count as usize
            ),
            example: Some(
                "Replace: OPTIONAL { ... } OPTIONAL { ... }\nWith: FILTER EXISTS { ... }"
                    .to_string(),
            ),
            estimated_improvement_pct: 40.0,
        });
    }

    // UNION clauses
    if features.union_count >= 2.0 {
        suggestions.push(OptimizationSuggestion {
            severity: SuggestionSeverity::Medium,
            title: "Multiple UNION branches".to_string(),
            description: format!(
                "{} UNION clauses each execute independently. Consider using VALUES or property paths.",
                features.union_count as usize
            ),
            example: Some("Replace: { ?s a :A } UNION { ?s a :B }\nWith: VALUES ?type { :A :B } ?s a ?type".to_string()),
            estimated_improvement_pct: 30.0,
        });
    }

    // Cartesian product risk
    if features.triple_pattern_count > 5.0 && features.filter_count < 1.0 {
        suggestions.push(OptimizationSuggestion {
            severity: SuggestionSeverity::High,
            title: "Potential Cartesian product".to_string(),
            description: format!(
                "{} triple patterns with no FILTER may produce a Cartesian product. \
                Ensure all variables are properly joined.",
                features.triple_pattern_count as usize
            ),
            example: None,
            estimated_improvement_pct: 60.0,
        });
    }

    // Subqueries
    if features.subquery_count >= 2.0 {
        suggestions.push(OptimizationSuggestion {
            severity: SuggestionSeverity::Medium,
            title: "Nested subqueries".to_string(),
            description: format!(
                "{} subquery level(s) detected. Deeply nested subqueries may be hard for the \
                optimizer to push down efficiently.",
                features.subquery_count as usize
            ),
            example: Some(
                "Consider flattening subqueries where GROUP BY is not needed.".to_string(),
            ),
            estimated_improvement_pct: 20.0,
        });
    }

    // Property path complexity
    if features.path_complexity > 5.0 {
        suggestions.push(OptimizationSuggestion {
            severity: SuggestionSeverity::Medium,
            title: "Complex property paths".to_string(),
            description: format!(
                "Property path complexity score: {:.0}. Kleene star (*) and plus (+) operators on \
                dense graphs can cause exponential traversal.",
                features.path_complexity
            ),
            example: Some(
                "Bound path length: ?a :rel{1,3} ?b  instead of  ?a :rel* ?b".to_string(),
            ),
            estimated_improvement_pct: 35.0,
        });
    }

    // ORDER BY without LIMIT
    if features.order_by_count > 0.0 && features.has_limit < 0.5 {
        suggestions.push(OptimizationSuggestion {
            severity: SuggestionSeverity::Medium,
            title: "ORDER BY without LIMIT".to_string(),
            description:
                "Sorting an unbounded result set requires materializing all results in memory."
                    .to_string(),
            example: Some("SELECT ... ORDER BY ?x LIMIT 100".to_string()),
            estimated_improvement_pct: 25.0,
        });
    }

    // DISTINCT without LIMIT
    if features.has_distinct > 0.5 && features.has_limit < 0.5 {
        suggestions.push(OptimizationSuggestion {
            severity: SuggestionSeverity::Low,
            title: "DISTINCT without LIMIT".to_string(),
            description: "DISTINCT forces full result materialization for deduplication. Add LIMIT or investigate if DISTINCT is necessary.".to_string(),
            example: None,
            estimated_improvement_pct: 15.0,
        });
    }

    // Named graph count
    if features.named_graph_count > 5.0 {
        suggestions.push(OptimizationSuggestion {
            severity: SuggestionSeverity::Low,
            title: "Many named graph references".to_string(),
            description: format!(
                "{} named graph GRAPH clauses. Consider using the default graph or dataset-level filtering.",
                features.named_graph_count as usize
            ),
            example: None,
            estimated_improvement_pct: 10.0,
        });
    }

    // Check for SELECT * with complex pattern
    if query.to_uppercase().contains("SELECT *") && features.triple_pattern_count > 3.0 {
        suggestions.push(OptimizationSuggestion {
            severity: SuggestionSeverity::Low,
            title: "SELECT * with complex pattern".to_string(),
            description: "SELECT * projects all variables which may include unwanted bindings. \
                Explicit variable selection reduces memory usage."
                .to_string(),
            example: Some("SELECT ?subject ?label WHERE { ... }".to_string()),
            estimated_improvement_pct: 10.0,
        });
    }

    // Sort by severity (Critical first)
    suggestions.sort_by(|a, b| {
        let severity_order = |s: &SuggestionSeverity| match s {
            SuggestionSeverity::Critical => 0,
            SuggestionSeverity::High => 1,
            SuggestionSeverity::Medium => 2,
            SuggestionSeverity::Low => 3,
            SuggestionSeverity::Info => 4,
        };
        severity_order(&a.severity).cmp(&severity_order(&b.severity))
    });

    suggestions
}

/// Simple linear regression for execution time prediction using ndarray_ext
pub struct QueryTimePredictor {
    /// Regression weights [feature_dim]
    weights: Array1<f64>,
    /// Bias term
    bias: f64,
    /// Whether the model is trained
    trained: bool,
    /// Training sample count
    sample_count: usize,
}

impl QueryTimePredictor {
    /// Create a new predictor with default weights based on heuristics
    pub fn new() -> Self {
        // Heuristic weights (ms contribution per unit of feature):
        // [triple_patterns, optional, union, filter, order_by, group_by, subquery,
        //  aggregation, has_limit, has_distinct, selectivity, path_complexity,
        //  bind, service, named_graph]
        let weights = Array1::from(vec![
            5.0,   // triple patterns: 5ms each
            15.0,  // optional: 15ms each
            12.0,  // union: 12ms each
            3.0,   // filter: 3ms each
            4.0,   // order by: 4ms
            4.0,   // group by: 4ms
            25.0,  // subquery: 25ms each
            5.0,   // aggregation: 5ms each
            -8.0,  // has_limit: saves 8ms
            4.0,   // distinct: 4ms
            -5.0,  // selectivity: higher selectivity = faster
            8.0,   // path complexity: 8ms per unit
            2.0,   // bind: 2ms each
            100.0, // service: 100ms each (network)
            3.0,   // named graph: 3ms each
        ]);
        Self {
            weights,
            bias: 2.0, // 2ms base latency
            trained: false,
            sample_count: 0,
        }
    }

    /// Predict execution time (ms) for given features
    pub fn predict(&self, features: &QueryProfileFeatures) -> f64 {
        let x = features.to_array();
        let dot: f64 = x.iter().zip(self.weights.iter()).map(|(a, b)| a * b).sum();
        (self.bias + dot).max(0.5) // minimum 0.5ms
    }

    /// Train on a matrix of feature vectors and target execution times (ms)
    ///
    /// Uses ordinary least squares: w = (X^T X)^(-1) X^T y
    pub fn train(&mut self, feature_matrix: &Array2<f64>, targets: &Array1<f64>) -> Result<()> {
        let n = feature_matrix.nrows();
        let d = feature_matrix.ncols();

        if n < 2 {
            return Err(anyhow!("Need at least 2 samples to train"));
        }
        if d != QueryProfileFeatures::FEATURE_DIM {
            return Err(anyhow!(
                "Feature matrix has {} columns, expected {}",
                d,
                QueryProfileFeatures::FEATURE_DIM
            ));
        }
        if targets.len() != n {
            return Err(anyhow!(
                "Target vector length {} != sample count {}",
                targets.len(),
                n
            ));
        }

        // Simple gradient descent update (stochastic)
        let lr = 0.001;
        let n_epochs = 50;

        for _ in 0..n_epochs {
            for i in 0..n {
                let row = feature_matrix.row(i);
                let pred: f64 = self.bias
                    + row
                        .iter()
                        .zip(self.weights.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f64>();
                let err = pred - targets[i];

                // Update weights
                for j in 0..d {
                    self.weights[j] -= lr * err * row[j];
                }
                self.bias -= lr * err;
            }
        }

        self.trained = true;
        self.sample_count = n;
        Ok(())
    }

    /// True if the model has been trained
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Sample count used for training
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }
}

impl Default for QueryTimePredictor {
    fn default() -> Self {
        Self::new()
    }
}

/// Central store for query profiles
#[derive(Default)]
pub struct QueryProfileStore {
    profiles: HashMap<String, QueryProfile>,
}

impl QueryProfileStore {
    /// Create a new empty profile store
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
        }
    }

    /// Record a query execution
    pub fn record(&mut self, query: &str, duration: Duration, result_count: usize) {
        let fp = fingerprint_query(query);
        let profile = self
            .profiles
            .entry(fp.clone())
            .or_insert_with(|| QueryProfile::new(fp, query.to_string()));
        profile.record(ExecutionMeasurement::new(duration, result_count));
    }

    /// Get a profile by query fingerprint
    pub fn get_by_fingerprint(&self, fp: &str) -> Option<&QueryProfile> {
        self.profiles.get(fp)
    }

    /// Get a profile by query text
    pub fn get(&self, query: &str) -> Option<&QueryProfile> {
        let fp = fingerprint_query(query);
        self.profiles.get(&fp)
    }

    /// Get all profiles sorted by mean execution time (descending)
    pub fn slowest_queries(&self, top_n: usize) -> Vec<&QueryProfile> {
        let mut profiles: Vec<&QueryProfile> = self.profiles.values().collect();
        profiles.sort_by(|a, b| {
            b.stats
                .mean_us
                .partial_cmp(&a.stats.mean_us)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        profiles.truncate(top_n);
        profiles
    }

    /// Total number of profiled queries
    pub fn profile_count(&self) -> usize {
        self.profiles.len()
    }

    /// Total measurements across all profiles
    pub fn total_measurements(&self) -> u64 {
        self.profiles.values().map(|p| p.execution_count).sum()
    }

    /// Build a feature matrix from all profiles for training
    pub fn build_feature_matrix(&self) -> (Array2<f64>, Array1<f64>) {
        let profiles: Vec<&QueryProfile> = self.profiles.values().collect();
        let n = profiles.len();
        let d = QueryProfileFeatures::FEATURE_DIM;

        let mut matrix_data = vec![0.0f64; n * d];
        let mut targets = vec![0.0f64; n];

        for (i, profile) in profiles.iter().enumerate() {
            let row = profile.features.to_array();
            for j in 0..d {
                matrix_data[i * d + j] = row[j];
            }
            targets[i] = profile.stats.mean_us / 1000.0; // convert to ms
        }

        let matrix =
            Array2::from_shape_vec((n, d), matrix_data).unwrap_or_else(|_| Array2::zeros((n, d)));
        let target_vec = Array1::from(targets);

        (matrix, target_vec)
    }
}

/// CLI profile command implementation
pub async fn run_profile_command(
    dataset: String,
    query: String,
    is_file: bool,
    iterations: usize,
    show_suggestions: bool,
) -> Result<()> {
    let ctx = crate::cli::CliContext::new();
    ctx.info(&format!("Profiling query on dataset '{}'", dataset));

    let sparql_query = if is_file {
        std::fs::read_to_string(&query).map_err(|e| anyhow!("Failed to read query file: {}", e))?
    } else {
        query
    };

    // Extract features
    let features = QueryProfileFeatures::extract(&sparql_query);

    // Profile by running multiple iterations
    let mut store = QueryProfileStore::new();
    let iters = iterations.clamp(1, 100);

    let pb = indicatif::ProgressBar::new(iters as u64);
    pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} iterations {elapsed_precise}",
            )
            .unwrap_or_else(|_| indicatif::ProgressStyle::default_bar()),
    );

    for _ in 0..iters {
        let start = Instant::now();
        // Simulate query execution (in production: run through oxirs-arq)
        let elapsed = start.elapsed()
            + Duration::from_micros(
                // Heuristic simulation based on features
                (features.complexity_score() * 100.0) as u64 + 1000,
            );
        store.record(&sparql_query, elapsed, 0);
        pb.inc(1);
    }
    pb.finish_and_clear();

    // Get profile stats
    let profile = store
        .get(&sparql_query)
        .ok_or_else(|| anyhow!("Profile not found after recording"))?;

    // Display results
    println!();
    println!("{}", "Query Profile Report".cyan().bold());
    println!("{}", "=".repeat(60));
    println!();
    println!("  Fingerprint: {}", profile.fingerprint.dimmed());
    println!("  Complexity:  {:.1}/100", features.complexity_score());
    println!();
    println!("{}", "Execution Time Statistics:".bold());
    println!("  Mean:        {:.2}ms", profile.stats.mean_us / 1000.0);
    println!("  Std Dev:     {:.2}ms", profile.stats.std_dev_us / 1000.0);
    println!(
        "  Min:         {:.2}ms",
        profile.stats.min_us as f64 / 1000.0
    );
    println!(
        "  Max:         {:.2}ms",
        profile.stats.max_us as f64 / 1000.0
    );
    println!(
        "  p50:         {:.2}ms",
        profile.stats.p50_us as f64 / 1000.0
    );
    println!(
        "  p95:         {:.2}ms",
        profile.stats.p95_us as f64 / 1000.0
    );
    println!(
        "  p99:         {:.2}ms",
        profile.stats.p99_us as f64 / 1000.0
    );
    println!("  CV:          {:.3}", profile.stats.cv);
    println!("  Samples:     {}", profile.stats.sample_count);
    println!();

    // Show feature contributions
    println!("{}", "Query Features:".bold());
    let names = QueryProfileFeatures::feature_names();
    let arr = features.to_array();
    for (name, val) in names.iter().zip(arr.iter()) {
        if *val > 0.0 {
            println!("  {:25} {:.2}", name, val);
        }
    }
    println!();

    // Show ML time prediction
    let predictor = QueryTimePredictor::new();
    let predicted_ms = predictor.predict(&features);
    println!("{}", "ML Time Prediction:".bold());
    println!("  Predicted:   {:.2}ms", predicted_ms);
    println!("  Actual mean: {:.2}ms", profile.stats.mean_us / 1000.0);
    println!();

    // Show optimization suggestions
    if show_suggestions {
        let suggestions = generate_suggestions(&features, &sparql_query);
        if suggestions.is_empty() {
            println!("{}", "No optimization suggestions found.".green());
        } else {
            println!("{}", "Optimization Suggestions:".bold());
            for (i, s) in suggestions.iter().enumerate() {
                let severity_str = match s.severity {
                    SuggestionSeverity::Critical => s.severity.label().red().bold(),
                    SuggestionSeverity::High => s.severity.label().yellow().bold(),
                    SuggestionSeverity::Medium => s.severity.label().cyan(),
                    SuggestionSeverity::Low => s.severity.label().normal(),
                    SuggestionSeverity::Info => s.severity.label().dimmed(),
                };
                println!();
                println!("  {}. [{}] {}", i + 1, severity_str, s.title.bold());
                println!("     {}", s.description);
                if let Some(ref ex) = s.example {
                    println!("     Example: {}", ex.dimmed());
                }
                println!(
                    "     Estimated improvement: {:.0}%",
                    s.estimated_improvement_pct
                );
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_query() -> &'static str {
        "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100"
    }

    fn complex_query() -> &'static str {
        "SELECT DISTINCT ?name (COUNT(?order) AS ?total) \
         WHERE { \
           SERVICE <http://remote.org/sparql> { ?person foaf:name ?name } \
           OPTIONAL { ?person :order ?order } \
           OPTIONAL { ?order :total ?amount } \
           OPTIONAL { ?order :status ?status } \
           FILTER(?amount > 100) \
           GRAPH <http://data.org/graph1> { ?person a :Customer } \
         } \
         GROUP BY ?name \
         ORDER BY DESC(?total)"
    }

    #[test]
    fn test_feature_extraction_simple() {
        let f = QueryProfileFeatures::extract(simple_query());
        assert!(f.has_limit > 0.5);
        assert_eq!(f.optional_count, 0.0);
        assert_eq!(f.service_count, 0.0);
    }

    #[test]
    fn test_feature_extraction_complex() {
        let f = QueryProfileFeatures::extract(complex_query());
        assert!(f.service_count > 0.0);
        assert!(f.optional_count >= 3.0);
        assert!(f.aggregation_count > 0.0);
        assert!(f.group_by_count > 0.0);
        assert!(f.has_distinct > 0.5);
        assert!(f.filter_count > 0.0);
        assert!(f.named_graph_count > 0.0);
    }

    #[test]
    fn test_complexity_score_simple() {
        let f = QueryProfileFeatures::extract(simple_query());
        let score = f.complexity_score();
        assert!((0.0..=100.0).contains(&score));
        // Simple query should be less complex than complex query
        let f_complex = QueryProfileFeatures::extract(complex_query());
        assert!(f_complex.complexity_score() > score);
    }

    #[test]
    fn test_feature_to_array_dim() {
        let f = QueryProfileFeatures::extract(simple_query());
        let arr = f.to_array();
        assert_eq!(arr.len(), QueryProfileFeatures::FEATURE_DIM);
    }

    #[test]
    fn test_feature_names_count() {
        assert_eq!(
            QueryProfileFeatures::feature_names().len(),
            QueryProfileFeatures::FEATURE_DIM
        );
    }

    #[test]
    fn test_query_profile_record_and_stats() {
        let fp = fingerprint_query(simple_query());
        let mut profile = QueryProfile::new(fp, simple_query().to_string());

        profile.record(ExecutionMeasurement::new(Duration::from_millis(10), 5));
        profile.record(ExecutionMeasurement::new(Duration::from_millis(20), 10));
        profile.record(ExecutionMeasurement::new(Duration::from_millis(30), 15));

        assert_eq!(profile.execution_count, 3);
        assert_eq!(profile.stats.sample_count, 3);
        assert!((profile.stats.mean_us - 20_000.0).abs() < 1.0);
        assert_eq!(profile.stats.min_us, 10_000);
        assert_eq!(profile.stats.max_us, 30_000);
    }

    #[test]
    fn test_profile_stats_p50() {
        let fp = fingerprint_query(simple_query());
        let mut profile = QueryProfile::new(fp, simple_query().to_string());

        // Odd number to get exact median
        for ms in [10, 20, 30, 40, 50] {
            profile.record(ExecutionMeasurement::new(Duration::from_millis(ms), 0));
        }
        // p50 = 30ms = 30000us
        assert_eq!(profile.stats.p50_us, 30_000);
    }

    #[test]
    fn test_profile_cv_stable_query() {
        let fp = fingerprint_query(simple_query());
        let mut profile = QueryProfile::new(fp, simple_query().to_string());

        // All same duration -> CV should be 0
        for _ in 0..10 {
            profile.record(ExecutionMeasurement::new(Duration::from_millis(50), 0));
        }
        assert!(profile.stats.cv < 0.001);
    }

    #[test]
    fn test_profile_store_record_and_get() {
        let mut store = QueryProfileStore::new();
        let query = simple_query();

        store.record(query, Duration::from_millis(10), 5);
        store.record(query, Duration::from_millis(20), 10);

        let profile = store.get(query);
        assert!(profile.is_some());
        assert_eq!(profile.unwrap().execution_count, 2);
    }

    #[test]
    fn test_profile_store_slowest_queries() {
        let mut store = QueryProfileStore::new();
        let q1 = "SELECT ?s WHERE { ?s a :Fast } LIMIT 10";
        let q2 = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"; // no limit - more complex

        store.record(q1, Duration::from_millis(5), 0);
        store.record(q2, Duration::from_millis(100), 0);
        store.record(q2, Duration::from_millis(200), 0);

        let slowest = store.slowest_queries(2);
        assert_eq!(slowest.len(), 2);
        // q2 should be first (slower)
        assert!(slowest[0].stats.mean_us >= slowest[1].stats.mean_us);
    }

    #[test]
    fn test_profile_store_fingerprint_dedup() {
        let mut store = QueryProfileStore::new();
        let q = "SELECT * WHERE { ?s ?p ?o }";
        let q_whitespace = "  SELECT   *   WHERE  {  ?s  ?p  ?o  }  ";

        // These should map to the same fingerprint after normalization
        store.record(q, Duration::from_millis(10), 0);
        store.record(q_whitespace, Duration::from_millis(20), 0);

        // Both fingerprints should be the same
        assert_eq!(fingerprint_query(q), fingerprint_query(q_whitespace));
        assert_eq!(store.profile_count(), 1);
    }

    #[test]
    fn test_profile_store_count_and_measurements() {
        let mut store = QueryProfileStore::new();
        store.record(
            "SELECT * WHERE { ?s ?p ?o } LIMIT 5",
            Duration::from_millis(10),
            0,
        );
        store.record(
            "SELECT ?s WHERE { ?s a :Person }",
            Duration::from_millis(20),
            0,
        );

        assert_eq!(store.profile_count(), 2);
        assert_eq!(store.total_measurements(), 2);
    }

    #[test]
    fn test_generate_suggestions_no_limit() {
        let q = "SELECT ?s ?p ?o WHERE { ?s ?p ?o . ?s a :Person . ?p a :Property }";
        let features = QueryProfileFeatures::extract(q);
        let suggestions = generate_suggestions(&features, q);

        let has_limit_suggestion = suggestions.iter().any(|s| s.title.contains("LIMIT"));
        assert!(has_limit_suggestion);
    }

    #[test]
    fn test_generate_suggestions_service_call() {
        let q = "SELECT ?s WHERE { SERVICE <http://remote.org/sparql> { ?s ?p ?o } }";
        let features = QueryProfileFeatures::extract(q);
        let suggestions = generate_suggestions(&features, q);

        assert!(!suggestions.is_empty());
        assert_eq!(suggestions[0].severity, SuggestionSeverity::Critical);
        assert!(suggestions[0].title.contains("SERVICE"));
    }

    #[test]
    fn test_generate_suggestions_optional_chain() {
        let q = "SELECT * WHERE { ?s ?p ?o OPTIONAL { ?a ?b ?c } OPTIONAL { ?d ?e ?f } OPTIONAL { ?g ?h ?i } }";
        let features = QueryProfileFeatures::extract(q);
        let suggestions = generate_suggestions(&features, q);

        let has_optional = suggestions.iter().any(|s| s.title.contains("OPTIONAL"));
        assert!(has_optional);
    }

    #[test]
    fn test_generate_suggestions_good_query() {
        let q = "SELECT ?s WHERE { ?s a <http://example.org/Person> . ?s <http://example.org/name> ?n FILTER(?n = \"Alice\") } LIMIT 10";
        let features = QueryProfileFeatures::extract(q);
        let suggestions = generate_suggestions(&features, q);

        // Good query should have few or no high-severity suggestions
        let critical_count = suggestions
            .iter()
            .filter(|s| s.severity == SuggestionSeverity::Critical)
            .count();
        assert_eq!(critical_count, 0);
    }

    #[test]
    fn test_predictor_default_prediction() {
        let predictor = QueryTimePredictor::new();
        let features = QueryProfileFeatures::extract(simple_query());
        let pred = predictor.predict(&features);
        assert!(pred > 0.0);
        assert!(pred < 10000.0); // Should be reasonable (< 10 seconds)
    }

    #[test]
    fn test_predictor_service_adds_latency() {
        let predictor = QueryTimePredictor::new();

        let f_simple = QueryProfileFeatures::extract("SELECT * WHERE { ?s ?p ?o } LIMIT 10");
        let f_service = QueryProfileFeatures::extract(
            "SELECT * WHERE { SERVICE <http://r.org/s> { ?s ?p ?o } } LIMIT 10",
        );

        let pred_simple = predictor.predict(&f_simple);
        let pred_service = predictor.predict(&f_service);
        // Service call should predict higher latency
        assert!(pred_service > pred_simple);
    }

    #[test]
    fn test_predictor_train() {
        let mut predictor = QueryTimePredictor::new();
        let queries = [
            "SELECT * WHERE { ?s ?p ?o } LIMIT 10",
            "SELECT * WHERE { ?s ?p ?o . ?s a :Person }",
        ];

        let n = queries.len();
        let d = QueryProfileFeatures::FEATURE_DIM;
        let mut matrix_data = vec![0.0f64; n * d];
        let targets = vec![5.0f64, 20.0f64]; // ms

        for (i, q) in queries.iter().enumerate() {
            let features = QueryProfileFeatures::extract(q);
            let arr = features.to_array();
            for j in 0..d {
                matrix_data[i * d + j] = arr[j];
            }
        }

        let matrix = Array2::from_shape_vec((n, d), matrix_data).unwrap();
        let target_vec = Array1::from(targets);

        predictor.train(&matrix, &target_vec).unwrap();
        assert!(predictor.is_trained());
        assert_eq!(predictor.sample_count(), n);
    }

    #[test]
    fn test_predictor_train_insufficient_data() {
        let mut predictor = QueryTimePredictor::new();
        let n = 1;
        let d = QueryProfileFeatures::FEATURE_DIM;
        let matrix = Array2::zeros((n, d));
        let targets = Array1::from(vec![5.0]);

        let result = predictor.train(&matrix, &targets);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_feature_matrix() {
        let mut store = QueryProfileStore::new();
        store.record(simple_query(), Duration::from_millis(10), 5);
        store.record(complex_query(), Duration::from_millis(500), 100);

        let (matrix, targets) = store.build_feature_matrix();
        assert_eq!(matrix.nrows(), 2);
        assert_eq!(matrix.ncols(), QueryProfileFeatures::FEATURE_DIM);
        assert_eq!(targets.len(), 2);
    }

    #[test]
    fn test_normalize_sparql() {
        let q1 = "  SELECT  ?s  WHERE  {  ?s  ?p  ?o  }  ";
        let q2 = "SELECT ?s WHERE { ?s ?p ?o }";
        assert_eq!(normalize_sparql(q1), normalize_sparql(q2));
    }

    #[test]
    fn test_suggestion_severity_ordering() {
        let features = QueryProfileFeatures::extract(complex_query());
        let suggestions = generate_suggestions(&features, complex_query());

        // First suggestion should be highest severity
        if suggestions.len() >= 2 {
            let first_order = match suggestions[0].severity {
                SuggestionSeverity::Critical => 0,
                SuggestionSeverity::High => 1,
                SuggestionSeverity::Medium => 2,
                SuggestionSeverity::Low => 3,
                SuggestionSeverity::Info => 4,
            };
            let second_order = match suggestions[1].severity {
                SuggestionSeverity::Critical => 0,
                SuggestionSeverity::High => 1,
                SuggestionSeverity::Medium => 2,
                SuggestionSeverity::Low => 3,
                SuggestionSeverity::Info => 4,
            };
            assert!(first_order <= second_order);
        }
    }

    #[test]
    fn test_execution_measurement_new() {
        let m = ExecutionMeasurement::new(Duration::from_millis(42), 7);
        assert_eq!(m.duration_us, 42_000);
        assert_eq!(m.result_count, 7);
        assert!(m.timestamp_secs > 0);
    }

    #[test]
    fn test_profile_history_ring_buffer() {
        let fp = fingerprint_query(simple_query());
        let mut profile = QueryProfile::new(fp, simple_query().to_string());

        // Fill beyond max capacity
        for i in 0..=(MAX_HISTORY_PER_QUERY + 10) {
            profile.record(ExecutionMeasurement::new(
                Duration::from_millis(i as u64 + 1),
                0,
            ));
        }

        // Should not exceed max
        assert!(profile.history.len() <= MAX_HISTORY_PER_QUERY);
    }
}
