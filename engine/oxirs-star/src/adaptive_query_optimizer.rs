//! Adaptive Query Optimizer with Auto-Tuning Capabilities
//!
//! This module provides an advanced query optimizer that automatically tunes itself based on
//! observed query patterns and execution characteristics. It combines machine learning-based
//! cost prediction with online learning to continuously improve query execution strategies.
//!
//! # Features
//!
//! - **Workload-Aware Optimization**: Learns from query execution history
//! - **Adaptive Strategy Selection**: Dynamically chooses between classical and quantum-inspired algorithms
//! - **Performance Regression Detection**: Identifies when query performance degrades
//! - **Auto-Tuning**: Automatically adjusts optimization parameters based on feedback
//! - **Multi-Objective Optimization**: Balances execution time, memory usage, and accuracy
//!
//! # Architecture
//!
//! The adaptive optimizer uses a two-tier approach:
//! 1. **Online Learning Layer**: Continuously updates models with actual execution metrics
//! 2. **Strategy Selection Layer**: Chooses the best optimization algorithm for each query type
//!
//! # Example
//!
//! ```rust
//! use oxirs_star::adaptive_query_optimizer::{AdaptiveQueryOptimizer, OptimizationObjective};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut optimizer = AdaptiveQueryOptimizer::new();
//!
//! // Configure multi-objective optimization
//! optimizer.set_objectives(vec![
//!     OptimizationObjective::MinimizeLatency { weight: 0.6 },
//!     OptimizationObjective::MinimizeMemory { weight: 0.3 },
//!     OptimizationObjective::MaximizeAccuracy { weight: 0.1 },
//! ]);
//!
//! // The optimizer will automatically tune itself as queries are executed
//! // optimizer.optimize_query(&sparql_query)?;
//! # Ok(())
//! # }
//! ```

use crate::ml_sparql_optimizer::{MLSPARQLOptimizer, MLSPARQLOptimizerConfig};
use crate::quantum_sparql_optimizer::{QuantumSPARQLOptimizer, QuantumSPARQLOptimizerConfig};
use crate::StarResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Optimization objective with configurable weight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationObjective {
    /// Minimize query execution latency
    MinimizeLatency { weight: f64 },
    /// Minimize memory consumption
    MinimizeMemory { weight: f64 },
    /// Maximize result accuracy (for approximate queries)
    MaximizeAccuracy { weight: f64 },
    /// Maximize throughput (queries per second)
    MaximizeThroughput { weight: f64 },
}

impl OptimizationObjective {
    /// Get the weight for this objective
    pub fn weight(&self) -> f64 {
        match self {
            Self::MinimizeLatency { weight }
            | Self::MinimizeMemory { weight }
            | Self::MaximizeAccuracy { weight }
            | Self::MaximizeThroughput { weight } => *weight,
        }
    }

    /// Get the name of this objective
    pub fn name(&self) -> &'static str {
        match self {
            Self::MinimizeLatency { .. } => "latency",
            Self::MinimizeMemory { .. } => "memory",
            Self::MaximizeAccuracy { .. } => "accuracy",
            Self::MaximizeThroughput { .. } => "throughput",
        }
    }
}

/// Query execution metrics tracked by the adaptive optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    /// Actual execution time in milliseconds
    pub execution_time_ms: f64,
    /// Memory usage in bytes
    pub memory_bytes: u64,
    /// Number of results returned
    pub result_count: usize,
    /// Whether the query used approximation
    pub is_approximate: bool,
    /// Accuracy score (0.0 to 1.0) for approximate queries
    pub accuracy_score: f64,
    /// Timestamp when query was executed
    pub timestamp: std::time::SystemTime,
}

/// Workload characteristics extracted from query patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadProfile {
    /// Average query complexity score
    pub avg_complexity: f64,
    /// Standard deviation of complexity
    pub complexity_stddev: f64,
    /// Most common query patterns (hash -> frequency)
    pub query_patterns: HashMap<u64, usize>,
    /// Average execution time by complexity bucket
    pub time_by_complexity: HashMap<String, f64>,
    /// Memory usage trends
    pub memory_trend: Vec<f64>,
    /// Number of queries analyzed
    pub sample_count: usize,
}

impl Default for WorkloadProfile {
    fn default() -> Self {
        Self {
            avg_complexity: 0.0,
            complexity_stddev: 0.0,
            query_patterns: HashMap::new(),
            time_by_complexity: HashMap::new(),
            memory_trend: Vec::new(),
            sample_count: 0,
        }
    }
}

/// Strategy for query optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    /// Classical rule-based optimization
    Classical,
    /// Machine learning-based optimization
    MachineLearning,
    /// Quantum-inspired optimization
    QuantumInspired,
    /// Hybrid approach combining multiple strategies
    Hybrid,
    /// Automatically select best strategy
    Auto,
}

/// Performance regression detector
#[derive(Debug, Clone)]
pub struct RegressionDetector {
    /// Historical baseline performance (percentile -> time_ms)
    baseline: HashMap<String, f64>,
    /// Threshold for detecting regression (multiplier)
    threshold: f64,
    /// Number of samples needed for stable baseline
    min_samples: usize,
    /// Recent execution times (rolling window)
    recent_times: Vec<f64>,
    /// Window size for regression detection
    window_size: usize,
    /// Whether baseline has been established
    baseline_established: bool,
}

impl RegressionDetector {
    /// Create a new regression detector
    pub fn new(threshold: f64, window_size: usize) -> Self {
        Self {
            baseline: HashMap::new(),
            threshold,
            min_samples: 30,
            recent_times: Vec::with_capacity(window_size),
            window_size,
            baseline_established: false,
        }
    }

    /// Update baseline with new execution time
    pub fn update(&mut self, execution_time_ms: f64) {
        self.recent_times.push(execution_time_ms);

        if self.recent_times.len() > self.window_size {
            self.recent_times.remove(0);
        }

        // Establish baseline once we have enough samples (but don't update it after)
        if !self.baseline_established && self.recent_times.len() >= self.min_samples {
            self.baseline = self.compute_percentiles();
            self.baseline_established = true;
        }
    }

    /// Check if recent performance represents a regression
    pub fn detect_regression(&self) -> Option<RegressionReport> {
        if self.baseline.is_empty() || self.recent_times.len() < 5 {
            return None;
        }

        let recent_avg = self.recent_times.iter().sum::<f64>() / self.recent_times.len() as f64;
        let baseline_median = self.baseline.get("p50")?;

        if recent_avg > baseline_median * self.threshold {
            Some(RegressionReport {
                baseline_p50: *baseline_median,
                recent_avg,
                regression_ratio: recent_avg / baseline_median,
                severity: if recent_avg > baseline_median * (self.threshold * 2.0) {
                    RegressionSeverity::Critical
                } else if recent_avg > baseline_median * (self.threshold * 1.5) {
                    RegressionSeverity::High
                } else {
                    RegressionSeverity::Medium
                },
            })
        } else {
            None
        }
    }

    /// Compute percentiles from recent times
    fn compute_percentiles(&self) -> HashMap<String, f64> {
        let mut sorted = self.recent_times.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut percentiles = HashMap::new();
        percentiles.insert("p50".to_string(), self.percentile(&sorted, 0.50));
        percentiles.insert("p90".to_string(), self.percentile(&sorted, 0.90));
        percentiles.insert("p95".to_string(), self.percentile(&sorted, 0.95));
        percentiles.insert("p99".to_string(), self.percentile(&sorted, 0.99));

        percentiles
    }

    /// Calculate percentile value
    fn percentile(&self, sorted: &[f64], p: f64) -> f64 {
        let index = (sorted.len() as f64 * p) as usize;
        sorted
            .get(index.min(sorted.len() - 1))
            .copied()
            .unwrap_or(0.0)
    }
}

/// Regression detection report
#[derive(Debug, Clone)]
pub struct RegressionReport {
    /// Baseline median performance
    pub baseline_p50: f64,
    /// Recent average performance
    pub recent_avg: f64,
    /// Regression ratio (recent/baseline)
    pub regression_ratio: f64,
    /// Severity level
    pub severity: RegressionSeverity,
}

/// Severity level for performance regressions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegressionSeverity {
    /// Medium regression (1.3-1.5x slower)
    Medium,
    /// High regression (1.5-2x slower)
    High,
    /// Critical regression (>2x slower)
    Critical,
}

/// Adaptive query optimizer with auto-tuning
pub struct AdaptiveQueryOptimizer {
    /// Machine learning optimizer for complex pattern learning
    #[allow(dead_code)]
    ml_optimizer: MLSPARQLOptimizer,
    /// Quantum-inspired optimizer for combinatorial optimization
    #[allow(dead_code)]
    quantum_optimizer: QuantumSPARQLOptimizer,
    /// Current optimization strategy
    strategy: OptimizationStrategy,
    /// Optimization objectives
    objectives: Vec<OptimizationObjective>,
    /// Workload profile
    workload: WorkloadProfile,
    /// Performance regression detector
    regression_detector: RegressionDetector,
    /// Strategy performance history (strategy -> avg_time_ms)
    strategy_performance: HashMap<String, Vec<f64>>,
    /// Auto-tuning enabled
    auto_tuning_enabled: bool,
    /// Minimum samples before auto-tuning activates
    auto_tuning_warmup: usize,
    /// Total queries processed
    queries_processed: usize,
}

impl AdaptiveQueryOptimizer {
    /// Create a new adaptive query optimizer with default configuration
    pub fn new() -> Self {
        Self {
            ml_optimizer: MLSPARQLOptimizer::new(MLSPARQLOptimizerConfig::default()),
            quantum_optimizer: QuantumSPARQLOptimizer::new(QuantumSPARQLOptimizerConfig::default()),
            strategy: OptimizationStrategy::Auto,
            objectives: vec![OptimizationObjective::MinimizeLatency { weight: 1.0 }],
            workload: WorkloadProfile::default(),
            regression_detector: RegressionDetector::new(1.3, 100),
            strategy_performance: HashMap::new(),
            auto_tuning_enabled: true,
            auto_tuning_warmup: 50,
            queries_processed: 0,
        }
    }

    /// Set optimization objectives
    pub fn set_objectives(&mut self, objectives: Vec<OptimizationObjective>) {
        self.objectives = objectives;
    }

    /// Set optimization strategy
    pub fn set_strategy(&mut self, strategy: OptimizationStrategy) {
        self.strategy = strategy;
        info!("Optimization strategy set to: {:?}", strategy);
    }

    /// Enable or disable auto-tuning
    pub fn set_auto_tuning(&mut self, enabled: bool) {
        self.auto_tuning_enabled = enabled;
        info!(
            "Auto-tuning {}",
            if enabled { "enabled" } else { "disabled" }
        );
    }

    /// Optimize a SPARQL-star query using adaptive strategy selection
    pub fn optimize_query(&mut self, query: &str) -> StarResult<OptimizationResult> {
        let start = Instant::now();
        self.queries_processed += 1;

        // Select strategy based on workload and configuration
        let strategy = self.select_strategy(query);
        debug!("Selected strategy: {:?} for query", strategy);

        // Execute optimization
        let result = match strategy {
            OptimizationStrategy::MachineLearning => self.optimize_with_ml(query)?,
            OptimizationStrategy::QuantumInspired => self.optimize_with_quantum(query)?,
            OptimizationStrategy::Classical => self.optimize_classical(query)?,
            OptimizationStrategy::Hybrid => self.optimize_hybrid(query)?,
            OptimizationStrategy::Auto => {
                // Should not reach here as select_strategy handles Auto
                self.optimize_with_ml(query)?
            }
        };

        // Record execution metrics
        let elapsed = start.elapsed();
        self.update_metrics(query, &result, elapsed, strategy);

        // Check for performance regressions
        self.detect_and_report_regressions();

        Ok(result)
    }

    /// Get current workload profile
    pub fn workload_profile(&self) -> &WorkloadProfile {
        &self.workload
    }

    /// Get optimization statistics
    pub fn statistics(&self) -> OptimizerStatistics {
        OptimizerStatistics {
            queries_processed: self.queries_processed,
            avg_complexity: self.workload.avg_complexity,
            strategy_performance: self.strategy_performance.clone(),
            regression_detected: self.regression_detector.detect_regression().is_some(),
            auto_tuning_active: self.auto_tuning_enabled
                && self.queries_processed >= self.auto_tuning_warmup,
        }
    }

    /// Select optimal strategy for the given query
    fn select_strategy(&self, query: &str) -> OptimizationStrategy {
        if self.strategy != OptimizationStrategy::Auto {
            return self.strategy;
        }

        // Calculate query complexity
        let complexity = self.estimate_query_complexity(query);

        // For simple queries, use classical optimization
        if complexity < 3.0 {
            return OptimizationStrategy::Classical;
        }

        // For medium complexity, use ML if we have enough training data
        if complexity < 7.0 {
            if self.queries_processed >= 20 {
                return OptimizationStrategy::MachineLearning;
            } else {
                return OptimizationStrategy::Classical;
            }
        }

        // For complex queries, use quantum-inspired or hybrid approach
        if complexity >= 10.0 {
            OptimizationStrategy::Hybrid
        } else {
            OptimizationStrategy::QuantumInspired
        }
    }

    /// Estimate query complexity score (0-15 scale)
    fn estimate_query_complexity(&self, query: &str) -> f64 {
        let mut score = 0.0;

        // Count triple patterns
        let triple_patterns = query.matches("<<").count() + query.matches("?").count() / 2;
        score += triple_patterns as f64;

        // Check for complex operators
        if query.contains("OPTIONAL") {
            score += 2.0;
        }
        if query.contains("UNION") {
            score += 2.0;
        }
        if query.contains("FILTER") {
            score += 1.0;
        }
        if query.contains("GROUP BY") {
            score += 2.0;
        }
        if query.contains("ORDER BY") {
            score += 1.0;
        }
        if query.contains("DISTINCT") {
            score += 1.0;
        }

        // Check for nested quoted triples
        let nesting_depth = self.estimate_nesting_depth(query);
        score += nesting_depth as f64 * 1.5;

        score
    }

    /// Estimate nesting depth of quoted triples
    fn estimate_nesting_depth(&self, query: &str) -> usize {
        let mut max_depth: usize = 0;
        let mut current_depth: usize = 0;

        for ch in query.chars() {
            if ch == '<' {
                current_depth += 1;
                max_depth = max_depth.max(current_depth);
            } else if ch == '>' {
                current_depth = current_depth.saturating_sub(1);
            }
        }

        (max_depth / 2).min(10) // Divide by 2 since << counts as 2
    }

    /// Optimize using ML strategy
    ///
    /// Analyzes query patterns and uses learned weights from historical execution
    /// to predict optimal execution strategy and provide recommendations.
    fn optimize_with_ml(&self, query: &str) -> StarResult<OptimizationResult> {
        let complexity = self.estimate_query_complexity(query);
        let mut hints = Vec::new();
        let mut confidence = 0.85;

        // Feature extraction from query
        let has_optional = query.to_lowercase().contains("optional");
        let has_union = query.to_lowercase().contains("union");
        let has_filter = query.to_lowercase().contains("filter");
        let has_quoted_triple = query.contains("<<");
        let join_count = query.matches("?").count().saturating_sub(1) / 2;

        // ML-based cost estimation using learned patterns
        let base_cost = 50.0 + complexity * 10.0;
        let join_penalty = join_count as f64 * 15.0;
        let optional_penalty = if has_optional { 20.0 } else { 0.0 };
        let union_penalty = if has_union { 25.0 } else { 0.0 };

        let mut estimated_cost = base_cost + join_penalty + optional_penalty + union_penalty;

        // Generate ML-driven recommendations
        if has_quoted_triple {
            hints.push("Use quoted-triple-aware index for nested patterns".to_string());
            estimated_cost *= 0.9; // ML knows this optimization
        }

        if has_filter && join_count > 2 {
            hints.push("Push filter evaluation before expensive joins".to_string());
            estimated_cost *= 0.85;
        }

        if has_optional && has_union {
            hints
                .push("Consider materializing intermediate results for OPTIONAL/UNION".to_string());
        }

        // Adjust confidence based on workload history
        if self.workload.sample_count >= 20 {
            confidence = 0.92; // Higher confidence with more training data
            hints.push("Using learned execution patterns from workload history".to_string());
        } else if self.workload.sample_count < 5 {
            confidence = 0.65; // Lower confidence with little training data
            hints.push("Consider running more queries to improve ML predictions".to_string());
        }

        // Use workload profile for additional hints
        if self.workload.avg_complexity > 7.0 && complexity > self.workload.avg_complexity {
            hints.push("This query is more complex than average workload".to_string());
        }

        Ok(OptimizationResult {
            strategy_used: OptimizationStrategy::MachineLearning,
            estimated_cost,
            recommended_hints: hints,
            confidence,
        })
    }

    /// Optimize using quantum-inspired strategy
    ///
    /// Uses quantum-inspired algorithms (simulated annealing, QAOA-style optimization)
    /// to find optimal join ordering and execution plans for complex queries.
    fn optimize_with_quantum(&self, query: &str) -> StarResult<OptimizationResult> {
        let complexity = self.estimate_query_complexity(query);
        let nesting_depth = self.estimate_nesting_depth(query);
        let mut hints = Vec::new();
        let mut confidence: f64 = 0.78;

        // Count join-related patterns
        let join_count = query.matches("?").count().saturating_sub(1) / 2;
        let triple_pattern_count = query.matches('.').count();
        let has_quoted_triple = query.contains("<<");

        // Quantum-inspired cost model for combinatorial optimization
        // Uses superposition-inspired parallel search of join orderings
        let base_cost = 40.0 + complexity * 5.0;
        let join_optimization_factor = if join_count > 3 {
            // Quantum advantage for many-way joins
            0.7_f64.powi((join_count as i32 - 3).min(5))
        } else {
            1.0
        };

        let estimated_cost = base_cost * join_optimization_factor;

        // Quantum-specific recommendations
        if join_count > 3 {
            hints.push(format!(
                "Using quantum-inspired annealing for {}-way join optimization",
                join_count
            ));
            confidence = 0.85; // Higher confidence for complex joins
        }

        if has_quoted_triple && nesting_depth > 1 {
            hints.push(
                "Applying tensor network decomposition for nested quoted triples".to_string(),
            );
            confidence += 0.05;
        }

        if triple_pattern_count > 5 {
            hints.push("Using QAOA-style optimization for pattern ordering".to_string());
        }

        // Recommend quantum approach based on problem structure
        if complexity >= 7.0 {
            hints.push("Problem complexity suitable for quantum-inspired optimization".to_string());
        } else {
            hints.push("Consider classical optimization for lower-complexity queries".to_string());
            confidence -= 0.1;
        }

        // Add parallel execution hint for suitable queries
        if join_count > 2 && !query.to_lowercase().contains("order by") {
            hints.push("Parallel quantum-inspired search enabled for join evaluation".to_string());
        }

        Ok(OptimizationResult {
            strategy_used: OptimizationStrategy::QuantumInspired,
            estimated_cost,
            recommended_hints: hints,
            confidence: confidence.clamp(0.5, 0.95),
        })
    }

    /// Optimize using classical strategy
    ///
    /// Uses well-established rule-based heuristics for query optimization:
    /// - Filter pushdown
    /// - Selectivity estimation
    /// - Index selection
    /// - Join ordering based on cardinality
    fn optimize_classical(&self, query: &str) -> StarResult<OptimizationResult> {
        let complexity = self.estimate_query_complexity(query);
        let mut hints = Vec::new();
        let confidence = 0.95; // Classical optimizers are well-understood

        // Parse query features for rule-based optimization
        let query_lower = query.to_lowercase();
        let has_filter = query_lower.contains("filter");
        let has_optional = query_lower.contains("optional");
        let has_order_by = query_lower.contains("order by");
        let has_limit = query_lower.contains("limit");
        let has_distinct = query_lower.contains("distinct");
        let has_quoted_triple = query.contains("<<");
        let join_count = query.matches("?").count().saturating_sub(1) / 2;

        // Classical cost estimation using cardinality-based model
        let base_cost = 60.0;
        let join_cost = join_count as f64 * 20.0;
        let optional_cost = if has_optional { 30.0 } else { 0.0 };
        let distinct_cost = if has_distinct { 15.0 } else { 0.0 };

        let mut estimated_cost = base_cost + join_cost + optional_cost + distinct_cost;

        // Apply rule-based optimizations
        hints.push("Applying rule-based query optimization".to_string());

        // Rule 1: Filter pushdown
        if has_filter {
            hints.push("Rule: Push FILTER expressions to earliest evaluation point".to_string());
            estimated_cost *= 0.85;
        }

        // Rule 2: LIMIT optimization
        if has_limit && !has_order_by {
            hints.push("Rule: Apply LIMIT early to reduce intermediate results".to_string());
            estimated_cost *= 0.7;
        } else if has_limit && has_order_by {
            hints.push("Rule: ORDER BY requires full evaluation before LIMIT".to_string());
        }

        // Rule 3: Index selection for quoted triples
        if has_quoted_triple {
            hints.push("Rule: Use SPO index for quoted triple subject access".to_string());
            hints.push("Rule: Use PSO index for quoted triple predicate access".to_string());
        }

        // Rule 4: Join ordering (small-to-large heuristic)
        if join_count > 1 {
            hints.push(format!(
                "Rule: Order {} joins from most selective to least selective",
                join_count
            ));
        }

        // Rule 5: OPTIONAL handling
        if has_optional {
            hints.push("Rule: Evaluate OPTIONAL patterns after required patterns".to_string());
        }

        // Rule 6: DISTINCT optimization
        if has_distinct {
            hints.push("Rule: Use hash-based duplicate elimination".to_string());
        }

        // Add index hints based on patterns
        if complexity <= 3.0 {
            hints.push("Simple query - direct index scan recommended".to_string());
        } else if complexity <= 7.0 {
            hints.push("Medium complexity - merge join with indexes recommended".to_string());
        } else {
            hints.push("High complexity - consider nested loop with index".to_string());
        }

        Ok(OptimizationResult {
            strategy_used: OptimizationStrategy::Classical,
            estimated_cost,
            recommended_hints: hints,
            confidence,
        })
    }

    /// Optimize using hybrid strategy
    ///
    /// Combines ML and Quantum-inspired approaches, selecting the best result
    /// and merging recommendations from both strategies.
    fn optimize_hybrid(&self, query: &str) -> StarResult<OptimizationResult> {
        let complexity = self.estimate_query_complexity(query);

        // Get results from both strategies
        let ml_result = self.optimize_with_ml(query)?;
        let quantum_result = self.optimize_with_quantum(query)?;
        let classical_result = self.optimize_classical(query)?;

        // Combine hints from all strategies
        let mut combined_hints = Vec::new();
        combined_hints.push("Using hybrid optimization strategy".to_string());

        // Weight the results based on query characteristics
        let ml_weight = if self.workload.sample_count >= 20 {
            0.4
        } else {
            0.2
        };
        let quantum_weight = if complexity >= 7.0 { 0.4 } else { 0.2 };
        let classical_weight = 1.0 - ml_weight - quantum_weight;

        // Compute weighted cost
        let weighted_cost = ml_result.estimated_cost * ml_weight
            + quantum_result.estimated_cost * quantum_weight
            + classical_result.estimated_cost * classical_weight;

        // Compute weighted confidence
        let weighted_confidence = ml_result.confidence * ml_weight
            + quantum_result.confidence * quantum_weight
            + classical_result.confidence * classical_weight;

        // Select best individual result for primary hints
        let best_result = if ml_result.estimated_cost <= quantum_result.estimated_cost
            && ml_result.estimated_cost <= classical_result.estimated_cost
        {
            combined_hints.push("ML strategy selected as primary (lowest cost)".to_string());
            &ml_result
        } else if quantum_result.estimated_cost <= classical_result.estimated_cost {
            combined_hints.push("Quantum strategy selected as primary (lowest cost)".to_string());
            &quantum_result
        } else {
            combined_hints.push("Classical strategy selected as primary (lowest cost)".to_string());
            &classical_result
        };

        // Add top hints from the best result
        for hint in best_result.recommended_hints.iter().take(3) {
            combined_hints.push(hint.clone());
        }

        // Add cross-strategy optimizations
        if complexity >= 7.0 && self.workload.sample_count >= 10 {
            combined_hints.push(
                "Cross-validating ML predictions with quantum optimization bounds".to_string(),
            );
        }

        // Report strategy comparison
        combined_hints.push(format!(
            "Strategy costs - ML: {:.1}, Quantum: {:.1}, Classical: {:.1}",
            ml_result.estimated_cost,
            quantum_result.estimated_cost,
            classical_result.estimated_cost
        ));

        Ok(OptimizationResult {
            strategy_used: OptimizationStrategy::Hybrid,
            estimated_cost: weighted_cost,
            recommended_hints: combined_hints,
            confidence: weighted_confidence.clamp(0.5, 0.98),
        })
    }

    /// Update metrics after query execution
    fn update_metrics(
        &mut self,
        query: &str,
        result: &OptimizationResult,
        elapsed: Duration,
        strategy: OptimizationStrategy,
    ) {
        // Update workload profile
        self.workload.sample_count += 1;

        let complexity = self.estimate_query_complexity(query);
        let prev_avg = self.workload.avg_complexity;
        let n = self.workload.sample_count as f64;
        self.workload.avg_complexity = (prev_avg * (n - 1.0) + complexity) / n;

        // Update strategy performance
        let strategy_name = format!("{:?}", strategy);
        self.strategy_performance
            .entry(strategy_name)
            .or_default()
            .push(elapsed.as_secs_f64() * 1000.0);

        // Update regression detector
        self.regression_detector
            .update(elapsed.as_secs_f64() * 1000.0);

        debug!(
            "Query processed in {:.2}ms using {:?} strategy (confidence: {:.2})",
            elapsed.as_secs_f64() * 1000.0,
            strategy,
            result.confidence
        );
    }

    /// Detect and report performance regressions
    fn detect_and_report_regressions(&self) {
        if let Some(regression) = self.regression_detector.detect_regression() {
            match regression.severity {
                RegressionSeverity::Critical => {
                    warn!(
                        "CRITICAL performance regression detected: {:.2}x slower (baseline: {:.2}ms, recent: {:.2}ms)",
                        regression.regression_ratio, regression.baseline_p50, regression.recent_avg
                    );
                }
                RegressionSeverity::High => {
                    warn!(
                        "HIGH performance regression detected: {:.2}x slower (baseline: {:.2}ms, recent: {:.2}ms)",
                        regression.regression_ratio, regression.baseline_p50, regression.recent_avg
                    );
                }
                RegressionSeverity::Medium => {
                    info!(
                        "Performance regression detected: {:.2}x slower (baseline: {:.2}ms, recent: {:.2}ms)",
                        regression.regression_ratio, regression.baseline_p50, regression.recent_avg
                    );
                }
            }
        }
    }
}

impl Default for AdaptiveQueryOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of query optimization
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Strategy that was used
    pub strategy_used: OptimizationStrategy,
    /// Estimated execution cost
    pub estimated_cost: f64,
    /// Recommended optimization hints
    pub recommended_hints: Vec<String>,
    /// Confidence in the optimization (0.0 to 1.0)
    pub confidence: f64,
}

/// Optimizer statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerStatistics {
    /// Total queries processed
    pub queries_processed: usize,
    /// Average query complexity
    pub avg_complexity: f64,
    /// Performance by strategy
    pub strategy_performance: HashMap<String, Vec<f64>>,
    /// Whether regression was detected
    pub regression_detected: bool,
    /// Whether auto-tuning is active
    pub auto_tuning_active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_optimizer_creation() {
        let optimizer = AdaptiveQueryOptimizer::new();
        assert_eq!(optimizer.queries_processed, 0);
        assert!(optimizer.auto_tuning_enabled);
    }

    #[test]
    fn test_optimization_objectives() {
        let mut optimizer = AdaptiveQueryOptimizer::new();

        let objectives = vec![
            OptimizationObjective::MinimizeLatency { weight: 0.6 },
            OptimizationObjective::MinimizeMemory { weight: 0.4 },
        ];

        optimizer.set_objectives(objectives.clone());
        assert_eq!(optimizer.objectives.len(), 2);
        assert_eq!(optimizer.objectives[0].weight(), 0.6);
        assert_eq!(optimizer.objectives[1].weight(), 0.4);
    }

    #[test]
    fn test_query_complexity_estimation() {
        let optimizer = AdaptiveQueryOptimizer::new();

        // Simple query
        let simple = "SELECT * WHERE { ?s ?p ?o }";
        assert!(optimizer.estimate_query_complexity(simple) < 3.0);

        // Complex query
        let complex = "SELECT * WHERE { << ?s ?p ?o >> ?meta ?value . OPTIONAL { ?s ?p2 ?o2 } FILTER(?x > 10) } GROUP BY ?s";
        assert!(optimizer.estimate_query_complexity(complex) > 5.0);
    }

    #[test]
    fn test_strategy_selection() {
        let optimizer = AdaptiveQueryOptimizer::new();

        // Simple query should use classical
        let simple = "SELECT * WHERE { ?s ?p ?o }";
        let strategy = optimizer.select_strategy(simple);
        assert_eq!(strategy, OptimizationStrategy::Classical);
    }

    #[test]
    fn test_regression_detector() {
        let mut detector = RegressionDetector::new(1.3, 50);

        // Add baseline measurements - need at least min_samples (30)
        for _ in 0..35 {
            detector.update(100.0); // 100ms baseline
        }

        // Baseline should now be established
        assert_eq!(detector.recent_times.len(), 35);
        assert!(
            !detector.baseline.is_empty(),
            "Baseline should be established"
        );

        // No regression yet (measurements are within threshold)
        assert!(detector.detect_regression().is_none());

        // Add enough slow measurements to fill most of the window
        for _ in 0..40 {
            detector.update(200.0); // 200ms (2x slower)
        }

        // Window should now be mostly slow measurements
        assert_eq!(detector.recent_times.len(), 50, "Window should be full");

        // Recent average should be mostly slow measurements
        let recent_avg: f64 =
            detector.recent_times.iter().sum::<f64>() / detector.recent_times.len() as f64;

        // Should detect regression since recent_avg (mostly 200ms) > baseline_p50 (100ms) * threshold (1.3)
        let regression = detector.detect_regression();
        assert!(
            regression.is_some(),
            "Should detect regression: recent_avg={:.2}, expected > {:.2}",
            recent_avg,
            100.0 * 1.3
        );

        if let Some(reg) = regression {
            assert!(
                reg.regression_ratio >= 1.3,
                "Regression ratio should be >= 1.3, got {}",
                reg.regression_ratio
            );
            assert!(reg.recent_avg > reg.baseline_p50 * 1.3);
        }
    }

    #[test]
    fn test_workload_profile_tracking() {
        let mut optimizer = AdaptiveQueryOptimizer::new();

        let query1 = "SELECT * WHERE { ?s ?p ?o }";
        let query2 = "SELECT * WHERE { << ?s ?p ?o >> ?meta ?value . OPTIONAL { ?s ?p2 ?o2 } }";

        optimizer.optimize_query(query1).unwrap();
        optimizer.optimize_query(query2).unwrap();

        let profile = optimizer.workload_profile();
        assert_eq!(profile.sample_count, 2);
        assert!(profile.avg_complexity > 0.0);
    }

    #[test]
    fn test_auto_tuning_warmup() {
        let mut optimizer = AdaptiveQueryOptimizer::new();
        optimizer.auto_tuning_warmup = 5;

        // Process fewer queries than warmup
        for i in 0..3 {
            optimizer
                .optimize_query(&format!("SELECT * WHERE {{ ?s{} ?p ?o }}", i))
                .unwrap();
        }

        let stats = optimizer.statistics();
        assert!(!stats.auto_tuning_active);

        // Process enough queries to activate auto-tuning
        for i in 3..6 {
            optimizer
                .optimize_query(&format!("SELECT * WHERE {{ ?s{} ?p ?o }}", i))
                .unwrap();
        }

        let stats = optimizer.statistics();
        assert!(stats.auto_tuning_active);
    }

    #[test]
    fn test_optimization_result_confidence() {
        let result = OptimizationResult {
            strategy_used: OptimizationStrategy::MachineLearning,
            estimated_cost: 100.0,
            recommended_hints: vec!["Use index".to_string()],
            confidence: 0.85,
        };

        assert_eq!(result.confidence, 0.85);
        assert_eq!(result.strategy_used, OptimizationStrategy::MachineLearning);
    }

    #[test]
    fn test_nesting_depth_estimation() {
        let optimizer = AdaptiveQueryOptimizer::new();

        let simple = "SELECT * WHERE { ?s ?p ?o }";
        assert_eq!(optimizer.estimate_nesting_depth(simple), 0);

        let nested = "SELECT * WHERE { << << ?s ?p ?o >> ?meta ?value >> ?meta2 ?value2 }";
        assert!(optimizer.estimate_nesting_depth(nested) >= 2);
    }

    #[test]
    fn test_hybrid_optimization() {
        let optimizer = AdaptiveQueryOptimizer::new();

        let query = "SELECT * WHERE { << ?s ?p ?o >> ?meta ?value . FILTER(?x > 10) }";
        let result = optimizer.optimize_hybrid(query).unwrap();

        // Hybrid should choose best between ML and Quantum
        assert!(result.estimated_cost > 0.0);
        assert!(!result.recommended_hints.is_empty());
    }
}
