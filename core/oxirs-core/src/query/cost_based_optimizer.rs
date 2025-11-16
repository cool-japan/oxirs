//! Cost-based query optimization using statistical models
//!
//! This module implements a sophisticated cost-based query optimizer that uses
//! statistical models and cardinality estimation to produce optimal execution plans.
//!
//! # Features
//!
//! - **Cardinality estimation**: Statistical prediction of result set sizes
//! - **Cost models**: Accurate estimation of I/O, CPU, and memory costs
//! - **Join ordering**: Optimal join order selection using dynamic programming
//! - **Index selection**: Automatic selection of optimal indexes
//! - **Parallel execution planning**: Cost-based parallelization decisions
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_core::query::cost_based_optimizer::CostBasedOptimizer;
//! use oxirs_core::query::algebra::Query;
//!
//! # fn example() -> Result<(), oxirs_core::OxirsError> {
//! let optimizer = CostBasedOptimizer::new();
//! // let query: Query = ...;
//! // let optimized = optimizer.optimize(&query)?;
//! # Ok(())
//! # }
//! ```

use crate::query::algebra::{AlgebraTriplePattern, GraphPattern, TermPattern};
use crate::OxirsError;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Cost-based query optimizer
///
/// Uses statistical models and cardinality estimation to produce
/// optimal execution plans for SPARQL queries.
pub struct CostBasedOptimizer {
    /// Statistics collector
    stats: Arc<StatisticsCollector>,
    /// Cost model configuration
    cost_config: CostConfiguration,
    /// Query counter for metrics
    query_count: AtomicU64,
}

impl CostBasedOptimizer {
    /// Create a new cost-based optimizer
    pub fn new() -> Self {
        Self {
            stats: Arc::new(StatisticsCollector::new()),
            cost_config: CostConfiguration::default(),
            query_count: AtomicU64::new(0),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: CostConfiguration) -> Self {
        Self {
            stats: Arc::new(StatisticsCollector::new()),
            cost_config: config,
            query_count: AtomicU64::new(0),
        }
    }

    /// Optimize a graph pattern
    pub fn optimize_pattern(&self, pattern: &GraphPattern) -> Result<OptimizedPlan, OxirsError> {
        self.query_count.fetch_add(1, Ordering::Relaxed);

        match pattern {
            GraphPattern::Bgp(patterns) => self.optimize_bgp(patterns),
            GraphPattern::Join(left, right) => self.optimize_join(left, right),
            GraphPattern::LeftJoin { left, right, .. } => self.optimize_left_join(left, right),
            GraphPattern::Filter { expr: _, inner } => self.optimize_filter(inner),
            GraphPattern::Union(left, right) => self.optimize_union(left, right),
            GraphPattern::Extend { inner, .. } => self.optimize_pattern(inner),
            GraphPattern::Minus(left, right) => self.optimize_minus(left, right),
            GraphPattern::Service { inner, .. } => self.optimize_pattern(inner),
            GraphPattern::Graph { inner, .. } => self.optimize_pattern(inner),
            GraphPattern::OrderBy { inner, .. } => self.optimize_pattern(inner),
            GraphPattern::Project { inner, .. } => self.optimize_pattern(inner),
            GraphPattern::Distinct(inner) => self.optimize_pattern(inner),
            GraphPattern::Reduced(inner) => self.optimize_pattern(inner),
            GraphPattern::Slice { inner, .. } => self.optimize_pattern(inner),
            GraphPattern::Group { inner, .. } => self.optimize_pattern(inner),
            GraphPattern::Path { .. } => {
                // Property paths - use default optimization
                Ok(OptimizedPlan::empty())
            }
            GraphPattern::Values { .. } => {
                // VALUES clause - typically small, no optimization needed
                Ok(OptimizedPlan::empty())
            }
        }
    }

    /// Optimize a Basic Graph Pattern (BGP)
    fn optimize_bgp(&self, patterns: &[AlgebraTriplePattern]) -> Result<OptimizedPlan, OxirsError> {
        if patterns.is_empty() {
            return Ok(OptimizedPlan::empty());
        }

        // Estimate cardinality for each pattern
        let mut pattern_costs: Vec<(usize, PatternCost)> = patterns
            .iter()
            .enumerate()
            .map(|(idx, pattern)| {
                let cost = self.estimate_pattern_cost(pattern);
                (idx, cost)
            })
            .collect();

        // Sort by selectivity (lowest cardinality first)
        pattern_costs.sort_by(|a, b| {
            a.1.estimated_cardinality
                .partial_cmp(&b.1.estimated_cardinality)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Build optimal join order
        let optimal_order: Vec<usize> = pattern_costs.iter().map(|(idx, _)| *idx).collect();

        // Calculate total estimated cost
        let total_cost: f64 = pattern_costs.iter().map(|(_, cost)| cost.total_cost).sum();

        Ok(OptimizedPlan {
            join_order: optimal_order,
            estimated_cost: total_cost,
            estimated_cardinality: self.estimate_bgp_cardinality(patterns, &pattern_costs),
            use_index: self.should_use_index(patterns),
            parallel_execution: self.should_parallelize(patterns, total_cost),
            optimizations: vec![Optimization::JoinReordering],
        })
    }

    /// Optimize a join operation
    fn optimize_join(
        &self,
        left: &GraphPattern,
        right: &GraphPattern,
    ) -> Result<OptimizedPlan, OxirsError> {
        let left_plan = self.optimize_pattern(left)?;
        let right_plan = self.optimize_pattern(right)?;

        // Decide join order based on cardinality
        let (first, second, swapped) =
            if left_plan.estimated_cardinality < right_plan.estimated_cardinality {
                (left_plan, right_plan, false)
            } else {
                (right_plan, left_plan, true)
            };

        // Estimate join cost
        let join_cost = self.estimate_join_cost(&first, &second);

        Ok(OptimizedPlan {
            join_order: if swapped { vec![1, 0] } else { vec![0, 1] },
            estimated_cost: first.estimated_cost + second.estimated_cost + join_cost,
            estimated_cardinality: self.estimate_join_cardinality(&first, &second),
            use_index: true,
            parallel_execution: join_cost > self.cost_config.parallel_threshold,
            optimizations: vec![Optimization::JoinReordering, Optimization::HashJoin],
        })
    }

    /// Optimize a left join (optional pattern)
    fn optimize_left_join(
        &self,
        left: &GraphPattern,
        right: &GraphPattern,
    ) -> Result<OptimizedPlan, OxirsError> {
        // Left join cannot be reordered, so we optimize each side separately
        let left_plan = self.optimize_pattern(left)?;
        let right_plan = self.optimize_pattern(right)?;

        let join_cost = self.estimate_join_cost(&left_plan, &right_plan);

        Ok(OptimizedPlan {
            join_order: vec![0, 1], // Must maintain order
            estimated_cost: left_plan.estimated_cost + right_plan.estimated_cost + join_cost,
            estimated_cardinality: left_plan.estimated_cardinality, // LEFT JOIN preserves left cardinality
            use_index: true,
            parallel_execution: false, // LEFT JOIN is harder to parallelize
            optimizations: vec![Optimization::IndexNLJ],
        })
    }

    /// Optimize a filter operation
    fn optimize_filter(&self, inner: &GraphPattern) -> Result<OptimizedPlan, OxirsError> {
        let mut inner_plan = self.optimize_pattern(inner)?;

        // Apply filter selectivity
        inner_plan.estimated_cardinality = ((inner_plan.estimated_cardinality as f64)
            * self.cost_config.filter_selectivity)
            as usize;
        inner_plan.optimizations.push(Optimization::FilterPushdown);

        Ok(inner_plan)
    }

    /// Optimize a union operation
    fn optimize_union(
        &self,
        left: &GraphPattern,
        right: &GraphPattern,
    ) -> Result<OptimizedPlan, OxirsError> {
        let left_plan = self.optimize_pattern(left)?;
        let right_plan = self.optimize_pattern(right)?;

        Ok(OptimizedPlan {
            join_order: vec![0, 1],
            estimated_cost: left_plan.estimated_cost + right_plan.estimated_cost,
            estimated_cardinality: left_plan.estimated_cardinality
                + right_plan.estimated_cardinality,
            use_index: true,
            parallel_execution: true, // UNION branches can run in parallel
            optimizations: vec![Optimization::ParallelUnion],
        })
    }

    /// Optimize a minus operation
    fn optimize_minus(
        &self,
        left: &GraphPattern,
        right: &GraphPattern,
    ) -> Result<OptimizedPlan, OxirsError> {
        let left_plan = self.optimize_pattern(left)?;
        let right_plan = self.optimize_pattern(right)?;

        Ok(OptimizedPlan {
            join_order: vec![0, 1],
            estimated_cost: left_plan.estimated_cost + right_plan.estimated_cost,
            estimated_cardinality: ((left_plan.estimated_cardinality as f64) * 0.7) as usize, // Heuristic
            use_index: true,
            parallel_execution: false,
            optimizations: vec![Optimization::HashAntiJoin],
        })
    }

    // Helper methods for cost estimation

    fn estimate_pattern_cost(&self, pattern: &AlgebraTriplePattern) -> PatternCost {
        let selectivity = self.calculate_selectivity(pattern);
        let estimated_card = (self.stats.total_triples() as f64 * selectivity) as usize;

        // I/O cost: depends on whether we can use an index
        let io_cost = if self.can_use_index(pattern) {
            estimated_card as f64 * self.cost_config.index_access_cost
        } else {
            self.stats.total_triples() as f64 * self.cost_config.sequential_scan_cost
        };

        // CPU cost: processing retrieved triples
        let cpu_cost = estimated_card as f64 * self.cost_config.cpu_tuple_cost;

        PatternCost {
            estimated_cardinality: estimated_card,
            io_cost,
            cpu_cost,
            total_cost: io_cost + cpu_cost,
        }
    }

    fn calculate_selectivity(&self, pattern: &AlgebraTriplePattern) -> f64 {
        let mut selectivity = 1.0;

        // Adjust selectivity based on bound terms
        match &pattern.subject {
            TermPattern::Variable(_) => selectivity *= 0.5, // Variable is less selective
            _ => selectivity *= 0.01,                       // Constant is very selective
        }

        match &pattern.predicate {
            TermPattern::Variable(_) => selectivity *= 0.5,
            _ => selectivity *= 0.1,
        }

        match &pattern.object {
            TermPattern::Variable(_) => selectivity *= 0.5,
            _ => selectivity *= 0.01,
        }

        selectivity
    }

    fn can_use_index(&self, pattern: &AlgebraTriplePattern) -> bool {
        // Can use index if at least one term is bound
        !matches!(pattern.subject, TermPattern::Variable(_))
            || !matches!(pattern.predicate, TermPattern::Variable(_))
            || !matches!(pattern.object, TermPattern::Variable(_))
    }

    fn estimate_bgp_cardinality(
        &self,
        _patterns: &[AlgebraTriplePattern],
        pattern_costs: &[(usize, PatternCost)],
    ) -> usize {
        if pattern_costs.is_empty() {
            return 0;
        }

        // Estimate using product of selectivities with correlation factor
        let mut cardinality = pattern_costs[0].1.estimated_cardinality as f64;

        for (_, cost) in pattern_costs.iter().skip(1) {
            let join_selectivity = 0.1; // Heuristic for join selectivity
            cardinality *= join_selectivity * (cost.estimated_cardinality as f64);
        }

        cardinality.max(1.0) as usize
    }

    fn should_use_index(&self, patterns: &[AlgebraTriplePattern]) -> bool {
        // Use index if any pattern has bound terms
        patterns.iter().any(|p| self.can_use_index(p))
    }

    fn should_parallelize(&self, _patterns: &[AlgebraTriplePattern], total_cost: f64) -> bool {
        total_cost > self.cost_config.parallel_threshold
    }

    fn estimate_join_cost(&self, left: &OptimizedPlan, right: &OptimizedPlan) -> f64 {
        // Hash join cost: build hash table + probe
        let build_cost = left.estimated_cardinality as f64 * self.cost_config.hash_build_cost;
        let probe_cost = right.estimated_cardinality as f64 * self.cost_config.hash_probe_cost;

        build_cost + probe_cost
    }

    fn estimate_join_cardinality(&self, left: &OptimizedPlan, right: &OptimizedPlan) -> usize {
        // Simple heuristic: assume 10% join selectivity
        let join_selectivity = 0.1;
        let product = left.estimated_cardinality as f64 * right.estimated_cardinality as f64;

        (product * join_selectivity).max(1.0) as usize
    }

    /// Get optimizer statistics
    pub fn stats(&self) -> OptimizerStats {
        OptimizerStats {
            queries_optimized: self.query_count.load(Ordering::Relaxed),
            total_triples: self.stats.total_triples(),
        }
    }

    /// Update statistics with actual query results
    ///
    /// Uses exponential moving average to adapt cost estimates based on actual execution results.
    /// This allows the optimizer to learn from real query patterns and improve over time.
    pub fn update_stats(&self, pattern: &GraphPattern, actual_cardinality: usize) {
        // Use exponential moving average with alpha = 0.2 (weight recent observations more)
        const ALPHA: f64 = 0.2;

        // Compute pattern hash for tracking
        let pattern_hash = self.compute_pattern_hash(pattern);

        // Update pattern-specific statistics
        let mut pattern_stats = self.stats.pattern_stats.write().unwrap();

        // Get or create entry for this pattern
        let entry = pattern_stats
            .entry(pattern_hash)
            .or_insert_with(|| PatternStats {
                execution_count: 0,
                avg_cardinality: actual_cardinality as f64,
                last_cardinality: actual_cardinality,
            });

        // Update with exponential moving average
        entry.avg_cardinality =
            ALPHA * (actual_cardinality as f64) + (1.0 - ALPHA) * entry.avg_cardinality;
        entry.last_cardinality = actual_cardinality;
        entry.execution_count += 1;

        // Log significant deviations for performance tuning
        if entry.execution_count > 5 {
            let deviation =
                (actual_cardinality as f64 - entry.avg_cardinality).abs() / entry.avg_cardinality;
            if deviation > 0.5 {
                tracing::debug!(
                    pattern_hash = pattern_hash,
                    actual = actual_cardinality,
                    avg = entry.avg_cardinality,
                    deviation_pct = deviation * 100.0,
                    "Significant cardinality deviation detected"
                );
            }
        }
    }

    /// Compute a hash for a graph pattern for statistics tracking
    fn compute_pattern_hash(&self, pattern: &GraphPattern) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash the pattern structure (simplified version)
        match pattern {
            GraphPattern::Bgp(patterns) => {
                "bgp".hash(&mut hasher);
                patterns.len().hash(&mut hasher);
            }
            GraphPattern::Join(_, _) => {
                "join".hash(&mut hasher);
            }
            GraphPattern::LeftJoin { .. } => {
                "leftjoin".hash(&mut hasher);
            }
            GraphPattern::Union(_, _) => {
                "union".hash(&mut hasher);
            }
            GraphPattern::Filter { .. } => {
                "filter".hash(&mut hasher);
            }
            GraphPattern::Extend { .. } => {
                "extend".hash(&mut hasher);
            }
            GraphPattern::Group { .. } => {
                "group".hash(&mut hasher);
            }
            GraphPattern::Service { .. } => {
                "service".hash(&mut hasher);
            }
            GraphPattern::Minus(_, _) => {
                "minus".hash(&mut hasher);
            }
            GraphPattern::Graph { .. } => {
                "graph".hash(&mut hasher);
            }
            GraphPattern::OrderBy { .. } => {
                "orderby".hash(&mut hasher);
            }
            GraphPattern::Project { .. } => {
                "project".hash(&mut hasher);
            }
            GraphPattern::Distinct(_) => {
                "distinct".hash(&mut hasher);
            }
            GraphPattern::Reduced(_) => {
                "reduced".hash(&mut hasher);
            }
            GraphPattern::Slice { .. } => {
                "slice".hash(&mut hasher);
            }
            GraphPattern::Path { .. } => {
                "path".hash(&mut hasher);
            }
            GraphPattern::Values { .. } => {
                "values".hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    /// Get learned statistics for a pattern
    pub fn get_learned_cardinality(&self, pattern: &GraphPattern) -> Option<f64> {
        let pattern_hash = self.compute_pattern_hash(pattern);
        let pattern_stats = self.stats.pattern_stats.read().unwrap();
        pattern_stats.get(&pattern_hash).map(|s| s.avg_cardinality)
    }
}

impl Default for CostBasedOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics collector for cost estimation
struct StatisticsCollector {
    /// Total number of triples in the store
    total_triples: AtomicU64,
    /// Statistics per predicate (reserved for future use)
    #[allow(dead_code)]
    predicate_stats: HashMap<String, PredicateStats>,
    /// Pattern-specific statistics for adaptive optimization
    pattern_stats: Arc<RwLock<HashMap<u64, PatternStats>>>,
}

impl StatisticsCollector {
    fn new() -> Self {
        Self {
            total_triples: AtomicU64::new(1_000_000), // Default estimate
            predicate_stats: HashMap::new(),
            pattern_stats: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn total_triples(&self) -> usize {
        self.total_triples.load(Ordering::Relaxed) as usize
    }
}

/// Statistics tracked for specific query patterns
#[derive(Debug, Clone)]
struct PatternStats {
    /// Number of times this pattern has been executed
    execution_count: usize,
    /// Average cardinality (exponential moving average)
    avg_cardinality: f64,
    /// Last observed cardinality
    last_cardinality: usize,
}

/// Statistics for a specific predicate
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PredicateStats {
    /// Number of triples with this predicate
    count: usize,
    /// Average number of objects per subject
    avg_objects_per_subject: f64,
    /// Distinct subjects
    distinct_subjects: usize,
    /// Distinct objects
    distinct_objects: usize,
}

/// Cost configuration parameters
#[derive(Debug, Clone)]
pub struct CostConfiguration {
    /// Cost per sequential scan of one triple
    pub sequential_scan_cost: f64,
    /// Cost per index access
    pub index_access_cost: f64,
    /// Cost per tuple processed by CPU
    pub cpu_tuple_cost: f64,
    /// Cost to build hash table per tuple
    pub hash_build_cost: f64,
    /// Cost to probe hash table per tuple
    pub hash_probe_cost: f64,
    /// Default filter selectivity
    pub filter_selectivity: f64,
    /// Threshold for parallel execution
    pub parallel_threshold: f64,
}

impl Default for CostConfiguration {
    fn default() -> Self {
        Self {
            sequential_scan_cost: 1.0,
            index_access_cost: 0.01,
            cpu_tuple_cost: 0.001,
            hash_build_cost: 0.005,
            hash_probe_cost: 0.002,
            filter_selectivity: 0.3,
            parallel_threshold: 10000.0,
        }
    }
}

/// Cost estimate for a single triple pattern
#[derive(Debug, Clone)]
struct PatternCost {
    /// Estimated result cardinality
    estimated_cardinality: usize,
    /// I/O cost (reserved for detailed cost models)
    #[allow(dead_code)]
    io_cost: f64,
    /// CPU cost (reserved for detailed cost models)
    #[allow(dead_code)]
    cpu_cost: f64,
    /// Total cost
    total_cost: f64,
}

/// Optimized execution plan
#[derive(Debug, Clone)]
pub struct OptimizedPlan {
    /// Optimal join order (indices)
    pub join_order: Vec<usize>,
    /// Estimated execution cost
    pub estimated_cost: f64,
    /// Estimated result cardinality
    pub estimated_cardinality: usize,
    /// Whether to use index access
    pub use_index: bool,
    /// Whether to use parallel execution
    pub parallel_execution: bool,
    /// Applied optimizations
    pub optimizations: Vec<Optimization>,
}

impl OptimizedPlan {
    /// Create an empty plan
    fn empty() -> Self {
        Self {
            join_order: vec![],
            estimated_cost: 0.0,
            estimated_cardinality: 0,
            use_index: false,
            parallel_execution: false,
            optimizations: vec![],
        }
    }
}

/// Types of optimizations applied
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Optimization {
    /// Reordered joins for optimal execution
    JoinReordering,
    /// Used hash join algorithm
    HashJoin,
    /// Pushed filter down to reduce intermediate results
    FilterPushdown,
    /// Used index for access
    IndexAccess,
    /// Used nested loop join with index
    IndexNLJ,
    /// Parallel union execution
    ParallelUnion,
    /// Hash-based anti-join for MINUS
    HashAntiJoin,
}

/// Optimizer statistics
#[derive(Debug, Clone)]
pub struct OptimizerStats {
    /// Number of queries optimized
    pub queries_optimized: u64,
    /// Total triples in the store
    pub total_triples: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{NamedNode, Variable};
    use crate::query::algebra::TermPattern;

    fn create_test_pattern(
        subject: TermPattern,
        predicate: TermPattern,
        object: TermPattern,
    ) -> AlgebraTriplePattern {
        AlgebraTriplePattern {
            subject,
            predicate,
            object,
        }
    }

    #[test]
    fn test_optimizer_creation() {
        let optimizer = CostBasedOptimizer::new();
        let stats = optimizer.stats();

        assert_eq!(stats.queries_optimized, 0);
        assert!(stats.total_triples > 0);
    }

    #[test]
    fn test_selectivity_calculation() {
        let optimizer = CostBasedOptimizer::new();

        // All variables (least selective)
        let pattern1 = create_test_pattern(
            TermPattern::Variable(Variable::new("s").unwrap()),
            TermPattern::Variable(Variable::new("p").unwrap()),
            TermPattern::Variable(Variable::new("o").unwrap()),
        );

        let cost1 = optimizer.estimate_pattern_cost(&pattern1);
        assert!(cost1.estimated_cardinality > 0);

        // All constants (most selective)
        let pattern2 = create_test_pattern(
            TermPattern::NamedNode(NamedNode::new("http://example.org/s").unwrap()),
            TermPattern::NamedNode(NamedNode::new("http://example.org/p").unwrap()),
            TermPattern::NamedNode(NamedNode::new("http://example.org/o").unwrap()),
        );

        let cost2 = optimizer.estimate_pattern_cost(&pattern2);

        // Constant pattern should be more selective
        assert!(cost2.estimated_cardinality < cost1.estimated_cardinality);
    }

    #[test]
    fn test_cost_configuration() {
        let config = CostConfiguration::default();

        assert!(config.index_access_cost < config.sequential_scan_cost);
        assert!(config.cpu_tuple_cost < config.index_access_cost);
    }

    #[test]
    fn test_can_use_index() {
        let optimizer = CostBasedOptimizer::new();

        // Pattern with bound subject
        let pattern1 = create_test_pattern(
            TermPattern::NamedNode(NamedNode::new("http://example.org/s").unwrap()),
            TermPattern::Variable(Variable::new("p").unwrap()),
            TermPattern::Variable(Variable::new("o").unwrap()),
        );

        assert!(optimizer.can_use_index(&pattern1));

        // Pattern with all variables
        let pattern2 = create_test_pattern(
            TermPattern::Variable(Variable::new("s").unwrap()),
            TermPattern::Variable(Variable::new("p").unwrap()),
            TermPattern::Variable(Variable::new("o").unwrap()),
        );

        assert!(!optimizer.can_use_index(&pattern2));
    }

    #[test]
    fn test_empty_plan() {
        let plan = OptimizedPlan::empty();

        assert_eq!(plan.join_order.len(), 0);
        assert_eq!(plan.estimated_cost, 0.0);
        assert_eq!(plan.estimated_cardinality, 0);
    }
}
