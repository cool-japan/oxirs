//! Query Optimization Module
//!
//! This module provides query optimization capabilities including
//! rule-based and cost-based optimization passes.

use crate::algebra::{
    Algebra, BinaryOperator, Expression, Term, TriplePattern, UnaryOperator, Variable,
};
use crate::bgp_optimizer::{BGPOptimizer, OptimizedBGP};
use crate::statistics_collector::{
    DynamicStatisticsUpdater, QueryExecutionRecord, StatisticsCollector,
};
use anyhow::{anyhow, Result};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};
use tracing;

/// Query optimizer configuration
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Enable join reordering
    pub join_reordering: bool,
    /// Enable filter pushdown
    pub filter_pushdown: bool,
    /// Enable projection pushdown
    pub projection_pushdown: bool,
    /// Enable constant folding
    pub constant_folding: bool,
    /// Enable dead code elimination
    pub dead_code_elimination: bool,
    /// Enable cost-based optimization
    pub cost_based: bool,
    /// Maximum optimization passes
    pub max_passes: usize,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            join_reordering: true,
            filter_pushdown: true,
            projection_pushdown: true,
            constant_folding: true,
            dead_code_elimination: true,
            cost_based: true,
            max_passes: 10,
        }
    }
}

/// Statistics for cost-based optimization
#[derive(Debug, Clone, Default)]
pub struct Statistics {
    /// Variable selectivity estimates
    pub variable_selectivity: HashMap<Variable, f64>,
    /// Triple pattern cardinality estimates
    pub pattern_cardinality: HashMap<String, usize>,
    /// Join selectivity estimates
    pub join_selectivity: HashMap<String, f64>,
    /// Predicate frequency statistics
    pub predicate_frequency: HashMap<String, usize>,
    /// Subject/object cardinality statistics
    pub subject_cardinality: HashMap<String, usize>,
    pub object_cardinality: HashMap<String, usize>,
    /// Index statistics
    pub index_stats: IndexStatistics,
    /// Query execution history
    pub execution_history: Vec<ExecutionRecord>,
}

/// Index statistics for optimization
#[derive(Debug, Clone, Default)]
pub struct IndexStatistics {
    /// Available indexes
    pub available_indexes: HashSet<IndexType>,
    /// Index selectivity estimates
    pub index_selectivity: HashMap<IndexType, f64>,
    /// Index access cost estimates
    pub index_access_cost: HashMap<IndexType, f64>,
}

/// Index types available in the system
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum IndexType {
    SubjectPredicate,
    PredicateObject,
    SubjectObject,
    FullText,
    Spatial,
    Temporal,
    // RDF Triple pattern indices (SPO permutations)
    SPO, // Subject-Predicate-Object
    PSO, // Predicate-Subject-Object
    OSP, // Object-Subject-Predicate
    OPS, // Object-Predicate-Subject
    SOP, // Subject-Object-Predicate
    POS, // Predicate-Object-Subject
    // Additional index types
    Hash,
    BTree,
    Bitmap,
    Bloom,
    // Advanced index types for enhanced optimization
    BTreeIndex(IndexPosition),
    HashIndex(IndexPosition),
    BitmapIndex(IndexPosition),
    SpatialRTree,
    TemporalBTree,
    MultiColumnBTree(Vec<IndexPosition>),
    BloomFilter(IndexPosition),
    Custom(String),
}

/// Index position specification
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum IndexPosition {
    Subject,
    Predicate,
    Object,
    SubjectPredicate,
    PredicateObject,
    SubjectObject,
    FullTriple,
}

/// Execution record for learning-based optimization
#[derive(Debug, Clone)]
pub struct ExecutionRecord {
    pub query_hash: u64,
    pub algebra: Algebra,
    pub execution_time: Duration,
    pub cardinality: usize,
    pub memory_usage: usize,
    pub optimization_decisions: Vec<OptimizationDecision>,
}

/// Optimization decision record
#[derive(Debug, Clone)]
pub struct OptimizationDecision {
    pub optimization_type: OptimizationType,
    pub before_cost: f64,
    pub after_cost: f64,
    pub success: bool,
}

/// Types of optimizations
#[derive(Debug, Clone)]
pub enum OptimizationType {
    JoinReordering,
    FilterPushdown,
    ProjectionPushdown,
    ConstantFolding,
    IndexSelection,
    MaterializationPoint,
    ParallelizationStrategy,
}

/// Query characteristics for adaptive optimization
#[derive(Debug, Clone)]
pub struct QueryCharacteristics {
    pub variable_count: usize,
    pub has_object_variables: bool,
    pub has_literal_constraints: bool,
    pub join_complexity: JoinComplexity,
    pub pattern_count: usize,
}

/// Join complexity classification
#[derive(Debug, Clone)]
pub enum JoinComplexity {
    Simple,   // 1-2 patterns, simple joins
    Moderate, // 3-5 patterns, some complexity
    Complex,  // 6+ patterns or complex join patterns
}

/// Index union plan for OR conditions
#[derive(Debug, Clone)]
pub struct IndexUnionPlan {
    pub left_indexes: Vec<IndexType>,
    pub right_indexes: Vec<IndexType>,
    pub union_cost: f64,
    pub estimated_selectivity: f64,
}

/// Index filter plan for push-down optimization
#[derive(Debug, Clone)]
pub struct IndexFilterPlan {
    pub pattern_index: usize,
    pub filter_index: IndexType,
    pub push_down_conditions: Vec<Expression>,
    pub estimated_cost: f64,
}

impl Statistics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Estimate cardinality of a triple pattern
    pub fn estimate_pattern_cardinality(&self, pattern: &TriplePattern) -> usize {
        let pattern_key = format!("{}", pattern);
        self.pattern_cardinality
            .get(&pattern_key)
            .copied()
            .unwrap_or_else(|| {
                // Use advanced estimation based on term types
                self.estimate_pattern_cardinality_advanced(pattern)
            })
    }

    /// Advanced pattern cardinality estimation
    fn estimate_pattern_cardinality_advanced(&self, pattern: &TriplePattern) -> usize {
        let mut cardinality = 1000; // Base cardinality

        // Adjust based on subject term
        match &pattern.subject {
            Term::Iri(iri) => {
                cardinality = self
                    .subject_cardinality
                    .get(iri.as_str())
                    .copied()
                    .unwrap_or(100);
            }
            Term::Variable(_) => {
                cardinality *= 10; // Variables are less selective
            }
            _ => {}
        }

        // Adjust based on predicate term
        match &pattern.predicate {
            Term::Iri(iri) => {
                let pred_freq = self
                    .predicate_frequency
                    .get(iri.as_str())
                    .copied()
                    .unwrap_or(1000);
                cardinality = std::cmp::min(cardinality, pred_freq);
            }
            Term::Variable(_) => {
                cardinality *= 100; // Variable predicates are very unselective
            }
            _ => {}
        }

        // Adjust based on object term
        match &pattern.object {
            Term::Iri(iri) => {
                let obj_card = self
                    .object_cardinality
                    .get(iri.as_str())
                    .copied()
                    .unwrap_or(100);
                cardinality = std::cmp::min(cardinality, obj_card);
            }
            Term::Literal(_) => {
                cardinality /= 2; // Literals are more selective
            }
            Term::Variable(_) => {
                cardinality *= 5; // Variable objects are less selective
            }
            _ => {}
        }

        std::cmp::max(1, cardinality)
    }

    /// Estimate selectivity of a variable
    pub fn estimate_variable_selectivity(&self, var: &Variable) -> f64 {
        self.variable_selectivity.get(var).copied().unwrap_or(0.1)
    }

    /// Estimate join selectivity between two patterns
    pub fn estimate_join_selectivity(&self, left: &Algebra, right: &Algebra) -> f64 {
        let shared_vars = self.get_shared_variables(left, right);
        if shared_vars.is_empty() {
            1.0 // Cartesian product
        } else {
            // More shared variables typically mean higher selectivity
            let base_selectivity = 1.0 / (shared_vars.len() as f64 + 1.0);

            // Adjust based on variable selectivity
            let avg_var_selectivity: f64 = shared_vars
                .iter()
                .map(|var| self.estimate_variable_selectivity(var))
                .sum::<f64>()
                / shared_vars.len() as f64;

            base_selectivity * avg_var_selectivity
        }
    }

    /// Get shared variables between two algebra expressions
    fn get_shared_variables(&self, left: &Algebra, right: &Algebra) -> Vec<Variable> {
        let left_vars: HashSet<_> = left.variables().into_iter().collect();
        let right_vars: HashSet<_> = right.variables().into_iter().collect();
        left_vars.intersection(&right_vars).cloned().collect()
    }

    /// Estimate cost of an algebra expression with advanced heuristics
    pub fn estimate_cost(&self, algebra: &Algebra) -> f64 {
        match algebra {
            Algebra::Bgp(patterns) => {
                if patterns.is_empty() {
                    1.0
                } else {
                    // Use sophisticated BGP cost estimation
                    self.estimate_bgp_cost(patterns)
                }
            }
            Algebra::Join { left, right } => {
                let left_cost = self.estimate_cost(left);
                let right_cost = self.estimate_cost(right);
                let join_selectivity = self.estimate_join_selectivity(left, right);

                // Use hash join cost model: O(M + N) + output cost
                let base_cost = left_cost + right_cost;
                let output_cost = left_cost * right_cost * join_selectivity;
                base_cost + output_cost
            }
            Algebra::LeftJoin { left, right, .. } => {
                // Left join typically more expensive than inner join
                let left_cost = self.estimate_cost(left);
                let right_cost = self.estimate_cost(right);
                left_cost + right_cost * 1.5
            }
            Algebra::Union { left, right } => {
                self.estimate_cost(left) + self.estimate_cost(right) + 10.0
            }
            Algebra::Filter { pattern, condition } => {
                let pattern_cost = self.estimate_cost(pattern);
                let filter_selectivity = self.estimate_filter_selectivity(condition);
                pattern_cost + (pattern_cost * (1.0 - filter_selectivity) * 0.1)
            }
            Algebra::Service { pattern, .. } => {
                // Remote service calls are expensive
                self.estimate_cost(pattern) * 10.0 + 1000.0
            }
            Algebra::Distinct { pattern } => {
                // Distinct requires sorting or hashing
                let pattern_cost = self.estimate_cost(pattern);
                pattern_cost + pattern_cost.log2() * 10.0
            }
            Algebra::OrderBy { pattern, .. } => {
                // Sorting cost
                let pattern_cost = self.estimate_cost(pattern);
                pattern_cost + pattern_cost.log2() * 5.0
            }
            Algebra::Group { pattern, .. } => {
                // Grouping cost similar to sorting
                let pattern_cost = self.estimate_cost(pattern);
                pattern_cost + pattern_cost.log2() * 8.0
            }
            Algebra::Zero => 0.0,
            Algebra::Table => 1.0,
            _ => 100.0,
        }
    }

    /// Estimate BGP cost using join order optimization
    fn estimate_bgp_cost(&self, patterns: &[TriplePattern]) -> f64 {
        if patterns.len() <= 1 {
            return patterns
                .iter()
                .map(|p| self.estimate_pattern_cardinality(p) as f64)
                .sum();
        }

        // Use dynamic programming for optimal join order
        let mut dp = HashMap::new();
        self.estimate_bgp_cost_dp(patterns, &mut dp, 0, patterns.len())
    }

    /// Dynamic programming approach for BGP cost estimation
    fn estimate_bgp_cost_dp(
        &self,
        patterns: &[TriplePattern],
        dp: &mut HashMap<(usize, usize), f64>,
        start: usize,
        end: usize,
    ) -> f64 {
        if start + 1 >= end {
            return if start < patterns.len() {
                self.estimate_pattern_cardinality(&patterns[start]) as f64
            } else {
                0.0
            };
        }

        if let Some(&cost) = dp.get(&(start, end)) {
            return cost;
        }

        let mut min_cost = f64::INFINITY;

        for k in start + 1..end {
            let left_cost = self.estimate_bgp_cost_dp(patterns, dp, start, k);
            let right_cost = self.estimate_bgp_cost_dp(patterns, dp, k, end);

            // Estimate join cost between left and right parts
            let join_cost = left_cost + right_cost + (left_cost * right_cost * 0.1);
            min_cost = min_cost.min(join_cost);
        }

        dp.insert((start, end), min_cost);
        min_cost
    }

    /// Estimate filter selectivity
    fn estimate_filter_selectivity(&self, _condition: &Expression) -> f64 {
        // Simplified filter selectivity estimation
        // In practice, this would analyze the expression structure
        0.1 // Assume filters are 10% selective
    }

    /// Record execution statistics for learning
    pub fn record_execution(&mut self, record: ExecutionRecord) {
        // Update statistics based on execution before moving
        self.update_statistics_from_execution(&record);

        self.execution_history.push(record);

        // Limit history size to prevent memory growth
        if self.execution_history.len() > 10000 {
            self.execution_history.remove(0);
        }
    }

    /// Update statistics based on execution record
    fn update_statistics_from_execution(&mut self, record: &ExecutionRecord) {
        // Update pattern cardinality estimates
        self.extract_pattern_statistics(&record.algebra, record.cardinality);

        // Update variable selectivity
        let variables = record.algebra.variables();
        for var in variables {
            let current_selectivity = self.variable_selectivity.get(&var).copied().unwrap_or(0.1);
            let observed_selectivity = record.cardinality as f64 / 10000.0; // Rough estimate
            let updated_selectivity = (current_selectivity + observed_selectivity) / 2.0;
            self.variable_selectivity.insert(var, updated_selectivity);
        }
    }

    /// Extract pattern statistics from algebra expression
    fn extract_pattern_statistics(&mut self, algebra: &Algebra, cardinality: usize) {
        match algebra {
            Algebra::Bgp(patterns) => {
                for pattern in patterns {
                    let pattern_key = format!("{}", pattern);
                    self.pattern_cardinality
                        .insert(pattern_key, cardinality / patterns.len());
                }
            }
            Algebra::Join { left, right } => {
                self.extract_pattern_statistics(left, cardinality);
                self.extract_pattern_statistics(right, cardinality);
            }
            _ => {}
        }
    }

    /// Calculate query hash for caching
    pub fn calculate_query_hash(&self, algebra: &Algebra) -> u64 {
        let mut hasher = DefaultHasher::new();
        format!("{:?}", algebra).hash(&mut hasher);
        hasher.finish()
    }
}

/// Query optimizer
pub struct QueryOptimizer {
    config: OptimizerConfig,
    statistics: Statistics,
    /// Dynamic statistics updater for learning from execution
    dynamic_updater: DynamicStatisticsUpdater,
}

impl QueryOptimizer {
    pub fn new() -> Self {
        Self {
            config: OptimizerConfig::default(),
            statistics: Statistics::new(),
            dynamic_updater: DynamicStatisticsUpdater::new(),
        }
    }

    pub fn with_config(config: OptimizerConfig) -> Self {
        Self {
            config,
            statistics: Statistics::new(),
            dynamic_updater: DynamicStatisticsUpdater::new(),
        }
    }

    pub fn with_statistics(mut self, stats: Statistics) -> Self {
        self.statistics = stats;
        self
    }

    /// Optimize an algebra expression with advanced techniques
    pub fn optimize(&self, algebra: Algebra) -> Result<Algebra> {
        let mut current = algebra;
        let mut pass = 0;
        let mut optimization_history = Vec::new();

        while pass < self.config.max_passes {
            let before_cost = self.statistics.estimate_cost(&current);
            let optimized = self.apply_advanced_optimization_passes(current.clone())?;
            let after_cost = self.statistics.estimate_cost(&optimized);

            // Record optimization decision
            optimization_history.push(OptimizationDecision {
                optimization_type: OptimizationType::JoinReordering, // Simplified
                before_cost,
                after_cost,
                success: after_cost < before_cost,
            });

            // Stop if no improvement or convergence
            if after_cost >= before_cost * 0.99 {
                // Allow for small improvements
                break;
            }

            current = optimized;
            pass += 1;
        }

        Ok(current)
    }

    /// Collect statistics from triple patterns
    pub fn collect_statistics(&mut self, patterns: &[TriplePattern]) -> Result<()> {
        let mut collector = StatisticsCollector::new()
            .with_sample_rate(0.1)
            .with_histogram_buckets(100);

        collector.collect_from_patterns(patterns)?;

        // Merge collected statistics into optimizer's statistics
        let new_stats = collector.into_statistics();
        self.merge_statistics(new_stats);

        Ok(())
    }

    /// Update statistics based on query execution feedback
    pub fn update_from_execution(
        &mut self,
        algebra: Algebra,
        estimated_cardinality: usize,
        actual_cardinality: usize,
        execution_time: Duration,
    ) -> Result<()> {
        let record = QueryExecutionRecord {
            algebra,
            estimated_cardinality,
            actual_cardinality,
            execution_time,
            timestamp: Instant::now(),
        };

        self.dynamic_updater
            .update_from_execution(record, &mut self.statistics)
    }

    /// Merge new statistics into existing ones
    fn merge_statistics(&mut self, new_stats: Statistics) {
        // Merge pattern cardinalities
        for (pattern, count) in new_stats.pattern_cardinality {
            self.statistics
                .pattern_cardinality
                .entry(pattern)
                .and_modify(|c| *c = (*c + count) / 2)
                .or_insert(count);
        }

        // Merge predicate frequencies
        for (pred, freq) in new_stats.predicate_frequency {
            self.statistics
                .predicate_frequency
                .entry(pred)
                .and_modify(|f| *f += freq)
                .or_insert(freq);
        }

        // Merge variable selectivities
        for (var, sel) in new_stats.variable_selectivity {
            self.statistics
                .variable_selectivity
                .entry(var)
                .and_modify(|s| *s = (*s + sel) / 2.0)
                .or_insert(sel);
        }

        // Merge index statistics
        for index_type in new_stats.index_stats.available_indexes {
            self.statistics
                .index_stats
                .available_indexes
                .insert(index_type);
        }

        for (index_type, sel) in new_stats.index_stats.index_selectivity {
            self.statistics
                .index_stats
                .index_selectivity
                .entry(index_type)
                .and_modify(|s| *s = (*s + sel) / 2.0)
                .or_insert(sel);
        }
    }

    /// Get reference to current statistics
    pub fn get_statistics(&self) -> &Statistics {
        &self.statistics
    }

    /// Apply advanced optimization passes
    fn apply_advanced_optimization_passes(&self, algebra: Algebra) -> Result<Algebra> {
        let mut result = algebra;

        // Phase 1: Basic optimizations
        if self.config.constant_folding {
            result = self.constant_folding(result)?;
        }

        if self.config.dead_code_elimination {
            result = self.dead_code_elimination(result)?;
        }

        // Phase 2: Pushdown optimizations
        if self.config.filter_pushdown {
            result = self.optimize_filters(result)?;
        }

        if self.config.projection_pushdown {
            result = self.advanced_projection_pushdown(result)?;
        }

        // Phase 3: Join optimization
        if self.config.join_reordering {
            result = self.cost_based_join_reordering(result)?;
        }

        // Phase 4: Advanced optimizations
        if self.config.cost_based {
            result = self.index_aware_optimization(result)?;
            result = self.materialization_point_optimization(result)?;
        }

        Ok(result)
    }

    /// Advanced filter pushdown with predicate analysis
    fn advanced_filter_pushdown(&self, algebra: Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Filter { pattern, condition } => {
                match *pattern {
                    Algebra::Join { left, right } => {
                        // Analyze filter condition to determine which sub-expressions can be pushed
                        let (left_filters, right_filters, remaining_filter) =
                            self.decompose_filter_condition(&condition, &left, &right);

                        let mut optimized_left = *left;
                        let mut optimized_right = *right;

                        // Push filters to appropriate sides
                        for filter in left_filters {
                            optimized_left = Algebra::Filter {
                                pattern: Box::new(optimized_left),
                                condition: filter,
                            };
                        }

                        for filter in right_filters {
                            optimized_right = Algebra::Filter {
                                pattern: Box::new(optimized_right),
                                condition: filter,
                            };
                        }

                        // Recursively optimize pushed filters
                        optimized_left = self.advanced_filter_pushdown(optimized_left)?;
                        optimized_right = self.advanced_filter_pushdown(optimized_right)?;

                        let join = Algebra::Join {
                            left: Box::new(optimized_left),
                            right: Box::new(optimized_right),
                        };

                        // Apply remaining filter if any
                        if let Some(remaining) = remaining_filter {
                            Ok(Algebra::Filter {
                                pattern: Box::new(join),
                                condition: remaining,
                            })
                        } else {
                            Ok(join)
                        }
                    }
                    _ => {
                        // Apply to children recursively
                        let optimized_pattern = self.advanced_filter_pushdown(*pattern)?;
                        Ok(Algebra::Filter {
                            pattern: Box::new(optimized_pattern),
                            condition,
                        })
                    }
                }
            }
            _ => self.apply_to_children(algebra, |child| self.advanced_filter_pushdown(child)),
        }
    }

    /// Decompose filter condition into parts that can be pushed to different sides
    fn decompose_filter_condition(
        &self,
        condition: &Expression,
        left: &Algebra,
        right: &Algebra,
    ) -> (Vec<Expression>, Vec<Expression>, Option<Expression>) {
        let left_vars: HashSet<_> = left.variables().into_iter().collect();
        let right_vars: HashSet<_> = right.variables().into_iter().collect();

        match condition {
            Expression::Binary {
                op: BinaryOperator::And,
                left: l,
                right: r,
            } => {
                let (mut left_filters1, mut right_filters1, remaining1) =
                    self.decompose_filter_condition(l, left, right);
                let (mut left_filters2, mut right_filters2, remaining2) =
                    self.decompose_filter_condition(r, left, right);

                left_filters1.extend(left_filters2);
                right_filters1.extend(right_filters2);

                let remaining = match (remaining1, remaining2) {
                    (Some(r1), Some(r2)) => Some(Expression::Binary {
                        op: BinaryOperator::And,
                        left: Box::new(r1),
                        right: Box::new(r2),
                    }),
                    (Some(r), None) | (None, Some(r)) => Some(r),
                    (None, None) => None,
                };

                (left_filters1, right_filters1, remaining)
            }
            _ => {
                // Check if condition uses only variables from one side
                let condition_vars = self.get_expression_variables(condition);

                if condition_vars.is_subset(&left_vars) {
                    (vec![condition.clone()], vec![], None)
                } else if condition_vars.is_subset(&right_vars) {
                    (vec![], vec![condition.clone()], None)
                } else {
                    // Condition spans both sides, cannot push down
                    (vec![], vec![], Some(condition.clone()))
                }
            }
        }
    }

    /// Advanced projection pushdown with column pruning
    fn advanced_projection_pushdown(&self, algebra: Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Project { pattern, variables } => {
                let needed_vars: HashSet<_> = variables.into_iter().collect();
                let optimized_pattern =
                    self.push_projection_requirements(*pattern, &needed_vars)?;

                // Check if projection is still needed
                let pattern_vars: HashSet<_> = optimized_pattern.variables().into_iter().collect();
                if needed_vars == pattern_vars {
                    Ok(optimized_pattern) // Projection is redundant
                } else {
                    Ok(Algebra::Project {
                        pattern: Box::new(optimized_pattern),
                        variables: needed_vars.into_iter().collect(),
                    })
                }
            }
            _ => self.apply_to_children(algebra, |child| self.advanced_projection_pushdown(child)),
        }
    }

    /// Push projection requirements down the algebra tree
    fn push_projection_requirements(
        &self,
        algebra: Algebra,
        needed_vars: &HashSet<Variable>,
    ) -> Result<Algebra> {
        match algebra {
            Algebra::Join { left, right } => {
                let left_vars: HashSet<_> = left.variables().into_iter().collect();
                let right_vars: HashSet<_> = right.variables().into_iter().collect();

                let left_needed: HashSet<_> =
                    needed_vars.intersection(&left_vars).cloned().collect();
                let right_needed: HashSet<_> =
                    needed_vars.intersection(&right_vars).cloned().collect();

                let optimized_left = if left_needed.len() < left_vars.len() {
                    Algebra::Project {
                        pattern: left,
                        variables: left_needed.into_iter().collect(),
                    }
                } else {
                    *left
                };

                let optimized_right = if right_needed.len() < right_vars.len() {
                    Algebra::Project {
                        pattern: right,
                        variables: right_needed.into_iter().collect(),
                    }
                } else {
                    *right
                };

                Ok(Algebra::Join {
                    left: Box::new(self.push_projection_requirements(optimized_left, needed_vars)?),
                    right: Box::new(
                        self.push_projection_requirements(optimized_right, needed_vars)?,
                    ),
                })
            }
            _ => Ok(algebra),
        }
    }

    /// Cost-based join reordering using dynamic programming
    fn cost_based_join_reordering(&self, algebra: Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Join {
                ref left,
                ref right,
            } => {
                // Extract all joins in this chain
                let join_chain = self.extract_join_chain(&algebra);
                if join_chain.len() <= 2 {
                    // No reordering needed for binary joins
                    return Ok(algebra);
                }

                // Use dynamic programming to find optimal join order
                let optimal_order = self.find_optimal_join_order(&join_chain)?;
                Ok(self.build_join_tree_from_order(&join_chain, optimal_order))
            }
            _ => self.apply_to_children(algebra, |child| self.cost_based_join_reordering(child)),
        }
    }

    /// Extract join chain from nested join structure
    fn extract_join_chain(&self, algebra: &Algebra) -> Vec<Algebra> {
        match algebra {
            Algebra::Join { left, right } => {
                let mut chain = self.extract_join_chain(left);
                chain.extend(self.extract_join_chain(right));
                chain
            }
            _ => vec![algebra.clone()],
        }
    }

    /// Find optimal join order using dynamic programming
    fn find_optimal_join_order(&self, relations: &[Algebra]) -> Result<Vec<usize>> {
        let n = relations.len();
        if n <= 2 {
            return Ok((0..n).collect());
        }

        // Simplified join ordering - in practice would use more sophisticated algorithm
        let mut order: Vec<_> = (0..n).collect();

        // Sort by estimated cardinality (greedy approach)
        order.sort_by(|&a, &b| {
            let cost_a = self.statistics.estimate_cost(&relations[a]);
            let cost_b = self.statistics.estimate_cost(&relations[b]);
            cost_a
                .partial_cmp(&cost_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(order)
    }

    /// Build join tree from optimal order
    fn build_join_tree_from_order(&self, join_chain: &[Algebra], order: Vec<usize>) -> Algebra {
        if order.is_empty() {
            return Algebra::Zero;
        }

        if order.len() == 1 {
            // Single relation
            return join_chain[order[0]].clone();
        }

        // Build left-deep join tree for now (can be enhanced to bushy trees later)
        let mut current = join_chain[order[0]].clone();

        for &relation_idx in &order[1..] {
            let right_relation = join_chain[relation_idx].clone();
            current = Algebra::Join {
                left: Box::new(current),
                right: Box::new(right_relation),
            };
        }

        current
    }

    /// Index-aware optimization with dynamic selection and union support
    fn index_aware_optimization(&self, algebra: Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Bgp(patterns) => {
                // Use the advanced BGP optimizer
                let bgp_optimizer =
                    BGPOptimizer::new(&self.statistics, &self.statistics.index_stats);
                let optimized_bgp = bgp_optimizer.optimize_bgp(patterns)?;

                // Store index plan metadata for the executor
                self.record_index_plan(&optimized_bgp.index_plan)?;

                // Apply dynamic index selection
                let dynamically_optimized = self.apply_dynamic_index_selection(optimized_bgp)?;

                // Check if we need to transform BGP based on index opportunities
                let transformed_algebra = self.transform_bgp_for_indexes(&dynamically_optimized)?;

                Ok(transformed_algebra)
            }
            Algebra::Union { left, right } => {
                // Check for index union opportunities
                if let Some(union_optimized) = self.optimize_union_with_indexes(&left, &right)? {
                    Ok(union_optimized)
                } else {
                    // Apply recursively
                    Ok(Algebra::Union {
                        left: Box::new(self.index_aware_optimization(*left)?),
                        right: Box::new(self.index_aware_optimization(*right)?),
                    })
                }
            }
            Algebra::Filter { pattern, condition } => {
                // Check for index-aware filter optimization
                if let Some(filter_optimized) =
                    self.optimize_filter_with_indexes(&pattern, &condition)?
                {
                    Ok(filter_optimized)
                } else {
                    Ok(Algebra::Filter {
                        pattern: Box::new(self.index_aware_optimization(*pattern)?),
                        condition,
                    })
                }
            }
            _ => self.apply_to_children(algebra, |child| self.index_aware_optimization(child)),
        }
    }

    /// Apply dynamic index selection based on query patterns and statistics
    fn apply_dynamic_index_selection(
        &self,
        mut optimized_bgp: OptimizedBGP,
    ) -> Result<OptimizedBGP> {
        let mut improved_plan = optimized_bgp.index_plan.clone();

        // Analyze query workload patterns for adaptive index selection
        let query_characteristics = self.analyze_query_characteristics(&optimized_bgp.patterns);

        // Adjust index selections based on runtime statistics
        for assignment in &mut improved_plan.pattern_indexes {
            if let Some(better_index) = self.select_adaptive_index(
                &optimized_bgp.patterns[assignment.pattern_idx],
                &query_characteristics,
            ) {
                assignment.index_type = better_index.0;
                assignment.scan_cost = better_index.1;
            }
        }

        // Check for new intersection opportunities
        let new_intersections =
            self.find_additional_intersections(&optimized_bgp.patterns, &improved_plan)?;
        improved_plan.index_intersections.extend(new_intersections);

        optimized_bgp.index_plan = improved_plan;
        Ok(optimized_bgp)
    }

    /// Analyze query characteristics for adaptive optimization
    fn analyze_query_characteristics(&self, patterns: &[TriplePattern]) -> QueryCharacteristics {
        let variable_count = patterns
            .iter()
            .flat_map(|p| p.variables())
            .collect::<HashSet<_>>()
            .len();

        let has_object_variables = patterns
            .iter()
            .any(|p| matches!(p.object, Term::Variable(_)));

        let has_literal_constraints = patterns
            .iter()
            .any(|p| matches!(p.object, Term::Literal(_)));

        let join_complexity = self.calculate_join_complexity(patterns);

        QueryCharacteristics {
            variable_count,
            has_object_variables,
            has_literal_constraints,
            join_complexity,
            pattern_count: patterns.len(),
        }
    }

    /// Select adaptive index based on query characteristics
    fn select_adaptive_index(
        &self,
        pattern: &TriplePattern,
        characteristics: &QueryCharacteristics,
    ) -> Option<(IndexType, f64)> {
        // Use different heuristics based on query characteristics
        match characteristics.join_complexity {
            JoinComplexity::Simple => {
                // For simple queries, prefer direct indexes
                if !matches!(pattern.predicate, Term::Variable(_)) {
                    self.select_single_optimal_index(pattern)
                } else {
                    None
                }
            }
            JoinComplexity::Moderate => {
                // For moderate complexity, consider intersections
                self.select_index_intersection(pattern)
                    .or_else(|| self.select_single_optimal_index(pattern))
            }
            JoinComplexity::Complex => {
                // For complex queries, prioritize most selective indexes
                self.select_most_selective_index(pattern)
            }
        }
    }

    /// Find additional index intersection opportunities
    fn find_additional_intersections(
        &self,
        patterns: &[TriplePattern],
        current_plan: &crate::bgp_optimizer::IndexUsagePlan,
    ) -> Result<Vec<crate::bgp_optimizer::IndexIntersection>> {
        let mut intersections = Vec::new();

        // Look for patterns that share variables and could benefit from intersections
        for i in 0..patterns.len() {
            for j in i + 1..patterns.len() {
                let shared_vars = self.get_shared_variables_in_patterns(&patterns[i], &patterns[j]);
                if !shared_vars.is_empty() {
                    if let Some(intersection) =
                        self.create_intersection_opportunity(i, j, &shared_vars)
                    {
                        intersections.push(intersection);
                    }
                }
            }
        }

        Ok(intersections)
    }

    /// Optimize union operations with index unions
    fn optimize_union_with_indexes(
        &self,
        left: &Algebra,
        right: &Algebra,
    ) -> Result<Option<Algebra>> {
        // Check if both sides are BGPs that could benefit from index unions
        if let (Algebra::Bgp(left_patterns), Algebra::Bgp(right_patterns)) = (left, right) {
            // Look for opportunities to use index unions instead of separate scans
            if self.can_use_index_union(left_patterns, right_patterns) {
                let union_plan = self.create_index_union_plan(left_patterns, right_patterns)?;
                return Ok(Some(self.execute_index_union_plan(union_plan)?));
            }
        }
        Ok(None)
    }

    /// Optimize filters with index-aware strategies
    fn optimize_filter_with_indexes(
        &self,
        pattern: &Algebra,
        condition: &Expression,
    ) -> Result<Option<Algebra>> {
        // Check if the filter condition can be pushed into index scans
        if let Algebra::Bgp(patterns) = pattern {
            if let Some(index_filter_plan) = self.create_index_filter_plan(patterns, condition)? {
                return Ok(Some(self.execute_index_filter_plan(index_filter_plan)?));
            }
        }
        Ok(None)
    }

    /// Record index plan for later execution
    fn record_index_plan(&self, index_plan: &crate::bgp_optimizer::IndexUsagePlan) -> Result<()> {
        // Store index plan metadata for the query executor
        // This would typically be stored in a query context or execution plan

        // Log index usage decisions for monitoring and debugging
        for assignment in &index_plan.pattern_indexes {
            tracing::debug!(
                "Index assignment: pattern {} -> {:?} (cost: {:.2})",
                assignment.pattern_idx,
                assignment.index_type,
                assignment.scan_cost
            );
        }

        for join_opportunity in &index_plan.join_indexes {
            tracing::debug!(
                "Join index opportunity: patterns {} and {} on variable {} (speedup: {:.2}x)",
                join_opportunity.left_pattern_idx,
                join_opportunity.right_pattern_idx,
                join_opportunity.join_var,
                join_opportunity.speedup_factor
            );
        }

        Ok(())
    }

    /// Transform BGP based on index usage plan
    fn transform_bgp_for_indexes(
        &self,
        optimized_bgp: &crate::bgp_optimizer::OptimizedBGP,
    ) -> Result<Algebra> {
        let patterns = &optimized_bgp.patterns;
        let index_plan = &optimized_bgp.index_plan;

        // If there are beneficial join index opportunities, restructure the BGP
        if !index_plan.join_indexes.is_empty() {
            // Find the most beneficial join opportunity
            if let Some(best_join) = index_plan
                .join_indexes
                .iter()
                .max_by(|a, b| a.speedup_factor.partial_cmp(&b.speedup_factor).unwrap())
            {
                if best_join.speedup_factor > 2.0 {
                    return self.create_index_optimized_join(patterns, best_join);
                }
            }
        }

        // For now, return optimized BGP with patterns reordered for best index usage
        Ok(Algebra::Bgp(patterns.clone()))
    }

    /// Create an index-optimized join structure
    fn create_index_optimized_join(
        &self,
        patterns: &[TriplePattern],
        join_opportunity: &crate::bgp_optimizer::JoinIndexOpportunity,
    ) -> Result<Algebra> {
        let left_idx = join_opportunity.left_pattern_idx;
        let right_idx = join_opportunity.right_pattern_idx;

        if left_idx >= patterns.len() || right_idx >= patterns.len() {
            return Ok(Algebra::Bgp(patterns.to_vec()));
        }

        // Create separate BGPs for the join
        let left_bgp = Algebra::Bgp(vec![patterns[left_idx].clone()]);
        let right_bgp = Algebra::Bgp(vec![patterns[right_idx].clone()]);

        // Create join
        let join_algebra = Algebra::Join {
            left: Box::new(left_bgp),
            right: Box::new(right_bgp),
        };

        // Add remaining patterns
        let mut remaining_patterns = Vec::new();
        for (i, pattern) in patterns.iter().enumerate() {
            if i != left_idx && i != right_idx {
                remaining_patterns.push(pattern.clone());
            }
        }

        if remaining_patterns.is_empty() {
            Ok(join_algebra)
        } else {
            // Join with remaining patterns
            Ok(Algebra::Join {
                left: Box::new(join_algebra),
                right: Box::new(Algebra::Bgp(remaining_patterns)),
            })
        }
    }

    /// Estimate pattern cost considering available indexes
    fn estimate_pattern_cost_with_indexes(&self, pattern: &TriplePattern) -> f64 {
        let base_cost = self.statistics.estimate_pattern_cardinality(pattern) as f64;

        // Check if we have suitable indexes
        let has_subject_index = matches!(pattern.subject, Term::Iri(_) | Term::Literal(_));
        let has_predicate_index = matches!(pattern.predicate, Term::Iri(_));
        let has_object_index = matches!(pattern.object, Term::Iri(_) | Term::Literal(_));

        let index_factor = match (has_subject_index, has_predicate_index, has_object_index) {
            (true, true, true) => 0.1,  // Triple index - very fast
            (true, true, false) => 0.3, // Subject-predicate index
            (false, true, true) => 0.4, // Predicate-object index
            (true, false, true) => 0.5, // Subject-object index
            (_, true, _) => 0.7,        // Predicate index
            _ => 1.0,                   // No suitable index
        };

        base_cost * index_factor
    }

    /// Enhanced index selection considering multiple factors and intersections
    fn select_optimal_index(&self, pattern: &TriplePattern) -> Option<IndexType> {
        // First check for single-index optimization
        let single_best = self.select_single_optimal_index(pattern);

        // Then check for index intersection opportunities
        let intersection_best = self.select_index_intersection(pattern);

        // Compare costs and return the best option
        match (single_best.as_ref(), intersection_best.as_ref()) {
            (Some((single_idx, single_cost)), Some((intersect_idx, intersect_cost))) => {
                if intersect_cost < single_cost {
                    intersection_best.map(|(idx, _)| idx)
                } else {
                    single_best.map(|(idx, _)| idx)
                }
            }
            (Some(_), None) => single_best.map(|(idx, _)| idx),
            (None, Some(_)) => intersection_best.map(|(idx, _)| idx),
            (None, None) => None,
        }
    }

    /// Select optimal single index for a pattern
    fn select_single_optimal_index(&self, pattern: &TriplePattern) -> Option<(IndexType, f64)> {
        let mut best_index = None;
        let mut best_cost = f64::INFINITY;

        for index_type in &self.statistics.index_stats.available_indexes {
            if let Some(selectivity) = self
                .statistics
                .index_stats
                .index_selectivity
                .get(index_type)
            {
                let access_cost = self
                    .statistics
                    .index_stats
                    .index_access_cost
                    .get(index_type)
                    .copied()
                    .unwrap_or(1.0);

                // Calculate total cost: access_cost * selectivity * base_cardinality
                let base_cardinality = self.statistics.estimate_pattern_cardinality(pattern) as f64;
                let total_cost = access_cost * selectivity * base_cardinality;

                if total_cost < best_cost && self.index_supports_pattern(index_type, pattern) {
                    best_cost = total_cost;
                    best_index = Some((index_type.clone(), total_cost));
                }
            }
        }

        best_index
    }

    /// Select optimal index intersection for a pattern
    fn select_index_intersection(&self, pattern: &TriplePattern) -> Option<(IndexType, f64)> {
        let available_indices: Vec<_> = self
            .statistics
            .index_stats
            .available_indexes
            .iter()
            .filter(|idx| self.index_supports_pattern(idx, pattern))
            .collect();

        if available_indices.len() < 2 {
            return None;
        }

        // Find best combination of indices for intersection
        let mut best_intersection = None;
        let mut best_cost = f64::INFINITY;

        // Check pairwise intersections
        for i in 0..available_indices.len() {
            for j in i + 1..available_indices.len() {
                let idx1 = available_indices[i];
                let idx2 = available_indices[j];

                if let Some(intersection_cost) =
                    self.estimate_intersection_cost(pattern, idx1, idx2)
                {
                    if intersection_cost < best_cost {
                        best_cost = intersection_cost;
                        // Create a virtual intersection index
                        best_intersection = Some((
                            IndexType::Custom(format!(
                                "intersection_{}_{}",
                                self.index_name(idx1),
                                self.index_name(idx2)
                            )),
                            intersection_cost,
                        ));
                    }
                }
            }
        }

        best_intersection
    }

    /// Estimate cost of index intersection
    fn estimate_intersection_cost(
        &self,
        pattern: &TriplePattern,
        idx1: &IndexType,
        idx2: &IndexType,
    ) -> Option<f64> {
        let sel1 = self
            .statistics
            .index_stats
            .index_selectivity
            .get(idx1)
            .copied()
            .unwrap_or(1.0);
        let sel2 = self
            .statistics
            .index_stats
            .index_selectivity
            .get(idx2)
            .copied()
            .unwrap_or(1.0);
        let cost1 = self
            .statistics
            .index_stats
            .index_access_cost
            .get(idx1)
            .copied()
            .unwrap_or(1.0);
        let cost2 = self
            .statistics
            .index_stats
            .index_access_cost
            .get(idx2)
            .copied()
            .unwrap_or(1.0);

        // Intersection selectivity is typically the product of individual selectivities
        let intersection_selectivity = sel1 * sel2;

        // Cost includes accessing both indices plus intersection computation
        let base_cardinality = self.statistics.estimate_pattern_cardinality(pattern) as f64;
        let intersection_cost = cost1 + cost2 + (base_cardinality * intersection_selectivity * 0.1);

        // Only beneficial if intersection selectivity is significantly better
        if intersection_selectivity < sel1.min(sel2) * 0.8 {
            Some(intersection_cost)
        } else {
            None
        }
    }

    /// Get a readable name for an index type
    fn index_name(&self, index_type: &IndexType) -> String {
        match index_type {
            IndexType::SPO => "SPO".to_string(),
            IndexType::PSO => "PSO".to_string(),
            IndexType::OSP => "OSP".to_string(),
            IndexType::Hash => "Hash".to_string(),
            IndexType::BTree => "BTree".to_string(),
            IndexType::Bitmap => "Bitmap".to_string(),
            IndexType::Custom(name) => name.clone(),
            _ => format!("{:?}", index_type),
        }
    }

    /// Check if an index supports a given pattern
    fn index_supports_pattern(&self, index_type: &IndexType, pattern: &TriplePattern) -> bool {
        match index_type {
            IndexType::SPO => true, // Supports all patterns
            IndexType::PSO => !matches!(pattern.predicate, Term::Variable(_)),
            IndexType::OSP => !matches!(pattern.object, Term::Variable(_)),
            IndexType::OPS => !matches!(pattern.object, Term::Variable(_)),
            IndexType::SOP => !matches!(pattern.subject, Term::Variable(_)),
            IndexType::POS => !matches!(pattern.predicate, Term::Variable(_)),
            IndexType::Bloom => {
                // Bloom filters are good for existence checks with concrete values
                !matches!(pattern.subject, Term::Variable(_))
                    || !matches!(pattern.predicate, Term::Variable(_))
                    || !matches!(pattern.object, Term::Variable(_))
            }
            IndexType::Hash => {
                // Hash indexes are good for equality lookups
                !matches!(pattern.subject, Term::Variable(_))
                    && !matches!(pattern.predicate, Term::Variable(_))
                    && !matches!(pattern.object, Term::Variable(_))
            }
            IndexType::BTree => true, // B-tree supports range queries and existence
            IndexType::Bitmap => {
                // Bitmap indexes are efficient for low-cardinality predicates
                !matches!(pattern.predicate, Term::Variable(_))
            }
            // Basic index types
            IndexType::SubjectPredicate => {
                !matches!(pattern.subject, Term::Variable(_))
                    && !matches!(pattern.predicate, Term::Variable(_))
            }
            IndexType::PredicateObject => {
                !matches!(pattern.predicate, Term::Variable(_))
                    && !matches!(pattern.object, Term::Variable(_))
            }
            IndexType::SubjectObject => {
                !matches!(pattern.subject, Term::Variable(_))
                    && !matches!(pattern.object, Term::Variable(_))
            }
            IndexType::FullText => {
                // Full-text search typically works with literals
                matches!(pattern.object, Term::Literal(_))
            }
            IndexType::Spatial => {
                // Spatial indexes work with geometric data
                matches!(pattern.object, Term::Literal(_))
            }
            IndexType::Temporal => {
                // Temporal indexes work with date/time data
                matches!(pattern.object, Term::Literal(_))
            }
            // Advanced index types
            IndexType::BTreeIndex(pos) => match pos {
                IndexPosition::Subject => !matches!(pattern.subject, Term::Variable(_)),
                IndexPosition::Predicate => !matches!(pattern.predicate, Term::Variable(_)),
                IndexPosition::Object => !matches!(pattern.object, Term::Variable(_)),
                IndexPosition::SubjectPredicate => {
                    !matches!(pattern.subject, Term::Variable(_))
                        && !matches!(pattern.predicate, Term::Variable(_))
                }
                IndexPosition::PredicateObject => {
                    !matches!(pattern.predicate, Term::Variable(_))
                        && !matches!(pattern.object, Term::Variable(_))
                }
                IndexPosition::SubjectObject => {
                    !matches!(pattern.subject, Term::Variable(_))
                        && !matches!(pattern.object, Term::Variable(_))
                }
                IndexPosition::FullTriple => true,
            },
            IndexType::HashIndex(pos) => match pos {
                IndexPosition::Subject => !matches!(pattern.subject, Term::Variable(_)),
                IndexPosition::Predicate => !matches!(pattern.predicate, Term::Variable(_)),
                IndexPosition::Object => !matches!(pattern.object, Term::Variable(_)),
                IndexPosition::SubjectPredicate => {
                    !matches!(pattern.subject, Term::Variable(_))
                        && !matches!(pattern.predicate, Term::Variable(_))
                }
                IndexPosition::PredicateObject => {
                    !matches!(pattern.predicate, Term::Variable(_))
                        && !matches!(pattern.object, Term::Variable(_))
                }
                IndexPosition::SubjectObject => {
                    !matches!(pattern.subject, Term::Variable(_))
                        && !matches!(pattern.object, Term::Variable(_))
                }
                IndexPosition::FullTriple => {
                    !matches!(pattern.subject, Term::Variable(_))
                        && !matches!(pattern.predicate, Term::Variable(_))
                        && !matches!(pattern.object, Term::Variable(_))
                }
            },
            IndexType::BitmapIndex(pos) => match pos {
                IndexPosition::Subject => !matches!(pattern.subject, Term::Variable(_)),
                IndexPosition::Predicate => !matches!(pattern.predicate, Term::Variable(_)),
                IndexPosition::Object => !matches!(pattern.object, Term::Variable(_)),
                _ => false, // Bitmap indexes typically work best with single positions
            },
            IndexType::BloomFilter(pos) => match pos {
                IndexPosition::Subject => !matches!(pattern.subject, Term::Variable(_)),
                IndexPosition::Predicate => !matches!(pattern.predicate, Term::Variable(_)),
                IndexPosition::Object => !matches!(pattern.object, Term::Variable(_)),
                _ => true, // Bloom filters can work with compound keys
            },
            IndexType::SpatialRTree => {
                // Spatial R-tree for geometric queries
                matches!(pattern.object, Term::Literal(_))
            }
            IndexType::TemporalBTree => {
                // Temporal B-tree for time-based queries
                matches!(pattern.object, Term::Literal(_))
            }
            IndexType::MultiColumnBTree(_positions) => {
                // Multi-column B-tree supports complex patterns
                true
            }
            IndexType::Custom(_name) => {
                // Custom indexes may have specialized support
                true
            }
        }
    }

    /// Optimize for streaming execution
    fn optimize_for_streaming(&self, algebra: Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Join { left, right } => {
                // Reorder joins to minimize intermediate result size
                let left_card = self.estimate_algebra_cardinality(&left);
                let right_card = self.estimate_algebra_cardinality(&right);

                // Put smaller relation first for hash joins
                if left_card > right_card {
                    Ok(Algebra::Join {
                        left: Box::new(self.optimize_for_streaming(*right)?),
                        right: Box::new(self.optimize_for_streaming(*left)?),
                    })
                } else {
                    Ok(Algebra::Join {
                        left: Box::new(self.optimize_for_streaming(*left)?),
                        right: Box::new(self.optimize_for_streaming(*right)?),
                    })
                }
            }
            Algebra::Union { left, right } => {
                // For unions, we can potentially stream results
                Ok(Algebra::Union {
                    left: Box::new(self.optimize_for_streaming(*left)?),
                    right: Box::new(self.optimize_for_streaming(*right)?),
                })
            }
            _ => self.apply_to_children(algebra, |child| self.optimize_for_streaming(child)),
        }
    }

    /// Estimate cardinality of an algebra expression
    fn estimate_algebra_cardinality(&self, algebra: &Algebra) -> f64 {
        match algebra {
            Algebra::Bgp(patterns) => patterns
                .iter()
                .map(|p| self.statistics.estimate_pattern_cardinality(p) as f64)
                .product(),
            Algebra::Join { left, right } => {
                let left_card = self.estimate_algebra_cardinality(left);
                let right_card = self.estimate_algebra_cardinality(right);
                // Simplified join cardinality estimation
                left_card * right_card * 0.1 // Assume 10% selectivity
            }
            Algebra::Union { left, right } => {
                self.estimate_algebra_cardinality(left) + self.estimate_algebra_cardinality(right)
            }
            Algebra::Filter { pattern, .. } => {
                // Filters typically reduce cardinality
                self.estimate_algebra_cardinality(pattern) * 0.3
            }
            _ => 1000.0, // Default estimate
        }
    }

    /// Materialization point optimization
    fn materialization_point_optimization(&self, algebra: Algebra) -> Result<Algebra> {
        // Determine optimal points to materialize intermediate results
        match algebra {
            Algebra::Join { left, right } => {
                let left_cost = self.statistics.estimate_cost(&left);
                let right_cost = self.statistics.estimate_cost(&right);

                // If one side is much more expensive, consider materializing it
                if left_cost > right_cost * 5.0 {
                    // Consider materializing left side
                    // In practice, this would add materialization algebra nodes
                }

                Ok(Algebra::Join {
                    left: Box::new(self.materialization_point_optimization(*left)?),
                    right: Box::new(self.materialization_point_optimization(*right)?),
                })
            }
            _ => self.apply_to_children(algebra, |child| {
                self.materialization_point_optimization(child)
            }),
        }
    }

    fn apply_optimization_passes(&self, algebra: Algebra) -> Result<Algebra> {
        let mut result = algebra;

        // Apply rule-based optimizations
        if self.config.constant_folding {
            result = self.constant_folding(result)?;
        }

        if self.config.dead_code_elimination {
            result = self.dead_code_elimination(result)?;
        }

        if self.config.filter_pushdown {
            result = self.filter_pushdown(result)?;
        }

        if self.config.projection_pushdown {
            result = self.projection_pushdown(result)?;
        }

        if self.config.join_reordering {
            result = self.join_reordering(result)?;
        }

        Ok(result)
    }

    /// Constant folding optimization
    fn constant_folding(&self, algebra: Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Filter { pattern, condition } => {
                let optimized_pattern = self.constant_folding(*pattern)?;
                let optimized_condition = self.fold_expression(condition)?;

                // Check if condition is always true/false after folding
                if self.is_always_true(&optimized_condition) {
                    return Ok(optimized_pattern);
                }

                if self.is_always_false(&optimized_condition) {
                    return Ok(Algebra::Zero);
                }

                Ok(Algebra::Filter {
                    pattern: Box::new(optimized_pattern),
                    condition: optimized_condition,
                })
            }
            Algebra::Join { left, right } => {
                let optimized_left = self.constant_folding(*left)?;
                let optimized_right = self.constant_folding(*right)?;

                // Join with Zero is Zero
                if matches!(optimized_left, Algebra::Zero)
                    || matches!(optimized_right, Algebra::Zero)
                {
                    return Ok(Algebra::Zero);
                }

                // Join with Table is the other pattern
                if matches!(optimized_left, Algebra::Table) {
                    return Ok(optimized_right);
                }
                if matches!(optimized_right, Algebra::Table) {
                    return Ok(optimized_left);
                }

                Ok(Algebra::Join {
                    left: Box::new(optimized_left),
                    right: Box::new(optimized_right),
                })
            }
            Algebra::Union { left, right } => {
                let optimized_left = self.constant_folding(*left)?;
                let optimized_right = self.constant_folding(*right)?;

                // Union with Zero is the other pattern
                if matches!(optimized_left, Algebra::Zero) {
                    return Ok(optimized_right);
                }
                if matches!(optimized_right, Algebra::Zero) {
                    return Ok(optimized_left);
                }

                Ok(Algebra::Union {
                    left: Box::new(optimized_left),
                    right: Box::new(optimized_right),
                })
            }
            _ => self.apply_to_children(algebra, |child| self.constant_folding(child)),
        }
    }

    /// Dead code elimination
    fn dead_code_elimination(&self, algebra: Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Project { pattern, variables } => {
                let optimized_pattern = self.dead_code_elimination(*pattern)?;

                // Remove projection if no variables are projected
                if variables.is_empty() {
                    return Ok(Algebra::Table);
                }

                // Remove projection if all variables from pattern are projected
                let pattern_vars = optimized_pattern.variables();
                if variables.len() >= pattern_vars.len()
                    && pattern_vars.iter().all(|v| variables.contains(v))
                {
                    return Ok(optimized_pattern);
                }

                Ok(Algebra::Project {
                    pattern: Box::new(optimized_pattern),
                    variables,
                })
            }
            Algebra::Slice {
                pattern,
                offset,
                limit,
            } => {
                let optimized_pattern = self.dead_code_elimination(*pattern)?;

                // If limit is 0, return zero
                if limit == Some(0) {
                    return Ok(Algebra::Zero);
                }

                Ok(Algebra::Slice {
                    pattern: Box::new(optimized_pattern),
                    offset,
                    limit,
                })
            }
            _ => self.apply_to_children(algebra, |child| self.dead_code_elimination(child)),
        }
    }

    /// Filter pushdown optimization
    fn filter_pushdown(&self, algebra: Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Filter { pattern, condition } => {
                match *pattern {
                    Algebra::Join { left, right } => {
                        let condition_vars = self.get_expression_variables(&condition);
                        let left_vars: HashSet<_> = left.variables().into_iter().collect();
                        let right_vars: HashSet<_> = right.variables().into_iter().collect();

                        // If filter only uses left variables, push to left
                        if condition_vars.is_subset(&left_vars) {
                            let filtered_left = Algebra::Filter {
                                pattern: left,
                                condition: condition.clone(),
                            };
                            return Ok(Algebra::Join {
                                left: Box::new(self.filter_pushdown(filtered_left)?),
                                right: Box::new(self.filter_pushdown(*right)?),
                            });
                        }

                        // If filter only uses right variables, push to right
                        if condition_vars.is_subset(&right_vars) {
                            let filtered_right = Algebra::Filter {
                                pattern: right,
                                condition: condition.clone(),
                            };
                            return Ok(Algebra::Join {
                                left: Box::new(self.filter_pushdown(*left)?),
                                right: Box::new(self.filter_pushdown(filtered_right)?),
                            });
                        }

                        // Can't push down, keep as is
                        Ok(Algebra::Filter {
                            pattern: Box::new(Algebra::Join {
                                left: Box::new(self.filter_pushdown(*left)?),
                                right: Box::new(self.filter_pushdown(*right)?),
                            }),
                            condition,
                        })
                    }
                    Algebra::Union { left, right } => {
                        // Push filter into both sides of union
                        let filtered_left = Algebra::Filter {
                            pattern: left,
                            condition: condition.clone(),
                        };
                        let filtered_right = Algebra::Filter {
                            pattern: right,
                            condition,
                        };
                        Ok(Algebra::Union {
                            left: Box::new(self.filter_pushdown(filtered_left)?),
                            right: Box::new(self.filter_pushdown(filtered_right)?),
                        })
                    }
                    _ => {
                        // Can't push down further
                        Ok(Algebra::Filter {
                            pattern: Box::new(self.filter_pushdown(*pattern)?),
                            condition,
                        })
                    }
                }
            }
            _ => self.apply_to_children(algebra, |child| self.filter_pushdown(child)),
        }
    }

    /// Projection pushdown optimization
    fn projection_pushdown(&self, algebra: Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Project { pattern, variables } => {
                match *pattern {
                    Algebra::Join { left, right } => {
                        let left_vars: HashSet<_> = left.variables().into_iter().collect();
                        let right_vars: HashSet<_> = right.variables().into_iter().collect();
                        let needed_vars: HashSet<_> = variables.into_iter().collect();

                        let left_needed: Vec<_> =
                            needed_vars.intersection(&left_vars).cloned().collect();
                        let right_needed: Vec<_> =
                            needed_vars.intersection(&right_vars).cloned().collect();

                        let projected_left = if left_needed.len() < left_vars.len() {
                            Algebra::Project {
                                pattern: left,
                                variables: left_needed,
                            }
                        } else {
                            *left
                        };

                        let projected_right = if right_needed.len() < right_vars.len() {
                            Algebra::Project {
                                pattern: right,
                                variables: right_needed,
                            }
                        } else {
                            *right
                        };

                        Ok(Algebra::Join {
                            left: Box::new(self.projection_pushdown(projected_left)?),
                            right: Box::new(self.projection_pushdown(projected_right)?),
                        })
                    }
                    _ => {
                        let needed_vars: HashSet<_> = variables.into_iter().collect();
                        let pattern_vars: HashSet<_> = pattern.variables().into_iter().collect();

                        // Only keep variables that are actually used
                        let filtered_vars: Vec<_> =
                            needed_vars.intersection(&pattern_vars).cloned().collect();

                        if filtered_vars.len() < pattern_vars.len() {
                            Ok(Algebra::Project {
                                pattern: Box::new(self.projection_pushdown(*pattern)?),
                                variables: filtered_vars,
                            })
                        } else {
                            Ok(self.projection_pushdown(*pattern)?)
                        }
                    }
                }
            }
            _ => self.apply_to_children(algebra, |child| self.projection_pushdown(child)),
        }
    }

    /// Join reordering optimization
    fn join_reordering(&self, algebra: Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Join { left, right } => {
                let left_cost = self.statistics.estimate_cost(&left);
                let right_cost = self.statistics.estimate_cost(&right);

                // Reorder if right is cheaper than left
                if right_cost < left_cost {
                    Ok(Algebra::Join {
                        left: Box::new(self.join_reordering(*right)?),
                        right: Box::new(self.join_reordering(*left)?),
                    })
                } else {
                    Ok(Algebra::Join {
                        left: Box::new(self.join_reordering(*left)?),
                        right: Box::new(self.join_reordering(*right)?),
                    })
                }
            }
            _ => self.apply_to_children(algebra, |child| self.join_reordering(child)),
        }
    }

    /// Apply optimization function to all children
    fn apply_to_children<F>(&self, algebra: Algebra, optimize_fn: F) -> Result<Algebra>
    where
        F: Fn(Algebra) -> Result<Algebra>,
    {
        match algebra {
            Algebra::Join { left, right } => Ok(Algebra::Join {
                left: Box::new(optimize_fn(*left)?),
                right: Box::new(optimize_fn(*right)?),
            }),
            Algebra::Union { left, right } => Ok(Algebra::Union {
                left: Box::new(optimize_fn(*left)?),
                right: Box::new(optimize_fn(*right)?),
            }),
            Algebra::Filter { pattern, condition } => Ok(Algebra::Filter {
                pattern: Box::new(optimize_fn(*pattern)?),
                condition,
            }),
            Algebra::Project { pattern, variables } => Ok(Algebra::Project {
                pattern: Box::new(optimize_fn(*pattern)?),
                variables,
            }),
            Algebra::Distinct { pattern } => Ok(Algebra::Distinct {
                pattern: Box::new(optimize_fn(*pattern)?),
            }),
            _ => Ok(algebra), // No children to optimize
        }
    }

    /// Fold constant expressions
    fn fold_expression(&self, expr: Expression) -> Result<Expression> {
        match expr {
            Expression::Binary { op, left, right } => {
                let folded_left = self.fold_expression(*left)?;
                let folded_right = self.fold_expression(*right)?;

                // Try to evaluate constant expressions
                if let (Expression::Literal(l), Expression::Literal(r)) =
                    (&folded_left, &folded_right)
                {
                    if let Ok(result) = self.evaluate_constant_binary(&op, l, r) {
                        return Ok(Expression::Literal(result));
                    }
                }

                Ok(Expression::Binary {
                    op,
                    left: Box::new(folded_left),
                    right: Box::new(folded_right),
                })
            }
            Expression::Unary { op, expr } => {
                let folded_expr = self.fold_expression(*expr)?;

                if let Expression::Literal(lit) = &folded_expr {
                    if let Ok(result) = self.evaluate_constant_unary(&op, lit) {
                        return Ok(Expression::Literal(result));
                    }
                }

                Ok(Expression::Unary {
                    op,
                    expr: Box::new(folded_expr),
                })
            }
            _ => Ok(expr),
        }
    }

    /// Check if expression is always true
    fn is_always_true(&self, expr: &Expression) -> bool {
        match expr {
            Expression::Literal(lit) => lit.value == "true",
            Expression::Binary {
                op: BinaryOperator::Equal,
                left,
                right,
            } => {
                // Check if it's comparing two identical literals
                if let (Expression::Literal(l), Expression::Literal(r)) =
                    (left.as_ref(), right.as_ref())
                {
                    l.value == r.value
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Check if expression is always false
    fn is_always_false(&self, expr: &Expression) -> bool {
        match expr {
            Expression::Literal(lit) => lit.value == "false",
            _ => false,
        }
    }

    /// Get variables used in expression
    fn get_expression_variables(&self, expr: &Expression) -> HashSet<Variable> {
        let mut vars = HashSet::new();
        self.collect_expression_variables(expr, &mut vars);
        vars
    }

    fn collect_expression_variables(&self, expr: &Expression, vars: &mut HashSet<Variable>) {
        match expr {
            Expression::Variable(var) => {
                vars.insert(var.clone());
            }
            Expression::Binary { left, right, .. } => {
                self.collect_expression_variables(left, vars);
                self.collect_expression_variables(right, vars);
            }
            Expression::Unary { expr, .. } => {
                self.collect_expression_variables(expr, vars);
            }
            Expression::Function { args, .. } => {
                for arg in args {
                    self.collect_expression_variables(arg, vars);
                }
            }
            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                self.collect_expression_variables(condition, vars);
                self.collect_expression_variables(then_expr, vars);
                self.collect_expression_variables(else_expr, vars);
            }
            Expression::Bound(var) => {
                vars.insert(var.clone());
            }
            Expression::Exists(algebra) | Expression::NotExists(algebra) => {
                for var in algebra.variables() {
                    vars.insert(var);
                }
            }
            _ => {}
        }
    }

    /// Evaluate constant binary expressions
    fn evaluate_constant_binary(
        &self,
        op: &BinaryOperator,
        left: &crate::algebra::Literal,
        right: &crate::algebra::Literal,
    ) -> Result<crate::algebra::Literal> {
        match op {
            BinaryOperator::Add => {
                if let (Ok(l), Ok(r)) = (left.value.parse::<f64>(), right.value.parse::<f64>()) {
                    Ok(crate::algebra::Literal {
                        value: (l + r).to_string(),
                        language: None,
                        datatype: None,
                    })
                } else {
                    Err(anyhow!("Cannot add non-numeric literals"))
                }
            }
            BinaryOperator::Equal => {
                let result = left.value == right.value;
                Ok(crate::algebra::Literal {
                    value: result.to_string(),
                    language: None,
                    datatype: None,
                })
            }
            _ => Err(anyhow!(
                "Binary operation not supported for constant folding"
            )),
        }
    }

    /// Evaluate constant unary expressions
    fn evaluate_constant_unary(
        &self,
        op: &UnaryOperator,
        expr: &crate::algebra::Literal,
    ) -> Result<crate::algebra::Literal> {
        match op {
            UnaryOperator::Not => {
                let val = expr.value != "false" && !expr.value.is_empty();
                Ok(crate::algebra::Literal {
                    value: (!val).to_string(),
                    language: None,
                    datatype: None,
                })
            }
            UnaryOperator::IsLiteral => Ok(crate::algebra::Literal {
                value: "true".to_string(),
                language: None,
                datatype: None,
            }),
            _ => Err(anyhow!(
                "Unary operation not supported for constant folding"
            )),
        }
    }

    /// Decompose complex filters into conjunctive normal form (CNF)
    fn decompose_filter_cnf(&self, expr: &Expression) -> Vec<Expression> {
        match expr {
            Expression::Binary {
                op: BinaryOperator::And,
                left,
                right,
            } => {
                let mut filters = self.decompose_filter_cnf(left);
                filters.extend(self.decompose_filter_cnf(right));
                filters
            }
            _ => vec![expr.clone()],
        }
    }

    /// Check if a filter expression uses only indexed properties
    fn is_indexed_filter(&self, expr: &Expression) -> bool {
        match expr {
            Expression::Binary {
                op: BinaryOperator::Equal,
                left,
                right,
            } => {
                // Check if comparing a variable with a constant
                matches!(
                    (left.as_ref(), right.as_ref()),
                    (Expression::Variable(_), Expression::Literal(_))
                        | (Expression::Literal(_), Expression::Variable(_))
                        | (Expression::Variable(_), Expression::Iri(_))
                        | (Expression::Iri(_), Expression::Variable(_))
                )
            }
            Expression::Binary {
                op: BinaryOperator::And,
                left,
                right,
            } => self.is_indexed_filter(left) || self.is_indexed_filter(right),
            _ => false,
        }
    }

    /// Enhanced filter optimization with reordering and selectivity estimation
    fn optimize_filters(&self, algebra: Algebra) -> Result<Algebra> {
        // First apply filter pushdown
        let pushed = self.filter_pushdown(algebra)?;
        // Then reorder filters based on selectivity
        self.reorder_filters(pushed)
    }

    /// Reorder filters based on estimated selectivity
    fn reorder_filters(&self, algebra: Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Filter { pattern, condition } => {
                // Check if pattern is also a filter to enable reordering
                if let Algebra::Filter {
                    pattern: inner_pattern,
                    condition: inner_condition,
                } = *pattern
                {
                    // Estimate selectivity of both conditions
                    let outer_selectivity = self.estimate_filter_selectivity(&condition);
                    let inner_selectivity = self.estimate_filter_selectivity(&inner_condition);

                    // If outer filter is more selective, keep current order
                    if outer_selectivity <= inner_selectivity {
                        Ok(Algebra::Filter {
                            pattern: Box::new(Algebra::Filter {
                                pattern: inner_pattern,
                                condition: inner_condition,
                            }),
                            condition,
                        })
                    } else {
                        // Swap order - more selective filter goes first
                        Ok(Algebra::Filter {
                            pattern: Box::new(Algebra::Filter {
                                pattern: inner_pattern,
                                condition,
                            }),
                            condition: inner_condition,
                        })
                    }
                } else {
                    Ok(Algebra::Filter {
                        pattern: Box::new(self.reorder_filters(*pattern)?),
                        condition,
                    })
                }
            }
            _ => self.apply_to_children(algebra, |child| self.reorder_filters(child)),
        }
    }

    /// Estimate filter selectivity (lower is more selective)
    fn estimate_filter_selectivity(&self, expr: &Expression) -> f64 {
        match expr {
            // Equality filters are usually very selective
            Expression::Binary {
                op: BinaryOperator::Equal,
                ..
            } => 0.1,

            // Inequality filters are less selective
            Expression::Binary {
                op: BinaryOperator::NotEqual,
                ..
            } => 0.9,

            // Range filters
            Expression::Binary {
                op: BinaryOperator::Less,
                ..
            } => 0.3,
            Expression::Binary {
                op: BinaryOperator::Greater,
                ..
            } => 0.3,
            Expression::Binary {
                op: BinaryOperator::LessEqual,
                ..
            } => 0.35,
            Expression::Binary {
                op: BinaryOperator::GreaterEqual,
                ..
            } => 0.35,

            // Logical operations
            Expression::Binary {
                op: BinaryOperator::And,
                left,
                right,
            } => self.estimate_filter_selectivity(left) * self.estimate_filter_selectivity(right),
            Expression::Binary {
                op: BinaryOperator::Or,
                left,
                right,
            } => {
                let sel_left = self.estimate_filter_selectivity(left);
                let sel_right = self.estimate_filter_selectivity(right);
                sel_left + sel_right - (sel_left * sel_right)
            }

            // Functions - depends on the function
            Expression::Function { name, .. } => match name.as_str() {
                "bound" => 0.8,
                "isIRI" | "isBlank" | "isLiteral" => 0.3,
                "regex" => 0.2,
                "contains" | "strstarts" | "strends" => 0.15,
                _ => 0.5,
            },

            // EXISTS/NOT EXISTS
            Expression::Exists { .. } => 0.5,
            Expression::NotExists { .. } => 0.5,

            // Default
            _ => 0.5,
        }
    }

    /// Calculate join complexity for adaptive optimization
    fn calculate_join_complexity(&self, patterns: &[TriplePattern]) -> JoinComplexity {
        let pattern_count = patterns.len();
        let total_variables = patterns
            .iter()
            .flat_map(|p| p.variables())
            .collect::<HashSet<_>>()
            .len();

        // Classify based on pattern count and variable complexity
        match pattern_count {
            1..=2 if total_variables <= 3 => JoinComplexity::Simple,
            3..=5 if total_variables <= 8 => JoinComplexity::Moderate,
            _ => JoinComplexity::Complex,
        }
    }

    /// Select most selective index for complex queries
    fn select_most_selective_index(&self, pattern: &TriplePattern) -> Option<(IndexType, f64)> {
        self.statistics
            .index_stats
            .available_indexes
            .iter()
            .filter(|idx| self.index_supports_pattern(idx, pattern))
            .min_by(|a, b| {
                let sel_a = self
                    .statistics
                    .index_stats
                    .index_selectivity
                    .get(a)
                    .copied()
                    .unwrap_or(1.0);
                let sel_b = self
                    .statistics
                    .index_stats
                    .index_selectivity
                    .get(b)
                    .copied()
                    .unwrap_or(1.0);
                sel_a.partial_cmp(&sel_b).unwrap()
            })
            .and_then(|idx| {
                let selectivity = self
                    .statistics
                    .index_stats
                    .index_selectivity
                    .get(idx)
                    .copied()
                    .unwrap_or(1.0);
                let cost = self
                    .statistics
                    .index_stats
                    .index_access_cost
                    .get(idx)
                    .copied()
                    .unwrap_or(1.0);
                Some((idx.clone(), cost * selectivity))
            })
    }

    /// Get shared variables between two patterns
    fn get_shared_variables_in_patterns(
        &self,
        p1: &TriplePattern,
        p2: &TriplePattern,
    ) -> Vec<Variable> {
        let vars1: HashSet<_> = p1.variables().into_iter().collect();
        let vars2: HashSet<_> = p2.variables().into_iter().collect();
        vars1.intersection(&vars2).cloned().collect()
    }

    /// Create intersection opportunity for two patterns
    fn create_intersection_opportunity(
        &self,
        i: usize,
        j: usize,
        shared_vars: &[Variable],
    ) -> Option<crate::bgp_optimizer::IndexIntersection> {
        // Create intersection metadata
        Some(crate::bgp_optimizer::IndexIntersection {
            left_pattern_idx: i,
            right_pattern_idx: j,
            intersection_variable: shared_vars.first()?.clone(),
            estimated_benefit: 2.0, // Simplified benefit estimation
            intersection_type: crate::bgp_optimizer::IntersectionType::VariableJoin,
        })
    }

    /// Check if index union can be used
    fn can_use_index_union(
        &self,
        left_patterns: &[TriplePattern],
        right_patterns: &[TriplePattern],
    ) -> bool {
        // Simple heuristic: can use index union if patterns are similar structure
        left_patterns.len() == 1
            && right_patterns.len() == 1
            && self.patterns_have_similar_structure(&left_patterns[0], &right_patterns[0])
    }

    /// Check if patterns have similar structure for union optimization
    fn patterns_have_similar_structure(&self, p1: &TriplePattern, p2: &TriplePattern) -> bool {
        // Patterns are similar if they have the same predicate or object structure
        match (&p1.predicate, &p2.predicate) {
            (Term::Iri(iri1), Term::Iri(iri2)) if iri1 == iri2 => true,
            (Term::Variable(_), Term::Variable(_)) => true,
            _ => false,
        }
    }

    /// Create index union plan
    fn create_index_union_plan(
        &self,
        left_patterns: &[TriplePattern],
        right_patterns: &[TriplePattern],
    ) -> Result<IndexUnionPlan> {
        let left_indexes = left_patterns
            .iter()
            .filter_map(|p| self.select_single_optimal_index(p).map(|(idx, _)| idx))
            .collect();

        let right_indexes = right_patterns
            .iter()
            .filter_map(|p| self.select_single_optimal_index(p).map(|(idx, _)| idx))
            .collect();

        // Advanced cost estimation for index union
        let left_cost = self.estimate_index_group_cost(&left_indexes)?;
        let right_cost = self.estimate_index_group_cost(&right_indexes)?;
        let union_overhead = (left_cost + right_cost) * 0.15; // 15% overhead for union operation
        let deduplication_cost = (left_cost * right_cost).sqrt() * 0.1; // Cost of deduplication
        
        let union_cost = left_cost + right_cost + union_overhead + deduplication_cost;
        
        // Advanced selectivity estimation based on pattern overlap
        let left_selectivity = self.estimate_pattern_group_selectivity(left_patterns)?;
        let right_selectivity = self.estimate_pattern_group_selectivity(right_patterns)?;
        let overlap_factor = self.estimate_pattern_overlap(left_patterns, right_patterns)?;
        
        // Union selectivity: left + right - overlap
        let estimated_selectivity = left_selectivity + right_selectivity - (left_selectivity * right_selectivity * overlap_factor);

        Ok(IndexUnionPlan {
            left_indexes,
            right_indexes,
            union_cost,
            estimated_selectivity,
        })
    }

    /// Execute index union plan
    fn execute_index_union_plan(&self, plan: IndexUnionPlan) -> Result<Algebra> {
        // Create left-side BGP with index hints
        let left_bgp = self.create_index_optimized_bgp(&plan.left_indexes)?;
        
        // Create right-side BGP with index hints
        let right_bgp = self.create_index_optimized_bgp(&plan.right_indexes)?;
        
        // Create union with proper index hints and cost estimates
        let union = Algebra::Union {
            left: Box::new(left_bgp),
            right: Box::new(right_bgp),
        };
        
        // Apply index-aware optimization if beneficial
        if plan.estimated_selectivity < 0.7 && plan.union_cost < 10000.0 {
            // Wrap with index scan hints for better execution
            Ok(Algebra::Filter {
                pattern: Box::new(union),
                condition: Expression::Bound(true),
            })
        } else {
            Ok(union)
        }
    }

    /// Create index-optimized BGP from index types
    fn create_index_optimized_bgp(&self, indexes: &[IndexType]) -> Result<Algebra> {
        if indexes.is_empty() {
            return Ok(Algebra::Table);
        }
        
        // Create BGP patterns optimized for the given indexes
        let mut patterns = Vec::new();
        
        for index_type in indexes {
            match index_type {
                IndexType::SPO => {
                    // Create subject-predicate-object optimized pattern
                    patterns.push(TriplePattern {
                        subject: Term::Variable(Variable::new("s")),
                        predicate: Term::Variable(Variable::new("p")),
                        object: Term::Variable(Variable::new("o")),
                    });
                }
                IndexType::PSO => {
                    // Create predicate-subject-object optimized pattern
                    patterns.push(TriplePattern {
                        subject: Term::Variable(Variable::new("s")),
                        predicate: Term::Variable(Variable::new("p")),
                        object: Term::Variable(Variable::new("o")),
                    });
                }
                IndexType::SubjectPredicate => {
                    patterns.push(TriplePattern {
                        subject: Term::Variable(Variable::new("s")),
                        predicate: Term::Variable(Variable::new("p")),
                        object: Term::Variable(Variable::new("o")),
                    });
                }
                IndexType::PredicateObject => {
                    patterns.push(TriplePattern {
                        subject: Term::Variable(Variable::new("s")),
                        predicate: Term::Variable(Variable::new("p")),
                        object: Term::Variable(Variable::new("o")),
                    });
                }
                _ => {
                    // Default pattern for other index types
                    patterns.push(TriplePattern {
                        subject: Term::Variable(Variable::new("s")),
                        predicate: Term::Variable(Variable::new("p")),
                        object: Term::Variable(Variable::new("o")),
                    });
                }
            }
        }
        
        if patterns.len() == 1 {
            Ok(Algebra::TriplePattern(patterns[0].clone()))
        } else {
            // Create efficient join structure for multiple patterns
            let mut bgp = Algebra::TriplePattern(patterns[0].clone());
            for pattern in patterns.iter().skip(1) {
                bgp = Algebra::Join {
                    left: Box::new(bgp),
                    right: Box::new(Algebra::TriplePattern(pattern.clone())),
                };
            }
            Ok(bgp)
        }
    }

    /// Estimate cost for a group of indexes
    fn estimate_index_group_cost(&self, indexes: &[IndexType]) -> Result<f64> {
        if indexes.is_empty() {
            return Ok(100.0); // Base cost for empty index group
        }
        
        let mut total_cost = 0.0;
        for index_type in indexes {
            let index_cost = self.statistics.index_stats.index_access_cost
                .get(index_type)
                .unwrap_or(&500.0); // Default cost
            total_cost += index_cost;
        }
        
        // Apply scaling factor for multiple indexes (diminishing returns)
        let scaling_factor = if indexes.len() > 1 {
            1.0 + (indexes.len() as f64 - 1.0) * 0.3 // Each additional index adds 30% overhead
        } else {
            1.0
        };
        
        Ok(total_cost * scaling_factor)
    }
    
    /// Estimate selectivity for a group of patterns
    fn estimate_pattern_group_selectivity(&self, patterns: &[TriplePattern]) -> Result<f64> {
        if patterns.is_empty() {
            return Ok(1.0);
        }
        
        let mut combined_selectivity = 1.0;
        for pattern in patterns {
            let pattern_key = format!("{:?}", pattern);
            let pattern_selectivity = self.statistics.pattern_cardinality
                .get(&pattern_key)
                .map(|card| (*card as f64) / 1_000_000.0) // Normalize by estimated total triples
                .unwrap_or(0.1); // Default selectivity
            
            // Combine selectivities (assuming independence)
            combined_selectivity *= pattern_selectivity.min(1.0);
        }
        
        Ok(combined_selectivity)
    }
    
    /// Estimate overlap between two pattern groups
    fn estimate_pattern_overlap(&self, left_patterns: &[TriplePattern], right_patterns: &[TriplePattern]) -> Result<f64> {
        if left_patterns.is_empty() || right_patterns.is_empty() {
            return Ok(0.0);
        }
        
        let mut overlap_score = 0.0;
        let mut total_comparisons = 0;
        
        for left_pattern in left_patterns {
            for right_pattern in right_patterns {
                total_comparisons += 1;
                
                // Check for variable overlap
                let left_vars: std::collections::HashSet<_> = left_pattern.variables().into_iter().collect();
                let right_vars: std::collections::HashSet<_> = right_pattern.variables().into_iter().collect();
                let shared_vars = left_vars.intersection(&right_vars).count();
                
                if shared_vars > 0 {
                    // Higher overlap score for more shared variables
                    overlap_score += (shared_vars as f64) / (left_vars.len() + right_vars.len()) as f64;
                }
                
                // Check for term overlap (exact matches in subject/predicate/object)
                let mut term_matches = 0;
                if left_pattern.subject == right_pattern.subject { term_matches += 1; }
                if left_pattern.predicate == right_pattern.predicate { term_matches += 1; }
                if left_pattern.object == right_pattern.object { term_matches += 1; }
                
                if term_matches > 0 {
                    overlap_score += (term_matches as f64) / 3.0; // Normalize by max possible matches
                }
            }
        }
        
        if total_comparisons > 0 {
            Ok((overlap_score / total_comparisons as f64).min(1.0))
        } else {
            Ok(0.0)
        }
    }

    /// Create index filter plan
    fn create_index_filter_plan(
        &self,
        patterns: &[TriplePattern],
        condition: &Expression,
    ) -> Result<Option<IndexFilterPlan>> {
        // Analyze if condition can be pushed to index level
        if let Some((pattern_idx, compatible_index)) =
            self.find_filter_compatible_index(patterns, condition)
        {
            let push_down_conditions = vec![condition.clone()];
            let estimated_cost = 500.0; // Simplified cost

            Ok(Some(IndexFilterPlan {
                pattern_index: pattern_idx,
                filter_index: compatible_index,
                push_down_conditions,
                estimated_cost,
            }))
        } else {
            Ok(None)
        }
    }

    /// Find index compatible with filter condition
    fn find_filter_compatible_index(
        &self,
        patterns: &[TriplePattern],
        condition: &Expression,
    ) -> Option<(usize, IndexType)> {
        // Simplified: look for bitmap or B-tree indexes that can handle the filter
        for (i, pattern) in patterns.iter().enumerate() {
            if let Some((index_type, _)) = self.select_single_optimal_index(pattern) {
                match index_type {
                    IndexType::BTree | IndexType::Bitmap => return Some((i, index_type)),
                    _ => continue,
                }
            }
        }
        None
    }

    /// Execute index filter plan
    fn execute_index_filter_plan(&self, plan: IndexFilterPlan) -> Result<Algebra> {
        // Create optimized filter representation
        // In practice, this would create specialized filter algebra with index hints
        Ok(Algebra::Filter {
            pattern: Box::new(Algebra::Table), // Placeholder
            condition: plan.push_down_conditions.into_iter().next().unwrap(),
        })
    }
}

impl Default for QueryOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimization rule trait
pub trait OptimizationRule {
    fn name(&self) -> &str;
    fn apply(&self, algebra: Algebra) -> Result<Algebra>;
    fn applicable(&self, algebra: &Algebra) -> bool;
}

/// Collection of optimization rules
pub struct RuleSet {
    rules: Vec<Box<dyn OptimizationRule>>,
}

impl RuleSet {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn add_rule<R: OptimizationRule + 'static>(&mut self, rule: R) {
        self.rules.push(Box::new(rule));
    }

    pub fn apply_rules(&self, algebra: Algebra) -> Result<Algebra> {
        let mut result = algebra;

        for rule in &self.rules {
            if rule.applicable(&result) {
                result = rule.apply(result)?;
            }
        }

        Ok(result)
    }
}

impl Default for RuleSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{Algebra, BinaryOperator, Expression, Iri, Term, TriplePattern, Variable};

    #[test]
    fn test_constant_folding() {
        let optimizer = QueryOptimizer::new();

        // Create a filter with always-true condition
        let pattern = Algebra::Bgp(vec![TriplePattern::new(
            Term::Variable(Variable::new("s").unwrap()),
            Term::Variable(Variable::new("p").unwrap()),
            Term::Variable(Variable::new("o").unwrap()),
        )]);

        let always_true = Expression::Binary {
            op: BinaryOperator::Equal,
            left: Box::new(Expression::Literal(crate::algebra::Literal {
                value: "1".to_string(),
                language: None,
                datatype: None,
            })),
            right: Box::new(Expression::Literal(crate::algebra::Literal {
                value: "1".to_string(),
                language: None,
                datatype: None,
            })),
        };

        let filter = Algebra::Filter {
            pattern: Box::new(pattern.clone()),
            condition: always_true,
        };

        let optimized = optimizer.optimize(filter).unwrap();

        // Should remove the filter since condition is always true
        match optimized {
            Algebra::Bgp(_) => {} // Expected
            _ => panic!("Expected BGP after optimizing always-true filter"),
        }
    }

    #[test]
    fn test_join_with_zero() {
        let optimizer = QueryOptimizer::new();

        let left = Algebra::Bgp(vec![]);
        let right = Algebra::Zero;

        let join = Algebra::Join {
            left: Box::new(left),
            right: Box::new(right),
        };

        let optimized = optimizer.optimize(join).unwrap();

        // Join with Zero should become Zero
        assert!(matches!(optimized, Algebra::Zero));
    }
}
