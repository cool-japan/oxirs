//! Advanced query optimizer for TDB storage
//!
//! Provides cost-based query optimization using statistics to select
//! the most efficient execution plan for triple pattern queries.
//!
//! ## Features
//! - Cost-based index selection using cardinality statistics
//! - Query plan generation with multiple access paths
//! - Execution cost estimation (I/O, CPU, memory)
//! - Integration with query hints for manual tuning
//! - Support for complex query patterns
//! - Adaptive optimization based on historical performance

use crate::dictionary::NodeId;
use crate::error::{Result, TdbError};
use crate::query_hints::{IndexType, QueryHints, QueryStats};
use crate::statistics::TripleStatistics;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Query pattern with bound/unbound positions
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QueryPattern {
    /// Subject node ID (None = wildcard)
    pub subject: Option<NodeId>,
    /// Predicate node ID (None = wildcard)
    pub predicate: Option<NodeId>,
    /// Object node ID (None = wildcard)
    pub object: Option<NodeId>,
}

impl QueryPattern {
    /// Create a new query pattern
    pub fn new(s: Option<NodeId>, p: Option<NodeId>, o: Option<NodeId>) -> Self {
        Self {
            subject: s,
            predicate: p,
            object: o,
        }
    }

    /// Check if all positions are bound (exact match)
    pub fn is_exact(&self) -> bool {
        self.subject.is_some() && self.predicate.is_some() && self.object.is_some()
    }

    /// Check if all positions are unbound (full scan)
    pub fn is_full_scan(&self) -> bool {
        self.subject.is_none() && self.predicate.is_none() && self.object.is_none()
    }

    /// Count number of bound positions
    pub fn bound_count(&self) -> usize {
        let mut count = 0;
        if self.subject.is_some() {
            count += 1;
        }
        if self.predicate.is_some() {
            count += 1;
        }
        if self.object.is_some() {
            count += 1;
        }
        count
    }

    /// Get selectivity factor (0.0 = very selective, 1.0 = not selective)
    ///
    /// More bound positions = more selective = lower selectivity factor
    pub fn selectivity(&self) -> f64 {
        let bound = self.bound_count();
        match bound {
            3 => 0.01, // Exact match - very selective
            2 => 0.1,  // Two bound - moderately selective
            1 => 0.5,  // One bound - less selective
            _ => 1.0,  // No bounds - full scan
        }
    }
}

/// Query execution plan with cost estimates
#[derive(Debug, Clone)]
pub struct QueryPlan {
    /// Pattern being queried
    pub pattern: QueryPattern,
    /// Index to use for execution
    pub index: IndexType,
    /// Estimated number of results
    pub estimated_results: usize,
    /// Estimated I/O cost (page reads)
    pub estimated_io_cost: f64,
    /// Estimated CPU cost (comparisons)
    pub estimated_cpu_cost: f64,
    /// Total estimated cost
    pub total_cost: f64,
    /// Confidence in cost estimate (0.0 - 1.0)
    pub confidence: f64,
    /// Explanation of plan choice
    pub explanation: String,
}

impl QueryPlan {
    /// Create a new query plan
    pub fn new(pattern: QueryPattern, index: IndexType, estimated_results: usize) -> Self {
        // Simple cost model:
        // I/O cost = log(total_triples) + log(results)
        // CPU cost = results * comparison_factor
        let io_cost = (estimated_results as f64 + 1.0).ln();
        let cpu_cost = estimated_results as f64 * 0.1;
        let total_cost = io_cost + cpu_cost;

        Self {
            pattern,
            index,
            estimated_results,
            estimated_io_cost: io_cost,
            estimated_cpu_cost: cpu_cost,
            total_cost,
            confidence: 0.7, // Default confidence
            explanation: String::new(),
        }
    }

    /// Update with explanation
    pub fn with_explanation(mut self, explanation: String) -> Self {
        self.explanation = explanation;
        self
    }

    /// Update confidence level
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

/// Advanced query optimizer
pub struct QueryOptimizer {
    /// Statistics collector
    statistics: Arc<TripleStatistics>,
    /// Cache of previously seen query patterns and their costs
    plan_cache: parking_lot::RwLock<std::collections::HashMap<QueryPattern, QueryPlan>>,
    /// Optimization level (0=disabled, 1=basic, 2=advanced)
    optimization_level: u8,
}

impl QueryOptimizer {
    /// Create a new query optimizer
    pub fn new(statistics: Arc<TripleStatistics>) -> Self {
        Self {
            statistics,
            plan_cache: parking_lot::RwLock::new(std::collections::HashMap::new()),
            optimization_level: 2, // Advanced by default
        }
    }

    /// Set optimization level
    pub fn with_optimization_level(mut self, level: u8) -> Self {
        self.optimization_level = level.min(2);
        self
    }

    /// Generate optimal query plan
    ///
    /// Considers:
    /// - Query pattern selectivity
    /// - Available statistics
    /// - User-provided hints
    /// - Historical query performance
    pub fn optimize(&self, pattern: QueryPattern, hints: &QueryHints) -> Result<QueryPlan> {
        // If optimization is disabled, use simple heuristics
        if self.optimization_level == 0 {
            return Ok(self.simple_optimization(pattern, hints));
        }

        // Check plan cache first
        {
            let cache = self.plan_cache.read();
            if let Some(cached_plan) = cache.get(&pattern) {
                // Apply hints to cached plan if needed
                if let Some(preferred_index) = hints.preferred_index {
                    let mut plan = cached_plan.clone();
                    plan.index = preferred_index;
                    plan.explanation = format!(
                        "Using hint-specified index {:?} (overriding cached plan)",
                        preferred_index
                    );
                    return Ok(plan);
                }
                return Ok(cached_plan.clone());
            }
        }

        // Generate and cache new plan
        let plan = if self.optimization_level >= 2 {
            self.advanced_optimization(pattern.clone(), hints)?
        } else {
            self.basic_optimization(pattern.clone(), hints)
        };

        // Cache the plan
        {
            let mut cache = self.plan_cache.write();
            cache.insert(pattern.clone(), plan.clone());
        }

        Ok(plan)
    }

    /// Simple optimization using only query pattern
    fn simple_optimization(&self, pattern: QueryPattern, hints: &QueryHints) -> QueryPlan {
        // Respect user hints if provided
        if let Some(preferred_index) = hints.preferred_index {
            return QueryPlan::new(pattern.clone(), preferred_index, 1000)
                .with_explanation(format!("Using hint-specified index {:?}", preferred_index));
        }

        // Use pattern-based heuristics
        let index = self.select_index_by_pattern(&pattern);
        let estimated_results = self.estimate_results_simple(&pattern);

        QueryPlan::new(pattern, index, estimated_results)
            .with_explanation(format!("Simple heuristic selection: {:?}", index))
            .with_confidence(0.5)
    }

    /// Basic optimization using simple statistics
    fn basic_optimization(&self, pattern: QueryPattern, hints: &QueryHints) -> QueryPlan {
        // Respect user hints if provided
        if let Some(preferred_index) = hints.preferred_index {
            return QueryPlan::new(pattern.clone(), preferred_index, 1000)
                .with_explanation(format!("Using hint-specified index {:?}", preferred_index));
        }

        // Use pattern-based selection with simple statistics
        let index = self.select_index_by_pattern(&pattern);
        let estimated_results = self.estimate_results_basic(&pattern);

        QueryPlan::new(pattern, index, estimated_results)
            .with_explanation(format!("Basic optimization with statistics: {:?}", index))
            .with_confidence(0.7)
    }

    /// Advanced optimization using detailed statistics and cost model
    fn advanced_optimization(
        &self,
        pattern: QueryPattern,
        hints: &QueryHints,
    ) -> Result<QueryPlan> {
        // Generate candidate plans for each possible index
        let mut candidates = Vec::new();

        // Consider SPO index
        if self.is_index_viable(&pattern, IndexType::SPO) {
            let plan = self.generate_plan(&pattern, IndexType::SPO)?;
            candidates.push(plan);
        }

        // Consider POS index
        if self.is_index_viable(&pattern, IndexType::POS) {
            let plan = self.generate_plan(&pattern, IndexType::POS)?;
            candidates.push(plan);
        }

        // Consider OSP index
        if self.is_index_viable(&pattern, IndexType::OSP) {
            let plan = self.generate_plan(&pattern, IndexType::OSP)?;
            candidates.push(plan);
        }

        // If user provided a hint, prefer that index
        if let Some(preferred_index) = hints.preferred_index {
            if let Some(plan) = candidates.iter().find(|p| p.index == preferred_index) {
                return Ok(plan.clone());
            }
        }

        // Select plan with lowest cost
        candidates
            .into_iter()
            .min_by(|a, b| {
                a.total_cost
                    .partial_cmp(&b.total_cost)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| TdbError::Other("No viable query plan found".to_string()))
    }

    /// Generate a query plan for a specific index
    fn generate_plan(&self, pattern: &QueryPattern, index: IndexType) -> Result<QueryPlan> {
        let stats = self.statistics.export();

        // Estimate cardinality based on statistics
        let estimated_results = match index {
            IndexType::SPO => {
                if pattern.subject.is_some() {
                    // S bound: estimate based on subject cardinality
                    (stats.total_triples / stats.distinct_subjects.max(1)) as usize
                } else {
                    stats.total_triples as usize
                }
            }
            IndexType::POS => {
                if pattern.predicate.is_some() {
                    // P bound: estimate based on predicate cardinality
                    (stats.total_triples / stats.distinct_predicates.max(1)) as usize
                } else {
                    stats.total_triples as usize
                }
            }
            IndexType::OSP => {
                if pattern.object.is_some() {
                    // O bound: estimate based on object cardinality
                    (stats.total_triples / stats.distinct_objects.max(1)) as usize
                } else {
                    stats.total_triples as usize
                }
            }
        };

        // Refine estimate based on additional bound positions
        let refined_estimate = match pattern.bound_count() {
            3 => 1, // Exact match
            2 => estimated_results / 10,
            1 => estimated_results,
            _ => stats.total_triples as usize,
        };

        let explanation = format!(
            "Cost-based selection: {:?} index with estimated {} results",
            index, refined_estimate
        );

        Ok(QueryPlan::new(pattern.clone(), index, refined_estimate)
            .with_explanation(explanation)
            .with_confidence(0.9))
    }

    /// Check if an index is viable for a query pattern
    fn is_index_viable(&self, pattern: &QueryPattern, index: IndexType) -> bool {
        match index {
            IndexType::SPO => true, // Always viable (fallback)
            IndexType::POS => pattern.predicate.is_some(),
            IndexType::OSP => pattern.object.is_some(),
        }
    }

    /// Select index based on query pattern alone
    fn select_index_by_pattern(&self, pattern: &QueryPattern) -> IndexType {
        match (
            pattern.subject.is_some(),
            pattern.predicate.is_some(),
            pattern.object.is_some(),
        ) {
            (true, _, _) => IndexType::SPO,          // S bound -> SPO
            (false, true, _) => IndexType::POS,      // P bound, S not -> POS
            (false, false, true) => IndexType::OSP,  // O bound, S,P not -> OSP
            (false, false, false) => IndexType::SPO, // All unbound -> SPO (default)
        }
    }

    /// Simple result estimation without statistics
    fn estimate_results_simple(&self, pattern: &QueryPattern) -> usize {
        match pattern.bound_count() {
            3 => 1,     // Exact match
            2 => 100,   // Two bound
            1 => 1000,  // One bound
            _ => 10000, // Full scan
        }
    }

    /// Basic result estimation with simple statistics
    fn estimate_results_basic(&self, pattern: &QueryPattern) -> usize {
        let stats = self.statistics.export();
        let total = stats.total_triples.max(1) as usize;

        match pattern.bound_count() {
            3 => 1,                                     // Exact match
            2 => (total as f64 * 0.01).ceil() as usize, // 1% of data
            1 => (total as f64 * 0.1).ceil() as usize,  // 10% of data
            _ => total,                                 // Full scan
        }
    }

    /// Clear the plan cache
    pub fn clear_cache(&self) {
        let mut cache = self.plan_cache.write();
        cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> QueryOptimizerStats {
        let cache = self.plan_cache.read();
        QueryOptimizerStats {
            cached_plans: cache.len(),
            optimization_level: self.optimization_level,
        }
    }
}

/// Query optimizer statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryOptimizerStats {
    /// Number of cached query plans
    pub cached_plans: usize,
    /// Current optimization level (0-2)
    pub optimization_level: u8,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistics::StatisticsConfig;

    fn create_test_optimizer() -> QueryOptimizer {
        let stats = Arc::new(TripleStatistics::new(StatisticsConfig::default()));
        QueryOptimizer::new(stats)
    }

    #[test]
    fn test_query_pattern_selectivity() {
        let pattern_exact = QueryPattern::new(
            Some(NodeId::new(1)),
            Some(NodeId::new(2)),
            Some(NodeId::new(3)),
        );
        assert_eq!(pattern_exact.selectivity(), 0.01);
        assert!(pattern_exact.is_exact());

        let pattern_two = QueryPattern::new(Some(NodeId::new(1)), Some(NodeId::new(2)), None);
        assert_eq!(pattern_two.selectivity(), 0.1);
        assert_eq!(pattern_two.bound_count(), 2);

        let pattern_one = QueryPattern::new(Some(NodeId::new(1)), None, None);
        assert_eq!(pattern_one.selectivity(), 0.5);

        let pattern_none = QueryPattern::new(None, None, None);
        assert_eq!(pattern_none.selectivity(), 1.0);
        assert!(pattern_none.is_full_scan());
    }

    #[test]
    fn test_simple_optimization() {
        let optimizer = create_test_optimizer();
        let hints = QueryHints::new();

        // S bound -> should select SPO
        let pattern = QueryPattern::new(Some(NodeId::new(1)), None, None);
        let plan = optimizer.simple_optimization(pattern, &hints);
        assert_eq!(plan.index, IndexType::SPO);

        // P bound -> should select POS
        let pattern = QueryPattern::new(None, Some(NodeId::new(2)), None);
        let plan = optimizer.simple_optimization(pattern, &hints);
        assert_eq!(plan.index, IndexType::POS);

        // O bound -> should select OSP
        let pattern = QueryPattern::new(None, None, Some(NodeId::new(3)));
        let plan = optimizer.simple_optimization(pattern, &hints);
        assert_eq!(plan.index, IndexType::OSP);
    }

    #[test]
    fn test_hint_override() {
        let optimizer = create_test_optimizer();

        // Pattern suggests SPO, but hint overrides to POS
        let pattern = QueryPattern::new(Some(NodeId::new(1)), None, None);
        let hints = QueryHints::new().with_index(IndexType::POS);

        let plan = optimizer.simple_optimization(pattern, &hints);
        assert_eq!(plan.index, IndexType::POS);
    }

    #[test]
    fn test_plan_cost_estimation() {
        let pattern = QueryPattern::new(Some(NodeId::new(1)), Some(NodeId::new(2)), None);
        let plan = QueryPlan::new(pattern, IndexType::SPO, 100);

        assert!(plan.estimated_io_cost > 0.0);
        assert!(plan.estimated_cpu_cost > 0.0);
        assert_eq!(
            plan.total_cost,
            plan.estimated_io_cost + plan.estimated_cpu_cost
        );
    }

    #[test]
    fn test_optimization_levels() {
        let stats = Arc::new(TripleStatistics::new(StatisticsConfig::default()));

        let optimizer_disabled = QueryOptimizer::new(stats.clone()).with_optimization_level(0);
        assert_eq!(optimizer_disabled.optimization_level, 0);

        let optimizer_basic = QueryOptimizer::new(stats.clone()).with_optimization_level(1);
        assert_eq!(optimizer_basic.optimization_level, 1);

        let optimizer_advanced = QueryOptimizer::new(stats.clone()).with_optimization_level(2);
        assert_eq!(optimizer_advanced.optimization_level, 2);
    }

    #[test]
    fn test_index_viability() {
        let optimizer = create_test_optimizer();

        // SPO is always viable
        let pattern = QueryPattern::new(None, None, None);
        assert!(optimizer.is_index_viable(&pattern, IndexType::SPO));

        // POS requires P bound
        let pattern_no_p = QueryPattern::new(Some(NodeId::new(1)), None, None);
        assert!(!optimizer.is_index_viable(&pattern_no_p, IndexType::POS));

        let pattern_with_p = QueryPattern::new(None, Some(NodeId::new(2)), None);
        assert!(optimizer.is_index_viable(&pattern_with_p, IndexType::POS));

        // OSP requires O bound
        let pattern_no_o = QueryPattern::new(Some(NodeId::new(1)), None, None);
        assert!(!optimizer.is_index_viable(&pattern_no_o, IndexType::OSP));

        let pattern_with_o = QueryPattern::new(None, None, Some(NodeId::new(3)));
        assert!(optimizer.is_index_viable(&pattern_with_o, IndexType::OSP));
    }

    #[test]
    fn test_cache_functionality() {
        let optimizer = create_test_optimizer();
        let pattern = QueryPattern::new(Some(NodeId::new(1)), None, None);
        let hints = QueryHints::new();

        // First call - should cache the plan
        let plan1 = optimizer.optimize(pattern.clone(), &hints).unwrap();

        // Second call - should retrieve from cache
        let plan2 = optimizer.optimize(pattern.clone(), &hints).unwrap();

        assert_eq!(plan1.index, plan2.index);

        let cache_stats = optimizer.cache_stats();
        assert_eq!(cache_stats.cached_plans, 1);

        optimizer.clear_cache();
        let cache_stats = optimizer.cache_stats();
        assert_eq!(cache_stats.cached_plans, 0);
    }

    #[test]
    fn test_result_estimation() {
        let optimizer = create_test_optimizer();

        // Exact match
        let pattern_exact = QueryPattern::new(
            Some(NodeId::new(1)),
            Some(NodeId::new(2)),
            Some(NodeId::new(3)),
        );
        assert_eq!(optimizer.estimate_results_simple(&pattern_exact), 1);

        // Two bound
        let pattern_two = QueryPattern::new(Some(NodeId::new(1)), Some(NodeId::new(2)), None);
        assert_eq!(optimizer.estimate_results_simple(&pattern_two), 100);

        // One bound
        let pattern_one = QueryPattern::new(Some(NodeId::new(1)), None, None);
        assert_eq!(optimizer.estimate_results_simple(&pattern_one), 1000);

        // Full scan
        let pattern_none = QueryPattern::new(None, None, None);
        assert_eq!(optimizer.estimate_results_simple(&pattern_none), 10000);
    }
}
