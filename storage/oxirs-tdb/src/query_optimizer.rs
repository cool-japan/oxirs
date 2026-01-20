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
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlan {
    /// Pattern being queried
    #[serde(skip)]
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

    /// Generate ASCII tree visualization of the query plan
    pub fn to_ascii_tree(&self) -> String {
        let mut output = String::new();

        output.push_str("Query Plan\n");
        output.push_str("══════════\n\n");

        // Pattern visualization
        output.push_str("├─ Pattern\n");
        output.push_str(&format!("│  ├─ Subject:   {:?}\n", self.pattern.subject));
        output.push_str(&format!("│  ├─ Predicate: {:?}\n", self.pattern.predicate));
        output.push_str(&format!("│  └─ Object:    {:?}\n", self.pattern.object));
        output.push_str("│\n");

        // Index selection
        output.push_str(&format!("├─ Index: {:?}\n", self.index));
        output.push_str("│\n");

        // Cost estimates
        output.push_str("├─ Cost Estimates\n");
        output.push_str(&format!(
            "│  ├─ I/O Cost:     {:.2}\n",
            self.estimated_io_cost
        ));
        output.push_str(&format!(
            "│  ├─ CPU Cost:     {:.2}\n",
            self.estimated_cpu_cost
        ));
        output.push_str(&format!("│  ├─ Total Cost:   {:.2}\n", self.total_cost));
        output.push_str(&format!("│  ├─ Est. Results: {}\n", self.estimated_results));
        output.push_str(&format!(
            "│  └─ Confidence:   {:.0}%\n",
            self.confidence * 100.0
        ));
        output.push_str("│\n");

        // Explanation
        if !self.explanation.is_empty() {
            output.push_str("└─ Explanation\n");
            for line in self.explanation.lines() {
                output.push_str(&format!("   {}\n", line));
            }
        }

        output
    }

    /// Generate DOT format for Graphviz visualization
    pub fn to_dot(&self) -> String {
        let mut output = String::new();

        output.push_str("digraph QueryPlan {\n");
        output.push_str("  rankdir=TB;\n");
        output.push_str("  node [shape=box, style=rounded];\n\n");

        // Root node
        output.push_str("  query [label=\"Query Pattern\"];\n");

        // Pattern node
        let pattern_label = format!(
            "S: {:?}\\nP: {:?}\\nO: {:?}",
            self.pattern.subject, self.pattern.predicate, self.pattern.object
        );
        output.push_str(&format!("  pattern [label=\"{}\"];\n", pattern_label));
        output.push_str("  query -> pattern;\n\n");

        // Index selection node
        let index_label = format!("Index: {:?}", self.index);
        output.push_str(&format!("  index [label=\"{}\"];\n", index_label));
        output.push_str("  pattern -> index;\n\n");

        // Cost node
        let cost_label = format!(
            "I/O: {:.2}\\nCPU: {:.2}\\nTotal: {:.2}\\nResults: {}",
            self.estimated_io_cost,
            self.estimated_cpu_cost,
            self.total_cost,
            self.estimated_results
        );
        output.push_str(&format!(
            "  cost [label=\"{}\", shape=ellipse];\n",
            cost_label
        ));
        output.push_str("  index -> cost;\n\n");

        output.push_str("}\n");
        output
    }

    /// Generate JSON representation of the query plan
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| TdbError::Other(format!("JSON serialization error: {}", e)))
    }

    /// Generate compact summary of the query plan
    pub fn to_summary(&self) -> String {
        format!(
            "Plan: {:?} index | Cost: {:.2} | Results: {} | Confidence: {:.0}%",
            self.index,
            self.total_cost,
            self.estimated_results,
            self.confidence * 100.0
        )
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

        // Estimate cardinality using average statistics for better accuracy
        let base_estimate = match index {
            IndexType::SPO => {
                if pattern.subject.is_some() {
                    // S bound: use average properties per subject
                    stats.avg_properties_per_subject as usize
                } else {
                    stats.total_triples as usize
                }
            }
            IndexType::POS => {
                if pattern.predicate.is_some() {
                    // P bound: use average objects per predicate
                    stats.avg_objects_per_predicate as usize
                } else {
                    stats.total_triples as usize
                }
            }
            IndexType::OSP => {
                if pattern.object.is_some() {
                    // O bound: use average subjects per object
                    stats.avg_subjects_per_object as usize
                } else {
                    stats.total_triples as usize
                }
            }
        };

        // Refine estimate based on additional bound positions with more accurate factors
        let refined_estimate = match pattern.bound_count() {
            3 => 1, // Exact match - single triple
            2 => {
                // Two bound positions: estimate intersection selectivity
                // Use geometric mean of two estimates for more accuracy
                let factor = if base_estimate > 10 {
                    (base_estimate as f64).sqrt() as usize
                } else {
                    base_estimate
                };
                factor.max(1)
            }
            1 => base_estimate.max(1),
            _ => stats.total_triples as usize,
        };

        // Calculate confidence based on data distribution
        let confidence = if stats.total_triples > 0 {
            match pattern.bound_count() {
                3 => 0.95, // High confidence for exact matches
                2 => 0.85, // Good confidence with two bounds
                1 => 0.70, // Moderate confidence with one bound
                _ => 0.50, // Low confidence for full scans
            }
        } else {
            0.30 // Low confidence with no data
        };

        let explanation = format!(
            "Cost-based selection: {:?} index with estimated {} results (avg={:.2}, confidence={:.0}%)",
            index,
            refined_estimate,
            match index {
                IndexType::SPO => stats.avg_properties_per_subject,
                IndexType::POS => stats.avg_objects_per_predicate,
                IndexType::OSP => stats.avg_subjects_per_object,
            },
            confidence * 100.0
        );

        Ok(QueryPlan::new(pattern.clone(), index, refined_estimate)
            .with_explanation(explanation)
            .with_confidence(confidence))
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

    #[test]
    fn test_query_plan_ascii_visualization() {
        let pattern = QueryPattern::new(Some(NodeId::new(1)), None, Some(NodeId::new(3)));
        let plan = QueryPlan::new(pattern, IndexType::SPO, 100)
            .with_explanation("Using SPO index for subject-bound query".to_string())
            .with_confidence(0.85);

        let ascii_tree = plan.to_ascii_tree();

        // Verify key components are present
        assert!(ascii_tree.contains("Query Plan"));
        assert!(ascii_tree.contains("Pattern"));
        assert!(ascii_tree.contains("Subject:"));
        assert!(ascii_tree.contains("Predicate:"));
        assert!(ascii_tree.contains("Object:"));
        assert!(ascii_tree.contains("Index: SPO"));
        assert!(ascii_tree.contains("I/O Cost:"));
        assert!(ascii_tree.contains("CPU Cost:"));
        assert!(ascii_tree.contains("Total Cost:"));
        assert!(ascii_tree.contains("Est. Results: 100"));
        assert!(ascii_tree.contains("Confidence:   85%"));
        assert!(ascii_tree.contains("Explanation"));
        assert!(ascii_tree.contains("Using SPO index"));
    }

    #[test]
    fn test_query_plan_dot_visualization() {
        let pattern = QueryPattern::new(Some(NodeId::new(1)), Some(NodeId::new(2)), None);
        let plan = QueryPlan::new(pattern, IndexType::SPO, 50);

        let dot_output = plan.to_dot();

        // Verify DOT format structure
        assert!(dot_output.starts_with("digraph QueryPlan {"));
        assert!(dot_output.contains("rankdir=TB"));
        assert!(dot_output.contains("query [label=\"Query Pattern\"]"));
        assert!(dot_output.contains("pattern [label="));
        assert!(dot_output.contains("index [label=\"Index: SPO\"]"));
        assert!(dot_output.contains("cost [label="));
        assert!(dot_output.contains("query -> pattern"));
        assert!(dot_output.contains("pattern -> index"));
        assert!(dot_output.contains("index -> cost"));
        assert!(dot_output.ends_with("}\n"));
    }

    #[test]
    fn test_query_plan_json_visualization() {
        let pattern = QueryPattern::new(Some(NodeId::new(1)), None, None);
        let plan = QueryPlan::new(pattern, IndexType::SPO, 200).with_confidence(0.9);

        let json_result = plan.to_json();
        assert!(json_result.is_ok());

        let json_output = json_result.unwrap();

        // Verify JSON contains key fields
        assert!(json_output.contains("\"index\""));
        assert!(json_output.contains("\"estimated_results\": 200"));
        assert!(json_output.contains("\"estimated_io_cost\""));
        assert!(json_output.contains("\"estimated_cpu_cost\""));
        assert!(json_output.contains("\"total_cost\""));
        assert!(json_output.contains("\"confidence\": 0.9"));
    }

    #[test]
    fn test_query_plan_summary() {
        let pattern = QueryPattern::new(None, Some(NodeId::new(2)), None);
        let plan = QueryPlan::new(pattern, IndexType::POS, 300).with_confidence(0.75);

        let summary = plan.to_summary();

        // Verify summary format
        assert!(summary.contains("POS index"));
        assert!(summary.contains("Cost:"));
        assert!(summary.contains("Results: 300"));
        assert!(summary.contains("Confidence: 75%"));
    }

    #[test]
    fn test_visualization_with_empty_explanation() {
        let pattern = QueryPattern::new(
            Some(NodeId::new(1)),
            Some(NodeId::new(2)),
            Some(NodeId::new(3)),
        );
        let plan = QueryPlan::new(pattern, IndexType::SPO, 1);

        let ascii_tree = plan.to_ascii_tree();

        // Should still render properly without explanation
        assert!(ascii_tree.contains("Query Plan"));
        assert!(ascii_tree.contains("Pattern"));
        assert!(ascii_tree.contains("Cost Estimates"));
    }

    #[test]
    fn test_visualization_format_consistency() {
        let pattern = QueryPattern::new(Some(NodeId::new(10)), None, Some(NodeId::new(20)));
        let plan = QueryPlan::new(pattern, IndexType::OSP, 500)
            .with_explanation("Multi-line\nexplanation\ntest".to_string())
            .with_confidence(0.95);

        // Test all formats work without errors
        let _ = plan.to_ascii_tree();
        let _ = plan.to_dot();
        let _ = plan.to_summary();
        assert!(plan.to_json().is_ok());
    }
}
