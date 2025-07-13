//! Query optimization for SHACL validation
//!
//! This module provides query optimization capabilities to improve
//! the performance of SPARQL queries generated during SHACL validation.

use crate::{PropertyConstraint, ShaclAiError, Shape};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Query optimizer for validation queries
#[derive(Debug, Clone)]
pub struct QueryOptimizer {
    query_cache: HashMap<String, OptimizedQuery>,
}

/// Optimized query representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedQuery {
    pub original_query: String,
    pub optimized_query: String,
    pub execution_plan: String,
    pub estimated_improvement: f64,
}

/// Query optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryOptimization {
    pub shape_id: String,
    pub original_complexity: QueryComplexity,
    pub optimized_complexity: QueryComplexity,
    pub optimization_techniques: Vec<OptimizationTechnique>,
    pub estimated_speedup: f64,
}

/// Query complexity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryComplexity {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Optimization techniques that can be applied
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationTechnique {
    IndexHints,
    ConstraintReordering,
    SubqueryOptimization,
    JoinOptimization,
    FilterPushdown,
    ConstantFolding,
    RedundantConstraintElimination,
}

impl Default for QueryOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryOptimizer {
    pub fn new() -> Self {
        Self {
            query_cache: HashMap::new(),
        }
    }

    /// Optimize queries for a set of shapes
    pub fn optimize_queries(
        &self,
        shapes: &[Shape],
    ) -> Result<Vec<QueryOptimization>, ShaclAiError> {
        let mut optimizations = Vec::new();

        for shape in shapes {
            let complexity = self.analyze_shape_complexity(shape);
            let optimization_techniques = self.identify_optimization_opportunities(shape);
            let optimized_complexity =
                self.calculate_optimized_complexity(&complexity, &optimization_techniques);
            let estimated_speedup = self.estimate_speedup(&complexity, &optimized_complexity);

            optimizations.push(QueryOptimization {
                shape_id: shape.id.to_string(),
                original_complexity: complexity,
                optimized_complexity,
                optimization_techniques,
                estimated_speedup,
            });
        }

        Ok(optimizations)
    }

    /// Analyze the complexity of a SHACL shape
    fn analyze_shape_complexity(&self, shape: &Shape) -> QueryComplexity {
        let constraint_count = shape.property_constraints.len();
        let has_complex_constraints = shape
            .property_constraints
            .iter()
            .any(|c| self.is_complex_constraint(c));

        match (constraint_count, has_complex_constraints) {
            (0..=5, false) => QueryComplexity::Low,
            (6..=15, false) | (0..=8, true) => QueryComplexity::Medium,
            (16..=30, _) | (9..=15, true) => QueryComplexity::High,
            _ => QueryComplexity::VeryHigh,
        }
    }

    /// Check if a constraint is computationally complex
    fn is_complex_constraint(&self, constraint: &PropertyConstraint) -> bool {
        // Complex constraints include regex patterns, mathematical operations, etc.
        constraint.path.contains("regex")
            || constraint.path.contains("math")
            || constraint.path.contains("aggregate")
            || constraint.path.len() > 50 // Very long paths are typically complex
    }

    /// Identify optimization opportunities for a shape
    fn identify_optimization_opportunities(&self, shape: &Shape) -> Vec<OptimizationTechnique> {
        let mut techniques = Vec::new();

        // Always consider constraint reordering
        techniques.push(OptimizationTechnique::ConstraintReordering);

        // Check for filter pushdown opportunities
        if shape
            .property_constraints
            .iter()
            .any(|c| self.can_push_down_filter(c))
        {
            techniques.push(OptimizationTechnique::FilterPushdown);
        }

        // Check for redundant constraints
        if self.has_redundant_constraints(shape) {
            techniques.push(OptimizationTechnique::RedundantConstraintElimination);
        }

        // Check for join optimization opportunities
        if shape.property_constraints.len() > 3 {
            techniques.push(OptimizationTechnique::JoinOptimization);
        }

        // Check for constant folding opportunities
        if shape
            .property_constraints
            .iter()
            .any(|c| self.has_constant_expressions(c))
        {
            techniques.push(OptimizationTechnique::ConstantFolding);
        }

        // Check for subquery optimization
        if shape
            .property_constraints
            .iter()
            .any(|c| self.has_subquery_pattern(c))
        {
            techniques.push(OptimizationTechnique::SubqueryOptimization);
        }

        // Always consider index hints for complex shapes
        if shape.property_constraints.len() > 1 {
            techniques.push(OptimizationTechnique::IndexHints);
        }

        techniques
    }

    /// Calculate optimized complexity based on applied techniques
    fn calculate_optimized_complexity(
        &self,
        original: &QueryComplexity,
        techniques: &[OptimizationTechnique],
    ) -> QueryComplexity {
        let optimization_factor = techniques.len() as f64 * 0.15; // Each technique reduces complexity by ~15%

        match (original, optimization_factor > 0.5) {
            (QueryComplexity::VeryHigh, true) => QueryComplexity::High,
            (QueryComplexity::High, true) => QueryComplexity::Medium,
            (QueryComplexity::Medium, true) => QueryComplexity::Low,
            (complexity, _) => complexity.clone(),
        }
    }

    /// Estimate speedup from optimizations
    fn estimate_speedup(&self, original: &QueryComplexity, optimized: &QueryComplexity) -> f64 {
        let original_factor = match original {
            QueryComplexity::Low => 1.0,
            QueryComplexity::Medium => 2.0,
            QueryComplexity::High => 4.0,
            QueryComplexity::VeryHigh => 8.0,
        };

        let optimized_factor = match optimized {
            QueryComplexity::Low => 1.0,
            QueryComplexity::Medium => 2.0,
            QueryComplexity::High => 4.0,
            QueryComplexity::VeryHigh => 8.0,
        };

        original_factor / optimized_factor
    }

    /// Check if filter can be pushed down
    fn can_push_down_filter(&self, constraint: &PropertyConstraint) -> bool {
        // Simple heuristic: filters on basic properties can often be pushed down
        constraint.path.contains("=")
            || constraint.path.contains(">")
            || constraint.path.contains("<")
    }

    /// Check for redundant constraints
    fn has_redundant_constraints(&self, shape: &Shape) -> bool {
        // Simple check: if multiple constraints have very similar paths
        let paths: Vec<&String> = shape.property_constraints.iter().map(|c| &c.path).collect();

        for (i, path1) in paths.iter().enumerate() {
            for path2 in paths.iter().skip(i + 1) {
                if self.are_paths_similar(path1, path2) {
                    return true;
                }
            }
        }

        false
    }

    /// Check if paths are similar (potential redundancy)
    fn are_paths_similar(&self, path1: &str, path2: &str) -> bool {
        // Simple similarity check
        let common_prefix_len = path1
            .chars()
            .zip(path2.chars())
            .take_while(|(a, b)| a == b)
            .count();

        let max_len = path1.len().max(path2.len());

        common_prefix_len as f64 / max_len as f64 > 0.8 // 80% similarity threshold
    }

    /// Check for constant expressions
    fn has_constant_expressions(&self, constraint: &PropertyConstraint) -> bool {
        // Look for patterns that suggest constant values
        constraint.path.contains("\"") || // String literals
        constraint.path.chars().any(|c| c.is_ascii_digit()) // Numbers
    }

    /// Check for subquery patterns
    fn has_subquery_pattern(&self, constraint: &PropertyConstraint) -> bool {
        // Look for nested query patterns
        constraint.path.contains("SELECT")
            || constraint.path.contains("EXISTS")
            || constraint.path.contains("NOT EXISTS")
    }

    /// Cache an optimized query
    pub fn cache_optimized_query(&mut self, query_id: String, optimized: OptimizedQuery) {
        self.query_cache.insert(query_id, optimized);
    }

    /// Get a cached optimized query
    pub fn get_cached_query(&self, query_id: &str) -> Option<&OptimizedQuery> {
        self.query_cache.get(query_id)
    }

    /// Clear the query cache
    pub fn clear_cache(&mut self) {
        self.query_cache.clear();
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> QueryCacheStats {
        QueryCacheStats {
            cached_queries: self.query_cache.len(),
            total_estimated_improvement: self
                .query_cache
                .values()
                .map(|q| q.estimated_improvement)
                .sum::<f64>()
                / self.query_cache.len() as f64,
        }
    }
}

/// Query cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryCacheStats {
    pub cached_queries: usize,
    pub total_estimated_improvement: f64,
}
