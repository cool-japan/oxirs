//! # PropertyPathEvaluator - optimize_path_group Methods
//!
//! This module contains method implementations for `PropertyPathEvaluator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::prelude::*;

impl PropertyPathEvaluator {
    /// Analyze and optimize a property path for better performance
    pub fn optimize_path(&self, path: &PropertyPath) -> OptimizedPropertyPath {
        let complexity = path.complexity();
        let can_use_sparql = path.can_use_sparql_path();
        let estimated_cost = self.estimate_path_cost(path);
        let optimization_strategy = if can_use_sparql && complexity <= 5 {
            PathOptimizationStrategy::SparqlPath
        } else if complexity > 50 {
            PathOptimizationStrategy::Programmatic
        } else {
            PathOptimizationStrategy::Hybrid
        };
        OptimizedPropertyPath {
            original_path: path.clone(),
            optimization_strategy,
            estimated_complexity: complexity,
            estimated_cost,
            can_use_sparql_path: can_use_sparql,
        }
    }
    /// Generate an optimized SPARQL query for property path evaluation
    pub fn generate_optimized_sparql_query(
        &self,
        start_node: &Term,
        path: &PropertyPath,
        graph_name: Option<&str>,
        optimization_hints: &PathOptimizationHints,
    ) -> Result<String> {
        let optimized_path = self.optimize_path(path);
        match optimized_path.optimization_strategy {
            PathOptimizationStrategy::SparqlPath => self.generate_native_sparql_path_query(
                start_node,
                path,
                graph_name,
                optimization_hints,
            ),
            PathOptimizationStrategy::Programmatic => {
                self.generate_fallback_sparql_query(start_node, path, graph_name)
            }
            PathOptimizationStrategy::Hybrid => {
                self.generate_hybrid_sparql_query(start_node, path, graph_name, optimization_hints)
            }
        }
    }
    /// Generate query plan for complex property path evaluation
    pub fn generate_query_plan(
        &self,
        start_node: &Term,
        path: &PropertyPath,
        graph_name: Option<&str>,
        optimization_hints: &PathOptimizationHints,
    ) -> Result<PropertyPathQueryPlan> {
        let optimized_path = self.optimize_path(path);
        let query =
            self.generate_optimized_sparql_query(start_node, path, graph_name, optimization_hints)?;
        let execution_strategy = match optimized_path.optimization_strategy {
            PathOptimizationStrategy::SparqlPath => PathExecutionStrategy::DirectSparql,
            PathOptimizationStrategy::Programmatic => PathExecutionStrategy::Programmatic,
            PathOptimizationStrategy::Hybrid => PathExecutionStrategy::HybridExecution,
        };
        Ok(PropertyPathQueryPlan {
            query,
            execution_strategy,
            estimated_cost: optimized_path.estimated_cost,
            estimated_complexity: optimized_path.estimated_complexity,
            optimization_hints: optimization_hints.clone(),
            cache_key: self.create_cache_key(start_node, path, graph_name),
        })
    }
    /// Estimate the cost of evaluating a property path
    #[allow(clippy::only_used_in_recursion)]
    pub(super) fn estimate_path_cost(&self, path: &PropertyPath) -> f64 {
        match path {
            PropertyPath::Predicate(_) => 1.0,
            PropertyPath::Inverse(inner) => inner.complexity() as f64 * 1.5,
            PropertyPath::Sequence(paths) => {
                paths
                    .iter()
                    .map(|p| self.estimate_path_cost(p))
                    .sum::<f64>()
                    * 1.2
            }
            PropertyPath::Alternative(paths) => {
                paths
                    .iter()
                    .map(|p| self.estimate_path_cost(p))
                    .sum::<f64>()
                    * 0.8
            }
            PropertyPath::ZeroOrMore(_) => 100.0,
            PropertyPath::OneOrMore(_) => 80.0,
            PropertyPath::ZeroOrOne(inner) => self.estimate_path_cost(inner) * 1.1,
        }
    }
}
