//! SERVICE Clause Optimization and Rewriting
//!
//! This module implements advanced optimization techniques for SPARQL SERVICE clauses,
//! including query pushdown, filter propagation, and intelligent service selection.
//!
//! The module is organized into the following components:
//! - `core`: Main ServiceOptimizer implementation with core optimization methods
//! - `source_selection`: Advanced source selection algorithms including pattern coverage analysis
//! - `cost_analysis`: Advanced cost-based selection algorithms with ML-based estimation
//! - `types`: Type definitions and data structures used throughout the module
//! - `enhanced_optimizer`: ML-driven enhanced optimizer with advanced pattern analysis

pub mod core;
pub mod cost_analysis;
pub mod enhanced_optimizer;
pub mod source_selection;
pub mod types;

// Re-export main types and structs for public API
pub use core::ServiceOptimizer;
pub use types::*;

// Re-export for use in other modules
pub use crate::planner::{
    FilterExpression as OptimizerFilterExpression, TriplePattern as OptimizerTriplePattern,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_service_optimizer_creation() {
        let optimizer = ServiceOptimizer::new();
        assert!(optimizer.config().enable_pattern_grouping);
    }

    #[test]
    fn test_pattern_variable_extraction() {
        let optimizer = ServiceOptimizer::new();
        let pattern = crate::planner::TriplePattern {
            subject: Some("?s".to_string()),
            predicate: Some("rdf:type".to_string()),
            object: Some("?o".to_string()),
            pattern_string: "?s rdf:type ?o".to_string(),
        };

        let vars = optimizer.extract_pattern_variables(&pattern);
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("?s"));
        assert!(vars.contains("?o"));
    }

    #[test]
    fn test_execution_strategy_determination() {
        let optimizer = ServiceOptimizer::new();

        // Single service
        let services = vec![OptimizedServiceClause {
            service_id: "test".to_string(),
            endpoint: "http://example.com/sparql".to_string(),
            patterns: vec![],
            filters: vec![],
            pushed_filters: vec![],
            strategy: ServiceExecutionStrategy {
                use_values_binding: false,
                stream_results: false,
                use_subqueries: false,
                batch_size: 1000,
                timeout_ms: 30000,
            },
            estimated_cost: 100.0,
            capabilities: HashSet::new(),
        }];

        let joins = vec![];
        let strategy = optimizer.determine_execution_strategy(&services, &joins);
        assert_eq!(strategy, ExecutionStrategy::Sequential);
    }
}
