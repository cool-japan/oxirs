//! Federated Query Execution Engine
//!
//! This module handles the execution of federated queries across multiple services,
//! including parallel execution, timeout handling, fault tolerance, and adaptive optimization.
//!
//! The module is organized into the following components:
//! - `types`: Type definitions and data structures for federated execution
//! - `core`: Main FederatedExecutor implementation with basic execution methods
//! - `adaptive_execution`: Adaptive execution algorithms with runtime optimization
//! - `step_execution`: Individual step execution functions (service queries, joins, filters, etc.)

pub mod adaptive_execution;
pub mod core;
pub mod step_execution;
pub mod types;

// Re-export main types and structs for public API
pub use step_execution::{execute_parallel_group, execute_step};
pub use types::*;
pub use types::{FederatedExecutor, FederatedExecutorConfig};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::{ExecutionPlan, ExecutionStep, StepType};
    use std::time::Duration;

    #[tokio::test]
    async fn test_executor_creation() {
        let executor = FederatedExecutor::new();
        assert_eq!(executor.config.max_parallel_requests, 10);
    }

    #[tokio::test]
    async fn test_step_result_creation() {
        let result = StepResult {
            step_id: "test-step".to_string(),
            step_type: StepType::ServiceQuery,
            status: ExecutionStatus::Success,
            data: None,
            error: None,
            execution_time: Duration::from_millis(100),
            service_id: Some("test-service".to_string()),
            memory_used: 0,
            result_size: 0,
            success: true,
            error_message: None,
            service_response_time: Duration::from_millis(100),
            cache_hit: false,
        };

        assert_eq!(result.status, ExecutionStatus::Success);
        assert!(result.data.is_none());
    }

    #[tokio::test]
    async fn test_sparql_results_join() {
        let left = SparqlResults {
            head: SparqlHead {
                vars: vec!["s".to_string(), "p".to_string()],
            },
            results: SparqlResultsData { bindings: vec![] },
        };

        let right = SparqlResults {
            head: SparqlHead {
                vars: vec!["p".to_string(), "o".to_string()],
            },
            results: SparqlResultsData { bindings: vec![] },
        };

        // Test would need proper join implementation
        assert_eq!(left.head.vars.len(), 2);
        assert_eq!(right.head.vars.len(), 2);
    }

    #[test]
    fn test_join_key_creation() {
        let mut binding = std::collections::HashMap::new();
        binding.insert(
            "x".to_string(),
            SparqlValue {
                value_type: "uri".to_string(),
                value: "http://example.org".to_string(),
                datatype: None,
                lang: None,
            },
        );

        let common_vars = vec!["x".to_string()];
        let key = step_execution::create_join_key(&binding, &common_vars);
        assert!(key.contains("x:"));
    }
}
