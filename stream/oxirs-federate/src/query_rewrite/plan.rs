//! Execution plan representation for federated query rewriting.
//!
//! This module defines the data structures that represent an optimized
//! federated execution plan after query decomposition and rewriting.
//! Plans capture the ordered set of subqueries, their inter-dependencies,
//! and the join strategy to be used when merging partial results.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

use super::decomposer::EndpointSubquery;

/// Strategy used to join results from multiple endpoint subqueries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JoinStrategy {
    /// Nested-loop join: for each result from the outer query, execute the inner query.
    NestedLoop,
    /// Hash join: build a hash table from the smaller result set and probe with the larger.
    Hash,
    /// Merge join: assume both inputs are sorted on the join variable.
    Merge,
    /// Bind join: push bindings from an upstream result as VALUES clause to the downstream endpoint.
    Bind,
    /// Broadcast join: replicate the smaller result to every endpoint.
    Broadcast,
}

impl std::fmt::Display for JoinStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JoinStrategy::NestedLoop => write!(f, "NestedLoop"),
            JoinStrategy::Hash => write!(f, "Hash"),
            JoinStrategy::Merge => write!(f, "Merge"),
            JoinStrategy::Bind => write!(f, "Bind"),
            JoinStrategy::Broadcast => write!(f, "Broadcast"),
        }
    }
}

/// A single node within the federated execution plan tree.
///
/// Leaf nodes correspond to individual endpoint subqueries while internal
/// nodes represent join operations over two child nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlanNode {
    /// Leaf: execute a subquery against a remote endpoint.
    Subquery(EndpointSubquery),
    /// Internal: join the results of two child plan nodes.
    Join {
        /// Left-hand child node.
        left: Box<PlanNode>,
        /// Right-hand child node.
        right: Box<PlanNode>,
        /// Variable(s) shared between left and right on which the join is performed.
        join_variables: Vec<String>,
        /// The chosen join strategy.
        strategy: JoinStrategy,
        /// Estimated cost of performing this join.
        estimated_cost: f64,
    },
    /// Union: merge results from two child nodes without joining on variables.
    Union {
        /// Left-hand child node.
        left: Box<PlanNode>,
        /// Right-hand child node.
        right: Box<PlanNode>,
    },
    /// Filter applied in-memory after retrieving results from a child node.
    Filter {
        /// The child node whose results are to be filtered.
        child: Box<PlanNode>,
        /// The SPARQL FILTER expression string.
        expression: String,
        /// Variables referenced in the filter expression.
        variables: Vec<String>,
    },
}

impl PlanNode {
    /// Recursively collect all leaf subqueries contained within this plan node.
    pub fn collect_subqueries(&self) -> Vec<&EndpointSubquery> {
        match self {
            PlanNode::Subquery(sq) => vec![sq],
            PlanNode::Join { left, right, .. } => {
                let mut result = left.collect_subqueries();
                result.extend(right.collect_subqueries());
                result
            }
            PlanNode::Union { left, right } => {
                let mut result = left.collect_subqueries();
                result.extend(right.collect_subqueries());
                result
            }
            PlanNode::Filter { child, .. } => child.collect_subqueries(),
        }
    }

    /// Compute the estimated total cost for this plan node, including all descendants.
    pub fn total_estimated_cost(&self) -> f64 {
        match self {
            PlanNode::Subquery(sq) => sq.priority,
            PlanNode::Join {
                left,
                right,
                estimated_cost,
                ..
            } => left.total_estimated_cost() + right.total_estimated_cost() + estimated_cost,
            PlanNode::Union { left, right } => {
                left.total_estimated_cost() + right.total_estimated_cost()
            }
            PlanNode::Filter { child, .. } => child.total_estimated_cost(),
        }
    }

    /// Return the number of leaf subquery nodes in this plan.
    pub fn subquery_count(&self) -> usize {
        self.collect_subqueries().len()
    }
}

/// Fully materialised federated execution plan produced by the optimizer.
///
/// The plan captures the complete tree of operations required to execute a
/// federated SPARQL query, together with planning metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// Unique identifier for this plan instance.
    pub plan_id: String,
    /// The original SPARQL query string that was decomposed.
    pub original_query: String,
    /// Root node of the physical plan tree.
    pub root: PlanNode,
    /// Wall-clock time spent generating this plan.
    pub planning_duration: Duration,
    /// Aggregate estimated cost across the entire plan.
    pub total_estimated_cost: f64,
    /// Maximum degree of parallelism available at the root level.
    pub max_parallelism: usize,
    /// Arbitrary key-value annotations attached during planning.
    pub annotations: HashMap<String, String>,
}

impl ExecutionPlan {
    /// Create a new execution plan with a freshly-generated unique identifier.
    pub fn new(
        original_query: impl Into<String>,
        root: PlanNode,
        planning_duration: Duration,
    ) -> Self {
        let total_estimated_cost = root.total_estimated_cost();
        let max_parallelism = root.subquery_count();
        Self {
            plan_id: Uuid::new_v4().to_string(),
            original_query: original_query.into(),
            root,
            planning_duration,
            total_estimated_cost,
            max_parallelism,
            annotations: HashMap::new(),
        }
    }

    /// Insert a key-value annotation into this plan.
    pub fn annotate(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.annotations.insert(key.into(), value.into());
    }

    /// Collect all leaf subqueries that will be dispatched to remote endpoints.
    pub fn endpoint_subqueries(&self) -> Vec<&EndpointSubquery> {
        self.root.collect_subqueries()
    }

    /// Return a human-readable summary of the plan.
    pub fn summary(&self) -> String {
        let subquery_count = self.max_parallelism;
        format!(
            "Plan[{}]: {} subqueries, estimated_cost={:.2}, parallelism={}",
            &self.plan_id[..8],
            subquery_count,
            self.total_estimated_cost,
            self.max_parallelism,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_rewrite::decomposer::EndpointSubquery;

    fn make_subquery(endpoint: &str, sparql: &str, priority: f64) -> EndpointSubquery {
        EndpointSubquery {
            endpoint_url: endpoint.to_string(),
            sparql: sparql.to_string(),
            estimated_results: 10,
            priority,
        }
    }

    #[test]
    fn test_leaf_node_cost() {
        let sq = make_subquery("http://ep1/sparql", "SELECT ?s WHERE { ?s ?p ?o }", 5.0);
        let node = PlanNode::Subquery(sq);
        assert!((node.total_estimated_cost() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_join_node_cost() {
        let left = PlanNode::Subquery(make_subquery(
            "http://ep1/sparql",
            "SELECT ?s ?p WHERE {?s ?p ?o}",
            3.0,
        ));
        let right = PlanNode::Subquery(make_subquery(
            "http://ep2/sparql",
            "SELECT ?s ?o WHERE {?s ?p ?o}",
            4.0,
        ));
        let join = PlanNode::Join {
            left: Box::new(left),
            right: Box::new(right),
            join_variables: vec!["?s".to_string()],
            strategy: JoinStrategy::Hash,
            estimated_cost: 2.0,
        };
        // 3 + 4 + 2 = 9
        assert!((join.total_estimated_cost() - 9.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_collect_subqueries() {
        let left = PlanNode::Subquery(make_subquery(
            "http://ep1/sparql",
            "SELECT ?s WHERE {?s a foaf:Person}",
            1.0,
        ));
        let right = PlanNode::Subquery(make_subquery(
            "http://ep2/sparql",
            "SELECT ?s WHERE {?s foaf:name ?n}",
            1.0,
        ));
        let join = PlanNode::Join {
            left: Box::new(left),
            right: Box::new(right),
            join_variables: vec!["?s".to_string()],
            strategy: JoinStrategy::Bind,
            estimated_cost: 0.5,
        };
        assert_eq!(join.subquery_count(), 2);
        assert_eq!(join.collect_subqueries().len(), 2);
    }

    #[test]
    fn test_execution_plan_creation() {
        let root = PlanNode::Subquery(make_subquery(
            "http://ep1/sparql",
            "SELECT * WHERE {?s ?p ?o}",
            10.0,
        ));
        let plan = ExecutionPlan::new(
            "SELECT * WHERE { ?s ?p ?o }",
            root,
            Duration::from_millis(5),
        );
        assert!(!plan.plan_id.is_empty());
        assert_eq!(plan.max_parallelism, 1);
        assert!((plan.total_estimated_cost - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_plan_summary_format() {
        let root = PlanNode::Subquery(make_subquery(
            "http://ep1/sparql",
            "SELECT ?s WHERE {?s ?p ?o}",
            7.5,
        ));
        let plan = ExecutionPlan::new("SELECT ?s WHERE {?s ?p ?o}", root, Duration::from_millis(2));
        let summary = plan.summary();
        assert!(summary.contains("subqueries"));
        assert!(summary.contains("estimated_cost"));
    }

    #[test]
    fn test_plan_annotations() {
        let root = PlanNode::Subquery(make_subquery("http://ep1/sparql", "ASK { ?s ?p ?o }", 1.0));
        let mut plan = ExecutionPlan::new("ASK { ?s ?p ?o }", root, Duration::from_millis(1));
        plan.annotate("optimizer_version", "v0.3.0");
        assert_eq!(
            plan.annotations.get("optimizer_version"),
            Some(&"v0.3.0".to_string())
        );
    }

    #[test]
    fn test_union_node() {
        let left = PlanNode::Subquery(make_subquery(
            "http://ep1/sparql",
            "SELECT ?s WHERE {?s a owl:Class}",
            2.0,
        ));
        let right = PlanNode::Subquery(make_subquery(
            "http://ep2/sparql",
            "SELECT ?s WHERE {?s a rdfs:Class}",
            2.0,
        ));
        let union_node = PlanNode::Union {
            left: Box::new(left),
            right: Box::new(right),
        };
        assert_eq!(union_node.subquery_count(), 2);
        assert!((union_node.total_estimated_cost() - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_filter_node() {
        let child = PlanNode::Subquery(make_subquery(
            "http://ep1/sparql",
            "SELECT ?s ?age WHERE {?s foaf:age ?age}",
            3.0,
        ));
        let filter = PlanNode::Filter {
            child: Box::new(child),
            expression: "FILTER(?age > 18)".to_string(),
            variables: vec!["?age".to_string()],
        };
        assert_eq!(filter.subquery_count(), 1);
        assert!((filter.total_estimated_cost() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_join_strategy_display() {
        assert_eq!(JoinStrategy::Hash.to_string(), "Hash");
        assert_eq!(JoinStrategy::Bind.to_string(), "Bind");
        assert_eq!(JoinStrategy::NestedLoop.to_string(), "NestedLoop");
        assert_eq!(JoinStrategy::Merge.to_string(), "Merge");
        assert_eq!(JoinStrategy::Broadcast.to_string(), "Broadcast");
    }
}
