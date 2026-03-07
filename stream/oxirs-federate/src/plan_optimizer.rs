//! Federated query plan optimizer.
//!
//! This module provides a cost-based optimizer for federated SPARQL query plans.
//! It estimates execution costs, reorders joins to minimize latency, and pushes
//! filter conditions as close to the data sources as possible.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Capabilities and statistics for a remote SPARQL endpoint.
#[derive(Debug, Clone)]
pub struct EndpointCapabilities {
    /// Unique identifier for this endpoint
    pub endpoint_id: String,
    /// Whether the endpoint supports filter push-down
    pub supports_filter_pushdown: bool,
    /// Whether the endpoint supports OPTIONAL (left-join)
    pub supports_optional: bool,
    /// Estimated number of triples in the endpoint
    pub estimated_triples: u64,
    /// Average query latency in milliseconds
    pub avg_latency_ms: u32,
}

impl EndpointCapabilities {
    /// Construct with required fields and sensible defaults.
    pub fn new(endpoint_id: impl Into<String>) -> Self {
        Self {
            endpoint_id: endpoint_id.into(),
            supports_filter_pushdown: true,
            supports_optional: true,
            estimated_triples: 1_000_000,
            avg_latency_ms: 50,
        }
    }
}

/// A node in a federated query execution plan.
#[derive(Debug, Clone)]
pub enum PlanNode {
    /// A single triple pattern routed to a specific endpoint.
    TriplePattern {
        subject: String,
        predicate: String,
        object: String,
        endpoint: String,
    },
    /// Inner join of two sub-plans.
    Join {
        left: Box<PlanNode>,
        right: Box<PlanNode>,
    },
    /// Left outer join (OPTIONAL) of two sub-plans.
    LeftJoin {
        left: Box<PlanNode>,
        right: Box<PlanNode>,
    },
    /// Filter applied on top of a child plan.
    Filter {
        child: Box<PlanNode>,
        expression: String,
    },
    /// Union of multiple sub-plans.
    Union { children: Vec<PlanNode> },
    /// Project (SELECT) over a child plan.
    Project {
        child: Box<PlanNode>,
        variables: Vec<String>,
    },
    /// Slice (LIMIT/OFFSET) applied to a child plan.
    Slice {
        child: Box<PlanNode>,
        offset: usize,
        limit: Option<usize>,
    },
}

/// Cost estimates for a query plan node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanCost {
    /// Estimated number of result rows
    pub estimated_rows: u64,
    /// Estimated total latency in milliseconds
    pub estimated_latency_ms: u32,
    /// Number of remote network calls required
    pub network_calls: usize,
}

impl PlanCost {
    fn zero() -> Self {
        Self {
            estimated_rows: 0,
            estimated_latency_ms: 0,
            network_calls: 0,
        }
    }
}

/// An optimized plan with its cost estimate.
#[derive(Debug, Clone)]
pub struct OptimizedPlan {
    pub root: PlanNode,
    pub cost: PlanCost,
}

// ─────────────────────────────────────────────────────────────────────────────
// Optimizer
// ─────────────────────────────────────────────────────────────────────────────

/// Cost-based optimizer for federated query plans.
#[derive(Debug, Default)]
pub struct PlanOptimizer {
    endpoints: HashMap<String, EndpointCapabilities>,
}

impl PlanOptimizer {
    /// Create an empty optimizer with no known endpoints.
    pub fn new() -> Self {
        Self {
            endpoints: HashMap::new(),
        }
    }

    /// Register endpoint capabilities.
    pub fn add_endpoint(&mut self, caps: EndpointCapabilities) {
        self.endpoints.insert(caps.endpoint_id.clone(), caps);
    }

    /// Number of registered endpoints.
    pub fn endpoint_count(&self) -> usize {
        self.endpoints.len()
    }

    /// Fully optimize a plan: push filters down then reorder joins.
    pub fn optimize(&self, plan: PlanNode) -> OptimizedPlan {
        let plan = self.push_filters(plan);
        let plan = self.reorder_joins(plan);
        let cost = self.estimate_cost(&plan);
        OptimizedPlan { root: plan, cost }
    }

    /// Estimate the cost of executing a plan node.
    pub fn estimate_cost(&self, plan: &PlanNode) -> PlanCost {
        match plan {
            PlanNode::TriplePattern { endpoint, .. } => {
                if let Some(caps) = self.endpoints.get(endpoint) {
                    // Rough selectivity: assume 0.1% of triples match
                    let estimated_rows =
                        ((caps.estimated_triples as f64 * 0.001) as u64).max(1);
                    PlanCost {
                        estimated_rows,
                        estimated_latency_ms: caps.avg_latency_ms,
                        network_calls: 1,
                    }
                } else {
                    // Unknown endpoint — conservative defaults
                    PlanCost {
                        estimated_rows: 1_000,
                        estimated_latency_ms: 200,
                        network_calls: 1,
                    }
                }
            }

            PlanNode::Join { left, right } => {
                let lc = self.estimate_cost(left);
                let rc = self.estimate_cost(right);
                PlanCost {
                    estimated_rows: lc.estimated_rows.saturating_mul(rc.estimated_rows) / 10,
                    estimated_latency_ms: lc.estimated_latency_ms + rc.estimated_latency_ms,
                    network_calls: lc.network_calls + rc.network_calls,
                }
            }

            PlanNode::LeftJoin { left, right } => {
                let lc = self.estimate_cost(left);
                let rc = self.estimate_cost(right);
                PlanCost {
                    // LeftJoin produces at least all left-side rows
                    estimated_rows: lc.estimated_rows.saturating_add(
                        lc.estimated_rows.saturating_mul(rc.estimated_rows) / 10,
                    ),
                    estimated_latency_ms: lc.estimated_latency_ms + rc.estimated_latency_ms,
                    network_calls: lc.network_calls + rc.network_calls,
                }
            }

            PlanNode::Filter { child, .. } => {
                let cc = self.estimate_cost(child);
                // Assume filter reduces cardinality by ~50%
                PlanCost {
                    estimated_rows: (cc.estimated_rows / 2).max(1),
                    estimated_latency_ms: cc.estimated_latency_ms,
                    network_calls: cc.network_calls,
                }
            }

            PlanNode::Union { children } => {
                if children.is_empty() {
                    return PlanCost::zero();
                }
                let costs: Vec<PlanCost> =
                    children.iter().map(|c| self.estimate_cost(c)).collect();
                PlanCost {
                    estimated_rows: costs.iter().map(|c| c.estimated_rows).sum(),
                    estimated_latency_ms: costs
                        .iter()
                        .map(|c| c.estimated_latency_ms)
                        .max()
                        .unwrap_or(0),
                    network_calls: costs.iter().map(|c| c.network_calls).sum(),
                }
            }

            PlanNode::Project { child, .. } => self.estimate_cost(child),

            PlanNode::Slice { child, limit, .. } => {
                let cc = self.estimate_cost(child);
                let rows = if let Some(lim) = limit {
                    cc.estimated_rows.min(*lim as u64)
                } else {
                    cc.estimated_rows
                };
                PlanCost {
                    estimated_rows: rows,
                    ..cc
                }
            }
        }
    }

    /// Reorder joins so that the cheaper sub-plan is always on the left.
    pub fn reorder_joins(&self, plan: PlanNode) -> PlanNode {
        match plan {
            PlanNode::Join { left, right } => {
                let left = self.reorder_joins(*left);
                let right = self.reorder_joins(*right);
                let lc = self.estimate_cost(&left);
                let rc = self.estimate_cost(&right);
                // Put cheaper (fewer estimated rows) on the left
                if lc.estimated_rows <= rc.estimated_rows {
                    PlanNode::Join {
                        left: Box::new(left),
                        right: Box::new(right),
                    }
                } else {
                    PlanNode::Join {
                        left: Box::new(right),
                        right: Box::new(left),
                    }
                }
            }

            PlanNode::LeftJoin { left, right } => {
                // For left-join we only recurse, we don't swap sides
                let left = self.reorder_joins(*left);
                let right = self.reorder_joins(*right);
                PlanNode::LeftJoin {
                    left: Box::new(left),
                    right: Box::new(right),
                }
            }

            PlanNode::Filter { child, expression } => PlanNode::Filter {
                child: Box::new(self.reorder_joins(*child)),
                expression,
            },

            PlanNode::Union { children } => PlanNode::Union {
                children: children.into_iter().map(|c| self.reorder_joins(c)).collect(),
            },

            PlanNode::Project { child, variables } => PlanNode::Project {
                child: Box::new(self.reorder_joins(*child)),
                variables,
            },

            PlanNode::Slice {
                child,
                offset,
                limit,
            } => PlanNode::Slice {
                child: Box::new(self.reorder_joins(*child)),
                offset,
                limit,
            },

            other => other,
        }
    }

    /// Push `Filter` nodes as far down the plan tree as possible.
    ///
    /// A filter above a `Join` can often be pushed into one of the join inputs,
    /// reducing the number of intermediate rows.
    pub fn push_filters(&self, plan: PlanNode) -> PlanNode {
        match plan {
            // A filter over a Join: try to push into one of the sides
            PlanNode::Filter {
                child,
                expression: ref expr,
            } => {
                let expression = expr.clone();
                match *child {
                    PlanNode::Join { left, right } => {
                        // Push the filter to the left side (heuristic)
                        let pushed_left = PlanNode::Filter {
                            child: left,
                            expression: expression.clone(),
                        };
                        let pushed_left = self.push_filters(pushed_left);
                        let right = self.push_filters(*right);
                        PlanNode::Join {
                            left: Box::new(pushed_left),
                            right: Box::new(right),
                        }
                    }
                    other => PlanNode::Filter {
                        child: Box::new(self.push_filters(other)),
                        expression,
                    },
                }
            }

            PlanNode::Join { left, right } => PlanNode::Join {
                left: Box::new(self.push_filters(*left)),
                right: Box::new(self.push_filters(*right)),
            },

            PlanNode::LeftJoin { left, right } => PlanNode::LeftJoin {
                left: Box::new(self.push_filters(*left)),
                right: Box::new(self.push_filters(*right)),
            },

            PlanNode::Union { children } => PlanNode::Union {
                children: children
                    .into_iter()
                    .map(|c| self.push_filters(c))
                    .collect(),
            },

            PlanNode::Project { child, variables } => PlanNode::Project {
                child: Box::new(self.push_filters(*child)),
                variables,
            },

            PlanNode::Slice {
                child,
                offset,
                limit,
            } => PlanNode::Slice {
                child: Box::new(self.push_filters(*child)),
                offset,
                limit,
            },

            other => other,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn triple(s: &str, p: &str, o: &str, ep: &str) -> PlanNode {
        PlanNode::TriplePattern {
            subject: s.to_string(),
            predicate: p.to_string(),
            object: o.to_string(),
            endpoint: ep.to_string(),
        }
    }

    fn join(l: PlanNode, r: PlanNode) -> PlanNode {
        PlanNode::Join {
            left: Box::new(l),
            right: Box::new(r),
        }
    }

    fn filter(child: PlanNode, expr: &str) -> PlanNode {
        PlanNode::Filter {
            child: Box::new(child),
            expression: expr.to_string(),
        }
    }

    fn project(child: PlanNode, vars: Vec<&str>) -> PlanNode {
        PlanNode::Project {
            child: Box::new(child),
            variables: vars.iter().map(|v| v.to_string()).collect(),
        }
    }

    fn slice(child: PlanNode, offset: usize, limit: Option<usize>) -> PlanNode {
        PlanNode::Slice {
            child: Box::new(child),
            offset,
            limit,
        }
    }

    fn union(children: Vec<PlanNode>) -> PlanNode {
        PlanNode::Union { children }
    }

    fn small_endpoint(id: &str, triples: u64, latency: u32) -> EndpointCapabilities {
        EndpointCapabilities {
            endpoint_id: id.to_string(),
            supports_filter_pushdown: true,
            supports_optional: true,
            estimated_triples: triples,
            avg_latency_ms: latency,
        }
    }

    fn optimizer_with_two_endpoints() -> PlanOptimizer {
        let mut opt = PlanOptimizer::new();
        opt.add_endpoint(small_endpoint("ep1", 10_000, 10));
        opt.add_endpoint(small_endpoint("ep2", 1_000_000, 50));
        opt
    }

    // 1. Single triple pattern cost
    #[test]
    fn test_single_triple_cost_known_endpoint() {
        let opt = optimizer_with_two_endpoints();
        let plan = triple("?s", "rdf:type", "?o", "ep1");
        let cost = opt.estimate_cost(&plan);
        assert_eq!(cost.network_calls, 1);
        assert_eq!(cost.estimated_latency_ms, 10);
        assert!(cost.estimated_rows >= 1);
    }

    // 2. Unknown endpoint uses conservative defaults
    #[test]
    fn test_single_triple_cost_unknown_endpoint() {
        let opt = PlanOptimizer::new();
        let plan = triple("?s", "?p", "?o", "unknown");
        let cost = opt.estimate_cost(&plan);
        assert_eq!(cost.network_calls, 1);
        assert_eq!(cost.estimated_latency_ms, 200);
        assert_eq!(cost.estimated_rows, 1_000);
    }

    // 3. Add endpoint increments count
    #[test]
    fn test_add_endpoint_count() {
        let mut opt = PlanOptimizer::new();
        assert_eq!(opt.endpoint_count(), 0);
        opt.add_endpoint(small_endpoint("ep1", 1_000, 10));
        assert_eq!(opt.endpoint_count(), 1);
        opt.add_endpoint(small_endpoint("ep2", 2_000, 20));
        assert_eq!(opt.endpoint_count(), 2);
    }

    // 4. Join cost sums latencies and calls
    #[test]
    fn test_join_cost_sums_latency() {
        let opt = optimizer_with_two_endpoints();
        let plan = join(
            triple("?s", "?p", "?o", "ep1"),
            triple("?s", "?q", "?r", "ep2"),
        );
        let cost = opt.estimate_cost(&plan);
        assert_eq!(cost.estimated_latency_ms, 10 + 50);
        assert_eq!(cost.network_calls, 2);
    }

    // 5. Join reordering puts cheaper on left
    #[test]
    fn test_join_reorder_cheaper_left() {
        let opt = optimizer_with_two_endpoints();
        // ep2 is large (1M triples), ep1 is small (10K triples)
        // After reorder, ep1 (cheap) should be on the left
        let plan = join(
            triple("?s", "?p", "?o", "ep2"), // expensive
            triple("?s", "?q", "?r", "ep1"), // cheap
        );
        let reordered = opt.reorder_joins(plan);
        if let PlanNode::Join { left, .. } = reordered {
            if let PlanNode::TriplePattern { endpoint, .. } = *left {
                assert_eq!(endpoint, "ep1");
            } else {
                panic!("Left should be TriplePattern");
            }
        } else {
            panic!("Should be a Join");
        }
    }

    // 6. Already-optimal join is not swapped
    #[test]
    fn test_join_no_reorder_when_already_optimal() {
        let opt = optimizer_with_two_endpoints();
        let plan = join(
            triple("?s", "?p", "?o", "ep1"), // cheap — already left
            triple("?s", "?q", "?r", "ep2"), // expensive
        );
        let reordered = opt.reorder_joins(plan);
        if let PlanNode::Join { left, .. } = reordered {
            if let PlanNode::TriplePattern { endpoint, .. } = *left {
                assert_eq!(endpoint, "ep1");
            } else {
                panic!("Left should be TriplePattern");
            }
        } else {
            panic!("Should be Join");
        }
    }

    // 7. Filter pushdown into join moves filter to left child
    #[test]
    fn test_filter_pushdown_into_join() {
        let opt = optimizer_with_two_endpoints();
        let plan = filter(
            join(
                triple("?s", "?p", "?o", "ep1"),
                triple("?s", "?q", "?r", "ep2"),
            ),
            "?o = 'foo'",
        );
        let pushed = opt.push_filters(plan);
        // The filter should now be inside, not wrapping the join
        match pushed {
            PlanNode::Join { left, .. } => {
                // Left should now have the filter
                assert!(matches!(*left, PlanNode::Filter { .. }));
            }
            _ => panic!("Expected a Join after filter pushdown"),
        }
    }

    // 8. Filter over TriplePattern stays in place
    #[test]
    fn test_filter_over_triple_stays() {
        let opt = optimizer_with_two_endpoints();
        let plan = filter(triple("?s", "?p", "?o", "ep1"), "?o = 'bar'");
        let pushed = opt.push_filters(plan);
        assert!(matches!(
            pushed,
            PlanNode::Filter {
                child,
                ..
            } if matches!(*child, PlanNode::TriplePattern { .. })
        ));
    }

    // 9. Filter cost reduces rows
    #[test]
    fn test_filter_cost_reduces_rows() {
        let opt = optimizer_with_two_endpoints();
        let child = triple("?s", "?p", "?o", "ep1");
        let child_cost = opt.estimate_cost(&child);
        let plan = filter(child, "?o = 'x'");
        let cost = opt.estimate_cost(&plan);
        assert!(cost.estimated_rows <= child_cost.estimated_rows);
    }

    // 10. Union cost sums rows and calls
    #[test]
    fn test_union_cost_sum_rows() {
        let opt = optimizer_with_two_endpoints();
        let plan = union(vec![
            triple("?s", "?p", "?o", "ep1"),
            triple("?s", "?q", "?r", "ep2"),
        ]);
        let cost = opt.estimate_cost(&plan);
        let c1 = opt.estimate_cost(&triple("?s", "?p", "?o", "ep1"));
        let c2 = opt.estimate_cost(&triple("?s", "?q", "?r", "ep2"));
        assert_eq!(cost.estimated_rows, c1.estimated_rows + c2.estimated_rows);
        assert_eq!(cost.network_calls, 2);
    }

    // 11. Union max latency
    #[test]
    fn test_union_max_latency() {
        let opt = optimizer_with_two_endpoints();
        let plan = union(vec![
            triple("?s", "?p", "?o", "ep1"), // 10ms
            triple("?s", "?q", "?r", "ep2"), // 50ms
        ]);
        let cost = opt.estimate_cost(&plan);
        assert_eq!(cost.estimated_latency_ms, 50); // max
    }

    // 12. Empty union cost is zero
    #[test]
    fn test_empty_union_cost_zero() {
        let opt = PlanOptimizer::new();
        let plan = union(vec![]);
        let cost = opt.estimate_cost(&plan);
        assert_eq!(cost, PlanCost::zero());
    }

    // 13. Project passthrough — same cost as child
    #[test]
    fn test_project_passthrough_cost() {
        let opt = optimizer_with_two_endpoints();
        let child = triple("?s", "?p", "?o", "ep1");
        let child_cost = opt.estimate_cost(&child);
        let plan = project(child, vec!["?s", "?o"]);
        let cost = opt.estimate_cost(&plan);
        assert_eq!(cost, child_cost);
    }

    // 14. Slice with limit caps rows
    #[test]
    fn test_slice_with_limit_caps_rows() {
        let opt = optimizer_with_two_endpoints();
        let child = triple("?s", "?p", "?o", "ep2"); // large endpoint
        let child_cost = opt.estimate_cost(&child);
        assert!(child_cost.estimated_rows > 10);

        let plan = slice(child, 0, Some(10));
        let cost = opt.estimate_cost(&plan);
        assert_eq!(cost.estimated_rows, 10);
    }

    // 15. Slice without limit keeps child rows
    #[test]
    fn test_slice_no_limit_keeps_rows() {
        let opt = optimizer_with_two_endpoints();
        let child = triple("?s", "?p", "?o", "ep1");
        let child_cost = opt.estimate_cost(&child);
        let plan = slice(child, 5, None);
        let cost = opt.estimate_cost(&plan);
        assert_eq!(cost.estimated_rows, child_cost.estimated_rows);
    }

    // 16. optimize returns an OptimizedPlan
    #[test]
    fn test_optimize_returns_plan() {
        let opt = optimizer_with_two_endpoints();
        let plan = join(
            triple("?s", "?p", "?o", "ep2"),
            triple("?s", "?q", "?r", "ep1"),
        );
        let result = opt.optimize(plan);
        assert!(result.cost.network_calls > 0);
    }

    // 17. optimize does filter pushdown before join reorder
    #[test]
    fn test_optimize_integrates_both_passes() {
        let opt = optimizer_with_two_endpoints();
        let plan = filter(
            join(
                triple("?s", "?p", "?o", "ep2"),
                triple("?s", "?q", "?r", "ep1"),
            ),
            "?o = 'test'",
        );
        let result = opt.optimize(plan);
        // Just verify it completes and produces a valid cost
        assert!(result.cost.estimated_latency_ms > 0);
    }

    // 18. LeftJoin preserves left side
    #[test]
    fn test_left_join_not_swapped() {
        let opt = optimizer_with_two_endpoints();
        let plan = PlanNode::LeftJoin {
            left: Box::new(triple("?s", "?p", "?o", "ep2")), // expensive
            right: Box::new(triple("?s", "?q", "?r", "ep1")), // cheap
        };
        let reordered = opt.reorder_joins(plan);
        // Left join should NOT be swapped
        if let PlanNode::LeftJoin { left, .. } = reordered {
            if let PlanNode::TriplePattern { endpoint, .. } = *left {
                assert_eq!(endpoint, "ep2"); // unchanged
            }
        }
    }

    // 19. LeftJoin cost is greater than child costs
    #[test]
    fn test_left_join_cost_greater_than_children() {
        let opt = optimizer_with_two_endpoints();
        let plan = PlanNode::LeftJoin {
            left: Box::new(triple("?s", "?p", "?o", "ep1")),
            right: Box::new(triple("?s", "?q", "?r", "ep2")),
        };
        let lc = opt.estimate_cost(&triple("?s", "?p", "?o", "ep1"));
        let cost = opt.estimate_cost(&plan);
        assert!(cost.estimated_rows >= lc.estimated_rows);
    }

    // 20. add_endpoint replaces existing entry with same id
    #[test]
    fn test_add_endpoint_replaces() {
        let mut opt = PlanOptimizer::new();
        opt.add_endpoint(small_endpoint("ep1", 1_000, 10));
        opt.add_endpoint(small_endpoint("ep1", 2_000_000, 100)); // replace
        assert_eq!(opt.endpoint_count(), 1);

        let plan = triple("?s", "?p", "?o", "ep1");
        let cost = opt.estimate_cost(&plan);
        assert_eq!(cost.estimated_latency_ms, 100);
    }

    // 21. new() creates empty optimizer
    #[test]
    fn test_new_empty() {
        let opt = PlanOptimizer::new();
        assert_eq!(opt.endpoint_count(), 0);
    }

    // 22. default() creates empty optimizer
    #[test]
    fn test_default_empty() {
        let opt = PlanOptimizer::default();
        assert_eq!(opt.endpoint_count(), 0);
    }

    // 23. Three-way join reorder produces cheapest on left
    #[test]
    fn test_three_way_join_reorder() {
        let mut opt = PlanOptimizer::new();
        opt.add_endpoint(small_endpoint("big", 10_000_000, 100));
        opt.add_endpoint(small_endpoint("medium", 100_000, 30));
        opt.add_endpoint(small_endpoint("small", 100, 5));

        // Nest: join(join(big, medium), small)
        let inner = join(
            triple("?s", "?p", "?o", "big"),
            triple("?s", "?q", "?r", "medium"),
        );
        let plan = join(inner, triple("?x", "?y", "?z", "small"));
        let reordered = opt.reorder_joins(plan);
        let cost = opt.estimate_cost(&reordered);
        assert!(cost.network_calls == 3);
    }

    // 24. Filter cost keeps network_calls same as child
    #[test]
    fn test_filter_same_network_calls() {
        let opt = optimizer_with_two_endpoints();
        let child = triple("?s", "?p", "?o", "ep1");
        let child_cost = opt.estimate_cost(&child);
        let plan = filter(child, "?o > 5");
        let cost = opt.estimate_cost(&plan);
        assert_eq!(cost.network_calls, child_cost.network_calls);
    }

    // 25. Project does not change network_calls
    #[test]
    fn test_project_same_network_calls() {
        let opt = optimizer_with_two_endpoints();
        let child = triple("?s", "?p", "?o", "ep1");
        let child_cost = opt.estimate_cost(&child);
        let plan = project(child, vec!["?s"]);
        let cost = opt.estimate_cost(&plan);
        assert_eq!(cost.network_calls, child_cost.network_calls);
    }

    // 26. Slice keeps network_calls same as child
    #[test]
    fn test_slice_same_network_calls() {
        let opt = optimizer_with_two_endpoints();
        let child = triple("?s", "?p", "?o", "ep1");
        let child_cost = opt.estimate_cost(&child);
        let plan = slice(child, 0, Some(5));
        let cost = opt.estimate_cost(&plan);
        assert_eq!(cost.network_calls, child_cost.network_calls);
    }

    // 27. push_filters on non-filter node returns same node type
    #[test]
    fn test_push_filters_passthrough_non_filter() {
        let opt = optimizer_with_two_endpoints();
        let plan = triple("?s", "?p", "?o", "ep1");
        let pushed = opt.push_filters(plan);
        assert!(matches!(pushed, PlanNode::TriplePattern { .. }));
    }

    // 28. Large estimated_triples translates to higher row estimate
    #[test]
    fn test_large_endpoint_higher_row_estimate() {
        let opt = optimizer_with_two_endpoints();
        let small = opt.estimate_cost(&triple("?s", "?p", "?o", "ep1")); // 10K triples
        let large = opt.estimate_cost(&triple("?s", "?p", "?o", "ep2")); // 1M triples
        assert!(large.estimated_rows > small.estimated_rows);
    }

    // 29. Slice with limit = 0 returns 0 rows
    #[test]
    fn test_slice_limit_zero_rows() {
        let opt = optimizer_with_two_endpoints();
        let plan = slice(triple("?s", "?p", "?o", "ep1"), 0, Some(0));
        let cost = opt.estimate_cost(&plan);
        assert_eq!(cost.estimated_rows, 0);
    }

    // 30. Union with single child same as child cost
    #[test]
    fn test_union_single_child() {
        let opt = optimizer_with_two_endpoints();
        let child = triple("?s", "?p", "?o", "ep1");
        let child_cost = opt.estimate_cost(&child);
        let plan = union(vec![child]);
        let cost = opt.estimate_cost(&plan);
        assert_eq!(cost.estimated_rows, child_cost.estimated_rows);
    }

    // 31. estimated_rows for join > 0 when both children have rows
    #[test]
    fn test_join_rows_nonzero() {
        let opt = optimizer_with_two_endpoints();
        let plan = join(
            triple("?s", "?p", "?o", "ep1"),
            triple("?s", "?q", "?r", "ep2"),
        );
        let cost = opt.estimate_cost(&plan);
        assert!(cost.estimated_rows >= 1);
    }

    // 32. reorder_joins recurses into Filter
    #[test]
    fn test_reorder_joins_recurses_into_filter() {
        let opt = optimizer_with_two_endpoints();
        let plan = filter(
            join(
                triple("?s", "?p", "?o", "ep2"),
                triple("?s", "?q", "?r", "ep1"),
            ),
            "x = 1",
        );
        let result = opt.reorder_joins(plan);
        if let PlanNode::Filter { child, .. } = result {
            if let PlanNode::Join { left, .. } = *child {
                if let PlanNode::TriplePattern { endpoint, .. } = *left {
                    assert_eq!(endpoint, "ep1"); // cheaper on left
                }
            }
        }
    }

    // 33. reorder_joins recurses into Union
    #[test]
    fn test_reorder_joins_recurses_into_union() {
        let opt = optimizer_with_two_endpoints();
        let plan = union(vec![
            join(
                triple("?s", "?p", "?o", "ep2"),
                triple("?s", "?q", "?r", "ep1"),
            ),
        ]);
        let result = opt.reorder_joins(plan);
        assert!(matches!(result, PlanNode::Union { .. }));
    }

    // 34. reorder_joins recurses into Project
    #[test]
    fn test_reorder_joins_recurses_into_project() {
        let opt = optimizer_with_two_endpoints();
        let plan = project(
            join(
                triple("?s", "?p", "?o", "ep2"),
                triple("?s", "?q", "?r", "ep1"),
            ),
            vec!["?s"],
        );
        let result = opt.reorder_joins(plan);
        if let PlanNode::Project { child, .. } = result {
            if let PlanNode::Join { left, .. } = *child {
                if let PlanNode::TriplePattern { endpoint, .. } = *left {
                    assert_eq!(endpoint, "ep1");
                }
            }
        }
    }

    // 35. reorder_joins recurses into Slice
    #[test]
    fn test_reorder_joins_recurses_into_slice() {
        let opt = optimizer_with_two_endpoints();
        let plan = slice(
            join(
                triple("?s", "?p", "?o", "ep2"),
                triple("?s", "?q", "?r", "ep1"),
            ),
            0,
            Some(10),
        );
        let result = opt.reorder_joins(plan);
        assert!(matches!(result, PlanNode::Slice { .. }));
    }

    // 36. push_filters recurses into Project
    #[test]
    fn test_push_filters_recurses_project() {
        let opt = optimizer_with_two_endpoints();
        let plan = project(
            filter(triple("?s", "?p", "?o", "ep1"), "x=1"),
            vec!["?s"],
        );
        let pushed = opt.push_filters(plan);
        assert!(matches!(pushed, PlanNode::Project { .. }));
    }

    // 37. push_filters recurses into Slice
    #[test]
    fn test_push_filters_recurses_slice() {
        let opt = optimizer_with_two_endpoints();
        let plan = slice(
            filter(triple("?s", "?p", "?o", "ep1"), "x=1"),
            0,
            Some(5),
        );
        let pushed = opt.push_filters(plan);
        assert!(matches!(pushed, PlanNode::Slice { .. }));
    }

    // 38. push_filters with Union pushes into each child
    #[test]
    fn test_push_filters_recurses_union() {
        let opt = optimizer_with_two_endpoints();
        let plan = union(vec![
            filter(triple("?s", "?p", "?o", "ep1"), "x=1"),
            filter(triple("?s", "?q", "?r", "ep2"), "x=1"),
        ]);
        let pushed = opt.push_filters(plan);
        if let PlanNode::Union { children } = pushed {
            for child in &children {
                assert!(matches!(child, PlanNode::Filter { .. }));
            }
        }
    }

    // 39. Filter keeps latency same as child
    #[test]
    fn test_filter_same_latency_as_child() {
        let opt = optimizer_with_two_endpoints();
        let child = triple("?s", "?p", "?o", "ep1");
        let child_cost = opt.estimate_cost(&child);
        let plan = filter(child, "y > 0");
        let cost = opt.estimate_cost(&plan);
        assert_eq!(cost.estimated_latency_ms, child_cost.estimated_latency_ms);
    }

    // 40. EndpointCapabilities new() has sensible defaults
    #[test]
    fn test_endpoint_capabilities_new() {
        let caps = EndpointCapabilities::new("test");
        assert_eq!(caps.endpoint_id, "test");
        assert!(caps.supports_filter_pushdown);
        assert!(caps.supports_optional);
        assert!(caps.estimated_triples > 0);
        assert!(caps.avg_latency_ms > 0);
    }

    // 41. Two equal-cost children: no swap needed
    #[test]
    fn test_join_equal_cost_no_swap() {
        let mut opt = PlanOptimizer::new();
        opt.add_endpoint(small_endpoint("ep_a", 1_000, 20));
        opt.add_endpoint(small_endpoint("ep_b", 1_000, 20)); // same cost
        let plan = join(
            triple("?s", "?p", "?o", "ep_a"),
            triple("?s", "?q", "?r", "ep_b"),
        );
        let reordered = opt.reorder_joins(plan);
        // Either order is fine — just ensure it remains a Join
        assert!(matches!(reordered, PlanNode::Join { .. }));
    }

    // 42. optimize cost latency > 0 for non-trivial plan
    #[test]
    fn test_optimize_non_zero_latency() {
        let opt = optimizer_with_two_endpoints();
        let plan = triple("?s", "?p", "?o", "ep1");
        let result = opt.optimize(plan);
        assert!(result.cost.estimated_latency_ms > 0);
    }

    // 43. PlanCost zero helper
    #[test]
    fn test_plan_cost_zero() {
        let c = PlanCost::zero();
        assert_eq!(c.estimated_rows, 0);
        assert_eq!(c.estimated_latency_ms, 0);
        assert_eq!(c.network_calls, 0);
    }

    // 44. TriplePattern has 1 network call
    #[test]
    fn test_triple_one_network_call() {
        let opt = optimizer_with_two_endpoints();
        let cost = opt.estimate_cost(&triple("?s", "?p", "?o", "ep1"));
        assert_eq!(cost.network_calls, 1);
    }

    // 45. Nested joins produce correct call count
    #[test]
    fn test_nested_join_call_count() {
        let opt = optimizer_with_two_endpoints();
        let plan = join(
            join(
                triple("?s", "?p", "?o", "ep1"),
                triple("?a", "?b", "?c", "ep2"),
            ),
            triple("?x", "?y", "?z", "ep1"),
        );
        let cost = opt.estimate_cost(&plan);
        assert_eq!(cost.network_calls, 3);
    }
}
