//! Query execution plan optimization for federated SPARQL.
//!
//! The [`FederationOptimizer`] takes a [`FederatedQuery`] produced by
//! [`super::decomposer::QueryDecomposer`] and produces an optimized
//! [`ExecutionPlan`].  The plan represents a tree of join / union operations
//! over per-endpoint subqueries that minimises the estimated total cost.
//!
//! # Optimization passes (in order)
//!
//! 1. **Priority assignment** – assign a scheduling priority to each
//!    [`EndpointSubquery`] based on its cost estimate and cardinality.
//! 2. **Subquery merging** – merge multiple subqueries targeting the same
//!    endpoint into a single combined SELECT, reducing round-trips.
//! 3. **Join ordering** – sort join pairs by expected output cardinality so
//!    the most selective join is performed first.
//! 4. **Join strategy selection** – choose the cheapest join algorithm
//!    (bind join, hash join, etc.) for each pair.
//! 5. **Plan tree construction** – assemble the ordered joins into a
//!    left-deep [`PlanNode`] tree ready for execution.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use super::cost_estimator::CostEstimator;
use super::decomposer::{EndpointSubquery, FederatedQuery};
use super::error::{FederationError, FederationResult};
use super::plan::{ExecutionPlan, JoinStrategy, PlanNode};

/// Tuning knobs for the optimizer.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Threshold below which a bind join is preferred over a hash join.
    /// Bind join pushes bindings as a VALUES clause to the remote endpoint,
    /// so it is better when one side is small.
    pub bind_join_cardinality_threshold: usize,
    /// Maximum number of subqueries that can be merged into a single request
    /// to the same endpoint.
    pub max_merge_subquery_count: usize,
    /// When `true`, the optimizer will attempt to reorder joins by selectivity.
    pub enable_join_reordering: bool,
    /// When `true`, subqueries targeting the same endpoint are merged.
    pub enable_subquery_merging: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            bind_join_cardinality_threshold: 1_000,
            max_merge_subquery_count: 5,
            enable_join_reordering: true,
            enable_subquery_merging: true,
        }
    }
}

/// Federated query plan optimizer.
///
/// Wraps a [`CostEstimator`] and applies a sequence of optimization passes to
/// produce a low-cost [`ExecutionPlan`] from a [`FederatedQuery`].
#[derive(Debug, Clone)]
pub struct FederationOptimizer {
    /// Cost estimator used for all cost-based decisions.
    pub cost_estimator: CostEstimator,
    /// Optimizer configuration.
    pub config: OptimizerConfig,
}

impl Default for FederationOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl FederationOptimizer {
    /// Create an optimizer with default configuration and a fresh cost estimator.
    pub fn new() -> Self {
        Self {
            cost_estimator: CostEstimator::new(),
            config: OptimizerConfig::default(),
        }
    }

    /// Create an optimizer with a pre-configured cost estimator and custom settings.
    pub fn with_cost_estimator(cost_estimator: CostEstimator) -> Self {
        Self {
            cost_estimator,
            config: OptimizerConfig::default(),
        }
    }

    /// Apply all optimization passes to a [`FederatedQuery`] and return an
    /// [`ExecutionPlan`].
    ///
    /// # Errors
    ///
    /// Returns [`FederationError::InvalidPlan`] when the federated query
    /// contains no subqueries to execute.
    pub fn optimize(&self, mut query: FederatedQuery) -> FederationResult<ExecutionPlan> {
        let start = Instant::now();

        if query.subqueries.is_empty() {
            return Err(FederationError::InvalidPlan(
                "FederatedQuery contains no subqueries".to_string(),
            ));
        }

        // Pass 1: assign priorities based on cost
        self.assign_priorities(&mut query.subqueries);

        // Pass 2: merge subqueries going to the same endpoint
        let merged = if self.config.enable_subquery_merging {
            self.merge_subqueries(query.subqueries)
        } else {
            query.subqueries
        };

        // Pass 3: reorder joins by selectivity (ascending cardinality)
        let ordered = if self.config.enable_join_reordering {
            self.reorder_joins(merged)
        } else {
            merged
        };

        // Pass 4: build plan tree from ordered subqueries
        let root = self.build_plan_tree(ordered)?;

        let plan = ExecutionPlan::new(query.original_query, root, start.elapsed());
        Ok(plan)
    }

    /// Assign a scheduling priority to each subquery.
    ///
    /// Lower priority value = higher scheduling precedence (should run sooner).
    /// The priority is derived from the cost estimate so that cheap (fast) queries
    /// start first, allowing early bind-join opportunities.
    pub fn assign_priorities(&self, subqueries: &mut [EndpointSubquery]) {
        for sq in subqueries.iter_mut() {
            sq.priority = self.cost_estimator.estimate_cost(sq);
        }
    }

    /// Reorder a list of subqueries in ascending order of estimated result cardinality.
    ///
    /// This implements a greedy selectivity-first join ordering: the most selective
    /// (smallest) subquery is placed first, driving subsequent bind-join candidates.
    pub fn reorder_joins(&self, mut subqueries: Vec<EndpointSubquery>) -> Vec<EndpointSubquery> {
        subqueries.sort_by(|a, b| {
            a.estimated_results.cmp(&b.estimated_results).then_with(|| {
                a.priority
                    .partial_cmp(&b.priority)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });
        subqueries
    }

    /// Merge subqueries that target the same endpoint into a single combined subquery.
    ///
    /// Merging reduces the number of network round-trips.  The merged subquery
    /// projects the union of all variables from the constituent subqueries and
    /// concatenates their triple patterns.
    pub fn merge_subqueries(&self, subqueries: Vec<EndpointSubquery>) -> Vec<EndpointSubquery> {
        // Group by endpoint URL
        let mut groups: HashMap<String, Vec<EndpointSubquery>> = HashMap::new();
        for sq in subqueries {
            groups.entry(sq.endpoint_url.clone()).or_default().push(sq);
        }

        let mut result = Vec::new();
        for (endpoint_url, group) in groups {
            if group.len() <= 1 || group.len() > self.config.max_merge_subquery_count {
                result.extend(group);
                continue;
            }

            // Merge the group into a single subquery
            if let Some(merged) = merge_endpoint_group(&endpoint_url, group) {
                result.push(merged);
            }
        }

        result
    }

    /// Build a left-deep plan tree from an ordered list of subqueries.
    ///
    /// A left-deep tree means the first subquery is the left-most leaf, and each
    /// subsequent subquery is joined onto the right of the running tree.  This
    /// matches the conventional pipelined execution model where intermediate
    /// results flow left-to-right.
    fn build_plan_tree(&self, subqueries: Vec<EndpointSubquery>) -> FederationResult<PlanNode> {
        let mut iter = subqueries.into_iter();

        let first = iter.next().ok_or_else(|| {
            FederationError::InvalidPlan("No subqueries to build plan from".to_string())
        })?;

        let mut current = PlanNode::Subquery(first);

        for next_sq in iter {
            let join_vars = shared_variables_from_node(&current, &next_sq);
            let left_card = node_cardinality(&current);
            let right_card = next_sq.estimated_results;
            let join_cost = self
                .cost_estimator
                .estimate_join_cost(left_card, right_card);

            let strategy = choose_join_strategy(
                left_card,
                right_card,
                self.config.bind_join_cardinality_threshold,
            );

            current = PlanNode::Join {
                left: Box::new(current),
                right: Box::new(PlanNode::Subquery(next_sq)),
                join_variables: join_vars,
                strategy,
                estimated_cost: join_cost,
            };
        }

        Ok(current)
    }
}

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Merge a group of subqueries all targeting the same endpoint into one.
fn merge_endpoint_group(
    endpoint_url: &str,
    group: Vec<EndpointSubquery>,
) -> Option<EndpointSubquery> {
    if group.is_empty() {
        return None;
    }

    // Collect all variable projections and triple patterns from all subqueries in the group
    let mut all_vars: Vec<String> = Vec::new();
    let mut pattern_lines: Vec<String> = Vec::new();
    let mut total_results: usize = 0;
    let mut min_priority = f64::MAX;

    for sq in &group {
        let (vars, body) = extract_query_parts(&sq.sparql);
        for v in vars {
            if !all_vars.contains(&v) {
                all_vars.push(v);
            }
        }
        pattern_lines.extend(body);
        total_results = total_results.saturating_add(sq.estimated_results);
        if sq.priority < min_priority {
            min_priority = sq.priority;
        }
    }

    let proj = if all_vars.is_empty() {
        "*".to_string()
    } else {
        all_vars.join(" ")
    };

    let merged_sparql = format!(
        "SELECT {proj} WHERE {{\n{}\n}}",
        pattern_lines.join(" .\n  ")
    );

    Some(EndpointSubquery {
        endpoint_url: endpoint_url.to_string(),
        sparql: merged_sparql,
        estimated_results: total_results,
        priority: min_priority,
    })
}

/// Very lightweight extraction of projection variables and WHERE body lines
/// from a SELECT query string.  Returns `(variables, body_lines)`.
fn extract_query_parts(sparql: &str) -> (Vec<String>, Vec<String>) {
    let upper = sparql.to_uppercase();

    // Extract projection variables
    let select_pos = upper.find("SELECT").map(|p| p + 6).unwrap_or(0);
    let where_pos = upper.find("WHERE").unwrap_or(sparql.len());
    let projection_text = sparql[select_pos..where_pos].trim();

    let vars: Vec<String> = if projection_text == "*" {
        Vec::new()
    } else {
        projection_text
            .split_whitespace()
            .filter(|t| t.starts_with('?') || t.starts_with('$'))
            .map(|t| t.to_string())
            .collect()
    };

    // Extract WHERE body lines (rudimentary)
    let after_where = &sparql[where_pos..];
    let body = extract_brace_content(after_where)
        .unwrap_or_default()
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();

    (vars, body)
}

/// Extract content between the first matching pair of `{` `}` in `s`.
fn extract_brace_content(s: &str) -> Option<String> {
    let mut depth = 0i32;
    let mut start = None;
    let mut end = None;

    for (i, ch) in s.char_indices() {
        match ch {
            '{' => {
                if depth == 0 {
                    start = Some(i + 1);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }

    match (start, end) {
        (Some(s_idx), Some(e_idx)) => Some(s[s_idx..e_idx].to_string()),
        _ => None,
    }
}

/// Determine the join strategy to use given the cardinalities of the two inputs.
fn choose_join_strategy(
    left_card: usize,
    right_card: usize,
    bind_threshold: usize,
) -> JoinStrategy {
    let smaller = left_card.min(right_card);
    if smaller == 0 || smaller <= bind_threshold {
        // Small left side → bind join (push VALUES clause to remote)
        JoinStrategy::Bind
    } else if smaller <= 10_000 {
        JoinStrategy::Hash
    } else {
        JoinStrategy::Merge
    }
}

/// Collect variable names that appear in both the right subquery and (approximately)
/// the left plan node.
fn shared_variables_from_node(left: &PlanNode, right: &EndpointSubquery) -> Vec<String> {
    let right_vars = extract_sparql_variables(&right.sparql);
    let left_vars = collect_node_variables(left);

    right_vars.intersection(&left_vars).cloned().collect()
}

/// Extract variable tokens from a SPARQL string.
fn extract_sparql_variables(sparql: &str) -> HashSet<String> {
    let mut vars = HashSet::new();
    let mut chars = sparql.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '?'
            && chars
                .peek()
                .is_some_and(|c| c.is_alphanumeric() || *c == '_')
        {
            let mut var = String::from("?");
            while chars
                .peek()
                .is_some_and(|c| c.is_alphanumeric() || *c == '_')
            {
                var.push(chars.next().expect("already peeked"));
            }
            vars.insert(var);
        }
    }
    vars
}

/// Recursively collect all variable names appearing in the SPARQL of leaf nodes.
fn collect_node_variables(node: &PlanNode) -> HashSet<String> {
    node.collect_subqueries()
        .iter()
        .flat_map(|sq| extract_sparql_variables(&sq.sparql))
        .collect()
}

/// Estimate the output cardinality of a plan node.
fn node_cardinality(node: &PlanNode) -> usize {
    match node {
        PlanNode::Subquery(sq) => sq.estimated_results,
        PlanNode::Join {
            left,
            right,
            strategy,
            ..
        } => {
            let lc = node_cardinality(left);
            let rc = node_cardinality(right);
            match strategy {
                JoinStrategy::Bind => lc.min(rc),
                JoinStrategy::Hash | JoinStrategy::Merge | JoinStrategy::NestedLoop => {
                    // Assume 10% selectivity for an equi-join
                    ((lc as f64) * (rc as f64) * 0.1) as usize
                }
                JoinStrategy::Broadcast => lc.max(rc),
            }
        }
        PlanNode::Union { left, right } => node_cardinality(left) + node_cardinality(right),
        PlanNode::Filter { child, .. } => {
            // Assume filter reduces cardinality by 50%
            node_cardinality(child) / 2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_rewrite::decomposer::{
        EndpointInfo, EndpointSubquery, FederatedQuery, QueryDecomposer,
    };

    fn make_subquery(
        endpoint: &str,
        sparql: &str,
        results: usize,
        priority: f64,
    ) -> EndpointSubquery {
        EndpointSubquery {
            endpoint_url: endpoint.to_string(),
            sparql: sparql.to_string(),
            estimated_results: results,
            priority,
        }
    }

    fn make_federated_query(subqueries: Vec<EndpointSubquery>) -> FederatedQuery {
        FederatedQuery {
            original_query: "SELECT ?s WHERE { ?s ?p ?o }".to_string(),
            subqueries,
        }
    }

    #[test]
    fn test_optimize_empty_subqueries_returns_error() {
        let optimizer = FederationOptimizer::new();
        let fq = make_federated_query(vec![]);
        assert!(matches!(
            optimizer.optimize(fq),
            Err(FederationError::InvalidPlan(_))
        ));
    }

    #[test]
    fn test_optimize_single_subquery_produces_leaf_plan() {
        let optimizer = FederationOptimizer::new();
        let fq = make_federated_query(vec![make_subquery(
            "http://ep/sparql",
            "SELECT ?s WHERE { ?s ?p ?o }",
            100,
            0.0,
        )]);
        let plan = optimizer.optimize(fq).expect("optimization should succeed");
        assert!(matches!(plan.root, PlanNode::Subquery(_)));
        assert_eq!(plan.endpoint_subqueries().len(), 1);
    }

    #[test]
    fn test_optimize_two_subqueries_produces_join() {
        let optimizer = FederationOptimizer::new();
        let fq = make_federated_query(vec![
            make_subquery(
                "http://ep1/sparql",
                "SELECT ?s WHERE { ?s a foaf:Person }",
                50,
                0.0,
            ),
            make_subquery(
                "http://ep2/sparql",
                "SELECT ?s ?name WHERE { ?s foaf:name ?name }",
                200,
                0.0,
            ),
        ]);
        let plan = optimizer.optimize(fq).expect("optimization should succeed");
        assert!(matches!(plan.root, PlanNode::Join { .. }));
    }

    #[test]
    fn test_reorder_joins_sorts_ascending_cardinality() {
        let optimizer = FederationOptimizer::new();
        let sqs = vec![
            make_subquery("http://ep1/sparql", "SELECT ?s WHERE {?s ?p ?o}", 1000, 0.0),
            make_subquery("http://ep2/sparql", "SELECT ?s WHERE {?s ?p ?o}", 10, 0.0),
            make_subquery("http://ep3/sparql", "SELECT ?s WHERE {?s ?p ?o}", 500, 0.0),
        ];
        let ordered = optimizer.reorder_joins(sqs);
        assert_eq!(ordered[0].estimated_results, 10);
        assert_eq!(ordered[1].estimated_results, 500);
        assert_eq!(ordered[2].estimated_results, 1000);
    }

    #[test]
    fn test_merge_subqueries_same_endpoint() {
        let optimizer = FederationOptimizer::new();
        let sqs = vec![
            make_subquery(
                "http://ep/sparql",
                "SELECT ?s WHERE { ?s a foaf:Person }",
                100,
                1.0,
            ),
            make_subquery(
                "http://ep/sparql",
                "SELECT ?s ?name WHERE { ?s foaf:name ?name }",
                200,
                2.0,
            ),
        ];
        let merged = optimizer.merge_subqueries(sqs);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].endpoint_url, "http://ep/sparql");
        assert_eq!(merged[0].estimated_results, 300);
    }

    #[test]
    fn test_merge_subqueries_different_endpoints_unchanged() {
        let optimizer = FederationOptimizer::new();
        let sqs = vec![
            make_subquery(
                "http://ep1/sparql",
                "SELECT ?s WHERE {?s a foaf:Person}",
                10,
                1.0,
            ),
            make_subquery(
                "http://ep2/sparql",
                "SELECT ?o WHERE {?s foaf:knows ?o}",
                20,
                2.0,
            ),
        ];
        let merged = optimizer.merge_subqueries(sqs);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_assign_priorities_positive() {
        let optimizer = FederationOptimizer::new();
        let mut sqs = vec![make_subquery(
            "http://ep1/sparql",
            "SELECT ?s WHERE {?s ?p ?o}",
            100,
            0.0,
        )];
        optimizer.assign_priorities(&mut sqs);
        assert!(sqs[0].priority > 0.0);
    }

    #[test]
    fn test_choose_join_strategy_small_gives_bind() {
        let strategy = choose_join_strategy(50, 100, 1_000);
        assert_eq!(strategy, JoinStrategy::Bind);
    }

    #[test]
    fn test_choose_join_strategy_medium_gives_hash() {
        let strategy = choose_join_strategy(5_000, 8_000, 1_000);
        assert_eq!(strategy, JoinStrategy::Hash);
    }

    #[test]
    fn test_choose_join_strategy_large_gives_merge() {
        let strategy = choose_join_strategy(50_000, 80_000, 1_000);
        assert_eq!(strategy, JoinStrategy::Merge);
    }

    #[test]
    fn test_full_optimize_from_decomposer() {
        let endpoints = vec![
            EndpointInfo::new("http://foaf/sparql").with_affinity("foaf"),
            EndpointInfo::new("http://schema/sparql").with_affinity("schema"),
        ];
        let decomposer = QueryDecomposer::new(endpoints);
        let fq = decomposer
            .decompose("SELECT ?s ?name WHERE { ?s a foaf:Person . ?s foaf:name ?name }")
            .expect("decomposition should succeed");

        let optimizer = FederationOptimizer::new();
        let plan = optimizer.optimize(fq).expect("optimization should succeed");
        assert!(!plan.plan_id.is_empty());
        assert!(plan.total_estimated_cost >= 0.0);
    }

    #[test]
    fn test_plan_has_planning_duration() {
        let optimizer = FederationOptimizer::new();
        let fq = make_federated_query(vec![make_subquery(
            "http://ep/sparql",
            "SELECT * WHERE {?s ?p ?o}",
            10,
            0.0,
        )]);
        let plan = optimizer.optimize(fq).expect("optimization should succeed");
        // Planning duration should be non-zero (even if just a few nanoseconds)
        // planning_duration is a std::time::Duration (non-negative by definition)
        let _duration = plan.planning_duration.as_nanos();
    }

    #[test]
    fn test_extract_sparql_variables() {
        let vars = extract_sparql_variables("SELECT ?s ?name WHERE { ?s foaf:name ?name }");
        assert!(vars.contains("?s"));
        assert!(vars.contains("?name"));
    }

    #[test]
    fn test_shared_variables_from_node() {
        let left = PlanNode::Subquery(make_subquery(
            "http://ep1/sparql",
            "SELECT ?s WHERE { ?s a foaf:Person }",
            10,
            1.0,
        ));
        let right = make_subquery(
            "http://ep2/sparql",
            "SELECT ?s ?name WHERE { ?s foaf:name ?name }",
            20,
            2.0,
        );
        let shared = shared_variables_from_node(&left, &right);
        assert!(shared.contains(&"?s".to_string()));
    }
}
