//! SPARQL query plan optimizer with rewrite rules.
//!
//! Provides a pipeline of optimization passes that transform a logical
//! query plan into a more efficient equivalent plan.

use std::collections::HashSet;

/// A rewrite rule metadata descriptor.
#[derive(Debug, Clone)]
pub struct OptimizerRule {
    pub name: &'static str,
    pub description: &'static str,
}

/// Join execution strategy.
#[derive(Debug, Clone, PartialEq)]
pub enum JoinStrategy {
    NestedLoop,
    HashJoin,
    MergeJoin,
}

/// A node in the logical query plan tree.
#[derive(Debug, Clone, PartialEq)]
pub enum PlanOperator {
    Scan {
        pattern: String,
    },
    Join {
        strategy: JoinStrategy,
    },
    Filter {
        expr: String,
    },
    Project {
        vars: Vec<String>,
    },
    Union,
    Optional,
    Distinct,
    Slice {
        offset: Option<usize>,
        limit: Option<usize>,
    },
}

/// A node in the query plan tree with cost estimates.
#[derive(Debug, Clone)]
pub struct QueryPlan {
    pub operator: PlanOperator,
    pub children: Vec<QueryPlan>,
    pub estimated_cost: f64,
}

impl QueryPlan {
    /// Create a new leaf plan node.
    pub fn new(operator: PlanOperator) -> Self {
        QueryPlan {
            operator,
            children: Vec::new(),
            estimated_cost: 0.0,
        }
    }

    /// Create a new plan node with children.
    pub fn with_children(operator: PlanOperator, children: Vec<QueryPlan>) -> Self {
        QueryPlan {
            operator,
            children,
            estimated_cost: 0.0,
        }
    }

    /// Recompute cost bottom-up.
    pub fn recompute_cost(&mut self) {
        for child in &mut self.children {
            child.recompute_cost();
        }
        self.estimated_cost = estimate_cost(self);
    }
}

/// Compute recursive cost estimate for a plan node.
pub fn estimate_cost(plan: &QueryPlan) -> f64 {
    match &plan.operator {
        PlanOperator::Scan { .. } => 100.0,
        PlanOperator::Join { .. } => {
            let left = plan
                .children
                .first()
                .map(|c| c.estimated_cost)
                .unwrap_or(100.0);
            let right = plan
                .children
                .get(1)
                .map(|c| c.estimated_cost)
                .unwrap_or(100.0);
            left * right * 0.1
        }
        PlanOperator::Filter { .. } => {
            let child = plan
                .children
                .first()
                .map(|c| c.estimated_cost)
                .unwrap_or(100.0);
            child * 0.5
        }
        PlanOperator::Project { .. } => plan
            .children
            .first()
            .map(|c| c.estimated_cost)
            .unwrap_or(0.0),
        PlanOperator::Union => plan.children.iter().map(|c| c.estimated_cost).sum::<f64>(),
        PlanOperator::Optional => {
            let left = plan
                .children
                .first()
                .map(|c| c.estimated_cost)
                .unwrap_or(100.0);
            let right = plan
                .children
                .get(1)
                .map(|c| c.estimated_cost)
                .unwrap_or(100.0);
            left + right * 0.3
        }
        PlanOperator::Distinct => plan
            .children
            .first()
            .map(|c| c.estimated_cost * 1.1)
            .unwrap_or(0.0),
        PlanOperator::Slice { limit, .. } => {
            let child = plan
                .children
                .first()
                .map(|c| c.estimated_cost)
                .unwrap_or(0.0);
            if let Some(lim) = limit {
                child * (*lim as f64 / 1000.0).min(1.0)
            } else {
                child
            }
        }
    }
}

/// Trait for a single optimization pass.
pub trait OptimizationPass: Send + Sync {
    fn name(&self) -> &str;
    fn apply(&self, plan: QueryPlan) -> QueryPlan;
}

// ---------------------------------------------------------------------------
// Pass 1: Filter Pushdown
// ---------------------------------------------------------------------------

/// Push filter nodes below joins so they run earlier.
pub struct FilterPushdownPass;

impl FilterPushdownPass {
    fn push_filter_into_children(filter_expr: String, child: QueryPlan) -> QueryPlan {
        match &child.operator {
            PlanOperator::Join { .. } => {
                // Push filter into the left child (simplistic: push to all that reference the expr)
                let mut new_children = child.children.clone();
                if let Some(first) = new_children.first_mut() {
                    let filter_plan = QueryPlan::with_children(
                        PlanOperator::Filter { expr: filter_expr },
                        vec![first.clone()],
                    );
                    *first = filter_plan;
                }
                QueryPlan::with_children(child.operator.clone(), new_children)
            }
            _ => {
                // Cannot push further; wrap as-is
                QueryPlan::with_children(PlanOperator::Filter { expr: filter_expr }, vec![child])
            }
        }
    }
}

impl OptimizationPass for FilterPushdownPass {
    fn name(&self) -> &str {
        "FilterPushdown"
    }

    fn apply(&self, plan: QueryPlan) -> QueryPlan {
        let children: Vec<QueryPlan> = plan.children.into_iter().map(|c| self.apply(c)).collect();
        match plan.operator {
            PlanOperator::Filter { ref expr } => {
                if let Some(child) = children.into_iter().next() {
                    Self::push_filter_into_children(expr.clone(), child)
                } else {
                    QueryPlan::new(plan.operator)
                }
            }
            op => QueryPlan {
                operator: op,
                children,
                estimated_cost: plan.estimated_cost,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 2: Projection Pruning
// ---------------------------------------------------------------------------

/// Remove projection nodes that include all upstream variables (redundant projections).
pub struct ProjectionPruningPass;

impl ProjectionPruningPass {
    fn collect_scan_vars(plan: &QueryPlan) -> HashSet<String> {
        let mut vars = HashSet::new();
        match &plan.operator {
            PlanOperator::Scan { pattern } => {
                // Extract ?var tokens from pattern string
                for token in pattern.split_whitespace() {
                    if token.starts_with('?') {
                        vars.insert(token.trim_start_matches('?').to_string());
                    }
                }
            }
            PlanOperator::Project { vars: pv } => {
                for v in pv {
                    vars.insert(v.clone());
                }
            }
            _ => {}
        }
        for child in &plan.children {
            vars.extend(Self::collect_scan_vars(child));
        }
        vars
    }
}

impl OptimizationPass for ProjectionPruningPass {
    fn name(&self) -> &str {
        "ProjectionPruning"
    }

    fn apply(&self, plan: QueryPlan) -> QueryPlan {
        let children: Vec<QueryPlan> = plan.children.into_iter().map(|c| self.apply(c)).collect();
        match &plan.operator {
            PlanOperator::Project { vars } => {
                // Collect upstream variables
                let upstream: HashSet<String> =
                    children.iter().flat_map(Self::collect_scan_vars).collect();
                // If all projected vars are covered and projection adds nothing new, skip it
                let all_included = vars.iter().all(|v| upstream.contains(v));
                let reduces = vars.len() < upstream.len();
                if all_included && !reduces && children.len() == 1 {
                    // Remove redundant projection
                    children
                        .into_iter()
                        .next()
                        .unwrap_or_else(|| QueryPlan::new(plan.operator.clone()))
                } else {
                    QueryPlan {
                        operator: plan.operator.clone(),
                        children,
                        estimated_cost: plan.estimated_cost,
                    }
                }
            }
            op => QueryPlan {
                operator: op.clone(),
                children,
                estimated_cost: plan.estimated_cost,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 3: Join Reordering
// ---------------------------------------------------------------------------

/// Reorder join children by ascending estimated cost.
pub struct JoinReorderPass;

impl OptimizationPass for JoinReorderPass {
    fn name(&self) -> &str {
        "JoinReorder"
    }

    fn apply(&self, plan: QueryPlan) -> QueryPlan {
        let children: Vec<QueryPlan> = plan.children.into_iter().map(|c| self.apply(c)).collect();
        match &plan.operator {
            PlanOperator::Join { .. } => {
                let mut sorted = children;
                sorted.sort_by(|a, b| {
                    a.estimated_cost
                        .partial_cmp(&b.estimated_cost)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                QueryPlan {
                    operator: plan.operator.clone(),
                    children: sorted,
                    estimated_cost: plan.estimated_cost,
                }
            }
            op => QueryPlan {
                operator: op.clone(),
                children,
                estimated_cost: plan.estimated_cost,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 4: Constant Folding
// ---------------------------------------------------------------------------

/// Fold constant boolean expressions in filters.
pub struct ConstantFoldingPass;

impl ConstantFoldingPass {
    fn fold_expr(expr: &str) -> Option<bool> {
        match expr.trim() {
            "true" | "TRUE" | "1=1" => Some(true),
            "false" | "FALSE" | "1=0" | "0=1" => Some(false),
            _ => None,
        }
    }
}

impl OptimizationPass for ConstantFoldingPass {
    fn name(&self) -> &str {
        "ConstantFolding"
    }

    fn apply(&self, plan: QueryPlan) -> QueryPlan {
        let operator = plan.operator.clone();
        let estimated_cost = plan.estimated_cost;
        let children: Vec<QueryPlan> = plan.children.into_iter().map(|c| self.apply(c)).collect();
        match operator {
            PlanOperator::Filter { ref expr } => {
                match Self::fold_expr(expr) {
                    Some(true) => {
                        // Filter always true — remove it; keep first child if present
                        let mut iter = children.into_iter();
                        iter.next().unwrap_or_else(|| QueryPlan::new(operator))
                    }
                    Some(false) => {
                        // Filter always false — return empty scan
                        QueryPlan::new(PlanOperator::Scan {
                            pattern: String::from("EMPTY"),
                        })
                    }
                    None => QueryPlan {
                        operator,
                        children,
                        estimated_cost,
                    },
                }
            }
            op => QueryPlan {
                operator: op,
                children,
                estimated_cost,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 5: Redundant Distinct Removal
// ---------------------------------------------------------------------------

/// Remove DISTINCT operators that are already guaranteed unique.
pub struct RedundantDistinctPass;

impl RedundantDistinctPass {
    fn is_unique_source(plan: &QueryPlan) -> bool {
        matches!(plan.operator, PlanOperator::Scan { .. })
    }
}

impl OptimizationPass for RedundantDistinctPass {
    fn name(&self) -> &str {
        "RedundantDistinct"
    }

    fn apply(&self, plan: QueryPlan) -> QueryPlan {
        let children: Vec<QueryPlan> = plan.children.into_iter().map(|c| self.apply(c)).collect();
        match &plan.operator {
            PlanOperator::Distinct => {
                // If already wrapping a DISTINCT or a Scan (inherently unique), remove
                if let Some(child) = children.first() {
                    if matches!(child.operator, PlanOperator::Distinct)
                        || Self::is_unique_source(child)
                    {
                        return children
                            .into_iter()
                            .next()
                            .unwrap_or_else(|| QueryPlan::new(plan.operator.clone()));
                    }
                }
                QueryPlan {
                    operator: plan.operator.clone(),
                    children,
                    estimated_cost: plan.estimated_cost,
                }
            }
            op => QueryPlan {
                operator: op.clone(),
                children,
                estimated_cost: plan.estimated_cost,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Main Optimizer
// ---------------------------------------------------------------------------

/// SPARQL query plan optimizer that runs a series of optimization passes.
pub struct SparqlOptimizer {
    rules: Vec<Box<dyn OptimizationPass>>,
}

impl SparqlOptimizer {
    /// Create a new optimizer with all default passes registered.
    pub fn new() -> Self {
        let rules: Vec<Box<dyn OptimizationPass>> = vec![
            Box::new(FilterPushdownPass),
            Box::new(ProjectionPruningPass),
            Box::new(JoinReorderPass),
            Box::new(ConstantFoldingPass),
            Box::new(RedundantDistinctPass),
        ];
        SparqlOptimizer { rules }
    }

    /// Apply all optimization passes sequentially.
    pub fn optimize(&self, mut plan: QueryPlan) -> QueryPlan {
        // Recompute costs before optimization
        plan.recompute_cost();
        for pass in &self.rules {
            plan = pass.apply(plan);
            plan.recompute_cost();
        }
        plan
    }

    /// Return optimizer rule metadata.
    pub fn rule_info(&self) -> Vec<OptimizerRule> {
        vec![
            OptimizerRule {
                name: "FilterPushdown",
                description: "Push filters below joins",
            },
            OptimizerRule {
                name: "ProjectionPruning",
                description: "Remove redundant projections",
            },
            OptimizerRule {
                name: "JoinReorder",
                description: "Reorder joins by cost ascending",
            },
            OptimizerRule {
                name: "ConstantFolding",
                description: "Fold constant expressions",
            },
            OptimizerRule {
                name: "RedundantDistinct",
                description: "Remove redundant DISTINCT",
            },
        ]
    }
}

impl Default for SparqlOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(pattern: &str) -> QueryPlan {
        let mut p = QueryPlan::new(PlanOperator::Scan {
            pattern: pattern.to_string(),
        });
        p.estimated_cost = 100.0;
        p
    }

    fn join(left: QueryPlan, right: QueryPlan) -> QueryPlan {
        let mut p = QueryPlan::with_children(
            PlanOperator::Join {
                strategy: JoinStrategy::NestedLoop,
            },
            vec![left, right],
        );
        p.recompute_cost();
        p
    }

    fn filter(expr: &str, child: QueryPlan) -> QueryPlan {
        let mut p = QueryPlan::with_children(
            PlanOperator::Filter {
                expr: expr.to_string(),
            },
            vec![child],
        );
        p.recompute_cost();
        p
    }

    fn project(vars: Vec<&str>, child: QueryPlan) -> QueryPlan {
        let mut p = QueryPlan::with_children(
            PlanOperator::Project {
                vars: vars.into_iter().map(String::from).collect(),
            },
            vec![child],
        );
        p.recompute_cost();
        p
    }

    fn distinct(child: QueryPlan) -> QueryPlan {
        let mut p = QueryPlan::with_children(PlanOperator::Distinct, vec![child]);
        p.recompute_cost();
        p
    }

    // Cost estimation tests
    #[test]
    fn test_cost_scan() {
        let p = scan("?s ?p ?o");
        assert_eq!(estimate_cost(&p), 100.0);
    }

    #[test]
    fn test_cost_join() {
        let j = join(scan("?s ?p ?o"), scan("?s ?p2 ?o2"));
        assert!((j.estimated_cost - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_cost_filter() {
        let mut f = filter("?s = 1", scan("?s ?p ?o"));
        f.recompute_cost();
        assert!((f.estimated_cost - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_cost_project() {
        let mut p = project(vec!["s"], scan("?s ?p ?o"));
        p.recompute_cost();
        assert_eq!(p.estimated_cost, 100.0);
    }

    #[test]
    fn test_cost_union() {
        let u = QueryPlan::with_children(
            PlanOperator::Union,
            vec![scan("?a ?b ?c"), scan("?x ?y ?z")],
        );
        assert_eq!(estimate_cost(&u), 200.0);
    }

    #[test]
    fn test_cost_distinct() {
        let d = distinct(scan("?s ?p ?o"));
        assert!((estimate_cost(&d) - 110.0).abs() < 0.01);
    }

    #[test]
    fn test_cost_slice_with_limit() {
        let mut s = QueryPlan::with_children(
            PlanOperator::Slice {
                offset: None,
                limit: Some(10),
            },
            vec![scan("?s ?p ?o")],
        );
        s.recompute_cost();
        // 100 * (10/1000) = 1.0
        assert!((s.estimated_cost - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cost_slice_no_limit() {
        let mut s = QueryPlan::with_children(
            PlanOperator::Slice {
                offset: Some(5),
                limit: None,
            },
            vec![scan("?s ?p ?o")],
        );
        s.recompute_cost();
        assert_eq!(s.estimated_cost, 100.0);
    }

    #[test]
    fn test_cost_optional() {
        let opt = QueryPlan::with_children(
            PlanOperator::Optional,
            vec![scan("?a ?b ?c"), scan("?x ?y ?z")],
        );
        // 100 + 100 * 0.3 = 130
        assert!((estimate_cost(&opt) - 130.0).abs() < 0.01);
    }

    // FilterPushdown tests
    #[test]
    fn test_filter_pushdown_through_join() {
        let pass = FilterPushdownPass;
        let j = join(scan("?s ?p ?o"), scan("?s ?p2 ?o2"));
        let f = filter("?s = 1", j);
        let result = pass.apply(f);
        // Filter should have been pushed into join's left child
        if let PlanOperator::Join { .. } = &result.operator {
            let left = &result.children[0];
            assert!(matches!(left.operator, PlanOperator::Filter { .. }));
        } else {
            panic!("Expected Join at top level after pushdown");
        }
    }

    #[test]
    fn test_filter_pushdown_non_join_child() {
        let pass = FilterPushdownPass;
        let f = filter("?x > 5", scan("?x ?y ?z"));
        let result = pass.apply(f);
        // Filter stays wrapping scan
        assert!(matches!(result.operator, PlanOperator::Filter { .. }));
    }

    #[test]
    fn test_filter_pushdown_empty_children() {
        let pass = FilterPushdownPass;
        let f = QueryPlan::new(PlanOperator::Filter {
            expr: "?x > 0".to_string(),
        });
        let result = pass.apply(f);
        assert!(matches!(result.operator, PlanOperator::Filter { .. }));
    }

    // ProjectionPruning tests
    #[test]
    fn test_projection_pruning_removes_redundant() {
        let pass = ProjectionPruningPass;
        // Project all vars that a scan provides — should be pruned
        let child = scan("?s ?p ?o");
        let p = project(vec!["s", "p", "o"], child);
        let result = pass.apply(p);
        // Redundant projection removed
        assert!(matches!(result.operator, PlanOperator::Scan { .. }));
    }

    #[test]
    fn test_projection_pruning_keeps_reducing() {
        let pass = ProjectionPruningPass;
        let child = scan("?s ?p ?o");
        let p = project(vec!["s"], child);
        let result = pass.apply(p);
        // Reduces variables — keep it
        assert!(matches!(result.operator, PlanOperator::Project { .. }));
    }

    #[test]
    fn test_projection_pruning_no_children() {
        let pass = ProjectionPruningPass;
        let p = QueryPlan::new(PlanOperator::Project {
            vars: vec!["s".to_string()],
        });
        let result = pass.apply(p);
        assert!(matches!(result.operator, PlanOperator::Project { .. }));
    }

    // JoinReorder tests
    #[test]
    fn test_join_reorder_sorts_by_cost() {
        let pass = JoinReorderPass;
        let expensive = {
            let mut p = scan("?a ?b ?c");
            p.estimated_cost = 500.0;
            p
        };
        let cheap = {
            let mut p = scan("?x ?y ?z");
            p.estimated_cost = 50.0;
            p
        };
        let j = QueryPlan::with_children(
            PlanOperator::Join {
                strategy: JoinStrategy::HashJoin,
            },
            vec![expensive, cheap],
        );
        let result = pass.apply(j);
        assert!(result.children[0].estimated_cost <= result.children[1].estimated_cost);
    }

    #[test]
    fn test_join_reorder_already_sorted() {
        let pass = JoinReorderPass;
        let mut c1 = scan("?a ?b ?c");
        c1.estimated_cost = 10.0;
        let mut c2 = scan("?x ?y ?z");
        c2.estimated_cost = 200.0;
        let j = QueryPlan::with_children(
            PlanOperator::Join {
                strategy: JoinStrategy::MergeJoin,
            },
            vec![c1, c2],
        );
        let result = pass.apply(j);
        assert!(result.children[0].estimated_cost <= result.children[1].estimated_cost);
    }

    #[test]
    fn test_join_reorder_non_join_unchanged() {
        let pass = JoinReorderPass;
        let s = scan("?s ?p ?o");
        let result = pass.apply(s);
        assert!(matches!(result.operator, PlanOperator::Scan { .. }));
    }

    // ConstantFolding tests
    #[test]
    fn test_constant_folding_true_removes_filter() {
        let pass = ConstantFoldingPass;
        let f = filter("true", scan("?s ?p ?o"));
        let result = pass.apply(f);
        assert!(matches!(result.operator, PlanOperator::Scan { .. }));
    }

    #[test]
    fn test_constant_folding_false_returns_empty() {
        let pass = ConstantFoldingPass;
        let f = filter("false", scan("?s ?p ?o"));
        let result = pass.apply(f);
        if let PlanOperator::Scan { pattern } = &result.operator {
            assert_eq!(pattern, "EMPTY");
        } else {
            panic!("Expected empty scan");
        }
    }

    #[test]
    fn test_constant_folding_tautology_1eq1() {
        let pass = ConstantFoldingPass;
        let f = filter("1=1", scan("?s ?p ?o"));
        let result = pass.apply(f);
        assert!(matches!(result.operator, PlanOperator::Scan { .. }));
    }

    #[test]
    fn test_constant_folding_false_1eq0() {
        let pass = ConstantFoldingPass;
        let f = filter("1=0", scan("?s ?p ?o"));
        let result = pass.apply(f);
        if let PlanOperator::Scan { pattern } = &result.operator {
            assert_eq!(pattern, "EMPTY");
        } else {
            panic!("Expected empty scan");
        }
    }

    #[test]
    fn test_constant_folding_non_constant_unchanged() {
        let pass = ConstantFoldingPass;
        let f = filter("?x > 5", scan("?s ?p ?o"));
        let result = pass.apply(f);
        assert!(matches!(result.operator, PlanOperator::Filter { .. }));
    }

    #[test]
    fn test_constant_folding_uppercase_true() {
        let pass = ConstantFoldingPass;
        let f = filter("TRUE", scan("?s ?p ?o"));
        let result = pass.apply(f);
        assert!(matches!(result.operator, PlanOperator::Scan { .. }));
    }

    // RedundantDistinct tests
    #[test]
    fn test_redundant_distinct_scan_removed() {
        let pass = RedundantDistinctPass;
        let d = distinct(scan("?s ?p ?o"));
        let result = pass.apply(d);
        assert!(matches!(result.operator, PlanOperator::Scan { .. }));
    }

    #[test]
    fn test_redundant_distinct_double_distinct() {
        let pass = RedundantDistinctPass;
        let d = distinct(distinct(scan("?s ?p ?o")));
        let result = pass.apply(d);
        // Inner distinct should collapse to scan; outer sees distinct wrapping scan and removes
        assert!(matches!(
            result.operator,
            PlanOperator::Scan { .. } | PlanOperator::Distinct
        ));
    }

    #[test]
    fn test_redundant_distinct_over_join_kept() {
        let pass = RedundantDistinctPass;
        let j = join(scan("?s ?p ?o"), scan("?s ?p2 ?o2"));
        let d = distinct(j);
        let result = pass.apply(d);
        assert!(matches!(result.operator, PlanOperator::Distinct));
    }

    #[test]
    fn test_redundant_distinct_empty_children() {
        let pass = RedundantDistinctPass;
        let d = QueryPlan::new(PlanOperator::Distinct);
        let result = pass.apply(d);
        assert!(matches!(result.operator, PlanOperator::Distinct));
    }

    // Full optimizer tests
    #[test]
    fn test_optimizer_new() {
        let opt = SparqlOptimizer::new();
        assert_eq!(opt.rules.len(), 5);
    }

    #[test]
    fn test_optimizer_default() {
        let opt = SparqlOptimizer::default();
        assert_eq!(opt.rules.len(), 5);
    }

    #[test]
    fn test_optimizer_rule_info() {
        let opt = SparqlOptimizer::new();
        let rules = opt.rule_info();
        assert_eq!(rules.len(), 5);
        assert!(rules.iter().any(|r| r.name == "FilterPushdown"));
        assert!(rules.iter().any(|r| r.name == "JoinReorder"));
    }

    #[test]
    fn test_optimizer_chain_simple() {
        let opt = SparqlOptimizer::new();
        let plan = scan("?s ?p ?o");
        let result = opt.optimize(plan);
        assert!(matches!(result.operator, PlanOperator::Scan { .. }));
    }

    #[test]
    fn test_optimizer_chain_filter_join() {
        let opt = SparqlOptimizer::new();
        let j = join(scan("?s ?p ?o"), scan("?s ?p2 ?o2"));
        let f = filter("?s = 1", j);
        let result = opt.optimize(f);
        // After optimization the filter should have been pushed or plan restructured
        assert!(result.estimated_cost >= 0.0);
    }

    #[test]
    fn test_optimizer_chain_constant_filter() {
        let opt = SparqlOptimizer::new();
        let f = filter("true", scan("?s ?p ?o"));
        let result = opt.optimize(f);
        assert!(matches!(result.operator, PlanOperator::Scan { .. }));
    }

    #[test]
    fn test_optimizer_chain_redundant_distinct() {
        let opt = SparqlOptimizer::new();
        let d = distinct(scan("?s ?p ?o"));
        let result = opt.optimize(d);
        assert!(matches!(result.operator, PlanOperator::Scan { .. }));
    }

    #[test]
    fn test_optimizer_empty_plan() {
        let opt = SparqlOptimizer::new();
        let plan = QueryPlan::new(PlanOperator::Union);
        let result = opt.optimize(plan);
        assert!(matches!(result.operator, PlanOperator::Union));
    }

    #[test]
    fn test_optimizer_deeply_nested() {
        let opt = SparqlOptimizer::new();
        let s1 = scan("?a ?b ?c");
        let s2 = scan("?d ?e ?f");
        let s3 = scan("?g ?h ?i");
        let j1 = join(s1, s2);
        let j2 = join(j1, s3);
        let f = filter("?a > 0", j2);
        let d = distinct(f);
        let result = opt.optimize(d);
        assert!(result.estimated_cost >= 0.0);
    }

    #[test]
    fn test_plan_recompute_cost() {
        let mut j = join(scan("?s ?p ?o"), scan("?x ?y ?z"));
        j.recompute_cost();
        assert!((j.estimated_cost - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_join_strategy_variants() {
        let strategies = [
            JoinStrategy::NestedLoop,
            JoinStrategy::HashJoin,
            JoinStrategy::MergeJoin,
        ];
        for strategy in &strategies {
            let p = QueryPlan::with_children(
                PlanOperator::Join {
                    strategy: strategy.clone(),
                },
                vec![scan("?a ?b ?c"), scan("?x ?y ?z")],
            );
            assert!(matches!(p.operator, PlanOperator::Join { .. }));
        }
    }
}
