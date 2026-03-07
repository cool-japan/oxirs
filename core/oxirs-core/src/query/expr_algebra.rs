//! SPARQL Algebra Expression Tree
//!
//! A standalone SPARQL 1.1 algebra representation for building, analyzing, and
//! transforming query trees. Covers BGP, Filter, Join, LeftJoin, Union, Minus,
//! Extend, Project, Distinct, OrderBy, Slice, Graph, and Group.

use std::collections::HashSet;

/// A SPARQL filter / projection expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Variable reference `?x`
    Var(String),
    /// Constant (IRI, literal, blank node represented as a string)
    Const(String),
    /// `?x = ?y` or `?x = "value"`
    Eq(Box<Expr>, Box<Expr>),
    /// `?x != ?y`
    Ne(Box<Expr>, Box<Expr>),
    /// `?x < ?y`
    Lt(Box<Expr>, Box<Expr>),
    /// `?x <= ?y`
    Le(Box<Expr>, Box<Expr>),
    /// `?x > ?y`
    Gt(Box<Expr>, Box<Expr>),
    /// `?x >= ?y`
    Ge(Box<Expr>, Box<Expr>),
    /// Logical `&&`
    And(Box<Expr>, Box<Expr>),
    /// Logical `||`
    Or(Box<Expr>, Box<Expr>),
    /// Logical `!`
    Not(Box<Expr>),
    /// `BOUND(?x)` — tests variable binding
    BoundVar(String),
    /// `REGEX(expr, pattern, flags)`
    Regex {
        expr: Box<Expr>,
        pattern: String,
        flags: String,
    },
    /// `isIRI(expr)`
    IsIri(Box<Expr>),
    /// `isLiteral(expr)`
    IsLiteral(Box<Expr>),
    /// `isBlank(expr)`
    IsBlank(Box<Expr>),
    /// `str(expr)` — converts to string representation
    Str(Box<Expr>),
    /// `lang(expr)` — extracts language tag
    Lang(Box<Expr>),
}

impl Expr {
    /// Collect all variable names referenced in this expression (sorted, deduplicated).
    pub fn variables(&self) -> Vec<String> {
        let mut set = HashSet::new();
        self.collect_vars(&mut set);
        let mut v: Vec<String> = set.into_iter().collect();
        v.sort();
        v
    }

    fn collect_vars(&self, out: &mut HashSet<String>) {
        match self {
            Expr::Var(n) | Expr::BoundVar(n) => {
                out.insert(n.clone());
            }
            Expr::Const(_) => {}
            Expr::Eq(l, r)
            | Expr::Ne(l, r)
            | Expr::Lt(l, r)
            | Expr::Le(l, r)
            | Expr::Gt(l, r)
            | Expr::Ge(l, r)
            | Expr::And(l, r)
            | Expr::Or(l, r) => {
                l.collect_vars(out);
                r.collect_vars(out);
            }
            Expr::Not(e)
            | Expr::IsIri(e)
            | Expr::IsLiteral(e)
            | Expr::IsBlank(e)
            | Expr::Str(e)
            | Expr::Lang(e) => e.collect_vars(out),
            Expr::Regex { expr, .. } => expr.collect_vars(out),
        }
    }

    /// Depth of the expression sub-tree (leaf = 1).
    pub fn depth(&self) -> usize {
        match self {
            Expr::Var(_) | Expr::Const(_) | Expr::BoundVar(_) => 1,
            Expr::Eq(l, r)
            | Expr::Ne(l, r)
            | Expr::Lt(l, r)
            | Expr::Le(l, r)
            | Expr::Gt(l, r)
            | Expr::Ge(l, r)
            | Expr::And(l, r)
            | Expr::Or(l, r) => 1 + l.depth().max(r.depth()),
            Expr::Not(e)
            | Expr::IsIri(e)
            | Expr::IsLiteral(e)
            | Expr::IsBlank(e)
            | Expr::Str(e)
            | Expr::Lang(e) => 1 + e.depth(),
            Expr::Regex { expr, .. } => 1 + expr.depth(),
        }
    }
}

/// A single triple pattern in a BGP — each position may be a variable or constant.
#[derive(Debug, Clone, PartialEq)]
pub struct TriplePattern {
    pub subject: Expr,
    pub predicate: Expr,
    pub object: Expr,
}

impl TriplePattern {
    pub fn new(subject: Expr, predicate: Expr, object: Expr) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }

    /// Variables present in this triple pattern (sorted).
    pub fn variables(&self) -> Vec<String> {
        let mut set = HashSet::new();
        self.subject.collect_vars(&mut set);
        self.predicate.collect_vars(&mut set);
        self.object.collect_vars(&mut set);
        let mut v: Vec<String> = set.into_iter().collect();
        v.sort();
        v
    }
}

/// Aggregation function used in GROUP BY / aggregation algebra nodes.
#[derive(Debug, Clone, PartialEq)]
pub enum AggFunc {
    /// `COUNT(*)` or `COUNT(expr)`
    Count(Option<Expr>),
    /// `SUM(expr)`
    Sum(Expr),
    /// `MIN(expr)`
    Min(Expr),
    /// `MAX(expr)`
    Max(Expr),
    /// `AVG(expr)`
    Avg(Expr),
    /// `GROUP_CONCAT(expr; separator="sep")`
    GroupConcat(Expr, String),
}

/// A SPARQL algebra tree node.
#[derive(Debug, Clone, PartialEq)]
pub enum AlgebraNode {
    /// Basic Graph Pattern — a set of triple patterns matched conjunctively.
    BGP(Vec<TriplePattern>),
    /// `FILTER` — restricts solutions by a boolean expression.
    Filter {
        child: Box<AlgebraNode>,
        expr: Expr,
    },
    /// Inner join of two solution sets.
    Join(Box<AlgebraNode>, Box<AlgebraNode>),
    /// Optional join (`OPTIONAL { ... }`).
    LeftJoin {
        left: Box<AlgebraNode>,
        right: Box<AlgebraNode>,
        /// Optional filter applied within the optional block.
        cond: Option<Expr>,
    },
    /// `UNION` of two sub-patterns.
    Union(Box<AlgebraNode>, Box<AlgebraNode>),
    /// `MINUS` — subtract compatible solutions from the right-hand side.
    Minus(Box<AlgebraNode>, Box<AlgebraNode>),
    /// `BIND(?var := expr)` — extends solutions with a new variable.
    Extend {
        child: Box<AlgebraNode>,
        var: String,
        expr: Expr,
    },
    /// `SELECT ?a ?b ...` — projects solutions to a specific variable set.
    Project {
        child: Box<AlgebraNode>,
        vars: Vec<String>,
    },
    /// `SELECT DISTINCT` — removes duplicate solutions.
    Distinct(Box<AlgebraNode>),
    /// `ORDER BY` — sorts solutions by one or more expressions.
    OrderBy {
        child: Box<AlgebraNode>,
        /// `(expression, ascending)` pairs.
        conditions: Vec<(Expr, bool)>,
    },
    /// `LIMIT` / `OFFSET` slice.
    Slice {
        child: Box<AlgebraNode>,
        offset: Option<usize>,
        limit: Option<usize>,
    },
    /// `GRAPH <name> { ... }` — evaluates against a named graph.
    Graph {
        name: Expr,
        child: Box<AlgebraNode>,
    },
    /// `GROUP BY` with aggregation functions.
    Group {
        child: Box<AlgebraNode>,
        by: Vec<String>,
        aggregates: Vec<(String, AggFunc)>,
    },
}

impl AlgebraNode {
    // ── variable collection ─────────────────────────────────────────────────

    /// Collect all variable names in this algebra sub-tree (sorted, deduplicated).
    pub fn variables(&self) -> Vec<String> {
        let mut set = HashSet::new();
        self.collect_vars(&mut set);
        let mut v: Vec<String> = set.into_iter().collect();
        v.sort();
        v
    }

    fn collect_vars(&self, out: &mut HashSet<String>) {
        match self {
            AlgebraNode::BGP(pats) => {
                for p in pats {
                    for v in p.variables() {
                        out.insert(v);
                    }
                }
            }
            AlgebraNode::Filter { child, expr } => {
                child.collect_vars(out);
                expr.collect_vars(out);
            }
            AlgebraNode::Join(l, r)
            | AlgebraNode::Union(l, r)
            | AlgebraNode::Minus(l, r) => {
                l.collect_vars(out);
                r.collect_vars(out);
            }
            AlgebraNode::LeftJoin { left, right, cond } => {
                left.collect_vars(out);
                right.collect_vars(out);
                if let Some(c) = cond {
                    c.collect_vars(out);
                }
            }
            AlgebraNode::Extend { child, var, expr } => {
                child.collect_vars(out);
                out.insert(var.clone());
                expr.collect_vars(out);
            }
            AlgebraNode::Project { child, vars } => {
                child.collect_vars(out);
                for v in vars {
                    out.insert(v.clone());
                }
            }
            AlgebraNode::Distinct(c)
            | AlgebraNode::Slice { child: c, .. } => c.collect_vars(out),
            AlgebraNode::OrderBy { child, conditions } => {
                child.collect_vars(out);
                for (e, _) in conditions {
                    e.collect_vars(out);
                }
            }
            AlgebraNode::Graph { name, child } => {
                name.collect_vars(out);
                child.collect_vars(out);
            }
            AlgebraNode::Group {
                child,
                by,
                aggregates,
            } => {
                child.collect_vars(out);
                for v in by {
                    out.insert(v.clone());
                }
                for (var, agg) in aggregates {
                    out.insert(var.clone());
                    match agg {
                        AggFunc::Count(Some(e))
                        | AggFunc::Sum(e)
                        | AggFunc::Min(e)
                        | AggFunc::Max(e)
                        | AggFunc::Avg(e)
                        | AggFunc::GroupConcat(e, _) => e.collect_vars(out),
                        AggFunc::Count(None) => {}
                    }
                }
            }
        }
    }

    // ── tree metrics ────────────────────────────────────────────────────────

    /// Maximum nesting depth of the algebra tree (BGP leaf = 1).
    pub fn depth(&self) -> usize {
        match self {
            AlgebraNode::BGP(_) => 1,
            AlgebraNode::Filter { child, .. }
            | AlgebraNode::Extend { child, .. }
            | AlgebraNode::Project { child, .. }
            | AlgebraNode::Distinct(child)
            | AlgebraNode::OrderBy { child, .. }
            | AlgebraNode::Slice { child, .. }
            | AlgebraNode::Graph { child, .. }
            | AlgebraNode::Group { child, .. } => 1 + child.depth(),
            AlgebraNode::Join(l, r)
            | AlgebraNode::Union(l, r)
            | AlgebraNode::Minus(l, r) => 1 + l.depth().max(r.depth()),
            AlgebraNode::LeftJoin { left, right, .. } => {
                1 + left.depth().max(right.depth())
            }
        }
    }

    /// Total number of algebra nodes in this sub-tree.
    pub fn node_count(&self) -> usize {
        match self {
            AlgebraNode::BGP(_) => 1,
            AlgebraNode::Filter { child, .. }
            | AlgebraNode::Extend { child, .. }
            | AlgebraNode::Project { child, .. }
            | AlgebraNode::Distinct(child)
            | AlgebraNode::OrderBy { child, .. }
            | AlgebraNode::Slice { child, .. }
            | AlgebraNode::Graph { child, .. }
            | AlgebraNode::Group { child, .. } => 1 + child.node_count(),
            AlgebraNode::Join(l, r)
            | AlgebraNode::Union(l, r)
            | AlgebraNode::Minus(l, r) => 1 + l.node_count() + r.node_count(),
            AlgebraNode::LeftJoin { left, right, .. } => {
                1 + left.node_count() + right.node_count()
            }
        }
    }

    /// Returns `true` iff this node is a bare `BGP` (no wrapping operators).
    pub fn is_bgp(&self) -> bool {
        matches!(self, AlgebraNode::BGP(_))
    }

    /// Total number of triple patterns summed across all `BGP` nodes in this sub-tree.
    pub fn triple_count(&self) -> usize {
        match self {
            AlgebraNode::BGP(pats) => pats.len(),
            AlgebraNode::Filter { child, .. }
            | AlgebraNode::Extend { child, .. }
            | AlgebraNode::Project { child, .. }
            | AlgebraNode::Distinct(child)
            | AlgebraNode::OrderBy { child, .. }
            | AlgebraNode::Slice { child, .. }
            | AlgebraNode::Graph { child, .. }
            | AlgebraNode::Group { child, .. } => child.triple_count(),
            AlgebraNode::Join(l, r)
            | AlgebraNode::Union(l, r)
            | AlgebraNode::Minus(l, r) => l.triple_count() + r.triple_count(),
            AlgebraNode::LeftJoin { left, right, .. } => {
                left.triple_count() + right.triple_count()
            }
        }
    }

    /// Returns `true` if this sub-tree contains a `LeftJoin` node (i.e. OPTIONAL).
    pub fn has_optional(&self) -> bool {
        match self {
            AlgebraNode::LeftJoin { .. } => true,
            AlgebraNode::BGP(_) => false,
            AlgebraNode::Filter { child, .. }
            | AlgebraNode::Extend { child, .. }
            | AlgebraNode::Project { child, .. }
            | AlgebraNode::Distinct(child)
            | AlgebraNode::OrderBy { child, .. }
            | AlgebraNode::Slice { child, .. }
            | AlgebraNode::Graph { child, .. }
            | AlgebraNode::Group { child, .. } => child.has_optional(),
            AlgebraNode::Join(l, r)
            | AlgebraNode::Union(l, r)
            | AlgebraNode::Minus(l, r) => l.has_optional() || r.has_optional(),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn var(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }

    fn cst(v: &str) -> Expr {
        Expr::Const(v.to_string())
    }

    fn tp_vv(s: &str, p: &str, o: &str) -> TriplePattern {
        TriplePattern::new(var(s), cst(p), var(o))
    }

    fn single_bgp() -> AlgebraNode {
        AlgebraNode::BGP(vec![tp_vv("s", "p:type", "cls")])
    }

    fn two_bgp() -> AlgebraNode {
        AlgebraNode::BGP(vec![tp_vv("s", "p1", "a"), tp_vv("s", "p2", "b")])
    }

    // ── Expr::variables ───────────────────────────────────────────────────────

    #[test]
    fn test_var_variables() {
        assert_eq!(var("x").variables(), vec!["x"]);
    }

    #[test]
    fn test_const_variables_empty() {
        assert!(cst("http://ex.org/").variables().is_empty());
    }

    #[test]
    fn test_bound_var_variables() {
        let e = Expr::BoundVar("z".to_string());
        assert_eq!(e.variables(), vec!["z"]);
    }

    #[test]
    fn test_eq_variables_sorted_deduped() {
        let e = Expr::Eq(Box::new(var("b")), Box::new(var("a")));
        assert_eq!(e.variables(), vec!["a", "b"]);
    }

    #[test]
    fn test_and_shares_variable() {
        let e = Expr::And(
            Box::new(Expr::Eq(Box::new(var("x")), Box::new(cst("1")))),
            Box::new(Expr::Lt(Box::new(var("x")), Box::new(var("y")))),
        );
        let mut vars = e.variables();
        vars.sort();
        assert_eq!(vars, vec!["x", "y"]);
    }

    #[test]
    fn test_ne_variables() {
        let e = Expr::Ne(Box::new(var("a")), Box::new(var("b")));
        assert_eq!(e.variables(), vec!["a", "b"]);
    }

    #[test]
    fn test_le_variables() {
        let e = Expr::Le(Box::new(var("a")), Box::new(cst("10")));
        assert_eq!(e.variables(), vec!["a"]);
    }

    #[test]
    fn test_ge_variables() {
        let e = Expr::Ge(Box::new(var("a")), Box::new(var("b")));
        assert_eq!(e.variables(), vec!["a", "b"]);
    }

    #[test]
    fn test_not_variables() {
        let e = Expr::Not(Box::new(Expr::BoundVar("x".to_string())));
        assert_eq!(e.variables(), vec!["x"]);
    }

    #[test]
    fn test_or_variables() {
        let e = Expr::Or(Box::new(var("p")), Box::new(var("q")));
        assert_eq!(e.variables(), vec!["p", "q"]);
    }

    #[test]
    fn test_regex_variables() {
        let e = Expr::Regex {
            expr: Box::new(var("name")),
            pattern: "^Alice".to_string(),
            flags: "i".to_string(),
        };
        assert_eq!(e.variables(), vec!["name"]);
    }

    #[test]
    fn test_is_iri_variables() {
        assert_eq!(Expr::IsIri(Box::new(var("s"))).variables(), vec!["s"]);
    }

    #[test]
    fn test_is_literal_variables() {
        assert_eq!(
            Expr::IsLiteral(Box::new(var("o"))).variables(),
            vec!["o"]
        );
    }

    #[test]
    fn test_is_blank_variables() {
        assert_eq!(Expr::IsBlank(Box::new(var("b"))).variables(), vec!["b"]);
    }

    #[test]
    fn test_str_variables() {
        assert_eq!(Expr::Str(Box::new(var("lit"))).variables(), vec!["lit"]);
    }

    #[test]
    fn test_lang_variables() {
        assert_eq!(Expr::Lang(Box::new(var("lit"))).variables(), vec!["lit"]);
    }

    #[test]
    fn test_expr_depth_leaf() {
        assert_eq!(var("x").depth(), 1);
        assert_eq!(cst("y").depth(), 1);
        assert_eq!(Expr::BoundVar("z".to_string()).depth(), 1);
    }

    #[test]
    fn test_expr_depth_unary() {
        assert_eq!(Expr::Not(Box::new(var("x"))).depth(), 2);
    }

    #[test]
    fn test_expr_depth_binary_asymmetric() {
        let e = Expr::And(
            Box::new(var("x")),
            Box::new(Expr::Not(Box::new(Expr::Not(Box::new(var("y")))))),
        );
        // And(1+1=2) → max(1, Not(Not(1))=3) + 1 = 4
        assert_eq!(e.depth(), 4);
    }

    // ── TriplePattern ─────────────────────────────────────────────────────────

    #[test]
    fn test_triple_pattern_variables_sorted() {
        let tp = tp_vv("subject", "p:type", "object");
        let vars = tp.variables();
        assert_eq!(vars, vec!["object", "subject"]);
    }

    #[test]
    fn test_triple_pattern_all_consts_no_vars() {
        let tp = TriplePattern::new(cst("s"), cst("p"), cst("o"));
        assert!(tp.variables().is_empty());
    }

    #[test]
    fn test_triple_pattern_predicate_var() {
        let tp = TriplePattern::new(cst("s"), var("pred"), cst("o"));
        assert_eq!(tp.variables(), vec!["pred"]);
    }

    // ── AlgebraNode::BGP ─────────────────────────────────────────────────────

    #[test]
    fn test_bgp_is_bgp() {
        assert!(single_bgp().is_bgp());
    }

    #[test]
    fn test_bgp_depth_is_1() {
        assert_eq!(single_bgp().depth(), 1);
    }

    #[test]
    fn test_bgp_node_count_is_1() {
        assert_eq!(single_bgp().node_count(), 1);
    }

    #[test]
    fn test_bgp_triple_count_single() {
        assert_eq!(single_bgp().triple_count(), 1);
    }

    #[test]
    fn test_bgp_triple_count_two() {
        assert_eq!(two_bgp().triple_count(), 2);
    }

    #[test]
    fn test_bgp_empty_triple_count() {
        assert_eq!(AlgebraNode::BGP(vec![]).triple_count(), 0);
    }

    #[test]
    fn test_bgp_has_no_optional() {
        assert!(!single_bgp().has_optional());
    }

    #[test]
    fn test_bgp_variables_collected() {
        let node = AlgebraNode::BGP(vec![tp_vv("s", "p:type", "cls"), tp_vv("s", "p:name", "nm")]);
        let vars = node.variables();
        assert!(vars.contains(&"s".to_string()));
        assert!(vars.contains(&"cls".to_string()));
        assert!(vars.contains(&"nm".to_string()));
    }

    // ── Filter ───────────────────────────────────────────────────────────────

    #[test]
    fn test_filter_not_bgp() {
        let f = AlgebraNode::Filter {
            child: Box::new(single_bgp()),
            expr: var("x"),
        };
        assert!(!f.is_bgp());
    }

    #[test]
    fn test_filter_depth() {
        let f = AlgebraNode::Filter {
            child: Box::new(single_bgp()),
            expr: var("x"),
        };
        assert_eq!(f.depth(), 2);
    }

    #[test]
    fn test_filter_node_count() {
        let f = AlgebraNode::Filter {
            child: Box::new(single_bgp()),
            expr: var("x"),
        };
        assert_eq!(f.node_count(), 2);
    }

    #[test]
    fn test_filter_triple_count() {
        let f = AlgebraNode::Filter {
            child: Box::new(two_bgp()),
            expr: var("x"),
        };
        assert_eq!(f.triple_count(), 2);
    }

    #[test]
    fn test_filter_variables_merged() {
        let f = AlgebraNode::Filter {
            child: Box::new(AlgebraNode::BGP(vec![tp_vv("s", "p:cls", "cls")])),
            expr: Expr::Eq(Box::new(var("cls")), Box::new(cst("foaf:Person"))),
        };
        let vars = f.variables();
        assert!(vars.contains(&"s".to_string()));
        assert!(vars.contains(&"cls".to_string()));
    }

    #[test]
    fn test_filter_no_optional() {
        let f = AlgebraNode::Filter {
            child: Box::new(single_bgp()),
            expr: var("x"),
        };
        assert!(!f.has_optional());
    }

    // ── Join ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_join_depth() {
        let j = AlgebraNode::Join(Box::new(single_bgp()), Box::new(single_bgp()));
        assert_eq!(j.depth(), 2);
    }

    #[test]
    fn test_join_node_count() {
        let j = AlgebraNode::Join(Box::new(single_bgp()), Box::new(single_bgp()));
        assert_eq!(j.node_count(), 3);
    }

    #[test]
    fn test_join_triple_count() {
        let j = AlgebraNode::Join(Box::new(two_bgp()), Box::new(single_bgp()));
        assert_eq!(j.triple_count(), 3);
    }

    #[test]
    fn test_join_no_optional() {
        let j = AlgebraNode::Join(Box::new(single_bgp()), Box::new(single_bgp()));
        assert!(!j.has_optional());
    }

    // ── LeftJoin ──────────────────────────────────────────────────────────────

    #[test]
    fn test_left_join_has_optional() {
        let lj = AlgebraNode::LeftJoin {
            left: Box::new(single_bgp()),
            right: Box::new(single_bgp()),
            cond: None,
        };
        assert!(lj.has_optional());
    }

    #[test]
    fn test_left_join_depth_symmetric() {
        let lj = AlgebraNode::LeftJoin {
            left: Box::new(single_bgp()),
            right: Box::new(single_bgp()),
            cond: None,
        };
        assert_eq!(lj.depth(), 2);
    }

    #[test]
    fn test_left_join_depth_asymmetric() {
        let lj = AlgebraNode::LeftJoin {
            left: Box::new(single_bgp()),
            right: Box::new(AlgebraNode::Filter {
                child: Box::new(single_bgp()),
                expr: var("x"),
            }),
            cond: None,
        };
        assert_eq!(lj.depth(), 3);
    }

    #[test]
    fn test_left_join_condition_variables() {
        let lj = AlgebraNode::LeftJoin {
            left: Box::new(AlgebraNode::BGP(vec![tp_vv("s", "p1", "x")])),
            right: Box::new(AlgebraNode::BGP(vec![tp_vv("s", "p2", "y")])),
            cond: Some(Expr::Gt(Box::new(var("x")), Box::new(var("z")))),
        };
        let vars = lj.variables();
        assert!(vars.contains(&"z".to_string()));
    }

    #[test]
    fn test_left_join_triple_count() {
        let lj = AlgebraNode::LeftJoin {
            left: Box::new(two_bgp()),
            right: Box::new(two_bgp()),
            cond: None,
        };
        assert_eq!(lj.triple_count(), 4);
    }

    // ── Union ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_union_node_count() {
        let u = AlgebraNode::Union(Box::new(single_bgp()), Box::new(two_bgp()));
        assert_eq!(u.node_count(), 3);
    }

    #[test]
    fn test_union_triple_count() {
        let u = AlgebraNode::Union(Box::new(single_bgp()), Box::new(two_bgp()));
        assert_eq!(u.triple_count(), 3);
    }

    #[test]
    fn test_union_has_optional_when_branch_has_optional() {
        let lj = AlgebraNode::LeftJoin {
            left: Box::new(single_bgp()),
            right: Box::new(single_bgp()),
            cond: None,
        };
        let u = AlgebraNode::Union(Box::new(single_bgp()), Box::new(lj));
        assert!(u.has_optional());
    }

    // ── Minus ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_minus_no_optional() {
        let m = AlgebraNode::Minus(Box::new(single_bgp()), Box::new(single_bgp()));
        assert!(!m.has_optional());
    }

    #[test]
    fn test_minus_triple_count() {
        let m = AlgebraNode::Minus(Box::new(two_bgp()), Box::new(two_bgp()));
        assert_eq!(m.triple_count(), 4);
    }

    // ── Extend (BIND) ─────────────────────────────────────────────────────────

    #[test]
    fn test_extend_includes_new_var() {
        let e = AlgebraNode::Extend {
            child: Box::new(AlgebraNode::BGP(vec![tp_vv("s", "p:price", "price")])),
            var: "discounted".to_string(),
            expr: Expr::Str(Box::new(var("price"))),
        };
        let vars = e.variables();
        assert!(vars.contains(&"discounted".to_string()));
        assert!(vars.contains(&"price".to_string()));
        assert!(vars.contains(&"s".to_string()));
    }

    #[test]
    fn test_extend_depth() {
        let e = AlgebraNode::Extend {
            child: Box::new(single_bgp()),
            var: "v".to_string(),
            expr: var("x"),
        };
        assert_eq!(e.depth(), 2);
    }

    // ── Project ───────────────────────────────────────────────────────────────

    #[test]
    fn test_project_node_count() {
        let p = AlgebraNode::Project {
            child: Box::new(single_bgp()),
            vars: vec!["s".to_string()],
        };
        assert_eq!(p.node_count(), 2);
    }

    #[test]
    fn test_project_vars_included() {
        let p = AlgebraNode::Project {
            child: Box::new(AlgebraNode::BGP(vec![tp_vv("s", "p:name", "nm")])),
            vars: vec!["s".to_string(), "nm".to_string()],
        };
        let vars = p.variables();
        assert!(vars.contains(&"s".to_string()));
        assert!(vars.contains(&"nm".to_string()));
    }

    // ── Distinct ──────────────────────────────────────────────────────────────

    #[test]
    fn test_distinct_depth() {
        let d = AlgebraNode::Distinct(Box::new(single_bgp()));
        assert_eq!(d.depth(), 2);
    }

    #[test]
    fn test_distinct_triple_count() {
        let d = AlgebraNode::Distinct(Box::new(two_bgp()));
        assert_eq!(d.triple_count(), 2);
    }

    #[test]
    fn test_distinct_not_bgp() {
        let d = AlgebraNode::Distinct(Box::new(single_bgp()));
        assert!(!d.is_bgp());
    }

    // ── OrderBy ───────────────────────────────────────────────────────────────

    #[test]
    fn test_order_by_depth() {
        let o = AlgebraNode::OrderBy {
            child: Box::new(AlgebraNode::Project {
                child: Box::new(single_bgp()),
                vars: vec!["s".to_string()],
            }),
            conditions: vec![(var("s"), true)],
        };
        assert_eq!(o.depth(), 3);
    }

    #[test]
    fn test_order_by_condition_vars() {
        let o = AlgebraNode::OrderBy {
            child: Box::new(single_bgp()),
            conditions: vec![(var("sortkey"), false)],
        };
        let vars = o.variables();
        assert!(vars.contains(&"sortkey".to_string()));
    }

    // ── Slice ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_slice_depth() {
        let s = AlgebraNode::Slice {
            child: Box::new(single_bgp()),
            offset: Some(10),
            limit: Some(20),
        };
        assert_eq!(s.depth(), 2);
    }

    #[test]
    fn test_slice_triple_count() {
        let s = AlgebraNode::Slice {
            child: Box::new(two_bgp()),
            offset: None,
            limit: Some(5),
        };
        assert_eq!(s.triple_count(), 2);
    }

    // ── Graph ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_graph_vars_merged() {
        let g = AlgebraNode::Graph {
            name: var("g"),
            child: Box::new(AlgebraNode::BGP(vec![tp_vv("s", "p:type", "cls")])),
        };
        let vars = g.variables();
        assert!(vars.contains(&"g".to_string()));
        assert!(vars.contains(&"s".to_string()));
        assert!(vars.contains(&"cls".to_string()));
    }

    #[test]
    fn test_graph_const_name_no_extra_var() {
        let g = AlgebraNode::Graph {
            name: cst("http://example.org/g1"),
            child: Box::new(AlgebraNode::BGP(vec![tp_vv("s", "p:type", "cls")])),
        };
        let vars = g.variables();
        assert!(!vars.contains(&"http://example.org/g1".to_string()));
    }

    // ── Group / aggregation ───────────────────────────────────────────────────

    #[test]
    fn test_group_agg_vars() {
        let g = AlgebraNode::Group {
            child: Box::new(AlgebraNode::BGP(vec![tp_vv("s", "p:price", "price")])),
            by: vec!["s".to_string()],
            aggregates: vec![
                ("total".to_string(), AggFunc::Sum(var("price"))),
                ("cnt".to_string(), AggFunc::Count(None)),
            ],
        };
        let vars = g.variables();
        assert!(vars.contains(&"s".to_string()));
        assert!(vars.contains(&"total".to_string()));
        assert!(vars.contains(&"cnt".to_string()));
        assert!(vars.contains(&"price".to_string()));
    }

    #[test]
    fn test_group_concat_vars() {
        let g = AlgebraNode::Group {
            child: Box::new(single_bgp()),
            by: vec![],
            aggregates: vec![(
                "names".to_string(),
                AggFunc::GroupConcat(var("name"), ", ".to_string()),
            )],
        };
        let vars = g.variables();
        assert!(vars.contains(&"name".to_string()));
        assert!(vars.contains(&"names".to_string()));
    }

    #[test]
    fn test_group_min_max_avg_vars() {
        let g = AlgebraNode::Group {
            child: Box::new(AlgebraNode::BGP(vec![tp_vv("s", "p:v", "v")])),
            by: vec!["s".to_string()],
            aggregates: vec![
                ("mn".to_string(), AggFunc::Min(var("v"))),
                ("mx".to_string(), AggFunc::Max(var("v"))),
                ("av".to_string(), AggFunc::Avg(var("v"))),
            ],
        };
        let vars = g.variables();
        assert!(vars.contains(&"mn".to_string()));
        assert!(vars.contains(&"mx".to_string()));
        assert!(vars.contains(&"av".to_string()));
    }

    // ── complex / integration ─────────────────────────────────────────────────

    #[test]
    fn test_complex_query_structure() {
        // SELECT DISTINCT ?s ?name
        // WHERE { ?s a :Person . OPTIONAL { ?s :name ?name } FILTER BOUND(?name) }
        // ORDER BY ?name LIMIT 10
        let main_bgp = AlgebraNode::BGP(vec![tp_vv("s", "rdf:type", "cls")]);
        let opt_bgp = AlgebraNode::BGP(vec![tp_vv("s", "foaf:name", "name")]);
        let lj = AlgebraNode::LeftJoin {
            left: Box::new(main_bgp),
            right: Box::new(opt_bgp),
            cond: None,
        };
        let filtered = AlgebraNode::Filter {
            child: Box::new(lj),
            expr: Expr::BoundVar("name".to_string()),
        };
        let projected = AlgebraNode::Project {
            child: Box::new(filtered),
            vars: vec!["s".to_string(), "name".to_string()],
        };
        let distinct = AlgebraNode::Distinct(Box::new(projected));
        let ordered = AlgebraNode::OrderBy {
            child: Box::new(distinct),
            conditions: vec![(var("name"), true)],
        };
        let sliced = AlgebraNode::Slice {
            child: Box::new(ordered),
            offset: None,
            limit: Some(10),
        };

        assert!(sliced.has_optional());
        assert_eq!(sliced.triple_count(), 2);
        assert_eq!(sliced.depth(), 7);
        assert_eq!(sliced.node_count(), 8);
        let vars = sliced.variables();
        assert!(vars.contains(&"s".to_string()));
        assert!(vars.contains(&"name".to_string()));
    }

    #[test]
    fn test_empty_bgp_no_vars() {
        let node = AlgebraNode::BGP(vec![]);
        assert!(node.variables().is_empty());
        assert_eq!(node.triple_count(), 0);
        assert_eq!(node.depth(), 1);
        assert_eq!(node.node_count(), 1);
    }

    #[test]
    fn test_nested_optional_in_project_slice() {
        let lj = AlgebraNode::LeftJoin {
            left: Box::new(single_bgp()),
            right: Box::new(single_bgp()),
            cond: None,
        };
        let projected = AlgebraNode::Project {
            child: Box::new(lj),
            vars: vec!["s".to_string()],
        };
        let sliced = AlgebraNode::Slice {
            child: Box::new(projected),
            offset: Some(0),
            limit: Some(100),
        };
        assert!(sliced.has_optional());
    }

    #[test]
    fn test_gt_expr_depth_and_variables() {
        let e = Expr::Gt(Box::new(var("a")), Box::new(var("b")));
        assert_eq!(e.depth(), 2);
        assert_eq!(e.variables(), vec!["a", "b"]);
    }
}
