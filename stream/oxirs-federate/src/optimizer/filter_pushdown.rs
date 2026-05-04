//! # Filter pushdown into SERVICE clauses
//!
//! Implements the SPARQL 1.1 Federation rewrite rule
//!
//! ```text
//!     Filter(Service(s, p), filter)
//!         ⇒  Service(s, Filter(p, filter))
//! ```
//!
//! when the filter expression's free variables are bound entirely by `p`.
//!
//! ## Why this is safe
//!
//! - The output bindings of `Service(p)` are exactly the bindings produced by
//!   the remote endpoint when evaluating `p`.  If a filter only mentions
//!   variables that `p` binds, evaluating it remotely produces the same
//!   solutions as evaluating it locally — the difference is just *where* the
//!   filter runs.
//! - Pushing the filter down reduces the size of the bag transmitted over the
//!   wire, so total work shrinks.
//!
//! ## Why this is sometimes blocked
//!
//! - If the filter references a variable bound *outside* the SERVICE clause
//!   (e.g. by a join with a local pattern), the filter cannot be pushed.  In
//!   that case the original `Filter(Service(...))` is preserved.
//! - We do **not** push filters into `SILENT` services in a way that could
//!   change visible error semantics; for the algebraic rewrite the SILENT flag
//!   is preserved verbatim.
//!
//! ## Combinations and conjunctions
//!
//! When a filter expression is a top-level conjunction (`A && B`), each
//! conjunct is considered independently.  A pushable conjunct is moved into
//! the service while non-pushable conjuncts remain at the outer filter.

use oxirs_arq::algebra::{Algebra, BinaryOperator, Expression};

use super::{bound_variables, expression_free_vars};

/// Apply filter pushdown to every SERVICE node in the tree.  Recurses into
/// every algebra constructor.
pub fn push_filters_into_services(algebra: Algebra) -> Algebra {
    match algebra {
        Algebra::Filter { pattern, condition } => rewrite_filter(*pattern, condition),
        Algebra::Join { left, right } => Algebra::Join {
            left: Box::new(push_filters_into_services(*left)),
            right: Box::new(push_filters_into_services(*right)),
        },
        Algebra::LeftJoin {
            left,
            right,
            filter,
        } => Algebra::LeftJoin {
            left: Box::new(push_filters_into_services(*left)),
            right: Box::new(push_filters_into_services(*right)),
            filter,
        },
        Algebra::Union { left, right } => Algebra::Union {
            left: Box::new(push_filters_into_services(*left)),
            right: Box::new(push_filters_into_services(*right)),
        },
        Algebra::Minus { left, right } => Algebra::Minus {
            left: Box::new(push_filters_into_services(*left)),
            right: Box::new(push_filters_into_services(*right)),
        },
        Algebra::Service {
            endpoint,
            pattern,
            silent,
        } => Algebra::Service {
            endpoint,
            pattern: Box::new(push_filters_into_services(*pattern)),
            silent,
        },
        Algebra::Graph { graph, pattern } => Algebra::Graph {
            graph,
            pattern: Box::new(push_filters_into_services(*pattern)),
        },
        Algebra::Project { pattern, variables } => Algebra::Project {
            pattern: Box::new(push_filters_into_services(*pattern)),
            variables,
        },
        Algebra::Distinct { pattern } => Algebra::Distinct {
            pattern: Box::new(push_filters_into_services(*pattern)),
        },
        Algebra::Reduced { pattern } => Algebra::Reduced {
            pattern: Box::new(push_filters_into_services(*pattern)),
        },
        Algebra::Slice {
            pattern,
            offset,
            limit,
        } => Algebra::Slice {
            pattern: Box::new(push_filters_into_services(*pattern)),
            offset,
            limit,
        },
        Algebra::OrderBy {
            pattern,
            conditions,
        } => Algebra::OrderBy {
            pattern: Box::new(push_filters_into_services(*pattern)),
            conditions,
        },
        Algebra::Group {
            pattern,
            variables,
            aggregates,
        } => Algebra::Group {
            pattern: Box::new(push_filters_into_services(*pattern)),
            variables,
            aggregates,
        },
        Algebra::Having { pattern, condition } => Algebra::Having {
            pattern: Box::new(push_filters_into_services(*pattern)),
            condition,
        },
        Algebra::Extend {
            pattern,
            variable,
            expr,
        } => Algebra::Extend {
            pattern: Box::new(push_filters_into_services(*pattern)),
            variable,
            expr,
        },
        // Leaves
        Algebra::Bgp(_)
        | Algebra::PropertyPath { .. }
        | Algebra::Values { .. }
        | Algebra::Table
        | Algebra::Zero
        | Algebra::Empty => algebra,
    }
}

/// Decide what to do with a single `Filter(pattern, condition)`.
///
/// 1. Recurse into `pattern` first so any nested SERVICE clauses get their
///    own filters pushed down.
/// 2. If the (recursed) pattern is a SERVICE, attempt to push pushable
///    conjuncts of the filter into the service body.
/// 3. Otherwise leave the filter wrapping its (recursed) child.
fn rewrite_filter(pattern: Algebra, condition: Expression) -> Algebra {
    let inner = push_filters_into_services(pattern);
    match inner {
        Algebra::Service {
            endpoint,
            pattern,
            silent,
        } => {
            let inner_vars = bound_variables(&pattern);
            let conjuncts = split_conjuncts(condition);
            let mut pushable = Vec::new();
            let mut residual = Vec::new();
            for c in conjuncts {
                let frees = expression_free_vars(&c);
                if frees.iter().all(|v| inner_vars.contains(v)) {
                    pushable.push(c);
                } else {
                    residual.push(c);
                }
            }

            // Build new SERVICE with pushed filters wrapped around its pattern.
            let mut new_pattern: Algebra = *pattern;
            for c in pushable {
                new_pattern = Algebra::Filter {
                    pattern: Box::new(new_pattern),
                    condition: c,
                };
            }
            let new_service = Algebra::Service {
                endpoint,
                pattern: Box::new(new_pattern),
                silent,
            };

            // Re-attach any non-pushable conjuncts at the outer level.
            wrap_with_conjuncts(new_service, residual)
        }
        other => Algebra::Filter {
            pattern: Box::new(other),
            condition,
        },
    }
}

/// Split a top-level conjunction `A && B && C` into `[A, B, C]`.  Other
/// expressions are returned as singletons.
fn split_conjuncts(expr: Expression) -> Vec<Expression> {
    let mut out = Vec::new();
    split_into(&expr, &mut out);
    out
}

fn split_into(expr: &Expression, out: &mut Vec<Expression>) {
    if let Expression::Binary {
        op: BinaryOperator::And,
        left,
        right,
    } = expr
    {
        split_into(left, out);
        split_into(right, out);
    } else {
        out.push(expr.clone());
    }
}

/// Re-AND a list of conjuncts on top of `base`.  Empty list ⇒ `base`.
fn wrap_with_conjuncts(base: Algebra, conjuncts: Vec<Expression>) -> Algebra {
    if conjuncts.is_empty() {
        return base;
    }
    let mut iter = conjuncts.into_iter();
    let mut combined = match iter.next() {
        Some(first) => first,
        None => return base,
    };
    for c in iter {
        combined = Expression::Binary {
            op: BinaryOperator::And,
            left: Box::new(combined),
            right: Box::new(c),
        };
    }
    Algebra::Filter {
        pattern: Box::new(base),
        condition: combined,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_arq::algebra::{Literal, Term, TriplePattern, Variable};
    use oxirs_core::model::NamedNode;

    fn var(s: &str) -> Variable {
        Variable::new(s).expect("valid var name")
    }

    fn iri(s: &str) -> Term {
        Term::Iri(NamedNode::new_unchecked(s))
    }

    fn bgp(subj: &str, pred: &str, obj: &str) -> Algebra {
        Algebra::Bgp(vec![TriplePattern {
            subject: Term::Variable(var(subj)),
            predicate: iri(pred),
            object: Term::Variable(var(obj)),
        }])
    }

    fn eq_expr(v: &str, value: &str) -> Expression {
        Expression::Binary {
            op: BinaryOperator::Equal,
            left: Box::new(Expression::Variable(var(v))),
            right: Box::new(Expression::Literal(Literal {
                value: value.into(),
                language: None,
                datatype: None,
            })),
        }
    }

    fn and_expr(a: Expression, b: Expression) -> Expression {
        Expression::Binary {
            op: BinaryOperator::And,
            left: Box::new(a),
            right: Box::new(b),
        }
    }

    fn service(endpoint: &str, pattern: Algebra, silent: bool) -> Algebra {
        Algebra::Service {
            endpoint: iri(endpoint),
            pattern: Box::new(pattern),
            silent,
        }
    }

    #[test]
    fn pushable_filter_moves_inside_service() {
        // Filter(?s == "x", Service(http://a, Bgp(?s ?p ?o)))
        let inner = bgp("s", "http://example.org/p", "o");
        let svc = service("http://a.example/sparql", inner, false);
        let outer = Algebra::Filter {
            pattern: Box::new(svc),
            condition: eq_expr("s", "x"),
        };
        let rewritten = push_filters_into_services(outer);
        // Expect: Service(http://a, Filter(Bgp, ?s == "x"))
        match rewritten {
            Algebra::Service {
                pattern, silent, ..
            } => {
                assert!(!silent);
                match *pattern {
                    Algebra::Filter { .. } => {}
                    other => panic!("expected Filter inside Service, got {other:?}"),
                }
            }
            other => panic!("expected Service at root, got {other:?}"),
        }
    }

    #[test]
    fn non_pushable_filter_stays_outside() {
        // Filter on a variable not bound by the service's pattern.
        let inner = bgp("s", "http://example.org/p", "o");
        let svc = service("http://a.example/sparql", inner, false);
        let outer = Algebra::Filter {
            pattern: Box::new(svc),
            condition: eq_expr("foreign", "x"), // ?foreign not in service
        };
        let rewritten = push_filters_into_services(outer.clone());
        // Should remain Filter(Service(...), …)
        match rewritten {
            Algebra::Filter { pattern, .. } => {
                assert!(matches!(*pattern, Algebra::Service { .. }));
            }
            other => panic!("expected Filter at root, got {other:?}"),
        }
    }

    #[test]
    fn split_conjunction_partially() {
        // Filter(?s == "x" AND ?foreign == "y", Service(?s ?p ?o))
        let inner = bgp("s", "http://example.org/p", "o");
        let svc = service("http://a.example/sparql", inner, false);
        let cond = and_expr(eq_expr("s", "x"), eq_expr("foreign", "y"));
        let outer = Algebra::Filter {
            pattern: Box::new(svc),
            condition: cond,
        };
        let rewritten = push_filters_into_services(outer);
        // Expect: Filter(Service(Filter(Bgp, ?s == "x")), ?foreign == "y")
        match rewritten {
            Algebra::Filter {
                pattern: outer_pat,
                condition: outer_cond,
            } => {
                // The outer condition should reference ?foreign
                let frees = super::expression_free_vars(&outer_cond);
                assert!(frees.contains(&var("foreign")));
                assert!(!frees.contains(&var("s")));
                // Inside, a Service wrapping a Filter
                match *outer_pat {
                    Algebra::Service { pattern, .. } => {
                        assert!(matches!(*pattern, Algebra::Filter { .. }));
                    }
                    other => panic!("expected Service, got {other:?}"),
                }
            }
            other => panic!("expected outer Filter, got {other:?}"),
        }
    }

    #[test]
    fn pushdown_preserves_silent_flag() {
        let inner = bgp("s", "http://example.org/p", "o");
        let svc = service("http://a.example/sparql", inner, true);
        let outer = Algebra::Filter {
            pattern: Box::new(svc),
            condition: eq_expr("s", "x"),
        };
        let rewritten = push_filters_into_services(outer);
        match rewritten {
            Algebra::Service { silent, .. } => assert!(silent),
            other => panic!("expected Service silent=true, got {other:?}"),
        }
    }

    #[test]
    fn pushdown_recurses_into_join() {
        // Join(Filter(Service(p1)), Filter(Service(p2)))
        let inner1 = bgp("s", "http://p", "o");
        let inner2 = bgp("a", "http://q", "b");
        let svc1 = service("http://x.example", inner1, false);
        let svc2 = service("http://y.example", inner2, false);
        let f1 = Algebra::Filter {
            pattern: Box::new(svc1),
            condition: eq_expr("s", "1"),
        };
        let f2 = Algebra::Filter {
            pattern: Box::new(svc2),
            condition: eq_expr("a", "2"),
        };
        let join = Algebra::Join {
            left: Box::new(f1),
            right: Box::new(f2),
        };
        let rewritten = push_filters_into_services(join);
        match rewritten {
            Algebra::Join { left, right } => {
                assert!(matches!(*left, Algebra::Service { .. }));
                assert!(matches!(*right, Algebra::Service { .. }));
            }
            other => panic!("expected Join, got {other:?}"),
        }
    }

    #[test]
    fn pushdown_no_op_for_bgp() {
        let bg = bgp("s", "http://p", "o");
        let out = push_filters_into_services(bg.clone());
        assert_eq!(out, bg);
    }

    #[test]
    fn pushdown_no_op_for_filter_over_bgp() {
        let bg = bgp("s", "http://p", "o");
        let f = Algebra::Filter {
            pattern: Box::new(bg),
            condition: eq_expr("s", "x"),
        };
        let out = push_filters_into_services(f.clone());
        assert_eq!(out, f);
    }

    #[test]
    fn empty_filter_no_op() {
        // No conjuncts pushable => filter stays
        let svc = service("http://a.example", bgp("s", "http://p", "o"), false);
        let out = push_filters_into_services(svc.clone());
        assert_eq!(out, svc);
    }

    #[test]
    fn pushdown_handles_optional() {
        let inner = bgp("s", "http://p", "o");
        let svc = service("http://a.example", inner, false);
        let outer = Algebra::LeftJoin {
            left: Box::new(bgp("a", "http://q", "b")),
            right: Box::new(Algebra::Filter {
                pattern: Box::new(svc),
                condition: eq_expr("s", "x"),
            }),
            filter: None,
        };
        let rewritten = push_filters_into_services(outer);
        match rewritten {
            Algebra::LeftJoin { right, .. } => {
                assert!(matches!(*right, Algebra::Service { .. }));
            }
            other => panic!("expected LeftJoin, got {other:?}"),
        }
    }

    #[test]
    fn pushdown_handles_minus() {
        let inner = bgp("s", "http://p", "o");
        let svc = service("http://a.example", inner, false);
        let outer = Algebra::Minus {
            left: Box::new(bgp("a", "http://q", "b")),
            right: Box::new(Algebra::Filter {
                pattern: Box::new(svc),
                condition: eq_expr("s", "x"),
            }),
        };
        let rewritten = push_filters_into_services(outer);
        match rewritten {
            Algebra::Minus { right, .. } => {
                assert!(matches!(*right, Algebra::Service { .. }));
            }
            other => panic!("expected Minus, got {other:?}"),
        }
    }

    #[test]
    fn split_conjuncts_smoke() {
        let e = and_expr(
            and_expr(eq_expr("a", "1"), eq_expr("b", "2")),
            eq_expr("c", "3"),
        );
        let parts = split_conjuncts(e);
        assert_eq!(parts.len(), 3);
    }

    #[test]
    fn wrap_with_conjuncts_empty_returns_base() {
        let base = bgp("s", "http://p", "o");
        let out = wrap_with_conjuncts(base.clone(), Vec::new());
        assert_eq!(out, base);
    }
}
