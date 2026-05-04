//! # Federated join decomposition by selectivity
//!
//! Reorders left-deep join chains so that the most-selective leg (the one
//! with the smallest estimated cardinality) is evaluated first, reducing
//! intermediate result sizes that have to flow between endpoints.
//!
//! ## Algorithm
//!
//! 1. Flatten a left-deep `Join(Join(A, B), C)` chain into a list `[A, B, C]`.
//! 2. Estimate each leaf's cardinality using a
//!    [`SourceSelectivityProvider`].
//! 3. Greedy reorder: pick the smallest cardinality first, then the next leg
//!    that produces the smallest intermediate result.
//! 4. Re-build a left-deep tree in that order.
//!
//! ## Scope
//!
//! - Only `Algebra::Join` chains are reordered; LEFT JOIN, MINUS, UNION are
//!   left untouched (their semantics are non-commutative).
//! - We only flatten **left-deep** chains because a *bushy* tree typically
//!   reflects an intent the planner already shaped (e.g. parallelism), and
//!   we don't want to silently undo it.
//! - We do not push/pull subqueries across SERVICE boundaries; that's
//!   handled by `service_merge` and `filter_pushdown`.
//!
//! ## Selectivity heuristics
//!
//! For a `Service(endpoint, Bgp([tp]))` leaf we ask the provider for the
//! selectivity of `tp` at `endpoint`.  For multi-pattern BGPs we sum
//! cardinalities (a coarse but standard approximation).  For non-Service
//! leaves we treat them as "local" and assign a cost based on the BGP size.

use oxirs_arq::algebra::{Algebra, Term, TriplePattern};
use oxirs_arq::optimizer::federated_plan::SourceSelectivityProvider;

/// Walk the tree and reorder all left-deep `Join` chains by estimated
/// cardinality.
pub fn reorder_joins_by_selectivity(
    algebra: Algebra,
    provider: &dyn SourceSelectivityProvider,
) -> Algebra {
    match algebra {
        Algebra::Join { .. } => {
            let legs = flatten_left_deep_join(algebra);
            let recursed: Vec<Algebra> = legs
                .into_iter()
                .map(|a| reorder_joins_by_selectivity(a, provider))
                .collect();
            let scored: Vec<(Algebra, f64)> = recursed
                .into_iter()
                .map(|a| {
                    let s = estimate_cardinality(&a, provider);
                    (a, s)
                })
                .collect();
            rebuild_join_in_selectivity_order(scored)
        }
        Algebra::LeftJoin {
            left,
            right,
            filter,
        } => Algebra::LeftJoin {
            left: Box::new(reorder_joins_by_selectivity(*left, provider)),
            right: Box::new(reorder_joins_by_selectivity(*right, provider)),
            filter,
        },
        Algebra::Union { left, right } => Algebra::Union {
            left: Box::new(reorder_joins_by_selectivity(*left, provider)),
            right: Box::new(reorder_joins_by_selectivity(*right, provider)),
        },
        Algebra::Minus { left, right } => Algebra::Minus {
            left: Box::new(reorder_joins_by_selectivity(*left, provider)),
            right: Box::new(reorder_joins_by_selectivity(*right, provider)),
        },
        Algebra::Filter { pattern, condition } => Algebra::Filter {
            pattern: Box::new(reorder_joins_by_selectivity(*pattern, provider)),
            condition,
        },
        Algebra::Service {
            endpoint,
            pattern,
            silent,
        } => Algebra::Service {
            endpoint,
            pattern: Box::new(reorder_joins_by_selectivity(*pattern, provider)),
            silent,
        },
        Algebra::Graph { graph, pattern } => Algebra::Graph {
            graph,
            pattern: Box::new(reorder_joins_by_selectivity(*pattern, provider)),
        },
        Algebra::Project { pattern, variables } => Algebra::Project {
            pattern: Box::new(reorder_joins_by_selectivity(*pattern, provider)),
            variables,
        },
        Algebra::Distinct { pattern } => Algebra::Distinct {
            pattern: Box::new(reorder_joins_by_selectivity(*pattern, provider)),
        },
        Algebra::Reduced { pattern } => Algebra::Reduced {
            pattern: Box::new(reorder_joins_by_selectivity(*pattern, provider)),
        },
        Algebra::Slice {
            pattern,
            offset,
            limit,
        } => Algebra::Slice {
            pattern: Box::new(reorder_joins_by_selectivity(*pattern, provider)),
            offset,
            limit,
        },
        Algebra::OrderBy {
            pattern,
            conditions,
        } => Algebra::OrderBy {
            pattern: Box::new(reorder_joins_by_selectivity(*pattern, provider)),
            conditions,
        },
        Algebra::Group {
            pattern,
            variables,
            aggregates,
        } => Algebra::Group {
            pattern: Box::new(reorder_joins_by_selectivity(*pattern, provider)),
            variables,
            aggregates,
        },
        Algebra::Having { pattern, condition } => Algebra::Having {
            pattern: Box::new(reorder_joins_by_selectivity(*pattern, provider)),
            condition,
        },
        Algebra::Extend {
            pattern,
            variable,
            expr,
        } => Algebra::Extend {
            pattern: Box::new(reorder_joins_by_selectivity(*pattern, provider)),
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

/// Recursively unwrap a left-deep join chain.  `Join(Join(A, B), C)` → `[A, B, C]`.
/// Right-deep or bushy children are kept as a single leg.
fn flatten_left_deep_join(algebra: Algebra) -> Vec<Algebra> {
    match algebra {
        Algebra::Join { left, right } => {
            let mut left_legs = flatten_left_deep_join(*left);
            left_legs.push(*right);
            left_legs
        }
        other => vec![other],
    }
}

/// Estimate a cardinality for a sub-plan using the selectivity provider.
///
/// - `Service(ep, Bgp([tps]))` → sum of provider's per-pattern cardinalities.
/// - `Bgp([tps])` → coarse estimate using `pattern_count * 1000` as a default.
/// - Other algebra constructs → cardinality of the inner pattern (filters
///   shrink, joins multiply, etc.).
fn estimate_cardinality(algebra: &Algebra, provider: &dyn SourceSelectivityProvider) -> f64 {
    match algebra {
        Algebra::Service {
            endpoint, pattern, ..
        } => {
            let endpoint_str = match endpoint {
                Term::Iri(n) => n.as_str().to_string(),
                _ => return f64::INFINITY,
            };
            estimate_pattern_cardinality(pattern, &endpoint_str, provider)
        }
        Algebra::Bgp(patterns) => {
            // Local BGPs: assume 1000 rows per pattern as a generic fallback.
            (patterns.len() as f64) * 1000.0
        }
        Algebra::Filter { pattern, .. } => estimate_cardinality(pattern, provider) * 0.5,
        Algebra::Project { pattern, .. }
        | Algebra::Distinct { pattern }
        | Algebra::Reduced { pattern }
        | Algebra::OrderBy { pattern, .. }
        | Algebra::Slice { pattern, .. }
        | Algebra::Group { pattern, .. }
        | Algebra::Having { pattern, .. }
        | Algebra::Extend { pattern, .. } => estimate_cardinality(pattern, provider),
        Algebra::Join { left, right } => {
            estimate_cardinality(left, provider) * estimate_cardinality(right, provider) / 100.0
        }
        Algebra::LeftJoin { left, .. } => estimate_cardinality(left, provider),
        Algebra::Union { left, right } => {
            estimate_cardinality(left, provider) + estimate_cardinality(right, provider)
        }
        Algebra::Minus { left, .. } => estimate_cardinality(left, provider),
        Algebra::Graph { pattern, .. } => estimate_cardinality(pattern, provider),
        Algebra::Values { bindings, .. } => bindings.len() as f64,
        Algebra::PropertyPath { .. } => 5000.0,
        Algebra::Table | Algebra::Zero | Algebra::Empty => 0.0,
    }
}

fn estimate_pattern_cardinality(
    algebra: &Algebra,
    endpoint: &str,
    provider: &dyn SourceSelectivityProvider,
) -> f64 {
    match algebra {
        Algebra::Bgp(patterns) => patterns
            .iter()
            .map(|tp| pattern_card(tp, endpoint, provider))
            .sum(),
        Algebra::Filter { pattern, .. } => {
            estimate_pattern_cardinality(pattern, endpoint, provider) * 0.5
        }
        other => estimate_cardinality(other, provider),
    }
}

fn pattern_card(
    tp: &TriplePattern,
    endpoint: &str,
    provider: &dyn SourceSelectivityProvider,
) -> f64 {
    provider
        .pattern_selectivity(endpoint, tp)
        .estimated_cardinality
}

/// Build a left-deep join from `legs` ordered by ascending cardinality.
///
/// `[A_5, B_2, C_8]` (cardinalities subscripted) → `Join(Join(B, A), C)`.
fn rebuild_join_in_selectivity_order(mut scored: Vec<(Algebra, f64)>) -> Algebra {
    if scored.is_empty() {
        return Algebra::Empty;
    }
    if scored.len() == 1 {
        return scored.remove(0).0;
    }

    // Sort ascending by cardinality (most selective first).
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut iter = scored.into_iter();
    let first = match iter.next() {
        Some(f) => f.0,
        None => return Algebra::Empty,
    };
    let mut acc = first;
    for (next_leg, _) in iter {
        acc = Algebra::Join {
            left: Box::new(acc),
            right: Box::new(next_leg),
        };
    }
    acc
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_arq::algebra::{Term, TriplePattern, Variable};
    use oxirs_arq::optimizer::federated_plan::{FederatedSelectivity, StaticSourceProvider};
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

    fn service(endpoint: &str, pattern: Algebra) -> Algebra {
        Algebra::Service {
            endpoint: iri(endpoint),
            pattern: Box::new(pattern),
            silent: false,
        }
    }

    fn build_provider() -> StaticSourceProvider {
        let mut p = StaticSourceProvider::new();
        p.register(
            "http://a.example",
            "http://a.example/sparql",
            FederatedSelectivity {
                estimated_cardinality: 100.0,
                estimated_latency_ms: 50.0,
                confidence: 0.9,
            },
        );
        p.register(
            "http://b.example",
            "http://b.example/sparql",
            FederatedSelectivity {
                estimated_cardinality: 5_000.0,
                estimated_latency_ms: 100.0,
                confidence: 0.9,
            },
        );
        p.register(
            "http://c.example",
            "http://c.example/sparql",
            FederatedSelectivity {
                estimated_cardinality: 1_000.0,
                estimated_latency_ms: 80.0,
                confidence: 0.9,
            },
        );
        p
    }

    #[test]
    fn flatten_left_deep_two() {
        let join = Algebra::Join {
            left: Box::new(bgp("s", "http://p", "o")),
            right: Box::new(bgp("a", "http://q", "b")),
        };
        let legs = flatten_left_deep_join(join);
        assert_eq!(legs.len(), 2);
    }

    #[test]
    fn flatten_left_deep_three() {
        let inner = Algebra::Join {
            left: Box::new(bgp("s", "http://p", "o")),
            right: Box::new(bgp("a", "http://q", "b")),
        };
        let outer = Algebra::Join {
            left: Box::new(inner),
            right: Box::new(bgp("c", "http://r", "d")),
        };
        let legs = flatten_left_deep_join(outer);
        assert_eq!(legs.len(), 3);
    }

    #[test]
    fn flatten_keeps_right_deep_intact() {
        let inner = Algebra::Join {
            left: Box::new(bgp("a", "http://q", "b")),
            right: Box::new(bgp("c", "http://r", "d")),
        };
        let outer = Algebra::Join {
            left: Box::new(bgp("s", "http://p", "o")),
            right: Box::new(inner),
        };
        let legs = flatten_left_deep_join(outer);
        // Right-deep: only 2 legs at the top — the inner join stays as one leg.
        assert_eq!(legs.len(), 2);
    }

    #[test]
    fn rebuild_single_leg_returns_self() {
        let only = bgp("s", "http://p", "o");
        let rebuilt = rebuild_join_in_selectivity_order(vec![(only.clone(), 1.0)]);
        assert_eq!(rebuilt, only);
    }

    #[test]
    fn rebuild_orders_ascending() {
        let bg = |i: usize| bgp(&format!("s{i}"), "http://p", "o");
        let scored = vec![(bg(0), 100.0), (bg(1), 10.0), (bg(2), 50.0)];
        let join = rebuild_join_in_selectivity_order(scored);
        // First (most selective) is bg(1), then bg(2), then bg(0).
        if let Algebra::Join {
            left: outer_left,
            right: outer_right,
        } = join
        {
            assert!(matches!(*outer_right, Algebra::Bgp(_)));
            // outer_left must be Join(bg(1), bg(2))
            if let Algebra::Join {
                left: inner_left,
                right: inner_right,
            } = *outer_left
            {
                // inner_left is bg(1) (smallest)
                if let Algebra::Bgp(pats) = *inner_left {
                    if let Term::Variable(v) = &pats[0].subject {
                        assert_eq!(v.name(), "s1");
                    } else {
                        panic!("expected Variable at inner_left subject");
                    }
                } else {
                    panic!("expected Bgp at inner_left");
                }
                if let Algebra::Bgp(pats) = *inner_right {
                    if let Term::Variable(v) = &pats[0].subject {
                        assert_eq!(v.name(), "s2");
                    } else {
                        panic!("expected Variable at inner_right subject");
                    }
                } else {
                    panic!("expected Bgp at inner_right");
                }
            } else {
                panic!("expected nested Join");
            }
        } else {
            panic!("expected Join at root");
        }
    }

    #[test]
    fn reorder_three_services_by_selectivity() {
        let provider = build_provider();
        // Original order: a (small), b (large), c (medium)
        let s_a = service(
            "http://a.example/sparql",
            bgp("s", "http://a.example/p", "o"),
        );
        let s_b = service(
            "http://b.example/sparql",
            bgp("s", "http://b.example/p", "o"),
        );
        let s_c = service(
            "http://c.example/sparql",
            bgp("s", "http://c.example/p", "o"),
        );
        // Build join: ((s_b ⨝ s_c) ⨝ s_a) — backwards on purpose
        let join = Algebra::Join {
            left: Box::new(Algebra::Join {
                left: Box::new(s_b),
                right: Box::new(s_c),
            }),
            right: Box::new(s_a.clone()),
        };
        let reordered = reorder_joins_by_selectivity(join, &provider);
        // Most selective leg should be at the deepest left position.
        // We expect s_a (cardinality 100) to be first.
        let mut walker: &Algebra = &reordered;
        loop {
            match walker {
                Algebra::Join { left, .. } => walker = left,
                Algebra::Service { endpoint, .. } => {
                    assert_eq!(
                        endpoint.to_string(),
                        iri("http://a.example/sparql").to_string()
                    );
                    break;
                }
                other => panic!("unexpected leg: {other:?}"),
            }
        }
    }

    #[test]
    fn no_op_for_single_service() {
        let provider = build_provider();
        let s = service(
            "http://a.example/sparql",
            bgp("s", "http://a.example/p", "o"),
        );
        let out = reorder_joins_by_selectivity(s.clone(), &provider);
        assert_eq!(out, s);
    }

    #[test]
    fn does_not_touch_left_join() {
        let provider = build_provider();
        let s_a = service(
            "http://a.example/sparql",
            bgp("s", "http://a.example/p", "o"),
        );
        let s_b = service(
            "http://b.example/sparql",
            bgp("s", "http://b.example/p", "o"),
        );
        let lj = Algebra::LeftJoin {
            left: Box::new(s_a),
            right: Box::new(s_b),
            filter: None,
        };
        let out = reorder_joins_by_selectivity(lj.clone(), &provider);
        assert_eq!(out, lj);
    }

    #[test]
    fn pattern_card_falls_back_to_default() {
        let provider = StaticSourceProvider::new();
        let tp = TriplePattern {
            subject: Term::Variable(var("s")),
            predicate: iri("http://p"),
            object: Term::Variable(var("o")),
        };
        let c = pattern_card(&tp, "http://nowhere", &provider);
        // Default selectivity gives 1000 cardinality.
        assert!(c > 0.0);
    }

    #[test]
    fn estimate_cardinality_local_bgp() {
        let provider = StaticSourceProvider::new();
        let bg = Algebra::Bgp(vec![
            TriplePattern {
                subject: Term::Variable(var("s")),
                predicate: iri("http://p"),
                object: Term::Variable(var("o")),
            },
            TriplePattern {
                subject: Term::Variable(var("s")),
                predicate: iri("http://q"),
                object: Term::Variable(var("o")),
            },
        ]);
        let c = estimate_cardinality(&bg, &provider);
        assert!(c > 0.0);
    }

    #[test]
    fn estimate_cardinality_filter_shrinks() {
        let provider = StaticSourceProvider::new();
        let bg = bgp("s", "http://p", "o");
        let c_bg = estimate_cardinality(&bg, &provider);
        let f = Algebra::Filter {
            pattern: Box::new(bg),
            condition: oxirs_arq::algebra::Expression::Literal(oxirs_arq::algebra::Literal {
                value: "true".into(),
                language: None,
                datatype: None,
            }),
        };
        let c_f = estimate_cardinality(&f, &provider);
        assert!(c_f < c_bg + 1e-6);
    }

    #[test]
    fn rebuild_empty_returns_empty() {
        let out = rebuild_join_in_selectivity_order(Vec::new());
        assert_eq!(out, Algebra::Empty);
    }
}
