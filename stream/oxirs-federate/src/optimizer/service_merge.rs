//! # Adjacent SERVICE merge
//!
//! Implements the rewrite
//!
//! ```text
//!     Join(Service(A, p1), Service(A, p2))
//!         ⇒  Service(A, Join(p1, p2))
//! ```
//!
//! when both services target the **same endpoint** with the **same SILENT
//! flag**.  Merging multiple round-trips into one reduces network overhead
//! and lets the remote endpoint optimize the joined sub-query locally.
//!
//! The rule is applied bottom-up: nested merges fold left, producing a
//! left-deep `Join(Join(p1, p2), p3)` inside one service when three or more
//! adjacent service nodes share an endpoint.
//!
//! ## When NOT to merge
//!
//! - Different endpoints: `Service(A, p1) ⨝ Service(B, p2)` cannot be merged.
//! - Different `SILENT` flags: merging would change error semantics for the
//!   non-silent leg, so we keep them separate.
//! - LEFT JOIN, MINUS, UNION: these constructs have non-symmetric or
//!   negation-style semantics.  Only the inner join (`Algebra::Join`) is
//!   handled.

use oxirs_arq::algebra::{Algebra, Term};

/// Walk the tree and merge adjacent `Service(A, ·)` joins.
pub fn merge_adjacent_services(algebra: Algebra) -> Algebra {
    let walked = walk(algebra);
    // A second pass collapses any newly-adjacent pairs that became visible
    // after a child rewrote into a Service node.  Run until fixed point.
    let mut current = walked;
    loop {
        let next = walk(current.clone());
        if next == current {
            return current;
        }
        current = next;
    }
}

fn walk(algebra: Algebra) -> Algebra {
    match algebra {
        Algebra::Join { left, right } => {
            let l = walk(*left);
            let r = walk(*right);
            try_merge_join(l, r)
        }
        Algebra::LeftJoin {
            left,
            right,
            filter,
        } => Algebra::LeftJoin {
            left: Box::new(walk(*left)),
            right: Box::new(walk(*right)),
            filter,
        },
        Algebra::Union { left, right } => Algebra::Union {
            left: Box::new(walk(*left)),
            right: Box::new(walk(*right)),
        },
        Algebra::Minus { left, right } => Algebra::Minus {
            left: Box::new(walk(*left)),
            right: Box::new(walk(*right)),
        },
        Algebra::Filter { pattern, condition } => Algebra::Filter {
            pattern: Box::new(walk(*pattern)),
            condition,
        },
        Algebra::Service {
            endpoint,
            pattern,
            silent,
        } => Algebra::Service {
            endpoint,
            pattern: Box::new(walk(*pattern)),
            silent,
        },
        Algebra::Graph { graph, pattern } => Algebra::Graph {
            graph,
            pattern: Box::new(walk(*pattern)),
        },
        Algebra::Project { pattern, variables } => Algebra::Project {
            pattern: Box::new(walk(*pattern)),
            variables,
        },
        Algebra::Distinct { pattern } => Algebra::Distinct {
            pattern: Box::new(walk(*pattern)),
        },
        Algebra::Reduced { pattern } => Algebra::Reduced {
            pattern: Box::new(walk(*pattern)),
        },
        Algebra::Slice {
            pattern,
            offset,
            limit,
        } => Algebra::Slice {
            pattern: Box::new(walk(*pattern)),
            offset,
            limit,
        },
        Algebra::OrderBy {
            pattern,
            conditions,
        } => Algebra::OrderBy {
            pattern: Box::new(walk(*pattern)),
            conditions,
        },
        Algebra::Group {
            pattern,
            variables,
            aggregates,
        } => Algebra::Group {
            pattern: Box::new(walk(*pattern)),
            variables,
            aggregates,
        },
        Algebra::Having { pattern, condition } => Algebra::Having {
            pattern: Box::new(walk(*pattern)),
            condition,
        },
        Algebra::Extend {
            pattern,
            variable,
            expr,
        } => Algebra::Extend {
            pattern: Box::new(walk(*pattern)),
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

fn try_merge_join(left: Algebra, right: Algebra) -> Algebra {
    if let (
        Algebra::Service {
            endpoint: ep_l,
            pattern: _pat_l,
            silent: silent_l,
        },
        Algebra::Service {
            endpoint: ep_r,
            pattern: _pat_r,
            silent: silent_r,
        },
    ) = (&left, &right)
    {
        if endpoints_equal(ep_l, ep_r) && silent_l == silent_r {
            // Pull data out of the patterns so we can build a single Service.
            // We need to clone here because the destructuring above only
            // borrowed; but we own the original `left` and `right`.
            let (ep, pat_l_owned, silent) = match left {
                Algebra::Service {
                    endpoint,
                    pattern,
                    silent,
                } => (endpoint, *pattern, silent),
                other => return rebuild_join(other, right),
            };
            let pat_r_owned = match right {
                Algebra::Service { pattern, .. } => *pattern,
                other => {
                    return rebuild_join(
                        Algebra::Service {
                            endpoint: ep,
                            pattern: Box::new(pat_l_owned),
                            silent,
                        },
                        other,
                    )
                }
            };
            return Algebra::Service {
                endpoint: ep,
                pattern: Box::new(Algebra::Join {
                    left: Box::new(pat_l_owned),
                    right: Box::new(pat_r_owned),
                }),
                silent,
            };
        }
    }
    rebuild_join(left, right)
}

fn rebuild_join(left: Algebra, right: Algebra) -> Algebra {
    Algebra::Join {
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn endpoints_equal(a: &Term, b: &Term) -> bool {
    a == b
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_arq::algebra::{Term, TriplePattern, Variable};
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

    fn service(endpoint: &str, pattern: Algebra, silent: bool) -> Algebra {
        Algebra::Service {
            endpoint: iri(endpoint),
            pattern: Box::new(pattern),
            silent,
        }
    }

    #[test]
    fn merges_two_services_same_endpoint() {
        let s1 = service(
            "http://a.example/sparql",
            bgp("s1", "http://p1", "o1"),
            false,
        );
        let s2 = service(
            "http://a.example/sparql",
            bgp("s2", "http://p2", "o2"),
            false,
        );
        let join = Algebra::Join {
            left: Box::new(s1),
            right: Box::new(s2),
        };
        let merged = merge_adjacent_services(join);
        match merged {
            Algebra::Service {
                pattern, silent, ..
            } => {
                assert!(!silent);
                assert!(matches!(*pattern, Algebra::Join { .. }));
            }
            other => panic!("expected merged Service, got {other:?}"),
        }
    }

    #[test]
    fn does_not_merge_different_endpoints() {
        let s1 = service("http://a.example/sparql", bgp("s", "http://p", "o"), false);
        let s2 = service("http://b.example/sparql", bgp("s", "http://q", "o"), false);
        let join = Algebra::Join {
            left: Box::new(s1.clone()),
            right: Box::new(s2.clone()),
        };
        let merged = merge_adjacent_services(join.clone());
        match merged {
            Algebra::Join { left, right } => {
                assert!(matches!(*left, Algebra::Service { .. }));
                assert!(matches!(*right, Algebra::Service { .. }));
            }
            other => panic!("expected Join unchanged, got {other:?}"),
        }
    }

    #[test]
    fn does_not_merge_different_silent_flags() {
        let s1 = service("http://a.example", bgp("s", "http://p", "o"), false);
        let s2 = service("http://a.example", bgp("a", "http://q", "b"), true);
        let join = Algebra::Join {
            left: Box::new(s1),
            right: Box::new(s2),
        };
        let merged = merge_adjacent_services(join);
        assert!(matches!(merged, Algebra::Join { .. }));
    }

    #[test]
    fn merges_three_services_to_left_deep_join() {
        let mk = |sub: &str, p: &str, obj: &str| {
            service("http://a.example/sparql", bgp(sub, p, obj), false)
        };
        let s1 = mk("s", "http://p1", "o1");
        let s2 = mk("s", "http://p2", "o2");
        let s3 = mk("s", "http://p3", "o3");
        // Left-deep join: ((s1 ⨝ s2) ⨝ s3)
        let join = Algebra::Join {
            left: Box::new(Algebra::Join {
                left: Box::new(s1),
                right: Box::new(s2),
            }),
            right: Box::new(s3),
        };
        let merged = merge_adjacent_services(join);
        match merged {
            Algebra::Service { pattern, .. } => {
                // Should be Join(Join(p1, p2), p3) inside one service.
                if let Algebra::Join {
                    left: outer_left,
                    right: outer_right,
                } = *pattern
                {
                    assert!(matches!(*outer_left, Algebra::Join { .. }));
                    assert!(matches!(*outer_right, Algebra::Bgp(_)));
                } else {
                    panic!("expected nested Join inside merged Service");
                }
            }
            other => panic!("expected merged Service, got {other:?}"),
        }
    }

    #[test]
    fn merges_with_silent_true_consistently() {
        let s1 = service("http://a.example", bgp("s", "http://p", "o"), true);
        let s2 = service("http://a.example", bgp("a", "http://q", "b"), true);
        let join = Algebra::Join {
            left: Box::new(s1),
            right: Box::new(s2),
        };
        let merged = merge_adjacent_services(join);
        if let Algebra::Service { silent, .. } = merged {
            assert!(silent);
        } else {
            panic!("expected merged Service with silent=true");
        }
    }

    #[test]
    fn no_merge_for_left_join() {
        let s1 = service("http://a.example", bgp("s", "http://p", "o"), false);
        let s2 = service("http://a.example", bgp("a", "http://q", "b"), false);
        let optional = Algebra::LeftJoin {
            left: Box::new(s1),
            right: Box::new(s2),
            filter: None,
        };
        let result = merge_adjacent_services(optional);
        // OPTIONAL is not symmetric — leave alone.
        assert!(matches!(result, Algebra::LeftJoin { .. }));
    }

    #[test]
    fn no_merge_for_union() {
        let s1 = service("http://a.example", bgp("s", "http://p", "o"), false);
        let s2 = service("http://a.example", bgp("a", "http://q", "b"), false);
        let union = Algebra::Union {
            left: Box::new(s1),
            right: Box::new(s2),
        };
        let result = merge_adjacent_services(union);
        assert!(matches!(result, Algebra::Union { .. }));
    }

    #[test]
    fn no_merge_for_minus() {
        let s1 = service("http://a.example", bgp("s", "http://p", "o"), false);
        let s2 = service("http://a.example", bgp("a", "http://q", "b"), false);
        let minus = Algebra::Minus {
            left: Box::new(s1),
            right: Box::new(s2),
        };
        let result = merge_adjacent_services(minus);
        assert!(matches!(result, Algebra::Minus { .. }));
    }

    #[test]
    fn merge_inside_filter() {
        // Filter(true, Join(Service(A, p1), Service(A, p2)))
        let s1 = service("http://a.example", bgp("s", "http://p", "o"), false);
        let s2 = service("http://a.example", bgp("a", "http://q", "b"), false);
        let join = Algebra::Join {
            left: Box::new(s1),
            right: Box::new(s2),
        };
        let filter = Algebra::Filter {
            pattern: Box::new(join),
            condition: oxirs_arq::algebra::Expression::Literal(oxirs_arq::algebra::Literal {
                value: "true".into(),
                language: None,
                datatype: None,
            }),
        };
        let merged = merge_adjacent_services(filter);
        match merged {
            Algebra::Filter { pattern, .. } => {
                assert!(matches!(*pattern, Algebra::Service { .. }));
            }
            other => panic!("expected Filter wrapping Service, got {other:?}"),
        }
    }

    #[test]
    fn merge_no_op_for_non_service_join() {
        let join = Algebra::Join {
            left: Box::new(bgp("s", "http://p1", "o")),
            right: Box::new(bgp("s", "http://p2", "o")),
        };
        let result = merge_adjacent_services(join.clone());
        assert_eq!(result, join);
    }

    #[test]
    fn merge_idempotent() {
        let s1 = service("http://a.example", bgp("s", "http://p", "o"), false);
        let s2 = service("http://a.example", bgp("a", "http://q", "b"), false);
        let join = Algebra::Join {
            left: Box::new(s1),
            right: Box::new(s2),
        };
        let once = merge_adjacent_services(join);
        let twice = merge_adjacent_services(once.clone());
        assert_eq!(once, twice);
    }
}
