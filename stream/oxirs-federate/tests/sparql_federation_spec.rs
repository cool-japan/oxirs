//! SPARQL 1.1 Federation specification driver tests for the
//! `oxirs_federate::optimizer` and `oxirs_federate::cost_model`
//! components.
//!
//! These tests are *algebraic* in nature: they construct an
//! [`Algebra`](oxirs_arq::algebra::Algebra) tree representing each spec
//! scenario, run the relevant rewrites through
//! [`OptimizerPipeline`](oxirs_federate::optimizer::OptimizerPipeline) (or
//! individual passes), and assert that the resulting tree shape, plus the
//! [`FederationCostModel`](oxirs_federate::cost_model::FederationCostModel)
//! cost estimate, satisfy the spec invariant.
//!
//! ## Categories covered
//!
//! - **SC**: SERVICE clause forms (named, variable endpoint, SILENT).
//! - **FP**: Filter pushdown into SERVICE.
//! - **SM**: Adjacent SERVICE merge.
//! - **JD**: Join decomposition by selectivity.
//! - **MR**: Result merging — UNION, OPTIONAL, MINUS.
//! - **CN**: Capability negotiation interplay (silent default, latency).
//! - **CC**: Cost-comparison checks.
//!
//! ## Pass-rate target
//!
//! Per the W3-S10 plan, ≥ 95 % of these scenarios must pass.  All 40
//! scenarios below currently pass.

use std::sync::Arc;

use oxirs_arq::algebra::{
    Algebra, BinaryOperator, Expression, Literal, Term, TriplePattern, Variable,
};
use oxirs_arq::optimizer::federated_plan::{
    FederatedSelectivity, SourceSelectivityProvider, StaticSourceProvider,
};
use oxirs_core::model::NamedNode;

use oxirs_federate::cost_model::{EndpointStats, FederationCostModel};
use oxirs_federate::optimizer::{
    merge_adjacent_services, push_filters_into_services, reorder_joins_by_selectivity,
    OptimizerPipeline,
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn var(s: &str) -> Variable {
    Variable::new(s).expect("valid variable name")
}

fn iri(s: &str) -> Term {
    Term::Iri(NamedNode::new_unchecked(s))
}

fn tp(s: Term, p: Term, o: Term) -> TriplePattern {
    TriplePattern {
        subject: s,
        predicate: p,
        object: o,
    }
}

fn bgp(s: &str, p: &str, o: &str) -> Algebra {
    Algebra::Bgp(vec![tp(
        Term::Variable(var(s)),
        iri(p),
        Term::Variable(var(o)),
    )])
}

fn service(endpoint: &str, pattern: Algebra, silent: bool) -> Algebra {
    Algebra::Service {
        endpoint: iri(endpoint),
        pattern: Box::new(pattern),
        silent,
    }
}

fn service_with_var_endpoint(endpoint_var: &str, pattern: Algebra, silent: bool) -> Algebra {
    Algebra::Service {
        endpoint: Term::Variable(var(endpoint_var)),
        pattern: Box::new(pattern),
        silent,
    }
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

fn build_provider() -> StaticSourceProvider {
    let mut p = StaticSourceProvider::new();
    p.register(
        "http://a.example",
        "http://a.example/sparql",
        FederatedSelectivity {
            estimated_cardinality: 50.0,
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
            estimated_cardinality: 500.0,
            estimated_latency_ms: 70.0,
            confidence: 0.9,
        },
    );
    p
}

fn build_cost_model() -> FederationCostModel {
    let mut m = FederationCostModel::new(5.0);
    m.register_endpoint(EndpointStats {
        endpoint_url: "http://a.example/sparql".into(),
        avg_latency_ms: 10.0,
        triples_count: 1_000,
        selectivity_estimate: 0.05,
        bandwidth_mbps: 100.0,
    });
    m.register_endpoint(EndpointStats {
        endpoint_url: "http://b.example/sparql".into(),
        avg_latency_ms: 20.0,
        triples_count: 100_000,
        selectivity_estimate: 0.05,
        bandwidth_mbps: 100.0,
    });
    m
}

// ─────────────────────────────────────────────────────────────────────────────
// SC — SERVICE clause forms
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn sc01_named_service_passes_through_pipeline() {
    // SELECT ?s WHERE { SERVICE <http://a.example/sparql> { ?s ?p ?o } }
    let svc = service("http://a.example/sparql", bgp("s", "http://p", "o"), false);
    let pipe = OptimizerPipeline::default();
    let out = pipe.run(svc.clone());
    // Pipeline preserves structurally well-formed plans.
    assert!(matches!(out, Algebra::Service { silent: false, .. }));
}

#[test]
fn sc02_silent_service_preserves_silent_flag() {
    let svc = service("http://a.example/sparql", bgp("s", "http://p", "o"), true);
    let out = push_filters_into_services(svc.clone());
    if let Algebra::Service { silent, .. } = out {
        assert!(silent, "SILENT flag must be preserved");
    } else {
        panic!("expected Service");
    }
}

#[test]
fn sc03_variable_endpoint_service_preserved() {
    // SERVICE ?endpoint { ?s ?p ?o }
    let svc = service_with_var_endpoint("endpoint", bgp("s", "http://p", "o"), false);
    let pipe = OptimizerPipeline::default();
    let out = pipe.run(svc.clone());
    if let Algebra::Service { endpoint, .. } = out {
        assert!(matches!(endpoint, Term::Variable(_)));
    } else {
        panic!("expected Service with variable endpoint");
    }
}

#[test]
fn sc04_silent_and_non_silent_to_same_endpoint_not_merged() {
    let s_loud = service("http://a.example", bgp("s", "http://p", "o"), false);
    let s_silent = service("http://a.example", bgp("a", "http://q", "b"), true);
    let join = Algebra::Join {
        left: Box::new(s_loud),
        right: Box::new(s_silent),
    };
    let merged = merge_adjacent_services(join);
    assert!(
        matches!(merged, Algebra::Join { .. }),
        "different SILENT flags must NOT merge"
    );
}

#[test]
fn sc05_variable_endpoint_blocks_merge() {
    let s_var = service_with_var_endpoint("endpoint", bgp("s", "http://p", "o"), false);
    let s_named = service("http://a.example", bgp("a", "http://q", "b"), false);
    let join = Algebra::Join {
        left: Box::new(s_var),
        right: Box::new(s_named),
    };
    let merged = merge_adjacent_services(join);
    assert!(matches!(merged, Algebra::Join { .. }));
}

// ─────────────────────────────────────────────────────────────────────────────
// FP — Filter pushdown
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fp01_simple_filter_pushdown() {
    // Filter(?s == "x", Service(A, ?s ?p ?o))  ⇒  Service(A, Filter(?s == "x", BGP))
    let svc = service("http://a.example", bgp("s", "http://p", "o"), false);
    let outer = Algebra::Filter {
        pattern: Box::new(svc),
        condition: eq_expr("s", "x"),
    };
    let rewritten = push_filters_into_services(outer);
    assert!(matches!(rewritten, Algebra::Service { .. }));
}

#[test]
fn fp02_unbound_filter_stays_outside() {
    let svc = service("http://a.example", bgp("s", "http://p", "o"), false);
    let outer = Algebra::Filter {
        pattern: Box::new(svc),
        condition: eq_expr("foreign", "x"),
    };
    let rewritten = push_filters_into_services(outer);
    assert!(matches!(rewritten, Algebra::Filter { .. }));
}

#[test]
fn fp03_partial_pushdown_with_conjunction() {
    let svc = service("http://a.example", bgp("s", "http://p", "o"), false);
    let cond = and_expr(eq_expr("s", "x"), eq_expr("foreign", "y"));
    let outer = Algebra::Filter {
        pattern: Box::new(svc),
        condition: cond,
    };
    let rewritten = push_filters_into_services(outer);
    if let Algebra::Filter { pattern, .. } = rewritten {
        assert!(matches!(*pattern, Algebra::Service { .. }));
    } else {
        panic!("expected Filter wrapping Service after partial pushdown");
    }
}

#[test]
fn fp04_pushdown_inside_join() {
    let s1 = service("http://a.example", bgp("s", "http://p", "o"), false);
    let s2 = service("http://b.example", bgp("a", "http://q", "b"), false);
    let f1 = Algebra::Filter {
        pattern: Box::new(s1),
        condition: eq_expr("s", "1"),
    };
    let f2 = Algebra::Filter {
        pattern: Box::new(s2),
        condition: eq_expr("a", "2"),
    };
    let join = Algebra::Join {
        left: Box::new(f1),
        right: Box::new(f2),
    };
    let rewritten = push_filters_into_services(join);
    if let Algebra::Join { left, right } = rewritten {
        assert!(matches!(*left, Algebra::Service { .. }));
        assert!(matches!(*right, Algebra::Service { .. }));
    } else {
        panic!("expected Join with two Services after pushdown");
    }
}

#[test]
fn fp05_pushdown_preserves_silent() {
    let svc = service("http://a.example", bgp("s", "http://p", "o"), true);
    let outer = Algebra::Filter {
        pattern: Box::new(svc),
        condition: eq_expr("s", "x"),
    };
    let rewritten = push_filters_into_services(outer);
    if let Algebra::Service { silent, .. } = rewritten {
        assert!(silent);
    } else {
        panic!("expected Service with SILENT preserved");
    }
}

#[test]
fn fp06_no_op_for_local_bgp() {
    let bg = bgp("s", "http://p", "o");
    let f = Algebra::Filter {
        pattern: Box::new(bg.clone()),
        condition: eq_expr("s", "x"),
    };
    let rewritten = push_filters_into_services(f.clone());
    assert_eq!(rewritten, f, "no-op on local BGP");
}

#[test]
fn fp07_pushdown_inside_left_join() {
    let svc = service("http://a.example", bgp("s", "http://p", "o"), false);
    let outer = Algebra::LeftJoin {
        left: Box::new(bgp("a", "http://q", "b")),
        right: Box::new(Algebra::Filter {
            pattern: Box::new(svc),
            condition: eq_expr("s", "x"),
        }),
        filter: None,
    };
    let rewritten = push_filters_into_services(outer);
    if let Algebra::LeftJoin { right, .. } = rewritten {
        assert!(matches!(*right, Algebra::Service { .. }));
    } else {
        panic!("expected LeftJoin with pushed Service on the right");
    }
}

#[test]
fn fp08_pushdown_inside_minus() {
    let svc = service("http://a.example", bgp("s", "http://p", "o"), false);
    let outer = Algebra::Minus {
        left: Box::new(bgp("a", "http://q", "b")),
        right: Box::new(Algebra::Filter {
            pattern: Box::new(svc),
            condition: eq_expr("s", "x"),
        }),
    };
    let rewritten = push_filters_into_services(outer);
    if let Algebra::Minus { right, .. } = rewritten {
        assert!(matches!(*right, Algebra::Service { .. }));
    } else {
        panic!("expected Minus with pushed Service on the right");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SM — SERVICE merge
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn sm01_two_services_same_endpoint_merge() {
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
    assert!(matches!(merged, Algebra::Service { .. }));
}

#[test]
fn sm02_three_services_collapse() {
    let mk = |s: &str, p: &str, o: &str| service("http://a.example/sparql", bgp(s, p, o), false);
    let join = Algebra::Join {
        left: Box::new(Algebra::Join {
            left: Box::new(mk("s1", "http://p1", "o1")),
            right: Box::new(mk("s2", "http://p2", "o2")),
        }),
        right: Box::new(mk("s3", "http://p3", "o3")),
    };
    let merged = merge_adjacent_services(join);
    if let Algebra::Service { pattern, .. } = merged {
        assert!(matches!(*pattern, Algebra::Join { .. }));
    } else {
        panic!("expected single Service wrapping nested Join");
    }
}

#[test]
fn sm03_different_endpoints_do_not_merge() {
    let s1 = service("http://a.example", bgp("s", "http://p", "o"), false);
    let s2 = service("http://b.example", bgp("a", "http://q", "b"), false);
    let join = Algebra::Join {
        left: Box::new(s1),
        right: Box::new(s2),
    };
    let merged = merge_adjacent_services(join);
    assert!(matches!(merged, Algebra::Join { .. }));
}

#[test]
fn sm04_merge_idempotent() {
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

#[test]
fn sm05_no_merge_for_optional() {
    let s1 = service("http://a.example", bgp("s", "http://p", "o"), false);
    let s2 = service("http://a.example", bgp("a", "http://q", "b"), false);
    let lj = Algebra::LeftJoin {
        left: Box::new(s1),
        right: Box::new(s2),
        filter: None,
    };
    let merged = merge_adjacent_services(lj);
    assert!(matches!(merged, Algebra::LeftJoin { .. }));
}

// ─────────────────────────────────────────────────────────────────────────────
// JD — Join decomposition by selectivity
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn jd01_most_selective_first_in_chain() {
    let provider = build_provider();
    let s_a = service(
        "http://a.example/sparql",
        bgp("s", "http://a.example/p", "o"),
        false,
    );
    let s_b = service(
        "http://b.example/sparql",
        bgp("s", "http://b.example/p", "o"),
        false,
    );
    let s_c = service(
        "http://c.example/sparql",
        bgp("s", "http://c.example/p", "o"),
        false,
    );
    // Worst order: b (5000) then c (500) then a (50)
    let join = Algebra::Join {
        left: Box::new(Algebra::Join {
            left: Box::new(s_b),
            right: Box::new(s_c),
        }),
        right: Box::new(s_a),
    };
    let reordered = reorder_joins_by_selectivity(join, &provider);
    // Walk to deepest left to find the most selective leg.
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
fn jd02_no_op_for_single_service() {
    let provider = build_provider();
    let svc = service(
        "http://a.example/sparql",
        bgp("s", "http://a.example/p", "o"),
        false,
    );
    let out = reorder_joins_by_selectivity(svc.clone(), &provider);
    assert_eq!(out, svc);
}

#[test]
fn jd03_left_join_not_reordered() {
    let provider = build_provider();
    let s_a = service(
        "http://a.example/sparql",
        bgp("s", "http://a.example/p", "o"),
        false,
    );
    let s_b = service(
        "http://b.example/sparql",
        bgp("s", "http://b.example/p", "o"),
        false,
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
fn jd04_pipeline_with_provider_reorders() {
    let provider: Arc<_> = Arc::new(build_provider());
    let pipe = OptimizerPipeline::default().with_selectivity(provider);
    let s_a = service(
        "http://a.example/sparql",
        bgp("s", "http://a.example/p", "o"),
        false,
    );
    let s_b = service(
        "http://b.example/sparql",
        bgp("s", "http://b.example/p", "o"),
        false,
    );
    let join = Algebra::Join {
        left: Box::new(s_b),
        right: Box::new(s_a),
    };
    let out = pipe.run(join);
    // Most selective (a) should be on the left after reorder.
    if let Algebra::Join { left, .. } = out {
        if let Algebra::Service { endpoint, .. } = *left {
            assert_eq!(
                endpoint.to_string(),
                iri("http://a.example/sparql").to_string()
            );
        } else {
            panic!("expected Service on the left");
        }
    } else {
        panic!("expected Join after reorder");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MR — Result merging (UNION, OPTIONAL, MINUS)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr01_union_preserved_through_pipeline() {
    let s1 = service("http://a.example", bgp("s", "http://p", "o"), false);
    let s2 = service("http://b.example", bgp("a", "http://q", "b"), false);
    let union = Algebra::Union {
        left: Box::new(s1),
        right: Box::new(s2),
    };
    let pipe = OptimizerPipeline::default();
    let out = pipe.run(union);
    assert!(matches!(out, Algebra::Union { .. }));
}

#[test]
fn mr02_optional_left_join_preserved() {
    let s1 = service("http://a.example", bgp("s", "http://p", "o"), false);
    let s2 = service("http://b.example", bgp("a", "http://q", "b"), false);
    let lj = Algebra::LeftJoin {
        left: Box::new(s1),
        right: Box::new(s2),
        filter: None,
    };
    let pipe = OptimizerPipeline::default();
    let out = pipe.run(lj);
    assert!(matches!(out, Algebra::LeftJoin { .. }));
}

#[test]
fn mr03_minus_preserved() {
    let s1 = service("http://a.example", bgp("s", "http://p", "o"), false);
    let s2 = service("http://b.example", bgp("a", "http://q", "b"), false);
    let minus = Algebra::Minus {
        left: Box::new(s1),
        right: Box::new(s2),
    };
    let pipe = OptimizerPipeline::default();
    let out = pipe.run(minus);
    assert!(matches!(out, Algebra::Minus { .. }));
}

#[test]
fn mr04_filter_pushdown_into_union_arms() {
    // Filter(?s == "x", Union(Service(A, ?s ?p ?o), Service(B, ?s ?p ?o)))
    let s1 = service("http://a.example", bgp("s", "http://p", "o"), false);
    let s2 = service("http://b.example", bgp("s", "http://q", "o"), false);
    let union = Algebra::Union {
        left: Box::new(s1),
        right: Box::new(s2),
    };
    let outer = Algebra::Filter {
        pattern: Box::new(union),
        condition: eq_expr("s", "x"),
    };
    // Filter outside Union: each arm independently contains a Service that
    // could host the filter, but the algebra rule does not push through Union.
    // We just verify the rewrite is structurally well-formed (Filter on top
    // of Union of Services).
    let rewritten = push_filters_into_services(outer);
    if let Algebra::Filter { pattern, .. } = rewritten {
        assert!(matches!(*pattern, Algebra::Union { .. }));
    } else {
        panic!("expected Filter wrapping Union (current rewrite scope)");
    }
}

#[test]
fn mr05_optional_with_pushable_filter_inside_right() {
    let svc = service("http://a.example", bgp("s", "http://p", "o"), false);
    let outer = Algebra::LeftJoin {
        left: Box::new(bgp("c", "http://r", "d")),
        right: Box::new(Algebra::Filter {
            pattern: Box::new(svc),
            condition: eq_expr("s", "x"),
        }),
        filter: None,
    };
    let rewritten = push_filters_into_services(outer);
    if let Algebra::LeftJoin { right, .. } = rewritten {
        assert!(matches!(*right, Algebra::Service { .. }));
    } else {
        panic!("expected pushed Service on right of LeftJoin");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CN — Capability negotiation (defaults + selectivity provider hooks)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cn01_default_pipeline_no_op_without_provider() {
    let pipe = OptimizerPipeline::default();
    // Without provider, join reorder is disabled.
    let s_a = service("http://a.example", bgp("s", "http://p", "o"), false);
    let s_b = service("http://b.example", bgp("a", "http://q", "b"), false);
    let join = Algebra::Join {
        left: Box::new(s_b.clone()),
        right: Box::new(s_a.clone()),
    };
    let out = pipe.run(join.clone());
    // Filter pushdown + service merge are no-ops here (different endpoints,
    // no filters), so the tree is structurally identical.
    assert_eq!(out, join);
}

#[test]
fn cn02_pipeline_with_disabled_passes_is_identity() {
    let pipe = OptimizerPipeline::default().with_passes(false, false, false);
    let s = service("http://a.example", bgp("s", "http://p", "o"), false);
    let out = pipe.run(s.clone());
    assert_eq!(out, s);
}

#[test]
fn cn03_provider_silent_default_consulted() {
    // Build a provider where the source is registered with silent semantics.
    let mut provider = StaticSourceProvider::new();
    provider.register_silent(
        "http://noisy.example",
        "http://noisy.example/sparql",
        FederatedSelectivity::default(),
    );
    // The provider's silent_default for that endpoint URL must be true.
    assert!(provider.silent_default("http://noisy.example/sparql"));
    // For an unknown endpoint, it falls back to false.
    assert!(!provider.silent_default("http://unknown.example"));
}

// ─────────────────────────────────────────────────────────────────────────────
// CC — Cost-comparison checks via FederationCostModel
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cc01_local_plan_cheaper_than_remote() {
    let mut m = build_cost_model();
    let local = bgp("s", "http://p", "o");
    let remote = service("http://b.example/sparql", bgp("s", "http://p", "o"), false);
    let (winner, _cost) = m.cheaper(&local, &remote);
    assert!(matches!(winner, Algebra::Bgp(_)));
}

#[test]
fn cc02_pushdown_does_not_increase_cost() {
    let mut m = build_cost_model();
    let svc = service("http://a.example/sparql", bgp("s", "http://p", "o"), false);
    let f_outside = Algebra::Filter {
        pattern: Box::new(svc.clone()),
        condition: eq_expr("s", "x"),
    };
    let pushed = push_filters_into_services(f_outside.clone());

    let c_outside = m.estimate_plan_cost(&f_outside);
    let c_pushed = m.estimate_plan_cost(&pushed);
    // After pushdown, the SERVICE wraps the filter — no extra hops.
    assert!(
        c_pushed.network_cost <= c_outside.network_cost + 1e-6,
        "pushdown should never increase network cost"
    );
}

#[test]
fn cc03_service_merge_reduces_round_trips() {
    // Counts the number of `Algebra::Service` nodes in a plan — a structural
    // proxy for "how many remote round trips are required".
    fn service_count(a: &Algebra) -> usize {
        match a {
            Algebra::Service { pattern, .. } => 1 + service_count(pattern),
            Algebra::Join { left, right }
            | Algebra::Union { left, right }
            | Algebra::Minus { left, right } => service_count(left) + service_count(right),
            Algebra::LeftJoin { left, right, .. } => service_count(left) + service_count(right),
            Algebra::Filter { pattern, .. }
            | Algebra::Project { pattern, .. }
            | Algebra::Distinct { pattern }
            | Algebra::Reduced { pattern }
            | Algebra::OrderBy { pattern, .. }
            | Algebra::Slice { pattern, .. }
            | Algebra::Group { pattern, .. }
            | Algebra::Having { pattern, .. }
            | Algebra::Extend { pattern, .. }
            | Algebra::Graph { pattern, .. } => service_count(pattern),
            _ => 0,
        }
    }

    let s1 = service(
        "http://a.example/sparql",
        bgp("s", "http://p1", "o1"),
        false,
    );
    let s2 = service(
        "http://a.example/sparql",
        bgp("s", "http://p2", "o2"),
        false,
    );
    let unmerged = Algebra::Join {
        left: Box::new(s1),
        right: Box::new(s2),
    };
    let merged = merge_adjacent_services(unmerged.clone());

    // The structural property of "merged round trips":
    assert_eq!(service_count(&unmerged), 2);
    assert_eq!(service_count(&merged), 1);
}

#[test]
fn cc04_estimate_total_cost_is_finite() {
    let mut m = build_cost_model();
    let s = service("http://a.example/sparql", bgp("s", "http://p", "o"), false);
    let cost = m.estimate_plan_cost(&s);
    assert!(cost.total_cost.is_finite());
    assert!(cost.total_cost >= 0.0);
}

#[test]
fn cc06_variable_endpoint_cost_finite() {
    // Variable-endpoint SERVICE — no concrete URL to bill, but the cost
    // walker should still produce a finite (defensive) estimate without
    // panicking.
    let mut m = build_cost_model();
    let svc = service_with_var_endpoint("endpoint", bgp("s", "http://p", "o"), false);
    let cost = m.estimate_plan_cost(&svc);
    assert!(cost.total_cost.is_finite());
    assert!(cost.total_cost >= 0.0);
}

#[test]
fn cc05_per_endpoint_local_cost_aggregated() {
    let mut m = build_cost_model();
    let plan = Algebra::Join {
        left: Box::new(service(
            "http://a.example/sparql",
            bgp("s", "http://p", "o"),
            false,
        )),
        right: Box::new(service(
            "http://b.example/sparql",
            bgp("s", "http://q", "o"),
            false,
        )),
    };
    let cost = m.estimate_plan_cost(&plan);
    assert!(cost
        .per_endpoint_local
        .contains_key("http://a.example/sparql"));
    assert!(cost
        .per_endpoint_local
        .contains_key("http://b.example/sparql"));
}

// ─────────────────────────────────────────────────────────────────────────────
// END-TO-END pipelines
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e01_full_pipeline_reduces_remote_calls() {
    let provider: Arc<_> = Arc::new(build_provider());
    let pipe = OptimizerPipeline::default().with_selectivity(provider);

    // Plan: Filter(?s == "x", Join(Service(A, p1), Service(A, p2)))
    let s1 = service("http://a.example", bgp("s", "http://p1", "o1"), false);
    let s2 = service("http://a.example", bgp("s", "http://p2", "o2"), false);
    let join = Algebra::Join {
        left: Box::new(s1),
        right: Box::new(s2),
    };
    let outer = Algebra::Filter {
        pattern: Box::new(join),
        condition: eq_expr("s", "x"),
    };
    let optimized = pipe.run(outer);

    // After: Service(A, Filter(?s == "x", Join(p1, p2)))
    if let Algebra::Service { pattern, .. } = optimized {
        // Inside the merged service, filter wraps the join (or the join wraps
        // a filter — both are spec-allowed).
        assert!(matches!(
            *pattern,
            Algebra::Filter { .. } | Algebra::Join { .. }
        ));
    } else {
        panic!("expected merged Service after full pipeline");
    }
}

#[test]
fn e2e02_full_pipeline_idempotent() {
    let provider: Arc<_> = Arc::new(build_provider());
    let pipe = OptimizerPipeline::default().with_selectivity(provider);

    let s1 = service("http://a.example", bgp("s", "http://p1", "o1"), false);
    let s2 = service("http://b.example", bgp("a", "http://q", "b"), false);
    let join = Algebra::Join {
        left: Box::new(s1),
        right: Box::new(s2),
    };
    let once = pipe.run(join);
    let twice = pipe.run(once.clone());
    assert_eq!(once, twice, "pipeline must be idempotent");
}

#[test]
fn e2e03_pipeline_preserves_semantics_for_local_only() {
    let pipe = OptimizerPipeline::default();
    let local = bgp("s", "http://p", "o");
    let out = pipe.run(local.clone());
    assert_eq!(out, local);
}

#[test]
fn e2e04_pipeline_handles_deep_nesting() {
    let pipe = OptimizerPipeline::default();
    // Service nested inside Project inside Distinct.
    let svc = service("http://a.example", bgp("s", "http://p", "o"), false);
    let project = Algebra::Project {
        pattern: Box::new(svc),
        variables: vec![var("s")],
    };
    let distinct = Algebra::Distinct {
        pattern: Box::new(project),
    };
    let out = pipe.run(distinct);
    assert!(matches!(out, Algebra::Distinct { .. }));
}
