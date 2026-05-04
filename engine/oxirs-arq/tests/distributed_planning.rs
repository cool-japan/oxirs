//! Integration tests for the W2-S4 federation-aware query planner.
//!
//! Covers:
//!
//! * IRI-namespace-driven endpoint selection.
//! * Multi-source cost-based ordering.
//! * Pass-through behaviour for local patterns.
//! * Trait-abstraction extension point for downstream embedders.

use std::sync::Arc;

use oxirs_arq::algebra::{Algebra, Term, TriplePattern, Variable};
use oxirs_arq::optimizer::federated_plan::{
    FederatedPlanner, FederatedSelectivity, SourceSelectivityProvider, StaticSourceProvider,
};
use oxirs_core::model::NamedNode;

fn iri(s: &str) -> Term {
    Term::Iri(NamedNode::new_unchecked(s))
}

fn var(name: &str) -> Term {
    Term::Variable(Variable::new(name).expect("valid variable name"))
}

fn pattern(s: Term, p: Term, o: Term) -> TriplePattern {
    TriplePattern {
        subject: s,
        predicate: p,
        object: o,
    }
}

fn three_endpoint_provider() -> StaticSourceProvider {
    let mut p = StaticSourceProvider::new();
    p.register(
        "http://dbpedia.org/",
        "https://dbpedia.org/sparql",
        FederatedSelectivity {
            estimated_cardinality: 200.0,
            estimated_latency_ms: 50.0,
            confidence: 0.9,
        },
    );
    p.register(
        "http://wikidata.org/",
        "https://wikidata.org/sparql",
        FederatedSelectivity {
            estimated_cardinality: 5_000.0,
            estimated_latency_ms: 120.0,
            confidence: 0.7,
        },
    );
    p.register(
        "http://swrc.ontoware.org/",
        "https://swrc.example/sparql",
        FederatedSelectivity {
            estimated_cardinality: 10.0,
            estimated_latency_ms: 30.0,
            confidence: 0.95,
        },
    );
    p
}

#[test]
fn known_iri_namespace_emits_service_node() {
    let planner = FederatedPlanner::new(Arc::new(three_endpoint_provider()));
    let alg = Algebra::Bgp(vec![pattern(
        var("s"),
        iri("http://dbpedia.org/property/birthDate"),
        var("o"),
    )]);

    let outcome = planner.plan(alg);
    assert!(outcome.touched_federation());
    assert_eq!(outcome.patterns_federated, 1);
    assert!(outcome
        .endpoints_used
        .contains_key("https://dbpedia.org/sparql"));
    assert!(matches!(outcome.algebra, Algebra::Service { .. }));
}

#[test]
fn unknown_iri_remains_local() {
    let planner = FederatedPlanner::new(Arc::new(three_endpoint_provider()));
    let alg = Algebra::Bgp(vec![pattern(
        iri("http://example.org/local/alice"),
        iri("http://example.org/local/knows"),
        var("friend"),
    )]);
    let outcome = planner.plan(alg.clone());
    assert!(!outcome.touched_federation());
    assert_eq!(outcome.algebra, alg);
}

#[test]
fn three_endpoint_cost_orders_cheapest_first() {
    let planner = FederatedPlanner::new(Arc::new(three_endpoint_provider()));

    let alg = Algebra::Bgp(vec![
        // Wikidata: high cost
        pattern(
            var("s"),
            iri("http://wikidata.org/data/predicate"),
            var("o1"),
        ),
        // SWRC: cheapest
        pattern(
            var("s"),
            iri("http://swrc.ontoware.org/ontology#title"),
            var("o2"),
        ),
        // DBpedia: middle
        pattern(
            var("s"),
            iri("http://dbpedia.org/property/birthDate"),
            var("o3"),
        ),
    ]);

    let outcome = planner.plan(alg);
    assert_eq!(outcome.patterns_federated, 3);

    // Three Service nodes, joined left-deep with the cheapest as the leftmost
    // operand.  Walk the tree to find the leftmost endpoint URL.
    fn first_endpoint(alg: &Algebra) -> Option<&str> {
        match alg {
            Algebra::Service {
                endpoint: Term::Iri(node),
                ..
            } => Some(node.as_str()),
            Algebra::Join { left, .. } => first_endpoint(left),
            _ => None,
        }
    }

    let leftmost = first_endpoint(&outcome.algebra).expect("at least one Service");
    assert_eq!(leftmost, "https://swrc.example/sparql");
}

#[test]
fn mixed_local_and_federated_creates_join() {
    let planner = FederatedPlanner::new(Arc::new(three_endpoint_provider()));

    let alg = Algebra::Bgp(vec![
        // Local pattern
        pattern(
            var("s"),
            iri("http://example.org/internal/predicate"),
            var("v"),
        ),
        // Federated pattern
        pattern(
            var("s"),
            iri("http://dbpedia.org/property/something"),
            var("d"),
        ),
    ]);
    let outcome = planner.plan(alg);
    assert_eq!(outcome.patterns_federated, 1);
    match outcome.algebra {
        Algebra::Join { left, right } => {
            assert!(matches!(*left, Algebra::Bgp(_)));
            assert!(matches!(*right, Algebra::Service { .. }));
        }
        other => panic!("expected Join, got {other:?}"),
    }
}

#[test]
fn nested_filter_and_optional_recurse_into_planner() {
    let planner = FederatedPlanner::new(Arc::new(three_endpoint_provider()));

    let inner_bgp = Algebra::Bgp(vec![pattern(
        var("s"),
        iri("http://dbpedia.org/property/foo"),
        var("o"),
    )]);
    let optional_alg = Algebra::LeftJoin {
        left: Box::new(Algebra::Bgp(vec![pattern(
            var("s"),
            iri("http://example.org/local/p"),
            var("o2"),
        )])),
        right: Box::new(inner_bgp),
        filter: None,
    };

    let outcome = planner.plan(optional_alg);
    assert_eq!(outcome.patterns_federated, 1);
    match outcome.algebra {
        Algebra::LeftJoin { right, .. } => assert!(matches!(*right, Algebra::Service { .. })),
        other => panic!("expected LeftJoin, got {other:?}"),
    }
}

#[test]
fn pre_existing_service_node_passes_through() {
    let planner = FederatedPlanner::new(Arc::new(three_endpoint_provider()));
    let original = Algebra::Service {
        endpoint: iri("https://other.example/sparql"),
        pattern: Box::new(Algebra::Bgp(vec![pattern(
            var("s"),
            iri("http://example.org/p"),
            var("o"),
        )])),
        silent: true,
    };
    let outcome = planner.plan(original.clone());
    assert_eq!(outcome.algebra, original);
    assert!(outcome
        .endpoints_used
        .contains_key("https://other.example/sparql"));
}

#[test]
fn longest_prefix_selection_preferred() {
    let mut provider = StaticSourceProvider::new();
    provider.register(
        "http://example.org/",
        "https://wide.example/sparql",
        FederatedSelectivity::default(),
    );
    provider.register(
        "http://example.org/specific/",
        "https://narrow.example/sparql",
        FederatedSelectivity::default(),
    );

    let planner = FederatedPlanner::new(Arc::new(provider));
    let alg = Algebra::Bgp(vec![pattern(
        var("s"),
        iri("http://example.org/specific/predicate"),
        var("o"),
    )]);
    let outcome = planner.plan(alg);
    assert!(outcome
        .endpoints_used
        .contains_key("https://narrow.example/sparql"));
    assert!(!outcome
        .endpoints_used
        .contains_key("https://wide.example/sparql"));
}

#[test]
fn custom_source_provider_implementation() {
    // Confirm the trait abstraction works with a custom impl, not just
    // the bundled StaticSourceProvider.
    struct AlwaysOne;
    impl SourceSelectivityProvider for AlwaysOne {
        fn endpoint_for_iri(&self, _iri: &NamedNode) -> Option<String> {
            Some("https://catch-all.example/sparql".to_string())
        }
    }

    let planner = FederatedPlanner::new(Arc::new(AlwaysOne));
    let alg = Algebra::Bgp(vec![pattern(
        var("s"),
        iri("http://anything.example/p"),
        var("o"),
    )]);
    let outcome = planner.plan(alg);
    assert!(outcome.touched_federation());
    assert_eq!(outcome.patterns_federated, 1);
    assert!(outcome
        .endpoints_used
        .contains_key("https://catch-all.example/sparql"));
}
