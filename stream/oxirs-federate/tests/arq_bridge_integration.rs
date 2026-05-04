//! Integration tests for the W2-S4 deepening:
//! `oxirs-federate::arq_bridge::ArqSourceSelectivityProvider` wired into
//! `oxirs-arq::optimizer::Optimizer::with_federated_planner(...)`.
//!
//! These tests validate the **end-to-end happy path**: an embedder builds a
//! federated source registry via `oxirs-federate`, hands the bridge to the
//! ARQ optimizer, and observes correctly-rewritten algebra with
//! `Algebra::Service` nodes targeting the right endpoints.
//!
//! No real network I/O — VoID metadata is faked in-memory.

use std::sync::Arc;

use oxirs_arq::algebra::{Algebra, Term, TriplePattern, Variable};
use oxirs_arq::optimizer::federated_plan::SourceSelectivityProvider;
use oxirs_arq::optimizer::{Optimizer, OptimizerConfig};
use oxirs_core::model::NamedNode;

use oxirs_federate::arq_bridge::ArqSourceSelectivityProvider;
use oxirs_federate::source_selector::{
    FederatedSource, SourceCapabilities, SourceMetrics, SourceSelector, SparqlVersion,
    VoidDescription,
};

// ─── Test helpers ────────────────────────────────────────────────────────────

fn iri_term(s: &str) -> Term {
    Term::Iri(NamedNode::new_unchecked(s))
}

fn var_term(name: &str) -> Term {
    Term::Variable(Variable::new(name).expect("valid variable name"))
}

fn pattern(s: Term, p: Term, o: Term) -> TriplePattern {
    TriplePattern {
        subject: s,
        predicate: p,
        object: o,
    }
}

fn dbpedia_source() -> FederatedSource {
    let mut void = VoidDescription::new("http://dbpedia.org/dataset", 1_000_000_000);
    void.property_partitions
        .insert("http://dbpedia.org/property/birthDate".to_string(), 100_000);
    void.property_partitions
        .insert("http://dbpedia.org/property/birthPlace".to_string(), 80_000);
    void.uri_spaces.push("http://dbpedia.org/".to_string());

    FederatedSource {
        id: "dbpedia".to_string(),
        endpoint_url: "https://dbpedia.org/sparql".to_string(),
        label: Some("DBpedia".to_string()),
        capabilities: SourceCapabilities {
            sparql_version: SparqlVersion::V1_1,
            supports_named_graphs: true,
            supports_update: false,
            supports_federation: true,
            supports_text_search: false,
            supports_geosparql: false,
            named_graphs: vec![],
        },
        void_description: Some(void),
        metrics: SourceMetrics {
            avg_latency_ms: 80.0,
            success_rate: 0.99,
            freshness_ms: 60_000,
            queries_processed: 100,
            is_reachable: true,
        },
        priority: 0,
        enabled: true,
    }
}

fn wikidata_source() -> FederatedSource {
    let mut void = VoidDescription::new("http://www.wikidata.org/dataset", 10_000_000_000);
    void.property_partitions
        .insert("http://www.wikidata.org/prop/P31".to_string(), 50_000_000);
    void.property_partitions
        .insert("http://www.wikidata.org/prop/P569".to_string(), 5_000_000);
    void.uri_spaces.push("http://www.wikidata.org/".to_string());

    FederatedSource {
        id: "wikidata".to_string(),
        endpoint_url: "https://query.wikidata.org/sparql".to_string(),
        label: Some("Wikidata".to_string()),
        capabilities: SourceCapabilities::default(),
        void_description: Some(void),
        metrics: SourceMetrics {
            avg_latency_ms: 200.0,
            success_rate: 0.95,
            freshness_ms: 30_000,
            queries_processed: 50,
            is_reachable: true,
        },
        priority: 0,
        enabled: true,
    }
}

fn yago_source() -> FederatedSource {
    let mut void = VoidDescription::new("http://yago-knowledge.org/dataset", 200_000_000);
    void.property_partitions.insert(
        "http://yago-knowledge.org/resource/wasBornIn".to_string(),
        20_000,
    );
    void.uri_spaces
        .push("http://yago-knowledge.org/".to_string());

    FederatedSource {
        id: "yago".to_string(),
        endpoint_url: "https://yago-knowledge.org/sparql".to_string(),
        label: Some("YAGO".to_string()),
        capabilities: SourceCapabilities::default(),
        void_description: Some(void),
        metrics: SourceMetrics {
            avg_latency_ms: 120.0,
            success_rate: 0.97,
            freshness_ms: 600_000,
            queries_processed: 25,
            is_reachable: true,
        },
        priority: 0,
        enabled: true,
    }
}

fn build_federation() -> Arc<SourceSelector> {
    let mut sel = SourceSelector::new();
    sel.register_source(dbpedia_source())
        .expect("register dbpedia");
    sel.register_source(wikidata_source())
        .expect("register wikidata");
    sel.register_source(yago_source()).expect("register yago");
    Arc::new(sel)
}

fn build_provider() -> Arc<dyn SourceSelectivityProvider> {
    Arc::new(ArqSourceSelectivityProvider::new(build_federation()))
}

fn find_service_endpoint(algebra: &Algebra) -> Option<String> {
    match algebra {
        Algebra::Service {
            endpoint: Term::Iri(node),
            ..
        } => Some(node.as_str().to_string()),
        Algebra::Service { .. } => None,
        Algebra::Join { left, right }
        | Algebra::Union { left, right }
        | Algebra::Minus { left, right } => {
            find_service_endpoint(left).or_else(|| find_service_endpoint(right))
        }
        Algebra::LeftJoin { left, right, .. } => {
            find_service_endpoint(left).or_else(|| find_service_endpoint(right))
        }
        Algebra::Filter { pattern, .. }
        | Algebra::Distinct { pattern }
        | Algebra::Reduced { pattern }
        | Algebra::Slice { pattern, .. }
        | Algebra::OrderBy { pattern, .. }
        | Algebra::Project { pattern, .. }
        | Algebra::Extend { pattern, .. }
        | Algebra::Group { pattern, .. }
        | Algebra::Having { pattern, .. }
        | Algebra::Graph { pattern, .. } => find_service_endpoint(pattern),
        _ => None,
    }
}

fn collect_service_endpoints(algebra: &Algebra, out: &mut Vec<String>) {
    match algebra {
        Algebra::Service {
            endpoint, pattern, ..
        } => {
            if let Term::Iri(node) = endpoint {
                out.push(node.as_str().to_string());
            }
            collect_service_endpoints(pattern, out);
        }
        Algebra::Join { left, right }
        | Algebra::Union { left, right }
        | Algebra::Minus { left, right } => {
            collect_service_endpoints(left, out);
            collect_service_endpoints(right, out);
        }
        Algebra::LeftJoin { left, right, .. } => {
            collect_service_endpoints(left, out);
            collect_service_endpoints(right, out);
        }
        Algebra::Filter { pattern, .. }
        | Algebra::Distinct { pattern }
        | Algebra::Reduced { pattern }
        | Algebra::Slice { pattern, .. }
        | Algebra::OrderBy { pattern, .. }
        | Algebra::Project { pattern, .. }
        | Algebra::Extend { pattern, .. }
        | Algebra::Group { pattern, .. }
        | Algebra::Having { pattern, .. }
        | Algebra::Graph { pattern, .. } => collect_service_endpoints(pattern, out),
        _ => {}
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn dbpedia_query_routes_to_dbpedia_endpoint_via_bridge() {
    let provider = build_provider();
    let mut optimizer = Optimizer::new(OptimizerConfig::default()).with_federated_planner(provider);

    // SELECT ?o WHERE { ?s dbo:birthDate ?o }
    let query = Algebra::Bgp(vec![pattern(
        var_term("s"),
        iri_term("http://dbpedia.org/property/birthDate"),
        var_term("o"),
    )]);

    let optimized = optimizer.optimize(query).expect("optimization succeeds");

    let endpoint =
        find_service_endpoint(&optimized).expect("a Service node must be present in the plan");
    assert_eq!(endpoint, "https://dbpedia.org/sparql");

    let outcome = optimizer
        .last_federated_outcome()
        .expect("optimization should record an outcome");
    assert!(outcome.touched_federation());
    assert_eq!(outcome.patterns_federated, 1);
    assert!(outcome
        .endpoints_used
        .contains_key("https://dbpedia.org/sparql"));
}

#[test]
fn wikidata_query_routes_to_wikidata_endpoint_via_bridge() {
    let provider = build_provider();
    let mut optimizer = Optimizer::new(OptimizerConfig::default()).with_federated_planner(provider);

    let query = Algebra::Bgp(vec![pattern(
        var_term("entity"),
        iri_term("http://www.wikidata.org/prop/P31"),
        iri_term("http://www.wikidata.org/entity/Q5"),
    )]);

    let optimized = optimizer.optimize(query).expect("optimization succeeds");
    let endpoint = find_service_endpoint(&optimized).expect("expected a Service node");
    assert_eq!(endpoint, "https://query.wikidata.org/sparql");
}

#[test]
fn unknown_iri_stays_local_via_bridge() {
    let provider = build_provider();
    let mut optimizer = Optimizer::new(OptimizerConfig::default()).with_federated_planner(provider);

    let query = Algebra::Bgp(vec![pattern(
        iri_term("http://example.org/local/alice"),
        iri_term("http://example.org/local/knows"),
        var_term("friend"),
    )]);

    let optimized = optimizer.optimize(query).expect("optimization succeeds");
    assert!(
        find_service_endpoint(&optimized).is_none(),
        "local-only query must not be federated, got: {optimized:?}"
    );

    let outcome = optimizer
        .last_federated_outcome()
        .expect("outcome must be recorded");
    assert!(!outcome.touched_federation());
}

#[test]
fn three_endpoint_query_emits_three_distinct_services() {
    let provider = build_provider();
    let mut optimizer = Optimizer::new(OptimizerConfig::default()).with_federated_planner(provider);

    // Query touches DBpedia, Wikidata, and YAGO.
    let query = Algebra::Bgp(vec![
        pattern(
            var_term("s"),
            iri_term("http://dbpedia.org/property/birthDate"),
            var_term("date"),
        ),
        pattern(
            var_term("s"),
            iri_term("http://www.wikidata.org/prop/P31"),
            iri_term("http://www.wikidata.org/entity/Q5"),
        ),
        pattern(
            var_term("s"),
            iri_term("http://yago-knowledge.org/resource/wasBornIn"),
            var_term("place"),
        ),
    ]);

    let optimized = optimizer.optimize(query).expect("optimization succeeds");

    let mut endpoints = Vec::new();
    collect_service_endpoints(&optimized, &mut endpoints);
    endpoints.sort();
    endpoints.dedup();

    assert_eq!(
        endpoints.len(),
        3,
        "expected 3 endpoints, got {endpoints:?}"
    );
    assert!(endpoints.contains(&"https://dbpedia.org/sparql".to_string()));
    assert!(endpoints.contains(&"https://query.wikidata.org/sparql".to_string()));
    assert!(endpoints.contains(&"https://yago-knowledge.org/sparql".to_string()));

    let outcome = optimizer
        .last_federated_outcome()
        .expect("outcome must be recorded");
    assert_eq!(outcome.patterns_federated, 3);
    assert_eq!(outcome.endpoints_used.len(), 3);
}

#[test]
fn mixed_local_and_federated_query_via_bridge() {
    let provider = build_provider();
    let mut optimizer = Optimizer::new(OptimizerConfig::default()).with_federated_planner(provider);

    // One local, one DBpedia.
    let query = Algebra::Bgp(vec![
        pattern(
            var_term("s"),
            iri_term("http://example.org/local/labelOf"),
            var_term("label"),
        ),
        pattern(
            var_term("s"),
            iri_term("http://dbpedia.org/property/birthDate"),
            var_term("date"),
        ),
    ]);

    let optimized = optimizer.optimize(query).expect("optimization succeeds");

    // Top-level should be a Join of local-BGP and the dbpedia Service.
    let endpoint =
        find_service_endpoint(&optimized).expect("federated portion should produce a Service node");
    assert_eq!(endpoint, "https://dbpedia.org/sparql");

    // Confirm the local BGP is still in the plan: the query should contain a
    // Bgp node with the local pattern.
    fn has_local_bgp(algebra: &Algebra) -> bool {
        match algebra {
            Algebra::Bgp(patterns) => patterns.iter().any(|p| {
                matches!(&p.predicate, Term::Iri(node) if node.as_str() == "http://example.org/local/labelOf")
            }),
            Algebra::Join { left, right }
            | Algebra::Union { left, right }
            | Algebra::Minus { left, right } => has_local_bgp(left) || has_local_bgp(right),
            Algebra::LeftJoin { left, right, .. } => has_local_bgp(left) || has_local_bgp(right),
            Algebra::Filter { pattern, .. }
            | Algebra::Project { pattern, .. }
            | Algebra::Distinct { pattern }
            | Algebra::Reduced { pattern }
            | Algebra::Slice { pattern, .. }
            | Algebra::OrderBy { pattern, .. }
            | Algebra::Extend { pattern, .. }
            | Algebra::Group { pattern, .. }
            | Algebra::Having { pattern, .. }
            | Algebra::Graph { pattern, .. } => has_local_bgp(pattern),
            _ => false,
        }
    }
    assert!(
        has_local_bgp(&optimized),
        "local triple pattern lost from plan: {optimized:?}"
    );
}

#[test]
fn excluded_endpoint_not_chosen_by_bridge() {
    // Build a federation, exclude DBpedia, and confirm the plan stays local.
    let mut sel = SourceSelector::new();
    sel.register_source(dbpedia_source())
        .expect("register dbpedia");
    sel.exclude_source("dbpedia");

    let provider: Arc<dyn SourceSelectivityProvider> =
        Arc::new(ArqSourceSelectivityProvider::new(Arc::new(sel)));
    let mut optimizer = Optimizer::new(OptimizerConfig::default()).with_federated_planner(provider);

    let query = Algebra::Bgp(vec![pattern(
        var_term("s"),
        iri_term("http://dbpedia.org/property/birthDate"),
        var_term("o"),
    )]);

    let optimized = optimizer.optimize(query).expect("optimization succeeds");
    assert!(
        find_service_endpoint(&optimized).is_none(),
        "excluded endpoint should not be chosen, got: {optimized:?}"
    );
}

#[test]
fn cheap_endpoint_wins_when_two_endpoints_tie_on_iri() {
    // Both endpoints' VoID covers the same prefix; the bridge should pick
    // the higher-scored one (lower latency, higher reliability).
    let mut fast_void = VoidDescription::new("http://shared.example/dataset/fast", 100_000);
    fast_void
        .property_partitions
        .insert("http://shared.example/property".to_string(), 1_000);
    fast_void
        .uri_spaces
        .push("http://shared.example/".to_string());

    let mut slow_void = VoidDescription::new("http://shared.example/dataset/slow", 100_000);
    slow_void
        .property_partitions
        .insert("http://shared.example/property".to_string(), 1_000);
    slow_void
        .uri_spaces
        .push("http://shared.example/".to_string());

    let fast = FederatedSource {
        id: "fast".to_string(),
        endpoint_url: "https://fast.example/sparql".to_string(),
        label: None,
        capabilities: SourceCapabilities::default(),
        void_description: Some(fast_void),
        metrics: SourceMetrics {
            avg_latency_ms: 5.0,
            success_rate: 0.999,
            freshness_ms: 0,
            queries_processed: 0,
            is_reachable: true,
        },
        priority: 0,
        enabled: true,
    };
    let slow = FederatedSource {
        id: "slow".to_string(),
        endpoint_url: "https://slow.example/sparql".to_string(),
        label: None,
        capabilities: SourceCapabilities::default(),
        void_description: Some(slow_void),
        metrics: SourceMetrics {
            avg_latency_ms: 5_000.0,
            success_rate: 0.5,
            freshness_ms: 3_600_000,
            queries_processed: 0,
            is_reachable: true,
        },
        priority: 0,
        enabled: true,
    };

    let mut sel = SourceSelector::new();
    sel.register_source(fast).expect("fast");
    sel.register_source(slow).expect("slow");
    let provider: Arc<dyn SourceSelectivityProvider> =
        Arc::new(ArqSourceSelectivityProvider::new(Arc::new(sel)));

    let mut optimizer = Optimizer::new(OptimizerConfig::default()).with_federated_planner(provider);
    let query = Algebra::Bgp(vec![pattern(
        var_term("s"),
        iri_term("http://shared.example/property"),
        var_term("o"),
    )]);

    let optimized = optimizer.optimize(query).expect("optimization succeeds");
    let endpoint = find_service_endpoint(&optimized).expect("a Service must be present");
    assert_eq!(endpoint, "https://fast.example/sparql");
}

#[test]
fn baseline_optimizer_is_unchanged_without_provider() {
    // Critical regression check: omitting with_federated_planner() must mean
    // the optimizer is byte-for-byte identical to the pre-W2-S4 baseline.
    let mut optimizer = Optimizer::new(OptimizerConfig::default());
    assert!(!optimizer.has_federated_planner());

    let query = Algebra::Bgp(vec![pattern(
        var_term("s"),
        iri_term("http://dbpedia.org/property/birthDate"),
        var_term("o"),
    )]);
    let optimized = optimizer.optimize(query).expect("baseline optimize");

    assert!(
        find_service_endpoint(&optimized).is_none(),
        "no federation without provider, got: {optimized:?}"
    );
    assert!(optimizer.last_federated_outcome().is_none());
}
