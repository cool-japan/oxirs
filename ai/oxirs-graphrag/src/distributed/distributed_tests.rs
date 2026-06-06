//! Tests for the distributed module.
//!
//! This file is only compiled during test builds (`#[cfg(test)] mod distributed_tests`
//! in `distributed/mod.rs`).  Here `super` resolves to the `distributed` module,
//! so `super::coordinator` and `super::worker` are the two sibling submodules.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::{GraphRAGResult, ScoreSource, ScoredEntity, Triple};

use super::coordinator::{
    ContextOrderingStrategy, DistributedEntityResolver, EndpointAuth, EndpointConfig,
    EndpointExecutor, FederatedContextBuilder, FederatedContextConfig, FederatedGraphRAGConfig,
    KnowledgeGraph,
};
use super::worker::{
    build_same_as_sparql, build_seed_expansion_sparql, parse_n_triples, AggregateMetrics,
    DistributedGraphRAGMetrics, EndpointMetrics, FederatedSubgraphExpander,
};

// ── Mock executor ────────────────────────────────────────────────────────────

struct MockExecutor {
    triples_by_endpoint: HashMap<String, Vec<Triple>>,
    same_as_by_endpoint: HashMap<String, Vec<(String, String)>>,
}

impl MockExecutor {
    fn new() -> Self {
        Self {
            triples_by_endpoint: HashMap::new(),
            same_as_by_endpoint: HashMap::new(),
        }
    }

    fn with_triples(mut self, endpoint: &str, triples: Vec<Triple>) -> Self {
        self.triples_by_endpoint
            .insert(endpoint.to_string(), triples);
        self
    }

    fn with_same_as(mut self, endpoint: &str, pairs: Vec<(String, String)>) -> Self {
        self.same_as_by_endpoint.insert(endpoint.to_string(), pairs);
        self
    }
}

#[async_trait]
impl EndpointExecutor for MockExecutor {
    async fn construct(
        &self,
        endpoint: &EndpointConfig,
        _sparql: &str,
        _timeout: Duration,
    ) -> GraphRAGResult<Vec<Triple>> {
        Ok(self
            .triples_by_endpoint
            .get(&endpoint.name)
            .cloned()
            .unwrap_or_default())
    }

    async fn select(
        &self,
        endpoint: &EndpointConfig,
        _sparql: &str,
        _timeout: Duration,
    ) -> GraphRAGResult<Vec<HashMap<String, String>>> {
        let pairs = self
            .same_as_by_endpoint
            .get(&endpoint.name)
            .cloned()
            .unwrap_or_default();
        Ok(pairs
            .into_iter()
            .map(|(a, b)| {
                let mut m = HashMap::new();
                m.insert("a".to_string(), a);
                m.insert("b".to_string(), b);
                m
            })
            .collect())
    }
}

// ── Helper constructors ──────────────────────────────────────────────────────

fn make_endpoint(name: &str, priority: f64) -> EndpointConfig {
    EndpointConfig {
        name: name.to_string(),
        url: format!("http://example.org/{}/sparql", name),
        auth: EndpointAuth::None,
        timeout_ms: Some(5_000),
        priority,
        enabled: true,
        graph_uri: None,
        max_triples: 1_000,
    }
}

fn make_seed(uri: &str, score: f64) -> ScoredEntity {
    ScoredEntity {
        uri: uri.to_string(),
        score,
        source: ScoreSource::Vector,
        metadata: HashMap::new(),
    }
}

fn make_triple(s: &str, p: &str, o: &str) -> Triple {
    Triple::new(s, p, o)
}

// ── test_federated_config_validation ─────────────────────────────────────────

#[test]
fn test_federated_config_validation_valid() {
    let config = FederatedGraphRAGConfig {
        endpoints: vec![make_endpoint("ep1", 1.0)],
        global_timeout_ms: 10_000,
        max_concurrency: 4,
        same_as_max_depth: 3,
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_federated_config_validation_zero_timeout() {
    let config = FederatedGraphRAGConfig {
        global_timeout_ms: 0,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_federated_config_validation_zero_concurrency() {
    let config = FederatedGraphRAGConfig {
        max_concurrency: 0,
        global_timeout_ms: 1_000,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_federated_config_validation_empty_url() {
    let mut ep = make_endpoint("ep1", 1.0);
    ep.url = String::new();
    let config = FederatedGraphRAGConfig {
        endpoints: vec![ep],
        global_timeout_ms: 5_000,
        max_concurrency: 2,
        same_as_max_depth: 3,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_federated_config_active_endpoints_filters_disabled() {
    let mut ep_disabled = make_endpoint("ep_off", 1.0);
    ep_disabled.enabled = false;
    let config = FederatedGraphRAGConfig {
        endpoints: vec![make_endpoint("ep_on", 1.0), ep_disabled],
        global_timeout_ms: 5_000,
        max_concurrency: 2,
        same_as_max_depth: 3,
        ..Default::default()
    };
    let active = config.active_endpoints();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].name, "ep_on");
}

// ── test_federated_subgraph_expander ─────────────────────────────────────────

#[tokio::test]
async fn test_federated_expansion_merges_two_endpoints() {
    let triples_a = vec![
        make_triple("http://a/s1", "http://p", "http://a/o1"),
        make_triple("http://a/s2", "http://p", "http://a/o2"),
    ];
    let triples_b = vec![
        make_triple("http://b/s1", "http://p", "http://b/o1"),
        make_triple("http://a/s1", "http://p", "http://a/o1"), // duplicate
    ];
    let executor = MockExecutor::new()
        .with_triples("ep_a", triples_a)
        .with_triples("ep_b", triples_b);

    let config = FederatedGraphRAGConfig {
        endpoints: vec![make_endpoint("ep_a", 2.0), make_endpoint("ep_b", 1.0)],
        global_timeout_ms: 5_000,
        max_concurrency: 4,
        same_as_max_depth: 3,
        partial_results_ok: true,
        ..Default::default()
    };

    let expander = FederatedSubgraphExpander::new(config, Arc::new(executor));
    let seeds = vec![make_seed("http://a/s1", 0.9)];
    let kg = expander
        .expand_federated(&seeds, None)
        .await
        .expect("should succeed");

    // 3 unique triples: 2 from ep_a + 1 new from ep_b (duplicate filtered)
    assert_eq!(kg.triple_count(), 3);
    assert!(!kg.is_empty());
}

#[tokio::test]
async fn test_federated_expansion_empty_seeds() {
    let executor = MockExecutor::new();
    let config = FederatedGraphRAGConfig {
        endpoints: vec![make_endpoint("ep_a", 1.0)],
        global_timeout_ms: 5_000,
        max_concurrency: 2,
        same_as_max_depth: 3,
        ..Default::default()
    };
    let expander = FederatedSubgraphExpander::new(config, Arc::new(executor));
    let kg = expander
        .expand_federated(&[], None)
        .await
        .expect("should succeed");
    assert!(kg.is_empty());
}

#[tokio::test]
async fn test_federated_expansion_no_active_endpoints() {
    let mut ep = make_endpoint("ep1", 1.0);
    ep.enabled = false;
    let executor = MockExecutor::new();
    let config = FederatedGraphRAGConfig {
        endpoints: vec![ep],
        global_timeout_ms: 5_000,
        max_concurrency: 2,
        same_as_max_depth: 3,
        ..Default::default()
    };
    let expander = FederatedSubgraphExpander::new(config, Arc::new(executor));
    let seeds = vec![make_seed("http://s", 0.9)];
    let result = expander.expand_federated(&seeds, None).await;
    assert!(result.is_err());
}

// ── test_distributed_entity_resolver_sameAs ──────────────────────────────────

#[tokio::test]
async fn test_distributed_entity_resolver_same_as_direct() {
    let same_as_pairs = vec![("http://a/e1".to_string(), "http://b/e1".to_string())];
    let executor = MockExecutor::new().with_same_as("ep_a", same_as_pairs);

    let config = FederatedGraphRAGConfig {
        endpoints: vec![make_endpoint("ep_a", 1.0)],
        global_timeout_ms: 5_000,
        max_concurrency: 2,
        same_as_max_depth: 3,
        ..Default::default()
    };

    let resolver = DistributedEntityResolver::new(config, Arc::new(executor));
    let uris = vec!["http://a/e1".to_string()];
    let closure = resolver
        .same_as_closure(&uris)
        .await
        .expect("should succeed");

    let canon_a = closure.get("http://a/e1").expect("should succeed");
    let canon_b = closure.get("http://b/e1").expect("should succeed");
    assert_eq!(
        canon_a, canon_b,
        "Same-as entities should share canonical URI"
    );
}

#[tokio::test]
async fn test_distributed_entity_resolver_no_links() {
    let executor = MockExecutor::new();
    let config = FederatedGraphRAGConfig {
        endpoints: vec![make_endpoint("ep_a", 1.0)],
        global_timeout_ms: 5_000,
        max_concurrency: 2,
        same_as_max_depth: 3,
        ..Default::default()
    };

    let resolver = DistributedEntityResolver::new(config, Arc::new(executor));
    let uris = vec!["http://example.org/e1".to_string()];
    let closure = resolver
        .same_as_closure(&uris)
        .await
        .expect("should succeed");

    let canon = closure
        .get("http://example.org/e1")
        .expect("should succeed");
    assert_eq!(canon, "http://example.org/e1");
}

#[tokio::test]
async fn test_distributed_entity_resolver_transitive_chain() {
    let same_as_pairs_ep1 = vec![("http://a/e1".to_string(), "http://b/e1".to_string())];
    let same_as_pairs_ep2 = vec![("http://b/e1".to_string(), "http://c/e1".to_string())];
    let executor = MockExecutor::new()
        .with_same_as("ep1", same_as_pairs_ep1)
        .with_same_as("ep2", same_as_pairs_ep2);

    let config = FederatedGraphRAGConfig {
        endpoints: vec![make_endpoint("ep1", 1.0), make_endpoint("ep2", 1.0)],
        global_timeout_ms: 5_000,
        max_concurrency: 2,
        same_as_max_depth: 5,
        ..Default::default()
    };

    let resolver = DistributedEntityResolver::new(config, Arc::new(executor));
    let uris = vec!["http://a/e1".to_string()];
    let closure = resolver
        .same_as_closure(&uris)
        .await
        .expect("should succeed");

    if let Some(canon_a) = closure.get("http://a/e1") {
        if let Some(canon_b) = closure.get("http://b/e1") {
            assert_eq!(canon_a, canon_b);
        }
    }
}

#[test]
fn test_apply_to_graph_rewrites_uris() {
    let executor = MockExecutor::new();
    let config = FederatedGraphRAGConfig::default();
    let resolver = DistributedEntityResolver::new(config, Arc::new(executor));

    let mut kg = KnowledgeGraph::new();
    kg.triples = vec![
        make_triple("http://a/e1", "http://p", "http://b/e1"),
        make_triple("http://a/e1", "http://p", "http://a/e1"),
    ];
    kg.provenance = vec!["ep_a".to_string(), "ep_a".to_string()];

    let mut canonical = HashMap::new();
    canonical.insert("http://a/e1".to_string(), "http://canonical/e1".to_string());
    canonical.insert("http://b/e1".to_string(), "http://canonical/e1".to_string());

    resolver.apply_to_graph(&mut kg, &canonical);

    assert_eq!(kg.triple_count(), 1);
    assert_eq!(kg.triples[0].subject, "http://canonical/e1");
    assert_eq!(kg.triples[0].object, "http://canonical/e1");
}

// ── test_federated_context_builder ────────────────────────────────────────────

#[tokio::test]
async fn test_federated_context_builder_basic() {
    let graphrag_config = FederatedGraphRAGConfig {
        endpoints: vec![make_endpoint("ep_a", 2.0), make_endpoint("ep_b", 1.0)],
        global_timeout_ms: 5_000,
        max_concurrency: 2,
        same_as_max_depth: 3,
        ..Default::default()
    };

    let ctx_config = FederatedContextConfig {
        max_context_triples: 100,
        max_context_chars: 10_000,
        ordering: ContextOrderingStrategy::ByEndpointPriority,
        include_provenance: true,
        include_equivalences: false,
        ..Default::default()
    };

    let builder = FederatedContextBuilder::new(ctx_config, &graphrag_config);

    let mut kg = KnowledgeGraph::new();
    kg.triples = vec![
        make_triple("http://s1", "http://p", "http://o1"),
        make_triple("http://s2", "http://p", "http://o2"),
    ];
    kg.provenance = vec!["ep_a".to_string(), "ep_b".to_string()];

    let context = builder
        .build_context(&kg, "test query")
        .await
        .expect("should succeed");

    assert!(context.contains("test query"));
    assert!(context.contains("http://s1"));
    assert!(context.contains("http://s2"));
    assert!(context.contains("[ep_a]") || context.contains("[ep_b]"));
}

#[tokio::test]
async fn test_federated_context_builder_empty_kg() {
    let graphrag_config = FederatedGraphRAGConfig::default();
    let ctx_config = FederatedContextConfig::default();
    let builder = FederatedContextBuilder::new(ctx_config, &graphrag_config);
    let kg = KnowledgeGraph::new();
    let context = builder
        .build_context(&kg, "test")
        .await
        .expect("should succeed");
    assert!(context.is_empty());
}

#[tokio::test]
async fn test_federated_context_builder_respects_max_triples() {
    let graphrag_config = FederatedGraphRAGConfig {
        endpoints: vec![make_endpoint("ep_a", 1.0)],
        global_timeout_ms: 5_000,
        max_concurrency: 2,
        same_as_max_depth: 3,
        ..Default::default()
    };

    let ctx_config = FederatedContextConfig {
        max_context_triples: 2,
        max_context_chars: 100_000,
        ordering: ContextOrderingStrategy::Insertion,
        include_provenance: false,
        include_equivalences: false,
        ..Default::default()
    };

    let builder = FederatedContextBuilder::new(ctx_config, &graphrag_config);

    let mut kg = KnowledgeGraph::new();
    kg.triples = (0..10)
        .map(|i| {
            make_triple(
                &format!("http://s{}", i),
                "http://p",
                &format!("http://o{}", i),
            )
        })
        .collect();
    kg.provenance = (0..10).map(|_| "ep_a".to_string()).collect();

    let context = builder
        .build_context(&kg, "test")
        .await
        .expect("should succeed");

    let triple_lines = context.lines().filter(|l| l.starts_with("- ")).count();
    assert!(
        triple_lines <= 2,
        "Expected at most 2 triples, got {}",
        triple_lines
    );
}

// ── test_distributed_metrics_tracking ────────────────────────────────────────

#[tokio::test]
async fn test_distributed_metrics_tracking_success() {
    let endpoints = vec![make_endpoint("ep_a", 1.0), make_endpoint("ep_b", 1.0)];
    let metrics = DistributedGraphRAGMetrics::new(&endpoints);

    metrics.record_success("ep_a", 150, 42).await;
    metrics.record_success("ep_a", 100, 30).await;

    let snap = metrics
        .endpoint_snapshot("ep_a")
        .await
        .expect("should succeed");
    assert_eq!(snap.total_queries, 2);
    assert_eq!(snap.successful_queries, 2);
    assert_eq!(snap.failed_queries, 0);
    assert_eq!(snap.total_triples, 72);
    assert!(snap.avg_latency_ms > 0.0);
}

#[tokio::test]
async fn test_distributed_metrics_tracking_failure() {
    let endpoints = vec![make_endpoint("ep_a", 1.0)];
    let metrics = DistributedGraphRAGMetrics::new(&endpoints);

    metrics.record_failure("ep_a").await;
    metrics.record_failure("ep_a").await;

    let snap = metrics
        .endpoint_snapshot("ep_a")
        .await
        .expect("should succeed");
    assert_eq!(snap.total_queries, 2);
    assert_eq!(snap.failed_queries, 2);
    assert_eq!(snap.successful_queries, 0);
    assert_eq!(snap.hit_rate, 0.0);
}

#[tokio::test]
async fn test_distributed_metrics_aggregate() {
    let endpoints = vec![make_endpoint("ep_a", 1.0)];
    let metrics = DistributedGraphRAGMetrics::new(&endpoints);

    metrics.record_federation_query(200, 100, false).await;
    metrics.record_federation_query(300, 50, true).await;
    metrics.record_entity_resolution().await;

    let agg = metrics.aggregate_snapshot().await;
    assert_eq!(agg.total_federation_queries, 2);
    assert_eq!(agg.total_triples_gathered, 150);
    assert_eq!(agg.entity_resolution_ops, 1);
    assert_eq!(agg.partial_failure_count, 1);
    assert!(agg.avg_federation_latency_ms > 0.0);
}

#[tokio::test]
async fn test_distributed_metrics_fastest_endpoint() {
    let endpoints = vec![make_endpoint("ep_a", 1.0), make_endpoint("ep_b", 1.0)];
    let metrics = DistributedGraphRAGMetrics::new(&endpoints);

    metrics.record_success("ep_a", 500, 10).await;
    metrics.record_success("ep_b", 50, 10).await;

    let fastest = metrics.fastest_endpoint().await.expect("should succeed");
    assert_eq!(fastest, "ep_b");
}

#[tokio::test]
async fn test_distributed_metrics_hit_rate() {
    let endpoints = vec![make_endpoint("ep_a", 1.0)];
    let metrics = DistributedGraphRAGMetrics::new(&endpoints);

    metrics.record_success("ep_a", 100, 5).await;
    metrics.record_failure("ep_a").await;

    let snap = metrics
        .endpoint_snapshot("ep_a")
        .await
        .expect("should succeed");
    assert_eq!(snap.total_queries, 2);
    assert!(snap.hit_rate >= 0.0 && snap.hit_rate <= 1.0);
}

// ── Parse helpers ─────────────────────────────────────────────────────────────

#[test]
fn test_parse_n_triples_basic() {
    let body = "<http://s> <http://p> <http://o> .\n";
    let triples = parse_n_triples(body).expect("should succeed");
    assert_eq!(triples.len(), 1);
    assert_eq!(triples[0].subject, "http://s");
    assert_eq!(triples[0].predicate, "http://p");
    assert_eq!(triples[0].object, "http://o");
}

#[test]
fn test_parse_n_triples_skips_comments() {
    let body = "# comment\n<http://s> <http://p> <http://o> .\n";
    let triples = parse_n_triples(body).expect("should succeed");
    assert_eq!(triples.len(), 1);
}

#[test]
fn test_parse_n_triples_empty() {
    let triples = parse_n_triples("").expect("should succeed");
    assert!(triples.is_empty());
}

#[test]
fn test_build_seed_expansion_sparql_includes_seeds() {
    let sparql = build_seed_expansion_sparql(
        &["http://example.org/e1", "http://example.org/e2"],
        None,
        500,
    );
    assert!(sparql.contains("<http://example.org/e1>"));
    assert!(sparql.contains("<http://example.org/e2>"));
    assert!(sparql.contains("LIMIT 500"));
}

#[test]
fn test_build_seed_expansion_sparql_with_graph() {
    let sparql = build_seed_expansion_sparql(
        &["http://example.org/e1"],
        Some("http://example.org/graph"),
        100,
    );
    assert!(sparql.contains("FROM <http://example.org/graph>"));
}

#[test]
fn test_build_same_as_sparql() {
    let sparql = build_same_as_sparql(&["http://a/e1", "http://b/e1"], None);
    assert!(sparql.contains("owl#sameAs"));
    assert!(sparql.contains("<http://a/e1>"));
}

#[test]
fn test_knowledge_graph_canonical_lookup() {
    let mut kg = KnowledgeGraph::new();
    kg.canonical_uris
        .insert("http://b/e1".to_string(), "http://canonical/e1".to_string());
    assert_eq!(kg.canonical("http://b/e1"), "http://canonical/e1");
    assert_eq!(kg.canonical("http://unknown"), "http://unknown");
}

#[test]
fn test_endpoint_auth_variants() {
    let bearer = EndpointAuth::Bearer {
        token: "tok123".to_string(),
    };
    let basic = EndpointAuth::Basic {
        username: "user".to_string(),
        password: "pass".to_string(),
    };
    let api = EndpointAuth::ApiKey {
        header: "X-API-Key".to_string(),
        key: "key123".to_string(),
    };
    assert_ne!(bearer, EndpointAuth::None);
    assert_ne!(basic, EndpointAuth::None);
    assert_ne!(api, EndpointAuth::None);
}

#[allow(dead_code)]
fn _use_types(_: EndpointMetrics, _: AggregateMetrics) {}
