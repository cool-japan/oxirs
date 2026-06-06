//! Worker types: HTTP endpoint executor implementation, federated expander, and metrics tracking.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, Semaphore};
use tracing::{debug, info, warn};

use crate::{GraphRAGError, GraphRAGResult, ScoredEntity, Triple};

use super::coordinator::{
    DistributedError, EndpointAuth, EndpointConfig, EndpointExecutor, FederatedGraphRAGConfig,
    KnowledgeGraph,
};

// ─────────────────────────────────────────────────────────────────────────────
// Internal result type
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a single endpoint query
#[derive(Debug)]
pub(super) struct EndpointResult {
    pub endpoint_name: String,
    pub triples: Vec<Triple>,
    pub latency_ms: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// SPARQL query builders
// ─────────────────────────────────────────────────────────────────────────────

/// Build a SPARQL CONSTRUCT query for seed expansion
pub(super) fn build_seed_expansion_sparql(
    seeds: &[&str],
    graph_uri: Option<&str>,
    limit: usize,
) -> String {
    let values: Vec<String> = seeds.iter().map(|s| format!("<{}>", s)).collect();
    let values_block = values.join(" ");

    let from_clause = match graph_uri {
        Some(g) => format!("FROM <{}>", g),
        None => String::new(),
    };

    format!(
        r#"CONSTRUCT {{
    ?s ?p ?o .
}}
{from}
WHERE {{
    VALUES ?seed {{ {seeds} }}
    {{
        BIND(?seed AS ?s)
        ?s ?p ?o .
    }} UNION {{
        ?s ?p ?seed .
        BIND(?seed AS ?o)
    }}
}}
LIMIT {limit}
"#,
        from = from_clause,
        seeds = values_block,
        limit = limit,
    )
}

/// Build a SPARQL SELECT query for sameAs links (re-exported for tests)
pub(super) fn build_same_as_sparql(uris: &[&str], graph_uri: Option<&str>) -> String {
    super::coordinator::build_same_as_sparql(uris, graph_uri)
}

// ─────────────────────────────────────────────────────────────────────────────
// HTTP executor implementation
// ─────────────────────────────────────────────────────────────────────────────

/// Default HTTP-based endpoint executor using reqwest
pub struct HttpEndpointExecutor {
    client: reqwest::Client,
}

impl HttpEndpointExecutor {
    /// Create a new HTTP executor
    pub fn new() -> GraphRAGResult<Self> {
        let client = reqwest::Client::builder()
            .build()
            .map_err(|e| GraphRAGError::InternalError(format!("HTTP client init: {e}")))?;
        Ok(Self { client })
    }

    /// Apply authentication headers to a request builder
    fn apply_auth(
        &self,
        builder: reqwest::RequestBuilder,
        auth: &EndpointAuth,
    ) -> reqwest::RequestBuilder {
        match auth {
            EndpointAuth::None => builder,
            EndpointAuth::Bearer { token } => {
                builder.header("Authorization", format!("Bearer {}", token))
            }
            EndpointAuth::Basic { username, password } => {
                builder.basic_auth(username, Some(password))
            }
            EndpointAuth::ApiKey { header, key } => builder.header(header.as_str(), key.as_str()),
        }
    }
}

#[async_trait::async_trait]
impl EndpointExecutor for HttpEndpointExecutor {
    async fn construct(
        &self,
        endpoint: &EndpointConfig,
        sparql: &str,
        timeout: Duration,
    ) -> GraphRAGResult<Vec<Triple>> {
        let builder: reqwest::RequestBuilder = self
            .client
            .post(&endpoint.url)
            .timeout(timeout)
            .header("Content-Type", "application/sparql-query")
            .header("Accept", "application/n-triples")
            .body(sparql.to_string());
        let builder = self.apply_auth(builder, &endpoint.auth);

        let response = builder
            .send()
            .await
            .map_err(|e| GraphRAGError::SparqlError(format!("HTTP error: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            return Err(GraphRAGError::SparqlError(format!(
                "HTTP {} from {}",
                status, endpoint.url
            )));
        }

        let body = response
            .text()
            .await
            .map_err(|e| GraphRAGError::SparqlError(format!("Response read error: {e}")))?;

        parse_n_triples(&body)
    }

    async fn select(
        &self,
        endpoint: &EndpointConfig,
        sparql: &str,
        timeout: Duration,
    ) -> GraphRAGResult<Vec<HashMap<String, String>>> {
        let builder: reqwest::RequestBuilder = self
            .client
            .post(&endpoint.url)
            .timeout(timeout)
            .header("Content-Type", "application/sparql-query")
            .header("Accept", "application/sparql-results+json")
            .body(sparql.to_string());
        let builder = self.apply_auth(builder, &endpoint.auth);

        let response = builder
            .send()
            .await
            .map_err(|e| GraphRAGError::SparqlError(format!("HTTP error: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            return Err(GraphRAGError::SparqlError(format!(
                "HTTP {} from {}",
                status, endpoint.url
            )));
        }

        let body = response
            .text()
            .await
            .map_err(|e| GraphRAGError::SparqlError(format!("Response read error: {e}")))?;

        parse_sparql_json_results(&body)
    }
}

/// Minimal N-Triples parser (handles `<s> <p> <o> .` and string literals)
pub(super) fn parse_n_triples(body: &str) -> GraphRAGResult<Vec<Triple>> {
    let mut triples = Vec::new();
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tokens: Vec<&str> = line.splitn(4, ' ').collect();
        if tokens.len() < 3 {
            continue;
        }
        let s = strip_angle_brackets(tokens[0]);
        let p = strip_angle_brackets(tokens[1]);
        let o = if tokens[2].starts_with('<') {
            strip_angle_brackets(tokens[2]).to_string()
        } else {
            tokens[2].to_string()
        };
        if !s.is_empty() && !p.is_empty() {
            triples.push(Triple::new(s, p, o));
        }
    }
    Ok(triples)
}

fn strip_angle_brackets(s: &str) -> &str {
    s.trim_start_matches('<').trim_end_matches('>')
}

/// Minimal SPARQL JSON results parser for SELECT queries
pub(super) fn parse_sparql_json_results(
    body: &str,
) -> GraphRAGResult<Vec<HashMap<String, String>>> {
    let json: serde_json::Value = serde_json::from_str(body)
        .map_err(|e| GraphRAGError::InternalError(format!("JSON parse error: {e}")))?;

    let vars: Vec<String> = json["head"]["vars"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();

    let bindings = json["results"]["bindings"]
        .as_array()
        .unwrap_or(&vec![])
        .clone();

    let mut rows = Vec::new();
    for binding in bindings {
        let mut row = HashMap::new();
        for var in &vars {
            if let Some(val) = binding.get(var) {
                let value = val["value"].as_str().unwrap_or("").to_string();
                row.insert(var.clone(), value);
            }
        }
        rows.push(row);
    }
    Ok(rows)
}

// ─────────────────────────────────────────────────────────────────────────────
// FederatedSubgraphExpander
// ─────────────────────────────────────────────────────────────────────────────

/// Expands subgraphs across multiple SPARQL endpoints concurrently and merges
/// the results into a single coherent [`KnowledgeGraph`].
pub struct FederatedSubgraphExpander<E: EndpointExecutor> {
    config: FederatedGraphRAGConfig,
    executor: Arc<E>,
}

impl<E: EndpointExecutor + 'static> FederatedSubgraphExpander<E> {
    /// Create a new expander with the given config and executor
    pub fn new(config: FederatedGraphRAGConfig, executor: Arc<E>) -> Self {
        Self { config, executor }
    }

    /// Expand subgraphs for the given seed entities across all active endpoints.
    pub async fn expand_federated(
        &self,
        seeds: &[ScoredEntity],
        endpoints: Option<&[String]>,
    ) -> GraphRAGResult<KnowledgeGraph> {
        if seeds.is_empty() {
            return Ok(KnowledgeGraph::new());
        }

        let seed_uris: Vec<&str> = seeds.iter().map(|s| s.uri.as_str()).collect();

        let active: Vec<&EndpointConfig> = match endpoints {
            Some(names) => self
                .config
                .active_endpoints()
                .into_iter()
                .filter(|ep| names.iter().any(|n| n == &ep.name))
                .collect(),
            None => self.config.active_endpoints(),
        };

        if active.is_empty() {
            return Err(DistributedError::NoHealthyEndpoints.into());
        }

        info!(
            "Federated expansion: {} seeds across {} endpoints",
            seeds.len(),
            active.len()
        );

        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrency));
        let results: Arc<Mutex<Vec<EndpointResult>>> = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();

        for ep in active {
            let ep = ep.clone();
            let executor = Arc::clone(&self.executor);
            let sem = Arc::clone(&semaphore);
            let results = Arc::clone(&results);
            let seed_uris: Vec<String> = seed_uris.iter().map(|s| s.to_string()).collect();
            let timeout_ms = ep.timeout_ms.unwrap_or(self.config.global_timeout_ms);
            let timeout = Duration::from_millis(timeout_ms);
            let retry_count = self.config.retry_count;
            let retry_delay = Duration::from_millis(self.config.retry_delay_ms);
            let partial_ok = self.config.partial_results_ok;

            let handle = tokio::spawn(async move {
                let _permit = match sem.acquire_owned().await {
                    Ok(p) => p,
                    Err(e) => {
                        warn!("Semaphore acquire failed: {e}");
                        return;
                    }
                };

                let sparql = build_seed_expansion_sparql(
                    &seed_uris.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                    ep.graph_uri.as_deref(),
                    ep.max_triples,
                );

                let start = Instant::now();
                let mut last_err = None;

                for attempt in 0..=retry_count {
                    if attempt > 0 {
                        tokio::time::sleep(retry_delay).await;
                    }

                    match executor.construct(&ep, &sparql, timeout).await {
                        Ok(triples) => {
                            let latency_ms = start.elapsed().as_millis() as u64;
                            debug!(
                                endpoint = %ep.name,
                                triples = triples.len(),
                                latency_ms,
                                "Endpoint query succeeded"
                            );
                            let mut guard = results.lock().await;
                            guard.push(EndpointResult {
                                endpoint_name: ep.name.clone(),
                                triples,
                                latency_ms,
                            });
                            return;
                        }
                        Err(e) => {
                            warn!(
                                endpoint = %ep.name,
                                attempt,
                                error = %e,
                                "Endpoint query failed"
                            );
                            last_err = Some(e);
                        }
                    }
                }

                if !partial_ok {
                    warn!(
                        endpoint = %ep.name,
                        error = ?last_err,
                        "Endpoint permanently failed and partial_results_ok=false"
                    );
                }
            });

            handles.push(handle);
        }

        for h in handles {
            if let Err(e) = h.await {
                warn!("Task join error: {e}");
            }
        }

        let endpoint_results = Arc::try_unwrap(results)
            .map_err(|_| GraphRAGError::InternalError("Arc unwrap failed".into()))?
            .into_inner();

        if endpoint_results.is_empty() && !self.config.partial_results_ok {
            return Err(DistributedError::NoHealthyEndpoints.into());
        }

        self.merge_results(endpoint_results)
    }

    /// Merge endpoint results into a unified [`KnowledgeGraph`], deduplicating
    /// triples and recording provenance.
    fn merge_results(&self, results: Vec<EndpointResult>) -> GraphRAGResult<KnowledgeGraph> {
        let mut kg = KnowledgeGraph::new();
        let mut seen: HashSet<(String, String, String)> = HashSet::new();

        let mut priority_map: HashMap<String, f64> = HashMap::new();
        for ep in &self.config.endpoints {
            priority_map.insert(ep.name.clone(), ep.priority);
        }

        let mut sorted_results = results;
        sorted_results.sort_by(|a, b| {
            let pa = priority_map.get(&a.endpoint_name).copied().unwrap_or(1.0);
            let pb = priority_map.get(&b.endpoint_name).copied().unwrap_or(1.0);
            pb.partial_cmp(&pa).unwrap_or(std::cmp::Ordering::Equal)
        });

        for result in sorted_results {
            for triple in result.triples {
                let key = (
                    triple.subject.clone(),
                    triple.predicate.clone(),
                    triple.object.clone(),
                );
                if seen.insert(key) {
                    kg.triples.push(triple);
                    kg.provenance.push(result.endpoint_name.clone());
                }
            }
        }

        Ok(kg)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DistributedGraphRAGMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Per-endpoint performance snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointMetrics {
    /// Endpoint name
    pub name: String,
    /// Total number of queries sent to this endpoint
    pub total_queries: u64,
    /// Number of successful queries
    pub successful_queries: u64,
    /// Number of failed queries
    pub failed_queries: u64,
    /// Total triples retrieved from this endpoint
    pub total_triples: u64,
    /// Exponential moving average of latency in milliseconds
    pub avg_latency_ms: f64,
    /// Minimum observed latency
    pub min_latency_ms: u64,
    /// Maximum observed latency
    pub max_latency_ms: u64,
    /// Hit rate: fraction of queries that returned ≥1 triple
    pub hit_rate: f64,
}

impl EndpointMetrics {
    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            total_queries: 0,
            successful_queries: 0,
            failed_queries: 0,
            total_triples: 0,
            avg_latency_ms: 0.0,
            min_latency_ms: u64::MAX,
            max_latency_ms: 0,
            hit_rate: 0.0,
        }
    }
}

/// Aggregate metrics across all endpoints
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregateMetrics {
    /// Total federation queries
    pub total_federation_queries: u64,
    /// Total triples gathered across all queries
    pub total_triples_gathered: u64,
    /// Number of entity resolution operations
    pub entity_resolution_ops: u64,
    /// Average federation latency (wall-clock)
    pub avg_federation_latency_ms: f64,
    /// Number of partial result failures (some endpoints failed)
    pub partial_failure_count: u64,
}

/// Thread-safe metrics tracker for distributed GraphRAG operations
pub struct DistributedGraphRAGMetrics {
    endpoint_metrics: Arc<tokio::sync::RwLock<HashMap<String, EndpointMetrics>>>,
    aggregate: Arc<tokio::sync::RwLock<AggregateMetrics>>,
    ema_alpha: f64,
}

impl DistributedGraphRAGMetrics {
    /// Create a new metrics tracker
    pub fn new(endpoints: &[EndpointConfig]) -> Self {
        let mut ep_map = HashMap::new();
        for ep in endpoints {
            ep_map.insert(ep.name.clone(), EndpointMetrics::new(&ep.name));
        }

        Self {
            endpoint_metrics: Arc::new(tokio::sync::RwLock::new(ep_map)),
            aggregate: Arc::new(tokio::sync::RwLock::new(AggregateMetrics::default())),
            ema_alpha: 0.2,
        }
    }

    /// Record a successful query result for an endpoint
    pub async fn record_success(&self, endpoint_name: &str, latency_ms: u64, triple_count: usize) {
        let mut guard = self.endpoint_metrics.write().await;
        let m = guard
            .entry(endpoint_name.to_string())
            .or_insert_with(|| EndpointMetrics::new(endpoint_name));

        m.total_queries += 1;
        m.successful_queries += 1;
        m.total_triples += triple_count as u64;

        if m.total_queries == 1 {
            m.avg_latency_ms = latency_ms as f64;
        } else {
            m.avg_latency_ms =
                self.ema_alpha * latency_ms as f64 + (1.0 - self.ema_alpha) * m.avg_latency_ms;
        }

        if latency_ms < m.min_latency_ms {
            m.min_latency_ms = latency_ms;
        }
        if latency_ms > m.max_latency_ms {
            m.max_latency_ms = latency_ms;
        }

        let hits = m.successful_queries - if triple_count == 0 { 1 } else { 0 };
        m.hit_rate = hits as f64 / m.total_queries as f64;
    }

    /// Record a failed query for an endpoint
    pub async fn record_failure(&self, endpoint_name: &str) {
        let mut guard = self.endpoint_metrics.write().await;
        let m = guard
            .entry(endpoint_name.to_string())
            .or_insert_with(|| EndpointMetrics::new(endpoint_name));

        m.total_queries += 1;
        m.failed_queries += 1;
        m.hit_rate = if m.total_queries > 0 {
            m.successful_queries as f64 / m.total_queries as f64
        } else {
            0.0
        };
    }

    /// Record a completed federation query
    pub async fn record_federation_query(
        &self,
        wall_latency_ms: u64,
        total_triples: usize,
        had_partial_failure: bool,
    ) {
        let mut agg = self.aggregate.write().await;
        agg.total_federation_queries += 1;
        agg.total_triples_gathered += total_triples as u64;
        if had_partial_failure {
            agg.partial_failure_count += 1;
        }
        if agg.total_federation_queries == 1 {
            agg.avg_federation_latency_ms = wall_latency_ms as f64;
        } else {
            agg.avg_federation_latency_ms = self.ema_alpha * wall_latency_ms as f64
                + (1.0 - self.ema_alpha) * agg.avg_federation_latency_ms;
        }
    }

    /// Record an entity resolution operation
    pub async fn record_entity_resolution(&self) {
        let mut agg = self.aggregate.write().await;
        agg.entity_resolution_ops += 1;
    }

    /// Retrieve a snapshot of metrics for a specific endpoint
    pub async fn endpoint_snapshot(&self, name: &str) -> Option<EndpointMetrics> {
        self.endpoint_metrics.read().await.get(name).cloned()
    }

    /// Retrieve a snapshot of all endpoint metrics
    pub async fn all_endpoint_snapshots(&self) -> Vec<EndpointMetrics> {
        self.endpoint_metrics
            .read()
            .await
            .values()
            .cloned()
            .collect()
    }

    /// Retrieve aggregate metrics
    pub async fn aggregate_snapshot(&self) -> AggregateMetrics {
        self.aggregate.read().await.clone()
    }

    /// Return the endpoint name with the lowest average latency
    pub async fn fastest_endpoint(&self) -> Option<String> {
        let guard = self.endpoint_metrics.read().await;
        guard
            .values()
            .filter(|m| m.successful_queries > 0)
            .min_by(|a, b| {
                a.avg_latency_ms
                    .partial_cmp(&b.avg_latency_ms)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|m| m.name.clone())
    }

    /// Return the endpoint with the highest hit rate
    pub async fn best_hit_rate_endpoint(&self) -> Option<String> {
        let guard = self.endpoint_metrics.read().await;
        guard
            .values()
            .filter(|m| m.total_queries > 0)
            .max_by(|a, b| {
                a.hit_rate
                    .partial_cmp(&b.hit_rate)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|m| m.name.clone())
    }
}
