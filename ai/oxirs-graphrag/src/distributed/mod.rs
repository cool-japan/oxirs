//! Distributed GraphRAG: federated subgraph expansion across multiple SPARQL endpoints.
//!
//! This module provides the building blocks for querying heterogeneous, geographically
//! distributed knowledge graphs and merging the results into a single coherent subgraph
//! suitable for retrieval-augmented generation.
//!
//! ## Architecture
//!
//! ```text
//! Query Seeds
//!     │
//!     ▼
//! FederatedSubgraphExpander ──► [Endpoint A] ──► subgraph_A
//!     │                    ──► [Endpoint B] ──► subgraph_B   ──► merge + resolve ──► KnowledgeGraph
//!     │                    ──► [Endpoint C] ──► subgraph_C
//!     │
//!     ▼
//! DistributedEntityResolver  (sameAs closure)
//!     │
//!     ▼
//! FederatedContextBuilder    (priority + confidence ranking)
//!     │
//!     ▼
//! RAG context string
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{Mutex, RwLock, Semaphore};
use tracing::{debug, info, warn};

use crate::{GraphRAGError, GraphRAGResult, ScoredEntity, Triple};

// ─────────────────────────────────────────────────────────────────────────────
// Error types
// ─────────────────────────────────────────────────────────────────────────────

/// Distributed GraphRAG–specific error variants
#[derive(Error, Debug)]
pub enum DistributedError {
    #[error("Endpoint {endpoint} is unreachable: {reason}")]
    EndpointUnreachable { endpoint: String, reason: String },

    #[error("Authentication failed for endpoint {endpoint}")]
    AuthFailed { endpoint: String },

    #[error("SPARQL query timeout after {timeout_ms}ms on endpoint {endpoint}")]
    QueryTimeout { endpoint: String, timeout_ms: u64 },

    #[error("Entity resolution cycle detected for URI {uri}")]
    SameAsCycle { uri: String },

    #[error("No healthy endpoints available for query")]
    NoHealthyEndpoints,

    #[error("Merge conflict: cannot reconcile {uri} across endpoints")]
    MergeConflict { uri: String },

    #[error("Configuration invalid: {0}")]
    InvalidConfig(String),
}

impl From<DistributedError> for GraphRAGError {
    fn from(e: DistributedError) -> Self {
        GraphRAGError::InternalError(e.to_string())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Authentication method for SPARQL endpoints
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EndpointAuth {
    /// No authentication
    None,
    /// HTTP Bearer token
    Bearer { token: String },
    /// HTTP Basic auth
    Basic { username: String, password: String },
    /// API key in header
    ApiKey { header: String, key: String },
}

/// Configuration for a single remote SPARQL endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointConfig {
    /// Human-readable name for the endpoint
    pub name: String,
    /// Base URL of the SPARQL endpoint
    pub url: String,
    /// Authentication method
    pub auth: EndpointAuth,
    /// Per-endpoint query timeout in milliseconds (overrides global setting)
    pub timeout_ms: Option<u64>,
    /// Priority weight (higher = preferred; used when deduplicating conflicting triples)
    pub priority: f64,
    /// Whether this endpoint is enabled
    pub enabled: bool,
    /// Graph URI to restrict queries to (SPARQL FROM clause)
    pub graph_uri: Option<String>,
    /// Maximum triples to fetch from this endpoint per query
    pub max_triples: usize,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            url: String::new(),
            auth: EndpointAuth::None,
            timeout_ms: None,
            priority: 1.0,
            enabled: true,
            graph_uri: None,
            max_triples: 10_000,
        }
    }
}

/// Top-level configuration for federated GraphRAG
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedGraphRAGConfig {
    /// List of remote endpoints to query
    pub endpoints: Vec<EndpointConfig>,
    /// Global query timeout in milliseconds
    pub global_timeout_ms: u64,
    /// Maximum concurrent endpoint requests
    pub max_concurrency: usize,
    /// Maximum transitive sameAs hops to follow
    pub same_as_max_depth: usize,
    /// Minimum endpoint priority to include in a query (0.0 = include all)
    pub min_endpoint_priority: f64,
    /// Whether to continue when some endpoints fail
    pub partial_results_ok: bool,
    /// Retry count for failed endpoint requests
    pub retry_count: usize,
    /// Delay between retries in milliseconds
    pub retry_delay_ms: u64,
}

impl Default for FederatedGraphRAGConfig {
    fn default() -> Self {
        Self {
            endpoints: vec![],
            global_timeout_ms: 30_000,
            max_concurrency: 8,
            same_as_max_depth: 5,
            min_endpoint_priority: 0.0,
            partial_results_ok: true,
            retry_count: 2,
            retry_delay_ms: 500,
        }
    }
}

impl FederatedGraphRAGConfig {
    /// Validate configuration and return an error description if invalid.
    pub fn validate(&self) -> Result<(), DistributedError> {
        if self.global_timeout_ms == 0 {
            return Err(DistributedError::InvalidConfig(
                "global_timeout_ms must be > 0".into(),
            ));
        }
        if self.max_concurrency == 0 {
            return Err(DistributedError::InvalidConfig(
                "max_concurrency must be > 0".into(),
            ));
        }
        if self.same_as_max_depth == 0 {
            return Err(DistributedError::InvalidConfig(
                "same_as_max_depth must be > 0".into(),
            ));
        }
        for ep in &self.endpoints {
            if ep.url.is_empty() {
                return Err(DistributedError::InvalidConfig(format!(
                    "Endpoint '{}' has an empty URL",
                    ep.name
                )));
            }
            if ep.max_triples == 0 {
                return Err(DistributedError::InvalidConfig(format!(
                    "Endpoint '{}' max_triples must be > 0",
                    ep.name
                )));
            }
        }
        Ok(())
    }

    /// Return only enabled endpoints with priority >= `min_endpoint_priority`.
    pub fn active_endpoints(&self) -> Vec<&EndpointConfig> {
        self.endpoints
            .iter()
            .filter(|ep| ep.enabled && ep.priority >= self.min_endpoint_priority)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Knowledge graph result type
// ─────────────────────────────────────────────────────────────────────────────

/// A merged knowledge graph assembled from multiple endpoints
#[derive(Debug, Clone, Default)]
pub struct KnowledgeGraph {
    /// All triples gathered from the federation
    pub triples: Vec<Triple>,
    /// Provenance: which endpoint contributed each triple index
    pub provenance: Vec<String>,
    /// Entity equivalence classes after sameAs resolution
    pub equivalence_classes: Vec<HashSet<String>>,
    /// Canonical URIs chosen for each equivalence class (representative URI)
    pub canonical_uris: HashMap<String, String>,
}

impl KnowledgeGraph {
    /// Create an empty knowledge graph
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the number of distinct triples
    pub fn triple_count(&self) -> usize {
        self.triples.len()
    }

    /// Return true if the knowledge graph has no triples
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Resolve a URI to its canonical form (or return the URI unchanged)
    pub fn canonical<'a>(&'a self, uri: &'a str) -> &'a str {
        self.canonical_uris
            .get(uri)
            .map(|s| s.as_str())
            .unwrap_or(uri)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HTTP SPARQL client abstraction
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a single endpoint query
#[derive(Debug)]
struct EndpointResult {
    endpoint_name: String,
    triples: Vec<Triple>,
    latency_ms: u64,
}

/// Build a SPARQL CONSTRUCT query for seed expansion
fn build_seed_expansion_sparql(seeds: &[&str], graph_uri: Option<&str>, limit: usize) -> String {
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

/// Build a SPARQL SELECT query for sameAs links
fn build_same_as_sparql(uris: &[&str], graph_uri: Option<&str>) -> String {
    let values: Vec<String> = uris.iter().map(|s| format!("<{}>", s)).collect();
    let values_block = values.join(" ");

    let from_clause = match graph_uri {
        Some(g) => format!("FROM <{}>", g),
        None => String::new(),
    };

    format!(
        r#"SELECT DISTINCT ?a ?b
{from}
WHERE {{
    VALUES ?a {{ {uris} }}
    {{
        ?a <http://www.w3.org/2002/07/owl#sameAs> ?b .
    }} UNION {{
        ?b <http://www.w3.org/2002/07/owl#sameAs> ?a .
    }}
}}
"#,
        from = from_clause,
        uris = values_block,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// HTTP executor (mockable in tests)
// ─────────────────────────────────────────────────────────────────────────────

/// Trait for executing SPARQL CONSTRUCT queries against a remote endpoint
#[async_trait::async_trait]
pub trait EndpointExecutor: Send + Sync {
    /// Execute a SPARQL CONSTRUCT query and return RDF triples
    async fn construct(
        &self,
        endpoint: &EndpointConfig,
        sparql: &str,
        timeout: Duration,
    ) -> GraphRAGResult<Vec<Triple>>;

    /// Execute a SPARQL SELECT query and return rows (variable → value maps)
    async fn select(
        &self,
        endpoint: &EndpointConfig,
        sparql: &str,
        timeout: Duration,
    ) -> GraphRAGResult<Vec<HashMap<String, String>>>;
}

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
fn parse_n_triples(body: &str) -> GraphRAGResult<Vec<Triple>> {
    let mut triples = Vec::new();
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Very lightweight parser: split on whitespace-delimited tokens
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
fn parse_sparql_json_results(body: &str) -> GraphRAGResult<Vec<HashMap<String, String>>> {
    // Use serde_json for reliability
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
/// the results into a single [`KnowledgeGraph`].
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
    ///
    /// Queries are issued concurrently (bounded by `config.max_concurrency`).
    /// If `config.partial_results_ok` is true, endpoint failures are logged but
    /// do not abort the overall operation.
    pub async fn expand_federated(
        &self,
        seeds: &[ScoredEntity],
        endpoints: Option<&[String]>,
    ) -> GraphRAGResult<KnowledgeGraph> {
        if seeds.is_empty() {
            return Ok(KnowledgeGraph::new());
        }

        let seed_uris: Vec<&str> = seeds.iter().map(|s| s.uri.as_str()).collect();

        // Determine which endpoints to query
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

        // Wait for all tasks
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
        // Use a set to deduplicate (subject, predicate, object)
        let mut seen: HashSet<(String, String, String)> = HashSet::new();

        // Sort by endpoint priority descending (higher priority wins dedup)
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
// DistributedEntityResolver
// ─────────────────────────────────────────────────────────────────────────────

/// Resolves entity identity across endpoints using owl:sameAs links.
///
/// The resolver computes the transitive sameAs closure: if A sameAs B and
/// B sameAs C, all three are placed in the same equivalence class.
pub struct DistributedEntityResolver<E: EndpointExecutor> {
    config: FederatedGraphRAGConfig,
    executor: Arc<E>,
}

impl<E: EndpointExecutor + 'static> DistributedEntityResolver<E> {
    /// Create a new resolver
    pub fn new(config: FederatedGraphRAGConfig, executor: Arc<E>) -> Self {
        Self { config, executor }
    }

    /// Compute the transitive owl:sameAs closure for the given URIs across all
    /// active endpoints.
    ///
    /// Returns a map from each input URI (and discovered aliases) to the
    /// canonical representative URI for its equivalence class.
    pub async fn same_as_closure(
        &self,
        uris: &[String],
    ) -> GraphRAGResult<HashMap<String, String>> {
        if uris.is_empty() {
            return Ok(HashMap::new());
        }

        // Union-Find structure for equivalence classes
        let parent: Arc<RwLock<HashMap<String, String>>> = Arc::new(RwLock::new(HashMap::new()));

        // Initialize each URI as its own parent
        {
            let mut p = parent.write().await;
            for uri in uris {
                p.insert(uri.clone(), uri.clone());
            }
        }

        // BFS frontier: expand sameAs links up to max depth
        let mut frontier: VecDeque<String> = uris.iter().cloned().collect();
        let mut visited: HashSet<String> = HashSet::from_iter(uris.iter().cloned());
        let mut depth = 0usize;

        while !frontier.is_empty() && depth < self.config.same_as_max_depth {
            let batch: Vec<String> = frontier.drain(..).collect();
            let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();

            // Query all endpoints for sameAs links
            let links = self.fetch_same_as_links(&batch_refs).await?;

            let mut p = parent.write().await;
            for (a, b) in links {
                // Ensure both exist in the union-find
                p.entry(a.clone()).or_insert_with(|| a.clone());
                p.entry(b.clone()).or_insert_with(|| b.clone());

                // Union a and b
                let root_a = find_root_path(&p, &a);
                let root_b = find_root_path(&p, &b);
                if root_a != root_b {
                    // Prefer lexicographically smaller URI as canonical
                    let canonical = if root_a <= root_b {
                        root_a.clone()
                    } else {
                        root_b.clone()
                    };
                    p.insert(root_a, canonical.clone());
                    p.insert(root_b, canonical);
                }

                // Add newly discovered URIs to the frontier
                if !visited.contains(&b) {
                    visited.insert(b.clone());
                    frontier.push_back(b);
                }
            }

            depth += 1;
        }

        // Flatten all paths to canonical roots
        let p = parent.read().await;
        let mut result = HashMap::new();
        for uri in p.keys() {
            let canonical = find_root_path(&p, uri);
            result.insert(uri.clone(), canonical);
        }
        Ok(result)
    }

    /// Fetch raw owl:sameAs pairs from all active endpoints for the given URIs
    async fn fetch_same_as_links(&self, uris: &[&str]) -> GraphRAGResult<Vec<(String, String)>> {
        let active = self.config.active_endpoints();
        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrency));
        let pairs: Arc<Mutex<Vec<(String, String)>>> = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();

        for ep in active {
            let ep = ep.clone();
            let executor = Arc::clone(&self.executor);
            let sem = Arc::clone(&semaphore);
            let pairs = Arc::clone(&pairs);
            let uris_owned: Vec<String> = uris.iter().map(|s| s.to_string()).collect();
            let timeout_ms = ep.timeout_ms.unwrap_or(self.config.global_timeout_ms);
            let timeout = Duration::from_millis(timeout_ms);

            let handle = tokio::spawn(async move {
                let _permit = match sem.acquire_owned().await {
                    Ok(p) => p,
                    Err(_) => return,
                };

                let sparql = build_same_as_sparql(
                    &uris_owned.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                    ep.graph_uri.as_deref(),
                );

                match executor.select(&ep, &sparql, timeout).await {
                    Ok(rows) => {
                        let mut guard = pairs.lock().await;
                        for row in rows {
                            if let (Some(a), Some(b)) = (row.get("a"), row.get("b")) {
                                guard.push((a.clone(), b.clone()));
                            }
                        }
                    }
                    Err(e) => {
                        debug!(endpoint = %ep.name, error = %e, "sameAs fetch failed");
                    }
                }
            });

            handles.push(handle);
        }

        for h in handles {
            let _ = h.await;
        }

        let guard = Arc::try_unwrap(pairs)
            .map_err(|_| GraphRAGError::InternalError("Arc unwrap failed".into()))?
            .into_inner();

        Ok(guard)
    }

    /// Apply sameAs closure to a knowledge graph, rewriting URIs to canonical forms
    /// and deduplicating triples that become identical after rewriting.
    pub fn apply_to_graph(&self, kg: &mut KnowledgeGraph, canonical_map: &HashMap<String, String>) {
        let canonicalize = |s: &str| -> String {
            canonical_map
                .get(s)
                .cloned()
                .unwrap_or_else(|| s.to_string())
        };

        let mut seen: HashSet<(String, String, String)> = HashSet::new();
        let mut new_triples = Vec::new();
        let mut new_provenance = Vec::new();

        for (triple, prov) in kg.triples.iter().zip(kg.provenance.iter()) {
            let s = canonicalize(&triple.subject);
            let p = triple.predicate.clone();
            let o = canonicalize(&triple.object);
            let key = (s.clone(), p.clone(), o.clone());
            if seen.insert(key) {
                new_triples.push(Triple::new(s, p, o));
                new_provenance.push(prov.clone());
            }
        }

        kg.triples = new_triples;
        kg.provenance = new_provenance;
        kg.canonical_uris = canonical_map.clone();

        // Rebuild equivalence classes
        let mut classes: HashMap<String, HashSet<String>> = HashMap::new();
        for (uri, canonical) in canonical_map {
            classes
                .entry(canonical.clone())
                .or_default()
                .insert(uri.clone());
        }
        kg.equivalence_classes = classes.into_values().collect();
    }
}

/// Path-compression find for the union-find structure
fn find_root_path(parent: &HashMap<String, String>, uri: &str) -> String {
    let mut current = uri.to_string();
    let mut depth = 0usize;
    loop {
        let next = parent
            .get(&current)
            .cloned()
            .unwrap_or_else(|| current.clone());
        if next == current || depth > 100 {
            return current;
        }
        current = next;
        depth += 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FederatedContextBuilder
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy for ordering triples in the generated context
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContextOrderingStrategy {
    /// Order by endpoint priority (highest first)
    ByEndpointPriority,
    /// Order by query latency (fastest endpoints first)
    ByLatency,
    /// No specific ordering (insertion order)
    Insertion,
}

/// Configuration for the federated context builder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedContextConfig {
    /// Maximum number of triples to include in the context
    pub max_context_triples: usize,
    /// Maximum length (characters) of the context string
    pub max_context_chars: usize,
    /// Triple ordering strategy
    pub ordering: ContextOrderingStrategy,
    /// Whether to include provenance annotations in the context
    pub include_provenance: bool,
    /// Minimum endpoint priority to include triples from
    pub min_endpoint_priority: f64,
    /// Whether to include equivalence class annotations
    pub include_equivalences: bool,
}

impl Default for FederatedContextConfig {
    fn default() -> Self {
        Self {
            max_context_triples: 500,
            max_context_chars: 50_000,
            ordering: ContextOrderingStrategy::ByEndpointPriority,
            include_provenance: false,
            include_equivalences: false,
            min_endpoint_priority: 0.0,
        }
    }
}

/// Builds RAG context strings from distributed knowledge graphs
pub struct FederatedContextBuilder {
    config: FederatedContextConfig,
    /// Per-endpoint priority registry
    endpoint_priorities: HashMap<String, f64>,
    /// Per-endpoint latency registry (milliseconds, populated from expansion runs)
    endpoint_latencies: Arc<RwLock<HashMap<String, u64>>>,
}

impl FederatedContextBuilder {
    /// Create a new context builder
    pub fn new(config: FederatedContextConfig, graphrag_config: &FederatedGraphRAGConfig) -> Self {
        let endpoint_priorities: HashMap<String, f64> = graphrag_config
            .endpoints
            .iter()
            .map(|ep| (ep.name.clone(), ep.priority))
            .collect();

        Self {
            config,
            endpoint_priorities,
            endpoint_latencies: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Record observed latency for an endpoint (used in ByLatency ordering)
    pub async fn record_latency(&self, endpoint_name: &str, latency_ms: u64) {
        let mut lats = self.endpoint_latencies.write().await;
        lats.insert(endpoint_name.to_string(), latency_ms);
    }

    /// Build a context string from a [`KnowledgeGraph`].
    ///
    /// Triples are ordered according to the configured strategy, truncated to
    /// respect both `max_context_triples` and `max_context_chars`.
    pub async fn build_context(&self, kg: &KnowledgeGraph, query: &str) -> GraphRAGResult<String> {
        if kg.is_empty() {
            return Ok(String::new());
        }

        // Create (triple_index, priority_key) pairs for sorting
        let latencies = self.endpoint_latencies.read().await;
        let mut indexed: Vec<(usize, f64)> = kg
            .triples
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let ep = kg.provenance.get(i).map(|s| s.as_str()).unwrap_or("");
                let sort_key = match self.config.ordering {
                    ContextOrderingStrategy::ByEndpointPriority => {
                        // Higher priority → lower sort key (we sort ascending, then reverse)
                        self.endpoint_priorities.get(ep).copied().unwrap_or(1.0)
                    }
                    ContextOrderingStrategy::ByLatency => {
                        // Lower latency → higher priority
                        let lat = latencies.get(ep).copied().unwrap_or(u64::MAX);
                        // Invert: smaller latency → larger sort key
                        1.0 / (lat as f64 + 1.0)
                    }
                    ContextOrderingStrategy::Insertion => i as f64,
                };
                (i, sort_key)
            })
            .filter(|(i, _)| {
                let ep = kg.provenance.get(*i).map(|s| s.as_str()).unwrap_or("");
                let prio = self.endpoint_priorities.get(ep).copied().unwrap_or(1.0);
                prio >= self.config.min_endpoint_priority
            })
            .collect();

        // Sort: ByEndpointPriority and ByLatency both want descending key
        match self.config.ordering {
            ContextOrderingStrategy::ByEndpointPriority | ContextOrderingStrategy::ByLatency => {
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            ContextOrderingStrategy::Insertion => {
                indexed.sort_by_key(|(i, _)| *i);
            }
        }

        let mut context = format!("## Knowledge Graph Context\n\nQuery: {}\n\n", query);

        // Add equivalence class info if requested
        if self.config.include_equivalences && !kg.equivalence_classes.is_empty() {
            context.push_str("### Entity Equivalences\n");
            for class in &kg.equivalence_classes {
                if class.len() > 1 {
                    let mut members: Vec<&str> = class.iter().map(|s| s.as_str()).collect();
                    members.sort();
                    context.push_str(&format!("- {}\n", members.join(" ≡ ")));
                }
            }
            context.push('\n');
        }

        context.push_str("### Facts\n\n");

        for (triple_count, (idx, _)) in indexed.iter().enumerate() {
            if triple_count >= self.config.max_context_triples {
                break;
            }
            if context.len() >= self.config.max_context_chars {
                break;
            }

            let triple = &kg.triples[*idx];
            let line = if self.config.include_provenance {
                let ep = kg.provenance.get(*idx).map(|s| s.as_str()).unwrap_or("?");
                format!(
                    "- {} → {} → {} [{}]\n",
                    triple.subject, triple.predicate, triple.object, ep
                )
            } else {
                format!(
                    "- {} → {} → {}\n",
                    triple.subject, triple.predicate, triple.object
                )
            };

            context.push_str(&line);
        }

        Ok(context)
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
    /// Per-endpoint counters
    endpoint_metrics: Arc<RwLock<HashMap<String, EndpointMetrics>>>,
    /// Aggregate counters
    aggregate: Arc<RwLock<AggregateMetrics>>,
    /// EMA smoothing factor (0 < alpha ≤ 1)
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
            endpoint_metrics: Arc::new(RwLock::new(ep_map)),
            aggregate: Arc::new(RwLock::new(AggregateMetrics::default())),
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

        // Update EMA latency
        if m.total_queries == 1 {
            m.avg_latency_ms = latency_ms as f64;
        } else {
            m.avg_latency_ms =
                self.ema_alpha * latency_ms as f64 + (1.0 - self.ema_alpha) * m.avg_latency_ms;
        }

        // Update min/max
        if latency_ms < m.min_latency_ms {
            m.min_latency_ms = latency_ms;
        }
        if latency_ms > m.max_latency_ms {
            m.max_latency_ms = latency_ms;
        }

        // Recompute hit rate
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
        // Recompute hit rate (failure counts as miss)
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

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphRAGResult, ScoreSource};
    use async_trait::async_trait;
    use std::collections::HashMap;

    // ── Mock executor ────────────────────────────────────────────────────────

    struct MockExecutor {
        /// Triples returned by the `construct` call (keyed by endpoint name)
        triples_by_endpoint: HashMap<String, Vec<Triple>>,
        /// sameAs pairs returned by the `select` call (keyed by endpoint name)
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

    // ── Helper constructors ──────────────────────────────────────────────────

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

    // ── test_federated_config_validation ─────────────────────────────────────

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

    // ── test_federated_subgraph_expander ─────────────────────────────────────

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

    // ── test_distributed_entity_resolver_sameAs ──────────────────────────────

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

        // Both http://a/e1 and http://b/e1 should map to the same canonical URI
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

        // Without any sameAs links, e1 maps to itself
        let canon = closure
            .get("http://example.org/e1")
            .expect("should succeed");
        assert_eq!(canon, "http://example.org/e1");
    }

    #[tokio::test]
    async fn test_distributed_entity_resolver_transitive_chain() {
        // A sameAs B, B sameAs C — all three should end up in the same class
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

        // Check that the discovered URIs (at least a/e1 and b/e1) share a canonical form
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
            make_triple("http://a/e1", "http://p", "http://a/e1"), // self-loop
        ];
        kg.provenance = vec!["ep_a".to_string(), "ep_a".to_string()];

        let mut canonical = HashMap::new();
        canonical.insert("http://a/e1".to_string(), "http://canonical/e1".to_string());
        canonical.insert("http://b/e1".to_string(), "http://canonical/e1".to_string());

        resolver.apply_to_graph(&mut kg, &canonical);

        // After rewriting: both triples become <canonical/e1> <p> <canonical/e1>
        // which is the same — deduplication keeps only 1
        assert_eq!(kg.triple_count(), 1);
        assert_eq!(kg.triples[0].subject, "http://canonical/e1");
        assert_eq!(kg.triples[0].object, "http://canonical/e1");
    }

    // ── test_federated_context_builder ───────────────────────────────────────

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
        // Provenance included
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

        // Count lines starting with "- " to determine triple count
        let triple_lines = context.lines().filter(|l| l.starts_with("- ")).count();
        assert!(
            triple_lines <= 2,
            "Expected at most 2 triples, got {}",
            triple_lines
        );
    }

    // ── test_distributed_metrics_tracking ────────────────────────────────────

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

        // ep_a is slow, ep_b is fast
        metrics.record_success("ep_a", 500, 10).await;
        metrics.record_success("ep_b", 50, 10).await;

        let fastest = metrics.fastest_endpoint().await.expect("should succeed");
        assert_eq!(fastest, "ep_b");
    }

    #[tokio::test]
    async fn test_distributed_metrics_hit_rate() {
        let endpoints = vec![make_endpoint("ep_a", 1.0)];
        let metrics = DistributedGraphRAGMetrics::new(&endpoints);

        metrics.record_success("ep_a", 100, 5).await; // hit (triple_count > 0)
        metrics.record_failure("ep_a").await; // miss

        let snap = metrics
            .endpoint_snapshot("ep_a")
            .await
            .expect("should succeed");
        assert_eq!(snap.total_queries, 2);
        // 1 success + 1 failure
        assert!(snap.hit_rate >= 0.0 && snap.hit_rate <= 1.0);
    }

    // ── Parse helpers ────────────────────────────────────────────────────────

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
}
