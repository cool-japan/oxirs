//! Coordinator types: configuration, knowledge graph, entity resolution, and context building.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{Mutex, RwLock, Semaphore};
use tracing::debug;

use crate::{GraphRAGError, GraphRAGResult, Triple};

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
// Authentication
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

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

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
// Endpoint executor trait (defined here to break the worker ↔ coordinator cycle)
// ─────────────────────────────────────────────────────────────────────────────

/// Trait for executing SPARQL queries against a remote endpoint.
/// Defined in `coordinator` so both coordinator and worker can reference it
/// without a circular dependency.
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

// ─────────────────────────────────────────────────────────────────────────────
// DistributedEntityResolver
// ─────────────────────────────────────────────────────────────────────────────

/// Resolves entity identity across endpoints using owl:sameAs links.
///
/// The resolver computes the transitive sameAs closure: if A sameAs B and
/// B sameAs C, all three are placed in the same equivalence class.
pub struct DistributedEntityResolver<E: EndpointExecutor> {
    pub(super) config: FederatedGraphRAGConfig,
    pub(super) executor: Arc<E>,
}

impl<E: EndpointExecutor + 'static> DistributedEntityResolver<E> {
    /// Create a new resolver
    pub fn new(config: FederatedGraphRAGConfig, executor: Arc<E>) -> Self {
        Self { config, executor }
    }

    /// Compute the transitive owl:sameAs closure for the given URIs across all
    /// active endpoints.
    pub async fn same_as_closure(
        &self,
        uris: &[String],
    ) -> GraphRAGResult<HashMap<String, String>> {
        if uris.is_empty() {
            return Ok(HashMap::new());
        }

        let parent: Arc<RwLock<HashMap<String, String>>> = Arc::new(RwLock::new(HashMap::new()));

        {
            let mut p = parent.write().await;
            for uri in uris {
                p.insert(uri.clone(), uri.clone());
            }
        }

        let mut frontier: VecDeque<String> = uris.iter().cloned().collect();
        let mut visited: HashSet<String> = HashSet::from_iter(uris.iter().cloned());
        let mut depth = 0usize;

        while !frontier.is_empty() && depth < self.config.same_as_max_depth {
            let batch: Vec<String> = frontier.drain(..).collect();
            let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();

            let links = self.fetch_same_as_links(&batch_refs).await?;

            let mut p = parent.write().await;
            for (a, b) in links {
                p.entry(a.clone()).or_insert_with(|| a.clone());
                p.entry(b.clone()).or_insert_with(|| b.clone());

                let root_a = find_root_path(&p, &a);
                let root_b = find_root_path(&p, &b);
                if root_a != root_b {
                    let canonical = if root_a <= root_b {
                        root_a.clone()
                    } else {
                        root_b.clone()
                    };
                    p.insert(root_a, canonical.clone());
                    p.insert(root_b, canonical);
                }

                if !visited.contains(&b) {
                    visited.insert(b.clone());
                    frontier.push_back(b);
                }
            }

            depth += 1;
        }

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
pub(super) fn find_root_path(parent: &HashMap<String, String>, uri: &str) -> String {
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

/// Build a SPARQL SELECT query for sameAs links (shared by coordinator and worker)
pub(super) fn build_same_as_sparql(uris: &[&str], graph_uri: Option<&str>) -> String {
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
    pub async fn build_context(&self, kg: &KnowledgeGraph, query: &str) -> GraphRAGResult<String> {
        if kg.is_empty() {
            return Ok(String::new());
        }

        let latencies = self.endpoint_latencies.read().await;
        let mut indexed: Vec<(usize, f64)> = kg
            .triples
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let ep = kg.provenance.get(i).map(|s| s.as_str()).unwrap_or("");
                let sort_key = match self.config.ordering {
                    ContextOrderingStrategy::ByEndpointPriority => {
                        self.endpoint_priorities.get(ep).copied().unwrap_or(1.0)
                    }
                    ContextOrderingStrategy::ByLatency => {
                        let lat = latencies.get(ep).copied().unwrap_or(u64::MAX);
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

        match self.config.ordering {
            ContextOrderingStrategy::ByEndpointPriority | ContextOrderingStrategy::ByLatency => {
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            ContextOrderingStrategy::Insertion => {
                indexed.sort_by_key(|(i, _)| *i);
            }
        }

        let mut context = format!("## Knowledge Graph Context\n\nQuery: {}\n\n", query);

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
