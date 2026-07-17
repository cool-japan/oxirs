//! # OxiRS GraphQL - GraphQL Interface for RDF
//!
//! [![Version](https://img.shields.io/badge/version-0.3.3-blue)](https://github.com/cool-japan/oxirs/releases)
//! [![docs.rs](https://docs.rs/oxirs-gql/badge.svg)](https://docs.rs/oxirs-gql)
//!
//! **Status**: Production Release (v0.3.3)
//! **Stability**: Public APIs are stable. Production-ready with comprehensive testing.
//!
//! GraphQL interface for RDF data with automatic schema generation from ontologies.
//! Enables modern GraphQL clients to query knowledge graphs with intuitive GraphQL syntax.
//!
//! ## Features
//!
//! - **Automatic Schema Generation** - Generate GraphQL schemas from RDF vocabularies
//! - **GraphQL to SPARQL** - Transparent translation of GraphQL queries to SPARQL
//! - **Type Mapping** - Map RDF classes to GraphQL types
//! - **Query Support** - Full GraphQL query capabilities
//! - **Subscriptions** - WebSocket-based subscriptions (experimental)
//!
//! ## See Also
//!
//! - [`oxirs-core`](https://docs.rs/oxirs-core) - RDF data model
//! - [`oxirs-arq`](https://docs.rs/oxirs-arq) - SPARQL query engine

use anyhow::Result;
use oxirs_core::model::{
    BlankNode, GraphName, Literal as OxiLiteral, NamedNode, Quad, Subject, Term, Variable,
};
use oxirs_core::ConcreteStore;
use std::sync::{Arc, Mutex};
use std::time::Duration;

// Re-export QueryResults for other modules
pub use oxirs_core::query::QueryResults;

// Module declarations are below after the main code

/// Render a bound SPARQL [`Term`] as its plain (bracket-free) lexical
/// value: the bare IRI for a named node, the bare label for a blank node,
/// and the lexical form for a literal.
///
/// `Term`'s `Display`/`to_string()` instead produces N-Triples-style
/// serialization (`<http://example.org/x>` for a named node), which is the
/// right form to re-embed in another SPARQL query but the *wrong* form for
/// GraphQL scalar output or for round-tripping back through this crate's
/// own `escape_iri`-style helpers (which expect a bare IRI to wrap in
/// `<...>` themselves).
fn term_to_plain_string(term: &Term) -> String {
    match term {
        Term::NamedNode(node) => node.as_str().to_string(),
        Term::BlankNode(node) => format!("_:{}", node.as_str()),
        Term::Literal(literal) => literal.value().to_string(),
        Term::Variable(var) => var.as_str().to_string(),
        Term::QuotedTriple(_) => term.to_string(),
    }
}

/// RDF store wrapper for GraphQL integration
pub struct RdfStore {
    store: Arc<Mutex<ConcreteStore>>,
}

impl std::fmt::Debug for RdfStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RdfStore")
            .field("store", &"Store { ... }")
            .finish()
    }
}

impl RdfStore {
    pub fn new() -> Result<Self> {
        Ok(Self {
            store: Arc::new(Mutex::new(ConcreteStore::new()?)),
        })
    }

    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        Ok(Self {
            store: Arc::new(Mutex::new(ConcreteStore::open(path)?)),
        })
    }

    /// Execute a SPARQL query and return results
    pub fn query(&self, query: &str) -> Result<QueryResults> {
        use oxirs_core::query::{QueryEngine, QueryResult};

        let store = self
            .store
            .lock()
            .map_err(|e| anyhow::anyhow!("Mutex lock error: {}", e))?;
        let engine = QueryEngine::new();
        let result = engine
            .query(query, &*store)
            .map_err(|e| anyhow::anyhow!("SPARQL query error: {}", e))?;

        match result {
            QueryResult::Select {
                variables: _,
                bindings,
            } => {
                let mut solutions = Vec::new();
                for binding in bindings {
                    let mut solution = oxirs_core::query::Solution::new();
                    for (var_name, term) in binding {
                        if let Ok(var) = oxirs_core::model::Variable::new(&var_name) {
                            solution.bind(var, term);
                        }
                    }
                    solutions.push(solution);
                }
                Ok(QueryResults::Solutions(solutions))
            }
            QueryResult::Ask(result) => Ok(QueryResults::Boolean(result)),
            QueryResult::Construct(triples) => {
                // Return triples directly (not quads)
                Ok(QueryResults::Graph(triples))
            }
        }
    }

    /// Get count of triples in the store
    pub fn triple_count(&self) -> Result<usize> {
        let query = "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }";
        if let QueryResults::Solutions(solutions) = self.query(query)? {
            if let Some(solution) = solutions.first() {
                let count_var = Variable::new("count")
                    .map_err(|e| anyhow::anyhow!("Failed to create count variable: {}", e))?;
                if let Some(Term::Literal(lit)) = solution.get(&count_var) {
                    let count = lit.value().parse::<usize>().map_err(|e| {
                        anyhow::anyhow!("Failed to parse count value '{}': {}", lit.value(), e)
                    })?;
                    return Ok(count);
                }
            }
        }
        Ok(0)
    }

    /// Get subjects with optional limit
    ///
    /// Note: the `LIMIT`/pagination is applied in Rust, *not* appended to
    /// the SPARQL text -- the query engine's parser only accepts a bare
    /// `WHERE { ... }` graph pattern with nothing trailing after the
    /// closing brace, so `... WHERE { ... } LIMIT n` fails to parse.
    /// Deduplication (`DISTINCT`) is likewise applied in Rust: the query
    /// engine parses the `SELECT`/`WHERE` boundary only and does not act on
    /// `DISTINCT` itself.
    pub fn get_subjects(&self, limit: Option<usize>) -> Result<Vec<String>> {
        let query = "SELECT DISTINCT ?s WHERE { ?s ?p ?o }";
        let mut subjects = Vec::new();
        let mut seen = std::collections::HashSet::new();

        let subject_var = Variable::new("s")
            .map_err(|e| anyhow::anyhow!("Failed to create subject variable: {}", e))?;

        if let QueryResults::Solutions(solutions) = self.query(query)? {
            for solution in &solutions {
                if let Some(subject) = solution.get(&subject_var) {
                    let value = term_to_plain_string(subject);
                    if seen.insert(value.clone()) {
                        subjects.push(value);
                    }
                }
            }
        }

        if let Some(limit) = limit {
            subjects.truncate(limit);
        }

        Ok(subjects)
    }

    /// Get predicates with optional limit (see the notes on
    /// [`Self::get_subjects`] about why `LIMIT`/deduplication are applied
    /// in Rust).
    pub fn get_predicates(&self, limit: Option<usize>) -> Result<Vec<String>> {
        let query = "SELECT DISTINCT ?p WHERE { ?s ?p ?o }";
        let mut predicates = Vec::new();
        let mut seen = std::collections::HashSet::new();

        let predicate_var = Variable::new("p")
            .map_err(|e| anyhow::anyhow!("Failed to create predicate variable: {}", e))?;

        if let QueryResults::Solutions(solutions) = self.query(query)? {
            for solution in &solutions {
                if let Some(predicate) = solution.get(&predicate_var) {
                    let value = term_to_plain_string(predicate);
                    if seen.insert(value.clone()) {
                        predicates.push(value);
                    }
                }
            }
        }

        if let Some(limit) = limit {
            predicates.truncate(limit);
        }

        Ok(predicates)
    }

    /// Get objects with optional limit (see the notes on
    /// [`Self::get_subjects`] about why `LIMIT`/deduplication are applied
    /// in Rust).
    pub fn get_objects(&self, limit: Option<usize>) -> Result<Vec<(String, String)>> {
        let query = "SELECT DISTINCT ?o WHERE { ?s ?p ?o }";
        let mut objects = Vec::new();
        let mut seen = std::collections::HashSet::new();

        let object_var = Variable::new("o")
            .map_err(|e| anyhow::anyhow!("Failed to create object variable: {}", e))?;

        if let QueryResults::Solutions(solutions) = self.query(query)? {
            for solution in &solutions {
                if let Some(object) = solution.get(&object_var) {
                    let object_type = match object {
                        Term::NamedNode(_) => "IRI".to_string(),
                        Term::BlankNode(_) => "BlankNode".to_string(),
                        Term::Literal(_) => "Literal".to_string(),
                        Term::Variable(_) => "Variable".to_string(),
                        Term::QuotedTriple(_) => "QuotedTriple".to_string(),
                    };
                    let entry = (term_to_plain_string(object), object_type);
                    if seen.insert(entry.clone()) {
                        objects.push(entry);
                    }
                }
            }
        }

        if let Some(limit) = limit {
            objects.truncate(limit);
        }

        Ok(objects)
    }

    /// Insert a triple into the store
    pub fn insert_triple(&mut self, subject: &str, predicate: &str, object: &str) -> Result<()> {
        // Parse terms
        let subject = if let Some(stripped) = subject.strip_prefix("_:") {
            Subject::BlankNode(BlankNode::new(stripped)?)
        } else {
            Subject::NamedNode(NamedNode::new(subject)?)
        };

        let predicate = NamedNode::new(predicate)?;

        let object = if object.starts_with("\"") && object.ends_with("\"") {
            // It's a literal
            let literal_value = &object[1..object.len() - 1];
            Term::Literal(OxiLiteral::new_simple_literal(literal_value))
        } else if let Some(stripped) = object.strip_prefix("_:") {
            // It's a blank node
            Term::BlankNode(BlankNode::new(stripped)?)
        } else {
            // It's a named node
            Term::NamedNode(NamedNode::new(object)?)
        };

        let quad = Quad::new(subject, predicate, object, GraphName::DefaultGraph);
        let store = self
            .store
            .lock()
            .map_err(|e| anyhow::anyhow!("Mutex lock error: {}", e))?;
        store.insert_quad(quad)?;
        Ok(())
    }

    /// Insert a quad into the store
    pub fn insert(&self, quad: &oxirs_core::model::Quad) -> Result<()> {
        let store = self
            .store
            .lock()
            .map_err(|e| anyhow::anyhow!("Mutex lock error: {}", e))?;
        store.insert_quad(quad.clone())?;
        Ok(())
    }

    /// Load data from a file
    pub fn load_file<P: AsRef<std::path::Path>>(&mut self, path: P, format: &str) -> Result<()> {
        use oxirs_core::parser::{Parser, RdfFormat};
        use std::fs;

        let format = match format.to_lowercase().as_str() {
            "turtle" | "ttl" => RdfFormat::Turtle,
            "ntriples" | "nt" => RdfFormat::NTriples,
            "rdfxml" | "rdf" => RdfFormat::RdfXml,
            "jsonld" | "json" => RdfFormat::JsonLd,
            _ => return Err(anyhow::anyhow!("Unsupported format: {}", format)),
        };

        // Read file content
        let content = fs::read_to_string(path)?;

        // Parse content to quads
        let parser = Parser::new(format);
        let quads = parser.parse_str_to_quads(&content)?;

        // Insert quads into store
        let store = self
            .store
            .lock()
            .map_err(|e| anyhow::anyhow!("Mutex lock error: {}", e))?;
        for quad in quads {
            store.insert_quad(quad)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod rdf_store_query_tests {
    use super::*;

    fn build_store() -> RdfStore {
        let mut store = RdfStore::new().expect("failed to create test store");
        store
            .insert_triple(
                "http://example.org/alice",
                "http://xmlns.com/foaf/0.1/name",
                "\"Alice\"",
            )
            .expect("insert triple should succeed");
        store
            .insert_triple(
                "http://example.org/bob",
                "http://xmlns.com/foaf/0.1/name",
                "\"Bob\"",
            )
            .expect("insert triple should succeed");
        store
    }

    /// Regression test: `get_subjects`/`get_predicates`/`get_objects` used
    /// to return `Term::to_string()` (N-Triples-style, e.g.
    /// `"<http://example.org/alice>"`) instead of the bare IRI/literal
    /// value, so every caller comparing against a plain IRI string (as the
    /// `subjects`/`predicates`/`objects` GraphQL fields' consumers do)
    /// would never match.
    #[test]
    fn test_get_subjects_returns_bare_iris_not_bracketed() {
        let store = build_store();
        let subjects = store
            .get_subjects(None)
            .expect("get_subjects should succeed");
        let subjects: std::collections::HashSet<String> = subjects.into_iter().collect();

        assert!(subjects.contains("http://example.org/alice"));
        assert!(subjects.contains("http://example.org/bob"));
        assert!(
            !subjects.iter().any(|s| s.starts_with('<')),
            "subjects must not be bracket-wrapped: {subjects:?}"
        );
    }

    #[test]
    fn test_get_predicates_returns_bare_iris_not_bracketed() {
        let store = build_store();
        let predicates = store
            .get_predicates(None)
            .expect("get_predicates should succeed");

        assert_eq!(
            predicates,
            vec!["http://xmlns.com/foaf/0.1/name".to_string()]
        );
    }

    #[test]
    fn test_get_objects_returns_plain_literal_values() {
        let store = build_store();
        let objects = store.get_objects(None).expect("get_objects should succeed");
        let values: std::collections::HashSet<String> =
            objects.into_iter().map(|(value, _)| value).collect();

        assert!(values.contains("Alice"));
        assert!(values.contains("Bob"));
        assert!(
            !values.iter().any(|v| v.starts_with('"')),
            "literal values must not retain quote delimiters: {values:?}"
        );
    }

    /// Regression test: the query engine's SPARQL parser rejects
    /// `WHERE { ... } LIMIT n` (it requires the graph pattern's closing
    /// brace to be the last thing in the query), so `limit` must be
    /// enforced in Rust rather than appended to the SPARQL text.
    #[test]
    fn test_get_subjects_respects_limit_without_breaking_the_query() {
        let store = build_store();
        let subjects = store
            .get_subjects(Some(1))
            .expect("get_subjects with a limit should still succeed");
        assert_eq!(subjects.len(), 1);
    }
}

/// Mock store for testing GraphQL functionality
#[derive(Debug)]
pub struct MockStore;

impl MockStore {
    pub fn new() -> Result<Self> {
        Ok(Self)
    }

    pub fn open(_path: String) -> Result<Self> {
        Ok(Self)
    }
}

// Individual modules
pub mod aggregation;
pub mod ai;
pub mod api_explorer;
pub mod ast;
pub mod auto_caching_strategies;
pub mod auto_schema_resolver;
pub mod custom_directives;
pub mod custom_type_mappings;
pub mod directive_registry;
pub mod enhanced_errors;
pub mod execution;
pub mod federation;
pub mod file_upload;
pub mod graphiql_integration;
pub mod historical_cost_estimator;
pub mod horizontal_scaling;
pub mod hybrid_optimizer;
pub mod intelligent_federation_gateway;
pub mod intelligent_query_cache;
pub mod introspection;
pub mod live_queries;
pub mod mapping;
pub mod ml_optimizer;
pub mod optimizer;
pub mod owl_enhanced_schema;
pub mod pagination_filtering;
pub mod parallel_field_resolver;
pub mod parser;
pub mod persisted_queries;
pub mod playground_integration;
pub mod quantum_optimizer;
pub mod query_builder;
pub mod query_debugger;
pub mod query_prefetcher;
pub mod rate_limiting;
pub mod rdf_scalars;
pub mod request_deduplication;
pub mod resolvers;
pub mod schema;
pub mod schema_cache;
pub mod schema_docs_generator;
pub mod schema_generator;
pub mod schema_loader;
pub mod schema_sdl;
pub mod schema_types;
pub mod server;
pub mod sse_subscriptions;
pub mod subscriptions;
pub mod types;
pub mod validation;
pub mod validation_spec;
pub mod zero_trust_security;

// v0.2.0 Operational Enhancements
pub mod api_versioning;
pub mod blue_green_deployment;
pub mod canary_release;
pub mod circuit_breaker;
pub mod graphql_mesh;
pub mod multi_region;
pub mod performance_insights;
pub mod schema_changelog;
pub mod visual_schema_designer;

// v0.3.0 Security & Integration Features
pub mod content_security_policy;
pub mod edge_caching;
pub mod query_sanitization;
pub mod response_streaming;
pub mod webhook_support;

// v0.4.0 Advanced Observability & Protocol Features
pub mod cost_based_optimizer;
pub mod incremental_execution;
pub mod query_batching;
pub mod query_plan_visualizer;
pub mod query_result_streaming;

// v0.5.0 Advanced Observability & Monitoring
pub mod graphql_span_attributes;
pub mod performance_anomaly_detector;
pub mod performance_heatmap;
pub mod query_pattern_analyzer;
pub mod trace_correlation;
pub mod trace_sampling;
pub mod trace_visualization;
pub mod tracing_exporters;

// Advanced performance modules
pub mod advanced_cache;
pub mod advanced_security_system;
pub mod ai_query_predictor;
pub mod async_streaming;
pub mod benchmarking;
pub mod dataloader;
pub mod neuromorphic_query_processor;
pub mod performance;
pub mod predictive_analytics;
pub mod quantum_real_time_analytics;
pub mod system_monitor;

// Ultra-modern enterprise modules (July 5, 2025 enhancements)
pub mod advanced_query_planner;
pub mod advanced_subscriptions;
pub mod ai_orchestration_engine;
pub mod observability;

// Production-ready features (November 2025)
pub mod production;

// v0.6.0 Enhanced Subscriptions, Multi-tenancy, and Query Caching
pub mod cache;
pub mod multitenancy;
pub mod subscription;
// v1.1.0 Enhanced tenant module with schema registry, query filter, rate limiter
pub mod tenant;
// v1.1.0 Apollo Federation v2 Subgraph Support
pub mod federation_v2;

// v1.2.0 GraphQL Schema Version Registry
pub mod schema_registry;

// v1.1.0 Relay Cursor-based Pagination
pub mod cursor_pagination;

// v1.2.0 GraphQL subscription lifecycle management
pub mod subscription_manager;

// v1.2.0 DataLoader / batch resolver for GraphQL N+1 resolution
pub mod batch_resolver;

// v1.5.0 GraphQL __schema / __type introspection engine
pub mod type_introspection;

// Organized module groups
pub mod core;
pub mod distributed_cache;
pub mod docs;
pub mod dynamic_query_planner;
pub mod features;
pub mod networking;
pub mod rdf;

// v1.6.0 Field-level resolver pipeline with middleware
pub mod field_resolver;

// v1.7.0 GraphQL query complexity analysis and limiting
pub mod query_complexity;

// v1.8.0 GraphQL error formatting, classification, and aggregation
pub mod error_formatter;

// v1.9.0 Custom GraphQL directive processing engine
pub mod directive_processor;

// v1.10.0 GraphQL pagination handler (Relay cursor spec)
pub mod pagination_handler;

// v1.11.0 GraphQL field-level validation rules engine
pub mod field_validator;

// v1.1.0 round 16 GraphQL enum type resolution and coercion
pub mod enum_resolver;

// v1.1.0 round 15 GraphQL argument type coercion and validation
pub mod argument_coercer;

// Juniper-based implementation with proper RDF integration (enabled)
pub mod juniper_schema;
pub mod juniper_server; // Complex Hyper v1 version - API issues fixed
pub mod simple_juniper_server; // Simplified version

// Juniper integration - comprehensive RDF GraphQL support
pub use juniper_schema::{create_schema, GraphQLContext, Schema as JuniperSchema};
pub use simple_juniper_server::{
    start_graphql_server, start_graphql_server_with_config, GraphQLServerBuilder,
    GraphQLServerConfig, JuniperGraphQLServer,
};

// Intelligent query caching
pub use intelligent_query_cache::{
    IntelligentCacheConfig, IntelligentQueryCache, QueryPattern, QueryUsageStats,
};

// Advanced Juniper server with full Hyper v1 support
pub use juniper_server::{
    start_graphql_server as start_advanced_graphql_server,
    start_graphql_server_with_config as start_advanced_graphql_server_with_config,
    GraphQLServerBuilder as AdvancedGraphQLServerBuilder,
    GraphQLServerConfig as AdvancedGraphQLServerConfig,
    JuniperGraphQLServer as AdvancedJuniperGraphQLServer,
};

#[cfg(test)]
mod tests;

/// GraphQL server configuration
#[derive(Debug, Clone)]
pub struct GraphQLConfig {
    pub enable_introspection: bool,
    pub enable_playground: bool,
    pub max_query_depth: Option<usize>,
    pub max_query_complexity: Option<usize>,
    pub validation_config: validation::ValidationConfig,
    pub enable_query_validation: bool,
    pub distributed_cache_config: Option<distributed_cache::CacheConfig>,
    /// Whether the raw, unauthenticated SPARQL passthrough field
    /// (`sparql(query: String!): String`) is exposed in the schema at all.
    /// **Disabled by default**: it bypasses every GraphQL-level
    /// depth/complexity limit and lets any client run arbitrary SPARQL
    /// against the store, with no authentication or authorization of its
    /// own. Only enable this on a deployment where the GraphQL endpoint
    /// itself is already access-controlled.
    pub enable_sparql_field: bool,
    /// Wall-clock timeout applied to each raw SPARQL passthrough query,
    /// only relevant when `enable_sparql_field` is `true`.
    pub sparql_query_timeout: Duration,
    /// Whether to attempt automatic GraphQL schema generation from the
    /// RDF store's vocabulary (`rdfs:Class`/`owl:Class` definitions) in
    /// addition to the built-in `hello`/`version`/`triples`/`subjects`/
    /// `predicates`/`objects` fields. Falls back to the built-in fields
    /// alone when the store has no discoverable vocabulary, or when
    /// generation fails.
    pub auto_generate_schema: bool,
    /// Maximum accepted HTTP request body size, in bytes.
    pub max_request_body_size: usize,
    /// Idle/read timeout applied to each socket read while parsing an
    /// HTTP request (Slowloris protection).
    pub request_read_timeout: Duration,
    /// Maximum number of concurrent HTTP connections served.
    pub max_connections: usize,
    /// Per-client-IP rate limiting configuration for incoming HTTP
    /// requests. **Disabled by default** (`None`): set this to enforce a
    /// request budget (token bucket / sliding window / fixed window /
    /// adaptive-under-load) before any GraphQL parsing/execution happens.
    /// See [`rate_limiting::RateLimiter`].
    pub rate_limit_config: Option<rate_limiting::RateLimitConfig>,
}

impl Default for GraphQLConfig {
    fn default() -> Self {
        Self {
            enable_introspection: true,
            enable_playground: true,
            max_query_depth: Some(10),
            max_query_complexity: Some(1000),
            validation_config: validation::ValidationConfig::default(),
            enable_query_validation: true,
            distributed_cache_config: None, // Disabled by default
            enable_sparql_field: false,
            sparql_query_timeout: Duration::from_secs(30),
            auto_generate_schema: true,
            max_request_body_size: 10 * 1024 * 1024,
            request_read_timeout: Duration::from_secs(30),
            max_connections: 1024,
            rate_limit_config: None,
        }
    }
}

impl GraphQLConfig {
    /// Merge the ergonomic top-level `max_query_depth`/`max_query_complexity`
    /// shorthands into a full [`validation::ValidationConfig`], overriding
    /// whatever `validation_config.max_depth`/`max_complexity` already
    /// contain. This is the single place those two fields take effect --
    /// previously they were accepted by `with_config` but never read
    /// anywhere, so setting them had no effect on query validation.
    fn effective_validation_config(&self) -> validation::ValidationConfig {
        let mut validation_config = self.validation_config.clone();
        if let Some(max_depth) = self.max_query_depth {
            validation_config.max_depth = max_depth;
        }
        if let Some(max_complexity) = self.max_query_complexity {
            validation_config.max_complexity = max_complexity;
        }
        validation_config
    }
}

#[cfg(test)]
mod graphql_config_tests {
    use super::*;

    /// Regression test: `max_query_depth`/`max_query_complexity` must
    /// actually override the validation config used to build the
    /// `QueryValidator`, not just sit on `GraphQLConfig` unread.
    #[test]
    fn test_max_query_depth_and_complexity_override_validation_config() {
        let config = GraphQLConfig {
            max_query_depth: Some(3),
            max_query_complexity: Some(42),
            ..Default::default()
        };

        let effective = config.effective_validation_config();
        assert_eq!(effective.max_depth, 3);
        assert_eq!(effective.max_complexity, 42);
    }

    /// When left `None`, the base `validation_config`'s own limits must be
    /// preserved rather than being reset to some other default.
    #[test]
    fn test_none_depth_and_complexity_preserve_validation_config_defaults() {
        let base_validation_config = validation::ValidationConfig {
            max_depth: 7,
            max_complexity: 77,
            ..Default::default()
        };

        let config = GraphQLConfig {
            max_query_depth: None,
            max_query_complexity: None,
            validation_config: base_validation_config,
            ..Default::default()
        };

        let effective = config.effective_validation_config();
        assert_eq!(effective.max_depth, 7);
        assert_eq!(effective.max_complexity, 77);
    }

    /// Regression test: the fully-implemented `rate_limiting` module used
    /// to be unreachable from `GraphQLServer` entirely; `rate_limit_config`
    /// must default to disabled (no surprise behavior change for existing
    /// deployments) while remaining settable.
    #[test]
    fn test_rate_limit_config_defaults_to_disabled_but_is_settable() {
        assert!(GraphQLConfig::default().rate_limit_config.is_none());

        let config = GraphQLConfig {
            rate_limit_config: Some(rate_limiting::RateLimitConfig {
                max_requests: 5,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(
            config
                .rate_limit_config
                .expect("should be set")
                .max_requests,
            5
        );
    }
}

/// Main GraphQL server
pub struct GraphQLServer {
    config: GraphQLConfig,
    store: Arc<RdfStore>,
    cache: Option<Arc<distributed_cache::GraphQLQueryCache>>,
}

impl GraphQLServer {
    pub fn new(store: Arc<RdfStore>) -> Self {
        Self {
            config: GraphQLConfig::default(),
            store,
            cache: None,
        }
    }

    pub fn new_with_mock(_store: Arc<MockStore>) -> Result<Self> {
        // For backward compatibility during transition
        let rdf_store = Arc::new(
            RdfStore::new()
                .map_err(|e| anyhow::anyhow!("Failed to create RDF store for mock: {}", e))?,
        );
        Ok(Self {
            config: GraphQLConfig::default(),
            store: rdf_store,
            cache: None,
        })
    }

    pub fn with_config(mut self, config: GraphQLConfig) -> Self {
        self.config = config;
        self
    }

    /// Enable distributed caching
    pub async fn with_distributed_cache(
        mut self,
        cache_config: distributed_cache::CacheConfig,
    ) -> Result<Self> {
        let cache = Arc::new(distributed_cache::GraphQLQueryCache::new(cache_config).await?);
        self.cache = Some(cache);
        Ok(self)
    }

    /// Get cache statistics if caching is enabled
    pub async fn get_cache_stats(&self) -> Option<distributed_cache::CacheStats> {
        if let Some(cache) = &self.cache {
            cache.get_stats().await.ok()
        } else {
            None
        }
    }

    pub async fn start(&self, addr: &str) -> Result<()> {
        tracing::info!("Starting GraphQL server on {}", addr);

        // `GraphQLConfig::max_query_depth`/`max_query_complexity` are the
        // ergonomic top-level knobs; thread them into the full
        // `ValidationConfig` before constructing the validator (previously
        // these fields were parsed by `with_config` but never read).
        let validation_config = self.config.effective_validation_config();

        // Create a basic schema with Query type
        let mut schema = types::Schema::new();

        // Add a Query type with more fields
        let mut query_type = types::ObjectType::new("Query".to_string())
            .with_description("The root query type for RDF data access".to_string())
            .with_field(
                "hello".to_string(),
                types::FieldType::new(
                    "hello".to_string(),
                    types::GraphQLType::Scalar(types::BuiltinScalars::string()),
                )
                .with_description("A simple greeting message".to_string()),
            )
            .with_field(
                "version".to_string(),
                types::FieldType::new(
                    "version".to_string(),
                    types::GraphQLType::Scalar(types::BuiltinScalars::string()),
                )
                .with_description("OxiRS GraphQL version".to_string()),
            )
            .with_field(
                "triples".to_string(),
                types::FieldType::new(
                    "triples".to_string(),
                    types::GraphQLType::Scalar(types::BuiltinScalars::int()),
                )
                .with_description("Count of triples in the store".to_string()),
            )
            .with_field(
                "subjects".to_string(),
                types::FieldType::new(
                    "subjects".to_string(),
                    types::GraphQLType::List(Box::new(types::GraphQLType::Scalar(
                        types::BuiltinScalars::string(),
                    ))),
                )
                .with_description("List of subject IRIs".to_string())
                .with_argument(
                    "limit".to_string(),
                    types::ArgumentType::new(
                        "limit".to_string(),
                        types::GraphQLType::Scalar(types::BuiltinScalars::int()),
                    )
                    .with_default_value(ast::Value::IntValue(10))
                    .with_description("Maximum number of subjects to return".to_string()),
                ),
            )
            .with_field(
                "predicates".to_string(),
                types::FieldType::new(
                    "predicates".to_string(),
                    types::GraphQLType::List(Box::new(types::GraphQLType::Scalar(
                        types::BuiltinScalars::string(),
                    ))),
                )
                .with_description("List of predicate IRIs".to_string())
                .with_argument(
                    "limit".to_string(),
                    types::ArgumentType::new(
                        "limit".to_string(),
                        types::GraphQLType::Scalar(types::BuiltinScalars::int()),
                    )
                    .with_default_value(ast::Value::IntValue(10))
                    .with_description("Maximum number of predicates to return".to_string()),
                ),
            )
            .with_field(
                "objects".to_string(),
                types::FieldType::new(
                    "objects".to_string(),
                    types::GraphQLType::List(Box::new(types::GraphQLType::Scalar(
                        types::BuiltinScalars::string(),
                    ))),
                )
                .with_description("List of objects".to_string())
                .with_argument(
                    "limit".to_string(),
                    types::ArgumentType::new(
                        "limit".to_string(),
                        types::GraphQLType::Scalar(types::BuiltinScalars::int()),
                    )
                    .with_default_value(ast::Value::IntValue(10))
                    .with_description("Maximum number of objects to return".to_string()),
                ),
            );

        // The raw SPARQL passthrough field bypasses all GraphQL-level
        // depth/complexity limits and lets any client run arbitrary SPARQL
        // against the store, with no authentication or authorization of
        // its own, so it is opt-in only (see
        // `GraphQLConfig::enable_sparql_field`, default `false`).
        if self.config.enable_sparql_field {
            query_type = query_type.with_field(
                "sparql".to_string(),
                types::FieldType::new(
                    "sparql".to_string(),
                    types::GraphQLType::Scalar(types::BuiltinScalars::string()),
                )
                .with_description("Execute a raw SPARQL query".to_string())
                .with_argument(
                    "query".to_string(),
                    types::ArgumentType::new(
                        "query".to_string(),
                        types::GraphQLType::NonNull(Box::new(types::GraphQLType::Scalar(
                            types::BuiltinScalars::string(),
                        ))),
                    )
                    .with_description("The SPARQL query to execute".to_string()),
                ),
            );
        }

        // Add introspection fields if enabled. `__schema`/`__type` are
        // modeled as real `Object` types (registered via
        // `introspection::introspection_meta_types`) rather than opaque
        // `Scalar`s, so that `QueryExecutor::complete_value` honors client
        // selection sets on them (see `QueryExecutor::project_introspection_value`).
        if self.config.enable_introspection {
            for meta_type in introspection::introspection_meta_types() {
                schema.add_type(meta_type);
            }

            query_type = query_type
                .with_field(
                    "__schema".to_string(),
                    types::FieldType::new(
                        "__schema".to_string(),
                        types::GraphQLType::NonNull(Box::new(types::GraphQLType::Object(
                            types::ObjectType::new("__Schema".to_string()),
                        ))),
                    )
                    .with_description("Access the current type schema of this server".to_string()),
                )
                .with_field(
                    "__type".to_string(),
                    types::FieldType::new(
                        "__type".to_string(),
                        types::GraphQLType::Object(types::ObjectType::new("__Type".to_string())),
                    )
                    .with_description("Request the type information of a single type".to_string())
                    .with_argument(
                        "name".to_string(),
                        types::ArgumentType::new(
                            "name".to_string(),
                            types::GraphQLType::NonNull(Box::new(types::GraphQLType::Scalar(
                                types::BuiltinScalars::string(),
                            ))),
                        )
                        .with_description("The name of the type to introspect".to_string()),
                    ),
                );
        }

        // Attempt to auto-generate additional object types and root query
        // fields from the store's RDF vocabulary (`rdfs:Class`/`owl:Class`
        // declarations), per `GraphQLConfig::auto_generate_schema` (default
        // `true`). This is a genuine best-effort enhancement: any failure
        // to extract a vocabulary or generate a schema from it is logged
        // and simply falls back to the hardcoded fields added above, which
        // remain present unconditionally.
        let mut auto_schema_resolver: Option<Arc<auto_schema_resolver::AutoSchemaResolver>> = None;
        let mut generated_type_names: Vec<String> = Vec::new();

        if self.config.auto_generate_schema {
            match schema_generator::SchemaGenerator::new()
                .extract_vocabulary_from_store(&self.store)
            {
                Ok(vocabulary) if !vocabulary.classes.is_empty() => {
                    let gen_config = schema_types::SchemaGenerationConfig {
                        // The top-level `sparql` field above already covers
                        // raw SPARQL passthrough when opted into; don't add
                        // a second copy from the generator.
                        enable_sparql_field: false,
                        ..Default::default()
                    };
                    let generator = schema_generator::SchemaGenerator::new()
                        .with_config(gen_config)
                        .with_vocabulary(vocabulary.clone());

                    match generator.generate_schema() {
                        Ok(generated) => {
                            let generated_query_fields = match generated.get_type("Query") {
                                Some(types::GraphQLType::Object(obj)) => obj.fields.clone(),
                                _ => std::collections::HashMap::new(),
                            };

                            for (type_name, generated_type) in generated.types {
                                if matches!(
                                    type_name.as_str(),
                                    "Query" | "Mutation" | "Subscription"
                                ) {
                                    continue;
                                }
                                schema.add_type(generated_type);
                            }

                            for (field_name, field_def) in generated_query_fields {
                                query_type.fields.entry(field_name).or_insert(field_def);
                            }

                            let resolver = auto_schema_resolver::AutoSchemaResolver::new(
                                Arc::clone(&self.store),
                                vocabulary,
                            );
                            generated_type_names = resolver.generated_type_names();
                            auto_schema_resolver = Some(Arc::new(resolver));
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Auto schema generation failed, falling back to built-in fields only: {}",
                                e
                            );
                        }
                    }
                }
                Ok(_) => {
                    tracing::debug!(
                        "Store has no discoverable rdfs:Class/owl:Class vocabulary; using built-in GraphQL fields only"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to extract RDF vocabulary from store, falling back to built-in fields only: {}",
                        e
                    );
                }
            }
        }

        schema.add_type(types::GraphQLType::Object(query_type));
        schema.set_query_type("Query".to_string());

        // Create the server with resolvers
        let schema_clone = schema.clone();
        let mut server = server::Server::new(schema.clone())
            .with_playground(self.config.enable_playground)
            .with_introspection(self.config.enable_introspection)
            .with_max_body_size(self.config.max_request_body_size)
            .with_read_timeout(self.config.request_read_timeout)
            .with_max_connections(self.config.max_connections);

        if let Some(ref rate_limit_config) = self.config.rate_limit_config {
            server = server.with_rate_limiting(rate_limit_config.clone());
        }

        // Configure validation if enabled
        if self.config.enable_query_validation {
            server = server.with_validation(validation_config, schema_clone.clone());
        }

        // Any RDF-class object type produced by auto schema generation must
        // be resolved "eagerly" (see `execution::QueryExecutor::add_eager_type`)
        // since `AutoSchemaResolver` returns one fully materialized value
        // tree per field rather than participating in per-type resolver
        // dispatch, which has no way to distinguish between items of a list.
        for type_name in &generated_type_names {
            server.add_eager_object_type(type_name.clone());
        }

        // Set up the root Query resolver: hardcoded RDF fields (and the
        // opt-in raw SPARQL passthrough), GraphQL introspection, and -- when
        // present -- fields generated from the store's RDF vocabulary.
        let query_resolvers = resolvers::QueryResolvers::new_with_sparql_config(
            Arc::clone(&self.store),
            self.config.enable_sparql_field,
            self.config.sparql_query_timeout,
        );
        let introspection_resolver = Arc::new(introspection::IntrospectionResolver::new(Arc::new(
            schema_clone,
        )));
        let root_resolver = QueryRootResolver {
            rdf_resolver: query_resolvers.rdf_resolver(),
            introspection_resolver,
            auto_schema_resolver,
        };
        server.add_resolver("Query".to_string(), Arc::new(root_resolver));

        // Parse the address
        let socket_addr: std::net::SocketAddr = addr
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid address '{}': {}", addr, e))?;

        server.start(socket_addr).await
    }
}

/// Root `Query`-type field resolver that dispatches each requested field to
/// the appropriate underlying resolver: the hardcoded RDF/SPARQL fields
/// (see [`resolvers::RdfResolver`]), GraphQL introspection (`__schema`/
/// `__type`, see [`introspection::IntrospectionResolver`]), and -- when the
/// store's vocabulary yielded at least one RDF class -- the fields
/// generated by [`schema_generator::SchemaGenerator`] (see
/// [`auto_schema_resolver::AutoSchemaResolver`]).
struct QueryRootResolver {
    rdf_resolver: Arc<resolvers::RdfResolver>,
    introspection_resolver: Arc<introspection::IntrospectionResolver>,
    auto_schema_resolver: Option<Arc<auto_schema_resolver::AutoSchemaResolver>>,
}

#[async_trait::async_trait]
impl execution::FieldResolver for QueryRootResolver {
    async fn resolve_field(
        &self,
        field_name: &str,
        args: &std::collections::HashMap<String, ast::Value>,
        context: &execution::ExecutionContext,
    ) -> Result<ast::Value> {
        match field_name {
            "__schema" | "__type" => {
                self.introspection_resolver
                    .resolve_field(field_name, args, context)
                    .await
            }
            _ => {
                if let Some(ref auto_resolver) = self.auto_schema_resolver {
                    if auto_resolver.handles(field_name) {
                        return auto_resolver.resolve_field(field_name, args, context).await;
                    }
                }
                self.rdf_resolver
                    .resolve_field(field_name, args, context)
                    .await
            }
        }
    }
}

// Comprehensive module declarations moved to top of file to avoid duplicates
