//! GraphQL resolvers for RDF data
//!
//! This module provides resolvers that translate GraphQL field selections
//! to SPARQL queries against RDF datasets with advanced performance optimizations.

use crate::advanced_cache::{AdvancedCache, AdvancedCacheConfig};
use crate::ast::Value;
use crate::dataloader::{DataLoader, DataLoaderFactory};
use crate::execution::{ExecutionContext, FieldResolver};
use crate::performance::PerformanceTracker;
use crate::RdfStore;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use oxirs_core::query::QueryResults;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Default timeout applied to raw SPARQL passthrough queries (the `sparql`
/// GraphQL field) when the resolver isn't given an explicit override.
const DEFAULT_SPARQL_QUERY_TIMEOUT: Duration = Duration::from_secs(30);

/// Enhanced RDF-based resolver with performance optimizations
pub struct RdfResolver {
    store: Arc<RdfStore>,
    subject_loader: Option<DataLoader<String, serde_json::Value>>,
    predicate_loader: Option<DataLoader<String, Vec<String>>>,
    cache: Option<Arc<AdvancedCache>>,
    performance_tracker: Option<Arc<PerformanceTracker>>,
    /// Whether the raw, unauthenticated SPARQL passthrough field
    /// (`sparql(query: String!): String`) is allowed to execute at all.
    /// Defaults to `false`: this field bypasses all GraphQL-level
    /// depth/complexity limits and lets a client run arbitrary SPARQL, so
    /// it must be explicitly opted into via
    /// `GraphQLConfig::enable_sparql_field`.
    enable_sparql_field: bool,
    /// Wall-clock timeout applied to each raw SPARQL passthrough query.
    sparql_query_timeout: Duration,
}

impl RdfResolver {
    pub fn new(store: Arc<RdfStore>) -> Self {
        Self {
            store,
            subject_loader: None,
            predicate_loader: None,
            cache: None,
            performance_tracker: None,
            enable_sparql_field: false,
            sparql_query_timeout: DEFAULT_SPARQL_QUERY_TIMEOUT,
        }
    }

    /// Create resolver with DataLoader optimization
    pub fn with_dataloader(store: Arc<RdfStore>) -> Self {
        let factory = DataLoaderFactory::new();
        let subject_loader = factory.create_subject_loader(Arc::clone(&store));
        let predicate_loader = factory.create_predicate_loader(Arc::clone(&store));

        Self {
            store,
            subject_loader: Some(subject_loader),
            predicate_loader: Some(predicate_loader),
            cache: None,
            performance_tracker: None,
            enable_sparql_field: false,
            sparql_query_timeout: DEFAULT_SPARQL_QUERY_TIMEOUT,
        }
    }

    /// Create resolver with full performance optimizations
    pub fn with_performance_optimizations(
        store: Arc<RdfStore>,
        cache_config: Option<AdvancedCacheConfig>,
        performance_tracker: Option<Arc<PerformanceTracker>>,
    ) -> Self {
        let factory = DataLoaderFactory::new();
        let subject_loader = factory.create_subject_loader(Arc::clone(&store));
        let predicate_loader = factory.create_predicate_loader(Arc::clone(&store));

        let cache = cache_config.map(|config| Arc::new(AdvancedCache::new(config)));

        Self {
            store,
            subject_loader: Some(subject_loader),
            predicate_loader: Some(predicate_loader),
            cache,
            performance_tracker,
            enable_sparql_field: false,
            sparql_query_timeout: DEFAULT_SPARQL_QUERY_TIMEOUT,
        }
    }

    /// Explicitly opt into (or out of) the raw SPARQL passthrough field.
    /// Disabled by default: see [`Self::enable_sparql_field`].
    pub fn with_sparql_enabled(mut self, enabled: bool) -> Self {
        self.enable_sparql_field = enabled;
        self
    }

    /// Set the wall-clock timeout applied to each raw SPARQL passthrough
    /// query executed via the `sparql` field.
    pub fn with_sparql_query_timeout(mut self, timeout: Duration) -> Self {
        self.sparql_query_timeout = timeout;
        self
    }
}

#[async_trait]
impl FieldResolver for RdfResolver {
    async fn resolve_field(
        &self,
        field_name: &str,
        args: &HashMap<String, Value>,
        context: &ExecutionContext,
    ) -> Result<Value> {
        let start_time = Instant::now();

        tracing::debug!(
            "Resolving field '{}' with args: {:?} in request {}",
            field_name,
            args,
            context.request_id
        );

        // Check cache first if available
        let cache_key = self.generate_cache_key(field_name, args, context);
        if let Some(ref cache) = self.cache {
            if let Some(cached_value) = cache.get(&cache_key).await {
                if let Ok(value) = serde_json::from_value::<Value>(cached_value) {
                    tracing::debug!(
                        "Cache hit for field '{}' in request {}",
                        field_name,
                        context.request_id
                    );
                    return Ok(value);
                }
            }
        }

        let result = match field_name {
            "hello" => {
                // Simple test resolver
                Ok(Value::StringValue("Hello from OxiRS GraphQL!".to_string()))
            }
            "version" => Ok(Value::StringValue(env!("CARGO_PKG_VERSION").to_string())),
            "triples" => {
                // Return count of triples in the store
                self.resolve_triples_count(args).await
            }
            "subjects" => {
                // Return list of subjects with DataLoader optimization
                self.resolve_subjects_optimized(args).await
            }
            "predicates" => {
                // Return list of predicates with DataLoader optimization
                self.resolve_predicates_optimized(args).await
            }
            "objects" => {
                // Return list of objects
                self.resolve_objects(args).await
            }
            "sparql" => {
                if !self.enable_sparql_field {
                    // The raw SPARQL passthrough bypasses all GraphQL-level
                    // depth/complexity limits and lets a client run
                    // arbitrary SPARQL against the store, so it is disabled
                    // by default. This check is defense-in-depth: callers
                    // should also avoid advertising the field in the schema
                    // at all when it's disabled (see GraphQLServer::start).
                    Err(anyhow!(
                        "The 'sparql' field is disabled by default; set GraphQLConfig::enable_sparql_field = true to opt in"
                    ))
                } else {
                    // Execute raw SPARQL query with caching and a bounded timeout
                    self.resolve_sparql_query_cached(args, &cache_key).await
                }
            }
            _ => {
                tracing::warn!("Unknown field '{}' requested", field_name);
                Ok(Value::NullValue)
            }
        };

        // Record performance metrics if tracker is available
        if let Some(ref tracker) = self.performance_tracker {
            let duration = start_time.elapsed();
            tracker.record_field_resolution(field_name, duration, result.is_err());
        }

        // Cache successful results
        if let (Ok(value), Some(cache)) = (&result, &self.cache) {
            if let Ok(json_value) = serde_json::to_value(value) {
                // Create dependencies for cache invalidation
                let mut dependencies = HashSet::new();
                dependencies.insert("rdf_data".to_string());

                // Create tags for categorization
                let mut tags = HashSet::new();
                tags.insert(field_name.to_string());
                tags.insert("query_result".to_string());

                cache
                    .set(
                        cache_key,
                        json_value,
                        None, // Use default TTL
                        Some(self.calculate_field_complexity(field_name, args)),
                        Some(dependencies),
                        Some(tags),
                    )
                    .await;
            }
        }

        result
    }
}

impl RdfResolver {
    /// Generate cache key for field resolution
    fn generate_cache_key(
        &self,
        field_name: &str,
        args: &HashMap<String, Value>,
        context: &ExecutionContext,
    ) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        field_name.hash(&mut hasher);
        format!("{args:?}").hash(&mut hasher);
        context.request_id.hash(&mut hasher);
        format!("field_{}_{}", field_name, hasher.finish())
    }

    /// Calculate complexity score for caching decisions
    fn calculate_field_complexity(&self, field_name: &str, args: &HashMap<String, Value>) -> usize {
        let base_complexity = match field_name {
            "sparql" => 100,
            "subjects" | "predicates" | "objects" => 50,
            "triples" => 30,
            _ => 10,
        };

        // Add complexity based on arguments
        let arg_complexity = args.len() * 5;

        base_complexity + arg_complexity
    }

    async fn resolve_triples_count(&self, _args: &HashMap<String, Value>) -> Result<Value> {
        match self.store.triple_count() {
            Ok(count) => Ok(Value::IntValue(count as i64)),
            Err(err) => {
                tracing::error!("Failed to count triples: {}", err);
                Ok(Value::IntValue(0))
            }
        }
    }

    /// Optimized subjects resolver using DataLoader
    ///
    /// The DataLoader batches/caches *per-subject detail* queries (it looks
    /// up the triples for a given subject IRI); it has no way to invent or
    /// enumerate subject IRIs on its own. So we always fetch the real,
    /// distinct subject IRIs from the store first via [`Self::resolve_subjects`],
    /// then warm the DataLoader's cache with those real keys so that any
    /// subsequent per-subject detail resolution (e.g. resolving nested
    /// fields for each subject) benefits from batching instead of issuing
    /// one SPARQL query per field.
    async fn resolve_subjects_optimized(&self, args: &HashMap<String, Value>) -> Result<Value> {
        let result = self.resolve_subjects(args).await?;

        if let (Some(ref loader), Value::ListValue(ref subjects)) = (&self.subject_loader, &result)
        {
            let keys: Vec<String> = subjects
                .iter()
                .filter_map(|v| match v {
                    Value::StringValue(s) => Some(s.clone()),
                    _ => None,
                })
                .collect();

            if !keys.is_empty() {
                if let Err(e) = loader.load_many(keys).await {
                    tracing::debug!("Subject DataLoader prefetch failed: {}", e);
                }
            }
        }

        Ok(result)
    }

    /// Optimized predicates resolver using DataLoader
    ///
    /// As with subjects, the DataLoader batches/caches per-predicate detail
    /// queries; it cannot enumerate predicates itself. We fetch the real
    /// `DISTINCT` predicate IRIs from the store first via
    /// [`Self::resolve_predicates`], then warm the DataLoader's cache with
    /// those real keys instead of a hardcoded, dataset-independent list.
    async fn resolve_predicates_optimized(&self, args: &HashMap<String, Value>) -> Result<Value> {
        let result = self.resolve_predicates(args).await?;

        if let (Some(ref loader), Value::ListValue(ref predicates)) =
            (&self.predicate_loader, &result)
        {
            let keys: Vec<String> = predicates
                .iter()
                .filter_map(|v| match v {
                    Value::StringValue(s) => Some(s.clone()),
                    _ => None,
                })
                .collect();

            if !keys.is_empty() {
                if let Err(e) = loader.load_many(keys).await {
                    tracing::debug!("Predicate DataLoader prefetch failed: {}", e);
                }
            }
        }

        Ok(result)
    }

    /// Cached SPARQL query resolver
    async fn resolve_sparql_query_cached(
        &self,
        args: &HashMap<String, Value>,
        cache_key: &str,
    ) -> Result<Value> {
        let query = args
            .get("query")
            .and_then(|v| match v {
                Value::StringValue(s) => Some(s.as_str()),
                _ => None,
            })
            .ok_or_else(|| anyhow!("SPARQL query argument required"))?;

        // For SPARQL queries, always check cache first due to complexity
        if let Some(ref cache) = self.cache {
            if let Some(cached_value) = cache.get(cache_key).await {
                if let Ok(value) = serde_json::from_value::<Value>(cached_value) {
                    tracing::info!("SPARQL cache hit for query: {}", query);
                    return Ok(value);
                }
            }
        }

        // Execute query if not cached, bounded by a wall-clock timeout so a
        // pathological or malicious query can't tie up the server
        // indefinitely. RdfStore::query is a blocking call (it takes a
        // std::sync::Mutex and runs the query synchronously), so it is run
        // on a blocking-pool thread rather than the async executor.
        let store = Arc::clone(&self.store);
        let owned_query = query.to_string();
        let timeout_duration = self.sparql_query_timeout;

        let join_result = tokio::time::timeout(
            timeout_duration,
            tokio::task::spawn_blocking(move || store.query(&owned_query)),
        )
        .await
        .map_err(|_| anyhow!("SPARQL query exceeded the {:?} timeout", timeout_duration))?;

        let results =
            join_result.map_err(|join_err| anyhow!("SPARQL query task failed: {}", join_err))??;

        let converted_results = self.convert_sparql_results_sync(results)?;

        // Cache SPARQL results with special handling
        if let Some(ref cache) = self.cache {
            if let Ok(json_value) = serde_json::to_value(&converted_results) {
                let mut dependencies = HashSet::new();
                dependencies.insert("sparql_query".to_string());
                dependencies.insert("rdf_data".to_string());

                let mut tags = HashSet::new();
                tags.insert("sparql".to_string());
                tags.insert("complex_query".to_string());

                cache
                    .set(
                        cache_key.to_string(),
                        json_value,
                        None,      // Use default TTL for SPARQL
                        Some(200), // High complexity for SPARQL queries
                        Some(dependencies),
                        Some(tags),
                    )
                    .await;
            }
        }

        Ok(converted_results)
    }

    async fn resolve_subjects(&self, args: &HashMap<String, Value>) -> Result<Value> {
        // Extract limit argument if provided
        let limit = args.get("limit").and_then(|v| match v {
            Value::IntValue(i) => Some(*i as usize),
            _ => None,
        });

        match self.store.get_subjects(limit) {
            Ok(subjects) => {
                let graphql_subjects: Vec<Value> =
                    subjects.into_iter().map(Value::StringValue).collect();
                Ok(Value::ListValue(graphql_subjects))
            }
            Err(err) => {
                tracing::error!("Failed to get subjects: {}", err);
                Ok(Value::ListValue(vec![]))
            }
        }
    }

    async fn resolve_predicates(&self, args: &HashMap<String, Value>) -> Result<Value> {
        let limit = args.get("limit").and_then(|v| match v {
            Value::IntValue(i) => Some(*i as usize),
            _ => None,
        });

        match self.store.get_predicates(limit) {
            Ok(predicates) => {
                let graphql_predicates: Vec<Value> =
                    predicates.into_iter().map(Value::StringValue).collect();
                Ok(Value::ListValue(graphql_predicates))
            }
            Err(err) => {
                tracing::error!("Failed to get predicates: {}", err);
                Ok(Value::ListValue(vec![]))
            }
        }
    }

    async fn resolve_objects(&self, args: &HashMap<String, Value>) -> Result<Value> {
        let limit = args.get("limit").and_then(|v| match v {
            Value::IntValue(i) => Some(*i as usize),
            _ => None,
        });

        match self.store.get_objects(limit) {
            Ok(objects) => {
                let graphql_objects: Vec<Value> = objects
                    .into_iter()
                    .map(|(value, object_type)| {
                        let mut obj = HashMap::new();
                        obj.insert("value".to_string(), Value::StringValue(value));
                        obj.insert("type".to_string(), Value::StringValue(object_type));
                        Value::ObjectValue(obj)
                    })
                    .collect();
                Ok(Value::ListValue(graphql_objects))
            }
            Err(err) => {
                tracing::error!("Failed to get objects: {}", err);
                Ok(Value::ListValue(vec![]))
            }
        }
    }

    /// Execute a raw SPARQL query
    #[allow(dead_code)]
    async fn resolve_sparql_query(&self, args: &HashMap<String, Value>) -> Result<Value> {
        let query = args
            .get("query")
            .and_then(|v| match v {
                Value::StringValue(s) => Some(s.as_str()),
                _ => None,
            })
            .ok_or_else(|| anyhow!("SPARQL query argument required"))?;

        // Execute query and convert results synchronously to avoid Send issues
        let results = self.store.query(query)?;
        let converted_results = self.convert_sparql_results_sync(results)?;
        Ok(converted_results)
    }

    /// Convert SPARQL query results to GraphQL Value synchronously
    fn convert_sparql_results_sync(&self, results: QueryResults) -> Result<Value> {
        match results {
            QueryResults::Solutions(solutions) => {
                let mut result_rows = Vec::new();

                // Collect all solutions synchronously
                for solution in solutions {
                    let mut row = HashMap::new();

                    // Iterate over variable-term bindings in the solution
                    for (var, term) in solution.iter() {
                        let value = match term {
                            oxirs_core::model::Term::NamedNode(node) => {
                                Value::StringValue(node.to_string())
                            }
                            oxirs_core::model::Term::BlankNode(node) => {
                                Value::StringValue(format!("_:{node}"))
                            }
                            oxirs_core::model::Term::Literal(literal) => {
                                // Try to parse as different types
                                if let Ok(int_val) = literal.value().parse::<i64>() {
                                    Value::IntValue(int_val)
                                } else if let Ok(float_val) = literal.value().parse::<f64>() {
                                    Value::FloatValue(float_val)
                                } else if let Ok(bool_val) = literal.value().parse::<bool>() {
                                    Value::BooleanValue(bool_val)
                                } else {
                                    Value::StringValue(literal.value().to_string())
                                }
                            }
                            // Note: Term::Triple is not currently supported
                            _ => Value::StringValue("Unknown term type".to_string()),
                        };
                        row.insert(var.name().to_string(), value);
                    }
                    result_rows.push(Value::ObjectValue(row));
                }

                Ok(Value::ListValue(result_rows))
            }
            QueryResults::Boolean(b) => Ok(Value::BooleanValue(b)),
            QueryResults::Graph(_) => {
                // For CONSTRUCT/DESCRIBE queries, we could serialize to RDF
                Ok(Value::StringValue("RDF graph result".to_string()))
            }
        }
    }
}

/// Minimal introspection resolver used only by [`ResolverRegistry`]'s
/// `FieldResolver`-trait wiring (itself exercised only in this crate's
/// tests, not by the production `GraphQLServer`/`Server` request paths).
///
/// This is **not** the schema introspection engine: it always reports an
/// empty `types` list and `null` for `__type` regardless of the real
/// schema. Real `__schema`/`__type` introspection is served by
/// [`crate::introspection::IntrospectionResolver`] (built from an
/// `Arc<Schema>`) together with [`crate::type_introspection`]; use those
/// for anything that needs accurate schema data. This stub is kept only so
/// `ResolverRegistry::setup_default_resolvers` has a type to register
/// under the `__Schema`/`__Type` type names in tests.
pub struct IntrospectionResolver;

impl IntrospectionResolver {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl FieldResolver for IntrospectionResolver {
    async fn resolve_field(
        &self,
        field_name: &str,
        _args: &HashMap<String, Value>,
        _context: &ExecutionContext,
    ) -> Result<Value> {
        match field_name {
            "__schema" => {
                // Return basic schema information
                let mut schema_obj = HashMap::new();
                schema_obj.insert("types".to_string(), Value::ListValue(vec![]));
                schema_obj.insert(
                    "queryType".to_string(),
                    Value::StringValue("Query".to_string()),
                );
                schema_obj.insert("mutationType".to_string(), Value::NullValue);
                schema_obj.insert("subscriptionType".to_string(), Value::NullValue);
                Ok(Value::ObjectValue(schema_obj))
            }
            "__type" => {
                // Return type information
                Ok(Value::NullValue)
            }
            _ => Ok(Value::NullValue),
        }
    }
}

impl Default for IntrospectionResolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Query resolvers container
pub struct QueryResolvers {
    rdf_resolver: Arc<RdfResolver>,
    introspection_resolver: Arc<IntrospectionResolver>,
}

impl QueryResolvers {
    pub fn new(store: Arc<RdfStore>) -> Self {
        Self {
            rdf_resolver: Arc::new(RdfResolver::new(store)),
            introspection_resolver: Arc::new(IntrospectionResolver::new()),
        }
    }

    /// Create query resolvers with explicit control over the raw SPARQL
    /// passthrough field's availability and timeout (see
    /// `GraphQLConfig::enable_sparql_field`/`sparql_query_timeout`).
    pub fn new_with_sparql_config(
        store: Arc<RdfStore>,
        enable_sparql_field: bool,
        sparql_query_timeout: std::time::Duration,
    ) -> Self {
        let rdf_resolver = RdfResolver::new(store)
            .with_sparql_enabled(enable_sparql_field)
            .with_sparql_query_timeout(sparql_query_timeout);
        Self {
            rdf_resolver: Arc::new(rdf_resolver),
            introspection_resolver: Arc::new(IntrospectionResolver::new()),
        }
    }

    pub fn new_with_mock(_store: Arc<crate::MockStore>) -> Self {
        // For backward compatibility during transition
        let rdf_store = Arc::new(RdfStore::new().expect("Failed to create RDF store"));
        Self {
            rdf_resolver: Arc::new(RdfResolver::new(rdf_store)),
            introspection_resolver: Arc::new(IntrospectionResolver::new()),
        }
    }

    pub fn rdf_resolver(&self) -> Arc<RdfResolver> {
        Arc::clone(&self.rdf_resolver)
    }

    pub fn introspection_resolver(&self) -> Arc<IntrospectionResolver> {
        Arc::clone(&self.introspection_resolver)
    }
}

/// Resolver registry for managing field resolvers
#[derive(Default)]
pub struct ResolverRegistry {
    resolvers: HashMap<String, Arc<dyn FieldResolver>>,
}

impl ResolverRegistry {
    pub fn new() -> Self {
        Self {
            resolvers: HashMap::new(),
        }
    }

    pub fn register<R: FieldResolver + 'static>(&mut self, type_name: String, resolver: R) {
        self.resolvers.insert(type_name, Arc::new(resolver));
    }

    pub fn register_arc(&mut self, type_name: String, resolver: Arc<dyn FieldResolver>) {
        self.resolvers.insert(type_name, resolver);
    }

    pub fn get(&self, type_name: &str) -> Option<Arc<dyn FieldResolver>> {
        self.resolvers.get(type_name).cloned()
    }

    pub fn setup_default_resolvers(&mut self, store: Arc<RdfStore>) {
        let query_resolvers = QueryResolvers::new(store);

        // Register the RDF resolver for Query type
        self.register_arc("Query".to_string(), query_resolvers.rdf_resolver());

        // Register introspection resolver for meta fields
        self.register_arc(
            "__Schema".to_string(),
            query_resolvers.introspection_resolver(),
        );
        self.register_arc(
            "__Type".to_string(),
            query_resolvers.introspection_resolver(),
        );
    }

    pub fn setup_default_resolvers_with_mock(&mut self, _store: Arc<crate::MockStore>) {
        // For backward compatibility during transition
        let rdf_store = Arc::new(RdfStore::new().expect("Failed to create RDF store"));
        self.setup_default_resolvers(rdf_store);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_store() -> Arc<RdfStore> {
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
        Arc::new(store)
    }

    /// Regression test for the "subjects" DataLoader resolver: it used to
    /// generate fake placeholder keys ("subject_0", "subject_1", ...) and
    /// look up a nonexistent "subject" JSON field, always returning an
    /// empty list regardless of what was in the store.
    #[tokio::test]
    async fn test_resolve_subjects_optimized_returns_real_iris() {
        let store = build_test_store();
        let resolver = RdfResolver::with_dataloader(store);

        let args = HashMap::new();
        let result = resolver
            .resolve_subjects_optimized(&args)
            .await
            .expect("resolve_subjects_optimized should succeed");

        let Value::ListValue(subjects) = result else {
            panic!("expected ListValue, got {result:?}");
        };
        let subjects: HashSet<String> = subjects
            .into_iter()
            .filter_map(|v| match v {
                Value::StringValue(s) => Some(s),
                _ => None,
            })
            .collect();

        assert_eq!(subjects.len(), 2);
        assert!(subjects.contains("http://example.org/alice"));
        assert!(subjects.contains("http://example.org/bob"));
        // Guard against the old bug reappearing: no fabricated placeholder keys.
        assert!(!subjects.iter().any(|s| s.starts_with("subject_")));
    }

    /// Regression test for the "predicates" DataLoader resolver: it used to
    /// hardcode three fixed predicate URIs (foaf:name, foaf:knows, rdf:type)
    /// regardless of the store's actual contents.
    #[tokio::test]
    async fn test_resolve_predicates_optimized_queries_store_not_hardcoded() {
        let store = build_test_store();
        let resolver = RdfResolver::with_dataloader(store);

        let args = HashMap::new();
        let result = resolver
            .resolve_predicates_optimized(&args)
            .await
            .expect("resolve_predicates_optimized should succeed");

        let Value::ListValue(predicates) = result else {
            panic!("expected ListValue, got {result:?}");
        };
        let predicates: HashSet<String> = predicates
            .into_iter()
            .filter_map(|v| match v {
                Value::StringValue(s) => Some(s),
                _ => None,
            })
            .collect();

        assert_eq!(predicates.len(), 1);
        assert!(predicates.contains("http://xmlns.com/foaf/0.1/name"));
        // Guard against the old bug reappearing: no hardcoded predicates
        // that aren't actually present in the store.
        assert!(!predicates.contains("http://xmlns.com/foaf/0.1/knows"));
        assert!(!predicates.contains("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"));
    }

    /// Regression test: the raw SPARQL passthrough field must refuse to run
    /// unless explicitly opted into, since it bypasses all GraphQL-level
    /// depth/complexity limits and lets a client run arbitrary SPARQL.
    #[tokio::test]
    async fn test_sparql_field_disabled_by_default() {
        let store = build_test_store();
        let resolver = RdfResolver::new(store);
        let context = ExecutionContext::new();

        let mut args = HashMap::new();
        args.insert(
            "query".to_string(),
            Value::StringValue("SELECT * WHERE { ?s ?p ?o }".to_string()),
        );

        let result = resolver.resolve_field("sparql", &args, &context).await;
        assert!(
            result.is_err(),
            "expected the sparql field to be disabled by default"
        );
    }

    /// Once explicitly opted in via `with_sparql_enabled(true)`, the field
    /// must execute the query and return real results.
    #[tokio::test]
    async fn test_sparql_field_executes_when_explicitly_enabled() {
        let store = build_test_store();
        let resolver = RdfResolver::new(store).with_sparql_enabled(true);
        let context = ExecutionContext::new();

        let mut args = HashMap::new();
        args.insert(
            "query".to_string(),
            Value::StringValue("SELECT ?s WHERE { ?s ?p ?o }".to_string()),
        );

        let result = resolver
            .resolve_field("sparql", &args, &context)
            .await
            .expect("sparql field should execute once enabled");

        match result {
            Value::ListValue(rows) => assert_eq!(rows.len(), 2),
            other => panic!("expected ListValue of solutions, got {other:?}"),
        }
    }

    /// A raw SPARQL query that exceeds the configured timeout must fail
    /// with an error rather than hang the resolver indefinitely.
    #[tokio::test]
    async fn test_sparql_field_respects_query_timeout() {
        let store = build_test_store();
        let resolver = RdfResolver::new(store)
            .with_sparql_enabled(true)
            .with_sparql_query_timeout(Duration::from_nanos(1));
        let context = ExecutionContext::new();

        let mut args = HashMap::new();
        args.insert(
            "query".to_string(),
            Value::StringValue("SELECT ?s WHERE { ?s ?p ?o }".to_string()),
        );

        let result = resolver.resolve_field("sparql", &args, &context).await;
        assert!(
            result.is_err(),
            "expected an effectively-zero timeout to trip, got {result:?}"
        );
    }
}
