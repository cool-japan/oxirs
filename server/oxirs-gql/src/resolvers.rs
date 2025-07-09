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
use std::time::Instant;

/// Enhanced RDF-based resolver with performance optimizations
pub struct RdfResolver {
    store: Arc<RdfStore>,
    subject_loader: Option<DataLoader<String, serde_json::Value>>,
    predicate_loader: Option<DataLoader<String, Vec<String>>>,
    cache: Option<Arc<AdvancedCache>>,
    performance_tracker: Option<Arc<PerformanceTracker>>,
}

impl RdfResolver {
    pub fn new(store: Arc<RdfStore>) -> Self {
        Self {
            store,
            subject_loader: None,
            predicate_loader: None,
            cache: None,
            performance_tracker: None,
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
        }
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
                // Execute raw SPARQL query with caching
                self.resolve_sparql_query_cached(args, &cache_key).await
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
    async fn resolve_subjects_optimized(&self, args: &HashMap<String, Value>) -> Result<Value> {
        // Extract limit argument if provided
        let limit = args.get("limit").and_then(|v| match v {
            Value::IntValue(i) => Some(*i as usize),
            _ => None,
        });

        // If DataLoader is available, use it for optimization
        if let Some(ref loader) = self.subject_loader {
            // For subject loading, we need to generate keys to load
            // This is a simplified implementation - in practice you'd want to
            // optimize based on actual query patterns
            let keys: Vec<String> = (0..limit.unwrap_or(10))
                .map(|i| format!("subject_{i}"))
                .collect();

            match loader.load_many(keys).await {
                Ok(loaded_data) => {
                    let graphql_subjects: Vec<Value> = loaded_data
                        .values()
                        .filter_map(|v| v.get("subject"))
                        .filter_map(|v| v.as_str())
                        .map(|s| Value::StringValue(s.to_string()))
                        .collect();
                    Ok(Value::ListValue(graphql_subjects))
                }
                Err(_) => {
                    // Fallback to direct store access
                    self.resolve_subjects(args).await
                }
            }
        } else {
            // Fallback to original implementation
            self.resolve_subjects(args).await
        }
    }

    /// Optimized predicates resolver using DataLoader
    async fn resolve_predicates_optimized(&self, args: &HashMap<String, Value>) -> Result<Value> {
        let limit = args.get("limit").and_then(|v| match v {
            Value::IntValue(i) => Some(*i as usize),
            _ => None,
        });

        // If DataLoader is available, use it for optimization
        if let Some(ref loader) = self.predicate_loader {
            // Generate predicate keys to load
            let keys: Vec<String> = vec![
                "http://xmlns.com/foaf/0.1/name".to_string(),
                "http://xmlns.com/foaf/0.1/knows".to_string(),
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
            ];

            match loader.load_many(keys).await {
                Ok(loaded_data) => {
                    let mut predicates: Vec<Value> = Vec::new();
                    for (predicate, _subjects) in loaded_data {
                        predicates.push(Value::StringValue(predicate));
                    }

                    // Apply limit if specified
                    if let Some(limit) = limit {
                        predicates.truncate(limit);
                    }

                    Ok(Value::ListValue(predicates))
                }
                Err(_) => {
                    // Fallback to direct store access
                    self.resolve_predicates(args).await
                }
            }
        } else {
            // Fallback to original implementation
            self.resolve_predicates(args).await
        }
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

        // Execute query if not cached
        let results = self.store.query(query)?;
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
                let graphql_subjects: Vec<Value> = subjects
                    .into_iter()
                    .map(Value::StringValue)
                    .collect();
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
                let graphql_predicates: Vec<Value> = predicates
                    .into_iter()
                    .map(Value::StringValue)
                    .collect();
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
                            _ => Value::StringValue("Unknown term type".to_string())
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

/// Introspection resolver for GraphQL schema introspection
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
