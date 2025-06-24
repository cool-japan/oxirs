//! GraphQL resolvers for RDF data
//!
//! This module provides resolvers that translate GraphQL field selections
//! to SPARQL queries against RDF datasets.

use crate::ast::Value;
use crate::execution::{ExecutionContext, FieldResolver};
use crate::RdfStore;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use oxigraph::sparql::QueryResults;

/// RDF-based resolver that executes SPARQL queries
pub struct RdfResolver {
    store: Arc<RdfStore>,
}

impl RdfResolver {
    pub fn new(store: Arc<RdfStore>) -> Self {
        Self { store }
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
        tracing::debug!(
            "Resolving field '{}' with args: {:?} in request {}",
            field_name,
            args,
            context.request_id
        );

        match field_name {
            "hello" => {
                // Simple test resolver
                Ok(Value::StringValue("Hello from OxiRS GraphQL!".to_string()))
            }
            "version" => {
                Ok(Value::StringValue(env!("CARGO_PKG_VERSION").to_string()))
            }
            "triples" => {
                // Return count of triples in the store
                self.resolve_triples_count(args).await
            }
            "subjects" => {
                // Return list of subjects
                self.resolve_subjects(args).await
            }
            "predicates" => {
                // Return list of predicates
                self.resolve_predicates(args).await
            }
            "objects" => {
                // Return list of objects
                self.resolve_objects(args).await
            }
            "sparql" => {
                // Execute raw SPARQL query
                self.resolve_sparql_query(args).await
            }
            _ => {
                tracing::warn!("Unknown field '{}' requested", field_name);
                Ok(Value::NullValue)
            }
        }
    }
}

impl RdfResolver {
    async fn resolve_triples_count(&self, _args: &HashMap<String, Value>) -> Result<Value> {
        match self.store.triple_count() {
            Ok(count) => Ok(Value::IntValue(count as i64)),
            Err(err) => {
                tracing::error!("Failed to count triples: {}", err);
                Ok(Value::IntValue(0))
            }
        }
    }

    async fn resolve_subjects(&self, args: &HashMap<String, Value>) -> Result<Value> {
        // Extract limit argument if provided
        let limit = args.get("limit")
            .and_then(|v| match v {
                Value::IntValue(i) => Some(*i as usize),
                _ => None,
            });

        match self.store.get_subjects(limit) {
            Ok(subjects) => {
                let graphql_subjects: Vec<Value> = subjects
                    .into_iter()
                    .map(|s| Value::StringValue(s))
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
        let limit = args.get("limit")
            .and_then(|v| match v {
                Value::IntValue(i) => Some(*i as usize),
                _ => None,
            });

        match self.store.get_predicates(limit) {
            Ok(predicates) => {
                let graphql_predicates: Vec<Value> = predicates
                    .into_iter()
                    .map(|p| Value::StringValue(p))
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
        let limit = args.get("limit")
            .and_then(|v| match v {
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
    async fn resolve_sparql_query(&self, args: &HashMap<String, Value>) -> Result<Value> {
        let query = args.get("query")
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
                    let solution = solution?;
                    let mut row = HashMap::new();
                    
                    for (var, term) in solution.iter() {
                        let value = match term {
                            oxigraph::model::Term::NamedNode(node) => {
                                Value::StringValue(node.to_string())
                            }
                            oxigraph::model::Term::BlankNode(node) => {
                                Value::StringValue(format!("_:{}", node))
                            }
                            oxigraph::model::Term::Literal(literal) => {
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
                            oxigraph::model::Term::Triple(_) => {
                                Value::StringValue("RDF-star triple".to_string())
                            }
                        };
                        row.insert(var.to_string(), value);
                    }
                    result_rows.push(Value::ObjectValue(row));
                }
                
                Ok(Value::ListValue(result_rows))
            }
            QueryResults::Boolean(b) => {
                Ok(Value::BooleanValue(b))
            }
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
                schema_obj.insert(
                    "types".to_string(),
                    Value::ListValue(vec![]),
                );
                schema_obj.insert(
                    "queryType".to_string(),
                    Value::StringValue("Query".to_string()),
                );
                schema_obj.insert(
                    "mutationType".to_string(),
                    Value::NullValue,
                );
                schema_obj.insert(
                    "subscriptionType".to_string(),
                    Value::NullValue,
                );
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
    
    pub fn new_with_mock(store: Arc<crate::MockStore>) -> Self {
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
        self.register_arc("__Schema".to_string(), query_resolvers.introspection_resolver());
        self.register_arc("__Type".to_string(), query_resolvers.introspection_resolver());
    }
    
    pub fn setup_default_resolvers_with_mock(&mut self, store: Arc<crate::MockStore>) {
        // For backward compatibility during transition
        let rdf_store = Arc::new(RdfStore::new().expect("Failed to create RDF store"));
        self.setup_default_resolvers(rdf_store);
    }
}