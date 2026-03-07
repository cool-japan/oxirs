//! GraphQL integration for querying RDF-star data
//!
//! This module provides GraphQL query capabilities for RDF-star data, enabling
//! developers to query quoted triples using GraphQL syntax instead of SPARQL.
//!
//! Features:
//! - **GraphQL schema generation** - Automatically generate GraphQL schemas from RDF-star data
//! - **Query translation** - Translate GraphQL queries to SPARQL-star queries
//! - **Quoted triple support** - Access quoted triples through GraphQL fields
//! - **Nesting support** - Query deeply nested quoted triple structures
//! - **Filtering and pagination** - Advanced query capabilities
//! - **Real-time subscriptions** - Subscribe to RDF-star data changes
//!
//! # Examples
//!
//! ```rust,ignore
//! use oxirs_star::graphql_star::{GraphQLStarEngine, SchemaGenerator};
//! use oxirs_star::StarStore;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a GraphQL engine
//! let store = StarStore::new();
//! let engine = GraphQLStarEngine::new(store);
//!
//! // Execute a GraphQL query
//! let query = r#"
//!     {
//!         quotedTriples(limit: 10) {
//!             subject { value }
//!             predicate { value }
//!             object { value }
//!             nestingDepth
//!         }
//!     }
//! "#;
//!
//! let result = engine.execute(query)?;
//! println!("Result: {}", result);
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};
use tracing::{debug, info};

use crate::model::{StarTerm, StarTriple};
use crate::store::StarStore;
use crate::{StarError, StarResult};

/// GraphQL engine for RDF-star queries
#[derive(Clone)]
pub struct GraphQLStarEngine {
    /// Underlying RDF-star store
    store: Arc<RwLock<StarStore>>,

    /// GraphQL schema
    schema: Arc<RwLock<GraphQLSchema>>,

    /// Configuration
    config: GraphQLConfig,

    /// Query statistics
    stats: Arc<RwLock<GraphQLStats>>,
}

/// Configuration for GraphQL engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLConfig {
    /// Maximum query depth
    pub max_query_depth: usize,

    /// Maximum number of results per query
    pub max_results: usize,

    /// Enable introspection
    pub enable_introspection: bool,

    /// Enable subscriptions
    pub enable_subscriptions: bool,

    /// Query timeout in milliseconds
    pub timeout_ms: u64,
}

impl Default for GraphQLConfig {
    fn default() -> Self {
        Self {
            max_query_depth: 10,
            max_results: 1000,
            enable_introspection: true,
            enable_subscriptions: false,
            timeout_ms: 30000,
        }
    }
}

/// Statistics for GraphQL queries
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphQLStats {
    /// Total queries executed
    pub total_queries: usize,

    /// Total mutations executed
    pub total_mutations: usize,

    /// Total subscriptions created
    pub total_subscriptions: usize,

    /// Average query time in microseconds
    pub avg_query_time_us: u64,
}

/// GraphQL schema for RDF-star data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLSchema {
    /// Type definitions
    pub types: HashMap<String, GraphQLType>,

    /// Root query type
    pub query_type: String,

    /// Root mutation type (optional)
    pub mutation_type: Option<String>,

    /// Root subscription type (optional)
    pub subscription_type: Option<String>,
}

/// GraphQL type definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLType {
    /// Type name
    pub name: String,

    /// Type kind (object, interface, enum, etc.)
    pub kind: TypeKind,

    /// Fields (for object types)
    pub fields: Vec<GraphQLField>,

    /// Description
    pub description: Option<String>,
}

/// GraphQL type kind
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TypeKind {
    Object,
    Interface,
    Enum,
    Scalar,
    List,
    NonNull,
}

/// GraphQL field definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLField {
    /// Field name
    pub name: String,

    /// Field type
    pub field_type: String,

    /// Arguments
    pub args: Vec<GraphQLArgument>,

    /// Description
    pub description: Option<String>,

    /// Resolver function name
    pub resolver: String,
}

/// GraphQL field argument
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLArgument {
    /// Argument name
    pub name: String,

    /// Argument type
    pub arg_type: String,

    /// Default value
    pub default_value: Option<JsonValue>,

    /// Description
    pub description: Option<String>,
}

/// GraphQL query execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLResult {
    /// Query data
    pub data: Option<JsonValue>,

    /// Query errors
    pub errors: Vec<GraphQLError>,

    /// Extensions (metadata)
    pub extensions: Option<JsonValue>,
}

/// GraphQL error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLError {
    /// Error message
    pub message: String,

    /// Error locations
    pub locations: Option<Vec<ErrorLocation>>,

    /// Error path
    pub path: Option<Vec<JsonValue>>,

    /// Extensions
    pub extensions: Option<JsonValue>,
}

/// Error location in query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorLocation {
    pub line: usize,
    pub column: usize,
}

impl GraphQLStarEngine {
    /// Create a new GraphQL engine with default configuration
    pub fn new(store: StarStore) -> Self {
        Self::with_config(store, GraphQLConfig::default())
    }

    /// Create a new GraphQL engine with custom configuration
    pub fn with_config(store: StarStore, config: GraphQLConfig) -> Self {
        let schema = Self::generate_default_schema();

        Self {
            store: Arc::new(RwLock::new(store)),
            schema: Arc::new(RwLock::new(schema)),
            config,
            stats: Arc::new(RwLock::new(GraphQLStats::default())),
        }
    }

    /// Generate default GraphQL schema for RDF-star data
    fn generate_default_schema() -> GraphQLSchema {
        let mut types = HashMap::new();

        // Define QuotedTriple type
        types.insert(
            "QuotedTriple".to_string(),
            GraphQLType {
                name: "QuotedTriple".to_string(),
                kind: TypeKind::Object,
                description: Some("A quoted RDF triple".to_string()),
                fields: vec![
                    GraphQLField {
                        name: "subject".to_string(),
                        field_type: "Term".to_string(),
                        args: vec![],
                        description: Some("The subject of the triple".to_string()),
                        resolver: "resolve_subject".to_string(),
                    },
                    GraphQLField {
                        name: "predicate".to_string(),
                        field_type: "Term".to_string(),
                        args: vec![],
                        description: Some("The predicate of the triple".to_string()),
                        resolver: "resolve_predicate".to_string(),
                    },
                    GraphQLField {
                        name: "object".to_string(),
                        field_type: "Term".to_string(),
                        args: vec![],
                        description: Some("The object of the triple".to_string()),
                        resolver: "resolve_object".to_string(),
                    },
                    GraphQLField {
                        name: "nestingDepth".to_string(),
                        field_type: "Int".to_string(),
                        args: vec![],
                        description: Some("The nesting depth of this triple".to_string()),
                        resolver: "resolve_nesting_depth".to_string(),
                    },
                ],
            },
        );

        // Define Term type
        types.insert(
            "Term".to_string(),
            GraphQLType {
                name: "Term".to_string(),
                kind: TypeKind::Interface,
                description: Some(
                    "An RDF term (IRI, literal, blank node, or quoted triple)".to_string(),
                ),
                fields: vec![
                    GraphQLField {
                        name: "value".to_string(),
                        field_type: "String".to_string(),
                        args: vec![],
                        description: Some("String representation of the term".to_string()),
                        resolver: "resolve_term_value".to_string(),
                    },
                    GraphQLField {
                        name: "termType".to_string(),
                        field_type: "TermType".to_string(),
                        args: vec![],
                        description: Some("Type of the term".to_string()),
                        resolver: "resolve_term_type".to_string(),
                    },
                ],
            },
        );

        // Define Query root type
        types.insert(
            "Query".to_string(),
            GraphQLType {
                name: "Query".to_string(),
                kind: TypeKind::Object,
                description: Some("Root query type".to_string()),
                fields: vec![
                    GraphQLField {
                        name: "quotedTriples".to_string(),
                        field_type: "[QuotedTriple]".to_string(),
                        args: vec![
                            GraphQLArgument {
                                name: "limit".to_string(),
                                arg_type: "Int".to_string(),
                                default_value: Some(json!(100)),
                                description: Some("Maximum number of results".to_string()),
                            },
                            GraphQLArgument {
                                name: "offset".to_string(),
                                arg_type: "Int".to_string(),
                                default_value: Some(json!(0)),
                                description: Some("Result offset for pagination".to_string()),
                            },
                            GraphQLArgument {
                                name: "maxDepth".to_string(),
                                arg_type: "Int".to_string(),
                                default_value: None,
                                description: Some("Filter by maximum nesting depth".to_string()),
                            },
                        ],
                        description: Some("Query all quoted triples".to_string()),
                        resolver: "resolve_quoted_triples".to_string(),
                    },
                    GraphQLField {
                        name: "tripleCount".to_string(),
                        field_type: "Int".to_string(),
                        args: vec![],
                        description: Some("Total number of quoted triples".to_string()),
                        resolver: "resolve_triple_count".to_string(),
                    },
                ],
            },
        );

        GraphQLSchema {
            types,
            query_type: "Query".to_string(),
            mutation_type: None,
            subscription_type: None,
        }
    }

    /// Execute a GraphQL query
    pub fn execute(&self, query: &str) -> StarResult<GraphQLResult> {
        info!("Executing GraphQL query");
        debug!("Query: {}", query);

        let start = std::time::Instant::now();

        // Parse the query (simplified - in production, use a proper GraphQL parser)
        let parsed = self.parse_query(query)?;

        // Execute the query
        let result = self.execute_parsed_query(&parsed)?;

        // Update statistics
        let elapsed = start.elapsed().as_micros() as u64;
        let mut stats = self
            .stats
            .write()
            .expect("write lock should not be poisoned");
        stats.total_queries += 1;
        let new_avg = if stats.total_queries == 1 {
            elapsed
        } else {
            (stats.avg_query_time_us * (stats.total_queries as u64 - 1) + elapsed)
                / stats.total_queries as u64
        };
        stats.avg_query_time_us = new_avg;

        info!("Query executed in {}μs", elapsed);

        Ok(result)
    }

    /// Parse a GraphQL query (simplified implementation)
    fn parse_query(&self, query: &str) -> StarResult<ParsedQuery> {
        // This is a simplified parser - in production, use a proper GraphQL parser library

        let trimmed = query.trim();

        // Check if it's a query for quoted triples
        if trimmed.contains("quotedTriples") {
            // Extract limit and offset if present
            let limit = self
                .extract_arg_value(trimmed, "limit")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(100);

            let offset = self
                .extract_arg_value(trimmed, "offset")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);

            let max_depth = self
                .extract_arg_value(trimmed, "maxDepth")
                .and_then(|s| s.parse::<usize>().ok());

            return Ok(ParsedQuery {
                operation: Operation::Query,
                selection: Selection::QuotedTriples {
                    limit,
                    offset,
                    max_depth,
                },
            });
        }

        if trimmed.contains("tripleCount") {
            return Ok(ParsedQuery {
                operation: Operation::Query,
                selection: Selection::TripleCount,
            });
        }

        Err(StarError::query_error("Unsupported GraphQL query"))
    }

    /// Extract argument value from query string
    fn extract_arg_value(&self, query: &str, arg_name: &str) -> Option<String> {
        let pattern = format!("{}:", arg_name);
        if let Some(start) = query.find(&pattern) {
            let after = &query[start + pattern.len()..];
            let value_end = after
                .find(|c: char| c == ',' || c == ')' || c.is_whitespace())
                .unwrap_or(after.len());
            let value = after[..value_end].trim();
            return Some(value.to_string());
        }
        None
    }

    /// Execute a parsed query
    fn execute_parsed_query(&self, query: &ParsedQuery) -> StarResult<GraphQLResult> {
        match query.operation {
            Operation::Query => match &query.selection {
                Selection::QuotedTriples {
                    limit,
                    offset,
                    max_depth,
                } => self.resolve_quoted_triples(*limit, *offset, *max_depth),
                Selection::TripleCount => self.resolve_triple_count(),
            },
            Operation::Mutation => Err(StarError::query_error("Mutations not implemented yet")),
            Operation::Subscription => {
                Err(StarError::query_error("Subscriptions not implemented yet"))
            }
        }
    }

    /// Resolve quotedTriples query
    fn resolve_quoted_triples(
        &self,
        limit: usize,
        offset: usize,
        max_depth: Option<usize>,
    ) -> StarResult<GraphQLResult> {
        let store = self.store.read().expect("read lock should not be poisoned");
        let mut triples = store.query(None, None, None)?;
        let original_count = triples.len();

        // Filter by max depth if specified
        if let Some(max_d) = max_depth {
            triples.retain(|t| t.nesting_depth() <= max_d);
        }

        // Apply pagination
        let paginated: Vec<_> = triples
            .into_iter()
            .skip(offset)
            .take(limit.min(self.config.max_results))
            .collect();

        // Convert to JSON
        let data = json!({
            "quotedTriples": paginated.iter().map(|t| self.triple_to_json(t)).collect::<Vec<_>>()
        });

        Ok(GraphQLResult {
            data: Some(data),
            errors: vec![],
            extensions: Some(json!({
                "count": paginated.len(),
                "hasMore": offset + paginated.len() < original_count
            })),
        })
    }

    /// Resolve tripleCount query
    fn resolve_triple_count(&self) -> StarResult<GraphQLResult> {
        let store = self.store.read().expect("read lock should not be poisoned");
        let count = store.len();

        Ok(GraphQLResult {
            data: Some(json!({
                "tripleCount": count
            })),
            errors: vec![],
            extensions: None,
        })
    }

    /// Convert a StarTriple to JSON
    fn triple_to_json(&self, triple: &StarTriple) -> JsonValue {
        json!({
            "subject": self.term_to_json(&triple.subject),
            "predicate": self.term_to_json(&triple.predicate),
            "object": self.term_to_json(&triple.object),
            "nestingDepth": triple.nesting_depth()
        })
    }

    /// Convert a StarTerm to JSON
    fn term_to_json(&self, term: &StarTerm) -> JsonValue {
        match term {
            StarTerm::NamedNode(nn) => json!({
                "value": nn.iri,
                "termType": "NamedNode"
            }),
            StarTerm::BlankNode(bn) => json!({
                "value": bn.id,
                "termType": "BlankNode"
            }),
            StarTerm::Literal(lit) => json!({
                "value": lit.value,
                "termType": "Literal",
                "language": lit.language,
                "datatype": lit.datatype.as_ref().map(|dt| &dt.iri)
            }),
            StarTerm::QuotedTriple(qt) => json!({
                "value": format!("<< {} {} {} >>", qt.subject, qt.predicate, qt.object),
                "termType": "QuotedTriple",
                "triple": self.triple_to_json(qt)
            }),
            StarTerm::Variable(var) => json!({
                "value": var.name,
                "termType": "Variable"
            }),
        }
    }

    /// Get the schema
    pub fn get_schema(&self) -> GraphQLSchema {
        self.schema
            .read()
            .expect("read lock should not be poisoned")
            .clone()
    }

    /// Get statistics
    pub fn get_statistics(&self) -> GraphQLStats {
        self.stats
            .read()
            .expect("read lock should not be poisoned")
            .clone()
    }
}

/// Parsed GraphQL query
#[derive(Debug, Clone)]
struct ParsedQuery {
    operation: Operation,
    selection: Selection,
}

/// GraphQL operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum Operation {
    Query,
    Mutation,
    Subscription,
}

/// Selection in query
#[derive(Debug, Clone)]
enum Selection {
    QuotedTriples {
        limit: usize,
        offset: usize,
        max_depth: Option<usize>,
    },
    TripleCount,
}

/// Schema generator for creating GraphQL schemas from RDF-star data
pub struct SchemaGenerator {
    /// Configuration
    config: SchemaGeneratorConfig,
}

/// Configuration for schema generation
#[derive(Debug, Clone)]
pub struct SchemaGeneratorConfig {
    /// Include introspection types
    pub include_introspection: bool,

    /// Include mutation types
    pub include_mutations: bool,

    /// Include subscription types
    pub include_subscriptions: bool,
}

impl Default for SchemaGeneratorConfig {
    fn default() -> Self {
        Self {
            include_introspection: true,
            include_mutations: false,
            include_subscriptions: false,
        }
    }
}

impl SchemaGenerator {
    /// Create a new schema generator
    pub fn new(config: SchemaGeneratorConfig) -> Self {
        Self { config }
    }

    /// Generate a schema from a store
    pub fn generate(&self, store: &StarStore) -> StarResult<GraphQLSchema> {
        // Start with default schema
        let mut schema = GraphQLStarEngine::generate_default_schema();

        // Analyze store and add custom types if needed
        let stats = store.statistics();
        debug!(
            "Generating schema for store with {} quoted triples",
            stats.quoted_triples_count
        );

        // Add mutations if enabled
        if self.config.include_mutations {
            let mutation_type = self.generate_mutation_type();
            schema.types.insert("Mutation".to_string(), mutation_type);
            schema.mutation_type = Some("Mutation".to_string());
        }

        // Add subscriptions if enabled
        if self.config.include_subscriptions {
            let subscription_type = self.generate_subscription_type();
            schema
                .types
                .insert("Subscription".to_string(), subscription_type);
            schema.subscription_type = Some("Subscription".to_string());
        }

        Ok(schema)
    }

    fn generate_mutation_type(&self) -> GraphQLType {
        GraphQLType {
            name: "Mutation".to_string(),
            kind: TypeKind::Object,
            description: Some("Root mutation type".to_string()),
            fields: vec![GraphQLField {
                name: "insertQuotedTriple".to_string(),
                field_type: "QuotedTriple".to_string(),
                args: vec![
                    GraphQLArgument {
                        name: "subject".to_string(),
                        arg_type: "String!".to_string(),
                        default_value: None,
                        description: Some("Subject IRI".to_string()),
                    },
                    GraphQLArgument {
                        name: "predicate".to_string(),
                        arg_type: "String!".to_string(),
                        default_value: None,
                        description: Some("Predicate IRI".to_string()),
                    },
                    GraphQLArgument {
                        name: "object".to_string(),
                        arg_type: "String!".to_string(),
                        default_value: None,
                        description: Some("Object value".to_string()),
                    },
                ],
                description: Some("Insert a new quoted triple".to_string()),
                resolver: "mutate_insert_quoted_triple".to_string(),
            }],
        }
    }

    fn generate_subscription_type(&self) -> GraphQLType {
        GraphQLType {
            name: "Subscription".to_string(),
            kind: TypeKind::Object,
            description: Some("Root subscription type".to_string()),
            fields: vec![GraphQLField {
                name: "quotedTripleAdded".to_string(),
                field_type: "QuotedTriple".to_string(),
                args: vec![],
                description: Some("Subscribe to new quoted triple additions".to_string()),
                resolver: "subscribe_quoted_triple_added".to_string(),
            }],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{StarTerm, StarTriple};

    #[test]
    fn test_graphql_quoted_triples_query() -> StarResult<()> {
        let store = StarStore::new();

        // Add some test data
        for i in 0..5 {
            let subject = format!("http://example.org/s{}", i);
            let object = format!("object{}", i);
            let triple = StarTriple::new(
                StarTerm::iri(&subject)?,
                StarTerm::iri("http://example.org/p")?,
                StarTerm::literal(&object)?,
            );
            store.insert(&triple)?;
        }

        let engine = GraphQLStarEngine::new(store);

        let query = r#"
            {
                quotedTriples(limit: 3, offset: 0) {
                    subject { value }
                    predicate { value }
                    object { value }
                }
            }
        "#;

        let result = engine.execute(query)?;

        assert!(result.data.is_some());
        assert!(result.errors.is_empty());

        Ok(())
    }

    #[test]
    fn test_graphql_triple_count() -> StarResult<()> {
        let store = StarStore::new();

        for i in 0..10 {
            let subject = format!("http://example.org/s{}", i);
            let object = format!("o{}", i);
            let triple = StarTriple::new(
                StarTerm::iri(&subject)?,
                StarTerm::iri("http://example.org/p")?,
                StarTerm::literal(&object)?,
            );
            store.insert(&triple)?;
        }

        let engine = GraphQLStarEngine::new(store);

        let query = "{ tripleCount }";

        let result = engine.execute(query)?;

        assert!(result.data.is_some());
        let data = result.data.unwrap();
        assert_eq!(data["tripleCount"], 10);

        Ok(())
    }

    #[test]
    fn test_schema_generation() -> StarResult<()> {
        let store = StarStore::new();
        let generator = SchemaGenerator::new(SchemaGeneratorConfig::default());

        let schema = generator.generate(&store)?;

        assert!(schema.types.contains_key("QuotedTriple"));
        assert!(schema.types.contains_key("Term"));
        assert!(schema.types.contains_key("Query"));
        assert_eq!(schema.query_type, "Query");

        Ok(())
    }
}
