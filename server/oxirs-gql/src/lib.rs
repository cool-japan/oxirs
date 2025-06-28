//! # OxiRS GraphQL
//!
//! GraphQL façade for OxiRS with automatic schema generation from RDF ontologies.
//!
//! This crate provides a GraphQL interface that automatically maps RDF data to GraphQL
//! schemas, enabling modern GraphQL clients to query knowledge graphs.

use anyhow::Result;
use oxirs_core::model::{
    BlankNode, GraphName, Literal as OxiLiteral, NamedNode, Quad, Subject, Term, Triple, Variable,
};
use oxirs_core::Store;
use std::sync::Arc;

// Re-export QueryResults for other modules
pub use oxirs_core::query::QueryResults;

// Module declarations are below after the main code

/// RDF store wrapper for GraphQL integration
pub struct RdfStore {
    store: Store,
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
            store: Store::new()?,
        })
    }

    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        Ok(Self {
            store: Store::open(path)?,
        })
    }

    /// Execute a SPARQL query and return results
    pub fn query(&self, query: &str) -> Result<QueryResults> {
        use oxirs_core::query::{QueryEngine, QueryResult};

        let engine = QueryEngine::new();
        let result = engine
            .query(query, &self.store)
            .map_err(|e| anyhow::anyhow!("SPARQL query error: {}", e))?;

        match result {
            QueryResult::Select {
                variables,
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
        match self.query(query)? {
            QueryResults::Solutions(solutions) => {
                if let Some(solution) = solutions.first() {
                    if let Some(Term::Literal(lit)) = solution.get(&Variable::new("count").unwrap())
                    {
                        if let Ok(count) = lit.value().parse::<usize>() {
                            return Ok(count);
                        }
                    }
                }
            }
            _ => {}
        }
        Ok(0)
    }

    /// Get subjects with optional limit
    pub fn get_subjects(&self, limit: Option<usize>) -> Result<Vec<String>> {
        let limit_clause = match limit {
            Some(l) => format!(" LIMIT {}", l),
            None => String::new(),
        };

        let query = format!("SELECT DISTINCT ?s WHERE {{ ?s ?p ?o }}{}", limit_clause);
        let mut subjects = Vec::new();

        match self.query(&query)? {
            QueryResults::Solutions(solutions) => {
                for solution in &solutions {
                    if let Some(subject) = solution.get(&Variable::new("s").unwrap()) {
                        subjects.push(subject.to_string());
                    }
                }
            }
            _ => {}
        }

        Ok(subjects)
    }

    /// Get predicates with optional limit
    pub fn get_predicates(&self, limit: Option<usize>) -> Result<Vec<String>> {
        let limit_clause = match limit {
            Some(l) => format!(" LIMIT {}", l),
            None => String::new(),
        };

        let query = format!("SELECT DISTINCT ?p WHERE {{ ?s ?p ?o }}{}", limit_clause);
        let mut predicates = Vec::new();

        match self.query(&query)? {
            QueryResults::Solutions(solutions) => {
                for solution in &solutions {
                    if let Some(predicate) = solution.get(&Variable::new("p").unwrap()) {
                        predicates.push(predicate.to_string());
                    }
                }
            }
            _ => {}
        }

        Ok(predicates)
    }

    /// Get objects with optional limit
    pub fn get_objects(&self, limit: Option<usize>) -> Result<Vec<(String, String)>> {
        let limit_clause = match limit {
            Some(l) => format!(" LIMIT {}", l),
            None => String::new(),
        };

        let query = format!("SELECT DISTINCT ?o WHERE {{ ?s ?p ?o }}{}", limit_clause);
        let mut objects = Vec::new();

        match self.query(&query)? {
            QueryResults::Solutions(solutions) => {
                for solution in &solutions {
                    if let Some(object) = solution.get(&Variable::new("o").unwrap()) {
                        let object_type = match object {
                            Term::NamedNode(_) => "IRI".to_string(),
                            Term::BlankNode(_) => "BlankNode".to_string(),
                            Term::Literal(_) => "Literal".to_string(),
                            Term::Variable(_) => "Variable".to_string(),
                            Term::QuotedTriple(_) => "QuotedTriple".to_string(),
                        };
                        objects.push((object.to_string(), object_type));
                    }
                }
            }
            _ => {}
        }

        Ok(objects)
    }

    /// Insert a triple into the store
    pub fn insert_triple(&mut self, subject: &str, predicate: &str, object: &str) -> Result<()> {
        // Parse terms
        let subject = if subject.starts_with("_:") {
            Subject::BlankNode(BlankNode::new(&subject[2..])?)
        } else {
            Subject::NamedNode(NamedNode::new(subject)?)
        };

        let predicate = NamedNode::new(predicate)?;

        let object = if object.starts_with("\"") && object.ends_with("\"") {
            // It's a literal
            let literal_value = &object[1..object.len() - 1];
            Term::Literal(OxiLiteral::new_simple_literal(literal_value))
        } else if object.starts_with("_:") {
            // It's a blank node
            Term::BlankNode(BlankNode::new(&object[2..])?)
        } else {
            // It's a named node
            Term::NamedNode(NamedNode::new(object)?)
        };

        let quad = Quad::new(subject, predicate, object, GraphName::DefaultGraph);
        self.store.insert_quad(quad)?;
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
        for quad in quads {
            self.store.insert_quad(quad)?;
        }

        Ok(())
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
pub mod ast;
pub mod execution;
pub mod federation;
pub mod introspection;
pub mod mapping;
pub mod optimizer;
pub mod parser;
pub mod rdf_scalars;
pub mod resolvers;
pub mod schema;
pub mod server;
pub mod subscriptions;
pub mod types;
pub mod validation;

// Organized module groups
pub mod core;
pub mod docs;
pub mod features;
pub mod networking;
pub mod rdf;

// Juniper-based implementation with proper RDF integration (WIP - disabled due to complex integration issues)
// pub mod juniper_schema;
// pub mod juniper_server; // Complex Hyper v1 version - disabled for now
// pub mod simple_juniper_server; // Simplified version

// Future Juniper integration - comprehensive RDF GraphQL support planned
// pub use juniper_schema::{Schema as JuniperSchema, GraphQLContext, create_schema};
// pub use simple_juniper_server::{JuniperGraphQLServer, GraphQLServerConfig, GraphQLServerBuilder, start_graphql_server, start_graphql_server_with_config};

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
        }
    }
}

/// Main GraphQL server
pub struct GraphQLServer {
    config: GraphQLConfig,
    store: Arc<RdfStore>,
}

impl GraphQLServer {
    pub fn new(store: Arc<RdfStore>) -> Self {
        Self {
            config: GraphQLConfig::default(),
            store,
        }
    }

    pub fn new_with_mock(store: Arc<MockStore>) -> Self {
        // For backward compatibility during transition
        let rdf_store = Arc::new(RdfStore::new().expect("Failed to create RDF store"));
        Self {
            config: GraphQLConfig::default(),
            store: rdf_store,
        }
    }

    pub fn with_config(mut self, config: GraphQLConfig) -> Self {
        self.config = config;
        self
    }

    pub async fn start(&self, addr: &str) -> Result<()> {
        tracing::info!("Starting GraphQL server on {}", addr);

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
            )
            .with_field(
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

        // Add introspection fields if enabled
        if self.config.enable_introspection {
            query_type = query_type
                .with_field(
                    "__schema".to_string(),
                    types::FieldType::new(
                        "__schema".to_string(),
                        types::GraphQLType::NonNull(Box::new(types::GraphQLType::Scalar(
                            types::ScalarType {
                                name: "__Schema".to_string(),
                                description: Some(
                                    "A GraphQL Schema defines the capabilities of a GraphQL server"
                                        .to_string(),
                                ),
                                serialize: |_| Ok(ast::Value::NullValue),
                                parse_value: |_| Err(anyhow::anyhow!("Cannot parse __Schema")),
                                parse_literal: |_| Err(anyhow::anyhow!("Cannot parse __Schema")),
                            },
                        ))),
                    )
                    .with_description("Access the current type schema of this server".to_string()),
                )
                .with_field(
                    "__type".to_string(),
                    types::FieldType::new(
                        "__type".to_string(),
                        types::GraphQLType::Scalar(types::ScalarType {
                            name: "__Type".to_string(),
                            description: Some(
                                "A GraphQL Schema defines the capabilities of a GraphQL server"
                                    .to_string(),
                            ),
                            serialize: |_| Ok(ast::Value::NullValue),
                            parse_value: |_| Err(anyhow::anyhow!("Cannot parse __Type")),
                            parse_literal: |_| Err(anyhow::anyhow!("Cannot parse __Type")),
                        }),
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

        schema.add_type(types::GraphQLType::Object(query_type));
        schema.set_query_type("Query".to_string());

        // Create the server with resolvers
        let schema_clone = schema.clone();
        let mut server = server::Server::new(schema.clone())
            .with_playground(self.config.enable_playground)
            .with_introspection(self.config.enable_introspection);

        // Configure validation if enabled
        if self.config.enable_query_validation {
            server =
                server.with_validation(self.config.validation_config.clone(), schema_clone.clone());
        }

        // Set up resolvers
        let query_resolvers = resolvers::QueryResolvers::new(Arc::clone(&self.store));
        server.add_resolver("Query".to_string(), query_resolvers.rdf_resolver());

        // Add introspection resolver
        let introspection_resolver = Arc::new(introspection::IntrospectionResolver::new(Arc::new(
            schema_clone,
        )));
        server.add_resolver("__Schema".to_string(), introspection_resolver.clone());
        server.add_resolver("__Type".to_string(), introspection_resolver.clone());
        server.add_resolver("__Field".to_string(), introspection_resolver.clone());
        server.add_resolver("__InputValue".to_string(), introspection_resolver.clone());
        server.add_resolver("__EnumValue".to_string(), introspection_resolver.clone());
        server.add_resolver("__Directive".to_string(), introspection_resolver);

        // Parse the address
        let socket_addr: std::net::SocketAddr = addr
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid address '{}': {}", addr, e))?;

        server.start(socket_addr).await
    }
}
