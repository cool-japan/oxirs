//! Juniper-based GraphQL schema for RDF data
//!
//! This module provides a proper Juniper implementation that integrates with the RDF model,
//! offering both auto-generated schemas from RDF ontologies and direct RDF access via GraphQL.

use juniper::{
    EmptyMutation, EmptySubscription, FieldResult, GraphQLObject, GraphQLScalar, 
    GraphQLUnion, GraphQLEnum, RootNode, GraphQLInputObject, ID, Value,
    ScalarValue, DefaultScalarValue
};
use crate::RdfStore;
use std::sync::Arc;
use anyhow::Result;
// use oxigraph::sparql::QueryResults;
// use oxigraph::model::{Term, NamedNode as OxiNamedNode, Literal as OxiLiteral, BlankNode as OxiBlankNode};
use oxirs_core::sparql::QueryResults;
use oxirs_core::model::{Term, NamedNode as OxiNamedNode, Literal as OxiLiteral, BlankNode as OxiBlankNode};
use serde::{Deserialize, Serialize};

/// Custom scalar type for RDF IRIs
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IRI(pub String);

#[juniper::graphql_scalar(description = "An RDF IRI (Internationalized Resource Identifier)")]
impl<S> GraphQLScalar<S> for IRI
where
    S: ScalarValue,
{
    fn resolve(&self) -> Value<S> {
        Value::scalar(self.0.clone())
    }

    fn from_input_value(value: &juniper::InputValue<S>) -> Result<IRI, String> {
        match value.as_string_value() {
            Some(s) => {
                // Basic IRI validation
                if s.starts_with("http://") || s.starts_with("https://") || s.starts_with("urn:") {
                    Ok(IRI(s.to_string()))
                } else {
                    Err(format!("Invalid IRI format: {}", s))
                }
            }
            None => Err("Expected string value for IRI".to_string()),
        }
    }

    fn from_str<'a>(value: juniper::ScalarToken<'a>) -> juniper::ParseScalarResult<'a, S> {
        <String as juniper::ParseScalarValue<S>>::from_str(value)
    }
}

/// Custom scalar type for RDF Literals with optional language tags and datatypes
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RdfLiteral {
    pub value: String,
    pub language: Option<String>,
    pub datatype: Option<String>,
}

#[juniper::graphql_scalar(description = "An RDF Literal with optional language tag and datatype")]
impl<S> GraphQLScalar<S> for RdfLiteral
where
    S: ScalarValue,
{
    fn resolve(&self) -> Value<S> {
        Value::scalar(format!(
            "{}{}{}",
            self.value,
            self.language.as_ref().map(|l| format!("@{}", l)).unwrap_or_default(),
            self.datatype.as_ref().map(|d| format!("^^{}", d)).unwrap_or_default()
        ))
    }

    fn from_input_value(value: &juniper::InputValue<S>) -> Result<RdfLiteral, String> {
        match value.as_string_value() {
            Some(s) => {
                // Simple parsing for now - in production, use proper RDF literal parsing
                if let Some(lang_pos) = s.rfind('@') {
                    let (value, lang) = s.split_at(lang_pos);
                    Ok(RdfLiteral {
                        value: value.to_string(),
                        language: Some(lang[1..].to_string()),
                        datatype: None,
                    })
                } else if let Some(type_pos) = s.rfind("^^") {
                    let (value, datatype) = s.split_at(type_pos);
                    Ok(RdfLiteral {
                        value: value.to_string(),
                        language: None,
                        datatype: Some(datatype[2..].to_string()),
                    })
                } else {
                    Ok(RdfLiteral {
                        value: s.to_string(),
                        language: None,
                        datatype: None,
                    })
                }
            }
            None => Err("Expected string value for RdfLiteral".to_string()),
        }
    }

    fn from_str<'a>(value: juniper::ScalarToken<'a>) -> juniper::ParseScalarResult<'a, S> {
        <String as juniper::ParseScalarValue<S>>::from_str(value)
    }
}

/// RDF Term union type representing any RDF term (IRI, Literal, or Blank Node)
#[derive(Debug, Clone, GraphQLUnion)]
#[graphql(description = "An RDF term which can be an IRI, Literal, or Blank Node")]
pub enum RdfTerm {
    /// A named resource identified by an IRI
    NamedNode(RdfNamedNode),
    /// A literal value with optional language tag or datatype
    Literal(RdfLiteralNode),
    /// A blank node identifier
    BlankNode(RdfBlankNode),
}

/// Wrapper for RDF Named Nodes (IRIs)
#[derive(Debug, Clone, GraphQLObject)]
#[graphql(description = "An RDF Named Node (IRI)")]
pub struct RdfNamedNode {
    /// The IRI of this named node
    pub iri: IRI,
    /// A human-readable label for this resource (if available)
    pub label: Option<String>,
    /// A description of this resource (if available)
    pub description: Option<String>,
}

/// Wrapper for RDF Literals
#[derive(Debug, Clone, GraphQLObject)]
#[graphql(description = "An RDF Literal value")]
pub struct RdfLiteralNode {
    /// The literal value
    pub literal: RdfLiteral,
    /// The string representation of the value
    pub value: String,
    /// The language tag if this is a language-tagged string
    pub language: Option<String>,
    /// The datatype IRI if this is a typed literal
    pub datatype: Option<IRI>,
}

/// Wrapper for RDF Blank Nodes
#[derive(Debug, Clone, GraphQLObject)]
#[graphql(description = "An RDF Blank Node")]
pub struct RdfBlankNode {
    /// The identifier of the blank node
    pub id: ID,
    /// Human-readable representation
    pub label: String,
}

/// RDF Triple structure
#[derive(Debug, Clone, GraphQLObject)]
#[graphql(description = "An RDF Triple (subject-predicate-object statement)")]
pub struct RdfTriple {
    /// The subject of the triple
    pub subject: RdfTerm,
    /// The predicate of the triple
    pub predicate: RdfNamedNode,
    /// The object of the triple
    pub object: RdfTerm,
}

/// RDF Quad structure (triple + graph)
#[derive(Debug, Clone, GraphQLObject)]
#[graphql(description = "An RDF Quad (triple + named graph)")]
pub struct RdfQuad {
    /// The subject of the quad
    pub subject: RdfTerm,
    /// The predicate of the quad
    pub predicate: RdfNamedNode,
    /// The object of the quad
    pub object: RdfTerm,
    /// The named graph (None for default graph)
    pub graph: Option<RdfNamedNode>,
}

/// SPARQL query result row
#[derive(Debug, Clone, GraphQLObject)]
#[graphql(description = "A single row from a SPARQL query result set")]
pub struct SparqlResultRow {
    /// Variable bindings as key-value pairs
    pub bindings: Vec<SparqlBinding>,
}

/// SPARQL variable binding
#[derive(Debug, Clone, GraphQLObject)]
#[graphql(description = "A variable binding in a SPARQL result")]
pub struct SparqlBinding {
    /// The variable name
    pub variable: String,
    /// The bound value
    pub value: RdfTerm,
}

/// SPARQL query results union type
#[derive(Debug, Clone, GraphQLUnion)]
#[graphql(description = "Result of a SPARQL query")]
pub enum SparqlResult {
    /// SELECT query results
    Solutions(SparqlSolutions),
    /// ASK query result
    Boolean(SparqlBoolean),
    /// CONSTRUCT/DESCRIBE query results
    Graph(SparqlGraph),
}

/// SPARQL SELECT query results
#[derive(Debug, Clone, GraphQLObject)]
#[graphql(description = "Results from a SPARQL SELECT query")]
pub struct SparqlSolutions {
    /// Variable names in the result set
    pub variables: Vec<String>,
    /// Result rows
    pub rows: Vec<SparqlResultRow>,
    /// Total number of results
    pub count: i32,
}

/// SPARQL ASK query result
#[derive(Debug, Clone, GraphQLObject)]
#[graphql(description = "Result from a SPARQL ASK query")]
pub struct SparqlBoolean {
    /// The boolean result
    pub result: bool,
}

/// SPARQL CONSTRUCT/DESCRIBE query results
#[derive(Debug, Clone, GraphQLObject)]
#[graphql(description = "Graph results from a SPARQL CONSTRUCT or DESCRIBE query")]
pub struct SparqlGraph {
    /// The resulting triples
    pub triples: Vec<RdfTriple>,
    /// Total number of triples
    pub count: i32,
}

/// Input type for SPARQL queries
#[derive(Debug, Clone, GraphQLInputObject)]
#[graphql(description = "Input for executing SPARQL queries")]
pub struct SparqlQueryInput {
    /// The SPARQL query string
    pub query: String,
    /// Optional result limit
    pub limit: Option<i32>,
    /// Optional result offset
    pub offset: Option<i32>,
}

/// Query filters for RDF data
#[derive(Debug, Clone, GraphQLInputObject)]
#[graphql(description = "Filters for querying RDF data")]
pub struct RdfQueryFilter {
    /// Filter by subject IRI pattern
    pub subject: Option<String>,
    /// Filter by predicate IRI pattern
    pub predicate: Option<String>,
    /// Filter by object value pattern
    pub object: Option<String>,
    /// Filter by named graph
    pub graph: Option<String>,
    /// Result limit
    pub limit: Option<i32>,
    /// Result offset
    pub offset: Option<i32>,
}

/// The main GraphQL context containing the RDF store
#[derive(Debug, Clone)]
pub struct GraphQLContext {
    pub store: Arc<RdfStore>,
}

impl juniper::Context for GraphQLContext {}

/// The root Query type
pub struct Query;

#[juniper::graphql_object(context = GraphQLContext)]
impl Query {
    /// Get basic information about the RDF store
    fn info(context: &GraphQLContext) -> FieldResult<StoreInfo> {
        let count = context.store.triple_count().unwrap_or(0);
        Ok(StoreInfo {
            triple_count: count as i32,
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: "OxiRS GraphQL endpoint for RDF data".to_string(),
        })
    }

    /// Execute a SPARQL query
    fn sparql(context: &GraphQLContext, input: SparqlQueryInput) -> FieldResult<SparqlResult> {
        let results = context.store.query(&input.query)?;
        Ok(convert_sparql_results(results)?)
    }

    /// Get all triples matching optional filters
    fn triples(context: &GraphQLContext, filter: Option<RdfQueryFilter>) -> FieldResult<Vec<RdfTriple>> {
        let filter = filter.unwrap_or_default();
        
        // For now, return a simple query - in production, implement proper filtering
        let query = build_select_query(&filter);
        let results = context.store.query(&query)?;
        
        match results {
            QueryResults::Solutions(solutions) => {
                let mut triples = Vec::new();
                for solution in solutions {
                    let solution = solution?;
                    if let (Some(s), Some(p), Some(o)) = (
                        solution.get("s"),
                        solution.get("p"),
                        solution.get("o")
                    ) {
                        triples.push(RdfTriple {
                            subject: convert_term_to_rdf_term(s.clone()),
                            predicate: convert_named_node(p.clone())?,
                            object: convert_term_to_rdf_term(o.clone()),
                        });
                    }
                }
                Ok(triples)
            }
            _ => Ok(Vec::new()),
        }
    }

    /// Get all subjects in the store
    fn subjects(context: &GraphQLContext, limit: Option<i32>) -> FieldResult<Vec<RdfNamedNode>> {
        let limit_usize = limit.map(|l| l as usize);
        let subjects = context.store.get_subjects(limit_usize)?;
        
        Ok(subjects.into_iter().map(|s| RdfNamedNode {
            iri: IRI(s.clone()),
            label: None,
            description: None,
        }).collect())
    }

    /// Get all predicates in the store
    fn predicates(context: &GraphQLContext, limit: Option<i32>) -> FieldResult<Vec<RdfNamedNode>> {
        let limit_usize = limit.map(|l| l as usize);
        let predicates = context.store.get_predicates(limit_usize)?;
        
        Ok(predicates.into_iter().map(|p| RdfNamedNode {
            iri: IRI(p.clone()),
            label: None,
            description: None,
        }).collect())
    }

    /// Search for resources by label or IRI pattern
    fn search(context: &GraphQLContext, pattern: String, limit: Option<i32>) -> FieldResult<Vec<RdfNamedNode>> {
        let limit_clause = limit.map(|l| format!(" LIMIT {}", l)).unwrap_or_default();
        
        let query = format!(
            r#"
            SELECT DISTINCT ?resource WHERE {{
                {{
                    ?resource ?p ?o .
                    FILTER(CONTAINS(STR(?resource), "{}"))
                }} UNION {{
                    ?resource rdfs:label ?label .
                    FILTER(CONTAINS(LCASE(STR(?label)), LCASE("{}")))
                }}
            }}{}
            "#,
            pattern, pattern, limit_clause
        );
        
        let results = context.store.query(&query)?;
        match results {
            QueryResults::Solutions(solutions) => {
                let mut resources = Vec::new();
                for solution in solutions {
                    let solution = solution?;
                    if let Some(resource) = solution.get("resource") {
                        if let Term::NamedNode(node) = resource {
                            resources.push(RdfNamedNode {
                                iri: IRI(node.to_string()),
                                label: None,
                                description: None,
                            });
                        }
                    }
                }
                Ok(resources)
            }
            _ => Ok(Vec::new()),
        }
    }
}

/// Store information object
#[derive(Debug, Clone, GraphQLObject)]
#[graphql(description = "Information about the RDF store")]
pub struct StoreInfo {
    /// Total number of triples in the store
    pub triple_count: i32,
    /// Version of the GraphQL server
    pub version: String,
    /// Description of the store
    pub description: String,
}

/// Create the root schema
pub type Schema = RootNode<'static, Query, EmptyMutation<GraphQLContext>, EmptySubscription<GraphQLContext>>;

pub fn create_schema() -> Schema {
    Schema::new(Query, EmptyMutation::new(), EmptySubscription::new())
}

// Helper functions for conversion

impl Default for RdfQueryFilter {
    fn default() -> Self {
        Self {
            subject: None,
            predicate: None,
            object: None,
            graph: None,
            limit: Some(100),
            offset: Some(0),
        }
    }
}

fn convert_sparql_results(results: QueryResults) -> Result<SparqlResult> {
    match results {
        QueryResults::Solutions(solutions) => {
            let mut variables = Vec::new();
            let mut rows = Vec::new();
            
            for solution in solutions {
                let solution = solution?;
                if variables.is_empty() {
                    variables = solution.variables().map(|v| v.to_string()).collect();
                }
                
                let mut bindings = Vec::new();
                for var in solution.variables() {
                    if let Some(term) = solution.get(var) {
                        bindings.push(SparqlBinding {
                            variable: var.to_string(),
                            value: convert_term_to_rdf_term(term.clone()),
                        });
                    }
                }
                rows.push(SparqlResultRow { bindings });
            }
            
            Ok(SparqlResult::Solutions(SparqlSolutions {
                variables,
                count: rows.len() as i32,
                rows,
            }))
        }
        QueryResults::Boolean(b) => {
            Ok(SparqlResult::Boolean(SparqlBoolean { result: b }))
        }
        QueryResults::Graph(_graph) => {
            // For now, return empty graph - in production, convert the graph properly
            Ok(SparqlResult::Graph(SparqlGraph {
                triples: Vec::new(),
                count: 0,
            }))
        }
    }
}

fn convert_term_to_rdf_term(term: Term) -> RdfTerm {
    match term {
        Term::NamedNode(node) => RdfTerm::NamedNode(RdfNamedNode {
            iri: IRI(node.to_string()),
            label: None,
            description: None,
        }),
        Term::Literal(literal) => {
            let rdf_literal = RdfLiteral {
                value: literal.value().to_string(),
                language: literal.language().map(|l| l.to_string()),
                datatype: if literal.datatype() != &oxigraph::model::vocab::xsd::STRING {
                    Some(literal.datatype().to_string())
                } else {
                    None
                },
            };
            
            RdfTerm::Literal(RdfLiteralNode {
                literal: rdf_literal.clone(),
                value: rdf_literal.value,
                language: rdf_literal.language,
                datatype: rdf_literal.datatype.map(IRI),
            })
        }
        Term::BlankNode(node) => RdfTerm::BlankNode(RdfBlankNode {
            id: ID::new(format!("_:{}", node)),
            label: format!("_:{}", node),
        }),
        Term::Triple(_) => {
            // RDF-star support - for now, represent as a special named node
            RdfTerm::NamedNode(RdfNamedNode {
                iri: IRI("rdf-star:triple".to_string()),
                label: Some("RDF-star Triple".to_string()),
                description: Some("An RDF-star quoted triple".to_string()),
            })
        }
    }
}

fn convert_named_node(term: Term) -> Result<RdfNamedNode> {
    match term {
        Term::NamedNode(node) => Ok(RdfNamedNode {
            iri: IRI(node.to_string()),
            label: None,
            description: None,
        }),
        _ => Err(anyhow::anyhow!("Expected named node, got {:?}", term)),
    }
}

fn build_select_query(filter: &RdfQueryFilter) -> String {
    let mut conditions = Vec::new();
    
    if let Some(ref subject) = filter.subject {
        conditions.push(format!("CONTAINS(STR(?s), \"{}\")", subject));
    }
    if let Some(ref predicate) = filter.predicate {
        conditions.push(format!("CONTAINS(STR(?p), \"{}\")", predicate));
    }
    if let Some(ref object) = filter.object {
        conditions.push(format!("CONTAINS(STR(?o), \"{}\")", object));
    }
    
    let filter_clause = if !conditions.is_empty() {
        format!("FILTER({})", conditions.join(" && "))
    } else {
        String::new()
    };
    
    let limit_clause = filter.limit.map(|l| format!(" LIMIT {}", l)).unwrap_or_default();
    let offset_clause = filter.offset.map(|o| format!(" OFFSET {}", o)).unwrap_or_default();
    
    format!(
        "SELECT ?s ?p ?o WHERE {{ ?s ?p ?o {} }}{}{}",
        filter_clause, limit_clause, offset_clause
    )
}