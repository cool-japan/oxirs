//! RDF to GraphQL mapping utilities

use crate::ast::{Document, Selection, SelectionSet, Field, Value};
use anyhow::{anyhow, Result};
use std::collections::HashMap;

/// Maps RDF data to GraphQL types and translates GraphQL queries to SPARQL
pub struct RdfGraphQLMapper {
    namespace_prefixes: HashMap<String, String>,
}

impl RdfGraphQLMapper {
    pub fn new() -> Self {
        let mut namespace_prefixes = HashMap::new();
        
        // Add common namespace prefixes
        namespace_prefixes.insert("rdf".to_string(), "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string());
        namespace_prefixes.insert("rdfs".to_string(), "http://www.w3.org/2000/01/rdf-schema#".to_string());
        namespace_prefixes.insert("owl".to_string(), "http://www.w3.org/2002/07/owl#".to_string());
        namespace_prefixes.insert("xsd".to_string(), "http://www.w3.org/2001/XMLSchema#".to_string());
        namespace_prefixes.insert("foaf".to_string(), "http://xmlns.com/foaf/0.1/".to_string());
        namespace_prefixes.insert("schema".to_string(), "http://schema.org/".to_string());
        
        Self {
            namespace_prefixes,
        }
    }

    pub fn add_namespace(&mut self, prefix: String, uri: String) {
        self.namespace_prefixes.insert(prefix, uri);
    }

    /// Convert a GraphQL query to SPARQL based on RDF vocabulary mapping
    pub fn graphql_to_sparql(&self, document: &Document, type_name: &str) -> Result<String> {
        // For now, implement a basic translation for the Query type
        if type_name != "Query" {
            return Err(anyhow!("Only Query type supported for now"));
        }

        let mut sparql_parts = Vec::new();
        
        // Add prefixes
        for (prefix, uri) in &self.namespace_prefixes {
            sparql_parts.push(format!("PREFIX {}: <{}>", prefix, uri));
        }
        
        // Extract the operation
        let operation = document.definitions.iter()
            .find_map(|def| match def {
                crate::ast::Definition::Operation(op) => Some(op),
                _ => None,
            })
            .ok_or_else(|| anyhow!("No operation found in document"))?;

        // Build the SELECT clause and WHERE patterns
        let (select_vars, where_patterns) = self.build_sparql_from_selection_set(&operation.selection_set)?;
        
        if select_vars.is_empty() {
            return Ok("SELECT * WHERE { ?s ?p ?o } LIMIT 10".to_string());
        }

        let select_clause = format!("SELECT {}", select_vars.join(" "));
        let where_clause = if where_patterns.is_empty() {
            "WHERE { ?s ?p ?o }".to_string()
        } else {
            format!("WHERE {{ {} }}", where_patterns.join(" . "))
        };

        sparql_parts.push(select_clause);
        sparql_parts.push(where_clause);
        sparql_parts.push("LIMIT 100".to_string()); // Default limit

        Ok(sparql_parts.join("\n"))
    }

    fn build_sparql_from_selection_set(&self, selection_set: &SelectionSet) -> Result<(Vec<String>, Vec<String>)> {
        let mut select_vars = Vec::new();
        let mut where_patterns = Vec::new();

        for selection in &selection_set.selections {
            match selection {
                Selection::Field(field) => {
                    match field.name.as_str() {
                        "hello" | "version" => {
                            // These are computed fields, no SPARQL needed
                        }
                        "triples" => {
                            // Count query
                            select_vars.push("(COUNT(*) as ?count)".to_string());
                            where_patterns.push("?s ?p ?o".to_string());
                        }
                        "subjects" => {
                            select_vars.push("DISTINCT ?s".to_string());
                            where_patterns.push("?s ?p ?o".to_string());
                            
                            // Handle limit argument
                            if let Some(limit) = self.extract_limit_from_field(field) {
                                // Limit will be added later
                            }
                        }
                        "predicates" => {
                            select_vars.push("DISTINCT ?p".to_string());
                            where_patterns.push("?s ?p ?o".to_string());
                        }
                        "objects" => {
                            select_vars.push("DISTINCT ?o".to_string());
                            where_patterns.push("?s ?p ?o".to_string());
                        }
                        "sparql" => {
                            // Raw SPARQL - handled separately
                        }
                        _ => {
                            // Default pattern for other fields
                            let var_name = format!("?{}", field.name);
                            select_vars.push(var_name.clone());
                            where_patterns.push(format!("?s {} {}", self.field_to_predicate(&field.name), var_name));
                        }
                    }
                }
                _ => {
                    // Handle fragments if needed
                }
            }
        }

        Ok((select_vars, where_patterns))
    }

    fn extract_limit_from_field(&self, field: &Field) -> Option<i64> {
        for arg in &field.arguments {
            if arg.name == "limit" {
                match &arg.value {
                    Value::IntValue(i) => return Some(*i),
                    _ => {}
                }
            }
        }
        None
    }

    fn field_to_predicate(&self, field_name: &str) -> String {
        // Simple mapping - in a real implementation, this would use vocabulary mappings
        match field_name {
            "name" => "foaf:name".to_string(),
            "email" => "foaf:mbox".to_string(),
            "knows" => "foaf:knows".to_string(),
            "label" => "rdfs:label".to_string(),
            "comment" => "rdfs:comment".to_string(),
            _ => format!(":{}", field_name), // Default to local namespace
        }
    }

    /// Create a GraphQL schema from RDF vocabulary
    pub fn rdf_to_graphql_schema(&self, vocabulary_uri: &str) -> Result<String> {
        // Placeholder implementation
        Ok(format!(r#"
type Query {{
  hello: String
  version: String
  triples: Int
  subjects(limit: Int = 10): [String!]!
  predicates(limit: Int = 10): [String!]!
  objects(limit: Int = 10): [String!]!
  sparql(query: String!): String
}}

# Generated from vocabulary: {}
"#, vocabulary_uri))
    }
}

impl Default for RdfGraphQLMapper {
    fn default() -> Self {
        Self::new()
    }
}