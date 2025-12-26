//! SPARQL extension functions for GraphRAG queries

use crate::{GraphRAGResult, Triple};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// GraphRAG SPARQL function definitions
///
/// These functions extend SPARQL with GraphRAG capabilities:
/// - graphrag:query(text) - Execute GraphRAG query
/// - graphrag:similar(entity, threshold) - Find similar entities
/// - graphrag:expand(entity, hops) - Expand from entity
/// - graphrag:community(graph) - Detect communities
#[derive(Debug, Clone)]
pub struct GraphRAGFunctions {
    /// Function registry
    functions: HashMap<String, FunctionDef>,
}

/// Function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    /// Function name
    pub name: String,
    /// Function URI
    pub uri: String,
    /// Parameter types
    pub params: Vec<ParamDef>,
    /// Return type
    pub return_type: ReturnType,
    /// Description
    pub description: String,
}

/// Parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamDef {
    pub name: String,
    pub param_type: ParamType,
    pub required: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ParamType {
    String,
    Integer,
    Float,
    Uri,
    Boolean,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReturnType {
    Binding,
    Triple,
    Graph,
    Scalar,
}

impl Default for GraphRAGFunctions {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphRAGFunctions {
    /// Create new function registry with default GraphRAG functions
    pub fn new() -> Self {
        let mut functions = HashMap::new();

        // graphrag:query - Main GraphRAG query function
        functions.insert(
            "query".to_string(),
            FunctionDef {
                name: "query".to_string(),
                uri: "http://oxirs.io/graphrag#query".to_string(),
                params: vec![
                    ParamDef {
                        name: "text".to_string(),
                        param_type: ParamType::String,
                        required: true,
                    },
                    ParamDef {
                        name: "top_k".to_string(),
                        param_type: ParamType::Integer,
                        required: false,
                    },
                ],
                return_type: ReturnType::Graph,
                description: "Execute GraphRAG query and return relevant subgraph".to_string(),
            },
        );

        // graphrag:similar - Vector similarity search
        functions.insert(
            "similar".to_string(),
            FunctionDef {
                name: "similar".to_string(),
                uri: "http://oxirs.io/graphrag#similar".to_string(),
                params: vec![
                    ParamDef {
                        name: "entity".to_string(),
                        param_type: ParamType::Uri,
                        required: true,
                    },
                    ParamDef {
                        name: "threshold".to_string(),
                        param_type: ParamType::Float,
                        required: false,
                    },
                    ParamDef {
                        name: "k".to_string(),
                        param_type: ParamType::Integer,
                        required: false,
                    },
                ],
                return_type: ReturnType::Binding,
                description: "Find entities similar to the given entity".to_string(),
            },
        );

        // graphrag:expand - Graph expansion
        functions.insert(
            "expand".to_string(),
            FunctionDef {
                name: "expand".to_string(),
                uri: "http://oxirs.io/graphrag#expand".to_string(),
                params: vec![
                    ParamDef {
                        name: "entity".to_string(),
                        param_type: ParamType::Uri,
                        required: true,
                    },
                    ParamDef {
                        name: "hops".to_string(),
                        param_type: ParamType::Integer,
                        required: false,
                    },
                    ParamDef {
                        name: "max_triples".to_string(),
                        param_type: ParamType::Integer,
                        required: false,
                    },
                ],
                return_type: ReturnType::Graph,
                description: "Expand subgraph from entity".to_string(),
            },
        );

        // graphrag:community - Community detection
        functions.insert(
            "community".to_string(),
            FunctionDef {
                name: "community".to_string(),
                uri: "http://oxirs.io/graphrag#community".to_string(),
                params: vec![
                    ParamDef {
                        name: "graph".to_string(),
                        param_type: ParamType::Uri,
                        required: true,
                    },
                    ParamDef {
                        name: "algorithm".to_string(),
                        param_type: ParamType::String,
                        required: false,
                    },
                ],
                return_type: ReturnType::Binding,
                description: "Detect communities in graph".to_string(),
            },
        );

        // graphrag:embed - Get entity embedding
        functions.insert(
            "embed".to_string(),
            FunctionDef {
                name: "embed".to_string(),
                uri: "http://oxirs.io/graphrag#embed".to_string(),
                params: vec![ParamDef {
                    name: "entity".to_string(),
                    param_type: ParamType::Uri,
                    required: true,
                }],
                return_type: ReturnType::Scalar,
                description: "Get embedding vector for entity".to_string(),
            },
        );

        Self { functions }
    }

    /// Get function definition by name
    pub fn get(&self, name: &str) -> Option<&FunctionDef> {
        self.functions.get(name)
    }

    /// Get all function definitions
    pub fn all(&self) -> impl Iterator<Item = &FunctionDef> {
        self.functions.values()
    }

    /// Generate SPARQL SERVICE clause for GraphRAG
    pub fn generate_service_clause(&self, function: &str, args: &[&str]) -> GraphRAGResult<String> {
        let func_def = self.get(function).ok_or_else(|| {
            crate::GraphRAGError::SparqlError(format!("Unknown function: {}", function))
        })?;

        let args_str = args.join(", ");
        Ok(format!(
            "SERVICE <{}> {{ ?result graphrag:{}({}) }}",
            func_def.uri, function, args_str
        ))
    }

    /// Parse SPARQL query for GraphRAG function calls
    pub fn parse_query(&self, sparql: &str) -> Vec<FunctionCall> {
        let mut calls = Vec::new();

        // Simple regex-based parsing (full implementation would use SPARQL parser)
        let re = regex::Regex::new(r"graphrag:(\w+)\(([^)]*)\)").unwrap();

        for cap in re.captures_iter(sparql) {
            if let (Some(func), Some(args)) = (cap.get(1), cap.get(2)) {
                let func_name = func.as_str().to_string();
                let args: Vec<String> = args
                    .as_str()
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();

                if self.functions.contains_key(&func_name) {
                    calls.push(FunctionCall {
                        function: func_name,
                        arguments: args,
                    });
                }
            }
        }

        calls
    }
}

/// Parsed function call
#[derive(Debug, Clone)]
pub struct FunctionCall {
    pub function: String,
    pub arguments: Vec<String>,
}

/// GraphRAG SPARQL query builder
pub struct QueryBuilder {
    prefixes: Vec<(String, String)>,
    select_vars: Vec<String>,
    where_patterns: Vec<String>,
    graphrag_calls: Vec<String>,
    limit: Option<usize>,
    offset: Option<usize>,
}

impl Default for QueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryBuilder {
    pub fn new() -> Self {
        Self {
            prefixes: vec![
                (
                    "graphrag".to_string(),
                    "http://oxirs.io/graphrag#".to_string(),
                ),
                (
                    "rdfs".to_string(),
                    "http://www.w3.org/2000/01/rdf-schema#".to_string(),
                ),
            ],
            select_vars: Vec::new(),
            where_patterns: Vec::new(),
            graphrag_calls: Vec::new(),
            limit: None,
            offset: None,
        }
    }

    pub fn prefix(mut self, prefix: &str, uri: &str) -> Self {
        self.prefixes.push((prefix.to_string(), uri.to_string()));
        self
    }

    pub fn select(mut self, vars: &[&str]) -> Self {
        self.select_vars = vars.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn triple(mut self, subject: &str, predicate: &str, object: &str) -> Self {
        self.where_patterns
            .push(format!("{} {} {}", subject, predicate, object));
        self
    }

    pub fn graphrag_query(mut self, text: &str, result_var: &str) -> Self {
        self.graphrag_calls.push(format!(
            "BIND(graphrag:query(\"{}\") AS {})",
            text, result_var
        ));
        self
    }

    pub fn graphrag_similar(mut self, entity: &str, threshold: f32, result_var: &str) -> Self {
        self.graphrag_calls.push(format!(
            "{} graphrag:similar(\"{}\", {})",
            result_var, entity, threshold
        ));
        self
    }

    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    pub fn build(self) -> String {
        let mut query = String::new();

        // Prefixes
        for (prefix, uri) in &self.prefixes {
            query.push_str(&format!("PREFIX {}: <{}>\n", prefix, uri));
        }
        query.push('\n');

        // SELECT
        if self.select_vars.is_empty() {
            query.push_str("SELECT * ");
        } else {
            query.push_str("SELECT ");
            query.push_str(&self.select_vars.join(" "));
            query.push(' ');
        }

        // WHERE
        query.push_str("WHERE {\n");

        for pattern in &self.where_patterns {
            query.push_str(&format!("  {} .\n", pattern));
        }

        for call in &self.graphrag_calls {
            query.push_str(&format!("  {} .\n", call));
        }

        query.push_str("}\n");

        // LIMIT/OFFSET
        if let Some(limit) = self.limit {
            query.push_str(&format!("LIMIT {}\n", limit));
        }
        if let Some(offset) = self.offset {
            query.push_str(&format!("OFFSET {}\n", offset));
        }

        query
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_registry() {
        let funcs = GraphRAGFunctions::new();

        assert!(funcs.get("query").is_some());
        assert!(funcs.get("similar").is_some());
        assert!(funcs.get("expand").is_some());
        assert!(funcs.get("unknown").is_none());
    }

    #[test]
    fn test_query_parsing() {
        let funcs = GraphRAGFunctions::new();

        let sparql = r#"
            SELECT ?entity WHERE {
                ?entity graphrag:similar("battery", 0.8) .
                BIND(graphrag:query("safety issues") AS ?result)
            }
        "#;

        let calls = funcs.parse_query(sparql);

        assert_eq!(calls.len(), 2);
        assert!(calls.iter().any(|c| c.function == "similar"));
        assert!(calls.iter().any(|c| c.function == "query"));
    }

    #[test]
    fn test_query_builder() {
        let query = QueryBuilder::new()
            .select(&["?entity", "?score"])
            .graphrag_similar("http://example.org/Battery", 0.8, "?entity")
            .triple("?entity", "rdfs:label", "?label")
            .limit(10)
            .build();

        assert!(query.contains("SELECT ?entity ?score"));
        assert!(query.contains("graphrag:similar"));
        assert!(query.contains("LIMIT 10"));
    }
}
