//! SPARQL query processing and execution

use std::collections::HashMap;
use std::str::FromStr;
use crate::model::*;
use crate::store::Store;
use crate::{OxirsError, Result};

/// SPARQL query types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryType {
    /// SELECT query
    Select,
    /// CONSTRUCT query
    Construct,
    /// ASK query
    Ask,
    /// DESCRIBE query
    Describe,
}

/// Variable binding in query results
pub type Binding = HashMap<String, Term>;

/// SPARQL query parser and executor
#[derive(Debug, Clone)]
pub struct QueryEngine {
    // Configuration options
    pub strict_mode: bool,
}

impl QueryEngine {
    /// Create a new query engine
    pub fn new() -> Self {
        QueryEngine {
            strict_mode: false,
        }
    }
    
    /// Create a query engine with strict mode enabled
    pub fn with_strict_mode(mut self, strict: bool) -> Self {
        self.strict_mode = strict;
        self
    }

    /// Parse a SPARQL query string
    pub fn parse(&self, query: &str) -> Result<Query> {
        // Simple query parsing - detect query type and extract basic components
        let query_type = self.detect_query_type(query)?;
        let variables = self.extract_variables(query);
        
        Ok(Query {
            query_string: query.to_string(),
            query_type,
            variables,
        })
    }
    
    /// Execute a query against a store
    pub fn execute(&self, query: &Query, store: &Store) -> Result<QueryResult> {
        match query.query_type {
            QueryType::Select => self.execute_select(query, store),
            QueryType::Construct => self.execute_construct(query, store),
            QueryType::Ask => self.execute_ask(query, store),
            QueryType::Describe => self.execute_describe(query, store),
        }
    }
    
    /// Parse and execute a SPARQL query in one step
    pub fn query(&self, query_str: &str, store: &Store) -> Result<QueryResult> {
        let query = self.parse(query_str)?;
        self.execute(&query, store)
    }
    
    fn detect_query_type(&self, query: &str) -> Result<QueryType> {
        let query_upper = query.to_uppercase();
        
        if query_upper.contains("SELECT") {
            Ok(QueryType::Select)
        } else if query_upper.contains("CONSTRUCT") {
            Ok(QueryType::Construct)
        } else if query_upper.contains("ASK") {
            Ok(QueryType::Ask)
        } else if query_upper.contains("DESCRIBE") {
            Ok(QueryType::Describe)
        } else {
            Err(OxirsError::Query("Unknown query type".to_string()))
        }
    }
    
    fn extract_variables(&self, query: &str) -> Vec<String> {
        // Simple variable extraction - find patterns like ?var
        let mut variables = Vec::new();
        let words: Vec<&str> = query.split_whitespace().collect();
        
        for word in words {
            if word.starts_with('?') && word.len() > 1 {
                let var_name = word[1..].trim_end_matches([',', ';', '.', ')', '}'].as_ref());
                if !var_name.is_empty() && !variables.contains(&var_name.to_string()) {
                    variables.push(var_name.to_string());
                }
            }
        }
        
        variables
    }
    
    fn execute_select(&self, query: &Query, store: &Store) -> Result<QueryResult> {
        // Simplified SELECT execution - return all quads for now
        // TODO: Implement proper BGP matching and variable binding
        let quads = store.iter_quads()?;
        let bindings = self.simple_pattern_match(&query.variables, &quads)?;
        
        Ok(QueryResult::Select {
            variables: query.variables.clone(),
            bindings,
        })
    }
    
    fn execute_construct(&self, _query: &Query, _store: &Store) -> Result<QueryResult> {
        // TODO: Implement CONSTRUCT query execution
        Ok(QueryResult::Construct {
            triples: Vec::new(),
        })
    }
    
    fn execute_ask(&self, _query: &Query, store: &Store) -> Result<QueryResult> {
        // Simple ASK - return true if store is not empty
        let has_data = !store.is_empty()?;
        Ok(QueryResult::Ask(has_data))
    }
    
    fn execute_describe(&self, _query: &Query, _store: &Store) -> Result<QueryResult> {
        // TODO: Implement DESCRIBE query execution
        Ok(QueryResult::Describe {
            graph: crate::graph::Graph::new(),
        })
    }
    
    fn simple_pattern_match(&self, variables: &[String], quads: &[Quad]) -> Result<Vec<Binding>> {
        // Simplified pattern matching - just create bindings from the first few quads
        let mut bindings = Vec::new();
        
        for quad in quads.iter().take(10) { // Limit to first 10 for demo
            let mut binding = HashMap::new();
            
            // Create simple bindings
            if let Some(var) = variables.get(0) {
                binding.insert(var.clone(), Term::from(quad.subject().clone()));
            }
            if let Some(var) = variables.get(1) {
                binding.insert(var.clone(), Term::from(quad.predicate().clone()));
            }
            if let Some(var) = variables.get(2) {
                binding.insert(var.clone(), Term::from(quad.object().clone()));
            }
            
            if !binding.is_empty() {
                bindings.push(binding);
            }
        }
        
        Ok(bindings)
    }
}

impl Default for QueryEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Parsed SPARQL query
#[derive(Debug, Clone)]
pub struct Query {
    query_string: String,
    query_type: QueryType,
    variables: Vec<String>,
}

impl Query {
    /// Create a new query with the given string
    pub fn new(query: &str) -> Self {
        Query {
            query_string: query.to_string(),
            query_type: QueryType::Select, // Default
            variables: Vec::new(),
        }
    }
    
    /// Get the query string
    pub fn query_string(&self) -> &str {
        &self.query_string
    }
    
    /// Get the query type
    pub fn query_type(&self) -> &QueryType {
        &self.query_type
    }
    
    /// Get the variables in this query
    pub fn variables(&self) -> &[String] {
        &self.variables
    }
    
    /// Check if this is a SELECT query
    pub fn is_select(&self) -> bool {
        matches!(self.query_type, QueryType::Select)
    }
    
    /// Check if this is a CONSTRUCT query
    pub fn is_construct(&self) -> bool {
        matches!(self.query_type, QueryType::Construct)
    }
    
    /// Check if this is an ASK query
    pub fn is_ask(&self) -> bool {
        matches!(self.query_type, QueryType::Ask)
    }
    
    /// Check if this is a DESCRIBE query
    pub fn is_describe(&self) -> bool {
        matches!(self.query_type, QueryType::Describe)
    }
}

/// Query execution result
#[derive(Debug, Clone)]
pub enum QueryResult {
    /// SELECT query result with variable bindings
    Select {
        variables: Vec<String>,
        bindings: Vec<Binding>,
    },
    /// CONSTRUCT query result with constructed triples
    Construct {
        triples: Vec<Triple>,
    },
    /// ASK query result with boolean answer
    Ask(bool),
    /// DESCRIBE query result with description graph
    Describe {
        graph: crate::graph::Graph,
    },
}

impl QueryResult {
    /// Create a new empty SELECT result
    pub fn new() -> Self {
        QueryResult::Select {
            variables: Vec::new(),
            bindings: Vec::new(),
        }
    }
    
    /// Check if this is a SELECT result
    pub fn is_select(&self) -> bool {
        matches!(self, QueryResult::Select { .. })
    }
    
    /// Check if this is a CONSTRUCT result
    pub fn is_construct(&self) -> bool {
        matches!(self, QueryResult::Construct { .. })
    }
    
    /// Check if this is an ASK result
    pub fn is_ask(&self) -> bool {
        matches!(self, QueryResult::Ask(_))
    }
    
    /// Check if this is a DESCRIBE result
    pub fn is_describe(&self) -> bool {
        matches!(self, QueryResult::Describe { .. })
    }
    
    /// Get SELECT variables and bindings
    pub fn as_select(&self) -> Option<(&Vec<String>, &Vec<Binding>)> {
        if let QueryResult::Select { variables, bindings } = self {
            Some((variables, bindings))
        } else {
            None
        }
    }
    
    /// Get CONSTRUCT triples
    pub fn as_construct(&self) -> Option<&Vec<Triple>> {
        if let QueryResult::Construct { triples } = self {
            Some(triples)
        } else {
            None
        }
    }
    
    /// Get ASK boolean result
    pub fn as_ask(&self) -> Option<bool> {
        if let QueryResult::Ask(result) = self {
            Some(*result)
        } else {
            None
        }
    }
    
    /// Get DESCRIBE graph
    pub fn as_describe(&self) -> Option<&crate::graph::Graph> {
        if let QueryResult::Describe { graph } = self {
            Some(graph)
        } else {
            None
        }
    }
    
    /// Get the number of results
    pub fn len(&self) -> usize {
        match self {
            QueryResult::Select { bindings, .. } => bindings.len(),
            QueryResult::Construct { triples } => triples.len(),
            QueryResult::Ask(_) => 1,
            QueryResult::Describe { graph } => graph.len(),
        }
    }
    
    /// Check if the result is empty
    pub fn is_empty(&self) -> bool {
        match self {
            QueryResult::Select { bindings, .. } => bindings.is_empty(),
            QueryResult::Construct { triples } => triples.is_empty(),
            QueryResult::Ask(_) => false,
            QueryResult::Describe { graph } => graph.is_empty(),
        }
    }
}

impl Default for QueryResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper functions for creating common queries
pub mod queries {
    use super::*;
    
    /// Create a simple SELECT * query
    pub fn select_all() -> String {
        "SELECT * WHERE { ?s ?p ?o }".to_string()
    }
    
    /// Create a SELECT query for specific variables
    pub fn select_vars(vars: &[&str]) -> String {
        let var_list = vars.iter().map(|v| format!("?{}", v)).collect::<Vec<_>>().join(" ");
        format!("SELECT {} WHERE {{ ?s ?p ?o }}", var_list)
    }
    
    /// Create an ASK query to check if any triples exist
    pub fn ask_any() -> String {
        "ASK { ?s ?p ?o }".to_string()
    }
    
    /// Create a CONSTRUCT query that returns all triples
    pub fn construct_all() -> String {
        "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }".to_string()
    }
    
    /// Create a DESCRIBE query for a specific resource
    pub fn describe_resource(resource: &str) -> String {
        format!("DESCRIBE <{}>", resource)
    }
}