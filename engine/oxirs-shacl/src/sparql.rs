//! SHACL-SPARQL constraint implementation
//! 
//! This module implements SPARQL-based constraints and validation logic.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use oxirs_core::{
    model::{NamedNode, Term, Triple, BlankNode, Literal, Variable, RdfTerm},
    store::Store,
    OxirsError,
};

use crate::{
    ShaclError, Result, Severity, constraints::ConstraintValidator,
    vocabulary::SHACL_PREFIXES,
};

/// SPARQL-based constraint
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparqlConstraint {
    /// SPARQL SELECT or ASK query
    pub query: String,
    
    /// Optional prefixes for the query
    pub prefixes: Option<String>,
    
    /// Custom violation message
    pub message: Option<String>,
    
    /// Severity level for violations
    pub severity: Option<Severity>,
    
    /// Optional SPARQL CONSTRUCT query for generating violation details
    pub construct_query: Option<String>,
}

impl SparqlConstraint {
    /// Create a new SPARQL constraint with a SELECT query
    pub fn select(query: String) -> Self {
        Self {
            query,
            prefixes: None,
            message: None,
            severity: None,
            construct_query: None,
        }
    }
    
    /// Create a new SPARQL constraint with an ASK query
    pub fn ask(query: String) -> Self {
        Self {
            query,
            prefixes: None,
            message: None,
            severity: None,
            construct_query: None,
        }
    }
    
    /// Set prefixes for the query
    pub fn with_prefixes(mut self, prefixes: String) -> Self {
        self.prefixes = Some(prefixes);
        self
    }
    
    /// Set custom violation message
    pub fn with_message(mut self, message: String) -> Self {
        self.message = Some(message);
        self
    }
    
    /// Set severity level
    pub fn with_severity(mut self, severity: Severity) -> Self {
        self.severity = Some(severity);
        self
    }
    
    /// Set CONSTRUCT query for violation details
    pub fn with_construct(mut self, construct_query: String) -> Self {
        self.construct_query = Some(construct_query);
        self
    }
    
    /// Get the complete query with prefixes
    pub fn get_complete_query(&self) -> String {
        let mut complete_query = String::new();
        
        // Add default SHACL prefixes
        complete_query.push_str(SHACL_PREFIXES);
        
        // Add custom prefixes if specified
        if let Some(ref prefixes) = self.prefixes {
            complete_query.push_str(prefixes);
            complete_query.push('\n');
        }
        
        // Add the main query
        complete_query.push_str(&self.query);
        
        complete_query
    }
    
    /// Check if this is an ASK query
    pub fn is_ask_query(&self) -> bool {
        self.query.trim_start().to_lowercase().starts_with("ask")
    }
    
    /// Check if this is a SELECT query
    pub fn is_select_query(&self) -> bool {
        self.query.trim_start().to_lowercase().starts_with("select")
    }
    
    /// Check if this is a CONSTRUCT query
    pub fn is_construct_query(&self) -> bool {
        self.query.trim_start().to_lowercase().starts_with("construct")
    }
    
    /// Prepare the query by substituting SHACL variables
    pub fn prepare_query(&self, bindings: &SparqlBindings) -> Result<String> {
        let mut query = self.get_complete_query();
        
        // Substitute standard SHACL variables
        if let Some(this_value) = &bindings.this {
            query = query.replace("$this", &format_term_for_sparql(this_value)?);
        }
        
        if let Some(current_shape) = &bindings.current_shape {
            query = query.replace("$currentShape", &format_term_for_sparql(current_shape)?);
        }
        
        if let Some(value) = &bindings.value {
            query = query.replace("$value", &format_term_for_sparql(value)?);
        }
        
        if let Some(path) = &bindings.path {
            query = query.replace("$PATH", path);
        }
        
        // Substitute custom bindings
        for (var, value) in &bindings.custom {
            let var_placeholder = format!("${}", var);
            query = query.replace(&var_placeholder, &format_term_for_sparql(value)?);
        }
        
        Ok(query)
    }
}

impl ConstraintValidator for SparqlConstraint {
    fn validate(&self) -> Result<()> {
        // Basic query validation
        if self.query.trim().is_empty() {
            return Err(ShaclError::ConstraintValidation(
                "SPARQL constraint query cannot be empty".to_string()
            ));
        }
        
        // Check that it's a supported query type
        if !self.is_ask_query() && !self.is_select_query() && !self.is_construct_query() {
            return Err(ShaclError::ConstraintValidation(
                "SPARQL constraint must be ASK, SELECT, or CONSTRUCT query".to_string()
            ));
        }
        
        // TODO: More thorough SPARQL syntax validation
        
        Ok(())
    }
}

/// SPARQL variable bindings for constraint evaluation
#[derive(Debug, Clone, Default)]
pub struct SparqlBindings {
    /// The focus node ($this)
    pub this: Option<Term>,
    
    /// The current shape being validated ($currentShape)
    pub current_shape: Option<Term>,
    
    /// The current value being validated ($value)
    pub value: Option<Term>,
    
    /// The property path ($PATH)
    pub path: Option<String>,
    
    /// Additional custom bindings
    pub custom: HashMap<String, Term>,
}

impl SparqlBindings {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_this(mut self, this: Term) -> Self {
        self.this = Some(this);
        self
    }
    
    pub fn with_current_shape(mut self, shape: Term) -> Self {
        self.current_shape = Some(shape);
        self
    }
    
    pub fn with_value(mut self, value: Term) -> Self {
        self.value = Some(value);
        self
    }
    
    pub fn with_path(mut self, path: String) -> Self {
        self.path = Some(path);
        self
    }
    
    pub fn with_custom_binding(mut self, name: String, value: Term) -> Self {
        self.custom.insert(name, value);
        self
    }
}

/// SPARQL constraint executor
#[derive(Debug)]
pub struct SparqlConstraintExecutor {
    /// Query timeout in milliseconds
    pub timeout_ms: Option<u64>,
    
    /// Maximum number of results for SELECT queries
    pub max_results: usize,
    
    /// Enable query optimization
    pub optimize_queries: bool,
}

impl SparqlConstraintExecutor {
    pub fn new() -> Self {
        Self {
            timeout_ms: Some(30000), // 30 seconds default
            max_results: 10000,
            optimize_queries: true,
        }
    }
    
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }
    
    pub fn with_max_results(mut self, max_results: usize) -> Self {
        self.max_results = max_results;
        self
    }
    
    pub fn with_optimization(mut self, optimize: bool) -> Self {
        self.optimize_queries = optimize;
        self
    }
    
    /// Execute a SPARQL constraint
    pub fn execute_constraint(
        &self, 
        store: &Store, 
        constraint: &SparqlConstraint, 
        bindings: &SparqlBindings,
        graph_name: Option<&str>
    ) -> Result<SparqlConstraintResult> {
        // Prepare the query with variable substitutions
        let prepared_query = constraint.prepare_query(bindings)?;
        
        // Execute the query based on its type
        if constraint.is_ask_query() {
            self.execute_ask_constraint(store, &prepared_query, graph_name)
        } else if constraint.is_select_query() {
            self.execute_select_constraint(store, &prepared_query, graph_name)
        } else {
            Err(ShaclError::SparqlExecution(
                "Unsupported SPARQL query type for constraint execution".to_string()
            ))
        }
    }
    
    /// Execute an ASK constraint
    fn execute_ask_constraint(
        &self,
        store: &Store,
        query: &str,
        graph_name: Option<&str>
    ) -> Result<SparqlConstraintResult> {
        tracing::debug!("Executing ASK constraint query: {}", query);
        
        // Wrap query in graph if needed
        let final_query = if let Some(graph) = graph_name {
            self.wrap_query_in_graph(query, graph)?
        } else {
            query.to_string()
        };
        
        // Execute the query using oxirs-core query engine
        match self.execute_sparql_query(store, &final_query) {
            Ok(result) => {
                match result {
                    oxirs_core::query::QueryResult::Ask(ask_result) => {
                        // For ASK queries in SHACL, true means constraint violation
                        Ok(SparqlConstraintResult::Ask(ask_result))
                    }
                    _ => {
                        Err(ShaclError::SparqlExecution(
                            "Expected ASK result but got different query result type".to_string()
                        ))
                    }
                }
            }
            Err(e) => {
                tracing::error!("ASK query execution failed: {}", e);
                Err(ShaclError::SparqlExecution(
                    format!("ASK query execution failed: {}", e)
                ))
            }
        }
    }
    
    /// Execute a SELECT constraint
    fn execute_select_constraint(
        &self,
        store: &Store,
        query: &str,
        graph_name: Option<&str>
    ) -> Result<SparqlConstraintResult> {
        tracing::debug!("Executing SELECT constraint query: {}", query);
        
        // Wrap query in graph if needed
        let final_query = if let Some(graph) = graph_name {
            self.wrap_query_in_graph(query, graph)?
        } else {
            query.to_string()
        };
        
        // Execute the query using oxirs-core query engine
        match self.execute_sparql_query(store, &final_query) {
            Ok(result) => {
                match result {
                    oxirs_core::query::QueryResult::Select { variables: _, bindings } => {
                        // Convert oxirs-core bindings to our format
                        let solutions: Vec<HashMap<String, Term>> = bindings.into_iter()
                            .take(self.max_results)
                            .collect();
                        
                        let truncated = solutions.len() >= self.max_results;
                        
                        Ok(SparqlConstraintResult::Select {
                            solutions,
                            truncated,
                        })
                    }
                    _ => {
                        Err(ShaclError::SparqlExecution(
                            "Expected SELECT result but got different query result type".to_string()
                        ))
                    }
                }
            }
            Err(e) => {
                tracing::error!("SELECT query execution failed: {}", e);
                Err(ShaclError::SparqlExecution(
                    format!("SELECT query execution failed: {}", e)
                ))
            }
        }
    }
    
    /// Execute a SPARQL query using oxirs-core query engine
    fn execute_sparql_query(&self, store: &Store, query: &str) -> Result<oxirs_core::query::QueryResult> {
        use oxirs_core::query::QueryEngine;
        
        let query_engine = QueryEngine::new();
        
        // Apply timeout if configured
        // TODO: Implement timeout handling once oxirs-core supports it
        
        let result = query_engine.query(query, store)
            .map_err(|e| ShaclError::SparqlExecution(format!("SPARQL query failed: {}", e)))?;
        
        Ok(result)
    }
    
    /// Wrap a query in a GRAPH clause if needed
    fn wrap_query_in_graph(&self, query: &str, graph_name: &str) -> Result<String> {
        // Simple graph wrapping for now
        // TODO: Implement more sophisticated query rewriting
        
        let query_upper = query.trim().to_uppercase();
        
        if query_upper.starts_with("ASK") {
            // For ASK queries, wrap the WHERE clause
            if let Some(where_pos) = query_upper.find("WHERE") {
                let prefix = &query[..where_pos + 5];
                let where_clause = &query[where_pos + 5..].trim();
                
                if where_clause.starts_with('{') && where_clause.ends_with('}') {
                    let inner = &where_clause[1..where_clause.len()-1];
                    return Ok(format!("{} {{ GRAPH <{}> {{ {} }} }}", prefix, graph_name, inner));
                }
            }
        } else if query_upper.starts_with("SELECT") {
            // For SELECT queries, wrap the WHERE clause
            if let Some(where_pos) = query_upper.find("WHERE") {
                let prefix = &query[..where_pos + 5];
                let where_clause = &query[where_pos + 5..].trim();
                
                if where_clause.starts_with('{') && where_clause.ends_with('}') {
                    let inner = &where_clause[1..where_clause.len()-1];
                    return Ok(format!("{} {{ GRAPH <{}> {{ {} }} }}", prefix, graph_name, inner));
                }
            }
        }
        
        // Fallback: return original query if we can't parse it
        tracing::warn!("Could not wrap query in GRAPH clause, using original: {}", query);
        Ok(query.to_string())
    }
}

impl Default for SparqlConstraintExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of SPARQL constraint execution
#[derive(Debug, Clone)]
pub enum SparqlConstraintResult {
    /// ASK query result
    Ask(bool),
    
    /// SELECT query results
    Select {
        /// Variable bindings for each solution
        solutions: Vec<HashMap<String, Term>>,
        
        /// Whether the query was truncated due to limits
        truncated: bool,
    },
}

impl SparqlConstraintResult {
    pub fn is_violation(&self) -> bool {
        match self {
            // For ASK queries, true means violation
            SparqlConstraintResult::Ask(result) => *result,
            
            // For SELECT queries, any results mean violations
            SparqlConstraintResult::Select { solutions, .. } => !solutions.is_empty(),
        }
    }
    
    pub fn violation_count(&self) -> usize {
        match self {
            SparqlConstraintResult::Ask(result) => if *result { 1 } else { 0 },
            SparqlConstraintResult::Select { solutions, .. } => solutions.len(),
        }
    }
}

/// Format a term for use in SPARQL queries
fn format_term_for_sparql(term: &Term) -> Result<String> {
    match term {
        Term::NamedNode(node) => Ok(format!("<{}>", node.as_str())),
        Term::BlankNode(node) => Ok(node.as_str().to_string()),
        Term::Literal(literal) => {
            // TODO: Proper literal formatting with datatype and language
            Ok(format!("\"{}\"", literal.as_str()))
        }
        Term::Variable(var) => Ok(format!("?{}", var.name())),
    }
}

/// SPARQL constraint validation context
#[derive(Debug, Clone)]
pub struct SparqlValidationContext {
    /// Pre-bound variables
    pub bindings: SparqlBindings,
    
    /// Query execution settings
    pub execution_settings: SparqlExecutionSettings,
    
    /// Validation metadata
    pub metadata: HashMap<String, String>,
}

impl SparqlValidationContext {
    pub fn new(bindings: SparqlBindings) -> Self {
        Self {
            bindings,
            execution_settings: SparqlExecutionSettings::default(),
            metadata: HashMap::new(),
        }
    }
    
    pub fn with_execution_settings(mut self, settings: SparqlExecutionSettings) -> Self {
        self.execution_settings = settings;
        self
    }
    
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// SPARQL execution settings
#[derive(Debug, Clone)]
pub struct SparqlExecutionSettings {
    /// Query timeout in milliseconds
    pub timeout_ms: Option<u64>,
    
    /// Maximum number of results
    pub max_results: usize,
    
    /// Enable query optimization
    pub optimize: bool,
    
    /// Enable result caching
    pub cache_results: bool,
    
    /// Graph name for queries
    pub graph_name: Option<String>,
}

impl Default for SparqlExecutionSettings {
    fn default() -> Self {
        Self {
            timeout_ms: Some(30000),
            max_results: 10000,
            optimize: true,
            cache_results: true,
            graph_name: None,
        }
    }
}

/// SPARQL constraint optimization hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlOptimizationHints {
    /// Cache query results
    pub cache_queries: bool,
    
    /// Parallelize constraint execution
    pub parallel_execution: bool,
    
    /// Maximum query complexity allowed
    pub max_query_complexity: usize,
    
    /// Query rewriting optimizations
    pub enable_rewriting: bool,
}

impl Default for SparqlOptimizationHints {
    fn default() -> Self {
        Self {
            cache_queries: true,
            parallel_execution: false,
            max_query_complexity: 1000,
            enable_rewriting: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sparql_constraint_creation() {
        let query = "ASK { $this a ex:Person }".to_string();
        let constraint = SparqlConstraint::ask(query.clone());
        
        assert_eq!(constraint.query, query);
        assert!(constraint.is_ask_query());
        assert!(!constraint.is_select_query());
        assert!(!constraint.is_construct_query());
    }
    
    #[test]
    fn test_sparql_constraint_with_prefixes() {
        let query = "SELECT ?violation WHERE { $this ex:age ?age . FILTER(?age < 0) }".to_string();
        let prefixes = "PREFIX ex: <http://example.org/>".to_string();
        
        let constraint = SparqlConstraint::select(query.clone())
            .with_prefixes(prefixes.clone());
        
        assert!(constraint.is_select_query());
        assert_eq!(constraint.prefixes, Some(prefixes));
        
        let complete_query = constraint.get_complete_query();
        assert!(complete_query.contains("PREFIX sh:"));
        assert!(complete_query.contains("PREFIX ex:"));
        assert!(complete_query.contains(&query));
    }
    
    #[test]
    fn test_sparql_bindings() {
        let this_node = Term::NamedNode(NamedNode::new("http://example.org/john").unwrap());
        let value_node = Term::NamedNode(NamedNode::new("http://example.org/age").unwrap());
        
        let bindings = SparqlBindings::new()
            .with_this(this_node.clone())
            .with_value(value_node.clone())
            .with_path("/ex:age".to_string())
            .with_custom_binding("customVar".to_string(), this_node.clone());
        
        assert_eq!(bindings.this, Some(this_node.clone()));
        assert_eq!(bindings.value, Some(value_node));
        assert_eq!(bindings.path, Some("/ex:age".to_string()));
        assert_eq!(bindings.custom.get("customVar"), Some(&this_node));
    }
    
    #[test]
    fn test_query_preparation() {
        let query = "ASK { $this ex:age $value }".to_string();
        let constraint = SparqlConstraint::ask(query);
        
        let this_node = Term::NamedNode(NamedNode::new("http://example.org/john").unwrap());
        let value_literal = Term::Literal(Literal::new("25"));
        
        let bindings = SparqlBindings::new()
            .with_this(this_node)
            .with_value(value_literal);
        
        let prepared = constraint.prepare_query(&bindings).unwrap();
        assert!(prepared.contains("<http://example.org/john>"));
        assert!(prepared.contains("\"25\""));
        assert!(!prepared.contains("$this"));
        assert!(!prepared.contains("$value"));
    }
    
    #[test]
    fn test_sparql_constraint_validation() {
        let valid_constraint = SparqlConstraint::ask("ASK { $this a ex:Person }".to_string());
        assert!(valid_constraint.validate().is_ok());
        
        let empty_constraint = SparqlConstraint::ask("".to_string());
        assert!(empty_constraint.validate().is_err());
        
        let invalid_constraint = SparqlConstraint {
            query: "INVALID QUERY TYPE { ... }".to_string(),
            prefixes: None,
            message: None,
            severity: None,
            construct_query: None,
        };
        assert!(invalid_constraint.validate().is_err());
    }
    
    #[test]
    fn test_sparql_constraint_result() {
        let ask_result = SparqlConstraintResult::Ask(true);
        assert!(ask_result.is_violation());
        assert_eq!(ask_result.violation_count(), 1);
        
        let no_violation_ask = SparqlConstraintResult::Ask(false);
        assert!(!no_violation_ask.is_violation());
        assert_eq!(no_violation_ask.violation_count(), 0);
        
        let select_result = SparqlConstraintResult::Select {
            solutions: vec![
                {
                    let mut solution = HashMap::new();
                    solution.insert("violation".to_string(), Term::NamedNode(NamedNode::new("http://example.org/v1").unwrap()));
                    solution
                }
            ],
            truncated: false,
        };
        assert!(select_result.is_violation());
        assert_eq!(select_result.violation_count(), 1);
    }
    
    #[test]
    fn test_term_formatting() {
        let named_node = Term::NamedNode(NamedNode::new("http://example.org/test").unwrap());
        let formatted = format_term_for_sparql(&named_node).unwrap();
        assert_eq!(formatted, "<http://example.org/test>");
        
        let literal = Term::Literal(Literal::new("test value"));
        let formatted = format_term_for_sparql(&literal).unwrap();
        assert_eq!(formatted, "\"test value\"");
        
        let variable = Term::Variable(Variable::new("x").unwrap());
        let formatted = format_term_for_sparql(&variable).unwrap();
        assert_eq!(formatted, "?x");
    }
}