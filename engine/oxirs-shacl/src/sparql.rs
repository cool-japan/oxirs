//! SHACL-SPARQL constraint implementation
//!
//! This module implements SPARQL-based constraints and validation logic.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use oxirs_core::{
    model::{BlankNode, Literal, NamedNode, RdfTerm, Term, Triple, Variable},
    OxirsError, Store,
};

use crate::{
    constraints::{
        ConstraintContext, ConstraintEvaluationResult, ConstraintEvaluator, ConstraintValidator,
    },
    optimization::{AdvancedConstraintEvaluator, ConstraintCache},
    security::{
        QueryExecutionSandbox, RecursionMonitor, SecurityAnalysisResult, SecurityConfig,
        SparqlSecurityAnalyzer,
    },
    vocabulary::SHACL_PREFIXES,
    Result, Severity, ShaclError, ShapeId,
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
        self.query
            .trim_start()
            .to_lowercase()
            .starts_with("construct")
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
                "SPARQL constraint query cannot be empty".to_string(),
            ));
        }

        // Check that it's a supported query type
        if !self.is_ask_query() && !self.is_select_query() && !self.is_construct_query() {
            return Err(ShaclError::ConstraintValidation(
                "SPARQL constraint must be ASK, SELECT, or CONSTRUCT query".to_string(),
            ));
        }

        // Additional SPARQL syntax validation
        let query_text = self.query.trim().to_lowercase();

        // Check for basic query structure (relaxed validation)
        if self.is_select_query() {
            if !query_text.contains("select") {
                return Err(ShaclError::ConstraintValidation(
                    "SELECT query must contain SELECT clause".to_string(),
                ));
            }
        } else if self.is_ask_query() {
            if !query_text.contains("ask") {
                return Err(ShaclError::ConstraintValidation(
                    "ASK query must contain ASK clause".to_string(),
                ));
            }
        } else if self.is_construct_query() {
            if !query_text.contains("construct") {
                return Err(ShaclError::ConstraintValidation(
                    "CONSTRUCT query must contain CONSTRUCT clause".to_string(),
                ));
            }
        }

        // Check for balanced braces
        let open_braces = query_text.matches('{').count();
        let close_braces = query_text.matches('}').count();
        if open_braces != close_braces {
            return Err(ShaclError::ConstraintValidation(
                "Query has unbalanced braces".to_string(),
            ));
        }

        // Check for dangerous operations (basic security check)
        let dangerous_keywords = ["drop", "clear", "insert", "delete", "load", "create"];
        for keyword in &dangerous_keywords {
            if query_text.contains(keyword) {
                return Err(ShaclError::ConstraintValidation(format!(
                    "Query contains potentially dangerous keyword: {}",
                    keyword
                )));
            }
        }

        Ok(())
    }
}

impl ConstraintEvaluator for SparqlConstraint {
    fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        // Create the SPARQL executor
        let executor = SparqlConstraintExecutor::new();

        // Create bindings from the context
        let mut bindings = SparqlBindings::new().with_this(context.focus_node.clone());

        // Add value binding if we have values to check
        if context.values.len() == 1 {
            bindings = bindings.with_value(context.values[0].clone());
        }

        // Add path binding if available
        if let Some(path) = &context.path {
            // Convert property path to SPARQL path string
            // For now, only handle simple predicate paths
            if let crate::PropertyPath::Predicate(pred) = path {
                bindings = bindings.with_path(format!("<{}>", pred.as_str()));
            }
        }

        // Add current shape binding
        bindings = bindings.with_current_shape(Term::NamedNode(
            NamedNode::new(context.shape_id.as_str()).unwrap(),
        ));

        // Note: Graph name would need to be passed separately as it's not in ConstraintContext
        let graph_name: Option<&str> = None;

        // Execute the constraint
        let result = executor.execute_constraint(store, self, &bindings, graph_name)?;

        // Convert SPARQL result to constraint evaluation result
        if result.is_violation() {
            let message = self.message.clone().unwrap_or_else(|| {
                format!(
                    "SPARQL constraint violated: {} violations found",
                    result.violation_count()
                )
            });

            match result {
                SparqlConstraintResult::Ask(_) => {
                    Ok(ConstraintEvaluationResult::violated(None, Some(message)))
                }
                SparqlConstraintResult::Select { solutions, .. } => {
                    // For SELECT queries, we can provide more detailed information
                    // Use the first solution's ?value binding if available
                    let violating_value =
                        solutions.first().and_then(|sol| sol.get("value")).cloned();

                    Ok(ConstraintEvaluationResult::violated(
                        violating_value,
                        Some(message),
                    ))
                }
            }
        } else {
            Ok(ConstraintEvaluationResult::satisfied())
        }
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
        graph_name: Option<&str>,
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
                "Unsupported SPARQL query type for constraint execution".to_string(),
            ))
        }
    }

    /// Execute an ASK constraint
    fn execute_ask_constraint(
        &self,
        store: &Store,
        query: &str,
        graph_name: Option<&str>,
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
                    _ => Err(ShaclError::SparqlExecution(
                        "Expected ASK result but got different query result type".to_string(),
                    )),
                }
            }
            Err(e) => {
                tracing::error!("ASK query execution failed: {}", e);
                Err(ShaclError::SparqlExecution(format!(
                    "ASK query execution failed: {}",
                    e
                )))
            }
        }
    }

    /// Execute a SELECT constraint
    fn execute_select_constraint(
        &self,
        store: &Store,
        query: &str,
        graph_name: Option<&str>,
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
                    oxirs_core::query::QueryResult::Select {
                        variables: _,
                        bindings,
                    } => {
                        // Convert oxirs-core bindings to our format
                        let solutions: Vec<HashMap<String, Term>> =
                            bindings.into_iter().take(self.max_results).collect();

                        let truncated = solutions.len() >= self.max_results;

                        Ok(SparqlConstraintResult::Select {
                            solutions,
                            truncated,
                        })
                    }
                    _ => Err(ShaclError::SparqlExecution(
                        "Expected SELECT result but got different query result type".to_string(),
                    )),
                }
            }
            Err(e) => {
                tracing::error!("SELECT query execution failed: {}", e);
                Err(ShaclError::SparqlExecution(format!(
                    "SELECT query execution failed: {}",
                    e
                )))
            }
        }
    }

    /// Execute a SPARQL query using oxirs-core query engine
    fn execute_sparql_query(
        &self,
        store: &Store,
        query: &str,
    ) -> Result<oxirs_core::query::QueryResult> {
        use oxirs_core::query::QueryEngine;

        let query_engine = QueryEngine::new();

        // Apply timeout if configured
        // TODO: Implement timeout handling once oxirs-core supports it

        let result = query_engine
            .query(query, store)
            .map_err(|e| ShaclError::SparqlExecution(format!("SPARQL query failed: {}", e)))?;

        Ok(result)
    }

    /// Wrap a query in a GRAPH clause if needed
    fn wrap_query_in_graph(&self, query: &str, graph_name: &str) -> Result<String> {
        let query_trimmed = query.trim();
        let query_upper = query_trimmed.to_uppercase();

        // More sophisticated query rewriting
        if query_upper.starts_with("ASK") {
            self.wrap_ask_query_in_graph(query_trimmed, graph_name)
        } else if query_upper.starts_with("SELECT") {
            self.wrap_select_query_in_graph(query_trimmed, graph_name)
        } else if query_upper.starts_with("CONSTRUCT") {
            self.wrap_construct_query_in_graph(query_trimmed, graph_name)
        } else {
            Err(ShaclError::SparqlExecution(
                "Unsupported query type for graph wrapping".to_string(),
            ))
        }
    }

    /// Wrap ASK query in GRAPH clause
    fn wrap_ask_query_in_graph(&self, query: &str, graph_name: &str) -> Result<String> {
        if let Some(where_pos) = query.to_uppercase().find("WHERE") {
            let prefix = &query[..where_pos + 5];
            let where_clause = query[where_pos + 5..].trim();

            if where_clause.starts_with('{') && where_clause.ends_with('}') {
                let inner = &where_clause[1..where_clause.len() - 1];
                return Ok(format!(
                    "{} {{ GRAPH <{}> {{ {} }} }}",
                    prefix, graph_name, inner
                ));
            }
        }

        Err(ShaclError::SparqlExecution(
            "Invalid ASK query structure for graph wrapping".to_string(),
        ))
    }

    /// Wrap SELECT query in GRAPH clause
    fn wrap_select_query_in_graph(&self, query: &str, graph_name: &str) -> Result<String> {
        if let Some(where_pos) = query.to_uppercase().find("WHERE") {
            let prefix = &query[..where_pos + 5];
            let where_clause = query[where_pos + 5..].trim();

            if where_clause.starts_with('{') && where_clause.ends_with('}') {
                let inner = &where_clause[1..where_clause.len() - 1];
                return Ok(format!(
                    "{} {{ GRAPH <{}> {{ {} }} }}",
                    prefix, graph_name, inner
                ));
            }
        }

        Err(ShaclError::SparqlExecution(
            "Invalid SELECT query structure for graph wrapping".to_string(),
        ))
    }

    /// Wrap CONSTRUCT query in GRAPH clause
    fn wrap_construct_query_in_graph(&self, query: &str, graph_name: &str) -> Result<String> {
        if let Some(where_pos) = query.to_uppercase().find("WHERE") {
            let prefix = &query[..where_pos + 5];
            let where_clause = query[where_pos + 5..].trim();

            if where_clause.starts_with('{') && where_clause.ends_with('}') {
                let inner = &where_clause[1..where_clause.len() - 1];
                return Ok(format!(
                    "{} {{ GRAPH <{}> {{ {} }} }}",
                    prefix, graph_name, inner
                ));
            }
        }

        Err(ShaclError::SparqlExecution(
            "Invalid CONSTRUCT query structure for graph wrapping".to_string(),
        ))
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
            SparqlConstraintResult::Ask(result) => {
                if *result {
                    1
                } else {
                    0
                }
            }
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
            let value = literal.as_str();
            let escaped_value = value.replace("\\", "\\\\").replace("\"", "\\\"");

            // Check for language tag
            if let Some(language) = literal.language() {
                Ok(format!("\"{}\"@{}", escaped_value, language))
            }
            // Check for datatype
            else if let datatype = literal.datatype() {
                let datatype_iri = datatype.as_str();
                // Only add datatype if it's not xsd:string (which is the default)
                if datatype_iri != "http://www.w3.org/2001/XMLSchema#string" {
                    Ok(format!("\"{}\"^^<{}>", escaped_value, datatype_iri))
                } else {
                    Ok(format!("\"{}\"", escaped_value))
                }
            } else {
                Ok(format!("\"{}\"", escaped_value))
            }
        }
        Term::Variable(var) => Ok(format!("?{}", var.name())),
        Term::QuotedTriple(_) => Err(ShaclError::SparqlExecution(
            "Quoted triples not supported in SPARQL constraints".to_string(),
        )),
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

/// SPARQL constraint component definition for custom constraints
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparqlConstraintComponent {
    /// Component identifier
    pub component_id: String,
    /// SPARQL query template
    pub query_template: String,
    /// Required parameters for this component
    pub parameters: Vec<SparqlParameter>,
    /// Optional label
    pub label: Option<String>,
    /// Optional description
    pub description: Option<String>,
    /// Validator query (ASK query to validate parameter values)
    pub validator: Option<String>,
    /// Library this component belongs to
    pub library: Option<String>,
    /// Reusable templates and snippets
    pub templates: HashMap<String, String>,
}

impl SparqlConstraintComponent {
    /// Create a new SPARQL constraint component
    pub fn new(component_id: String, query_template: String) -> Self {
        Self {
            component_id,
            query_template,
            parameters: Vec::new(),
            label: None,
            description: None,
            validator: None,
            library: None,
            templates: HashMap::new(),
        }
    }

    /// Add a parameter to this component
    pub fn with_parameter(mut self, parameter: SparqlParameter) -> Self {
        self.parameters.push(parameter);
        self
    }

    /// Set the validator query
    pub fn with_validator(mut self, validator: String) -> Self {
        self.validator = Some(validator);
        self
    }

    /// Set the library
    pub fn with_library(mut self, library: String) -> Self {
        self.library = Some(library);
        self
    }

    /// Add a template
    pub fn with_template(mut self, name: String, template: String) -> Self {
        self.templates.insert(name, template);
        self
    }

    /// Instantiate this component with parameter values
    pub fn instantiate(
        &self,
        parameter_values: &HashMap<String, Term>,
    ) -> Result<SparqlConstraint> {
        // Validate parameters
        self.validate_parameters(parameter_values)?;

        // Replace parameters in query template
        let mut instantiated_query = self.query_template.clone();
        for parameter in &self.parameters {
            if let Some(value) = parameter_values.get(&parameter.name) {
                let placeholder = format!("${}", parameter.name);
                let formatted_value = format_term_for_sparql(value)?;
                instantiated_query = instantiated_query.replace(&placeholder, &formatted_value);
            } else if !parameter.optional {
                return Err(ShaclError::ConstraintValidation(format!(
                    "Required parameter '{}' not provided for SPARQL component '{}'",
                    parameter.name, self.component_id
                )));
            }
        }

        // Create constraint with instantiated query
        let mut constraint = if instantiated_query.trim().to_lowercase().starts_with("ask") {
            SparqlConstraint::ask(instantiated_query)
        } else {
            SparqlConstraint::select(instantiated_query)
        };

        // Add metadata
        if let Some(label) = &self.label {
            constraint = constraint.with_message(label.clone());
        }

        Ok(constraint)
    }

    /// Validate parameter values against component requirements
    fn validate_parameters(&self, parameter_values: &HashMap<String, Term>) -> Result<()> {
        for parameter in &self.parameters {
            if let Some(value) = parameter_values.get(&parameter.name) {
                // Validate parameter type if specified
                if let Some(expected_type) = &parameter.parameter_type {
                    if !self.validate_parameter_type(value, expected_type)? {
                        return Err(ShaclError::ConstraintValidation(format!(
                            "Parameter '{}' value {:?} does not match expected type '{}'",
                            parameter.name, value, expected_type
                        )));
                    }
                }
            } else if !parameter.optional {
                return Err(ShaclError::ConstraintValidation(format!(
                    "Required parameter '{}' not provided",
                    parameter.name
                )));
            }
        }
        Ok(())
    }

    /// Validate parameter type
    fn validate_parameter_type(&self, value: &Term, expected_type: &str) -> Result<bool> {
        match expected_type {
            "NamedNode" | "IRI" => Ok(matches!(value, Term::NamedNode(_))),
            "Literal" => Ok(matches!(value, Term::Literal(_))),
            "BlankNode" => Ok(matches!(value, Term::BlankNode(_))),
            "Variable" => Ok(matches!(value, Term::Variable(_))),
            _ => {
                // Custom type validation could be added here
                Ok(true)
            }
        }
    }
}

/// Parameter definition for SPARQL constraint components
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparqlParameter {
    /// Parameter name
    pub name: String,
    /// Parameter description
    pub description: Option<String>,
    /// Whether this parameter is optional
    pub optional: bool,
    /// Expected parameter type
    pub parameter_type: Option<String>,
    /// Default value if optional
    pub default_value: Option<Term>,
}

impl SparqlParameter {
    pub fn new(name: String) -> Self {
        Self {
            name,
            description: None,
            optional: false,
            parameter_type: None,
            default_value: None,
        }
    }

    pub fn optional(mut self) -> Self {
        self.optional = true;
        self
    }

    pub fn with_type(mut self, parameter_type: String) -> Self {
        self.parameter_type = Some(parameter_type);
        self
    }

    pub fn with_default(mut self, default_value: Term) -> Self {
        self.default_value = Some(default_value);
        self
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }
}

/// SPARQL constraint library containing reusable components
#[derive(Debug, Clone)]
pub struct SparqlConstraintLibrary {
    /// Library identifier
    pub library_id: String,
    /// Library metadata
    pub metadata: SparqlLibraryMetadata,
    /// Constraint components in this library
    pub components: HashMap<String, SparqlConstraintComponent>,
    /// Common prefixes used by this library
    pub prefixes: HashMap<String, String>,
    /// Common SPARQL functions and templates
    pub functions: HashMap<String, String>,
}

impl SparqlConstraintLibrary {
    pub fn new(library_id: String, metadata: SparqlLibraryMetadata) -> Self {
        Self {
            library_id,
            metadata,
            components: HashMap::new(),
            prefixes: HashMap::new(),
            functions: HashMap::new(),
        }
    }

    /// Add a constraint component to the library
    pub fn add_component(&mut self, component: SparqlConstraintComponent) {
        self.components
            .insert(component.component_id.clone(), component);
    }

    /// Get a component by ID
    pub fn get_component(&self, component_id: &str) -> Option<&SparqlConstraintComponent> {
        self.components.get(component_id)
    }

    /// Add a prefix
    pub fn add_prefix(&mut self, prefix: String, namespace: String) {
        self.prefixes.insert(prefix, namespace);
    }

    /// Add a function
    pub fn add_function(&mut self, function_name: String, function_body: String) {
        self.functions.insert(function_name, function_body);
    }

    /// Get prefixes as SPARQL prefix declarations
    pub fn get_prefix_declarations(&self) -> String {
        let mut declarations = String::new();
        for (prefix, namespace) in &self.prefixes {
            declarations.push_str(&format!("PREFIX {}: <{}>\n", prefix, namespace));
        }
        declarations
    }
}

/// Metadata for SPARQL constraint library
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlLibraryMetadata {
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    pub author: Option<String>,
    pub license: Option<String>,
    pub dependencies: Vec<String>,
}

/// Registry for managing SPARQL constraint components and libraries
#[derive(Debug)]
pub struct SparqlConstraintRegistry {
    /// Registered libraries
    libraries: HashMap<String, SparqlConstraintLibrary>,
    /// Component index for quick lookup
    component_index: HashMap<String, String>, // component_id -> library_id
    /// Performance optimizer
    optimizer: AdvancedConstraintEvaluator,
    /// Constraint cache
    constraint_cache: HashMap<String, SparqlConstraint>,
}

impl SparqlConstraintRegistry {
    pub fn new() -> Self {
        let cache = ConstraintCache::default();
        let optimizer = AdvancedConstraintEvaluator::new(cache, true, 100, true);

        Self {
            libraries: HashMap::new(),
            component_index: HashMap::new(),
            optimizer,
            constraint_cache: HashMap::new(),
        }
    }

    /// Register a library
    pub fn register_library(&mut self, library: SparqlConstraintLibrary) -> Result<()> {
        let library_id = library.library_id.clone();

        // Update component index
        for component_id in library.components.keys() {
            if self.component_index.contains_key(component_id) {
                return Err(ShaclError::ConstraintValidation(format!(
                    "Component '{}' already registered",
                    component_id
                )));
            }
            self.component_index
                .insert(component_id.clone(), library_id.clone());
        }

        self.libraries.insert(library_id, library);
        Ok(())
    }

    /// Get a component by ID
    pub fn get_component(&self, component_id: &str) -> Option<&SparqlConstraintComponent> {
        if let Some(library_id) = self.component_index.get(component_id) {
            if let Some(library) = self.libraries.get(library_id) {
                return library.get_component(component_id);
            }
        }
        None
    }

    /// Get a library by ID
    pub fn get_library(&self, library_id: &str) -> Option<&SparqlConstraintLibrary> {
        self.libraries.get(library_id)
    }

    /// Create and cache a constraint from a component
    pub fn create_constraint(
        &mut self,
        component_id: &str,
        parameter_values: &HashMap<String, Term>,
    ) -> Result<SparqlConstraint> {
        // Create cache key
        let cache_key = format!(
            "{}:{}",
            component_id,
            self.hash_parameters(parameter_values)
        );

        // Check cache first
        if let Some(cached_constraint) = self.constraint_cache.get(&cache_key) {
            return Ok(cached_constraint.clone());
        }

        // Get component and instantiate
        let component = self.get_component(component_id).ok_or_else(|| {
            ShaclError::ConstraintValidation(format!("Component '{}' not found", component_id))
        })?;

        let constraint = component.instantiate(parameter_values)?;

        // Cache the result
        self.constraint_cache.insert(cache_key, constraint.clone());

        Ok(constraint)
    }

    /// Hash parameter values for caching
    fn hash_parameters(&self, parameters: &HashMap<String, Term>) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        let mut sorted_params: Vec<_> = parameters.iter().collect();
        sorted_params.sort_by_key(|(k, _)| *k);

        for (key, value) in sorted_params {
            key.hash(&mut hasher);
            format!("{:?}", value).hash(&mut hasher);
        }

        hasher.finish()
    }

    /// List all available components
    pub fn list_components(&self) -> Vec<(&str, &SparqlConstraintComponent)> {
        let mut components = Vec::new();
        for library in self.libraries.values() {
            for (id, component) in &library.components {
                components.push((id.as_str(), component));
            }
        }
        components
    }

    /// Get all libraries
    pub fn list_libraries(&self) -> Vec<&SparqlConstraintLibrary> {
        self.libraries.values().collect()
    }

    /// Clear constraint cache
    pub fn clear_cache(&mut self) {
        self.constraint_cache.clear();
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> (usize, usize) {
        (
            self.constraint_cache.len(),
            self.constraint_cache.capacity(),
        )
    }
}

impl Default for SparqlConstraintRegistry {
    fn default() -> Self {
        let mut registry = Self::new();

        // Register standard SHACL-SPARQL library
        if let Ok(std_library) = create_standard_sparql_library() {
            let _ = registry.register_library(std_library);
        }

        registry
    }
}

/// Create standard SHACL-SPARQL constraint library
pub fn create_standard_sparql_library() -> Result<SparqlConstraintLibrary> {
    let metadata = SparqlLibraryMetadata {
        name: "Standard SHACL-SPARQL".to_string(),
        version: "1.0.0".to_string(),
        description: Some("Standard SHACL-SPARQL constraint components".to_string()),
        author: Some("OxiRS SHACL Engine".to_string()),
        license: Some("MIT".to_string()),
        dependencies: vec![],
    };

    let mut library = SparqlConstraintLibrary::new("std".to_string(), metadata);

    // Add standard prefixes
    library.add_prefix("sh".to_string(), "http://www.w3.org/ns/shacl#".to_string());
    library.add_prefix(
        "rdf".to_string(),
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
    );
    library.add_prefix(
        "rdfs".to_string(),
        "http://www.w3.org/2000/01/rdf-schema#".to_string(),
    );
    library.add_prefix(
        "xsd".to_string(),
        "http://www.w3.org/2001/XMLSchema#".to_string(),
    );

    // Add example constraint components

    // 1. Custom range validation component
    let range_component = SparqlConstraintComponent::new(
        "rangeValidation".to_string(),
        "SELECT $this WHERE { $this $predicate ?value . FILTER(?value < $minValue || ?value > $maxValue) }".to_string()
    )
    .with_parameter(SparqlParameter::new("predicate".to_string()).with_type("NamedNode".to_string()))
    .with_parameter(SparqlParameter::new("minValue".to_string()).with_type("Literal".to_string()))
    .with_parameter(SparqlParameter::new("maxValue".to_string()).with_type("Literal".to_string()))
    .with_library("std".to_string());

    library.add_component(range_component);

    // 2. Custom pattern matching component
    let pattern_component = SparqlConstraintComponent::new(
        "customPattern".to_string(),
        "SELECT $this WHERE { $this $predicate ?value . FILTER(!REGEX(STR(?value), $pattern, $flags)) }".to_string()
    )
    .with_parameter(SparqlParameter::new("predicate".to_string()).with_type("NamedNode".to_string()))
    .with_parameter(SparqlParameter::new("pattern".to_string()).with_type("Literal".to_string()))
    .with_parameter(SparqlParameter::new("flags".to_string()).with_type("Literal".to_string()).optional())
    .with_library("std".to_string());

    library.add_component(pattern_component);

    // 3. Custom uniqueness validation component
    let uniqueness_component = SparqlConstraintComponent::new(
        "uniquenessValidation".to_string(),
        "SELECT $this WHERE { $this $predicate ?value . ?other $predicate ?value . FILTER($this != ?other) }".to_string()
    )
    .with_parameter(SparqlParameter::new("predicate".to_string()).with_type("NamedNode".to_string()))
    .with_library("std".to_string());

    library.add_component(uniqueness_component);

    // 4. Custom dependency validation component
    let dependency_component = SparqlConstraintComponent::new(
        "dependencyValidation".to_string(),
        "SELECT $this WHERE { $this $sourcePredicate ?sourceValue . FILTER NOT EXISTS { $this $targetPredicate ?targetValue } }".to_string()
    )
    .with_parameter(SparqlParameter::new("sourcePredicate".to_string()).with_type("NamedNode".to_string()))
    .with_parameter(SparqlParameter::new("targetPredicate".to_string()).with_type("NamedNode".to_string()))
    .with_library("std".to_string());

    library.add_component(dependency_component);

    Ok(library)
}

/// Enhanced SPARQL constraint executor with library support
#[derive(Debug)]
pub struct EnhancedSparqlConstraintExecutor {
    /// Base executor
    base_executor: SparqlConstraintExecutor,
    /// Constraint registry
    registry: SparqlConstraintRegistry,
    /// Query optimization enabled
    optimize_queries: bool,
    /// Query result caching
    cache_results: bool,
}

impl EnhancedSparqlConstraintExecutor {
    pub fn new() -> Self {
        Self {
            base_executor: SparqlConstraintExecutor::new(),
            registry: SparqlConstraintRegistry::default(),
            optimize_queries: true,
            cache_results: true,
        }
    }

    pub fn with_registry(mut self, registry: SparqlConstraintRegistry) -> Self {
        self.registry = registry;
        self
    }

    /// Execute a constraint using component from registry
    pub fn execute_component_constraint(
        &mut self,
        store: &Store,
        component_id: &str,
        parameter_values: &HashMap<String, Term>,
        bindings: &SparqlBindings,
        graph_name: Option<&str>,
    ) -> Result<SparqlConstraintResult> {
        // Create constraint from component
        let constraint = self
            .registry
            .create_constraint(component_id, parameter_values)?;

        // Execute using base executor
        self.base_executor
            .execute_constraint(store, &constraint, bindings, graph_name)
    }

    /// Register a new library
    pub fn register_library(&mut self, library: SparqlConstraintLibrary) -> Result<()> {
        self.registry.register_library(library)
    }

    /// Get available components
    pub fn list_components(&self) -> Vec<(&str, &SparqlConstraintComponent)> {
        self.registry.list_components()
    }

    /// Clear caches
    pub fn clear_caches(&mut self) {
        self.registry.clear_cache();
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> SparqlExecutorStats {
        let (cache_size, cache_capacity) = self.registry.get_cache_stats();
        SparqlExecutorStats {
            cache_size,
            cache_capacity,
            libraries_count: self.registry.libraries.len(),
            components_count: self.registry.component_index.len(),
        }
    }
}

impl Default for EnhancedSparqlConstraintExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for SPARQL executor performance
#[derive(Debug, Clone)]
pub struct SparqlExecutorStats {
    pub cache_size: usize,
    pub cache_capacity: usize,
    pub libraries_count: usize,
    pub components_count: usize,
}

/// SPARQL constraint target for custom target selection
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparqlTarget {
    /// SPARQL SELECT query to find target nodes
    pub select_query: String,
    /// Optional prefixes
    pub prefixes: Option<String>,
    /// Target label
    pub label: Option<String>,
}

impl SparqlTarget {
    pub fn new(select_query: String) -> Self {
        Self {
            select_query,
            prefixes: None,
            label: None,
        }
    }

    pub fn with_prefixes(mut self, prefixes: String) -> Self {
        self.prefixes = Some(prefixes);
        self
    }

    pub fn with_label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }

    /// Execute target selection query
    pub fn execute_target_selection(&self, store: &Store) -> Result<Vec<Term>> {
        let executor = SparqlConstraintExecutor::new();

        // Prepare complete query
        let mut complete_query = String::new();
        complete_query.push_str(SHACL_PREFIXES);
        if let Some(prefixes) = &self.prefixes {
            complete_query.push_str(prefixes);
            complete_query.push('\n');
        }
        complete_query.push_str(&self.select_query);

        // Execute query
        match executor.execute_sparql_query(store, &complete_query)? {
            oxirs_core::query::QueryResult::Select { bindings, .. } => {
                let mut targets = Vec::new();
                for binding in bindings {
                    // Look for ?this variable or first variable
                    if let Some(term) = binding.get("this").or_else(|| binding.values().next()) {
                        targets.push(term.clone());
                    }
                }
                Ok(targets)
            }
            _ => Err(ShaclError::SparqlExecution(
                "Target selection query must be a SELECT query".to_string(),
            )),
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

        let constraint = SparqlConstraint::select(query.clone()).with_prefixes(prefixes.clone());

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
            solutions: vec![{
                let mut solution = HashMap::new();
                solution.insert(
                    "violation".to_string(),
                    Term::NamedNode(NamedNode::new("http://example.org/v1").unwrap()),
                );
                solution
            }],
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

/// Secure SPARQL constraint evaluator with comprehensive security features
#[derive(Debug)]
pub struct SecureSparqlConstraintExecutor {
    /// Base executor
    base_executor: SparqlConstraintExecutor,
    /// Security analyzer
    security_analyzer: SparqlSecurityAnalyzer,
    /// Security configuration
    security_config: SecurityConfig,
    /// Query execution sandbox
    execution_sandbox: Option<QueryExecutionSandbox>,
    /// Recursion monitor
    recursion_monitor: RecursionMonitor,
    /// Execution statistics
    execution_stats: SecureExecutionStats,
}

impl SecureSparqlConstraintExecutor {
    /// Create a new secure SPARQL constraint executor
    pub fn new(security_config: SecurityConfig) -> Result<Self> {
        let security_analyzer = SparqlSecurityAnalyzer::new(security_config.clone())?;
        let recursion_monitor = RecursionMonitor::new(security_config.max_recursion_depth);

        Ok(Self {
            base_executor: SparqlConstraintExecutor::new(),
            security_analyzer,
            security_config: security_config.clone(),
            execution_sandbox: None,
            recursion_monitor,
            execution_stats: SecureExecutionStats::default(),
        })
    }

    /// Create with default security configuration
    pub fn with_default_security() -> Result<Self> {
        Self::new(SecurityConfig::default())
    }

    /// Execute a SPARQL constraint with full security validation
    pub fn execute_secure_constraint(
        &mut self,
        store: &Store,
        constraint: &SparqlConstraint,
        bindings: &SparqlBindings,
        graph_name: Option<&str>,
        shape_id: &str,
    ) -> Result<SecureSparqlConstraintResult> {
        // Track recursion for this shape
        self.recursion_monitor.enter_shape(shape_id)?;

        // Prepare the query with bindings
        let prepared_query = constraint.prepare_query(bindings)?;

        // Perform security analysis
        let security_analysis = self.security_analyzer.analyze_query(&prepared_query)?;

        // Check if query is safe to execute
        if !security_analysis.is_safe {
            self.execution_stats.security_violations += 1;

            if self.security_config.enable_security_logging {
                tracing::warn!(
                    "Security violation in SPARQL constraint for shape {}: {} violations found",
                    shape_id,
                    security_analysis.violations.len()
                );

                for violation in &security_analysis.violations {
                    tracing::warn!("Security violation: {}", violation.message);
                }
            }

            return Err(ShaclError::SecurityViolation(format!(
                "Query failed security validation: {} violations",
                security_analysis.violations.len()
            )));
        }

        // Sanitize the query
        let sanitized_query = self.security_analyzer.sanitize_query(&prepared_query)?;

        // Create execution sandbox
        let mut sandbox = QueryExecutionSandbox::new(self.security_config.clone());
        sandbox.start_execution()?;

        // Execute the constraint with monitoring
        let start_time = std::time::Instant::now();
        let result = self.execute_with_monitoring(
            store,
            constraint,
            &sanitized_query,
            bindings,
            graph_name,
            &mut sandbox,
        );

        // Stop sandbox and collect stats
        let execution_stats = sandbox.stop_execution()?;

        // Update execution statistics
        self.execution_stats.total_executions += 1;
        self.execution_stats.total_execution_time += start_time.elapsed();
        self.execution_stats.total_memory_used += execution_stats.memory_used;
        self.execution_stats.average_complexity_score =
            (self.execution_stats.average_complexity_score
                * (self.execution_stats.total_executions - 1) as f64
                + security_analysis.complexity_score)
                / self.execution_stats.total_executions as f64;

        // Exit recursion tracking
        self.recursion_monitor.exit_shape(shape_id);

        match result {
            Ok(constraint_result) => {
                self.execution_stats.successful_executions += 1;

                Ok(SecureSparqlConstraintResult {
                    constraint_result,
                    security_analysis,
                    execution_stats,
                    sanitized_query: Some(sanitized_query),
                })
            }
            Err(e) => {
                self.execution_stats.failed_executions += 1;
                Err(e)
            }
        }
    }

    /// Execute constraint with sandbox monitoring
    fn execute_with_monitoring(
        &mut self,
        store: &Store,
        constraint: &SparqlConstraint,
        sanitized_query: &str,
        bindings: &SparqlBindings,
        graph_name: Option<&str>,
        sandbox: &mut QueryExecutionSandbox,
    ) -> Result<SparqlConstraintResult> {
        // Create a modified constraint with the sanitized query
        let secure_constraint = SparqlConstraint {
            query: sanitized_query.to_string(),
            prefixes: constraint.prefixes.clone(),
            message: constraint.message.clone(),
            severity: constraint.severity.clone(),
            construct_query: constraint.construct_query.clone(),
        };

        // Monitor execution periodically
        let result = if constraint.is_ask_query() {
            self.execute_ask_with_monitoring(
                store,
                &secure_constraint,
                bindings,
                graph_name,
                sandbox,
            )
        } else if constraint.is_select_query() {
            self.execute_select_with_monitoring(
                store,
                &secure_constraint,
                bindings,
                graph_name,
                sandbox,
            )
        } else {
            Err(ShaclError::SparqlExecution(
                "Unsupported query type for secure execution".to_string(),
            ))
        };

        result
    }

    /// Execute ASK query with monitoring
    fn execute_ask_with_monitoring(
        &mut self,
        store: &Store,
        constraint: &SparqlConstraint,
        bindings: &SparqlBindings,
        graph_name: Option<&str>,
        sandbox: &mut QueryExecutionSandbox,
    ) -> Result<SparqlConstraintResult> {
        // Check execution limits before starting
        sandbox.check_execution_limits()?;

        // Execute the ASK query
        let result = self
            .base_executor
            .execute_constraint(store, constraint, bindings, graph_name)?;

        // Record result and check limits
        sandbox.record_result()?;

        Ok(result)
    }

    /// Execute SELECT query with monitoring
    fn execute_select_with_monitoring(
        &mut self,
        store: &Store,
        constraint: &SparqlConstraint,
        bindings: &SparqlBindings,
        graph_name: Option<&str>,
        sandbox: &mut QueryExecutionSandbox,
    ) -> Result<SparqlConstraintResult> {
        // Check execution limits before starting
        sandbox.check_execution_limits()?;

        // Execute the SELECT query
        let result = self
            .base_executor
            .execute_constraint(store, constraint, bindings, graph_name)?;

        // Check result count and record
        match &result {
            SparqlConstraintResult::Select { solutions, .. } => {
                for _ in solutions {
                    sandbox.record_result()?;
                    sandbox.check_execution_limits()?;
                }
            }
            SparqlConstraintResult::Ask(_) => {
                sandbox.record_result()?;
            }
        }

        Ok(result)
    }

    /// Get execution statistics
    pub fn get_execution_stats(&self) -> &SecureExecutionStats {
        &self.execution_stats
    }

    /// Reset execution statistics
    pub fn reset_stats(&mut self) {
        self.execution_stats = SecureExecutionStats::default();
    }

    /// Update security configuration
    pub fn update_security_config(&mut self, config: SecurityConfig) -> Result<()> {
        self.security_analyzer = SparqlSecurityAnalyzer::new(config.clone())?;
        self.recursion_monitor = RecursionMonitor::new(config.max_recursion_depth);
        self.security_config = config;
        Ok(())
    }

    /// Get current recursion depth for a shape
    pub fn get_recursion_depth(&self, shape_id: &str) -> usize {
        self.recursion_monitor.current_depth(shape_id)
    }

    /// Validate a query without executing it
    pub fn validate_query_security(&self, query: &str) -> Result<SecurityAnalysisResult> {
        self.security_analyzer.analyze_query(query)
    }

    /// Pre-validate a constraint before execution
    pub fn pre_validate_constraint(
        &self,
        constraint: &SparqlConstraint,
        bindings: &SparqlBindings,
    ) -> Result<SecurityAnalysisResult> {
        let prepared_query = constraint.prepare_query(bindings)?;
        self.security_analyzer.analyze_query(&prepared_query)
    }
}

/// Result of secure SPARQL constraint execution
#[derive(Debug, Clone)]
pub struct SecureSparqlConstraintResult {
    /// The actual constraint result
    pub constraint_result: SparqlConstraintResult,
    /// Security analysis performed
    pub security_analysis: SecurityAnalysisResult,
    /// Execution statistics
    pub execution_stats: crate::security::ExecutionStats,
    /// Sanitized query that was executed
    pub sanitized_query: Option<String>,
}

impl SecureSparqlConstraintResult {
    /// Check if the constraint was violated
    pub fn is_violation(&self) -> bool {
        self.constraint_result.is_violation()
    }

    /// Get violation count
    pub fn violation_count(&self) -> usize {
        self.constraint_result.violation_count()
    }

    /// Check if security analysis found issues
    pub fn has_security_issues(&self) -> bool {
        !self.security_analysis.is_safe
    }

    /// Get security violations
    pub fn get_security_violations(&self) -> &[crate::security::SecurityViolation] {
        &self.security_analysis.violations
    }
}

/// Execution statistics for secure SPARQL executor
#[derive(Debug, Clone, Default)]
pub struct SecureExecutionStats {
    /// Total number of executions
    pub total_executions: usize,
    /// Successful executions
    pub successful_executions: usize,
    /// Failed executions
    pub failed_executions: usize,
    /// Security violations detected
    pub security_violations: usize,
    /// Total execution time
    pub total_execution_time: std::time::Duration,
    /// Total memory used
    pub total_memory_used: usize,
    /// Average query complexity score
    pub average_complexity_score: f64,
}

impl SecureExecutionStats {
    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_executions == 0 {
            0.0
        } else {
            self.successful_executions as f64 / self.total_executions as f64
        }
    }

    /// Calculate average execution time
    pub fn average_execution_time(&self) -> std::time::Duration {
        if self.total_executions == 0 {
            std::time::Duration::from_secs(0)
        } else {
            self.total_execution_time / self.total_executions as u32
        }
    }

    /// Calculate average memory usage
    pub fn average_memory_usage(&self) -> usize {
        if self.total_executions == 0 {
            0
        } else {
            self.total_memory_used / self.total_executions
        }
    }

    /// Security violation rate
    pub fn security_violation_rate(&self) -> f64 {
        if self.total_executions == 0 {
            0.0
        } else {
            self.security_violations as f64 / self.total_executions as f64
        }
    }
}

#[cfg(test)]
mod secure_tests {
    use super::*;

    #[test]
    fn test_secure_executor_creation() {
        let config = SecurityConfig::default();
        let executor = SecureSparqlConstraintExecutor::new(config);
        assert!(executor.is_ok());
    }

    #[test]
    fn test_query_security_validation() {
        let executor = SecureSparqlConstraintExecutor::with_default_security().unwrap();

        // Test safe query
        let safe_query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
        let analysis = executor.validate_query_security(safe_query).unwrap();
        assert!(analysis.is_safe);

        // Test potentially dangerous query
        let dangerous_query = "DROP GRAPH <http://example.org/graph>";
        let analysis = executor.validate_query_security(dangerous_query).unwrap();
        assert!(!analysis.is_safe);
    }

    #[test]
    fn test_constraint_pre_validation() {
        let executor = SecureSparqlConstraintExecutor::with_default_security().unwrap();

        let constraint = SparqlConstraint::ask("ASK { $this a ?type }".to_string());
        let bindings = SparqlBindings::new().with_this(Term::NamedNode(
            NamedNode::new("http://example.org/test").unwrap(),
        ));

        let analysis = executor
            .pre_validate_constraint(&constraint, &bindings)
            .unwrap();
        assert!(analysis.is_safe);
    }

    #[test]
    fn test_execution_stats() {
        let mut stats = SecureExecutionStats::default();
        stats.total_executions = 10;
        stats.successful_executions = 8;
        stats.failed_executions = 2;
        stats.security_violations = 1;
        stats.total_execution_time = std::time::Duration::from_millis(1000);
        stats.total_memory_used = 1024 * 1024;

        assert_eq!(stats.success_rate(), 0.8);
        assert_eq!(
            stats.average_execution_time(),
            std::time::Duration::from_millis(100)
        );
        assert_eq!(stats.average_memory_usage(), 1024 * 1024 / 10);
        assert_eq!(stats.security_violation_rate(), 0.1);
    }

    #[test]
    fn test_recursion_monitoring() {
        let config = SecurityConfig {
            max_recursion_depth: 3,
            ..SecurityConfig::default()
        };
        let mut executor = SecureSparqlConstraintExecutor::new(config).unwrap();

        assert_eq!(executor.get_recursion_depth("shape1"), 0);

        // This would be called internally during execution
        assert!(executor.recursion_monitor.enter_shape("shape1").is_ok());
        assert_eq!(executor.get_recursion_depth("shape1"), 1);

        assert!(executor.recursion_monitor.enter_shape("shape1").is_ok());
        assert_eq!(executor.get_recursion_depth("shape1"), 2);

        assert!(executor.recursion_monitor.enter_shape("shape1").is_ok());
        assert_eq!(executor.get_recursion_depth("shape1"), 3);

        // Should fail on exceeding depth
        assert!(executor.recursion_monitor.enter_shape("shape1").is_err());
    }
}
