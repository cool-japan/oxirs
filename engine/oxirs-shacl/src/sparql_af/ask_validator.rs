//! `sh:SPARQLAskValidator` — ASK-query-based SHACL constraint validation
//!
//! An ASK validator uses a SPARQL ASK query to determine whether a focus node
//! satisfies a constraint.  The semantics are:
//!
//! - ASK returns `true`  → the node conforms (constraint satisfied)
//! - ASK returns `false` → the node violates the constraint
//!
//! This matches Apache Jena's behaviour for `sh:sparql` with ASK queries
//! and the SHACL-AF `sh:SPARQLAskValidator` extension.
//!
//! Reference: <https://www.w3.org/TR/shacl-af/#SPARQLAskValidator>

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::{AfResult, PrefixMap, SparqlAfError, SubstitutionContext};

/// A single SPARQL-AF ASK-based validator violation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparqlAskViolation {
    /// The focus node IRI or blank node label that violated the constraint
    pub focus_node: String,
    /// Violation message (may reference query variables via `{?var}` syntax)
    pub message: String,
    /// Optional SPARQL variable bindings from an associated SELECT query
    pub bindings: HashMap<String, String>,
}

impl SparqlAskViolation {
    /// Create a minimal violation with only a focus node and message
    pub fn new(focus_node: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            focus_node: focus_node.into(),
            message: message.into(),
            bindings: HashMap::new(),
        }
    }

    /// Create a violation with additional bindings
    pub fn with_bindings(mut self, bindings: HashMap<String, String>) -> Self {
        self.bindings = bindings;
        self
    }
}

/// Result of evaluating a `SparqlAskValidator` for one focus node
#[derive(Debug, Clone)]
pub struct SparqlAskResult {
    /// The focus node evaluated
    pub focus_node: String,
    /// Whether the constraint is satisfied (ASK returned true)
    pub conforms: bool,
    /// Violation details when `conforms` is false
    pub violation: Option<SparqlAskViolation>,
    /// Whether this validator was deactivated (always conforms when true)
    pub was_deactivated: bool,
}

impl SparqlAskResult {
    /// Returns `true` when the focus node satisfies the constraint
    pub fn is_satisfied(&self) -> bool {
        self.conforms
    }

    /// Returns `true` when the focus node violates the constraint
    pub fn is_violated(&self) -> bool {
        !self.conforms && !self.was_deactivated
    }
}

/// A SHACL-AF `sh:SPARQLAskValidator`
///
/// Uses a SPARQL ASK query to test each focus node.  The focus node is
/// injected as `$this` in the query body.
///
/// ## Example
///
/// ```text
/// sh:sparql [
///   a sh:SPARQLConstraint ;
///   sh:select """
///     ASK { $this ex:status "active" }
///   """ ;
///   sh:message "The node must have ex:status \"active\""
/// ] ;
/// ```
///
/// ## Parameter substitution
///
/// Beyond `$this`, arbitrary parameters can be registered via
/// `with_parameter(name, value)` and will be substituted before execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlAskValidator {
    /// The ASK query template.  Must use `$this` for the focus node.
    pub ask_query: String,

    /// Namespace prefixes to include
    pub prefixes: PrefixMap,

    /// Optional violation message template.
    /// Use `{?varname}` to embed query variable values.
    pub message: Option<String>,

    /// Whether this validator is deactivated (always conforms if true)
    pub deactivated: bool,

    /// Optional constraint IRI / label for tracing
    pub constraint_iri: Option<String>,

    /// Named parameter bindings
    pub parameters: HashMap<String, String>,
}

impl SparqlAskValidator {
    /// Create a validator from an ASK query template
    pub fn new(ask_query: impl Into<String>) -> Self {
        Self {
            ask_query: ask_query.into(),
            prefixes: PrefixMap::new(),
            message: None,
            deactivated: false,
            constraint_iri: None,
            parameters: HashMap::new(),
        }
    }

    /// Set a violation message template (builder pattern)
    pub fn with_message(mut self, msg: impl Into<String>) -> Self {
        self.message = Some(msg.into());
        self
    }

    /// Add a namespace prefix (builder pattern)
    pub fn with_prefix(mut self, prefix: impl Into<String>, iri: impl Into<String>) -> Self {
        self.prefixes.0.insert(prefix.into(), iri.into());
        self
    }

    /// Add a named parameter binding (builder pattern)
    pub fn with_parameter(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.parameters.insert(name.into(), value.into());
        self
    }

    /// Set the constraint IRI for tracing (builder pattern)
    pub fn with_iri(mut self, iri: impl Into<String>) -> Self {
        self.constraint_iri = Some(iri.into());
        self
    }

    /// Deactivate this validator (builder pattern)
    pub fn deactivated(mut self) -> Self {
        self.deactivated = true;
        self
    }

    /// Build the full ASK query with prefix declarations and variable substitutions
    pub fn build_query(&self, focus_node: &str) -> String {
        let mut ctx = SubstitutionContext::new().with_this(focus_node);
        for (name, value) in &self.parameters {
            ctx = ctx.bind(name, value);
        }

        let query_body = ctx.apply(&self.ask_query);
        let prefix_block = self.prefixes.render_declarations();

        if !prefix_block.is_empty() {
            format!("{prefix_block}\n{query_body}")
        } else {
            query_body
        }
    }

    /// Format a violation message by substituting `{?varname}` tokens
    pub fn format_message(&self, bindings: &HashMap<String, String>) -> String {
        let template = match &self.message {
            Some(t) => t.clone(),
            None => return "SPARQL ASK constraint violated".to_string(),
        };

        let mut result = template.clone();
        for (var, value) in bindings {
            let placeholder = format!("{{?{var}}}");
            result = result.replace(&placeholder, value);
        }
        result
    }

    /// Validate a single focus node using the provided executor
    pub fn validate_node(
        &self,
        focus_node: &str,
        executor: &dyn SparqlAskExecutor,
    ) -> AfResult<SparqlAskResult> {
        if self.deactivated {
            return Ok(SparqlAskResult {
                focus_node: focus_node.to_string(),
                conforms: true,
                violation: None,
                was_deactivated: true,
            });
        }

        let query = self.build_query(focus_node);
        let ask_result = executor.execute_ask(&query)?;

        if ask_result {
            Ok(SparqlAskResult {
                focus_node: focus_node.to_string(),
                conforms: true,
                violation: None,
                was_deactivated: false,
            })
        } else {
            let message = self.format_message(&HashMap::new());
            let violation = SparqlAskViolation::new(focus_node, message);
            Ok(SparqlAskResult {
                focus_node: focus_node.to_string(),
                conforms: false,
                violation: Some(violation),
                was_deactivated: false,
            })
        }
    }

    /// Validate multiple focus nodes in sequence
    pub fn validate_nodes(
        &self,
        focus_nodes: &[&str],
        executor: &dyn SparqlAskExecutor,
    ) -> AfResult<Vec<SparqlAskResult>> {
        focus_nodes
            .iter()
            .map(|node| self.validate_node(node, executor))
            .collect()
    }
}

/// Builder for `SparqlAskValidator`
///
/// Allows constructing validators with a fluent API.
#[derive(Debug, Default)]
pub struct SparqlAskValidatorBuilder {
    ask_query: String,
    prefixes: PrefixMap,
    message: Option<String>,
    constraint_iri: Option<String>,
    parameters: HashMap<String, String>,
}

impl SparqlAskValidatorBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the ASK query
    pub fn with_query(mut self, query: impl Into<String>) -> Self {
        self.ask_query = query.into();
        self
    }

    /// Add a prefix
    pub fn with_prefix(mut self, prefix: impl Into<String>, iri: impl Into<String>) -> Self {
        self.prefixes.0.insert(prefix.into(), iri.into());
        self
    }

    /// Set a message template
    pub fn with_message(mut self, msg: impl Into<String>) -> Self {
        self.message = Some(msg.into());
        self
    }

    /// Set the constraint IRI
    pub fn with_iri(mut self, iri: impl Into<String>) -> Self {
        self.constraint_iri = Some(iri.into());
        self
    }

    /// Add a parameter binding
    pub fn with_param(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.parameters.insert(name.into(), value.into());
        self
    }

    /// Build the validator
    pub fn build(self) -> AfResult<SparqlAskValidator> {
        if self.ask_query.trim().is_empty() {
            return Err(SparqlAfError::Configuration(
                "ASK query must not be empty".to_string(),
            ));
        }
        Ok(SparqlAskValidator {
            ask_query: self.ask_query,
            prefixes: self.prefixes,
            message: self.message,
            deactivated: false,
            constraint_iri: self.constraint_iri,
            parameters: self.parameters,
        })
    }
}

/// Trait for executing SPARQL ASK queries
pub trait SparqlAskExecutor: Send + Sync {
    /// Execute an ASK query and return the boolean result
    fn execute_ask(&self, query: &str) -> AfResult<bool>;
}

/// Mock ASK executor for testing
///
/// Returns `true` (conforms) for all queries unless a matching key
/// is found in the `violations` map.
#[derive(Debug, Default)]
pub struct MockAskExecutor {
    /// `query_substring` -> result (`true` = conforms, `false` = violates)
    pub responses: HashMap<String, bool>,
    /// Default result when no key matches (default: true = conforms)
    pub default_result: bool,
}

impl MockAskExecutor {
    /// Create a conforming mock (default all conforms)
    pub fn conforming() -> Self {
        Self {
            responses: HashMap::new(),
            default_result: true,
        }
    }

    /// Create a violating mock (default all violate)
    pub fn violating() -> Self {
        Self {
            responses: HashMap::new(),
            default_result: false,
        }
    }

    /// Register a response for a specific query substring
    pub fn with_response(mut self, key: impl Into<String>, conforms: bool) -> Self {
        self.responses.insert(key.into(), conforms);
        self
    }
}

impl SparqlAskExecutor for MockAskExecutor {
    fn execute_ask(&self, query: &str) -> AfResult<bool> {
        for (key, result) in &self.responses {
            if query.contains(key.as_str()) {
                return Ok(*result);
            }
        }
        Ok(self.default_result)
    }
}

/// A failing ASK executor for error-path testing
#[derive(Debug)]
pub struct FailingAskExecutor {
    message: String,
}

impl FailingAskExecutor {
    /// Create an executor that always returns an error
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl SparqlAskExecutor for FailingAskExecutor {
    fn execute_ask(&self, _query: &str) -> AfResult<bool> {
        Err(SparqlAfError::QueryExecution(self.message.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- SparqlAskViolation tests ----

    #[test]
    fn test_violation_creation() {
        let v = SparqlAskViolation::new("http://example.org/Node", "Test violation");
        assert_eq!(v.focus_node, "http://example.org/Node");
        assert_eq!(v.message, "Test violation");
        assert!(v.bindings.is_empty());
    }

    #[test]
    fn test_violation_with_bindings() {
        let mut bindings = HashMap::new();
        bindings.insert("prop".to_string(), "ex:age".to_string());
        let v = SparqlAskViolation::new("http://example.org/X", "msg").with_bindings(bindings);
        assert_eq!(v.bindings.get("prop"), Some(&"ex:age".to_string()));
    }

    // ---- SparqlAskValidator construction ----

    #[test]
    fn test_validator_creation() {
        let v = SparqlAskValidator::new("ASK { $this a ex:ValidThing }");
        assert!(!v.deactivated);
        assert!(v.message.is_none());
        assert!(v.parameters.is_empty());
    }

    #[test]
    fn test_validator_builder_pattern() {
        let v = SparqlAskValidator::new("ASK { $this a $class }")
            .with_message("Must be an instance of {?class}")
            .with_prefix("ex", "http://example.org/")
            .with_parameter("class", "<http://example.org/Person>")
            .with_iri("http://example.org/constraint/PersonType");

        assert!(v.message.is_some());
        assert!(v.prefixes.0.contains_key("ex"));
        assert_eq!(
            v.parameters.get("class"),
            Some(&"<http://example.org/Person>".to_string())
        );
        assert!(v.constraint_iri.is_some());
    }

    // ---- build_query tests ----

    #[test]
    fn test_build_query_substitutes_this() {
        let v = SparqlAskValidator::new("ASK { $this ex:status \"active\" }");
        let q = v.build_query("<http://example.org/Alice>");
        assert!(q.contains("<http://example.org/Alice>"));
        assert!(!q.contains("$this"));
    }

    #[test]
    fn test_build_query_prepends_prefixes() {
        let v = SparqlAskValidator::new("ASK { $this a ex:Person }")
            .with_prefix("ex", "http://example.org/");
        let q = v.build_query("<http://example.org/X>");
        assert!(q.starts_with("PREFIX ex: <http://example.org/>"));
    }

    #[test]
    fn test_build_query_substitutes_params() {
        let v = SparqlAskValidator::new("ASK { $this a $targetClass }")
            .with_parameter("targetClass", "<http://example.org/Employee>");
        let q = v.build_query("<http://example.org/Alice>");
        assert!(q.contains("<http://example.org/Employee>"));
        assert!(!q.contains("$targetClass"));
    }

    // ---- format_message tests ----

    #[test]
    fn test_format_message_default() {
        let v = SparqlAskValidator::new("ASK { $this ?p ?o }");
        let msg = v.format_message(&HashMap::new());
        assert_eq!(msg, "SPARQL ASK constraint violated");
    }

    #[test]
    fn test_format_message_with_template() {
        let v = SparqlAskValidator::new("ASK { $this ?p ?o }")
            .with_message("Node {?this} is missing required type");
        let mut bindings = HashMap::new();
        bindings.insert("this".to_string(), "http://example.org/Alice".to_string());
        let msg = v.format_message(&bindings);
        assert!(msg.contains("http://example.org/Alice"));
    }

    // ---- validate_node tests ----

    #[test]
    fn test_validate_node_conforms() {
        let v = SparqlAskValidator::new("ASK { $this a ex:ActivePerson }");
        let executor = MockAskExecutor::conforming();
        let result = v
            .validate_node("<http://example.org/Alice>", &executor)
            .expect("validation should succeed");
        assert!(result.conforms);
        assert!(result.violation.is_none());
        assert!(!result.was_deactivated);
    }

    #[test]
    fn test_validate_node_violates() {
        let v = SparqlAskValidator::new("ASK { $this a ex:ActivePerson }")
            .with_message("Node must be an ex:ActivePerson");
        let executor = MockAskExecutor::violating();
        let result = v
            .validate_node("<http://example.org/Bob>", &executor)
            .expect("validation should succeed");
        assert!(!result.conforms);
        assert!(result.is_violated());
        assert!(result.violation.is_some());
        let v = result.violation.expect("should succeed");
        assert!(v.message.contains("ActivePerson"));
    }

    #[test]
    fn test_validate_node_deactivated_always_conforms() {
        let v = SparqlAskValidator::new("ASK { $this ?p ?o }").deactivated();
        let executor = MockAskExecutor::violating();
        let result = v
            .validate_node("<http://example.org/X>", &executor)
            .expect("validation should succeed");
        assert!(result.conforms);
        assert!(result.was_deactivated);
    }

    #[test]
    fn test_validate_node_executor_error() {
        let v = SparqlAskValidator::new("ASK { $this ?p ?o }");
        let executor = FailingAskExecutor::new("endpoint unreachable");
        let result = v.validate_node("<http://example.org/X>", &executor);
        assert!(result.is_err());
    }

    // ---- validate_nodes batch tests ----

    #[test]
    fn test_validate_nodes_batch() {
        let v = SparqlAskValidator::new("ASK { $this a ex:Person }");
        let executor = MockAskExecutor::conforming();
        let nodes = vec!["<http://ex.org/Alice>", "<http://ex.org/Bob>"];
        let results = v.validate_nodes(&nodes, &executor).expect("should succeed");
        assert_eq!(results.len(), 2);
        for r in &results {
            assert!(r.conforms);
        }
    }

    // ---- SparqlAskValidatorBuilder tests ----

    #[test]
    fn test_builder_success() {
        let v = SparqlAskValidatorBuilder::new()
            .with_query("ASK { $this a ex:Valid }")
            .with_message("Violation message")
            .with_prefix("ex", "http://example.org/")
            .build()
            .expect("builder should succeed");
        assert!(v.message.is_some());
        assert!(v.prefixes.0.contains_key("ex"));
    }

    #[test]
    fn test_builder_empty_query_fails() {
        let result = SparqlAskValidatorBuilder::new().with_message("msg").build();
        assert!(result.is_err());
        match result {
            Err(SparqlAfError::Configuration(_)) => {}
            _ => panic!("Expected Configuration error"),
        }
    }

    // ---- Combined scenario: parameterised ASK validator ----

    #[test]
    fn test_parameterised_ask_validator() {
        let v = SparqlAskValidator::new("ASK { $this a $requiredType }")
            .with_parameter("requiredType", "<http://example.org/Person>")
            .with_message("Node {?this} is not a Person");

        let executor =
            MockAskExecutor::violating().with_response("<http://example.org/Person>", true);

        // This node's query will contain the substituted IRI and find the response
        let result = v
            .validate_node("<http://example.org/Alice>", &executor)
            .expect("should succeed");
        assert!(result.conforms);
    }
}
