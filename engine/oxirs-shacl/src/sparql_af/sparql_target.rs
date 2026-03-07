//! `sh:SPARQLTarget` — SPARQL SELECT-based focus node selection
//!
//! A `sh:SPARQLTarget` uses a SPARQL SELECT query to identify focus nodes.
//! The query MUST bind `?this` for each target node.
//!
//! Reference: <https://www.w3.org/TR/shacl-af/#SPARQLTarget>

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::{AfResult, PrefixMap, SparqlRow, SubstitutionContext};

/// A single parameter binding used when evaluating a SPARQL-AF target
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterBinding {
    /// Parameter name (without `$` prefix)
    pub name: String,
    /// SPARQL-serialised value string (e.g. `<http://ex.org/>` or `"42"^^xsd:integer`)
    pub value: String,
}

impl ParameterBinding {
    /// Create a new parameter binding
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }
}

/// A SHACL-AF `sh:SPARQLTarget`
///
/// Uses a SPARQL SELECT query to identify the set of focus nodes.
/// The query result MUST include at least a `?this` variable whose bindings
/// become the focus nodes to validate.
///
/// ## Example
///
/// ```text
/// sh:target [
///   a sh:SPARQLTarget ;
///   sh:select """
///     PREFIX ex: <http://example.org/>
///     SELECT ?this WHERE { ?this a ex:Person . ?this ex:active true }
///   """
/// ]
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlAfTarget {
    /// The SPARQL SELECT query body (may or may not include PREFIX declarations).
    /// MUST produce `?this` bindings.
    pub select_query: String,

    /// Additional namespace prefixes to prepend
    pub prefixes: PrefixMap,

    /// Optional parameter bindings for parameterised targets
    pub parameters: Vec<ParameterBinding>,

    /// Whether deactivated (if true, returns empty target set)
    pub deactivated: bool,

    /// Optional human-readable label
    pub label: Option<String>,
}

/// Result of evaluating a SPARQL-AF target
#[derive(Debug, Clone)]
pub struct SparqlAfTargetResult {
    /// The resolved focus node IRIs / blank node IDs
    pub focus_nodes: Vec<String>,
    /// Number of SPARQL result rows processed
    pub result_rows: usize,
    /// Whether the target was deactivated (result is always empty if true)
    pub was_deactivated: bool,
}

impl SparqlAfTargetResult {
    /// Returns `true` when focus nodes were found
    pub fn has_nodes(&self) -> bool {
        !self.focus_nodes.is_empty()
    }

    /// Returns the count of focus nodes
    pub fn count(&self) -> usize {
        self.focus_nodes.len()
    }
}

impl SparqlAfTarget {
    /// Create a minimal SPARQL target from a SELECT query body
    pub fn new(select_query: impl Into<String>) -> Self {
        Self {
            select_query: select_query.into(),
            prefixes: PrefixMap::new(),
            parameters: Vec::new(),
            deactivated: false,
            label: None,
        }
    }

    /// Add a namespace prefix (builder pattern)
    pub fn with_prefix(mut self, prefix: impl Into<String>, iri: impl Into<String>) -> Self {
        self.prefixes.0.insert(prefix.into(), iri.into());
        self
    }

    /// Add a parameter binding (builder pattern)
    pub fn with_parameter(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.parameters.push(ParameterBinding::new(name, value));
        self
    }

    /// Set label (builder pattern)
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Deactivate target (builder pattern)
    pub fn deactivated(mut self) -> Self {
        self.deactivated = true;
        self
    }

    /// Build the full SPARQL query string with prefix declarations and parameter substitutions
    pub fn build_query(&self) -> String {
        if self.deactivated {
            return String::new();
        }

        // Build substitution context from declared parameters
        let mut ctx = SubstitutionContext::new();
        for param in &self.parameters {
            ctx = ctx.bind(&param.name, &param.value);
        }

        // Prepend prefix declarations
        let prefix_block = self.prefixes.render_declarations();
        let query_body = ctx.apply(&self.select_query);

        // If the query already starts with PREFIX declarations, don't prepend duplicates
        if !prefix_block.is_empty() && !query_body.trim_start().to_uppercase().starts_with("PREFIX")
        {
            format!("{prefix_block}\n{query_body}")
        } else if !prefix_block.is_empty() {
            // Merge: put our prefixes first and then the body (which may have its own)
            format!("{prefix_block}\n{query_body}")
        } else {
            query_body
        }
    }

    /// Evaluate the target using the provided evaluator
    pub fn evaluate(
        &self,
        evaluator: &dyn SparqlTargetEvaluator,
    ) -> AfResult<SparqlAfTargetResult> {
        if self.deactivated {
            return Ok(SparqlAfTargetResult {
                focus_nodes: Vec::new(),
                result_rows: 0,
                was_deactivated: true,
            });
        }

        let query = self.build_query();
        let rows = evaluator.execute_select(&query)?;
        let result_rows = rows.len();

        // Extract `?this` variable from each row
        let focus_nodes: Vec<String> = rows
            .into_iter()
            .filter_map(|row| row.get("this").cloned())
            .collect();

        Ok(SparqlAfTargetResult {
            focus_nodes,
            result_rows,
            was_deactivated: false,
        })
    }
}

/// Trait for executing SPARQL SELECT queries in the context of SPARQL-AF targets
pub trait SparqlTargetEvaluator: Send + Sync {
    /// Execute a SPARQL SELECT and return all result rows
    fn execute_select(&self, query: &str) -> AfResult<Vec<SparqlRow>>;
}

/// A mock SPARQL target evaluator for testing
///
/// Results are keyed by query substring.  If no key matches, returns empty.
#[derive(Debug, Default)]
pub struct SparqlTargetMock {
    /// `key_substring` -> rows to return when the query contains this substring
    pub responses: HashMap<String, Vec<SparqlRow>>,
    /// Default rows to return when no key matches (default: empty)
    pub default_rows: Vec<SparqlRow>,
}

impl SparqlTargetMock {
    /// Create an empty mock evaluator
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a mock response
    pub fn with_response(mut self, key: impl Into<String>, rows: Vec<SparqlRow>) -> Self {
        self.responses.insert(key.into(), rows);
        self
    }

    /// Set default response rows when no key matches
    pub fn with_default(mut self, rows: Vec<SparqlRow>) -> Self {
        self.default_rows = rows;
        self
    }
}

impl SparqlTargetEvaluator for SparqlTargetMock {
    fn execute_select(&self, query: &str) -> AfResult<Vec<SparqlRow>> {
        for (key, rows) in &self.responses {
            if query.contains(key.as_str()) {
                return Ok(rows.clone());
            }
        }
        Ok(self.default_rows.clone())
    }
}

/// Helper: build a single-column `?this` result row
pub fn this_row(iri: impl Into<String>) -> SparqlRow {
    let mut m = HashMap::new();
    m.insert("this".to_string(), iri.into());
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- SparqlAfTarget construction ----

    #[test]
    fn test_sparql_target_creation() {
        let target = SparqlAfTarget::new("SELECT ?this WHERE { ?this a <http://ex.org/Person> }");
        assert!(!target.deactivated);
        assert!(target.parameters.is_empty());
        assert!(target.label.is_none());
    }

    #[test]
    fn test_sparql_target_builder_pattern() {
        let target = SparqlAfTarget::new("SELECT ?this WHERE { ?this a $class }")
            .with_prefix("ex", "http://example.org/")
            .with_parameter("class", "<http://example.org/Person>")
            .with_label("PersonTarget");

        assert_eq!(target.parameters.len(), 1);
        assert_eq!(target.label, Some("PersonTarget".to_string()));
        assert!(target.prefixes.0.contains_key("ex"));
    }

    #[test]
    fn test_build_query_simple() {
        let target = SparqlAfTarget::new("SELECT ?this WHERE { ?this a <http://ex.org/P> }");
        let query = target.build_query();
        assert!(query.contains("SELECT ?this"));
        assert!(query.contains("http://ex.org/P"));
    }

    #[test]
    fn test_build_query_with_prefix() {
        let target = SparqlAfTarget::new("SELECT ?this WHERE { ?this a ex:Person }")
            .with_prefix("ex", "http://example.org/");
        let query = target.build_query();
        assert!(query.contains("PREFIX ex: <http://example.org/>"));
        assert!(query.contains("SELECT ?this"));
    }

    #[test]
    fn test_build_query_with_param_substitution() {
        let target = SparqlAfTarget::new("SELECT ?this WHERE { ?this a $targetClass }")
            .with_parameter("targetClass", "<http://example.org/Employee>");
        let query = target.build_query();
        assert!(query.contains("<http://example.org/Employee>"));
        assert!(!query.contains("$targetClass"));
    }

    #[test]
    fn test_deactivated_target_returns_empty() {
        let target = SparqlAfTarget::new("SELECT ?this WHERE { ?this ?p ?o }").deactivated();
        let evaluator =
            SparqlTargetMock::new().with_default(vec![this_row("http://example.org/Node1")]);

        let result = target
            .evaluate(&evaluator)
            .expect("evaluate should succeed");
        assert!(result.was_deactivated);
        assert!(result.focus_nodes.is_empty());
        assert_eq!(result.count(), 0);
    }

    #[test]
    fn test_evaluate_returns_focus_nodes() {
        let target = SparqlAfTarget::new("SELECT ?this WHERE { ?this a ex:Person }");
        let evaluator = SparqlTargetMock::new().with_response(
            "ex:Person",
            vec![
                this_row("http://example.org/Alice"),
                this_row("http://example.org/Bob"),
            ],
        );

        let result = target
            .evaluate(&evaluator)
            .expect("evaluate should succeed");
        assert!(!result.was_deactivated);
        assert_eq!(result.count(), 2);
        assert!(result.has_nodes());
        assert!(result
            .focus_nodes
            .contains(&"http://example.org/Alice".to_string()));
        assert!(result
            .focus_nodes
            .contains(&"http://example.org/Bob".to_string()));
    }

    #[test]
    fn test_evaluate_filters_rows_without_this() {
        let target = SparqlAfTarget::new("SELECT ?this WHERE { ?this a ex:Thing }");
        // One row has `this`, one has different key (should be filtered out)
        let mut bad_row = HashMap::new();
        bad_row.insert("node".to_string(), "http://example.org/X".to_string());

        let evaluator = SparqlTargetMock::new().with_response(
            "ex:Thing",
            vec![this_row("http://example.org/Valid"), bad_row],
        );

        let result = target
            .evaluate(&evaluator)
            .expect("evaluate should succeed");
        assert_eq!(result.result_rows, 2); // 2 rows were returned by SPARQL
        assert_eq!(result.count(), 1); // but only 1 had a `?this` binding
    }

    #[test]
    fn test_evaluate_empty_result_no_nodes() {
        let target = SparqlAfTarget::new("SELECT ?this WHERE { ?this a ex:Unknown }");
        let evaluator = SparqlTargetMock::new(); // returns empty by default

        let result = target
            .evaluate(&evaluator)
            .expect("evaluate should succeed");
        assert!(!result.has_nodes());
        assert_eq!(result.count(), 0);
        assert_eq!(result.result_rows, 0);
    }

    #[test]
    fn test_parameter_binding_struct() {
        let binding = ParameterBinding::new("class", "<http://example.org/Person>");
        assert_eq!(binding.name, "class");
        assert_eq!(binding.value, "<http://example.org/Person>");
    }
}
