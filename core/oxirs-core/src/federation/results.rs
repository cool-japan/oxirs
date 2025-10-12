//! SPARQL Results JSON parser for federated query results

use crate::model::{Literal, NamedNode, Term, Variable};
use crate::OxirsError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, warn};

/// A single variable binding (solution)
pub type Binding = HashMap<String, Term>;

/// SPARQL Results JSON format (W3C standard)
/// See: https://www.w3.org/TR/sparql11-results-json/
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlResults {
    pub head: ResultsHead,
    pub results: ResultsBody,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultsHead {
    #[serde(default)]
    pub vars: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultsBody {
    #[serde(default)]
    pub bindings: Vec<HashMap<String, BindingValue>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distinct: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ordered: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum BindingValue {
    Uri {
        value: String,
    },
    Literal {
        value: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        datatype: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none", rename = "xml:lang")]
        lang: Option<String>,
    },
    Bnode {
        value: String,
    },
}

/// Parser for SPARQL Results JSON format
pub struct SparqlResultsParser;

impl SparqlResultsParser {
    /// Parse SPARQL Results JSON into variable bindings
    pub fn parse(json_str: &str) -> Result<Vec<Binding>, OxirsError> {
        debug!("Parsing SPARQL Results JSON");

        let results: SparqlResults = serde_json::from_str(json_str).map_err(|e| {
            OxirsError::Parse(format!("Failed to parse SPARQL Results JSON: {}", e))
        })?;

        debug!("Found {} variables", results.head.vars.len());
        debug!("Found {} bindings", results.results.bindings.len());

        let mut parsed_bindings = Vec::new();

        for (idx, binding_row) in results.results.bindings.iter().enumerate() {
            let mut parsed_row = HashMap::new();

            for (var_name, binding_value) in binding_row {
                match Self::parse_binding_value(binding_value) {
                    Ok(term) => {
                        parsed_row.insert(var_name.clone(), term);
                    }
                    Err(e) => {
                        warn!(
                            "Failed to parse binding at row {}, var {}: {}",
                            idx, var_name, e
                        );
                        // Continue parsing other bindings
                    }
                }
            }

            parsed_bindings.push(parsed_row);
        }

        debug!("Successfully parsed {} bindings", parsed_bindings.len());
        Ok(parsed_bindings)
    }

    /// Parse a single binding value into a Term
    fn parse_binding_value(value: &BindingValue) -> Result<Term, OxirsError> {
        match value {
            BindingValue::Uri { value } => {
                let node = NamedNode::new(value)
                    .map_err(|e| OxirsError::Parse(format!("Invalid IRI: {}", e)))?;
                Ok(Term::NamedNode(node))
            }
            BindingValue::Literal {
                value,
                datatype,
                lang,
            } => {
                let literal = if let Some(lang_tag) = lang {
                    // Language-tagged literal
                    Literal::new_language_tagged_literal(value, lang_tag)
                        .map_err(|e| OxirsError::Parse(format!("Invalid language tag: {}", e)))?
                } else if let Some(dt) = datatype {
                    // Typed literal
                    let datatype_node = NamedNode::new(dt)
                        .map_err(|e| OxirsError::Parse(format!("Invalid datatype IRI: {}", e)))?;
                    Literal::new_typed_literal(value, datatype_node)
                } else {
                    // Simple literal (xsd:string)
                    Literal::new_simple_literal(value)
                };
                Ok(Term::Literal(literal))
            }
            BindingValue::Bnode { value } => {
                let blank_node = crate::model::BlankNode::new(value)
                    .map_err(|e| OxirsError::Parse(format!("Invalid blank node: {}", e)))?;
                Ok(Term::BlankNode(blank_node))
            }
        }
    }

    /// Parse SPARQL Results JSON and extract variables
    pub fn parse_with_variables(
        json_str: &str,
    ) -> Result<(Vec<Variable>, Vec<Binding>), OxirsError> {
        let results: SparqlResults = serde_json::from_str(json_str).map_err(|e| {
            OxirsError::Parse(format!("Failed to parse SPARQL Results JSON: {}", e))
        })?;

        let variables: Result<Vec<Variable>, _> =
            results.head.vars.iter().map(Variable::new).collect();

        let variables = variables
            .map_err(|e| OxirsError::Parse(format!("Invalid variable name in results: {}", e)))?;

        let bindings = Self::parse(json_str)?;

        Ok((variables, bindings))
    }

    /// Check if the JSON represents an empty result set
    pub fn is_empty_result(json_str: &str) -> bool {
        if let Ok(results) = serde_json::from_str::<SparqlResults>(json_str) {
            results.results.bindings.is_empty()
        } else {
            false
        }
    }

    /// Get the number of results from JSON
    pub fn count_results(json_str: &str) -> Result<usize, OxirsError> {
        let results: SparqlResults = serde_json::from_str(json_str).map_err(|e| {
            OxirsError::Parse(format!("Failed to parse SPARQL Results JSON: {}", e))
        })?;

        Ok(results.results.bindings.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_result() {
        let json = r#"{
            "head": { "vars": ["name"] },
            "results": {
                "bindings": [
                    { "name": { "type": "literal", "value": "Alice" } },
                    { "name": { "type": "literal", "value": "Bob" } }
                ]
            }
        }"#;

        let bindings = SparqlResultsParser::parse(json).unwrap();
        assert_eq!(bindings.len(), 2);

        if let Some(Term::Literal(lit)) = bindings[0].get("name") {
            assert_eq!(lit.value(), "Alice");
        } else {
            panic!("Expected literal value");
        }
    }

    #[test]
    fn test_parse_uri_result() {
        let json = r#"{
            "head": { "vars": ["person"] },
            "results": {
                "bindings": [
                    { "person": { "type": "uri", "value": "http://example.org/alice" } }
                ]
            }
        }"#;

        let bindings = SparqlResultsParser::parse(json).unwrap();
        assert_eq!(bindings.len(), 1);

        if let Some(Term::NamedNode(node)) = bindings[0].get("person") {
            assert_eq!(node.as_str(), "http://example.org/alice");
        } else {
            panic!("Expected named node");
        }
    }

    #[test]
    fn test_parse_language_tagged_literal() {
        let json = r#"{
            "head": { "vars": ["label"] },
            "results": {
                "bindings": [
                    { "label": { "type": "literal", "value": "Hello", "xml:lang": "en" } }
                ]
            }
        }"#;

        let bindings = SparqlResultsParser::parse(json).unwrap();
        assert_eq!(bindings.len(), 1);

        if let Some(Term::Literal(lit)) = bindings[0].get("label") {
            assert_eq!(lit.value(), "Hello");
            assert_eq!(lit.language(), Some("en"));
        } else {
            panic!("Expected language-tagged literal");
        }
    }

    #[test]
    fn test_parse_typed_literal() {
        let json = r#"{
            "head": { "vars": ["age"] },
            "results": {
                "bindings": [
                    {
                        "age": {
                            "type": "literal",
                            "value": "30",
                            "datatype": "http://www.w3.org/2001/XMLSchema#integer"
                        }
                    }
                ]
            }
        }"#;

        let bindings = SparqlResultsParser::parse(json).unwrap();
        assert_eq!(bindings.len(), 1);

        if let Some(Term::Literal(lit)) = bindings[0].get("age") {
            assert_eq!(lit.value(), "30");
            assert_eq!(
                lit.datatype().as_str(),
                "http://www.w3.org/2001/XMLSchema#integer"
            );
        } else {
            panic!("Expected typed literal");
        }
    }

    #[test]
    fn test_parse_empty_result() {
        let json = r#"{
            "head": { "vars": [] },
            "results": { "bindings": [] }
        }"#;

        let bindings = SparqlResultsParser::parse(json).unwrap();
        assert_eq!(bindings.len(), 0);
        assert!(SparqlResultsParser::is_empty_result(json));
    }

    #[test]
    fn test_parse_with_variables() {
        let json = r#"{
            "head": { "vars": ["name", "age"] },
            "results": {
                "bindings": [
                    {
                        "name": { "type": "literal", "value": "Alice" },
                        "age": { "type": "literal", "value": "30" }
                    }
                ]
            }
        }"#;

        let (variables, bindings) = SparqlResultsParser::parse_with_variables(json).unwrap();
        assert_eq!(variables.len(), 2);
        assert_eq!(variables[0].name(), "name");
        assert_eq!(variables[1].name(), "age");
        assert_eq!(bindings.len(), 1);
    }

    #[test]
    fn test_count_results() {
        let json = r#"{
            "head": { "vars": ["x"] },
            "results": {
                "bindings": [
                    { "x": { "type": "literal", "value": "1" } },
                    { "x": { "type": "literal", "value": "2" } },
                    { "x": { "type": "literal", "value": "3" } }
                ]
            }
        }"#;

        let count = SparqlResultsParser::count_results(json).unwrap();
        assert_eq!(count, 3);
    }
}
