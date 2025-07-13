//! SPARQL Query Results Format Support
//!
//! This module provides support for various SPARQL query result formats
//! including JSON, XML, CSV, and TSV as specified by W3C standards.

use crate::algebra::{Binding, Solution, Term, Variable};
use anyhow::{anyhow, Result};
use serde_json::{json, Map, Value as JsonValue};
use std::io::Write;

/// SPARQL Query Result Types
#[derive(Debug, Clone)]
pub enum QueryResult {
    /// Boolean result (for ASK queries)
    Boolean(bool),
    /// Variable bindings (for SELECT queries)
    Bindings {
        variables: Vec<Variable>,
        solutions: Vec<Binding>,
    },
    /// RDF graph (for CONSTRUCT/DESCRIBE queries)
    Graph(Vec<crate::algebra::TriplePattern>),
}

/// Result format types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultFormat {
    /// SPARQL Results JSON Format
    Json,
    /// SPARQL Results XML Format
    Xml,
    /// CSV format
    Csv,
    /// TSV format
    Tsv,
    /// Binary format for efficiency
    Binary,
}

impl ResultFormat {
    /// Get the MIME type for this format
    pub fn mime_type(&self) -> &'static str {
        match self {
            ResultFormat::Json => "application/sparql-results+json",
            ResultFormat::Xml => "application/sparql-results+xml",
            ResultFormat::Csv => "text/csv",
            ResultFormat::Tsv => "text/tab-separated-values",
            ResultFormat::Binary => "application/octet-stream",
        }
    }

    /// Get the file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            ResultFormat::Json => "srj",
            ResultFormat::Xml => "srx",
            ResultFormat::Csv => "csv",
            ResultFormat::Tsv => "tsv",
            ResultFormat::Binary => "bin",
        }
    }
}

/// SPARQL Results JSON Format Serializer
pub struct JsonResultSerializer;

impl JsonResultSerializer {
    /// Serialize query result to SPARQL Results JSON format
    pub fn serialize<W: Write>(result: &QueryResult, writer: &mut W) -> Result<()> {
        let json = Self::to_json(result)?;
        serde_json::to_writer_pretty(writer, &json)?;
        Ok(())
    }

    /// Convert query result to JSON Value
    pub fn to_json(result: &QueryResult) -> Result<JsonValue> {
        match result {
            QueryResult::Boolean(value) => Ok(json!({
                "head": {},
                "boolean": value
            })),
            QueryResult::Bindings {
                variables,
                solutions,
            } => {
                let vars: Vec<String> = variables.iter().map(|v| v.to_string()).collect();
                let bindings: Vec<JsonValue> = solutions
                    .iter()
                    .map(Self::binding_to_json)
                    .collect::<Result<Vec<_>>>()?;

                Ok(json!({
                    "head": {
                        "vars": vars
                    },
                    "results": {
                        "bindings": bindings
                    }
                }))
            }
            QueryResult::Graph(_triples) => {
                // For CONSTRUCT/DESCRIBE queries, we return the triples in a simplified format
                // In practice, these would typically be serialized as RDF, not SPARQL Results JSON
                Err(anyhow!(
                    "Graph results should be serialized as RDF, not SPARQL Results JSON"
                ))
            }
        }
    }

    /// Convert a single binding to JSON
    fn binding_to_json(binding: &Binding) -> Result<JsonValue> {
        let mut obj = Map::new();

        for (var, term) in binding {
            let term_json = Self::term_to_json(term)?;
            obj.insert(var.to_string(), term_json);
        }

        Ok(JsonValue::Object(obj))
    }

    /// Convert a term to JSON representation
    fn term_to_json(term: &Term) -> Result<JsonValue> {
        match term {
            Term::Iri(iri) => Ok(json!({
                "type": "uri",
                "value": iri.to_string()
            })),
            Term::Literal(lit) => {
                let mut obj = Map::new();
                obj.insert("type".to_string(), JsonValue::String("literal".to_string()));
                obj.insert("value".to_string(), JsonValue::String(lit.value.clone()));

                if let Some(lang) = &lit.language {
                    obj.insert("xml:lang".to_string(), JsonValue::String(lang.clone()));
                } else if let Some(datatype) = &lit.datatype {
                    obj.insert(
                        "datatype".to_string(),
                        JsonValue::String(datatype.to_string()),
                    );
                }

                Ok(JsonValue::Object(obj))
            }
            Term::BlankNode(id) => Ok(json!({
                "type": "bnode",
                "value": id
            })),
            Term::Variable(_var) => Err(anyhow!("Variables should not appear in result bindings")),
            Term::QuotedTriple(triple) => {
                // For RDF-star support - represent as a special literal
                Ok(json!({
                    "type": "triple",
                    "value": format!("<<{} {} {}>>", triple.subject, triple.predicate, triple.object)
                }))
            }
            Term::PropertyPath(_path) => Err(anyhow!(
                "Property paths should not appear in result bindings"
            )),
        }
    }
}

/// CSV Result Serializer
pub struct CsvResultSerializer;

impl CsvResultSerializer {
    /// Serialize query result to CSV format
    pub fn serialize<W: Write>(result: &QueryResult, writer: &mut W) -> Result<()> {
        match result {
            QueryResult::Boolean(value) => {
                writeln!(writer, "result")?;
                writeln!(writer, "{value}")?;
                Ok(())
            }
            QueryResult::Bindings {
                variables,
                solutions,
            } => {
                // Write header
                let header: Vec<String> = variables.iter().map(|v| v.to_string()).collect();
                writeln!(writer, "{}", header.join(","))?;

                // Write data rows
                for binding in solutions {
                    let row: Vec<String> = variables
                        .iter()
                        .map(|var| {
                            binding
                                .get(var)
                                .map(Self::term_to_csv_value)
                                .unwrap_or_else(|| "".to_string())
                        })
                        .collect();
                    writeln!(writer, "{}", row.join(","))?;
                }
                Ok(())
            }
            QueryResult::Graph(_triples) => {
                Err(anyhow!("Graph results cannot be serialized as CSV"))
            }
        }
    }

    /// Convert term to CSV-safe value
    fn term_to_csv_value(term: &Term) -> String {
        match term {
            Term::Iri(iri) => iri.to_string(),
            Term::Literal(lit) => {
                if lit.language.is_some() || lit.datatype.is_some() {
                    format!("\"{}\"", lit.value.replace('"', "\"\""))
                } else {
                    lit.value.clone()
                }
            }
            Term::BlankNode(id) => format!("_:{id}"),
            Term::Variable(var) => format!("?{var}"),
            Term::QuotedTriple(triple) => {
                format!(
                    "\"<<{} {} {}>>\"",
                    triple.subject, triple.predicate, triple.object
                )
            }
            Term::PropertyPath(path) => format!("\"{path}\""),
        }
    }
}

/// TSV Result Serializer  
pub struct TsvResultSerializer;

impl TsvResultSerializer {
    /// Serialize query result to TSV format
    pub fn serialize<W: Write>(result: &QueryResult, writer: &mut W) -> Result<()> {
        match result {
            QueryResult::Boolean(value) => {
                writeln!(writer, "result")?;
                writeln!(writer, "{value}")?;
                Ok(())
            }
            QueryResult::Bindings {
                variables,
                solutions,
            } => {
                // Write header
                let header: Vec<String> = variables.iter().map(|v| v.to_string()).collect();
                writeln!(writer, "{}", header.join("\t"))?;

                // Write data rows
                for binding in solutions {
                    let row: Vec<String> = variables
                        .iter()
                        .map(|var| {
                            binding
                                .get(var)
                                .map(Self::term_to_tsv_value)
                                .unwrap_or_else(|| "".to_string())
                        })
                        .collect();
                    writeln!(writer, "{}", row.join("\t"))?;
                }
                Ok(())
            }
            QueryResult::Graph(_triples) => {
                Err(anyhow!("Graph results cannot be serialized as TSV"))
            }
        }
    }

    /// Convert term to TSV-safe value
    fn term_to_tsv_value(term: &Term) -> String {
        match term {
            Term::Iri(iri) => iri.to_string(),
            Term::Literal(lit) => lit.value.replace(['\t', '\n'], " "),
            Term::BlankNode(id) => format!("_:{id}"),
            Term::Variable(var) => format!("?{var}"),
            Term::QuotedTriple(triple) => format!(
                "<<{} {} {}>>",
                triple.subject, triple.predicate, triple.object
            )
            .replace(['\t', '\n'], " "),
            Term::PropertyPath(path) => path.to_string().replace(['\t', '\n'], " "),
        }
    }
}

/// Generic result serializer that can handle multiple formats
pub struct ResultSerializer;

impl ResultSerializer {
    /// Serialize result in the specified format
    pub fn serialize<W: Write>(
        result: &QueryResult,
        format: ResultFormat,
        writer: &mut W,
    ) -> Result<()> {
        match format {
            ResultFormat::Json => JsonResultSerializer::serialize(result, writer),
            ResultFormat::Csv => CsvResultSerializer::serialize(result, writer),
            ResultFormat::Tsv => TsvResultSerializer::serialize(result, writer),
            ResultFormat::Xml => {
                // TODO: Implement XML serializer
                Err(anyhow!("XML format not yet implemented"))
            }
            ResultFormat::Binary => {
                // TODO: Implement binary format
                Err(anyhow!("Binary format not yet implemented"))
            }
        }
    }

    /// Get appropriate format from MIME type
    pub fn format_from_mime_type(mime_type: &str) -> Option<ResultFormat> {
        match mime_type {
            "application/sparql-results+json" | "application/json" => Some(ResultFormat::Json),
            "application/sparql-results+xml" | "application/xml" => Some(ResultFormat::Xml),
            "text/csv" => Some(ResultFormat::Csv),
            "text/tab-separated-values" | "text/tsv" => Some(ResultFormat::Tsv),
            "application/octet-stream" => Some(ResultFormat::Binary),
            _ => None,
        }
    }

    /// Get appropriate format from file extension
    pub fn format_from_extension(ext: &str) -> Option<ResultFormat> {
        match ext.to_lowercase().as_str() {
            "json" | "srj" => Some(ResultFormat::Json),
            "xml" | "srx" => Some(ResultFormat::Xml),
            "csv" => Some(ResultFormat::Csv),
            "tsv" => Some(ResultFormat::Tsv),
            "bin" => Some(ResultFormat::Binary),
            _ => None,
        }
    }
}

/// Utility functions for working with Solutions
pub fn solution_to_query_result(solution: Solution, variables: Vec<Variable>) -> QueryResult {
    QueryResult::Bindings {
        variables,
        solutions: solution,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::{NamedNode, Variable};
    use std::collections::HashMap;

    fn create_test_variable(name: &str) -> Variable {
        Variable::new(name).unwrap()
    }

    fn create_test_iri(iri: &str) -> NamedNode {
        NamedNode::new(iri).unwrap()
    }

    #[test]
    fn test_json_boolean_result() {
        let result = QueryResult::Boolean(true);
        let json = JsonResultSerializer::to_json(&result).unwrap();

        assert_eq!(json["boolean"], JsonValue::Bool(true));
        assert!(json["head"].is_object());
    }

    #[test]
    fn test_json_bindings_result() {
        let var1 = create_test_variable("x");
        let var2 = create_test_variable("y");
        let variables = vec![var1.clone(), var2.clone()];

        let mut binding = HashMap::new();
        binding.insert(var1, Term::Iri(create_test_iri("http://example.org/alice")));
        binding.insert(
            var2,
            Term::Literal(crate::algebra::Literal {
                value: "Alice".to_string(),
                language: Some("en".to_string()),
                datatype: None,
            }),
        );

        let result = QueryResult::Bindings {
            variables,
            solutions: vec![binding],
        };

        let json = JsonResultSerializer::to_json(&result).unwrap();

        assert!(json["head"]["vars"].is_array());
        assert!(json["results"]["bindings"].is_array());
        assert_eq!(json["results"]["bindings"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_csv_serialization() {
        let var1 = create_test_variable("name");
        let variables = vec![var1.clone()];

        let mut binding = HashMap::new();
        binding.insert(
            var1,
            Term::Literal(crate::algebra::Literal {
                value: "Alice".to_string(),
                language: None,
                datatype: None,
            }),
        );

        let result = QueryResult::Bindings {
            variables,
            solutions: vec![binding],
        };

        let mut output = Vec::new();
        CsvResultSerializer::serialize(&result, &mut output).unwrap();

        let csv_string = String::from_utf8(output).unwrap();
        assert!(csv_string.contains("name"));
        assert!(csv_string.contains("Alice"));
    }

    #[test]
    fn test_format_detection() {
        assert_eq!(
            ResultSerializer::format_from_mime_type("application/sparql-results+json"),
            Some(ResultFormat::Json)
        );
        assert_eq!(
            ResultSerializer::format_from_extension("csv"),
            Some(ResultFormat::Csv)
        );
    }

    #[test]
    fn test_result_format_properties() {
        assert_eq!(
            ResultFormat::Json.mime_type(),
            "application/sparql-results+json"
        );
        assert_eq!(ResultFormat::Json.extension(), "srj");
        assert_eq!(ResultFormat::Csv.mime_type(), "text/csv");
        assert_eq!(ResultFormat::Csv.extension(), "csv");
    }
}
