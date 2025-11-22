//! Custom Template Formatter
//!
//! Provides a flexible template-based formatter for SPARQL query results using Handlebars.
//! Users can define custom output formats with templates for complete control over result presentation.

use crate::cli::formatters::{QueryResults, RdfTerm, ResultFormatter};
use handlebars::{Context, Handlebars, Helper, HelperResult, Output, RenderContext};
use serde_json::{json, Value};
use std::fs;
use std::io::Write;
use std::path::PathBuf;

/// Template formatter using Handlebars templates
pub struct TemplateFormatter {
    handlebars: Handlebars<'static>,
    template_name: String,
}

impl TemplateFormatter {
    /// Create a new template formatter from a template string
    pub fn from_string(
        template: String,
        template_name: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut handlebars = Handlebars::new();

        // Register custom helpers
        Self::register_helpers(&mut handlebars);

        // Register the template
        handlebars.register_template_string(template_name, template)?;

        Ok(Self {
            handlebars,
            template_name: template_name.to_string(),
        })
    }

    /// Create a new template formatter from a template file
    pub fn from_file(template_path: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let template_content = fs::read_to_string(&template_path)?;
        let template_name = template_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("template");

        Self::from_string(template_content, template_name)
    }

    /// Register custom Handlebars helpers for RDF term handling
    fn register_helpers(handlebars: &mut Handlebars<'static>) {
        // Disable strict mode to allow missing variables
        handlebars.set_strict_mode(false);

        // Helper to format RDF terms with full syntax
        handlebars.register_helper(
            "rdf_format",
            Box::new(
                |h: &Helper,
                 _: &Handlebars,
                 _: &Context,
                 _: &mut RenderContext,
                 out: &mut dyn Output|
                 -> HelperResult {
                    if let Some(param) = h.param(0) {
                        let value = param.value();
                        let formatted = Self::format_rdf_term_from_json(value);
                        out.write(&formatted)?;
                    }
                    Ok(())
                },
            ),
        );

        // Helper to get plain value (without RDF syntax)
        handlebars.register_helper(
            "rdf_plain",
            Box::new(
                |h: &Helper,
                 _: &Handlebars,
                 _: &Context,
                 _: &mut RenderContext,
                 out: &mut dyn Output|
                 -> HelperResult {
                    if let Some(param) = h.param(0) {
                        let value = param.value();
                        let plain = Self::get_plain_value_from_json(value);
                        out.write(&plain)?;
                    }
                    Ok(())
                },
            ),
        );

        // Helper to check term type
        handlebars.register_helper(
            "is_uri",
            Box::new(
                |h: &Helper,
                 _: &Handlebars,
                 _: &Context,
                 _: &mut RenderContext,
                 out: &mut dyn Output|
                 -> HelperResult {
                    if let Some(param) = h.param(0) {
                        let value = param.value();
                        let is_uri = value.get("type").and_then(|t| t.as_str()) == Some("uri");
                        out.write(&is_uri.to_string())?;
                    }
                    Ok(())
                },
            ),
        );

        // Helper to check if term is a literal
        handlebars.register_helper(
            "is_literal",
            Box::new(
                |h: &Helper,
                 _: &Handlebars,
                 _: &Context,
                 _: &mut RenderContext,
                 out: &mut dyn Output|
                 -> HelperResult {
                    if let Some(param) = h.param(0) {
                        let value = param.value();
                        let is_literal =
                            value.get("type").and_then(|t| t.as_str()) == Some("literal");
                        out.write(&is_literal.to_string())?;
                    }
                    Ok(())
                },
            ),
        );

        // Helper to truncate long strings
        handlebars.register_helper(
            "truncate",
            Box::new(
                |h: &Helper,
                 _: &Handlebars,
                 _: &Context,
                 _: &mut RenderContext,
                 out: &mut dyn Output|
                 -> HelperResult {
                    if let Some(text_param) = h.param(0) {
                        let text = text_param.value().as_str().unwrap_or("");
                        let max_len =
                            h.param(1).and_then(|p| p.value().as_u64()).unwrap_or(80) as usize;

                        let truncated = if text.len() > max_len {
                            format!("{}...", &text[..max_len - 3])
                        } else {
                            text.to_string()
                        };

                        out.write(&truncated)?;
                    }
                    Ok(())
                },
            ),
        );

        // Helper to count results
        handlebars.register_helper(
            "count",
            Box::new(
                |h: &Helper,
                 _: &Handlebars,
                 _: &Context,
                 _: &mut RenderContext,
                 out: &mut dyn Output|
                 -> HelperResult {
                    if let Some(param) = h.param(0) {
                        if let Some(arr) = param.value().as_array() {
                            out.write(&arr.len().to_string())?;
                        }
                    }
                    Ok(())
                },
            ),
        );
    }

    /// Format RDF term from JSON representation
    fn format_rdf_term_from_json(value: &Value) -> String {
        match value.get("type").and_then(|t| t.as_str()) {
            Some("uri") => {
                let val = value.get("value").and_then(|v| v.as_str()).unwrap_or("");
                format!("<{}>", val)
            }
            Some("literal") => {
                let val = value.get("value").and_then(|v| v.as_str()).unwrap_or("");
                if let Some(lang) = value.get("lang").and_then(|l| l.as_str()) {
                    format!("\"{}\"@{}", val, lang)
                } else if let Some(datatype) = value.get("datatype").and_then(|d| d.as_str()) {
                    format!("\"{}\"^^<{}>", val, datatype)
                } else {
                    format!("\"{}\"", val)
                }
            }
            Some("bnode") => {
                let val = value.get("value").and_then(|v| v.as_str()).unwrap_or("");
                format!("_:{}", val)
            }
            _ => String::new(),
        }
    }

    /// Get plain value from JSON representation (without RDF syntax)
    fn get_plain_value_from_json(value: &Value) -> String {
        value
            .get("value")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string()
    }

    /// Convert QueryResults to JSON for template rendering
    fn results_to_json(results: &QueryResults) -> Value {
        // Create bindings as objects with variable names as keys
        let bindings_json: Vec<Value> = results
            .bindings
            .iter()
            .map(|binding| {
                let mut binding_map = serde_json::Map::new();

                for (i, var) in results.variables.iter().enumerate() {
                    if let Some(Some(term)) = binding.values.get(i) {
                        binding_map.insert(var.clone(), Self::term_to_json(term));
                    } else {
                        binding_map.insert(var.clone(), json!(null));
                    }
                }

                Value::Object(binding_map)
            })
            .collect();

        // Also create a row-based format for easier iteration
        let rows_json: Vec<Value> = results
            .bindings
            .iter()
            .map(|binding| {
                let values: Vec<Value> = binding
                    .values
                    .iter()
                    .map(|opt_term| {
                        opt_term
                            .as_ref()
                            .map(Self::term_to_json)
                            .unwrap_or(json!(null))
                    })
                    .collect();

                json!({ "cells": values })
            })
            .collect();

        json!({
            "variables": results.variables,
            "bindings": bindings_json,
            "rows": rows_json,
            "count": results.bindings.len()
        })
    }

    /// Convert RDF term to JSON representation
    fn term_to_json(term: &RdfTerm) -> Value {
        match term {
            RdfTerm::Uri { value } => json!({
                "type": "uri",
                "value": value
            }),
            RdfTerm::Literal {
                value,
                lang,
                datatype,
            } => {
                let mut obj = serde_json::Map::new();
                obj.insert("type".to_string(), json!("literal"));
                obj.insert("value".to_string(), json!(value));
                if let Some(l) = lang {
                    obj.insert("lang".to_string(), json!(l));
                }
                if let Some(dt) = datatype {
                    obj.insert("datatype".to_string(), json!(dt));
                }
                Value::Object(obj)
            }
            RdfTerm::Bnode { value } => json!({
                "type": "bnode",
                "value": value
            }),
        }
    }
}

impl ResultFormatter for TemplateFormatter {
    fn format(&self, results: &QueryResults, writer: &mut dyn Write) -> std::io::Result<()> {
        let data = Self::results_to_json(results);

        let rendered = self
            .handlebars
            .render(&self.template_name, &data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        write!(writer, "{}", rendered)?;

        Ok(())
    }
}

/// Built-in template presets
pub struct TemplatePresets;

impl TemplatePresets {
    /// Simple HTML table template
    pub fn html_table() -> &'static str {
        r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>SPARQL Query Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .uri { color: #0066cc; }
        .literal { color: #006600; }
        .bnode { color: #666666; font-style: italic; }
        .summary { margin: 20px 0; font-weight: bold; }
    </style>
</head>
<body>
    <h1>SPARQL Query Results</h1>
    <div class="summary">Total Results: {{count bindings}} result(s)</div>
    <table>
        <thead>
            <tr>
                {{#each variables}}
                <th>?{{this}}</th>
                {{/each}}
            </tr>
        </thead>
        <tbody>
            {{#each rows}}
            <tr>
                {{#each cells}}
                <td>
                    {{#if this}}
                    {{rdf_format this}}
                    {{else}}
                    -
                    {{/if}}
                </td>
                {{/each}}
            </tr>
            {{/each}}
        </tbody>
    </table>
</body>
</html>"#
    }

    /// JSON-LD template
    pub fn jsonld() -> &'static str {
        r#"{
  "@context": {
    "results": "http://www.w3.org/ns/sparql-service-description#",
    "bindings": "http://www.w3.org/ns/sparql-service-description#resultSet"
  },
  "head": {
    "vars": [
      {{#each variables}}
      "{{this}}"{{#unless @last}},{{/unless}}
      {{/each}}
    ]
  },
  "results": {
    "bindings": [
      {{#each bindings}}
      {
        {{#each ../variables}}
        "{{this}}": {{#if (lookup .. this)}}{{json (lookup .. this)}}{{else}}null{{/if}}{{#unless @last}},{{/unless}}
        {{/each}}
      }{{#unless @last}},{{/unless}}
      {{/each}}
    ]
  }
}"#
    }

    /// Simple text template with plain values
    pub fn text_plain() -> &'static str {
        r#"SPARQL Query Results
====================

{{#each variables}}
{{this}}{{#unless @last}} | {{/unless}}
{{/each}}
{{#each variables}}
{{#repeat (add (len this) 2)}}=={{/repeat}}{{#unless @last}}{{/unless}}
{{/each}}

{{#each bindings}}
{{#each ../variables}}
{{#if (lookup .. this)}}{{rdf_plain (lookup .. this)}}{{else}}-{{/if}}{{#unless @last}} | {{/unless}}
{{/each}}
{{/each}}

Total: {{count}} result(s)
"#
    }

    /// Custom CSV template with escaped values
    pub fn csv_custom() -> &'static str {
        r#"{{#each variables}}"{{this}}"{{#unless @last}},{{/unless}}{{/each}}
{{#each bindings}}{{#each ../variables}}{{#if (lookup .. this)}}"{{rdf_plain (lookup .. this)}}"{{else}}""{{/if}}{{#unless @last}},{{/unless}}{{/each}}
{{/each}}"#
    }

    /// Markdown table template
    pub fn markdown_table() -> &'static str {
        r#"# SPARQL Query Results

| {{#each variables}}{{this}} | {{/each}}
| {{#each variables}}--- | {{/each}}
{{#each rows}}| {{#each cells}}{{#if this}}{{truncate (rdf_format this) 50}} | {{else}}- | {{/if}}{{/each}}
{{/each}}

**Total Results:** {{count bindings}}
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::formatters::Binding;

    fn create_test_results() -> QueryResults {
        QueryResults {
            variables: vec![
                "subject".to_string(),
                "predicate".to_string(),
                "object".to_string(),
            ],
            bindings: vec![
                Binding {
                    values: vec![
                        Some(RdfTerm::Uri {
                            value: "http://example.org/alice".to_string(),
                        }),
                        Some(RdfTerm::Uri {
                            value: "http://xmlns.com/foaf/0.1/name".to_string(),
                        }),
                        Some(RdfTerm::Literal {
                            value: "Alice".to_string(),
                            lang: Some("en".to_string()),
                            datatype: None,
                        }),
                    ],
                },
                Binding {
                    values: vec![
                        Some(RdfTerm::Uri {
                            value: "http://example.org/bob".to_string(),
                        }),
                        Some(RdfTerm::Uri {
                            value: "http://xmlns.com/foaf/0.1/name".to_string(),
                        }),
                        Some(RdfTerm::Literal {
                            value: "Bob".to_string(),
                            lang: None,
                            datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
                        }),
                    ],
                },
            ],
        }
    }

    #[test]
    fn test_template_from_string() {
        let template = "{{count}} results";
        let formatter = TemplateFormatter::from_string(template.to_string(), "test").unwrap();
        assert_eq!(formatter.template_name, "test");
    }

    #[test]
    fn test_simple_template_rendering() {
        let template =
            "Total: {{count bindings}} results\nVariables: {{#each variables}}{{this}} {{/each}}";
        let formatter = TemplateFormatter::from_string(template.to_string(), "test").unwrap();

        let results = create_test_results();
        let mut output = Vec::new();
        formatter.format(&results, &mut output).unwrap();

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Total: 2 results"));
        assert!(rendered.contains("subject predicate object"));
    }

    #[test]
    fn test_rdf_format_helper() {
        let template = "{{#each bindings}}{{rdf_format subject}}\n{{/each}}";
        let formatter = TemplateFormatter::from_string(template.to_string(), "test").unwrap();

        let results = create_test_results();
        let mut output = Vec::new();
        formatter.format(&results, &mut output).unwrap();

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("<http://example.org/alice>"));
    }

    #[test]
    fn test_rdf_plain_helper() {
        let template = "{{#each bindings}}{{rdf_plain subject}}\n{{/each}}";
        let formatter = TemplateFormatter::from_string(template.to_string(), "test").unwrap();

        let results = create_test_results();
        let mut output = Vec::new();
        formatter.format(&results, &mut output).unwrap();

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("http://example.org/alice"));
        assert!(!rendered.contains("<http://"));
    }

    #[test]
    fn test_truncate_helper() {
        let template = "{{#each bindings}}{{truncate (rdf_plain subject) 20}}\n{{/each}}";
        let formatter = TemplateFormatter::from_string(template.to_string(), "test").unwrap();

        let results = create_test_results();
        let mut output = Vec::new();
        formatter.format(&results, &mut output).unwrap();

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("http://example.or..."));
    }

    #[test]
    fn test_html_table_preset() {
        let template = TemplatePresets::html_table();
        let formatter = TemplateFormatter::from_string(template.to_string(), "html").unwrap();

        let results = create_test_results();
        let mut output = Vec::new();
        formatter.format(&results, &mut output).unwrap();

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("<!DOCTYPE html>"));
        assert!(rendered.contains("<table>"));
        assert!(rendered.contains("Total Results: 2"));
        assert!(rendered.contains("<th>?subject</th>"));
    }

    #[test]
    fn test_markdown_preset() {
        let template = TemplatePresets::markdown_table();
        let formatter = TemplateFormatter::from_string(template.to_string(), "md").unwrap();

        let results = create_test_results();
        let mut output = Vec::new();
        formatter.format(&results, &mut output).unwrap();

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("# SPARQL Query Results"));
        assert!(rendered.contains("| subject | predicate | object |"));
        assert!(rendered.contains("| --- | --- | --- |"));
        assert!(rendered.contains("**Total Results:** 2"));
    }

    #[test]
    fn test_empty_results() {
        // Test with the count helper instead
        let template = "Count: {{count bindings}}";
        let formatter = TemplateFormatter::from_string(template.to_string(), "test").unwrap();

        let empty_results = QueryResults {
            variables: vec!["x".to_string()],
            bindings: vec![],
        };

        let mut output = Vec::new();
        formatter.format(&empty_results, &mut output).unwrap();

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("0"));
    }
}
