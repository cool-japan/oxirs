//! Report Generation Functions for All Supported Formats
//!
//! This module provides comprehensive report generation functions that can export
//! SHACL validation reports in multiple formats including RDF formats (Turtle, JSON-LD,
//! RDF/XML, N-Triples) and structured data formats (JSON, HTML, CSV, YAML).

use chrono::{DateTime, Utc};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::Write;

use crate::{
    report::{ReportFormat, ValidationReport},
    validation::ValidationViolation,
    Result, Severity, ShaclError,
};

/// Generate validation report in the specified format
pub fn generate_report(report: &ValidationReport, format: &ReportFormat) -> Result<String> {
    match format {
        ReportFormat::Turtle => generate_turtle_report(report),
        ReportFormat::JsonLd => generate_jsonld_report(report),
        ReportFormat::RdfXml => generate_rdfxml_report(report),
        ReportFormat::NTriples => generate_ntriples_report(report),
        ReportFormat::Json => generate_json_report(report),
        ReportFormat::Html => generate_html_report(report),
        ReportFormat::Csv => generate_csv_report(report),
        ReportFormat::Text => generate_text_report(report),
        ReportFormat::Yaml => generate_yaml_report(report),
    }
}

/// Generate Turtle format validation report
pub fn generate_turtle_report(report: &ValidationReport) -> Result<String> {
    let mut output = String::new();

    // Add prefixes
    writeln!(output, "@prefix sh: <http://www.w3.org/ns/shacl#> .")?;
    writeln!(
        output,
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> ."
    )?;
    writeln!(
        output,
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> ."
    )?;
    writeln!(output, "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .")?;
    writeln!(output)?;

    // Generate report IRI
    let report_iri = format!("_:ValidationReport_{}", generate_timestamp_id());

    // Validation report
    writeln!(output, "{} a sh:ValidationReport ;", report_iri)?;
    writeln!(
        output,
        "    sh:conforms {} ;",
        if report.conforms() { "true" } else { "false" }
    )?;

    // Add validation results
    if !report.violations().is_empty() {
        write!(output, "    sh:result ")?;

        for (i, violation) in report.violations().iter().enumerate() {
            if i > 0 {
                write!(output, " ,")?;
            }
            writeln!(output)?;
            writeln!(output, "        [ a sh:ValidationResult ;")?;

            // Focus node
            writeln!(
                output,
                "            sh:focusNode {} ;",
                format_term_turtle(&violation.focus_node)
            )?;

            // Result path
            if let Some(path) = &violation.result_path {
                writeln!(
                    output,
                    "            sh:resultPath {} ;",
                    format_path_turtle(path)
                )?;
            }

            // Value
            if let Some(value) = &violation.value {
                writeln!(
                    output,
                    "            sh:value {} ;",
                    format_term_turtle(value)
                )?;
            }

            // Source constraint component
            writeln!(
                output,
                "            sh:sourceConstraintComponent {} ;",
                format_iri_turtle(violation.source_constraint_component.as_str())
            )?;

            // Source shape
            writeln!(
                output,
                "            sh:sourceShape {} ;",
                format_iri_turtle(violation.source_shape.as_str())
            )?;

            // Result severity
            let severity_iri = match violation.result_severity {
                Severity::Violation => "sh:Violation",
                Severity::Warning => "sh:Warning",
                Severity::Info => "sh:Info",
            };
            writeln!(output, "            sh:resultSeverity {} ;", severity_iri)?;

            // Result message
            if let Some(message) = &violation.result_message {
                writeln!(
                    output,
                    "            sh:resultMessage \"{}\" ;",
                    escape_turtle_string(message)
                )?;
            }

            write!(output, "        ]")?;
        }
        writeln!(output, " .")?;
    } else {
        writeln!(output, "    .")?;
    }

    Ok(output)
}

/// Generate JSON-LD format validation report
pub fn generate_jsonld_report(report: &ValidationReport) -> Result<String> {
    let mut json_report = serde_json::Map::new();

    // Context
    let mut context = serde_json::Map::new();
    context.insert(
        "sh".to_string(),
        Value::String("http://www.w3.org/ns/shacl#".to_string()),
    );
    context.insert(
        "rdf".to_string(),
        Value::String("http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string()),
    );
    context.insert(
        "conforms".to_string(),
        Value::String("sh:conforms".to_string()),
    );
    context.insert("result".to_string(), Value::String("sh:result".to_string()));
    json_report.insert("@context".to_string(), Value::Object(context));

    // Type
    json_report.insert(
        "@type".to_string(),
        Value::String("sh:ValidationReport".to_string()),
    );

    // Conforms
    json_report.insert("conforms".to_string(), Value::Bool(report.conforms()));

    // Results
    if !report.violations().is_empty() {
        let results: Vec<Value> = report
            .violations()
            .iter()
            .map(|violation| {
                let mut result = serde_json::Map::new();
                result.insert(
                    "@type".to_string(),
                    Value::String("sh:ValidationResult".to_string()),
                );

                result.insert(
                    "sh:focusNode".to_string(),
                    format_term_jsonld(&violation.focus_node),
                );

                if let Some(path) = &violation.result_path {
                    result.insert("sh:resultPath".to_string(), format_path_jsonld(path));
                }

                if let Some(value) = &violation.value {
                    result.insert("sh:value".to_string(), format_term_jsonld(value));
                }

                result.insert(
                    "sh:sourceConstraintComponent".to_string(),
                    Value::String(format!(
                        "sh:{}",
                        violation.source_constraint_component.as_str()
                    )),
                );

                result.insert(
                    "sh:sourceShape".to_string(),
                    Value::String(violation.source_shape.as_str().to_string()),
                );

                let severity = match violation.result_severity {
                    Severity::Violation => "sh:Violation",
                    Severity::Warning => "sh:Warning",
                    Severity::Info => "sh:Info",
                };
                result.insert(
                    "sh:resultSeverity".to_string(),
                    Value::String(severity.to_string()),
                );

                if let Some(message) = &violation.result_message {
                    result.insert(
                        "sh:resultMessage".to_string(),
                        Value::String(message.clone()),
                    );
                }

                Value::Object(result)
            })
            .collect();

        json_report.insert("result".to_string(), Value::Array(results));
    }

    serde_json::to_string_pretty(&json_report)
        .map_err(|e| ShaclError::ReportGeneration(format!("JSON-LD serialization error: {}", e)))
}

/// Generate RDF/XML format validation report
pub fn generate_rdfxml_report(report: &ValidationReport) -> Result<String> {
    let mut output = String::new();

    // XML declaration and RDF root
    writeln!(output, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")?;
    writeln!(output, "<rdf:RDF")?;
    writeln!(
        output,
        "    xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\""
    )?;
    writeln!(output, "    xmlns:sh=\"http://www.w3.org/ns/shacl#\"")?;
    writeln!(
        output,
        "    xmlns:xsd=\"http://www.w3.org/2001/XMLSchema#\">"
    )?;

    // Validation report
    let report_id = format!("report_{}", generate_timestamp_id());
    writeln!(
        output,
        "  <sh:ValidationReport rdf:about=\"#{}\"_>",
        report_id
    )?;
    writeln!(output, "    <sh:conforms rdf:datatype=\"http://www.w3.org/2001/XMLSchema#boolean\">{}</sh:conforms>", 
        if report.conforms() { "true" } else { "false" })?;

    // Validation results
    for (i, violation) in report.violations().iter().enumerate() {
        let result_id = format!("result_{}_{}", report_id, i);
        writeln!(output, "    <sh:result>")?;
        writeln!(
            output,
            "      <sh:ValidationResult rdf:about=\"#{}\">",
            result_id
        )?;

        writeln!(
            output,
            "        <sh:focusNode rdf:resource=\"{}\"/>",
            format_term_rdfxml(&violation.focus_node)
        )?;

        if let Some(path) = &violation.result_path {
            writeln!(
                output,
                "        <sh:resultPath rdf:resource=\"{}\"/>",
                format_path_rdfxml(path)
            )?;
        }

        if let Some(value) = &violation.value {
            writeln!(
                output,
                "        <sh:value>{}</sh:value>",
                format_term_rdfxml_value(value)
            )?;
        }

        writeln!(output, "        <sh:sourceConstraintComponent rdf:resource=\"http://www.w3.org/ns/shacl#{}\"/>", 
            violation.source_constraint_component.as_str())?;

        writeln!(
            output,
            "        <sh:sourceShape rdf:resource=\"{}\"/>",
            violation.source_shape.as_str()
        )?;

        let severity_iri = match violation.result_severity {
            Severity::Violation => "http://www.w3.org/ns/shacl#Violation",
            Severity::Warning => "http://www.w3.org/ns/shacl#Warning",
            Severity::Info => "http://www.w3.org/ns/shacl#Info",
        };
        writeln!(
            output,
            "        <sh:resultSeverity rdf:resource=\"{}\"/>",
            severity_iri
        )?;

        if let Some(message) = &violation.result_message {
            writeln!(
                output,
                "        <sh:resultMessage>{}</sh:resultMessage>",
                escape_xml_string(message)
            )?;
        }

        writeln!(output, "      </sh:ValidationResult>")?;
        writeln!(output, "    </sh:result>")?;
    }

    writeln!(output, "  </sh:ValidationReport>")?;
    writeln!(output, "</rdf:RDF>")?;

    Ok(output)
}

/// Generate N-Triples format validation report
pub fn generate_ntriples_report(report: &ValidationReport) -> Result<String> {
    let mut output = String::new();

    let report_iri = format!("_:report{}", generate_timestamp_id());

    // Report type
    writeln!(output, "{} <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/ns/shacl#ValidationReport> .", report_iri)?;

    // Conforms
    writeln!(output, "{} <http://www.w3.org/ns/shacl#conforms> \"{}\"^^<http://www.w3.org/2001/XMLSchema#boolean> .", 
        report_iri, if report.conforms() { "true" } else { "false" })?;

    // Results
    for (i, violation) in report.violations().iter().enumerate() {
        let result_iri = format!("_:result{}_{}", generate_timestamp_id(), i);

        // Link to result
        writeln!(
            output,
            "{} <http://www.w3.org/ns/shacl#result> {} .",
            report_iri, result_iri
        )?;

        // Result type
        writeln!(output, "{} <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/ns/shacl#ValidationResult> .", result_iri)?;

        // Focus node
        writeln!(
            output,
            "{} <http://www.w3.org/ns/shacl#focusNode> {} .",
            result_iri,
            format_term_ntriples(&violation.focus_node)
        )?;

        // Result path
        if let Some(path) = &violation.result_path {
            writeln!(
                output,
                "{} <http://www.w3.org/ns/shacl#resultPath> {} .",
                result_iri,
                format_path_ntriples(path)
            )?;
        }

        // Value
        if let Some(value) = &violation.value {
            writeln!(
                output,
                "{} <http://www.w3.org/ns/shacl#value> {} .",
                result_iri,
                format_term_ntriples(value)
            )?;
        }

        // Source constraint component
        writeln!(output, "{} <http://www.w3.org/ns/shacl#sourceConstraintComponent> <http://www.w3.org/ns/shacl#{}> .", 
            result_iri, violation.source_constraint_component.as_str())?;

        // Source shape
        writeln!(
            output,
            "{} <http://www.w3.org/ns/shacl#sourceShape> <{}> .",
            result_iri,
            violation.source_shape.as_str()
        )?;

        // Result severity
        let severity_iri = match violation.result_severity {
            Severity::Violation => "<http://www.w3.org/ns/shacl#Violation>",
            Severity::Warning => "<http://www.w3.org/ns/shacl#Warning>",
            Severity::Info => "<http://www.w3.org/ns/shacl#Info>",
        };
        writeln!(
            output,
            "{} <http://www.w3.org/ns/shacl#resultSeverity> {} .",
            result_iri, severity_iri
        )?;

        // Result message
        if let Some(message) = &violation.result_message {
            writeln!(
                output,
                "{} <http://www.w3.org/ns/shacl#resultMessage> \"{}\" .",
                result_iri,
                escape_ntriples_string(message)
            )?;
        }
    }

    Ok(output)
}

/// Generate JSON format validation report (non-RDF)
pub fn generate_json_report(report: &ValidationReport) -> Result<String> {
    let mut json_report = serde_json::Map::new();

    // Basic information
    json_report.insert("conforms".to_string(), Value::Bool(report.conforms()));
    json_report.insert(
        "violationCount".to_string(),
        Value::Number(report.violations().len().into()),
    );
    json_report.insert(
        "timestamp".to_string(),
        Value::String(Utc::now().to_rfc3339()),
    );

    // Violations
    let violations: Vec<Value> = report
        .violations()
        .iter()
        .map(|violation| {
            let mut v = serde_json::Map::new();

            v.insert(
                "focusNode".to_string(),
                Value::String(format!("{:?}", violation.focus_node)),
            );

            if let Some(path) = &violation.result_path {
                v.insert(
                    "resultPath".to_string(),
                    Value::String(format!("{:?}", path)),
                );
            }

            if let Some(value) = &violation.value {
                v.insert("value".to_string(), Value::String(format!("{:?}", value)));
            }

            v.insert(
                "sourceConstraintComponent".to_string(),
                Value::String(violation.source_constraint_component.as_str().to_string()),
            );

            v.insert(
                "sourceShape".to_string(),
                Value::String(violation.source_shape.as_str().to_string()),
            );

            v.insert(
                "severity".to_string(),
                Value::String(format!("{:?}", violation.result_severity)),
            );

            if let Some(message) = &violation.result_message {
                v.insert("message".to_string(), Value::String(message.clone()));
            }

            Value::Object(v)
        })
        .collect();

    json_report.insert("violations".to_string(), Value::Array(violations));

    // Summary statistics
    let mut stats = serde_json::Map::new();
    let mut severity_counts = HashMap::new();
    for violation in report.violations() {
        *severity_counts
            .entry(&violation.result_severity)
            .or_insert(0) += 1;
    }

    stats.insert(
        "violations".to_string(),
        Value::Number(
            severity_counts
                .get(&Severity::Violation)
                .unwrap_or(&0)
                .clone()
                .into(),
        ),
    );
    stats.insert(
        "warnings".to_string(),
        Value::Number(
            severity_counts
                .get(&Severity::Warning)
                .unwrap_or(&0)
                .clone()
                .into(),
        ),
    );
    stats.insert(
        "info".to_string(),
        Value::Number(
            severity_counts
                .get(&Severity::Info)
                .unwrap_or(&0)
                .clone()
                .into(),
        ),
    );

    json_report.insert("statistics".to_string(), Value::Object(stats));

    serde_json::to_string_pretty(&json_report)
        .map_err(|e| ShaclError::ReportGeneration(format!("JSON serialization error: {}", e)))
}

/// Generate HTML format validation report
pub fn generate_html_report(report: &ValidationReport) -> Result<String> {
    let mut output = String::new();

    // HTML document structure
    writeln!(output, "<!DOCTYPE html>")?;
    writeln!(output, "<html lang=\"en\">")?;
    writeln!(output, "<head>")?;
    writeln!(output, "    <meta charset=\"UTF-8\">")?;
    writeln!(
        output,
        "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">"
    )?;
    writeln!(output, "    <title>SHACL Validation Report</title>")?;
    writeln!(output, "    <style>")?;
    writeln!(output, "{}", include_str!("../html_report_style.css"))?;
    writeln!(output, "    </style>")?;
    writeln!(output, "</head>")?;
    writeln!(output, "<body>")?;

    // Report header
    writeln!(output, "    <div class=\"container\">")?;
    writeln!(output, "        <header class=\"report-header\">")?;
    writeln!(output, "            <h1>SHACL Validation Report</h1>")?;
    writeln!(output, "            <div class=\"report-metadata\">")?;
    writeln!(
        output,
        "                <p><strong>Generated:</strong> {}</p>",
        Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    )?;
    writeln!(
        output,
        "                <p><strong>Conforms:</strong> <span class=\"{}\">{}</span></p>",
        if report.conforms() {
            "conforms-yes"
        } else {
            "conforms-no"
        },
        if report.conforms() {
            "✓ Yes"
        } else {
            "✗ No"
        }
    )?;
    writeln!(
        output,
        "                <p><strong>Total Violations:</strong> {}</p>",
        report.violations().len()
    )?;
    writeln!(output, "            </div>")?;
    writeln!(output, "        </header>")?;

    if !report.violations().is_empty() {
        // Summary statistics
        let mut severity_counts = HashMap::new();
        for violation in report.violations() {
            *severity_counts
                .entry(&violation.result_severity)
                .or_insert(0) += 1;
        }

        writeln!(output, "        <section class=\"summary\">")?;
        writeln!(output, "            <h2>Summary</h2>")?;
        writeln!(output, "            <div class=\"stats-grid\">")?;
        writeln!(
            output,
            "                <div class=\"stat-card violation\">"
        )?;
        writeln!(output, "                    <h3>Violations</h3>")?;
        writeln!(
            output,
            "                    <span class=\"count\">{}</span>",
            severity_counts.get(&Severity::Violation).unwrap_or(&0)
        )?;
        writeln!(output, "                </div>")?;
        writeln!(output, "                <div class=\"stat-card warning\">")?;
        writeln!(output, "                    <h3>Warnings</h3>")?;
        writeln!(
            output,
            "                    <span class=\"count\">{}</span>",
            severity_counts.get(&Severity::Warning).unwrap_or(&0)
        )?;
        writeln!(output, "                </div>")?;
        writeln!(output, "                <div class=\"stat-card info\">")?;
        writeln!(output, "                    <h3>Info</h3>")?;
        writeln!(
            output,
            "                    <span class=\"count\">{}</span>",
            severity_counts.get(&Severity::Info).unwrap_or(&0)
        )?;
        writeln!(output, "                </div>")?;
        writeln!(output, "            </div>")?;
        writeln!(output, "        </section>")?;

        // Violations table
        writeln!(output, "        <section class=\"violations\">")?;
        writeln!(output, "            <h2>Validation Results</h2>")?;
        writeln!(output, "            <table class=\"violations-table\">")?;
        writeln!(output, "                <thead>")?;
        writeln!(output, "                    <tr>")?;
        writeln!(output, "                        <th>Severity</th>")?;
        writeln!(output, "                        <th>Focus Node</th>")?;
        writeln!(output, "                        <th>Path</th>")?;
        writeln!(output, "                        <th>Value</th>")?;
        writeln!(output, "                        <th>Message</th>")?;
        writeln!(output, "                        <th>Shape</th>")?;
        writeln!(output, "                        <th>Constraint</th>")?;
        writeln!(output, "                    </tr>")?;
        writeln!(output, "                </thead>")?;
        writeln!(output, "                <tbody>")?;

        for violation in report.violations() {
            let severity_class = match violation.result_severity {
                Severity::Violation => "severity-violation",
                Severity::Warning => "severity-warning",
                Severity::Info => "severity-info",
            };

            writeln!(
                output,
                "                    <tr class=\"{}\">",
                severity_class
            )?;
            writeln!(
                output,
                "                        <td><span class=\"severity-badge {}\">{:?}</span></td>",
                severity_class, violation.result_severity
            )?;
            writeln!(
                output,
                "                        <td class=\"focus-node\">{}</td>",
                format!("{:?}", violation.focus_node)
            )?;
            writeln!(
                output,
                "                        <td class=\"path\">{}</td>",
                violation
                    .result_path
                    .as_ref()
                    .map(|p| format!("{:?}", p))
                    .unwrap_or_else(|| "-".to_string())
            )?;
            writeln!(
                output,
                "                        <td class=\"value\">{}</td>",
                violation
                    .value
                    .as_ref()
                    .map(|v| format!("{:?}", v))
                    .unwrap_or_else(|| "-".to_string())
            )?;
            writeln!(
                output,
                "                        <td class=\"message\">{}</td>",
                violation
                    .result_message
                    .as_ref()
                    .unwrap_or(&"-".to_string())
            )?;
            writeln!(
                output,
                "                        <td class=\"shape\">{}</td>",
                violation.source_shape.as_str()
            )?;
            writeln!(
                output,
                "                        <td class=\"constraint\">{}</td>",
                violation.source_constraint_component.as_str()
            )?;
            writeln!(output, "                    </tr>")?;
        }

        writeln!(output, "                </tbody>")?;
        writeln!(output, "            </table>")?;
        writeln!(output, "        </section>")?;
    } else {
        writeln!(output, "        <section class=\"no-violations\">")?;
        writeln!(output, "            <h2>✓ Validation Successful</h2>")?;
        writeln!(output, "            <p>All data conforms to the specified SHACL shapes. No violations found.</p>")?;
        writeln!(output, "        </section>")?;
    }

    writeln!(output, "    </div>")?;
    writeln!(output, "</body>")?;
    writeln!(output, "</html>")?;

    Ok(output)
}

/// Generate CSV format validation report
pub fn generate_csv_report(report: &ValidationReport) -> Result<String> {
    let mut output = String::new();

    // CSV header
    writeln!(
        output,
        "Severity,Focus Node,Result Path,Value,Message,Source Shape,Source Constraint Component"
    )?;

    // CSV rows
    for violation in report.violations() {
        let severity = format!("{:?}", violation.result_severity);
        let focus_node = format!("{:?}", violation.focus_node);
        let result_path = violation
            .result_path
            .as_ref()
            .map(|p| format!("{:?}", p))
            .unwrap_or_else(|| "".to_string());
        let value = violation
            .value
            .as_ref()
            .map(|v| format!("{:?}", v))
            .unwrap_or_else(|| "".to_string());
        let message = violation
            .result_message
            .as_ref()
            .unwrap_or(&"".to_string())
            .clone();
        let source_shape = violation.source_shape.as_str();
        let source_constraint = violation.source_constraint_component.as_str();

        writeln!(
            output,
            "\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\"",
            escape_csv_field(&severity),
            escape_csv_field(&focus_node),
            escape_csv_field(&result_path),
            escape_csv_field(&value),
            escape_csv_field(&message),
            escape_csv_field(source_shape),
            escape_csv_field(source_constraint)
        )?;
    }

    Ok(output)
}

/// Generate plain text format validation report
pub fn generate_text_report(report: &ValidationReport) -> Result<String> {
    let mut output = String::new();

    // Report header
    writeln!(output, "SHACL Validation Report")?;
    writeln!(output, "=======================")?;
    writeln!(
        output,
        "Generated: {}",
        Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    )?;
    writeln!(
        output,
        "Conforms: {}",
        if report.conforms() { "YES" } else { "NO" }
    )?;
    writeln!(output, "Total Violations: {}", report.violations().len())?;
    writeln!(output)?;

    if !report.violations().is_empty() {
        // Summary by severity
        let mut severity_counts = HashMap::new();
        for violation in report.violations() {
            *severity_counts
                .entry(&violation.result_severity)
                .or_insert(0) += 1;
        }

        writeln!(output, "Summary by Severity:")?;
        writeln!(
            output,
            "  Violations: {}",
            severity_counts.get(&Severity::Violation).unwrap_or(&0)
        )?;
        writeln!(
            output,
            "  Warnings:   {}",
            severity_counts.get(&Severity::Warning).unwrap_or(&0)
        )?;
        writeln!(
            output,
            "  Info:       {}",
            severity_counts.get(&Severity::Info).unwrap_or(&0)
        )?;
        writeln!(output)?;

        // Detailed violations
        writeln!(output, "Detailed Results:")?;
        writeln!(output, "-----------------")?;

        for (i, violation) in report.violations().iter().enumerate() {
            writeln!(
                output,
                "{}. [{:?}] {}",
                i + 1,
                violation.result_severity,
                violation
                    .result_message
                    .as_ref()
                    .unwrap_or(&"No message".to_string())
            )?;

            writeln!(output, "   Focus Node: {:?}", violation.focus_node)?;

            if let Some(path) = &violation.result_path {
                writeln!(output, "   Result Path: {:?}", path)?;
            }

            if let Some(value) = &violation.value {
                writeln!(output, "   Value: {:?}", value)?;
            }

            writeln!(
                output,
                "   Source Shape: {}",
                violation.source_shape.as_str()
            )?;

            writeln!(
                output,
                "   Constraint Component: {}",
                violation.source_constraint_component.as_str()
            )?;

            writeln!(output)?;
        }
    } else {
        writeln!(output, "✓ All data conforms to the specified SHACL shapes.")?;
        writeln!(output, "  No violations found.")?;
    }

    Ok(output)
}

/// Generate YAML format validation report
pub fn generate_yaml_report(report: &ValidationReport) -> Result<String> {
    let mut output = String::new();

    // YAML header
    writeln!(output, "---")?;
    writeln!(output, "validationReport:")?;
    writeln!(
        output,
        "  conforms: {}",
        if report.conforms() { "true" } else { "false" }
    )?;
    writeln!(output, "  timestamp: \"{}\"", Utc::now().to_rfc3339())?;
    writeln!(output, "  violationCount: {}", report.violations().len())?;

    if !report.violations().is_empty() {
        // Summary statistics
        let mut severity_counts = HashMap::new();
        for violation in report.violations() {
            *severity_counts
                .entry(&violation.result_severity)
                .or_insert(0) += 1;
        }

        writeln!(output, "  summary:")?;
        writeln!(
            output,
            "    violations: {}",
            severity_counts.get(&Severity::Violation).unwrap_or(&0)
        )?;
        writeln!(
            output,
            "    warnings: {}",
            severity_counts.get(&Severity::Warning).unwrap_or(&0)
        )?;
        writeln!(
            output,
            "    info: {}",
            severity_counts.get(&Severity::Info).unwrap_or(&0)
        )?;

        // Violations
        writeln!(output, "  violations:")?;
        for violation in report.violations() {
            writeln!(output, "    - severity: {:?}", violation.result_severity)?;

            writeln!(
                output,
                "      focusNode: \"{}\"",
                format!("{:?}", violation.focus_node).replace("\"", "\\\"")
            )?;

            if let Some(path) = &violation.result_path {
                writeln!(
                    output,
                    "      resultPath: \"{}\"",
                    format!("{:?}", path).replace("\"", "\\\"")
                )?;
            }

            if let Some(value) = &violation.value {
                writeln!(
                    output,
                    "      value: \"{}\"",
                    format!("{:?}", value).replace("\"", "\\\"")
                )?;
            }

            if let Some(message) = &violation.result_message {
                writeln!(
                    output,
                    "      message: \"{}\"",
                    message.replace("\"", "\\\"")
                )?;
            }

            writeln!(
                output,
                "      sourceShape: \"{}\"",
                violation.source_shape.as_str()
            )?;

            writeln!(
                output,
                "      sourceConstraintComponent: \"{}\"",
                violation.source_constraint_component.as_str()
            )?;
        }
    } else {
        writeln!(output, "  violations: []")?;
    }

    Ok(output)
}

// Helper functions for formatting

fn generate_timestamp_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
        .to_string()
}

fn format_term_turtle(term: &oxirs_core::model::Term) -> String {
    format!("{:?}", term) // Placeholder - should use proper Turtle formatting
}

fn format_path_turtle(path: &crate::paths::PropertyPath) -> String {
    format!("{:?}", path) // Placeholder - should use proper Turtle formatting
}

fn format_iri_turtle(iri: &str) -> String {
    format!("<{}>", iri)
}

fn escape_turtle_string(s: &str) -> String {
    s.replace("\\", "\\\\")
        .replace("\"", "\\\"")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
}

fn format_term_jsonld(term: &oxirs_core::model::Term) -> Value {
    // Placeholder - should use proper JSON-LD formatting
    Value::String(format!("{:?}", term))
}

fn format_path_jsonld(path: &crate::paths::PropertyPath) -> Value {
    // Placeholder - should use proper JSON-LD formatting
    Value::String(format!("{:?}", path))
}

fn format_term_rdfxml(term: &oxirs_core::model::Term) -> String {
    format!("{:?}", term) // Placeholder - should use proper RDF/XML formatting
}

fn format_path_rdfxml(path: &crate::paths::PropertyPath) -> String {
    format!("{:?}", path) // Placeholder - should use proper RDF/XML formatting
}

fn format_term_rdfxml_value(term: &oxirs_core::model::Term) -> String {
    escape_xml_string(&format!("{:?}", term))
}

fn escape_xml_string(s: &str) -> String {
    s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\"", "&quot;")
        .replace("'", "&apos;")
}

fn format_term_ntriples(term: &oxirs_core::model::Term) -> String {
    format!("{:?}", term) // Placeholder - should use proper N-Triples formatting
}

fn format_path_ntriples(path: &crate::paths::PropertyPath) -> String {
    format!("{:?}", path) // Placeholder - should use proper N-Triples formatting
}

fn escape_ntriples_string(s: &str) -> String {
    s.replace("\\", "\\\\")
        .replace("\"", "\\\"")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
}

fn escape_csv_field(s: &str) -> String {
    s.replace("\"", "\"\"")
}
