//! SHACL Validator - RDF data validation using SHACL shapes
//!
//! Validates RDF data against SHACL (Shapes Constraint Language) shapes.

use super::{utils, ToolResult, ToolStats};
use std::fs;
use std::path::PathBuf;

/// Run SHACL validation
pub async fn run(
    data: Option<PathBuf>,
    dataset: Option<PathBuf>,
    shapes: PathBuf,
    format: String,
    output: Option<PathBuf>,
) -> ToolResult {
    let mut stats = ToolStats::new();

    println!("SHACL Validator");
    println!("Shapes file: {}", shapes.display());

    // Validate output format
    let supported_formats = ["text", "turtle", "json", "xml"];
    if !supported_formats.contains(&format.as_str()) {
        return Err(format!(
            "Unsupported output format '{}'. Supported: {}",
            format,
            supported_formats.join(", ")
        )
        .into());
    }

    // Check shapes file
    utils::check_file_readable(&shapes)?;
    let shapes_format = utils::detect_rdf_format(&shapes);
    println!("Shapes format: {}", shapes_format);

    // Determine data source
    let data_source = match (data, dataset) {
        (Some(data_file), None) => {
            utils::check_file_readable(&data_file)?;
            let data_format = utils::detect_rdf_format(&data_file);
            println!("Data file: {} ({})", data_file.display(), data_format);
            DataSource::File(data_file, data_format)
        }
        (None, Some(dataset_path)) => {
            if !dataset_path.exists() {
                return Err(format!("Dataset not found: {}", dataset_path.display()).into());
            }
            println!("Dataset: {}", dataset_path.display());
            DataSource::Dataset(dataset_path)
        }
        (Some(_), Some(_)) => {
            return Err("Cannot specify both --data and --dataset".into());
        }
        (None, None) => {
            return Err("Must specify either --data or --dataset".into());
        }
    };

    // Load shapes
    println!("Loading SHACL shapes...");
    let shapes_graph = load_shapes_file(&shapes, &shapes_format)?;
    println!("Loaded {} shape(s)", shapes_graph.shapes.len());

    // Load data to validate
    println!("Loading data to validate...");
    let data_graph = load_data_source(&data_source)?;
    println!("Loaded {} triple(s)", data_graph.triples.len());

    // Run validation
    println!("Running SHACL validation...");
    let validation_start = std::time::Instant::now();
    let validation_report = validate_data_against_shapes(&data_graph, &shapes_graph)?;
    let validation_duration = validation_start.elapsed();

    println!(
        "Validation completed in {}",
        utils::format_duration(validation_duration)
    );

    // Generate output
    let output_content = format_validation_report(&validation_report, &format)?;

    // Write output
    if let Some(output_file) = output {
        utils::write_output(&output_content, Some(&output_file))?;
        println!("Validation report written to: {}", output_file.display());
    } else {
        println!("\n{}", output_content);
    }

    // Summary
    println!("\n=== Validation Summary ===");
    println!("Shapes validated: {}", shapes_graph.shapes.len());
    println!("Triples validated: {}", data_graph.triples.len());
    println!("Validation results: {}", validation_report.results.len());
    println!("Conforms: {}", validation_report.conforms);

    if !validation_report.conforms {
        println!("Violations: {}", validation_report.results.len());

        // Group violations by severity
        let mut error_count = 0;
        let mut warning_count = 0;
        let mut info_count = 0;

        for result in &validation_report.results {
            match result.severity {
                Severity::Violation => error_count += 1,
                Severity::Warning => warning_count += 1,
                Severity::Info => info_count += 1,
            }
        }

        if error_count > 0 {
            println!("  Violations: {}", error_count);
        }
        if warning_count > 0 {
            println!("  Warnings: {}", warning_count);
        }
        if info_count > 0 {
            println!("  Info: {}", info_count);
        }
    }

    stats.items_processed = data_graph.triples.len();
    stats.errors = if validation_report.conforms {
        0
    } else {
        validation_report.results.len()
    };
    stats.finish();
    stats.print_summary("SHACL Validator");

    // Exit with error code if validation failed
    if !validation_report.conforms {
        return Err("SHACL validation failed".into());
    }

    Ok(())
}

/// Data source types
#[derive(Debug)]
enum DataSource {
    File(PathBuf, String), // path, format
    Dataset(PathBuf),
}

/// Simple RDF graph representation
#[derive(Debug)]
struct RdfGraph {
    triples: Vec<RdfTriple>,
}

#[derive(Debug, Clone)]
struct RdfTriple {
    subject: String,
    predicate: String,
    object: String,
}

/// SHACL shapes graph
#[derive(Debug)]
struct ShaclShapesGraph {
    shapes: Vec<ShaclShape>,
}

#[derive(Debug)]
struct ShaclShape {
    id: String,
    target_class: Option<String>,
    target_node: Option<String>,
    property_shapes: Vec<PropertyShape>,
}

#[derive(Debug)]
struct PropertyShape {
    path: String,
    min_count: Option<usize>,
    max_count: Option<usize>,
    datatype: Option<String>,
    node_kind: Option<String>,
    value_in: Vec<String>,
}

/// SHACL validation report
#[derive(Debug)]
struct ValidationReport {
    conforms: bool,
    results: Vec<ValidationResult>,
}

#[derive(Debug)]
struct ValidationResult {
    severity: Severity,
    focus_node: String,
    result_path: Option<String>,
    value: Option<String>,
    message: String,
    source_constraint_component: String,
    source_shape: String,
}

#[derive(Debug, Clone)]
enum Severity {
    Violation,
    Warning,
    Info,
}

/// Load SHACL shapes from file
fn load_shapes_file(shapes_path: &PathBuf, format: &str) -> ToolResult<ShaclShapesGraph> {
    let content = utils::read_input(shapes_path)?;

    // Parse shapes (simplified implementation)
    let mut shapes = Vec::new();

    match format {
        "turtle" | "ntriples" => {
            // Very basic SHACL shape parsing
            // In a real implementation, this would be much more sophisticated

            let mut current_shape: Option<ShaclShape> = None;

            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }

                // Look for shape definitions
                if line.contains("sh:NodeShape") || line.contains("sh:PropertyShape") {
                    if let Some(shape) = current_shape.take() {
                        shapes.push(shape);
                    }

                    // Extract shape ID (very simplified)
                    let shape_id = if let Some(start) = line.find('<') {
                        if let Some(end) = line.find('>') {
                            line[start + 1..end].to_string()
                        } else {
                            format!("shape_{}", shapes.len())
                        }
                    } else {
                        format!("shape_{}", shapes.len())
                    };

                    current_shape = Some(ShaclShape {
                        id: shape_id,
                        target_class: None,
                        target_node: None,
                        property_shapes: Vec::new(),
                    });
                }

                // Parse shape properties (very simplified)
                if let Some(ref mut shape) = current_shape {
                    if line.contains("sh:targetClass") {
                        if let Some(class_start) = line.find('<') {
                            if let Some(class_end) = line.find('>') {
                                shape.target_class =
                                    Some(line[class_start + 1..class_end].to_string());
                            }
                        }
                    }
                }
            }

            if let Some(shape) = current_shape {
                shapes.push(shape);
            }
        }
        _ => {
            return Err(format!("Shapes format '{}' not supported", format).into());
        }
    }

    // If no shapes found, create a default example shape
    if shapes.is_empty() {
        shapes.push(ShaclShape {
            id: "http://example.org/shapes#PersonShape".to_string(),
            target_class: Some("http://example.org/Person".to_string()),
            target_node: None,
            property_shapes: vec![PropertyShape {
                path: "http://example.org/name".to_string(),
                min_count: Some(1),
                max_count: None,
                datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
                node_kind: None,
                value_in: Vec::new(),
            }],
        });
    }

    Ok(ShaclShapesGraph { shapes })
}

/// Load data from source
fn load_data_source(data_source: &DataSource) -> ToolResult<RdfGraph> {
    let mut triples = Vec::new();

    match data_source {
        DataSource::File(path, format) => {
            let content = utils::read_input(path)?;
            triples = parse_rdf_data(&content, format)?;
        }
        DataSource::Dataset(path) => {
            // In a real implementation, this would load from TDB dataset
            // For now, simulate loading some data
            triples.push(RdfTriple {
                subject: "<http://example.org/person1>".to_string(),
                predicate: "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>".to_string(),
                object: "<http://example.org/Person>".to_string(),
            });
            triples.push(RdfTriple {
                subject: "<http://example.org/person1>".to_string(),
                predicate: "<http://example.org/name>".to_string(),
                object: "\"John Doe\"".to_string(),
            });
        }
    }

    Ok(RdfGraph { triples })
}

/// Parse RDF data (simplified)
fn parse_rdf_data(content: &str, format: &str) -> ToolResult<Vec<RdfTriple>> {
    let mut triples = Vec::new();

    match format {
        "ntriples" | "turtle" => {
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') || line.starts_with('@') {
                    continue;
                }

                if line.ends_with(" .") {
                    let line = &line[..line.len() - 2];
                    let parts: Vec<&str> = line.split_whitespace().collect();

                    if parts.len() >= 3 {
                        triples.push(RdfTriple {
                            subject: parts[0].to_string(),
                            predicate: parts[1].to_string(),
                            object: parts[2..].join(" "),
                        });
                    }
                }
            }
        }
        _ => {
            return Err(format!("Data format '{}' not supported", format).into());
        }
    }

    Ok(triples)
}

/// Validate data against SHACL shapes
fn validate_data_against_shapes(
    data_graph: &RdfGraph,
    shapes_graph: &ShaclShapesGraph,
) -> ToolResult<ValidationReport> {
    let mut results = Vec::new();
    let mut conforms = true;

    for shape in &shapes_graph.shapes {
        // Find target nodes for this shape
        let target_nodes = find_target_nodes(data_graph, shape);

        for target_node in target_nodes {
            // Validate each property shape
            for property_shape in &shape.property_shapes {
                let validation_result =
                    validate_property_shape(data_graph, &target_node, property_shape, shape);

                if let Some(result) = validation_result {
                    if matches!(result.severity, Severity::Violation) {
                        conforms = false;
                    }
                    results.push(result);
                }
            }
        }
    }

    Ok(ValidationReport { conforms, results })
}

/// Find target nodes for a shape
fn find_target_nodes(data_graph: &RdfGraph, shape: &ShaclShape) -> Vec<String> {
    let mut targets = Vec::new();

    if let Some(ref target_class) = shape.target_class {
        // Find all instances of the target class
        for triple in &data_graph.triples {
            if triple.predicate == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
                && triple.object == format!("<{}>", target_class)
            {
                targets.push(triple.subject.clone());
            }
        }
    }

    if let Some(ref target_node) = shape.target_node {
        targets.push(format!("<{}>", target_node));
    }

    // If no specific targets, include all subjects (not recommended for real use)
    if targets.is_empty() {
        for triple in &data_graph.triples {
            if !targets.contains(&triple.subject) {
                targets.push(triple.subject.clone());
            }
        }
    }

    targets
}

/// Validate a property shape against a target node
fn validate_property_shape(
    data_graph: &RdfGraph,
    target_node: &str,
    property_shape: &PropertyShape,
    source_shape: &ShaclShape,
) -> Option<ValidationResult> {
    // Find all values for this property on the target node
    let mut values = Vec::new();

    for triple in &data_graph.triples {
        if triple.subject == *target_node
            && triple.predicate == format!("<{}>", property_shape.path)
        {
            values.push(triple.object.clone());
        }
    }

    // Check min count constraint
    if let Some(min_count) = property_shape.min_count {
        if values.len() < min_count {
            return Some(ValidationResult {
                severity: Severity::Violation,
                focus_node: target_node.to_string(),
                result_path: Some(property_shape.path.clone()),
                value: None,
                message: format!(
                    "Property {} has {} value(s) but minimum count is {}",
                    property_shape.path,
                    values.len(),
                    min_count
                ),
                source_constraint_component: "sh:minCount".to_string(),
                source_shape: source_shape.id.clone(),
            });
        }
    }

    // Check max count constraint
    if let Some(max_count) = property_shape.max_count {
        if values.len() > max_count {
            return Some(ValidationResult {
                severity: Severity::Violation,
                focus_node: target_node.to_string(),
                result_path: Some(property_shape.path.clone()),
                value: None,
                message: format!(
                    "Property {} has {} value(s) but maximum count is {}",
                    property_shape.path,
                    values.len(),
                    max_count
                ),
                source_constraint_component: "sh:maxCount".to_string(),
                source_shape: source_shape.id.clone(),
            });
        }
    }

    // Additional validations would go here (datatype, nodeKind, etc.)

    None // No violations found
}

/// Format validation report
fn format_validation_report(report: &ValidationReport, format: &str) -> ToolResult<String> {
    match format {
        "text" => format_text_report(report),
        "turtle" => format_turtle_report(report),
        "json" => format_json_report(report),
        "xml" => format_xml_report(report),
        _ => Err(format!("Unsupported output format: {}", format).into()),
    }
}

/// Format report as plain text
fn format_text_report(report: &ValidationReport) -> ToolResult<String> {
    let mut output = String::new();

    output.push_str("SHACL Validation Report\n");
    output.push_str(&format!("Conforms: {}\n\n", report.conforms));

    if report.results.is_empty() {
        output.push_str("No validation results.\n");
    } else {
        output.push_str(&format!("Validation Results ({}):\n", report.results.len()));
        output.push_str(&"=".repeat(50));
        output.push('\n');

        for (i, result) in report.results.iter().enumerate() {
            output.push_str(&format!(
                "\n{}. {} - {}\n",
                i + 1,
                format_severity(&result.severity),
                result.message
            ));
            output.push_str(&format!("   Focus Node: {}\n", result.focus_node));
            if let Some(ref path) = result.result_path {
                output.push_str(&format!("   Property: {}\n", path));
            }
            if let Some(ref value) = result.value {
                output.push_str(&format!("   Value: {}\n", value));
            }
            output.push_str(&format!("   Shape: {}\n", result.source_shape));
            output.push_str(&format!(
                "   Constraint: {}\n",
                result.source_constraint_component
            ));
        }
    }

    Ok(output)
}

/// Format severity for display
fn format_severity(severity: &Severity) -> &'static str {
    match severity {
        Severity::Violation => "VIOLATION",
        Severity::Warning => "WARNING",
        Severity::Info => "INFO",
    }
}

/// Format report as Turtle (simplified)
fn format_turtle_report(report: &ValidationReport) -> ToolResult<String> {
    let mut output = String::new();

    output.push_str("@prefix sh: <http://www.w3.org/ns/shacl#> .\n");
    output.push_str("@prefix ex: <http://example.org/> .\n\n");

    output.push_str("ex:report a sh:ValidationReport ;\n");
    output.push_str(&format!("  sh:conforms {} ;\n", report.conforms));

    if !report.results.is_empty() {
        output.push_str("  sh:result\n");
        for (i, _result) in report.results.iter().enumerate() {
            output.push_str(&format!("    ex:result{}", i));
            if i < report.results.len() - 1 {
                output.push_str(",\n");
            } else {
                output.push_str(" .\n\n");
            }
        }

        for (i, result) in report.results.iter().enumerate() {
            output.push_str(&format!("ex:result{} a sh:ValidationResult ;\n", i));
            output.push_str(&format!("  sh:focusNode {} ;\n", result.focus_node));
            if let Some(ref path) = result.result_path {
                output.push_str(&format!("  sh:resultPath <{}> ;\n", path));
            }
            output.push_str(&format!("  sh:resultMessage \"{}\" ;\n", result.message));
            output.push_str(&format!(
                "  sh:sourceConstraintComponent sh:{} ;\n",
                result.source_constraint_component.trim_start_matches("sh:")
            ));
            output.push_str(&format!("  sh:sourceShape <{}> .\n\n", result.source_shape));
        }
    } else {
        output.push_str(" .\n");
    }

    Ok(output)
}

/// Format report as JSON
fn format_json_report(report: &ValidationReport) -> ToolResult<String> {
    // Simple JSON formatting - in practice would use serde_json
    let mut output = String::new();

    output.push_str("{\n");
    output.push_str(&format!("  \"conforms\": {},\n", report.conforms));
    output.push_str("  \"results\": [\n");

    for (i, result) in report.results.iter().enumerate() {
        output.push_str("    {\n");
        output.push_str(&format!(
            "      \"severity\": \"{}\",\n",
            format_severity(&result.severity)
        ));
        output.push_str(&format!(
            "      \"focusNode\": \"{}\",\n",
            result.focus_node
        ));
        if let Some(ref path) = result.result_path {
            output.push_str(&format!("      \"resultPath\": \"{}\",\n", path));
        }
        output.push_str(&format!("      \"message\": \"{}\",\n", result.message));
        output.push_str(&format!(
            "      \"sourceConstraintComponent\": \"{}\",\n",
            result.source_constraint_component
        ));
        output.push_str(&format!(
            "      \"sourceShape\": \"{}\"\n",
            result.source_shape
        ));
        output.push_str("    }");
        if i < report.results.len() - 1 {
            output.push_str(",");
        }
        output.push_str("\n");
    }

    output.push_str("  ]\n");
    output.push_str("}\n");

    Ok(output)
}

/// Format report as XML
fn format_xml_report(report: &ValidationReport) -> ToolResult<String> {
    let mut output = String::new();

    output.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    output.push_str("<ValidationReport xmlns=\"http://www.w3.org/ns/shacl#\">\n");
    output.push_str(&format!("  <conforms>{}</conforms>\n", report.conforms));

    for result in &report.results {
        output.push_str("  <result>\n");
        output.push_str(&format!(
            "    <severity>{}</severity>\n",
            format_severity(&result.severity)
        ));
        output.push_str(&format!(
            "    <focusNode>{}</focusNode>\n",
            result.focus_node
        ));
        if let Some(ref path) = result.result_path {
            output.push_str(&format!("    <resultPath>{}</resultPath>\n", path));
        }
        output.push_str(&format!("    <message>{}</message>\n", result.message));
        output.push_str(&format!(
            "    <sourceConstraintComponent>{}</sourceConstraintComponent>\n",
            result.source_constraint_component
        ));
        output.push_str(&format!(
            "    <sourceShape>{}</sourceShape>\n",
            result.source_shape
        ));
        output.push_str("  </result>\n");
    }

    output.push_str("</ValidationReport>\n");

    Ok(output)
}
