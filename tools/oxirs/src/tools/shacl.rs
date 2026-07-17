//! SHACL Validator - RDF data validation using SHACL shapes
//!
//! Validates RDF data against SHACL (Shapes Constraint Language) shapes using
//! the real engine/oxirs-shacl validator (full SHACL Core + SHACL-AF), the
//! same engine wired into `oxirs-fuseki`'s `/shacl` HTTP endpoint
//! (`server/oxirs-fuseki/src/handlers/shacl.rs`).

use super::{utils, ToolResult, ToolStats};
use indexmap::IndexMap;
use oxirs_core::format::{FormatHandler, JsonLdProfileSet, RdfFormat as CoreRdfFormat};
use oxirs_core::model::Triple;
use oxirs_core::rdf_store::RdfStore;
use oxirs_shacl::{Shape, ShapeId, ShapeParser, ValidationConfig, ValidationEngine};
use std::fs::File;
use std::path::{Path, PathBuf};

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
    println!("Shapes format: {shapes_format}");

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

    // Load shapes via the real SHACL shape parser
    println!("Loading SHACL shapes...");
    let shapes_map = load_shapes_file(&shapes, &shapes_format)?;
    println!("Loaded {} shape(s)", shapes_map.len());

    // Load data to validate via the real oxirs-core store
    println!("Loading data to validate...");
    let data_store = load_data_source(&data_source)?;
    let triple_count = data_store
        .quads()
        .map_err(|e| format!("Failed to enumerate data quads: {e}"))?
        .len();
    println!("Loaded {triple_count} triple(s)");

    // Run validation using the real oxirs-shacl validation engine
    println!("Running SHACL validation...");
    let validation_start = std::time::Instant::now();
    let validation_report = validate_data_against_shapes(&data_store, &shapes_map)?;
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
        println!("\n{output_content}");
    }

    // Summary
    println!("\n=== Validation Summary ===");
    println!("Shapes validated: {}", shapes_map.len());
    println!("Triples validated: {triple_count}");
    println!(
        "Validation results: {}",
        validation_report.violations().len()
    );
    println!("Conforms: {}", validation_report.conforms);

    if !validation_report.conforms {
        println!("Violations: {}", validation_report.violations().len());

        // Group violations by severity
        let mut error_count = 0;
        let mut warning_count = 0;
        let mut info_count = 0;

        for violation in validation_report.violations() {
            match violation.result_severity {
                oxirs_shacl::Severity::Violation => error_count += 1,
                oxirs_shacl::Severity::Warning => warning_count += 1,
                oxirs_shacl::Severity::Info => info_count += 1,
            }
        }

        if error_count > 0 {
            println!("  Violations: {error_count}");
        }
        if warning_count > 0 {
            println!("  Warnings: {warning_count}");
        }
        if info_count > 0 {
            println!("  Info: {info_count}");
        }
    }

    stats.items_processed = triple_count;
    stats.errors = if validation_report.conforms {
        0
    } else {
        validation_report.violations().len()
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

/// Map a `utils::detect_rdf_format` string to oxirs-core's parser-facing
/// `RdfFormat`, as used elsewhere for real (non-fabricated) RDF parsing.
fn tool_format_to_core(format: &str) -> ToolResult<CoreRdfFormat> {
    match format {
        "turtle" | "ttl" => Ok(CoreRdfFormat::Turtle),
        "ntriples" | "nt" => Ok(CoreRdfFormat::NTriples),
        "nquads" | "nq" => Ok(CoreRdfFormat::NQuads),
        "trig" => Ok(CoreRdfFormat::TriG),
        "n3" => Ok(CoreRdfFormat::N3),
        "rdfxml" | "rdf" | "xml" => Ok(CoreRdfFormat::RdfXml),
        "jsonld" | "json-ld" | "json" => Ok(CoreRdfFormat::JsonLd {
            profile: JsonLdProfileSet::empty(),
        }),
        other => Err(format!("Unsupported RDF format for SHACL: '{other}'").into()),
    }
}

/// Parse an RDF file's real content into triples via oxirs-core's format
/// handlers. No placeholder/synthesized triples are ever produced.
fn parse_triples_file(path: &Path, format: &str) -> ToolResult<Vec<Triple>> {
    let core_format = tool_format_to_core(format)?;
    let file = File::open(path).map_err(|e| format!("Failed to open '{}': {e}", path.display()))?;
    FormatHandler::new(core_format)
        .parse_triples(file)
        .map_err(|e| format!("Failed to parse '{}' as {format}: {e}", path.display()).into())
}

/// Load SHACL shapes from file using the real `oxirs_shacl::ShapeParser`
/// (full SHACL Core target/constraint discovery — `sh:NodeShape`,
/// `sh:PropertyShape`, `sh:targetClass`, `sh:property`, `sh:path`,
/// `sh:pattern`, `sh:class`, `sh:node`, `sh:or`/`sh:and`/`sh:not`,
/// `sh:closed`, qualified value shapes, etc.), not a hand-rolled subset.
fn load_shapes_file(shapes_path: &Path, format: &str) -> ToolResult<IndexMap<ShapeId, Shape>> {
    let triples = parse_triples_file(shapes_path, format)?;

    let mut store = RdfStore::new().map_err(|e| format!("Failed to create shapes store: {e}"))?;
    for triple in triples {
        store
            .insert_triple(triple)
            .map_err(|e| format!("Failed to load shape triple: {e}"))?;
    }

    let mut shape_parser = ShapeParser::new();
    let shapes_vec = shape_parser
        .parse_shapes_from_store(&store, None)
        .map_err(|e| format!("Shape parsing failed: {e}"))?;

    Ok(shapes_vec
        .into_iter()
        .map(|shape| (shape.id.clone(), shape))
        .collect())
}

/// Load data from source into a real `RdfStore` — a `File` source is parsed
/// via oxirs-core's format handlers; a `Dataset` source is opened directly
/// (read-only) via `RdfStore::open`. No hardcoded/simulated triples are ever
/// substituted for the actual dataset content.
fn load_data_source(data_source: &DataSource) -> ToolResult<RdfStore> {
    match data_source {
        DataSource::File(path, format) => {
            let triples = parse_triples_file(path, format)?;
            let mut store =
                RdfStore::new().map_err(|e| format!("Failed to create data store: {e}"))?;
            for triple in triples {
                store
                    .insert_triple(triple)
                    .map_err(|e| format!("Failed to load data triple: {e}"))?;
            }
            Ok(store)
        }
        DataSource::Dataset(path) => RdfStore::open(path)
            .map_err(|e| format!("Failed to open dataset '{}': {e}", path.display()).into()),
    }
}

/// Validate data against SHACL shapes using the real
/// `oxirs_shacl::ValidationEngine` (SHACL Core + SHACL-AF).
fn validate_data_against_shapes(
    data_store: &RdfStore,
    shapes: &IndexMap<ShapeId, Shape>,
) -> ToolResult<oxirs_shacl::ValidationReport> {
    let config = ValidationConfig::default();
    let mut engine = ValidationEngine::new(shapes, config);
    engine
        .validate_store(data_store)
        .map_err(|e| format!("SHACL validation failed: {e}").into())
}

/// Format validation report
fn format_validation_report(
    report: &oxirs_shacl::ValidationReport,
    format: &str,
) -> ToolResult<String> {
    match format {
        "text" => format_text_report(report),
        "turtle" => format_turtle_report(report),
        "json" => format_json_report(report),
        "xml" => format_xml_report(report),
        _ => Err(format!("Unsupported output format: {format}").into()),
    }
}

/// Format report as plain text
fn format_text_report(report: &oxirs_shacl::ValidationReport) -> ToolResult<String> {
    let mut output = String::new();

    output.push_str("SHACL Validation Report\n");
    output.push_str(&format!("Conforms: {}\n\n", report.conforms));

    let violations = report.violations();
    if violations.is_empty() {
        output.push_str("No validation results.\n");
    } else {
        output.push_str(&format!("Validation Results ({}):\n", violations.len()));
        output.push_str(&"=".repeat(50));
        output.push('\n');

        for (i, violation) in violations.iter().enumerate() {
            output.push_str(&format!(
                "\n{}. {} - {}\n",
                i + 1,
                format_severity(violation.result_severity),
                violation
                    .result_message
                    .as_deref()
                    .unwrap_or("(no message)")
            ));
            output.push_str(&format!("   Focus Node: {}\n", violation.focus_node));
            if let Some(ref path) = violation.result_path {
                output.push_str(&format!("   Property: {path}\n"));
            }
            if let Some(ref value) = violation.value {
                output.push_str(&format!("   Value: {value}\n"));
            }
            output.push_str(&format!("   Shape: {}\n", violation.source_shape));
            output.push_str(&format!(
                "   Constraint: {}\n",
                violation.source_constraint_component
            ));
        }
    }

    Ok(output)
}

/// Format severity for display
fn format_severity(severity: oxirs_shacl::Severity) -> &'static str {
    match severity {
        oxirs_shacl::Severity::Violation => "VIOLATION",
        oxirs_shacl::Severity::Warning => "WARNING",
        oxirs_shacl::Severity::Info => "INFO",
    }
}

/// Escape a string for embedding in a Turtle string literal.
fn turtle_escape(input: &str) -> String {
    input.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Escape a string for embedding in a JSON string literal.
fn json_escape(input: &str) -> String {
    input
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// Escape a string for embedding in XML character data.
fn xml_escape(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Format report as Turtle (SHACL `sh:ValidationReport` shape)
fn format_turtle_report(report: &oxirs_shacl::ValidationReport) -> ToolResult<String> {
    let mut output = String::new();

    output.push_str("@prefix sh: <http://www.w3.org/ns/shacl#> .\n");
    output.push_str("@prefix ex: <http://example.org/> .\n\n");

    output.push_str("ex:report a sh:ValidationReport ;\n");
    output.push_str(&format!("  sh:conforms {} ;\n", report.conforms));

    let violations = report.violations();
    if !violations.is_empty() {
        output.push_str("  sh:result\n");
        for (i, _violation) in violations.iter().enumerate() {
            output.push_str(&format!("    ex:result{i}"));
            if i < violations.len() - 1 {
                output.push_str(",\n");
            } else {
                output.push_str(" .\n\n");
            }
        }

        for (i, violation) in violations.iter().enumerate() {
            output.push_str(&format!("ex:result{i} a sh:ValidationResult ;\n"));
            output.push_str(&format!(
                "  sh:focusNode {} ;\n",
                turtle_term(&violation.focus_node.to_string())
            ));
            if let Some(ref path) = violation.result_path {
                output.push_str(&format!("  sh:resultPath <{path}> ;\n"));
            }
            if let Some(ref message) = violation.result_message {
                output.push_str(&format!(
                    "  sh:resultMessage \"{}\" ;\n",
                    turtle_escape(message)
                ));
            }
            output.push_str(&format!(
                "  sh:sourceConstraintComponent sh:{} ;\n",
                violation
                    .source_constraint_component
                    .to_string()
                    .trim_start_matches("sh:")
            ));
            output.push_str(&format!(
                "  sh:sourceShape <{}> .\n\n",
                violation.source_shape
            ));
        }
    } else {
        output.push_str(" .\n");
    }

    Ok(output)
}

/// Render an already-Display-formatted RDF term (e.g. `<iri>` or `"literal"`)
/// unchanged if it already looks like a Turtle term, otherwise wrap as an IRI.
fn turtle_term(rendered: &str) -> String {
    if rendered.starts_with('<') || rendered.starts_with('"') || rendered.starts_with("_:") {
        rendered.to_string()
    } else {
        format!("<{rendered}>")
    }
}

/// Format report as JSON
fn format_json_report(report: &oxirs_shacl::ValidationReport) -> ToolResult<String> {
    let mut output = String::new();

    output.push_str("{\n");
    output.push_str(&format!("  \"conforms\": {},\n", report.conforms));
    output.push_str("  \"results\": [\n");

    let violations = report.violations();
    for (i, violation) in violations.iter().enumerate() {
        output.push_str("    {\n");
        output.push_str(&format!(
            "      \"severity\": \"{}\",\n",
            format_severity(violation.result_severity)
        ));
        output.push_str(&format!(
            "      \"focusNode\": \"{}\",\n",
            json_escape(&violation.focus_node.to_string())
        ));
        if let Some(ref path) = violation.result_path {
            output.push_str(&format!(
                "      \"resultPath\": \"{}\",\n",
                json_escape(&path.to_string())
            ));
        }
        output.push_str(&format!(
            "      \"message\": \"{}\",\n",
            json_escape(violation.result_message.as_deref().unwrap_or(""))
        ));
        output.push_str(&format!(
            "      \"sourceConstraintComponent\": \"{}\",\n",
            json_escape(&violation.source_constraint_component.to_string())
        ));
        output.push_str(&format!(
            "      \"sourceShape\": \"{}\"\n",
            json_escape(&violation.source_shape.to_string())
        ));
        output.push_str("    }");
        if i < violations.len() - 1 {
            output.push(',');
        }
        output.push('\n');
    }

    output.push_str("  ]\n");
    output.push_str("}\n");

    Ok(output)
}

/// Format report as XML
fn format_xml_report(report: &oxirs_shacl::ValidationReport) -> ToolResult<String> {
    let mut output = String::new();

    output.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    output.push_str("<ValidationReport xmlns=\"http://www.w3.org/ns/shacl#\">\n");
    output.push_str(&format!("  <conforms>{}</conforms>\n", report.conforms));

    for violation in report.violations() {
        output.push_str("  <result>\n");
        output.push_str(&format!(
            "    <severity>{}</severity>\n",
            format_severity(violation.result_severity)
        ));
        output.push_str(&format!(
            "    <focusNode>{}</focusNode>\n",
            xml_escape(&violation.focus_node.to_string())
        ));
        if let Some(ref path) = violation.result_path {
            output.push_str(&format!(
                "    <resultPath>{}</resultPath>\n",
                xml_escape(&path.to_string())
            ));
        }
        output.push_str(&format!(
            "    <message>{}</message>\n",
            xml_escape(violation.result_message.as_deref().unwrap_or(""))
        ));
        output.push_str(&format!(
            "    <sourceConstraintComponent>{}</sourceConstraintComponent>\n",
            xml_escape(&violation.source_constraint_component.to_string())
        ));
        output.push_str(&format!(
            "    <sourceShape>{}</sourceShape>\n",
            xml_escape(&violation.source_shape.to_string())
        ));
        output.push_str("  </result>\n");
    }

    output.push_str("</ValidationReport>\n");

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_temp_path(label: &str, ext: &str) -> PathBuf {
        let n = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "oxirs_shacl_tool_test_{label}_{}_{}.{ext}",
            std::process::id(),
            n
        ))
    }

    const SHAPES_TTL: &str = r#"
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:PersonShape a sh:NodeShape ;
    sh:targetClass ex:Person ;
    sh:property [
        sh:path ex:name ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] .
"#;

    #[test]
    fn test_load_shapes_file_parses_real_shacl_shapes() {
        let path = unique_temp_path("shapes", "ttl");
        std::fs::write(&path, SHAPES_TTL).unwrap();

        let shapes = load_shapes_file(&path, "turtle").expect("parse shapes");
        // The real oxirs_shacl::ShapeParser returns 2 entries here: the named
        // `ex:PersonShape` NodeShape itself, plus the inline `sh:property [ ... ]`
        // blank-node PropertyShape under a generated IRI. This is not a
        // fabrication artifact — `ValidationEngine::validate_node_against_shape`
        // (engine/oxirs-shacl/src/validation/engine.rs) resolves
        // `shape.property_shapes` IDs by looking them up in this exact map, so the
        // inline property shape must be present for `sh:minCount`/`sh:maxCount`
        // constraints to be enforced at all (see
        // `test_validate_nonconforming_data_reports_real_violation` below, which
        // only detects the missing-`ex:name` violation because of this lookup).
        assert_eq!(
            shapes.len(),
            2,
            "must load the real NodeShape plus its inline sh:property blank-node \
             PropertyShape, not a fabricated default"
        );
        assert!(shapes.contains_key(&ShapeId::new("http://example.org/PersonShape")));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_validate_conforming_data_real_engine() {
        let shapes_path = unique_temp_path("shapes_conform", "ttl");
        std::fs::write(&shapes_path, SHAPES_TTL).unwrap();
        let shapes = load_shapes_file(&shapes_path, "turtle").expect("parse shapes");

        let data_path = unique_temp_path("data_conform", "ttl");
        std::fs::write(
            &data_path,
            "@prefix ex: <http://example.org/> .\nex:alice a ex:Person ; ex:name \"Alice\" .\n",
        )
        .unwrap();
        let data_store =
            load_data_source(&DataSource::File(data_path.clone(), "turtle".to_string()))
                .expect("load data");

        let report = validate_data_against_shapes(&data_store, &shapes).expect("validate");
        assert!(
            report.conforms,
            "data satisfying minCount=1/maxCount=1 must conform"
        );
        assert!(report.violations().is_empty());

        let _ = std::fs::remove_file(&shapes_path);
        let _ = std::fs::remove_file(&data_path);
    }

    #[test]
    fn test_validate_nonconforming_data_reports_real_violation() {
        let shapes_path = unique_temp_path("shapes_violate", "ttl");
        std::fs::write(&shapes_path, SHAPES_TTL).unwrap();
        let shapes = load_shapes_file(&shapes_path, "turtle").expect("parse shapes");

        // Person with no ex:name at all -> violates sh:minCount 1.
        let data_path = unique_temp_path("data_violate", "ttl");
        std::fs::write(
            &data_path,
            "@prefix ex: <http://example.org/> .\nex:bob a ex:Person .\n",
        )
        .unwrap();
        let data_store =
            load_data_source(&DataSource::File(data_path.clone(), "turtle".to_string()))
                .expect("load data");

        let report = validate_data_against_shapes(&data_store, &shapes).expect("validate");
        assert!(
            !report.conforms,
            "missing required ex:name must NOT conform (real engine, not the old 2-hardcoded-triples simulation)"
        );
        assert!(!report.violations().is_empty());

        let _ = std::fs::remove_file(&shapes_path);
        let _ = std::fs::remove_file(&data_path);
    }

    #[test]
    fn test_load_data_source_dataset_uses_real_dataset_not_hardcoded_triples() {
        let dataset_dir = unique_temp_path("dataset", "dir");
        std::fs::create_dir_all(&dataset_dir).unwrap();

        let seed_path = unique_temp_path("dataset_seed", "ttl");
        std::fs::write(
            &seed_path,
            "@prefix ex: <http://example.org/> .\nex:x ex:y ex:z .\nex:x ex:y2 ex:z2 .\nex:x ex:y3 ex:z3 .\n",
        )
        .unwrap();
        let seed_triples = parse_triples_file(&seed_path, "turtle").expect("parse seed");

        {
            let mut store = RdfStore::open(&dataset_dir).expect("open dataset");
            for t in seed_triples {
                store.insert_triple(t).expect("insert");
            }
            store.flush().expect("flush");
        }

        let store =
            load_data_source(&DataSource::Dataset(dataset_dir.clone())).expect("load dataset");
        let quads = store.quads().expect("quads");
        // The old implementation always returned exactly 2 hardcoded triples
        // regardless of dataset content; assert we see the real 3.
        assert_eq!(
            quads.len(),
            3,
            "must reflect the real dataset content, not 2 hardcoded triples"
        );

        let _ = std::fs::remove_file(&seed_path);
        let _ = std::fs::remove_dir_all(&dataset_dir);
    }
}
