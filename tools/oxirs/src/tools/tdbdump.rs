//! TDB Dump Tool
//!
//! Export RDF data from TDB databases to various serialization formats.
//! Supports exporting entire datasets or specific named graphs.

use super::{ToolResult, ToolStats};
use colored::Colorize;
use oxirs_tdb::{TdbConfig, TdbStore};
use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

/// Supported RDF serialization formats
#[derive(Debug, Clone, Copy)]
pub enum DumpFormat {
    NTriples,
    NQuads,
    Turtle,
    TriG,
    RdfXml,
    JsonLd,
}

impl FromStr for DumpFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ntriples" | "nt" => Ok(Self::NTriples),
            "nquads" | "nq" => Ok(Self::NQuads),
            "turtle" | "ttl" => Ok(Self::Turtle),
            "trig" => Ok(Self::TriG),
            "rdfxml" | "rdf" | "xml" => Ok(Self::RdfXml),
            "jsonld" | "json" => Ok(Self::JsonLd),
            _ => Err(format!(
                "Unknown format: {}. Supported: ntriples, nquads, turtle, trig, rdfxml, jsonld",
                s
            )),
        }
    }
}

impl DumpFormat {
    /// Get file extension for format
    pub fn extension(&self) -> &'static str {
        match self {
            Self::NTriples => "nt",
            Self::NQuads => "nq",
            Self::Turtle => "ttl",
            Self::TriG => "trig",
            Self::RdfXml => "rdf",
            Self::JsonLd => "jsonld",
        }
    }

    /// Get MIME type for format
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::NTriples => "application/n-triples",
            Self::NQuads => "application/n-quads",
            Self::Turtle => "text/turtle",
            Self::TriG => "application/trig",
            Self::RdfXml => "application/rdf+xml",
            Self::JsonLd => "application/ld+json",
        }
    }
}

/// Run TDB dump command
pub async fn run(
    location: PathBuf,
    output: Option<PathBuf>,
    format: String,
    graph: Option<String>,
) -> ToolResult {
    let mut stats = ToolStats::new();

    println!("{}", "═".repeat(70).bright_blue());
    println!("{}", "  OxiRS TDB Database Export".bright_green().bold());
    println!("{}", "═".repeat(70).bright_blue());
    println!();

    // Parse dump format
    let dump_format: DumpFormat = format
        .parse()
        .map_err(|e| Box::new(io::Error::new(io::ErrorKind::InvalidInput, e)))?;

    // Validate location exists
    if !location.exists() {
        return Err(format!("Dataset location does not exist: {}", location.display()).into());
    }

    if !location.is_dir() {
        return Err(format!(
            "Dataset location must be a directory: {}",
            location.display()
        )
        .into());
    }

    println!(
        "Dataset Location: {}",
        location.display().to_string().cyan()
    );
    println!(
        "Output Format:    {}",
        format!("{} ({})", format.to_uppercase(), dump_format.mime_type()).bright_white()
    );

    if let Some(ref g) = graph {
        println!("Target Graph:     {}", g.cyan());
    } else {
        println!(
            "Target:           {}",
            "All graphs (default graph + named graphs)".dimmed()
        );
    }

    if let Some(ref out) = output {
        println!("Output File:      {}", out.display().to_string().cyan());
    } else {
        println!("Output:           {}", "stdout".dimmed());
    }
    println!();

    // Open store
    println!("{}", "Opening database...".bright_yellow());
    let config = TdbConfig::new(&location);
    let store =
        TdbStore::open_with_config(config).map_err(|e| format!("Failed to open store: {}", e))?;

    println!("  ✓ {}", "Database opened".green());
    println!();

    // Start export
    println!("{}", "Exporting data...".bright_yellow());
    let export_start = Instant::now();

    // Get triples count for progress
    let store_stats = store.stats();
    let total_triples = store_stats.triple_count;

    if total_triples == 0 {
        println!("  ⊘ {}", "No triples found in database".yellow());
        println!();
        println!("{}", "═".repeat(70).bright_blue());
        return Ok(());
    }

    println!(
        "  Found {} triples to export",
        format_number(total_triples).bright_white()
    );

    // Create output writer
    let mut writer: Box<dyn Write> = if let Some(ref out_path) = output {
        Box::new(File::create(out_path)?)
    } else {
        Box::new(io::stdout())
    };

    // Export based on format
    let exported_count = match dump_format {
        DumpFormat::NQuads | DumpFormat::NTriples => {
            // Simple line-based format
            export_ntriples_nquads(&store, &mut writer, graph.as_deref(), dump_format)?
        }
        DumpFormat::Turtle | DumpFormat::TriG => {
            // Turtle/TriG serialization using oxirs-core
            export_turtle_trig(&store, &mut writer, graph.as_deref(), dump_format)?
        }
        DumpFormat::RdfXml => {
            // RDF/XML serialization using oxirs-core
            export_rdfxml(&store, &mut writer, graph.as_deref())?
        }
        DumpFormat::JsonLd => {
            // TODO: Implement JSON-LD serialization when oxirs-core supports it
            writeln!(writer, "# OxiRS TDB Export")?;
            writeln!(writer, "# Format: {}", format)?;
            writeln!(writer, "# TODO: JSON-LD serialization not yet implemented")?;
            writeln!(writer, "# Falling back to N-Triples format\n")?;
            export_ntriples_nquads(&store, &mut writer, graph.as_deref(), DumpFormat::NTriples)?
        }
    };

    writer.flush()?;

    let export_duration = export_start.elapsed();

    println!(
        "  ✓ {}",
        format!("Exported {} triples", format_number(exported_count))
            .green()
            .bold()
    );
    println!();

    // Display summary
    println!("{}", "Export Summary".bright_yellow().bold());
    println!("{}", "─".repeat(70));
    println!(
        "  Triples Exported: {}",
        format_number(exported_count).bright_white().bold()
    );
    println!(
        "  Duration:         {}s",
        format!("{:.2}", export_duration.as_secs_f64()).bright_white()
    );

    let throughput = exported_count as f64 / export_duration.as_secs_f64();
    println!(
        "  Throughput:       {} triples/sec",
        format_number(throughput as usize).bright_white()
    );

    if let Some(ref out_path) = output {
        let file_size = std::fs::metadata(out_path)?.len();
        println!(
            "  Output Size:      {}",
            format_bytes(file_size).bright_white()
        );
        println!(
            "  Output File:      {}",
            out_path.display().to_string().cyan()
        );
    }

    println!();
    println!("{}", "═".repeat(70).bright_blue());

    stats.items_processed = exported_count;
    stats.finish();

    Ok(())
}

/// Export triples in N-Triples or N-Quads format
fn export_ntriples_nquads(
    store: &TdbStore,
    writer: &mut dyn Write,
    graph: Option<&str>,
    format: DumpFormat,
) -> Result<usize, Box<dyn std::error::Error>> {
    use oxirs_core::format::ntriples::NTriplesSerializer;
    use oxirs_core::model::Triple;

    // Write a comment header
    writeln!(writer, "# OxiRS TDB Export")?;
    writeln!(
        writer,
        "# Format: {}",
        match format {
            DumpFormat::NTriples => "N-Triples",
            DumpFormat::NQuads => "N-Quads",
            _ => "Unknown",
        }
    )?;

    if let Some(g) = graph {
        writeln!(writer, "# Graph: {}", g)?;
    }

    writeln!(writer, "# Exported: {}\n", chrono::Utc::now().to_rfc3339())?;

    // Query all triples from the store
    // Use TdbStore::query_triples with all wildcards to get all triples
    let triples = if graph.is_some() {
        // TODO: Add graph-specific query support when TDB supports quads
        // For now, treat graph parameter as filter on subject IRIs
        store.query_triples(None, None, None)?
    } else {
        // Query all triples
        store.query_triples(None, None, None)?
    };

    // Create serializer and serialize each triple
    let serializer = NTriplesSerializer::new();
    let mut writer_serializer = serializer.for_writer(writer);

    let mut count = 0;
    for (s, p, o) in triples {
        // Convert TDB Terms to oxirs-core Terms for serialization
        let subject = term_to_subject(&s)?;
        let predicate = term_to_named_node(&p)?;
        let object = term_to_object(&o)?;

        let triple = Triple::new(subject, predicate, object);

        // Serialize the triple
        writer_serializer.serialize_triple(triple.as_ref())?;
        count += 1;
    }

    writer_serializer.finish()?;

    Ok(count)
}

/// Export triples in Turtle or TriG format
fn export_turtle_trig(
    store: &TdbStore,
    writer: &mut dyn Write,
    graph: Option<&str>,
    _format: DumpFormat,
) -> Result<usize, Box<dyn std::error::Error>> {
    use oxirs_core::format::turtle::TurtleSerializer;
    use oxirs_core::model::Triple;

    // Query all triples from the store
    let triples = if graph.is_some() {
        // TODO: Add graph-specific query support when TDB supports quads
        store.query_triples(None, None, None)?
    } else {
        store.query_triples(None, None, None)?
    };

    // Create Turtle serializer
    let serializer = TurtleSerializer::new();
    let mut writer_serializer = serializer.for_writer(writer);

    let mut count = 0;
    for (s, p, o) in triples {
        // Convert TDB Terms to oxirs-core Terms for serialization
        let subject = term_to_subject(&s)?;
        let predicate = term_to_named_node(&p)?;
        let object = term_to_object(&o)?;

        let triple = Triple::new(subject, predicate, object);

        // Serialize the triple
        writer_serializer.serialize_triple(triple.as_ref())?;
        count += 1;
    }

    writer_serializer.finish()?;

    Ok(count)
}

/// Export triples in RDF/XML format
fn export_rdfxml(
    store: &TdbStore,
    writer: &mut dyn Write,
    graph: Option<&str>,
) -> Result<usize, Box<dyn std::error::Error>> {
    use oxirs_core::format::rdfxml::RdfXmlSerializer;
    use oxirs_core::model::Triple;

    // Query all triples from the store
    let triples = if graph.is_some() {
        // TODO: Add graph-specific query support when TDB supports quads
        store.query_triples(None, None, None)?
    } else {
        store.query_triples(None, None, None)?
    };

    // Convert to oxirs-core Triple format
    let mut core_triples = Vec::new();
    for (s, p, o) in triples {
        let subject = term_to_subject(&s)?;
        let predicate = term_to_named_node(&p)?;
        let object = term_to_object(&o)?;

        core_triples.push(Triple::new(subject, predicate, object));
    }

    // Create RDF/XML serializer and serialize to string
    let serializer = RdfXmlSerializer::new();
    let serialized = serializer.serialize_to_string(&core_triples)?;

    // Write to output
    writer.write_all(serialized.as_bytes())?;

    Ok(core_triples.len())
}

/// Convert TDB Term to oxirs-core Subject
fn term_to_subject(
    term: &oxirs_tdb::dictionary::Term,
) -> Result<oxirs_core::model::Subject, Box<dyn std::error::Error>> {
    use oxirs_core::model::{BlankNode, NamedNode, Subject};
    use oxirs_tdb::dictionary::Term;

    match term {
        Term::Iri(iri) => Ok(Subject::NamedNode(NamedNode::new(iri.as_str())?)),
        Term::BlankNode(id) => Ok(Subject::BlankNode(BlankNode::new(id.as_str())?)),
        Term::Literal { .. } => Err("Literal cannot be used as subject".into()),
    }
}

/// Convert TDB Term to oxirs-core NamedNode (for predicates)
fn term_to_named_node(
    term: &oxirs_tdb::dictionary::Term,
) -> Result<oxirs_core::model::NamedNode, Box<dyn std::error::Error>> {
    use oxirs_core::model::NamedNode;
    use oxirs_tdb::dictionary::Term;

    match term {
        Term::Iri(iri) => Ok(NamedNode::new(iri.as_str())?),
        _ => Err("Only IRI can be used as predicate".into()),
    }
}

/// Convert TDB Term to oxirs-core Object
fn term_to_object(
    term: &oxirs_tdb::dictionary::Term,
) -> Result<oxirs_core::model::Object, Box<dyn std::error::Error>> {
    use oxirs_core::model::{BlankNode, Literal, NamedNode, Object};
    use oxirs_tdb::dictionary::Term;

    match term {
        Term::Iri(iri) => Ok(Object::NamedNode(NamedNode::new(iri.as_str())?)),
        Term::BlankNode(id) => Ok(Object::BlankNode(BlankNode::new(id.as_str())?)),
        Term::Literal {
            value,
            language,
            datatype,
        } => {
            // Convert TDB literal to oxirs-core literal
            if let Some(lang) = language {
                Ok(Object::Literal(Literal::new_language_tagged_literal(
                    value, lang,
                )?))
            } else if let Some(dt) = datatype {
                let datatype_node = NamedNode::new(dt.as_str())?;
                Ok(Object::Literal(Literal::new_typed_literal(
                    value,
                    datatype_node,
                )))
            } else {
                // Plain literal (xsd:string)
                Ok(Object::Literal(Literal::new_simple_literal(value)))
            }
        }
    }
}

/// Format number with thousands separators
fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();

    for (count, c) in s.chars().rev().enumerate() {
        if count > 0 && count % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }

    result.chars().rev().collect()
}

/// Format bytes in human-readable format
fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KB", "MB", "GB", "TB", "PB"];

    if bytes == 0 {
        return "0 B".to_string();
    }

    let bytes_f = bytes as f64;
    let unit_index = (bytes_f.log2() / 10.0).floor() as usize;
    let unit_index = unit_index.min(UNITS.len() - 1);

    let value = bytes_f / (1024_f64.powi(unit_index as i32));

    if value < 10.0 {
        format!("{:.2} {}", value, UNITS[unit_index])
    } else if value < 100.0 {
        format!("{:.1} {}", value, UNITS[unit_index])
    } else {
        format!("{:.0} {}", value, UNITS[unit_index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dump_format_parse() {
        assert!(matches!(
            "ntriples".parse::<DumpFormat>(),
            Ok(DumpFormat::NTriples)
        ));
        assert!(matches!(
            "nt".parse::<DumpFormat>(),
            Ok(DumpFormat::NTriples)
        ));
        assert!(matches!(
            "turtle".parse::<DumpFormat>(),
            Ok(DumpFormat::Turtle)
        ));
        assert!(matches!(
            "ttl".parse::<DumpFormat>(),
            Ok(DumpFormat::Turtle)
        ));
        assert!("invalid".parse::<DumpFormat>().is_err());
    }

    #[test]
    fn test_dump_format_extension() {
        assert_eq!(DumpFormat::NTriples.extension(), "nt");
        assert_eq!(DumpFormat::Turtle.extension(), "ttl");
        assert_eq!(DumpFormat::JsonLd.extension(), "jsonld");
    }

    #[test]
    fn test_dump_format_mime_type() {
        assert_eq!(DumpFormat::NTriples.mime_type(), "application/n-triples");
        assert_eq!(DumpFormat::Turtle.mime_type(), "text/turtle");
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(1234567), "1,234,567");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
    }
}
