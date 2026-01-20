//! RDF Concatenation Tool
//!
//! Concatenates multiple RDF files and optionally converts format.

use super::{utils, ToolResult};
use crate::export::{ExportFormat, Exporter};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;

/// Supported input format detection patterns
const TURTLE_PATTERNS: &[&str] = &["@prefix", "@base", "a ", " a ", ":"];
const NTRIPLES_PATTERNS: &[&str] = &["<", "> <", "> \"", "> .", "_:"];
const RDFXML_PATTERNS: &[&str] = &["<?xml", "<rdf:", "xmlns"];
const JSONLD_PATTERNS: &[&str] = &["@context", "@id", "@type", "@graph"];

/// Detect RDF format from file content
fn detect_format(file_path: &PathBuf) -> Result<ExportFormat, Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    // Read first few lines to detect format
    let mut sample_lines = Vec::new();
    for (i, line_result) in reader.lines().enumerate() {
        if i >= 10 {
            break;
        } // Only check first 10 lines
        let line = line_result?;
        sample_lines.push(line.to_lowercase());
    }

    let content = sample_lines.join(" ");

    // Check for JSON-LD first (most specific)
    if JSONLD_PATTERNS
        .iter()
        .any(|pattern| content.contains(pattern))
    {
        return Ok(ExportFormat::JsonLd);
    }

    // Check for RDF/XML
    if RDFXML_PATTERNS
        .iter()
        .any(|pattern| content.contains(pattern))
    {
        return Ok(ExportFormat::RdfXml);
    }

    // Check for Turtle
    if TURTLE_PATTERNS
        .iter()
        .any(|pattern| content.contains(pattern))
    {
        return Ok(ExportFormat::Turtle);
    }

    // Check for N-Triples (most generic, check last)
    if NTRIPLES_PATTERNS
        .iter()
        .any(|pattern| content.contains(pattern))
    {
        return Ok(ExportFormat::NTriples);
    }

    // Default to Turtle if unable to detect
    eprintln!("Warning: Unable to detect format for {file_path:?}, assuming Turtle");
    Ok(ExportFormat::Turtle)
}

/// Parse format string to ExportFormat
fn parse_output_format(format_str: &str) -> Result<ExportFormat, Box<dyn std::error::Error>> {
    match format_str.to_lowercase().as_str() {
        "turtle" | "ttl" => Ok(ExportFormat::Turtle),
        "ntriples" | "nt" => Ok(ExportFormat::NTriples),
        "rdfxml" | "rdf" | "xml" => Ok(ExportFormat::RdfXml),
        "jsonld" | "json-ld" | "json" => Ok(ExportFormat::JsonLd),
        "trig" => Ok(ExportFormat::TriG),
        "nquads" | "nq" => Ok(ExportFormat::NQuads),
        _ => Err(format!("Unsupported format: {format_str}").into()),
    }
}

/// Simple RDF triple structure for concatenation
#[derive(Debug, Clone)]
struct RdfTriple {
    subject: String,
    predicate: String,
    object: String,
}

impl RdfTriple {
    fn from_ntriples_line(line: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return Err("Empty or comment line".into());
        }

        // Simple parsing - split by whitespace, recombine properly
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Err("Invalid N-Triples line".into());
        }

        let subject = parts[0].to_string();
        let predicate = parts[1].to_string();
        let object = parts[2..parts.len() - 1].join(" "); // Everything except the final "."

        Ok(RdfTriple {
            subject,
            predicate,
            object,
        })
    }

    fn to_ntriples(&self) -> String {
        format!("{} {} {} .", self.subject, self.predicate, self.object)
    }
}

/// Read RDF data from file (simplified implementation)
fn read_rdf_file(file_path: &PathBuf) -> Result<Vec<RdfTriple>, Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut triples = Vec::new();

    let format = detect_format(file_path)?;

    match format {
        ExportFormat::NTriples => {
            for line_result in reader.lines() {
                let line = line_result?;
                if let Ok(triple) = RdfTriple::from_ntriples_line(&line) {
                    triples.push(triple);
                }
            }
        }
        _ => {
            // For other formats, return a placeholder
            eprintln!(
                "Warning: Full parsing for {format:?} format not implemented, creating placeholder"
            );
            triples.push(RdfTriple {
                subject: format!("<file://{}>", file_path.display()),
                predicate: "<http://example.org/source>".to_string(),
                object: format!(
                    "\"{}\"",
                    file_path.file_name().unwrap_or_default().to_string_lossy()
                ),
            });
        }
    }

    Ok(triples)
}

/// Write concatenated RDF data
fn write_concatenated_output<W: Write + 'static>(
    triples: &[RdfTriple],
    output_format: ExportFormat,
    mut writer: W,
) -> Result<(), Box<dyn std::error::Error>> {
    match output_format {
        ExportFormat::NTriples => {
            for triple in triples {
                writeln!(writer, "{}", triple.to_ntriples())?;
            }
        }
        ExportFormat::Turtle => {
            writeln!(writer, "# Concatenated RDF data in Turtle format")?;
            writeln!(writer, "@prefix ex: <http://example.org/> .")?;
            writeln!(writer)?;
            for triple in triples {
                // Simple conversion - would need proper Turtle serialization
                writeln!(writer, "{}", triple.to_ntriples())?;
            }
        }
        _ => {
            // For other formats, use the Exporter
            let exporter = Exporter::new().with_format(output_format);
            let data = triples
                .iter()
                .map(|t| t.to_ntriples())
                .collect::<Vec<_>>()
                .join("\n");
            exporter.export_to_writer(&data, writer)?;
        }
    }

    Ok(())
}

/// Run rdfcat command
pub async fn run(files: Vec<PathBuf>, format: String, output: Option<PathBuf>) -> ToolResult {
    println!("RDF Concatenation Tool");
    println!("Input files: {}", files.len());
    println!("Output format: {format}");

    if !utils::is_supported_output_format(&format) {
        return Err(format!("Unsupported output format: {format}").into());
    }

    // Parse output format
    let output_format = parse_output_format(&format)?;

    // Collect all triples from input files
    let mut all_triples = Vec::new();
    let mut total_triples = 0;

    for file_path in &files {
        if !file_path.exists() {
            eprintln!("Warning: File {file_path:?} does not exist, skipping");
            continue;
        }

        match read_rdf_file(file_path) {
            Ok(triples) => {
                let count = triples.len();
                total_triples += count;
                all_triples.extend(triples);
                println!("Read {count} triples from {file_path:?}");
            }
            Err(e) => {
                eprintln!("Error reading {file_path:?}: {e}");
                if files.len() == 1 {
                    return Err(e);
                }
                // Continue with other files if multiple files
            }
        }
    }

    if all_triples.is_empty() {
        return Err("No RDF data found in input files".into());
    }

    println!("Total triples collected: {total_triples}");

    // Write output
    match output {
        Some(output_path) => {
            let file = File::create(&output_path)?;
            write_concatenated_output(&all_triples, output_format, file)?;
            println!("Concatenated data written to {output_path:?}");
        }
        None => {
            let stdout = io::stdout();
            write_concatenated_output(&all_triples, output_format, stdout.lock())?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_output_format() {
        assert!(matches!(
            parse_output_format("turtle").unwrap(),
            ExportFormat::Turtle
        ));
        assert!(matches!(
            parse_output_format("nt").unwrap(),
            ExportFormat::NTriples
        ));
        assert!(matches!(
            parse_output_format("json-ld").unwrap(),
            ExportFormat::JsonLd
        ));
        assert!(parse_output_format("invalid").is_err());
    }

    #[test]
    fn test_rdf_triple_parsing() {
        let line = "<http://example.org/s> <http://example.org/p> \"object\" .";
        let triple = RdfTriple::from_ntriples_line(line).unwrap();
        assert_eq!(triple.subject, "<http://example.org/s>");
        assert_eq!(triple.predicate, "<http://example.org/p>");
        assert_eq!(triple.object, "\"object\"");
    }

    #[test]
    fn test_write_ntriples_output() {
        let triples = vec![
            RdfTriple {
                subject: "<http://example.org/s1>".to_string(),
                predicate: "<http://example.org/p1>".to_string(),
                object: "\"object1\"".to_string(),
            },
            RdfTriple {
                subject: "<http://example.org/s2>".to_string(),
                predicate: "<http://example.org/p2>".to_string(),
                object: "\"object2\"".to_string(),
            },
        ];

        // Create a temporary file for testing
        use std::env::temp_dir;
        let temp_file = temp_dir().join("test_rdfcat_output.nt");
        let file = File::create(&temp_file).unwrap();
        write_concatenated_output(&triples, ExportFormat::NTriples, file).unwrap();

        // Read back the output
        let output_str = std::fs::read_to_string(&temp_file).unwrap();
        assert!(output_str.contains("<http://example.org/s1>"));
        assert!(output_str.contains("<http://example.org/s2>"));

        // Cleanup
        let _ = std::fs::remove_file(&temp_file);
    }
}
