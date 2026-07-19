//! RDF Concatenation Tool
//!
//! Concatenates multiple RDF files and optionally converts format.

use super::{compression, utils, ToolResult};
use crate::export::{ExportFormat, Exporter};
use oxirs_core::format::{FormatHandler, JsonLdProfileSet, RdfFormat as CoreRdfFormat};
use std::fs::File;
use std::io::{self, Cursor, Read, Write};
use std::path::{Path, PathBuf};

/// Supported input format detection patterns
const TURTLE_PATTERNS: &[&str] = &["@prefix", "@base", "a ", " a ", ":"];
const NTRIPLES_PATTERNS: &[&str] = &["<", "> <", "> \"", "> .", "_:"];
const RDFXML_PATTERNS: &[&str] = &["<?xml", "<rdf:", "xmlns"];
const JSONLD_PATTERNS: &[&str] = &["@context", "@id", "@type", "@graph"];

/// Detect RDF format from a decompressed content sample (first 10 lines).
///
/// Operating on the already-decompressed content means `.gz` inputs are
/// classified from their real inner content, not the gzip wrapper.
fn detect_format_from_content(content: &str, file_path: &Path) -> ExportFormat {
    // Only the first 10 lines are needed to classify the format.
    let sample = content
        .lines()
        .take(10)
        .map(|line| line.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ");

    // Check for JSON-LD first (most specific)
    if JSONLD_PATTERNS
        .iter()
        .any(|pattern| sample.contains(pattern))
    {
        return ExportFormat::JsonLd;
    }

    // Check for RDF/XML
    if RDFXML_PATTERNS
        .iter()
        .any(|pattern| sample.contains(pattern))
    {
        return ExportFormat::RdfXml;
    }

    // Check for Turtle
    if TURTLE_PATTERNS
        .iter()
        .any(|pattern| sample.contains(pattern))
    {
        return ExportFormat::Turtle;
    }

    // Check for N-Triples (most generic, check last)
    if NTRIPLES_PATTERNS
        .iter()
        .any(|pattern| sample.contains(pattern))
    {
        return ExportFormat::NTriples;
    }

    // Default to Turtle if unable to detect
    eprintln!("Warning: Unable to detect format for {file_path:?}, assuming Turtle");
    ExportFormat::Turtle
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
    fn to_ntriples(&self) -> String {
        format!("{} {} {} .", self.subject, self.predicate, self.object)
    }
}

/// Map the tool's `ExportFormat` to oxirs-core's parser-facing `RdfFormat`.
fn export_format_to_core(format: ExportFormat) -> CoreRdfFormat {
    match format {
        ExportFormat::Turtle => CoreRdfFormat::Turtle,
        ExportFormat::NTriples => CoreRdfFormat::NTriples,
        ExportFormat::RdfXml => CoreRdfFormat::RdfXml,
        ExportFormat::JsonLd => CoreRdfFormat::JsonLd {
            profile: JsonLdProfileSet::empty(),
        },
        ExportFormat::TriG => CoreRdfFormat::TriG,
        ExportFormat::NQuads => CoreRdfFormat::NQuads,
    }
}

/// Read RDF data from file, parsing the real content via oxirs-core's format
/// handlers (Turtle, N-Triples, N-Quads, TriG, RDF/XML, JSON-LD). Every
/// triple returned here comes from the file's actual content — no
/// placeholder/synthesized triples are ever emitted.
fn read_rdf_file(file_path: &Path) -> Result<Vec<RdfTriple>, Box<dyn std::error::Error>> {
    // Transparent decompression: `.gz` inputs are inflated on the fly via the
    // crate's single compression module; plain files pass through unchanged.
    // Corrupt gzip surfaces an explicit error. Decompress once, then reuse the
    // same bytes for both format detection and real parsing.
    let mut reader = compression::open_reader(file_path)?;
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;

    let format = detect_format_from_content(&String::from_utf8_lossy(&bytes), file_path);
    let core_format = export_format_to_core(format);

    let handler = FormatHandler::new(core_format);
    let parsed = handler.parse_triples(Cursor::new(bytes)).map_err(|e| {
        format!(
            "Failed to parse '{}' as {:?}: {e}",
            file_path.display(),
            format
        )
    })?;

    Ok(parsed
        .into_iter()
        .map(|t| RdfTriple {
            subject: t.subject().to_string(),
            predicate: t.predicate().to_string(),
            object: t.object().to_string(),
        })
        .collect())
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

    fn unique_temp_path(label: &str, ext: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "oxirs_rdfcat_test_{label}_{}_{}.{ext}",
            std::process::id(),
            n
        ))
    }

    #[test]
    fn test_read_rdf_file_turtle_real_data_not_placeholder() {
        let path = unique_temp_path("turtle_real", "ttl");
        std::fs::write(
            &path,
            "@prefix ex: <http://example.org/> .\nex:alice ex:knows ex:bob .\nex:bob ex:knows ex:carol .\n",
        )
        .unwrap();

        let triples = read_rdf_file(&path).expect("should parse real turtle content");
        assert_eq!(
            triples.len(),
            2,
            "must reflect the real triple count, not a single placeholder"
        );
        for t in &triples {
            assert!(
                !t.predicate.contains("example.org/source"),
                "must not emit the old fabricated placeholder predicate"
            );
        }
        assert!(triples.iter().any(|t| t.subject.contains("alice")));

        let _ = std::fs::remove_file(&path);
    }

    /// Transparent `.gz` handling: a gzipped Turtle file must be inflated on
    /// the fly and its real triples parsed, exactly as an uncompressed file.
    #[test]
    fn test_read_rdf_file_transparent_gzip() {
        use std::io::Write;

        let path = unique_temp_path("gz_turtle", "ttl.gz");
        {
            let mut writer = compression::open_writer(&path, compression::CompressionFormat::Gzip)
                .expect("open gzip writer");
            writer
                .write_all(
                    b"@prefix ex: <http://example.org/> .\n\
                      ex:alice ex:knows ex:bob .\n\
                      ex:bob ex:knows ex:carol .\n",
                )
                .expect("write turtle payload");
            writer.flush().expect("flush gzip writer");
        }

        let triples = read_rdf_file(&path).expect("gzipped turtle should parse transparently");
        assert_eq!(
            triples.len(),
            2,
            "gzipped turtle must be decompressed and its real triples parsed"
        );
        assert!(triples.iter().any(|t| t.subject.contains("alice")));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_read_rdf_file_invalid_content_errors_instead_of_placeholder() {
        let path = unique_temp_path("invalid", "rdf");
        // Content matching the RDF/XML detection heuristic but not valid XML;
        // real parsing must surface an error rather than silently emitting a
        // synthesized placeholder triple.
        std::fs::write(&path, "<?xml this is not well-formed rdf/xml at all").unwrap();

        let result = read_rdf_file(&path);
        assert!(
            result.is_err(),
            "malformed content must fail loudly, not produce a fabricated triple"
        );

        let _ = std::fs::remove_file(&path);
    }
}
