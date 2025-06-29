//! Advanced Format Detection and Conversion Tools
//!
//! This module provides comprehensive format detection, validation, and conversion
//! capabilities for the Oxide CLI toolkit, supporting automatic MIME type detection,
//! content-based analysis, and intelligent format conversion.

use crate::{
    cli::{error::CliError, output::OutputFormatter},
    config::OxideConfig,
};
use oxirs_core::{
    format::{FormatHandler, RdfFormat, RdfParseError},
    model::Graph,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};
use tracing::{debug, info, warn};

/// Supported file formats with detection metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatInfo {
    pub format: RdfFormat,
    pub mime_types: Vec<String>,
    pub file_extensions: Vec<String>,
    pub magic_bytes: Option<Vec<u8>>,
    pub confidence_markers: Vec<String>,
    pub encoding_support: Vec<String>,
    pub compression_support: Vec<String>,
}

/// Format detection result with confidence score
#[derive(Debug, Clone)]
pub struct DetectionResult {
    pub format: RdfFormat,
    pub confidence: f32,
    pub detection_method: DetectionMethod,
    pub encoding: Option<String>,
    pub compression: Option<String>,
    pub additional_info: HashMap<String, String>,
}

/// Detection method used to identify format
#[derive(Debug, Clone)]
pub enum DetectionMethod {
    FileExtension,
    MimeType,
    ContentAnalysis,
    MagicBytes,
    CombinedHeuristics,
}

/// Format detection and conversion engine
pub struct FormatDetector {
    format_registry: HashMap<RdfFormat, FormatInfo>,
    content_patterns: HashMap<RdfFormat, Vec<ContentPattern>>,
    mime_mappings: HashMap<String, RdfFormat>,
    extension_mappings: HashMap<String, RdfFormat>,
}

/// Content pattern for format detection
#[derive(Debug, Clone)]
struct ContentPattern {
    pattern: String,
    weight: f32,
    required: bool,
}

/// Conversion options
#[derive(Debug, Clone)]
pub struct ConversionOptions {
    pub input_format: Option<RdfFormat>,
    pub output_format: RdfFormat,
    pub input_encoding: Option<String>,
    pub output_encoding: Option<String>,
    pub validate_input: bool,
    pub preserve_prefixes: bool,
    pub optimize_output: bool,
    pub chunk_size: Option<usize>,
}

impl Default for ConversionOptions {
    fn default() -> Self {
        Self {
            input_format: None,
            output_format: RdfFormat::Turtle,
            input_encoding: None,
            output_encoding: Some("UTF-8".to_string()),
            validate_input: true,
            preserve_prefixes: true,
            optimize_output: false,
            chunk_size: None,
        }
    }
}

impl FormatDetector {
    /// Create a new format detector with comprehensive format registry
    pub fn new() -> Self {
        let mut detector = Self {
            format_registry: HashMap::new(),
            content_patterns: HashMap::new(),
            mime_mappings: HashMap::new(),
            extension_mappings: HashMap::new(),
        };

        detector.initialize_format_registry();
        detector.initialize_content_patterns();
        detector.build_mappings();
        detector
    }

    /// Detect format from file path using multiple detection methods
    pub fn detect_format<P: AsRef<Path>>(&self, path: P) -> Result<DetectionResult, CliError> {
        let path = path.as_ref();
        let mut results = Vec::new();

        // Method 1: File extension detection
        if let Some(extension_result) = self.detect_by_extension(path)? {
            results.push(extension_result);
        }

        // Method 2: Content analysis (if file exists and is readable)
        if path.exists() && path.is_file() {
            if let Ok(content_result) = self.detect_by_content(path) {
                results.push(content_result);
            }

            // Method 3: Magic bytes detection
            if let Ok(magic_result) = self.detect_by_magic_bytes(path) {
                results.push(magic_result);
            }
        }

        // Combine results using weighted confidence scoring
        self.combine_detection_results(results)
    }

    /// Detect format from content buffer
    pub fn detect_format_from_buffer(&self, buffer: &[u8]) -> Result<DetectionResult, CliError> {
        let mut results = Vec::new();

        // Magic bytes detection
        if let Some(magic_result) = self.detect_magic_bytes_from_buffer(buffer) {
            results.push(magic_result);
        }

        // Content pattern analysis
        if let Ok(content) = std::str::from_utf8(buffer) {
            if let Some(pattern_result) = self.detect_by_content_patterns(content) {
                results.push(pattern_result);
            }
        }

        self.combine_detection_results(results)
    }

    /// Validate file format against detected/specified format
    pub fn validate_format<P: AsRef<Path>>(
        &self,
        path: P,
        expected_format: Option<RdfFormat>,
    ) -> Result<bool, CliError> {
        let path = path.as_ref();
        let detected = self.detect_format(path)?;

        if let Some(expected) = expected_format {
            Ok(detected.format == expected && detected.confidence > 0.7)
        } else {
            Ok(detected.confidence > 0.8)
        }
    }

    /// Convert between RDF formats with comprehensive options
    pub fn convert_format<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input_path: P,
        output_path: Q,
        options: ConversionOptions,
    ) -> Result<ConversionStats, CliError> {
        let input_path = input_path.as_ref();
        let output_path = output_path.as_ref();

        info!(
            "Converting {} to {} format",
            input_path.display(),
            format_name(&options.output_format)
        );

        let start_time = std::time::Instant::now();

        // Detect input format if not specified
        let input_format = if let Some(format) = options.input_format {
            format
        } else {
            let detection = self.detect_format(input_path)?;
            if detection.confidence < 0.7 {
                warn!(
                    "Low confidence ({:.2}) in format detection for {}",
                    detection.confidence,
                    input_path.display()
                );
            }
            detection.format
        };

        // Validate input if requested
        if options.validate_input {
            self.validate_input_file(input_path, input_format)?;
        }

        // Perform conversion
        let stats = self.perform_conversion(input_path, output_path, input_format, &options)?;

        let duration = start_time.elapsed();
        info!(
            "Conversion completed in {:.2}s: {} triples processed",
            duration.as_secs_f64(),
            stats.triples_processed
        );

        Ok(stats)
    }

    /// Get comprehensive format information
    pub fn get_format_info(&self, format: &RdfFormat) -> Option<&FormatInfo> {
        self.format_registry.get(format)
    }

    /// List all supported formats with details
    pub fn list_supported_formats(&self) -> Vec<(&RdfFormat, &FormatInfo)> {
        self.format_registry.iter().collect()
    }

    /// Get MIME type for format
    pub fn get_mime_type(&self, format: &RdfFormat) -> Option<&str> {
        self.format_registry
            .get(format)
            .and_then(|info| info.mime_types.first())
            .map(|s| s.as_str())
    }

    // Private implementation methods

    fn initialize_format_registry(&mut self) {
        // Turtle format
        self.format_registry.insert(
            RdfFormat::Turtle,
            FormatInfo {
                format: RdfFormat::Turtle,
                mime_types: vec![
                    "text/turtle".to_string(),
                    "application/x-turtle".to_string(),
                ],
                file_extensions: vec!["ttl".to_string(), "turtle".to_string()],
                magic_bytes: None,
                confidence_markers: vec!["@prefix".to_string(), "@base".to_string()],
                encoding_support: vec!["UTF-8".to_string(), "UTF-16".to_string()],
                compression_support: vec!["gzip".to_string(), "bzip2".to_string()],
            },
        );

        // N-Triples format
        self.format_registry.insert(
            RdfFormat::NTriples,
            FormatInfo {
                format: RdfFormat::NTriples,
                mime_types: vec!["application/n-triples".to_string()],
                file_extensions: vec!["nt".to_string(), "ntriples".to_string()],
                magic_bytes: None,
                confidence_markers: vec![" .".to_string(), ">\n".to_string()],
                encoding_support: vec!["UTF-8".to_string()],
                compression_support: vec!["gzip".to_string(), "bzip2".to_string()],
            },
        );

        // RDF/XML format
        self.format_registry.insert(
            RdfFormat::RdfXml,
            FormatInfo {
                format: RdfFormat::RdfXml,
                mime_types: vec![
                    "application/rdf+xml".to_string(),
                    "application/xml".to_string(),
                ],
                file_extensions: vec!["rdf".to_string(), "xml".to_string(), "owl".to_string()],
                magic_bytes: Some(b"<?xml".to_vec()),
                confidence_markers: vec![
                    "rdf:RDF".to_string(),
                    "xmlns:rdf".to_string(),
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
                ],
                encoding_support: vec![
                    "UTF-8".to_string(),
                    "UTF-16".to_string(),
                    "ISO-8859-1".to_string(),
                ],
                compression_support: vec!["gzip".to_string(), "bzip2".to_string()],
            },
        );

        // JSON-LD format
        self.format_registry.insert(
            RdfFormat::JsonLd,
            FormatInfo {
                format: RdfFormat::JsonLd,
                mime_types: vec![
                    "application/ld+json".to_string(),
                    "application/json".to_string(),
                ],
                file_extensions: vec!["jsonld".to_string(), "json".to_string()],
                magic_bytes: Some(b"{".to_vec()),
                confidence_markers: vec![
                    "@context".to_string(),
                    "@type".to_string(),
                    "@id".to_string(),
                ],
                encoding_support: vec!["UTF-8".to_string()],
                compression_support: vec!["gzip".to_string(), "bzip2".to_string()],
            },
        );

        // TriG format
        self.format_registry.insert(
            RdfFormat::TriG,
            FormatInfo {
                format: RdfFormat::TriG,
                mime_types: vec!["application/trig".to_string()],
                file_extensions: vec!["trig".to_string()],
                magic_bytes: None,
                confidence_markers: vec!["@prefix".to_string(), "{".to_string(), "}".to_string()],
                encoding_support: vec!["UTF-8".to_string(), "UTF-16".to_string()],
                compression_support: vec!["gzip".to_string(), "bzip2".to_string()],
            },
        );

        // N-Quads format
        self.format_registry.insert(
            RdfFormat::NQuads,
            FormatInfo {
                format: RdfFormat::NQuads,
                mime_types: vec!["application/n-quads".to_string()],
                file_extensions: vec!["nq".to_string(), "nquads".to_string()],
                magic_bytes: None,
                confidence_markers: vec![" .".to_string(), "> .".to_string()],
                encoding_support: vec!["UTF-8".to_string()],
                compression_support: vec!["gzip".to_string(), "bzip2".to_string()],
            },
        );
    }

    fn initialize_content_patterns(&mut self) {
        // Turtle patterns
        self.content_patterns.insert(
            RdfFormat::Turtle,
            vec![
                ContentPattern {
                    pattern: r"@prefix\s+\w*:\s*<[^>]+>\s*\.".to_string(),
                    weight: 0.9,
                    required: false,
                },
                ContentPattern {
                    pattern: r"@base\s+<[^>]+>\s*\.".to_string(),
                    weight: 0.8,
                    required: false,
                },
                ContentPattern {
                    pattern: r"<[^>]+>\s+<[^>]+>\s+[^.]+\.".to_string(),
                    weight: 0.6,
                    required: false,
                },
            ],
        );

        // N-Triples patterns
        self.content_patterns.insert(
            RdfFormat::NTriples,
            vec![
                ContentPattern {
                    pattern: r"<[^>]+>\s+<[^>]+>\s+<[^>]+>\s*\.".to_string(),
                    weight: 0.8,
                    required: false,
                },
                ContentPattern {
                    pattern: "<[^>]+>\\s+<[^>]+>\\s+\"[^\"]*\"(\\^\\^<[^>]+>)?\\s*\\.".to_string(),
                    weight: 0.7,
                    required: false,
                },
            ],
        );

        // RDF/XML patterns
        self.content_patterns.insert(
            RdfFormat::RdfXml,
            vec![
                ContentPattern {
                    pattern: r"<rdf:RDF[^>]*>".to_string(),
                    weight: 0.9,
                    required: false,
                },
                ContentPattern {
                    pattern: "xmlns:rdf\\s*=\\s*[\"']http://www\\.w3\\.org/1999/02/22-rdf-syntax-ns#[\"']".to_string(),
                    weight: 0.8,
                    required: false,
                },
            ],
        );

        // JSON-LD patterns
        self.content_patterns.insert(
            RdfFormat::JsonLd,
            vec![
                ContentPattern {
                    pattern: r#""@context"\s*:"#.to_string(),
                    weight: 0.9,
                    required: false,
                },
                ContentPattern {
                    pattern: r#""@type"\s*:"#.to_string(),
                    weight: 0.7,
                    required: false,
                },
                ContentPattern {
                    pattern: r#""@id"\s*:"#.to_string(),
                    weight: 0.6,
                    required: false,
                },
            ],
        );
    }

    fn build_mappings(&mut self) {
        for info in self.format_registry.values() {
            // MIME type mappings
            for mime_type in &info.mime_types {
                self.mime_mappings.insert(mime_type.clone(), info.format);
            }

            // Extension mappings
            for extension in &info.file_extensions {
                self.extension_mappings
                    .insert(extension.clone(), info.format);
            }
        }
    }

    fn detect_by_extension(&self, path: &Path) -> Result<Option<DetectionResult>, CliError> {
        if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
            let extension = extension.to_lowercase();
            if let Some(&format) = self.extension_mappings.get(&extension) {
                return Ok(Some(DetectionResult {
                    format,
                    confidence: 0.6, // Medium confidence for extension-based detection
                    detection_method: DetectionMethod::FileExtension,
                    encoding: None,
                    compression: self.detect_compression_from_extension(&extension),
                    additional_info: HashMap::new(),
                }));
            }
        }
        Ok(None)
    }

    fn detect_by_content(&self, path: &Path) -> Result<DetectionResult, CliError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut buffer = Vec::new();

        // Read first few KB for analysis
        reader.by_ref().take(8192).read_to_end(&mut buffer)?;

        let content = std::str::from_utf8(&buffer)
            .map_err(|_| CliError::InvalidFormat("File contains invalid UTF-8".to_string()))?;

        self.detect_by_content_patterns(content).ok_or_else(|| {
            CliError::UnknownFormat("Could not detect format from content".to_string())
        })
    }

    fn detect_by_content_patterns(&self, content: &str) -> Option<DetectionResult> {
        let mut best_match = None;
        let mut best_confidence = 0.0;

        for (format, patterns) in &self.content_patterns {
            let mut total_weight = 0.0;
            let mut matched_weight = 0.0;

            for pattern in patterns {
                total_weight += pattern.weight;

                if regex::Regex::new(&pattern.pattern)
                    .map(|re| re.is_match(content))
                    .unwrap_or(false)
                {
                    matched_weight += pattern.weight;
                }
            }

            let confidence = if total_weight > 0.0 {
                matched_weight / total_weight
            } else {
                0.0
            };

            if confidence > best_confidence {
                best_confidence = confidence;
                best_match = Some(DetectionResult {
                    format: *format,
                    confidence,
                    detection_method: DetectionMethod::ContentAnalysis,
                    encoding: Some("UTF-8".to_string()),
                    compression: None,
                    additional_info: HashMap::new(),
                });
            }
        }

        best_match
    }

    fn detect_by_magic_bytes(&self, path: &Path) -> Result<DetectionResult, CliError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut buffer = [0; 16];
        let bytes_read = reader.read(&mut buffer)?;

        self.detect_magic_bytes_from_buffer(&buffer[..bytes_read])
            .ok_or_else(|| CliError::UnknownFormat("No magic bytes detected".to_string()))
    }

    fn detect_magic_bytes_from_buffer(&self, buffer: &[u8]) -> Option<DetectionResult> {
        for info in self.format_registry.values() {
            if let Some(magic_bytes) = &info.magic_bytes {
                if buffer.starts_with(magic_bytes) {
                    return Some(DetectionResult {
                        format: info.format,
                        confidence: 0.95, // High confidence for magic bytes
                        detection_method: DetectionMethod::MagicBytes,
                        encoding: None,
                        compression: None,
                        additional_info: HashMap::new(),
                    });
                }
            }
        }
        None
    }

    fn combine_detection_results(
        &self,
        results: Vec<DetectionResult>,
    ) -> Result<DetectionResult, CliError> {
        if results.is_empty() {
            return Err(CliError::UnknownFormat("No format detected".to_string()));
        }

        if results.len() == 1 {
            return Ok(results.into_iter().next().unwrap());
        }

        // Weight results by detection method reliability
        let mut weighted_scores: HashMap<RdfFormat, f32> = HashMap::new();
        let mut best_result = &results[0];

        for result in &results {
            let method_weight = match result.detection_method {
                DetectionMethod::MagicBytes => 0.95,
                DetectionMethod::ContentAnalysis => 0.85,
                DetectionMethod::MimeType => 0.75,
                DetectionMethod::FileExtension => 0.60,
                DetectionMethod::CombinedHeuristics => 0.90,
            };

            let weighted_confidence = result.confidence * method_weight;
            *weighted_scores.entry(result.format).or_insert(0.0) += weighted_confidence;

            if weighted_confidence > best_result.confidence * 0.85 {
                best_result = result;
            }
        }

        // Create combined result
        let mut combined_result = best_result.clone();
        combined_result.detection_method = DetectionMethod::CombinedHeuristics;
        combined_result.confidence = weighted_scores
            .values()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(0.0);

        Ok(combined_result)
    }

    fn detect_compression_from_extension(&self, extension: &str) -> Option<String> {
        match extension {
            "gz" | "gzip" => Some("gzip".to_string()),
            "bz2" | "bzip2" => Some("bzip2".to_string()),
            "xz" => Some("xz".to_string()),
            "lz4" => Some("lz4".to_string()),
            _ => None,
        }
    }

    fn validate_input_file(&self, path: &Path, format: RdfFormat) -> Result<(), CliError> {
        debug!("Validating input file {} as {:?}", path.display(), format);

        // Basic validation by attempting to parse a small portion
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        // Try to parse first few statements
        let handler = FormatHandler::new(format);
        match handler.parse_triples(reader) {
            Ok(_) => Ok(()),
            Err(e) => Err(CliError::InvalidFormat(format!("Validation failed: {}", e))),
        }
    }

    fn perform_conversion(
        &self,
        input_path: &Path,
        output_path: &Path,
        input_format: RdfFormat,
        options: &ConversionOptions,
    ) -> Result<ConversionStats, CliError> {
        let start_time = std::time::Instant::now();
        let mut stats = ConversionStats::default();

        // Read input
        let input_file = File::open(input_path)?;
        let input_reader = BufReader::new(input_file);

        let mut graph = Graph::new();

        // Parse input format
        let handler = FormatHandler::new(input_format);
        match handler.parse_triples(input_reader) {
            Ok(triples) => {
                stats.triples_processed = triples.len();
                stats.triples_valid = triples.len();
                // Add triples to graph
                for triple in triples {
                    graph.insert(triple);
                }
            }
            Err(e) => {
                return Err(CliError::ConversionError(format!(
                    "Failed to parse input: {}",
                    e
                )));
            }
        }

        // Write output
        let output_file = File::create(output_path)?;
        let output_handler = FormatHandler::new(options.output_format);
        let graph_triples: Vec<_> = graph.iter().cloned().collect();

        match output_handler.serialize_triples(output_file, &graph_triples) {
            Ok(_) => {
                stats.conversion_time = start_time.elapsed();
                Ok(stats)
            }
            Err(e) => Err(CliError::ConversionError(format!(
                "Failed to write output: {}",
                e
            ))),
        }
    }
}

impl Default for FormatDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Conversion statistics
#[derive(Debug, Default)]
pub struct ConversionStats {
    pub triples_processed: usize,
    pub triples_valid: usize,
    pub triples_errors: usize,
    pub conversion_time: std::time::Duration,
    pub input_size: u64,
    pub output_size: u64,
}

fn format_name(format: &RdfFormat) -> &'static str {
    match format {
        RdfFormat::Turtle => "Turtle",
        RdfFormat::NTriples => "N-Triples",
        RdfFormat::RdfXml => "RDF/XML",
        RdfFormat::JsonLd => "JSON-LD",
        RdfFormat::TriG => "TriG",
        RdfFormat::NQuads => "N-Quads",
    }
}

/// CLI command for format detection
pub fn detect_format_command(
    path: PathBuf,
    verbose: bool,
    output_format: Option<String>,
) -> Result<(), CliError> {
    let detector = FormatDetector::new();
    let result = detector.detect_format(&path)?;

    if verbose {
        let formatter = OutputFormatter::new(output_format.as_deref().unwrap_or("table"));
        formatter.print_detection_result(&result, &path)?;
    } else {
        println!("{}", format_name(&result.format));
    }

    Ok(())
}

/// CLI command for format conversion
pub fn convert_format_command(
    input_path: PathBuf,
    output_path: PathBuf,
    input_format: Option<String>,
    output_format: String,
    options: ConversionOptions,
) -> Result<(), CliError> {
    let detector = FormatDetector::new();

    let input_format = if let Some(format_str) = input_format {
        parse_format(&format_str)?
    } else {
        detector.detect_format(&input_path)?.format
    };

    let output_format = parse_format(&output_format)?;

    let mut conversion_options = options;
    conversion_options.input_format = Some(input_format);
    conversion_options.output_format = output_format;

    let stats = detector.convert_format(&input_path, &output_path, conversion_options)?;

    println!(
        "Converted {} triples in {:.2}s",
        stats.triples_processed,
        stats.conversion_time.as_secs_f64()
    );

    Ok(())
}

fn parse_format(format_str: &str) -> Result<RdfFormat, CliError> {
    match format_str.to_lowercase().as_str() {
        "turtle" | "ttl" => Ok(RdfFormat::Turtle),
        "ntriples" | "nt" => Ok(RdfFormat::NTriples),
        "rdfxml" | "rdf" | "xml" => Ok(RdfFormat::RdfXml),
        "jsonld" | "json" => Ok(RdfFormat::JsonLd),
        "trig" => Ok(RdfFormat::TriG),
        "nquads" | "nq" => Ok(RdfFormat::NQuads),
        _ => Err(CliError::InvalidFormat(format!(
            "Unknown format: {}",
            format_str
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_format_detection_by_extension() {
        let detector = FormatDetector::new();

        let path = Path::new("test.ttl");
        let result = detector.detect_by_extension(path).unwrap().unwrap();

        assert_eq!(result.format, RdfFormat::Turtle);
        assert_eq!(result.detection_method, DetectionMethod::FileExtension);
    }

    #[test]
    fn test_content_pattern_detection() {
        let detector = FormatDetector::new();

        let turtle_content =
            "@prefix ex: <http://example.org/> .\nex:subject ex:predicate ex:object .";
        let result = detector.detect_by_content_patterns(turtle_content).unwrap();

        assert_eq!(result.format, RdfFormat::Turtle);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_magic_bytes_detection() {
        let detector = FormatDetector::new();

        let xml_content = b"<?xml version=\"1.0\"?><rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">";
        let result = detector
            .detect_magic_bytes_from_buffer(xml_content)
            .unwrap();

        assert_eq!(result.format, RdfFormat::RdfXml);
        assert_eq!(result.confidence, 0.95);
    }

    #[test]
    fn test_format_info_retrieval() {
        let detector = FormatDetector::new();

        let info = detector.get_format_info(&RdfFormat::Turtle).unwrap();
        assert!(!info.mime_types.is_empty());
        assert!(!info.file_extensions.is_empty());
        assert!(info.mime_types.contains(&"text/turtle".to_string()));
    }

    #[test]
    fn test_conversion_options_defaults() {
        let options = ConversionOptions::default();
        assert_eq!(options.output_format, RdfFormat::Turtle);
        assert!(options.validate_input);
        assert!(options.preserve_prefixes);
    }
}
