//! RDF format detection and conversion utilities
//!
//! Provides comprehensive format detection, validation, and conversion
//! capabilities for all supported RDF serialization formats.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use super::ToolResult;

/// Supported RDF formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RdfFormat {
    /// Turtle (Terse RDF Triple Language)
    Turtle,
    /// N-Triples
    NTriples,
    /// RDF/XML
    RdfXml,
    /// JSON-LD
    JsonLd,
    /// TriG (Turtle with named graphs)
    TriG,
    /// N-Quads
    NQuads,
    /// RDF/JSON (W3C format)
    RdfJson,
    /// RDFa (in HTML/XML)
    RdFa,
    /// Notation3 (N3)
    Notation3,
}

impl RdfFormat {
    /// Get file extensions for this format
    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            RdfFormat::Turtle => &["ttl", "turtle"],
            RdfFormat::NTriples => &["nt", "ntriples"],
            RdfFormat::RdfXml => &["rdf", "xml", "owl"],
            RdfFormat::JsonLd => &["jsonld", "json-ld"],
            RdfFormat::TriG => &["trig"],
            RdfFormat::NQuads => &["nq", "nquads"],
            RdfFormat::RdfJson => &["rj", "rdf-json"],
            RdfFormat::RdFa => &["html", "xhtml"],
            RdfFormat::Notation3 => &["n3"],
        }
    }

    /// Get MIME types for this format
    pub fn mime_types(&self) -> &'static [&'static str] {
        match self {
            RdfFormat::Turtle => &["text/turtle", "application/x-turtle"],
            RdfFormat::NTriples => &["application/n-triples", "text/plain"],
            RdfFormat::RdfXml => &["application/rdf+xml", "application/xml", "text/xml"],
            RdfFormat::JsonLd => &["application/ld+json", "application/json"],
            RdfFormat::TriG => &["application/trig", "application/x-trig"],
            RdfFormat::NQuads => &["application/n-quads", "text/x-nquads"],
            RdfFormat::RdfJson => &["application/rdf+json"],
            RdfFormat::RdFa => &["text/html", "application/xhtml+xml"],
            RdfFormat::Notation3 => &["text/n3", "text/rdf+n3"],
        }
    }

    /// Get the canonical name for this format
    pub fn name(&self) -> &'static str {
        match self {
            RdfFormat::Turtle => "Turtle",
            RdfFormat::NTriples => "N-Triples",
            RdfFormat::RdfXml => "RDF/XML",
            RdfFormat::JsonLd => "JSON-LD",
            RdfFormat::TriG => "TriG",
            RdfFormat::NQuads => "N-Quads",
            RdfFormat::RdfJson => "RDF/JSON",
            RdfFormat::RdFa => "RDFa",
            RdfFormat::Notation3 => "Notation3",
        }
    }

    /// Check if this format supports named graphs
    pub fn supports_graphs(&self) -> bool {
        matches!(
            self,
            RdfFormat::TriG | RdfFormat::NQuads | RdfFormat::JsonLd
        )
    }

    /// Check if this is a line-based format
    pub fn is_line_based(&self) -> bool {
        matches!(self, RdfFormat::NTriples | RdfFormat::NQuads)
    }

    /// Get default file extension
    pub fn default_extension(&self) -> &'static str {
        self.extensions()[0]
    }
}

impl std::fmt::Display for RdfFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Format detection result with confidence
#[derive(Debug, Clone)]
pub struct FormatDetection {
    pub format: RdfFormat,
    pub confidence: f32, // 0.0 to 1.0
    pub reasoning: String,
}

/// Format detector with multiple detection strategies
pub struct FormatDetector;

impl FormatDetector {
    /// Detect format from file path, content, and MIME type
    pub fn detect(
        path: Option<&Path>,
        content: Option<&str>,
        mime_type: Option<&str>,
    ) -> Vec<FormatDetection> {
        let mut detections = Vec::new();

        // Extension-based detection
        if let Some(path) = path {
            if let Some(detection) = Self::detect_by_extension(path) {
                detections.push(detection);
            }
        }

        // MIME type detection
        if let Some(mime) = mime_type {
            if let Some(detection) = Self::detect_by_mime_type(mime) {
                detections.push(detection);
            }
        }

        // Content-based detection
        if let Some(content) = content {
            detections.extend(Self::detect_by_content(content));
        }

        // Sort by confidence (highest first)
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

        // Remove duplicates, keeping highest confidence
        let mut seen = std::collections::HashSet::new();
        detections.retain(|d| seen.insert(d.format));

        detections
    }

    /// Detect format from file extension
    fn detect_by_extension(path: &Path) -> Option<FormatDetection> {
        let ext = path.extension()?.to_str()?.to_lowercase();

        for format in &[
            RdfFormat::Turtle,
            RdfFormat::NTriples,
            RdfFormat::RdfXml,
            RdfFormat::JsonLd,
            RdfFormat::TriG,
            RdfFormat::NQuads,
            RdfFormat::RdfJson,
            RdfFormat::RdFa,
            RdfFormat::Notation3,
        ] {
            if format.extensions().contains(&ext.as_str()) {
                return Some(FormatDetection {
                    format: *format,
                    confidence: 0.8,
                    reasoning: format!("File extension '.{}' matches {}", ext, format.name()),
                });
            }
        }

        None
    }

    /// Detect format from MIME type
    fn detect_by_mime_type(mime: &str) -> Option<FormatDetection> {
        let mime_lower = mime.to_lowercase();

        for format in &[
            RdfFormat::Turtle,
            RdfFormat::NTriples,
            RdfFormat::RdfXml,
            RdfFormat::JsonLd,
            RdfFormat::TriG,
            RdfFormat::NQuads,
            RdfFormat::RdfJson,
            RdfFormat::RdFa,
            RdfFormat::Notation3,
        ] {
            if format.mime_types().iter().any(|&m| m == mime_lower) {
                return Some(FormatDetection {
                    format: *format,
                    confidence: 0.9,
                    reasoning: format!("MIME type '{}' indicates {}", mime, format.name()),
                });
            }
        }

        None
    }

    /// Detect format from content patterns
    fn detect_by_content(content: &str) -> Vec<FormatDetection> {
        let mut detections = Vec::new();
        let trimmed = content.trim();
        let first_lines: Vec<&str> = trimmed.lines().take(20).collect();

        // Check for Turtle/N3 prefixes
        if first_lines
            .iter()
            .any(|line| line.starts_with("@prefix") || line.starts_with("@base"))
        {
            detections.push(FormatDetection {
                format: RdfFormat::Turtle,
                confidence: 0.95,
                reasoning: "Contains @prefix or @base directives".to_string(),
            });
        }

        // Check for N-Triples patterns
        if first_lines.iter().all(|line| {
            line.is_empty()
                || line.starts_with('#')
                || (line.contains(" .") && (line.starts_with('<') || line.starts_with('_')))
        }) {
            let confidence = if trimmed.contains(" .") { 0.85 } else { 0.6 };
            detections.push(FormatDetection {
                format: RdfFormat::NTriples,
                confidence,
                reasoning: "Line-based format with N-Triples patterns".to_string(),
            });
        }

        // Check for XML/RDF
        if trimmed.starts_with("<?xml") || trimmed.starts_with("<rdf:RDF") {
            detections.push(FormatDetection {
                format: RdfFormat::RdfXml,
                confidence: 0.95,
                reasoning: "XML declaration or rdf:RDF root element".to_string(),
            });
        }

        // Check for JSON-LD
        if trimmed.starts_with('{')
            && (trimmed.contains("\"@context\"") || trimmed.contains("'@context'"))
        {
            detections.push(FormatDetection {
                format: RdfFormat::JsonLd,
                confidence: 0.95,
                reasoning: "JSON object with @context".to_string(),
            });
        }

        // Check for TriG (graphs)
        if first_lines
            .iter()
            .any(|line| line.trim().starts_with("GRAPH") || line.contains(" {"))
        {
            detections.push(FormatDetection {
                format: RdfFormat::TriG,
                confidence: 0.8,
                reasoning: "Contains GRAPH keyword or graph brackets".to_string(),
            });
        }

        // Check for N-Quads (fourth component)
        if first_lines.iter().any(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            parts.len() >= 4 && line.ends_with(" .")
        }) {
            detections.push(FormatDetection {
                format: RdfFormat::NQuads,
                confidence: 0.7,
                reasoning: "Line format with four or more components".to_string(),
            });
        }

        detections
    }

    /// Detect format from file with content sampling
    pub async fn detect_file(path: &Path) -> ToolResult<Vec<FormatDetection>> {
        // Read first 4KB for content detection
        let mut file = File::open(path)?;
        let mut buffer = vec![0; 4096];
        let bytes_read = file.read(&mut buffer)?;
        buffer.truncate(bytes_read);

        let content = String::from_utf8_lossy(&buffer);

        Ok(Self::detect(Some(path), Some(&content), None))
    }
}

/// Format converter interface
pub trait FormatConverter {
    /// Convert from one format to another
    fn convert(&self, input: &str, from: RdfFormat, to: RdfFormat) -> ToolResult<String>;

    /// Check if conversion is supported
    fn supports_conversion(&self, from: RdfFormat, to: RdfFormat) -> bool;
}

/// Basic format converter implementation
pub struct BasicFormatConverter;

impl FormatConverter for BasicFormatConverter {
    fn convert(&self, _input: &str, from: RdfFormat, to: RdfFormat) -> ToolResult<String> {
        // For now, return a placeholder - actual implementation would use RDF parsing/serialization
        Err(format!(
            "Conversion from {} to {} not yet implemented",
            from.name(),
            to.name()
        )
        .into())
    }

    fn supports_conversion(&self, _from: RdfFormat, _to: RdfFormat) -> bool {
        // Will support all conversions once implemented
        false
    }
}

/// Format validation
pub struct FormatValidator;

impl FormatValidator {
    /// Validate content against a specific format
    pub fn validate(content: &str, format: RdfFormat) -> ToolResult<ValidationResult> {
        let mut result = ValidationResult {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            stats: ValidationStats::default(),
        };

        match format {
            RdfFormat::NTriples => Self::validate_ntriples(content, &mut result),
            RdfFormat::Turtle => Self::validate_turtle(content, &mut result),
            RdfFormat::RdfXml => Self::validate_rdf_xml(content, &mut result),
            RdfFormat::JsonLd => Self::validate_json_ld(content, &mut result),
            _ => {
                result.warnings.push(format!(
                    "Validation for {} format not yet implemented",
                    format.name()
                ));
            }
        }

        Ok(result)
    }

    fn validate_ntriples(content: &str, result: &mut ValidationResult) {
        for (line_num, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            result.stats.triple_count += 1;

            // Basic N-Triples validation
            if !line.ends_with(" .") {
                result.errors.push(ValidationError {
                    line: Some(line_num + 1),
                    column: None,
                    message: "N-Triples line must end with ' .'".to_string(),
                });
                result.valid = false;
            }

            // Check for basic structure
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 3 {
                result.errors.push(ValidationError {
                    line: Some(line_num + 1),
                    column: None,
                    message: "N-Triples line must have at least 3 components".to_string(),
                });
                result.valid = false;
            }
        }
    }

    fn validate_turtle(_content: &str, result: &mut ValidationResult) {
        // Basic Turtle validation would go here
        result
            .warnings
            .push("Turtle validation not fully implemented".to_string());
    }

    fn validate_rdf_xml(_content: &str, result: &mut ValidationResult) {
        // RDF/XML validation would go here
        result
            .warnings
            .push("RDF/XML validation not fully implemented".to_string());
    }

    fn validate_json_ld(content: &str, result: &mut ValidationResult) {
        // Try to parse as JSON first
        match serde_json::from_str::<serde_json::Value>(content) {
            Ok(value) => {
                // Check for JSON-LD specific requirements
                Self::validate_json_ld_semantics(&value, result);
            }
            Err(e) => {
                result.errors.push(ValidationError {
                    line: None,
                    column: None,
                    message: format!("Invalid JSON: {}", e),
                });
                result.valid = false;
            }
        }
    }

    fn validate_json_ld_semantics(value: &serde_json::Value, result: &mut ValidationResult) {
        match value {
            serde_json::Value::Object(obj) => {
                // Check if it's a valid JSON-LD document
                let has_context = obj.contains_key("@context");
                let has_graph = obj.contains_key("@graph");
                let has_id = obj.contains_key("@id");
                let has_type = obj.contains_key("@type");
                let has_value = obj.contains_key("@value");

                // A JSON-LD document should have at least one of these
                if !has_context && !has_graph && !has_id && !has_type && !has_value {
                    // Check if any keys start with @ (JSON-LD keywords)
                    let has_jsonld_keywords = obj.keys().any(|k| k.starts_with('@'));
                    if !has_jsonld_keywords {
                        result.warnings.push(
                            "Document appears to be plain JSON rather than JSON-LD (no @context, @graph, @id, @type, or other keywords found)".to_string()
                        );
                    }
                }

                // Validate @context if present
                if let Some(context) = obj.get("@context") {
                    Self::validate_context(context, result);
                }

                // Validate @type if present
                if let Some(type_value) = obj.get("@type") {
                    Self::validate_type_value(type_value, result);
                }

                // Validate @id if present
                if let Some(id_value) = obj.get("@id") {
                    Self::validate_id_value(id_value, result);
                }

                // Validate @value if present
                if let Some(value_obj) = obj.get("@value") {
                    Self::validate_value_object(obj, result);
                }

                // Check for invalid keyword combinations
                if has_value && (has_id || has_type) {
                    // @value objects cannot have @id or @type (except in specific cases)
                    if !obj.contains_key("@type") || obj.len() > 3 {
                        result.warnings.push(
                            "@value objects should only contain @value, @type (for datatype), and @language".to_string()
                        );
                    }
                }

                // Recursively validate nested objects
                for (key, nested_value) in obj {
                    if !key.starts_with('@') {
                        Self::validate_json_ld_semantics(nested_value, result);
                    }
                }
            }
            serde_json::Value::Array(arr) => {
                // Validate each item in the array
                for item in arr {
                    Self::validate_json_ld_semantics(item, result);
                }
            }
            _ => {
                // Primitive values are valid at the top level in some contexts
            }
        }
    }

    fn validate_context(context: &serde_json::Value, result: &mut ValidationResult) {
        match context {
            serde_json::Value::String(s) => {
                // Should be a valid IRI
                if !s.starts_with("http://") && !s.starts_with("https://") && !s.starts_with("file://") {
                    if !s.contains(':') {
                        result.warnings.push(
                            format!("@context IRI '{}' may not be valid (no protocol specified)", s)
                        );
                    }
                }
            }
            serde_json::Value::Object(obj) => {
                // Validate context object
                for (key, value) in obj {
                    if key.starts_with('@') && key != "@base" && key != "@vocab" && key != "@language" && key != "@version" {
                        result.warnings.push(
                            format!("Unknown keyword '{}' in @context", key)
                        );
                    }
                    
                    // Validate term definitions
                    if !key.starts_with('@') {
                        match value {
                            serde_json::Value::String(_) => {
                                // Simple term mapping - valid
                            }
                            serde_json::Value::Object(term_def) => {
                                // Expanded term definition
                                Self::validate_term_definition(key, term_def, result);
                            }
                            _ => {
                                result.errors.push(ValidationError {
                                    line: None,
                                    column: None,
                                    message: format!("Invalid term definition for '{}': must be string or object", key),
                                });
                                result.valid = false;
                            }
                        }
                    }
                }
            }
            serde_json::Value::Array(arr) => {
                // Array of contexts
                for item in arr {
                    Self::validate_context(item, result);
                }
            }
            _ => {
                result.errors.push(ValidationError {
                    line: None,
                    column: None,
                    message: "@context must be a string, object, or array".to_string(),
                });
                result.valid = false;
            }
        }
    }

    fn validate_term_definition(term: &str, def: &serde_json::Map<String, serde_json::Value>, result: &mut ValidationResult) {
        let valid_keys = ["@id", "@type", "@container", "@context", "@language", "@reverse", "@nest"];
        
        for key in def.keys() {
            if !valid_keys.contains(&key.as_str()) {
                result.warnings.push(
                    format!("Unknown key '{}' in term definition for '{}'", key, term)
                );
            }
        }

        // Check for required @id in reverse properties
        if def.contains_key("@reverse") && !def.contains_key("@id") {
            result.warnings.push(
                format!("Term '{}' has @reverse but no @id", term)
            );
        }

        // Validate @container values
        if let Some(container) = def.get("@container") {
            match container {
                serde_json::Value::String(s) => {
                    let valid_containers = ["@list", "@set", "@index", "@language", "@id", "@type", "@graph"];
                    if !valid_containers.contains(&s.as_str()) {
                        result.warnings.push(
                            format!("Unknown @container value '{}' for term '{}'", s, term)
                        );
                    }
                }
                serde_json::Value::Array(arr) => {
                    for item in arr {
                        if let serde_json::Value::String(s) = item {
                            let valid_containers = ["@list", "@set", "@index", "@language", "@id", "@type", "@graph"];
                            if !valid_containers.contains(&s.as_str()) {
                                result.warnings.push(
                                    format!("Unknown @container value '{}' for term '{}'", s, term)
                                );
                            }
                        }
                    }
                }
                _ => {
                    result.errors.push(ValidationError {
                        line: None,
                        column: None,
                        message: format!("@container for term '{}' must be string or array", term),
                    });
                    result.valid = false;
                }
            }
        }
    }

    fn validate_type_value(type_value: &serde_json::Value, result: &mut ValidationResult) {
        match type_value {
            serde_json::Value::String(_) => {
                // Single type - valid
            }
            serde_json::Value::Array(arr) => {
                if arr.is_empty() {
                    result.warnings.push("@type array should not be empty".to_string());
                }
                for item in arr {
                    if !item.is_string() {
                        result.errors.push(ValidationError {
                            line: None,
                            column: None,
                            message: "All @type values must be strings".to_string(),
                        });
                        result.valid = false;
                    }
                }
            }
            _ => {
                result.errors.push(ValidationError {
                    line: None,
                    column: None,
                    message: "@type must be a string or array of strings".to_string(),
                });
                result.valid = false;
            }
        }
    }

    fn validate_id_value(id_value: &serde_json::Value, result: &mut ValidationResult) {
        match id_value {
            serde_json::Value::String(s) => {
                if s.is_empty() {
                    result.warnings.push("@id should not be empty".to_string());
                }
                // Could add IRI validation here
            }
            _ => {
                result.errors.push(ValidationError {
                    line: None,
                    column: None,
                    message: "@id must be a string".to_string(),
                });
                result.valid = false;
            }
        }
    }

    fn validate_value_object(obj: &serde_json::Map<String, serde_json::Value>, result: &mut ValidationResult) {
        let has_value = obj.contains_key("@value");
        let has_language = obj.contains_key("@language");
        let has_type = obj.contains_key("@type");
        let has_index = obj.contains_key("@index");

        if !has_value {
            return; // Not a value object
        }

        // @language and @type are mutually exclusive
        if has_language && has_type {
            result.errors.push(ValidationError {
                line: None,
                column: None,
                message: "@language and @type cannot both be present in a value object".to_string(),
            });
            result.valid = false;
        }

        // Check for invalid keys in value object
        for key in obj.keys() {
            if !["@value", "@type", "@language", "@index"].contains(&key.as_str()) {
                result.warnings.push(
                    format!("Unexpected key '{}' in @value object", key)
                );
            }
        }

        // Validate @language value
        if let Some(lang) = obj.get("@language") {
            if !lang.is_string() {
                result.errors.push(ValidationError {
                    line: None,
                    column: None,
                    message: "@language must be a string".to_string(),
                });
                result.valid = false;
            } else if let Some(lang_str) = lang.as_str() {
                if lang_str.is_empty() {
                    result.warnings.push("@language should not be empty".to_string());
                }
                // Could add BCP 47 language tag validation here
            }
        }
    }
}

/// Validation result
#[derive(Debug)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<String>,
    pub stats: ValidationStats,
}

#[derive(Debug)]
pub struct ValidationError {
    pub line: Option<usize>,
    pub column: Option<usize>,
    pub message: String,
}

#[derive(Debug, Default)]
pub struct ValidationStats {
    pub triple_count: usize,
    pub prefix_count: usize,
    pub blank_node_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_extensions() {
        assert!(RdfFormat::Turtle.extensions().contains(&"ttl"));
        assert!(RdfFormat::NTriples.extensions().contains(&"nt"));
        assert!(RdfFormat::RdfXml.extensions().contains(&"rdf"));
    }

    #[test]
    fn test_format_detection_by_extension() {
        let detections = FormatDetector::detect(Some(Path::new("test.ttl")), None, None);
        assert!(!detections.is_empty());
        assert_eq!(detections[0].format, RdfFormat::Turtle);
    }

    #[test]
    fn test_format_detection_by_content() {
        let turtle_content = "@prefix dc: <http://purl.org/dc/elements/1.1/> .\n";
        let detections = FormatDetector::detect(None, Some(turtle_content), None);
        assert!(!detections.is_empty());
        assert_eq!(detections[0].format, RdfFormat::Turtle);

        let xml_content = "<?xml version=\"1.0\"?>\n<rdf:RDF>";
        let detections = FormatDetector::detect(None, Some(xml_content), None);
        assert!(!detections.is_empty());
        assert_eq!(detections[0].format, RdfFormat::RdfXml);
    }

    #[test]
    fn test_ntriples_validation() {
        let valid_content =
            "<http://example.org/s> <http://example.org/p> <http://example.org/o> .\n";
        let result = FormatValidator::validate(valid_content, RdfFormat::NTriples).unwrap();
        assert!(result.valid);
        assert_eq!(result.stats.triple_count, 1);

        let invalid_content =
            "<http://example.org/s> <http://example.org/p> <http://example.org/o>\n";
        let result = FormatValidator::validate(invalid_content, RdfFormat::NTriples).unwrap();
        assert!(!result.valid);
        assert!(!result.errors.is_empty());
    }
}
