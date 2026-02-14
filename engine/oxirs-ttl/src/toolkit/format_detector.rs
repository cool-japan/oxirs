//! Automatic RDF format detection
//!
//! This module provides utilities for detecting RDF serialization formats from:
//! - File extensions
//! - Content analysis
//! - MIME types
//!
//! Supported formats:
//! - Turtle (.ttl)
//! - N-Triples (.nt)
//! - N-Quads (.nq)
//! - TriG (.trig)

use crate::toolkit::FastScanner;
use std::path::Path;

/// RDF serialization formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RdfFormat {
    /// Turtle (Terse RDF Triple Language)
    Turtle,
    /// N-Triples (line-oriented triple format)
    NTriples,
    /// N-Quads (line-oriented quad format)
    NQuads,
    /// TriG (Turtle with named graphs)
    TriG,
}

impl RdfFormat {
    /// Get the standard file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            RdfFormat::Turtle => "ttl",
            RdfFormat::NTriples => "nt",
            RdfFormat::NQuads => "nq",
            RdfFormat::TriG => "trig",
        }
    }

    /// Get the MIME type for this format
    pub fn mime_type(&self) -> &'static str {
        match self {
            RdfFormat::Turtle => "text/turtle",
            RdfFormat::NTriples => "application/n-triples",
            RdfFormat::NQuads => "application/n-quads",
            RdfFormat::TriG => "application/trig",
        }
    }

    /// Get a human-readable name for this format
    pub fn name(&self) -> &'static str {
        match self {
            RdfFormat::Turtle => "Turtle",
            RdfFormat::NTriples => "N-Triples",
            RdfFormat::NQuads => "N-Quads",
            RdfFormat::TriG => "TriG",
        }
    }
}

/// Format detection result with confidence score
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Detected format
    pub format: RdfFormat,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Detection method used
    pub method: DetectionMethod,
}

/// Method used for format detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionMethod {
    /// Detected from file extension
    FileExtension,
    /// Detected from MIME type
    MimeType,
    /// Detected by analyzing content
    ContentAnalysis,
    /// Multiple methods agreed
    Combined,
}

/// Format detector with configurable strategies
#[derive(Debug, Clone)]
pub struct FormatDetector {
    /// Number of bytes to analyze for content detection
    sample_size: usize,
    /// Minimum confidence threshold (0.0 to 1.0)
    min_confidence: f64,
}

impl Default for FormatDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl FormatDetector {
    /// Create a new format detector with default settings
    pub fn new() -> Self {
        Self {
            sample_size: 4096, // Analyze first 4KB
            min_confidence: 0.6,
        }
    }

    /// Set the sample size for content analysis
    pub fn with_sample_size(mut self, size: usize) -> Self {
        self.sample_size = size;
        self
    }

    /// Set the minimum confidence threshold
    pub fn with_min_confidence(mut self, confidence: f64) -> Self {
        self.min_confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Detect format from file path
    pub fn detect_from_path(&self, path: &Path) -> Option<DetectionResult> {
        path.extension()
            .and_then(|ext| ext.to_str())
            .and_then(|ext| self.detect_from_extension(ext))
    }

    /// Detect format from file extension
    pub fn detect_from_extension(&self, extension: &str) -> Option<DetectionResult> {
        let ext_lower = extension.to_lowercase();
        let format = match ext_lower.as_str() {
            "ttl" | "turtle" => RdfFormat::Turtle,
            "nt" | "ntriples" => RdfFormat::NTriples,
            "nq" | "nquads" => RdfFormat::NQuads,
            "trig" => RdfFormat::TriG,
            _ => return None,
        };

        Some(DetectionResult {
            format,
            confidence: 0.9, // High confidence from extension
            method: DetectionMethod::FileExtension,
        })
    }

    /// Detect format from MIME type
    pub fn detect_from_mime_type(&self, mime_type: &str) -> Option<DetectionResult> {
        let mime_lower = mime_type.to_lowercase();
        let format = if mime_lower.contains("turtle") {
            RdfFormat::Turtle
        } else if mime_lower.contains("n-triples") || mime_lower.contains("ntriples") {
            RdfFormat::NTriples
        } else if mime_lower.contains("n-quads") || mime_lower.contains("nquads") {
            RdfFormat::NQuads
        } else if mime_lower.contains("trig") {
            RdfFormat::TriG
        } else {
            return None;
        };

        Some(DetectionResult {
            format,
            confidence: 0.95, // Very high confidence from MIME type
            method: DetectionMethod::MimeType,
        })
    }

    /// Detect format by analyzing content
    pub fn detect_from_content(&self, content: &[u8]) -> Option<DetectionResult> {
        let sample = &content[..content.len().min(self.sample_size)];
        let scanner = FastScanner::new(sample);

        let mut scores = FormatScores::default();

        // Analyze content for format-specific patterns
        self.analyze_directives(&scanner, &mut scores);
        self.analyze_syntax(&scanner, &mut scores);
        self.analyze_structure(&scanner, &mut scores);

        // Determine format from scores
        scores.determine_format(self.min_confidence)
    }

    /// Detect format using all available information
    pub fn detect(
        &self,
        path: Option<&Path>,
        mime_type: Option<&str>,
        content: Option<&[u8]>,
    ) -> Option<DetectionResult> {
        let mut results = Vec::new();

        // Try file extension
        if let Some(path) = path {
            if let Some(result) = self.detect_from_path(path) {
                results.push(result);
            }
        }

        // Try MIME type
        if let Some(mime) = mime_type {
            if let Some(result) = self.detect_from_mime_type(mime) {
                results.push(result);
            }
        }

        // Try content analysis
        if let Some(content) = content {
            if let Some(result) = self.detect_from_content(content) {
                results.push(result);
            }
        }

        // Combine results
        self.combine_results(results)
    }

    /// Analyze directives (@prefix, @base, PREFIX, BASE)
    fn analyze_directives(&self, scanner: &FastScanner, scores: &mut FormatScores) {
        let mut pos = 0;
        let mut has_prefix_directive = false;

        while pos < scanner.len() {
            pos = scanner.skip_whitespace_and_comments(pos);
            if pos >= scanner.len() {
                break;
            }

            // Check for @prefix or @base (Turtle/TriG)
            // Give Turtle slightly higher base score since it's more common
            if scanner.byte_at(pos) == Some(b'@') {
                scores.turtle += 0.8;
                scores.trig += 0.5; // Lower base score for TriG
                has_prefix_directive = true;
                break; // Found strong indicator
            }

            // Check for PREFIX or BASE (also Turtle/TriG)
            let slice = scanner.slice(pos, pos + 6);
            if slice.starts_with(b"PREFIX") || slice.starts_with(b"BASE") {
                scores.turtle += 0.7;
                scores.trig += 0.4; // Lower base score for TriG
                has_prefix_directive = true;
                break;
            }

            // Move to next line
            if let Some(newline) = scanner.find_line_end(pos) {
                pos = newline + 1;
            } else {
                break;
            }
        }

        // Prefix directives rule OUT N-Triples and N-Quads
        if has_prefix_directive {
            scores.ntriples = 0.0;
            scores.nquads = 0.0;
        }
    }

    /// Analyze syntax features (abbreviated syntax, named graphs)
    fn analyze_syntax(&self, scanner: &FastScanner, scores: &mut FormatScores) {
        let content = scanner.slice(0, scanner.len());
        let mut has_curly_braces = false;

        // Look for Turtle/TriG abbreviated syntax
        for byte in content {
            match byte {
                b';' => {
                    scores.turtle += 0.1;
                    scores.trig += 0.1;
                }
                b',' => {
                    scores.turtle += 0.05;
                    scores.trig += 0.05;
                }
                b'[' | b']' => {
                    scores.turtle += 0.08;
                    scores.trig += 0.08;
                }
                b'{' | b'}' => {
                    // Named graphs indicate TriG - strong indicator!
                    has_curly_braces = true;
                    scores.trig += 0.5;
                }
                _ => {}
            }
        }

        // If curly braces found, it's likely TriG not Turtle
        if has_curly_braces {
            scores.turtle *= 0.5; // Reduce Turtle confidence
        }
    }

    /// Analyze line-based structure (N-Triples/N-Quads)
    fn analyze_structure(&self, scanner: &FastScanner, scores: &mut FormatScores) {
        let mut pos = 0;
        let mut line_count = 0;
        let mut triple_lines = 0;
        let mut quad_lines = 0;
        let mut has_prefixed_names = false;
        let mut has_full_iris = false;

        while pos < scanner.len() && line_count < 10 {
            pos = scanner.skip_whitespace_and_comments(pos);
            if pos >= scanner.len() {
                break;
            }

            let line_start = pos;
            let line_end = scanner.find_line_end(pos).unwrap_or(scanner.len());

            // Count elements on this line
            let line_content = scanner.slice(line_start, line_end);
            let element_count = self.count_line_elements(line_content);

            // Check for prefixed names (e.g., ex:subject)
            // N-Triples/N-Quads only use full IRIs or blank nodes
            if self.has_prefixed_name(line_content) {
                has_prefixed_names = true;
            }

            // Check if line starts with < (full IRI) or _: (blank node)
            let trimmed_start = scanner.skip_whitespace(line_start);
            if trimmed_start < scanner.len() {
                match scanner.byte_at(trimmed_start) {
                    Some(b'<') | Some(b'_') => has_full_iris = true,
                    _ => {}
                }
            }

            match element_count {
                3 => triple_lines += 1,
                4 => quad_lines += 1,
                _ => {}
            }

            line_count += 1;
            pos = line_end + 1;
        }

        // Prefixed names rule OUT N-Triples and N-Quads
        if has_prefixed_names {
            scores.turtle += 0.4;
            scores.trig += 0.4;
            scores.ntriples = 0.0;
            scores.nquads = 0.0;
            return;
        }

        // Score based on line patterns (only if no prefixed names)
        if line_count > 0 && has_full_iris {
            let triple_ratio = triple_lines as f64 / line_count as f64;
            let quad_ratio = quad_lines as f64 / line_count as f64;

            if triple_ratio > 0.7 {
                scores.ntriples += 0.6;
            }

            if quad_ratio > 0.7 {
                scores.nquads += 0.7;
            }
        }
    }

    /// Check if a line contains a prefixed name (e.g., ex:subject)
    fn has_prefixed_name(&self, line: &[u8]) -> bool {
        let mut i = 0;
        while i < line.len() {
            // Skip whitespace
            while i < line.len() && (line[i] == b' ' || line[i] == b'\t') {
                i += 1;
            }

            if i >= line.len() {
                break;
            }

            // Skip angle bracket IRIs
            if line[i] == b'<' {
                while i < line.len() && line[i] != b'>' {
                    i += 1;
                }
                i += 1;
                continue;
            }

            // Skip strings
            if line[i] == b'"' {
                i += 1;
                while i < line.len() && line[i] != b'"' {
                    if line[i] == b'\\' {
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                i += 1;
                continue;
            }

            // Check for prefixed name pattern: [a-zA-Z]+:
            if line[i].is_ascii_alphabetic() {
                while i < line.len() && (line[i].is_ascii_alphanumeric() || line[i] == b'_') {
                    i += 1;
                }

                // Found a colon after alphanumeric chars?
                if i < line.len() && line[i] == b':' {
                    // Make sure it's not a trailing colon at end
                    if i + 1 < line.len() && line[i + 1] != b' ' {
                        return true;
                    }
                }

                continue;
            }

            i += 1;
        }

        false
    }

    /// Count elements on a line (space-separated)
    ///
    /// Counts RDF elements on a line, excluding the trailing period.
    /// N-Triples has 3 elements (subject predicate object)
    /// N-Quads has 4 elements (subject predicate object graph)
    fn count_line_elements(&self, line: &[u8]) -> usize {
        let mut count = 0;
        let mut in_string = false;
        let mut in_angle_bracket = false;
        let mut prev_space = true;
        let mut elements = Vec::new();
        let mut current_element_start = 0;

        for (i, &byte) in line.iter().enumerate() {
            match byte {
                b'"' => in_string = !in_string,
                b'<' if !in_string => in_angle_bracket = true,
                b'>' if !in_string => in_angle_bracket = false,
                b' ' | b'\t' if !in_string && !in_angle_bracket => {
                    if !prev_space {
                        elements.push(&line[current_element_start..i]);
                    }
                    prev_space = true;
                    current_element_start = i + 1;
                }
                _ => {
                    if prev_space && !in_string {
                        prev_space = false;
                    }
                }
            }
        }

        // Add final element if any
        if !prev_space && current_element_start < line.len() {
            elements.push(&line[current_element_start..]);
        }

        // Filter out trailing period and count
        for elem in elements {
            // Skip if it's just a period or period with whitespace
            if elem.is_empty() || elem == b"." {
                continue;
            }
            count += 1;
        }

        count
    }

    /// Combine multiple detection results
    fn combine_results(&self, results: Vec<DetectionResult>) -> Option<DetectionResult> {
        if results.is_empty() {
            return None;
        }

        if results.len() == 1 {
            return Some(results[0].clone());
        }

        // Weight results by confidence and method
        let mut format_scores: std::collections::HashMap<RdfFormat, f64> =
            std::collections::HashMap::new();

        for result in &results {
            let weight = match result.method {
                DetectionMethod::MimeType => 1.5,
                DetectionMethod::FileExtension => 1.2,
                DetectionMethod::ContentAnalysis => 1.0,
                DetectionMethod::Combined => 1.3,
            };

            *format_scores.entry(result.format).or_insert(0.0) += result.confidence * weight;
        }

        // Find format with highest score
        format_scores
            .into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(format, score)| DetectionResult {
                format,
                confidence: (score / results.len() as f64).min(1.0),
                method: DetectionMethod::Combined,
            })
    }
}

/// Scores for each format during content analysis
#[derive(Debug, Default)]
struct FormatScores {
    turtle: f64,
    ntriples: f64,
    nquads: f64,
    trig: f64,
}

impl FormatScores {
    fn determine_format(&self, min_confidence: f64) -> Option<DetectionResult> {
        let formats = [
            (RdfFormat::Turtle, self.turtle),
            (RdfFormat::NTriples, self.ntriples),
            (RdfFormat::NQuads, self.nquads),
            (RdfFormat::TriG, self.trig),
        ];

        formats
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .filter(|(_, score)| *score >= min_confidence)
            .map(|(format, score)| DetectionResult {
                format: *format,
                confidence: *score,
                method: DetectionMethod::ContentAnalysis,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_turtle_from_extension() {
        let detector = FormatDetector::new();

        let result = detector
            .detect_from_extension("ttl")
            .expect("detection should succeed");
        assert_eq!(result.format, RdfFormat::Turtle);
        assert!(result.confidence > 0.8);
    }

    #[test]
    fn test_detect_ntriples_from_extension() {
        let detector = FormatDetector::new();

        let result = detector
            .detect_from_extension("nt")
            .expect("detection should succeed");
        assert_eq!(result.format, RdfFormat::NTriples);
    }

    #[test]
    fn test_detect_from_path() {
        let detector = FormatDetector::new();
        let path = Path::new("test.ttl");

        let result = detector
            .detect_from_path(path)
            .expect("detection should succeed");
        assert_eq!(result.format, RdfFormat::Turtle);
    }

    #[test]
    fn test_detect_turtle_from_content() {
        let detector = FormatDetector::new();
        let content = b"@prefix ex: <http://example.org/> .\nex:subject ex:predicate ex:object .";

        let result = detector
            .detect_from_content(content)
            .expect("detection should succeed");
        assert_eq!(result.format, RdfFormat::Turtle);
    }

    #[test]
    fn test_detect_trig_from_content() {
        let detector = FormatDetector::new();
        let content = b"@prefix ex: <http://example.org/> .\nex:graph { ex:s ex:p ex:o . }";

        let result = detector
            .detect_from_content(content)
            .expect("detection should succeed");
        assert_eq!(result.format, RdfFormat::TriG);
    }

    #[test]
    fn test_detect_ntriples_from_content() {
        let detector = FormatDetector::new();
        let content = b"<http://example.org/s> <http://example.org/p> <http://example.org/o> .\n<http://example.org/s2> <http://example.org/p2> <http://example.org/o2> .";

        let result = detector
            .detect_from_content(content)
            .expect("detection should succeed");
        assert_eq!(result.format, RdfFormat::NTriples);
    }

    #[test]
    fn test_detect_from_mime_type() {
        let detector = FormatDetector::new();

        let result = detector
            .detect_from_mime_type("text/turtle")
            .expect("detection should succeed");
        assert_eq!(result.format, RdfFormat::Turtle);

        let result = detector
            .detect_from_mime_type("application/n-triples")
            .expect("detection should succeed");
        assert_eq!(result.format, RdfFormat::NTriples);
    }

    #[test]
    fn test_combined_detection() {
        let detector = FormatDetector::new();
        let path = Path::new("test.ttl");
        let content = b"@prefix ex: <http://example.org/> .";

        let result = detector
            .detect(Some(path), None, Some(content))
            .expect("detection should succeed");
        assert_eq!(result.format, RdfFormat::Turtle);
        assert!(result.confidence > 0.8);
    }

    #[test]
    fn test_format_properties() {
        assert_eq!(RdfFormat::Turtle.extension(), "ttl");
        assert_eq!(RdfFormat::Turtle.mime_type(), "text/turtle");
        assert_eq!(RdfFormat::Turtle.name(), "Turtle");
    }
}
