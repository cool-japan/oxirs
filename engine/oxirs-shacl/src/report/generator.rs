//\! Report generation utilities and high-level interface

use super::{ReportConfig, ReportFormat, ValidationReport};
use crate::{Result, ShaclError};

/// High-level report generator
#[derive(Debug, Clone)]
pub struct ReportGenerator {
    config: ReportConfig,
}

impl ReportGenerator {
    /// Create a new report generator with default configuration
    pub fn new() -> Self {
        Self {
            config: ReportConfig::default(),
        }
    }

    /// Create a report generator with custom configuration
    pub fn with_config(config: ReportConfig) -> Self {
        Self { config }
    }

    /// Generate a report in the configured format
    pub fn generate(&self, report: &ValidationReport) -> Result<String> {
        match self.config.format {
            ReportFormat::Json => report.to_json_with_config(&self.config),
            ReportFormat::Html => report.to_html_with_config(&self.config),
            ReportFormat::Turtle => report.to_turtle_with_config(&self.config),
            ReportFormat::Csv => report.to_csv_with_config(&self.config),
            ReportFormat::Text => Ok(report.to_text()?),
            ReportFormat::Yaml => report.to_yaml_with_config(&self.config),
            ReportFormat::JsonLd => self.generate_jsonld(report),
            ReportFormat::RdfXml => self.generate_rdfxml(report),
            ReportFormat::NTriples => self.generate_ntriples(report),
        }
    }

    /// Generate report and write to file
    pub fn generate_to_file(&self, report: &ValidationReport, path: &str) -> Result<()> {
        let content = self.generate(report)?;
        std::fs::write(path, content)
            .map_err(|e| ShaclError::ReportError(format!("Failed to write file: {e}")))?;
        Ok(())
    }

    /// Set output format
    pub fn with_format(mut self, format: ReportFormat) -> Self {
        self.config.format = format;
        self
    }

    /// Set pretty printing
    pub fn with_pretty_print(mut self, pretty: bool) -> Self {
        self.config.pretty_print = pretty;
        self
    }

    /// Set maximum violations
    pub fn with_max_violations(mut self, max: usize) -> Self {
        self.config.max_violations = Some(max);
        self
    }

    /// Enable or disable details
    pub fn with_details(mut self, include: bool) -> Self {
        self.config.include_details = include;
        self
    }

    /// Generate multiple formats at once
    pub fn generate_multiple(
        &self,
        report: &ValidationReport,
        formats: &[ReportFormat],
    ) -> Result<Vec<(ReportFormat, String)>> {
        let mut results = Vec::new();

        for format in formats {
            let mut generator = self.clone();
            generator.config.format = *format;
            let content = generator.generate(report)?;
            results.push((*format, content));
        }

        Ok(results)
    }

    fn generate_jsonld(&self, report: &ValidationReport) -> Result<String> {
        // For now, delegate to JSON and add JSON-LD context
        let json_content = report.to_json_with_config(&self.config)?;

        // Parse and add @context
        let mut json_value: serde_json::Value = serde_json::from_str(&json_content)
            .map_err(|e| ShaclError::ReportError(format!("JSON parsing failed: {e}")))?;

        if let serde_json::Value::Object(ref mut obj) = json_value {
            obj.insert(
                "@context".to_string(),
                serde_json::json!({
                    "sh": "http://www.w3.org/ns/shacl#",
                    "xsd": "http://www.w3.org/2001/XMLSchema#",
                    "conforms": "sh:conforms",
                    "result": "sh:result"
                }),
            );
        }

        if self.config.pretty_print {
            serde_json::to_string_pretty(&json_value)
        } else {
            serde_json::to_string(&json_value)
        }
        .map_err(|e| ShaclError::ReportError(format!("JSON-LD serialization failed: {e}")))
    }

    fn generate_rdfxml(&self, report: &ValidationReport) -> Result<String> {
        // Convert to RDF/XML format
        let turtle = report.to_turtle_with_config(&self.config)?;

        // Simple conversion (in practice, would use a proper RDF library)
        let rdfxml = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:sh="http://www.w3.org/ns/shacl#">
  <sh:ValidationReport>
    <sh:conforms rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">{}</sh:conforms>
    <!-- Violations would be here in full implementation -->
  </sh:ValidationReport>
</rdf:RDF>"#,
            report.conforms
        );

        Ok(rdfxml)
    }

    fn generate_ntriples(&self, report: &ValidationReport) -> Result<String> {
        // Convert to N-Triples format
        let mut ntriples = Vec::new();

        // Basic conformance triple
        ntriples.push(format!(
            "<urn:report> <http://www.w3.org/ns/shacl#conforms> \"{}\"^^<http://www.w3.org/2001/XMLSchema#boolean> .",
            report.conforms
        ));

        // Add violation triples (simplified)
        for (i, violation) in report.violations.iter().enumerate() {
            let violation_uri = format!("<urn:violation{i}>");
            ntriples.push(format!(
                "{violation_uri} <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/ns/shacl#ValidationResult> ."
            ));
            ntriples.push(format!(
                "{} <http://www.w3.org/ns/shacl#focusNode> <{}> .",
                violation_uri, violation.focus_node
            ));
        }

        Ok(ntriples.join("\n"))
    }
}

impl Default for ReportGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenient function to generate a report in a specific format
pub fn generate_report(
    validation_report: &ValidationReport,
    format: &ReportFormat,
) -> Result<String> {
    let config = ReportConfig::default().with_format(*format);
    let generator = ReportGenerator::with_config(config);
    generator.generate(validation_report)
}

/// Generate report with custom configuration
pub fn generate_report_with_config(
    validation_report: &ValidationReport,
    config: &ReportConfig,
) -> Result<String> {
    let generator = ReportGenerator::with_config(config.clone());
    generator.generate(validation_report)
}

/// Batch generate reports in multiple formats
pub fn generate_reports_batch(
    validation_report: &ValidationReport,
    formats: &[ReportFormat],
) -> Result<Vec<(ReportFormat, String)>> {
    let generator = ReportGenerator::new();
    generator.generate_multiple(validation_report, formats)
}

/// Generate and save reports to files
pub fn generate_reports_to_files(
    validation_report: &ValidationReport,
    base_path: &str,
    formats: &[ReportFormat],
) -> Result<Vec<String>> {
    let mut file_paths = Vec::new();

    for format in formats {
        let file_path = format!("{}.{}", base_path, format.file_extension());
        let config = ReportConfig::default().with_format(*format);
        let generator = ReportGenerator::with_config(config);
        generator.generate_to_file(validation_report, &file_path)?;
        file_paths.push(file_path);
    }

    Ok(file_paths)
}

/// Create a summary report (minimal violations, focused on statistics)
pub fn generate_summary_report(validation_report: &ValidationReport) -> Result<String> {
    let config = ReportConfig {
        include_details: false,
        max_violations: Some(10),
        include_summary: true,
        include_metadata: true,
        ..ReportConfig::default()
    };

    generate_report_with_config(validation_report, &config)
}

/// Create a detailed report (all violations, full details)
pub fn generate_detailed_report(validation_report: &ValidationReport) -> Result<String> {
    let config = ReportConfig::detailed();
    generate_report_with_config(validation_report, &config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_report_generation() {
        let report = ValidationReport::new();
        let generator = ReportGenerator::new();

        let json_result = generator.generate(&report);
        assert!(json_result.is_ok());

        let html_generator = generator.with_format(ReportFormat::Html);
        let html_result = html_generator.generate(&report);
        assert!(html_result.is_ok());
    }

    #[test]
    fn test_multiple_formats() {
        let report = ValidationReport::new();
        let generator = ReportGenerator::new();

        let formats = vec![ReportFormat::Json, ReportFormat::Html, ReportFormat::Text];
        let results = generator.generate_multiple(&report, &formats);

        assert!(results.is_ok());
        let results = results.unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_convenience_functions() {
        let report = ValidationReport::new();

        let json_result = generate_report(&report, &ReportFormat::Json);
        assert!(json_result.is_ok());

        let summary_result = generate_summary_report(&report);
        assert!(summary_result.is_ok());

        let detailed_result = generate_detailed_report(&report);
        assert!(detailed_result.is_ok());
    }
}
