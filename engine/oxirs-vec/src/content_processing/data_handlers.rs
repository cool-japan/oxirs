//! Data format handlers for content processing
//!
//! This module provides handlers for JSON, CSV, and other structured data formats.

#[cfg(feature = "content-processing")]
use crate::content_processing::{
    ContentExtractionConfig, ContentLocation, DocumentFormat, DocumentStructure, ExtractedContent,
    ExtractedTable, FormatHandler, ProcessingStats,
};
#[cfg(feature = "content-processing")]
use anyhow::Result;
#[cfg(feature = "content-processing")]
use std::collections::HashMap;

/// JSON handler
#[cfg(feature = "content-processing")]
pub struct JsonHandler;

#[cfg(feature = "content-processing")]
impl FormatHandler for JsonHandler {
    fn extract_content(
        &self,
        data: &[u8],
        _config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent> {
        let json_str = String::from_utf8_lossy(data);

        // Parse JSON and extract text content
        let text = match serde_json::from_str::<serde_json::Value>(&json_str) {
            Ok(value) => extract_text_from_json(&value),
            Err(_) => json_str.to_string(),
        };

        Ok(ExtractedContent {
            format: DocumentFormat::Json,
            text,
            metadata: HashMap::new(),
            images: Vec::new(),
            tables: Vec::new(),
            links: Vec::new(),
            structure: DocumentStructure {
                title: None,
                headings: Vec::new(),
                page_count: 1,
                section_count: 1,
                table_of_contents: Vec::new(),
            },
            chunks: Vec::new(),
            language: None,
            processing_stats: ProcessingStats::default(),
            audio_content: Vec::new(),
            video_content: Vec::new(),
            cross_modal_embeddings: Vec::new(),
        })
    }

    fn can_handle(&self, data: &[u8]) -> bool {
        let content = String::from_utf8_lossy(data);
        serde_json::from_str::<serde_json::Value>(&content).is_ok()
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["json"]
    }
}

#[cfg(feature = "content-processing")]
fn extract_text_from_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(arr) => arr
            .iter()
            .map(extract_text_from_json)
            .collect::<Vec<_>>()
            .join(" "),
        serde_json::Value::Object(obj) => obj
            .values()
            .map(extract_text_from_json)
            .collect::<Vec<_>>()
            .join(" "),
        _ => value.to_string(),
    }
}

/// CSV handler
#[cfg(feature = "content-processing")]
pub struct CsvHandler;

#[cfg(feature = "content-processing")]
impl FormatHandler for CsvHandler {
    fn extract_content(
        &self,
        data: &[u8],
        _config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent> {
        let csv_str = String::from_utf8_lossy(data);

        // Parse CSV and extract content
        let (text, tables) = self.parse_csv(&csv_str);

        Ok(ExtractedContent {
            format: DocumentFormat::Csv,
            text,
            metadata: HashMap::new(),
            images: Vec::new(),
            tables,
            links: Vec::new(),
            structure: DocumentStructure {
                title: None,
                headings: Vec::new(),
                page_count: 1,
                section_count: 1,
                table_of_contents: Vec::new(),
            },
            chunks: Vec::new(),
            language: None,
            processing_stats: ProcessingStats::default(),
            audio_content: Vec::new(),
            video_content: Vec::new(),
            cross_modal_embeddings: Vec::new(),
        })
    }

    fn can_handle(&self, data: &[u8]) -> bool {
        let content = String::from_utf8_lossy(data);
        // Basic CSV detection
        content.contains(',') || content.contains(';')
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["csv"]
    }
}

#[cfg(feature = "content-processing")]
impl CsvHandler {
    fn parse_csv(&self, csv_str: &str) -> (String, Vec<ExtractedTable>) {
        let lines: Vec<&str> = csv_str.lines().collect();
        if lines.is_empty() {
            return (String::new(), Vec::new());
        }

        // Simple CSV parsing (in practice, would use csv crate)
        let headers: Vec<String> = lines[0].split(',').map(|s| s.trim().to_string()).collect();
        let mut rows = Vec::new();

        for line in lines.iter().skip(1) {
            let row: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();
            if row.len() == headers.len() {
                rows.push(row);
            }
        }

        // Create text representation
        let mut text_parts = vec![headers.join(" | ")];
        for row in &rows {
            text_parts.push(row.join(" | "));
        }
        let text = text_parts.join("\n");

        // Create table structure
        let table = ExtractedTable {
            headers,
            rows,
            caption: None,
            location: ContentLocation {
                page: Some(1),
                section: None,
                char_offset: None,
                line: None,
                column: None,
            },
        };

        (text, vec![table])
    }
}

/// Fallback handler for unsupported formats
#[cfg(feature = "content-processing")]
pub struct FallbackHandler(pub String);

#[cfg(feature = "content-processing")]
impl FormatHandler for FallbackHandler {
    fn extract_content(
        &self,
        data: &[u8],
        _config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent> {
        // Basic text extraction attempt
        let text = if let Ok(utf8_text) = String::from_utf8(data.to_vec()) {
            utf8_text
        } else {
            format!(
                "Binary content ({} bytes) - {} format not fully supported",
                data.len(),
                self.0
            )
        };

        Ok(ExtractedContent {
            format: DocumentFormat::Unknown,
            text,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("format".to_string(), self.0.clone());
                meta.insert("size".to_string(), data.len().to_string());
                meta
            },
            images: Vec::new(),
            tables: Vec::new(),
            links: Vec::new(),
            structure: DocumentStructure {
                title: None,
                headings: Vec::new(),
                page_count: 1,
                section_count: 1,
                table_of_contents: Vec::new(),
            },
            chunks: Vec::new(),
            language: None,
            processing_stats: ProcessingStats::default(),
            audio_content: Vec::new(),
            video_content: Vec::new(),
            cross_modal_embeddings: Vec::new(),
        })
    }

    fn can_handle(&self, _data: &[u8]) -> bool {
        true // Fallback handles everything
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec![] // No specific extensions
    }
}
