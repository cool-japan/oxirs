//! Document content validators

use async_trait::async_trait;
use std::collections::HashMap;

use super::traits::*;
use super::types::*;
use crate::Result;

/// PDF document validator
#[derive(Debug)]
pub struct PDFValidator;

impl Default for PDFValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl PDFValidator {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DocumentValidator for PDFValidator {
    async fn validate(&self, content: &MultiModalContent) -> Result<Option<ValidationResult>> {
        if !self.supports_content(content) {
            return Ok(None);
        }

        let is_pdf = content.data.len() >= 4 && &content.data[0..4] == b"%PDF";

        let mut details = HashMap::new();
        details.insert("file_size".to_string(), content.data.len().to_string());
        details.insert("is_pdf".to_string(), is_pdf.to_string());

        Ok(Some(ValidationResult {
            is_valid: is_pdf,
            confidence: if is_pdf { 0.95 } else { 0.2 },
            error_message: if is_pdf {
                None
            } else {
                Some("Invalid PDF format".to_string())
            },
            details,
        }))
    }

    fn name(&self) -> &str {
        "pdf_validator"
    }

    fn description(&self) -> &str {
        "Validates PDF document format"
    }
}

/// Office document validator
#[derive(Debug)]
pub struct OfficeDocumentValidator;

impl Default for OfficeDocumentValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl OfficeDocumentValidator {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DocumentValidator for OfficeDocumentValidator {
    async fn validate(&self, content: &MultiModalContent) -> Result<Option<ValidationResult>> {
        if !self.supports_content(content) {
            return Ok(None);
        }

        let is_zip = content.data.len() >= 4 && content.data[0..4] == [0x50, 0x4B, 0x03, 0x04];

        let mut details = HashMap::new();
        details.insert("file_size".to_string(), content.data.len().to_string());
        details.insert("is_office_format".to_string(), is_zip.to_string());

        Ok(Some(ValidationResult {
            is_valid: is_zip,
            confidence: if is_zip { 0.9 } else { 0.2 },
            error_message: if is_zip {
                None
            } else {
                Some("Invalid Office format".to_string())
            },
            details,
        }))
    }

    fn name(&self) -> &str {
        "office_validator"
    }

    fn description(&self) -> &str {
        "Validates Microsoft Office document formats"
    }
}

/// Markdown document validator
#[derive(Debug)]
pub struct MarkdownValidator;

impl Default for MarkdownValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl MarkdownValidator {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DocumentValidator for MarkdownValidator {
    async fn validate(&self, content: &MultiModalContent) -> Result<Option<ValidationResult>> {
        if !self.supports_content(content) {
            return Ok(None);
        }

        let text = String::from_utf8_lossy(&content.data);

        let mut details = HashMap::new();
        details.insert("file_size".to_string(), content.data.len().to_string());
        details.insert("line_count".to_string(), text.lines().count().to_string());

        Ok(Some(ValidationResult {
            is_valid: true,
            confidence: 0.9,
            error_message: None,
            details,
        }))
    }

    fn name(&self) -> &str {
        "markdown_validator"
    }

    fn description(&self) -> &str {
        "Validates Markdown document format"
    }
}
