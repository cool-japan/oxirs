//! PDF document handler for content processing
//!
//! This module provides PDF document parsing and content extraction capabilities.

#[cfg(feature = "content-processing")]
use crate::content_processing::{
    ContentExtractionConfig, ContentLocation, DocumentFormat, DocumentStructure, ExtractedContent,
    ExtractedImage, ExtractedLink, ExtractedTable, FormatHandler, Heading, ProcessingStats,
    TocEntry,
};
#[cfg(feature = "content-processing")]
use anyhow::{anyhow, Result};
#[cfg(feature = "content-processing")]
use std::collections::HashMap;

/// PDF document handler
#[cfg(feature = "content-processing")]
pub struct PdfHandler;

#[cfg(feature = "content-processing")]
impl FormatHandler for PdfHandler {
    fn extract_content(
        &self,
        data: &[u8],
        config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent> {
        let text = match pdf_extract::extract_text_from_mem(data) {
            Ok(extracted_text) => {
                if extracted_text.trim().is_empty() {
                    return Err(anyhow!("No text content found in PDF"));
                }
                extracted_text
            }
            Err(e) => {
                return Err(anyhow!("Failed to extract text from PDF: {}", e));
            }
        };

        // Enhanced metadata extraction
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "PDF".to_string());
        metadata.insert("size".to_string(), data.len().to_string());
        metadata.insert("extraction_method".to_string(), "pdf-extract".to_string());

        // Try to extract metadata from PDF header
        if let Some(pdf_metadata) = self.extract_pdf_metadata(data) {
            for (key, value) in pdf_metadata {
                metadata.insert(key, value);
            }
        }

        // Enhanced page count estimation
        let estimated_pages = text.matches("\x0C").count().max(1); // Form feed character

        // Extract structural elements
        let headings = self.extract_pdf_headings(&text);
        let tables = if config.extract_tables {
            self.extract_pdf_tables(&text)
        } else {
            Vec::new()
        };

        let links = if config.extract_links {
            self.extract_pdf_links(&text)
        } else {
            Vec::new()
        };

        // Enhanced table of contents generation
        let toc = self.generate_table_of_contents(&headings);

        // Extract images (basic implementation)
        let images = if config.extract_images {
            self.extract_pdf_images(data, config).unwrap_or_default()
        } else {
            Vec::new()
        };

        Ok(ExtractedContent {
            format: DocumentFormat::Pdf,
            text: text.trim().to_string(),
            metadata,
            images,
            tables,
            links,
            structure: DocumentStructure {
                title: self.extract_pdf_title(&text),
                headings: headings.clone(),
                page_count: estimated_pages,
                section_count: headings.len().max(1),
                table_of_contents: toc,
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
        data.len() >= 4 && &data[0..4] == b"%PDF"
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["pdf"]
    }
}

#[cfg(feature = "content-processing")]
impl PdfHandler {
    fn extract_pdf_title(&self, text: &str) -> Option<String> {
        // Look for title-like patterns at the beginning of the document
        let lines: Vec<&str> = text.lines().take(10).collect();
        for line in lines {
            let trimmed = line.trim();
            if trimmed.len() > 5
                && trimmed.len() < 100
                && !trimmed.contains("http")
                && !trimmed.contains("www")
            {
                // Simple heuristic: first substantial line might be title
                return Some(trimmed.to_string());
            }
        }
        None
    }

    fn extract_pdf_headings(&self, text: &str) -> Vec<Heading> {
        let mut headings = Vec::new();

        // Simple heuristic: lines that are shorter, have capital words, and are followed by text
        let lines: Vec<&str> = text.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.len() > 5 && trimmed.len() < 80 {
                // Check if it looks like a heading
                let words: Vec<&str> = trimmed.split_whitespace().collect();
                let capitalized_words = words
                    .iter()
                    .filter(|w| w.chars().next().is_some_and(|c| c.is_uppercase()))
                    .count();

                if capitalized_words >= words.len() / 2 && words.len() <= 10 {
                    headings.push(Heading {
                        level: 1, // Simple assumption - would need better analysis
                        text: trimmed.to_string(),
                        location: ContentLocation {
                            page: None,
                            section: None,
                            char_offset: None,
                            line: Some(i + 1),
                            column: None,
                        },
                    });
                }
            }
        }

        headings
    }

    /// Extract tables from PDF text using pattern recognition
    fn extract_pdf_tables(&self, text: &str) -> Vec<ExtractedTable> {
        let mut tables = Vec::new();
        let lines: Vec<&str> = text.lines().collect();

        let mut current_table: Vec<Vec<String>> = Vec::new();
        let mut in_table = false;

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Detect table-like patterns (multiple columns separated by spaces/tabs)
            let parts: Vec<&str> = trimmed.split_whitespace().collect();

            if parts.len() >= 2 && parts.len() <= 8 {
                // Check if this looks like a table row
                let has_numbers = parts.iter().any(|p| p.parse::<f64>().is_ok());
                let consistent_spacing =
                    trimmed.contains('\t') || trimmed.matches("  ").count() >= 2;

                if has_numbers || consistent_spacing {
                    if !in_table {
                        in_table = true;
                        current_table.clear();
                    }

                    let row: Vec<String> = parts.iter().map(|s| s.to_string()).collect();
                    current_table.push(row);
                } else if in_table && current_table.len() >= 2 {
                    // End of table detected
                    tables.push(ExtractedTable {
                        headers: if current_table.len() > 1 {
                            current_table[0].clone()
                        } else {
                            Vec::new()
                        },
                        rows: current_table[1..].to_vec(),
                        caption: None,
                        location: ContentLocation {
                            page: None,
                            section: None,
                            char_offset: None,
                            line: Some(i + 1),
                            column: None,
                        },
                    });

                    in_table = false;
                    current_table.clear();
                }
            } else if in_table {
                // Non-table line encountered, end current table
                if current_table.len() >= 2 {
                    tables.push(ExtractedTable {
                        headers: if current_table.len() > 1 {
                            current_table[0].clone()
                        } else {
                            Vec::new()
                        },
                        rows: current_table[1..].to_vec(),
                        caption: None,
                        location: ContentLocation {
                            page: None,
                            section: None,
                            char_offset: None,
                            line: Some(i + 1),
                            column: None,
                        },
                    });
                }

                in_table = false;
                current_table.clear();
            }
        }

        // Handle table at end of document
        if in_table && current_table.len() >= 2 {
            tables.push(ExtractedTable {
                headers: if current_table.len() > 1 {
                    current_table[0].clone()
                } else {
                    Vec::new()
                },
                rows: current_table[1..].to_vec(),
                caption: None,
                location: ContentLocation {
                    page: None,
                    section: None,
                    char_offset: None,
                    line: Some(lines.len()),
                    column: None,
                },
            });
        }

        tables
    }

    /// Extract links from PDF text
    fn extract_pdf_links(&self, text: &str) -> Vec<ExtractedLink> {
        let mut links = Vec::new();

        // Regular expressions for different link types
        let url_regex =
            regex::Regex::new(r"https?://[^\s\)]+").expect("URL regex pattern is valid");
        let email_regex = regex::Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
            .expect("email regex pattern is valid");

        // Find HTTP/HTTPS URLs
        for mat in url_regex.find_iter(text) {
            let url = mat
                .as_str()
                .trim_end_matches(&['.', ',', ')', ']', '}'][..]);
            links.push(ExtractedLink {
                url: url.to_string(),
                text: url.to_string(),
                title: None,
                location: ContentLocation {
                    page: None,
                    section: None,
                    char_offset: None,
                    line: None,
                    column: None,
                },
            });
        }

        // Find email addresses
        for mat in email_regex.find_iter(text) {
            let email = mat.as_str();
            links.push(ExtractedLink {
                url: format!("mailto:{}", email),
                text: email.to_string(),
                title: None,
                location: ContentLocation {
                    page: None,
                    section: None,
                    char_offset: None,
                    line: None,
                    column: None,
                },
            });
        }

        links
    }

    /// Extract basic metadata from PDF bytes
    fn extract_pdf_metadata(&self, data: &[u8]) -> Option<HashMap<String, String>> {
        let mut metadata = HashMap::new();

        // Simple PDF metadata extraction (looking for Info dictionary patterns)
        let content = String::from_utf8_lossy(data).into_owned();

        // Look for title
        if let Some(title_match) = regex::Regex::new(r"/Title\s*\(\s*([^)]+)\s*\)")
            .expect("title regex pattern is valid")
            .captures(&content)
        {
            if let Some(title) = title_match.get(1) {
                metadata.insert("title".to_string(), title.as_str().to_string());
            }
        }

        // Look for author
        if let Some(author_match) = regex::Regex::new(r"/Author\s*\(\s*([^)]+)\s*\)")
            .expect("author regex pattern is valid")
            .captures(&content)
        {
            if let Some(author) = author_match.get(1) {
                metadata.insert("author".to_string(), author.as_str().to_string());
            }
        }

        // Look for subject
        if let Some(subject_match) = regex::Regex::new(r"/Subject\s*\(\s*([^)]+)\s*\)")
            .expect("subject regex pattern is valid")
            .captures(&content)
        {
            if let Some(subject) = subject_match.get(1) {
                metadata.insert("subject".to_string(), subject.as_str().to_string());
            }
        }

        // Look for creation date
        if let Some(date_match) = regex::Regex::new(r"/CreationDate\s*\(\s*([^)]+)\s*\)")
            .expect("creation date regex pattern is valid")
            .captures(&content)
        {
            if let Some(date) = date_match.get(1) {
                metadata.insert("creation_date".to_string(), date.as_str().to_string());
            }
        }

        if metadata.is_empty() {
            None
        } else {
            Some(metadata)
        }
    }

    /// Extract images from PDF (basic implementation)
    fn extract_pdf_images(
        &self,
        _data: &[u8],
        config: &ContentExtractionConfig,
    ) -> Result<Vec<ExtractedImage>> {
        if config.extract_images {
            // This is a placeholder implementation
            // In a real scenario, you'd use a PDF library that can extract embedded images
            // For now, just return empty vector
            Ok(Vec::new())
        } else {
            Ok(Vec::new())
        }
    }

    /// Generate table of contents from headings
    fn generate_table_of_contents(&self, headings: &[Heading]) -> Vec<TocEntry> {
        headings
            .iter()
            .map(|heading| TocEntry {
                title: heading.text.clone(),
                level: heading.level,
                page: heading.location.page,
                location: heading.location.clone(),
            })
            .collect()
    }
}
