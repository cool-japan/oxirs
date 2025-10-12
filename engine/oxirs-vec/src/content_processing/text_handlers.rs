//! Text format handlers for content processing
//!
//! This module provides handlers for plain text, HTML, XML, and Markdown documents.

#[cfg(feature = "content-processing")]
use crate::content_processing::{
    ContentExtractionConfig, ContentLocation, DocumentFormat, DocumentStructure, ExtractedContent,
    FormatHandler, Heading, ProcessingStats,
};
#[cfg(feature = "content-processing")]
use anyhow::Result;
#[cfg(feature = "content-processing")]
use std::collections::HashMap;

/// Plain text handler
#[cfg(feature = "content-processing")]
pub struct PlainTextHandler;

#[cfg(feature = "content-processing")]
impl FormatHandler for PlainTextHandler {
    fn extract_content(
        &self,
        data: &[u8],
        _config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent> {
        let text = String::from_utf8_lossy(data).to_string();

        Ok(ExtractedContent {
            format: DocumentFormat::PlainText,
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
        // Check if data is valid UTF-8
        String::from_utf8(data.to_vec()).is_ok()
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["txt", "text"]
    }
}

/// HTML handler
#[cfg(feature = "content-processing")]
pub struct HtmlHandler;

#[cfg(feature = "content-processing")]
impl FormatHandler for HtmlHandler {
    fn extract_content(
        &self,
        data: &[u8],
        config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent> {
        let html = String::from_utf8_lossy(data);

        // Simple HTML text extraction (in practice, would use a proper HTML parser)
        let text = self.extract_text_from_html(&html);
        let headings = self.extract_headings(&html);
        let links = if config.extract_links {
            self.extract_links(&html)
        } else {
            Vec::new()
        };

        let metadata = self.extract_metadata(&html);
        let title = metadata.get("title").cloned();

        Ok(ExtractedContent {
            format: DocumentFormat::Html,
            text,
            metadata,
            images: Vec::new(), // Would implement image extraction
            tables: Vec::new(), // Would implement table extraction
            links,
            structure: DocumentStructure {
                title,
                headings,
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
        content.contains("<html") || content.contains("<!DOCTYPE")
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["html", "htm"]
    }
}

#[cfg(feature = "content-processing")]
impl HtmlHandler {
    fn extract_text_from_html(&self, html: &str) -> String {
        // Very basic HTML text extraction
        // In practice, would use html5ever or similar
        let mut text = html.to_string();

        // Remove script and style elements
        text = regex::Regex::new(r"<script[^>]*>.*?</script>")
            .unwrap()
            .replace_all(&text, "")
            .to_string();
        text = regex::Regex::new(r"<style[^>]*>.*?</style>")
            .unwrap()
            .replace_all(&text, "")
            .to_string();

        // Remove HTML tags
        text = regex::Regex::new(r"<[^>]*>")
            .unwrap()
            .replace_all(&text, " ")
            .to_string();

        // Clean up whitespace
        text = regex::Regex::new(r"\s+")
            .unwrap()
            .replace_all(&text, " ")
            .to_string();

        text.trim().to_string()
    }

    fn extract_headings(&self, html: &str) -> Vec<Heading> {
        let mut headings = Vec::new();

        for level in 1..=6 {
            let pattern = format!(r"<h{}[^>]*>(.*?)</h{}>", level, level);
            if let Ok(re) = regex::Regex::new(&pattern) {
                for (i, capture) in re.captures_iter(html).enumerate() {
                    if let Some(heading_text) = capture.get(1) {
                        let text = regex::Regex::new(r"<[^>]*>")
                            .unwrap()
                            .replace_all(heading_text.as_str(), "")
                            .trim()
                            .to_string();

                        headings.push(Heading {
                            level,
                            text,
                            location: ContentLocation {
                                page: None,
                                section: Some(i),
                                char_offset: None,
                                line: None,
                                column: None,
                            },
                        });
                    }
                }
            }
        }

        headings
    }

    fn extract_links(&self, html: &str) -> Vec<crate::content_processing::ExtractedLink> {
        let mut links = Vec::new();

        if let Ok(re) = regex::Regex::new(r#"<a[^>]*href\s*=\s*["']([^"']*)["'][^>]*>(.*?)</a>"#) {
            for capture in re.captures_iter(html) {
                if let (Some(url), Some(text)) = (capture.get(1), capture.get(2)) {
                    links.push(crate::content_processing::ExtractedLink {
                        url: url.as_str().to_string(),
                        text: regex::Regex::new(r"<[^>]*>")
                            .unwrap()
                            .replace_all(text.as_str(), "")
                            .trim()
                            .to_string(),
                        title: None,
                        location: crate::content_processing::ContentLocation {
                            page: None,
                            section: None,
                            char_offset: None,
                            line: None,
                            column: None,
                        },
                    });
                }
            }
        }

        links
    }

    fn extract_metadata(&self, html: &str) -> HashMap<String, String> {
        let mut metadata = HashMap::new();

        // Extract title
        if let Ok(re) = regex::Regex::new(r"<title[^>]*>(.*?)</title>") {
            if let Some(capture) = re.captures(html) {
                if let Some(title) = capture.get(1) {
                    metadata.insert("title".to_string(), title.as_str().trim().to_string());
                }
            }
        }

        // Extract meta tags
        if let Ok(re) = regex::Regex::new(
            r#"<meta[^>]*name\s*=\s*["']([^"']*)["'][^>]*content\s*=\s*["']([^"']*)["'][^>]*>"#,
        ) {
            for capture in re.captures_iter(html) {
                if let (Some(name), Some(content)) = (capture.get(1), capture.get(2)) {
                    metadata.insert(name.as_str().to_string(), content.as_str().to_string());
                }
            }
        }

        metadata
    }
}

/// XML handler
#[cfg(feature = "content-processing")]
pub struct XmlHandler;

#[cfg(feature = "content-processing")]
impl FormatHandler for XmlHandler {
    fn extract_content(
        &self,
        data: &[u8],
        _config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent> {
        let xml = String::from_utf8_lossy(data);

        // Basic XML text extraction
        let text = self.extract_text_from_xml(&xml);

        Ok(ExtractedContent {
            format: DocumentFormat::Xml,
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
        content.trim_start().starts_with("<?xml") || content.contains("<") && content.contains(">")
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["xml"]
    }
}

#[cfg(feature = "content-processing")]
impl XmlHandler {
    fn extract_text_from_xml(&self, xml: &str) -> String {
        // Basic XML text extraction - strip tags and return text content
        let text = regex::Regex::new(r"<[^>]*>")
            .unwrap()
            .replace_all(xml, " ")
            .to_string();

        // Clean up whitespace
        regex::Regex::new(r"\s+")
            .unwrap()
            .replace_all(&text, " ")
            .trim()
            .to_string()
    }
}

/// Markdown handler
#[cfg(feature = "content-processing")]
pub struct MarkdownHandler;

#[cfg(feature = "content-processing")]
impl FormatHandler for MarkdownHandler {
    fn extract_content(
        &self,
        data: &[u8],
        config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent> {
        let markdown = String::from_utf8_lossy(data);

        let text = self.extract_text_from_markdown(&markdown);
        let headings = self.extract_headings(&markdown);
        let links = if config.extract_links {
            self.extract_links(&markdown)
        } else {
            Vec::new()
        };

        Ok(ExtractedContent {
            format: DocumentFormat::Markdown,
            text,
            metadata: HashMap::new(),
            images: Vec::new(),
            tables: Vec::new(),
            links,
            structure: DocumentStructure {
                title: None,
                headings,
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
        // Check for common markdown patterns
        content.contains("#")
            || content.contains("*")
            || content.contains("```")
            || content.contains("[")
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["md", "markdown"]
    }
}

#[cfg(feature = "content-processing")]
impl MarkdownHandler {
    fn extract_text_from_markdown(&self, markdown: &str) -> String {
        let mut text = markdown.to_string();

        // Remove code blocks
        text = regex::Regex::new(r"```[\s\S]*?```")
            .unwrap()
            .replace_all(&text, "")
            .to_string();

        // Remove inline code
        text = regex::Regex::new(r"`[^`]*`")
            .unwrap()
            .replace_all(&text, "")
            .to_string();

        // Remove markdown formatting
        text = regex::Regex::new(r"[*_]{1,2}([^*_]*)[*_]{1,2}")
            .unwrap()
            .replace_all(&text, "$1")
            .to_string();

        // Remove headers
        text = regex::Regex::new(r"^#+\s*(.*)$")
            .unwrap()
            .replace_all(&text, "$1")
            .to_string();

        // Remove links
        text = regex::Regex::new(r"\[([^\]]*)\]\([^)]*\)")
            .unwrap()
            .replace_all(&text, "$1")
            .to_string();

        // Clean up whitespace
        regex::Regex::new(r"\s+")
            .unwrap()
            .replace_all(&text, " ")
            .trim()
            .to_string()
    }

    fn extract_headings(&self, markdown: &str) -> Vec<Heading> {
        let mut headings = Vec::new();

        for (i, line) in markdown.lines().enumerate() {
            if let Some(captures) = regex::Regex::new(r"^(#{1,6})\s+(.+)$")
                .unwrap()
                .captures(line)
            {
                let level = captures[1].len();
                let text = captures[2].to_string();

                headings.push(Heading {
                    level,
                    text,
                    location: ContentLocation {
                        page: None,
                        section: Some(i),
                        char_offset: None,
                        line: Some(i),
                        column: None,
                    },
                });
            }
        }

        headings
    }

    fn extract_links(&self, markdown: &str) -> Vec<crate::content_processing::ExtractedLink> {
        let mut links = Vec::new();

        if let Ok(re) = regex::Regex::new(r"\[([^\]]*)\]\(([^)]*)\)") {
            for capture in re.captures_iter(markdown) {
                if let (Some(text), Some(url)) = (capture.get(1), capture.get(2)) {
                    links.push(crate::content_processing::ExtractedLink {
                        url: url.as_str().to_string(),
                        text: text.as_str().to_string(),
                        title: None,
                        location: crate::content_processing::ContentLocation {
                            page: None,
                            section: None,
                            char_offset: None,
                            line: None,
                            column: None,
                        },
                    });
                }
            }
        }

        links
    }
}
