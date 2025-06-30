//! Office document handlers for content processing
//!
//! This module provides handlers for Microsoft Office documents (DOCX, PPTX, XLSX).

#[cfg(feature = "content-processing")]
use crate::content_processing::{
    ContentExtractionConfig, ContentLocation, DocumentFormat, DocumentStructure, ExtractedContent,
    ExtractedTable, FormatHandler, Heading, ProcessingStats,
};
#[cfg(feature = "content-processing")]
use anyhow::{anyhow, Result};
#[cfg(feature = "content-processing")]
use std::collections::HashMap;

/// Base handler for Office documents (DOCX, PPTX, XLSX)
#[cfg(feature = "content-processing")]
pub trait OfficeDocumentHandler {
    fn extract_from_zip(&self, data: &[u8], main_xml_path: &str) -> Result<String> {
        let cursor = std::io::Cursor::new(data);
        let mut archive = zip::ZipArchive::new(cursor)
            .map_err(|e| anyhow!("Failed to open ZIP archive: {}", e))?;

        // Try to find the main content file
        let file = archive
            .by_name(main_xml_path)
            .map_err(|e| anyhow!("Main content file not found: {}", e))?;

        let content =
            std::io::read_to_string(file).map_err(|e| anyhow!("Failed to read content: {}", e))?;

        self.extract_text_from_xml(&content)
    }

    fn extract_text_from_xml(&self, xml: &str) -> Result<String> {
        let mut reader = quick_xml::Reader::from_str(xml);
        let mut buf = Vec::new();
        let mut text_content = Vec::new();
        let mut in_text = false;

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(quick_xml::events::Event::Start(ref e)) => {
                    match e.name().as_ref() {
                        b"w:t" | b"a:t" | b"c" => in_text = true, // Word text, PowerPoint text, Excel cell
                        _ => {}
                    }
                }
                Ok(quick_xml::events::Event::End(ref e)) => match e.name().as_ref() {
                    b"w:t" | b"a:t" | b"c" => in_text = false,
                    _ => {}
                },
                Ok(quick_xml::events::Event::Text(e)) if in_text => {
                    let text = e.unescape().unwrap_or_default();
                    text_content.push(text.to_string());
                }
                Ok(quick_xml::events::Event::Eof) => break,
                Err(e) => return Err(anyhow!("XML parsing error: {}", e)),
                _ => {}
            }
            buf.clear();
        }

        Ok(text_content.join(" "))
    }

    fn extract_metadata_from_zip(&self, data: &[u8]) -> HashMap<String, String> {
        let mut metadata = HashMap::new();

        let cursor = std::io::Cursor::new(data);
        if let Ok(mut archive) = zip::ZipArchive::new(cursor) {
            // Try to read core properties
            if let Ok(file) = archive.by_name("docProps/core.xml") {
                if let Ok(content) = std::io::read_to_string(file) {
                    // Parse core properties XML
                    let mut reader = quick_xml::Reader::from_str(&content);
                    let mut buf = Vec::new();
                    let mut current_element = String::new();

                    loop {
                        match reader.read_event_into(&mut buf) {
                            Ok(quick_xml::events::Event::Start(ref e)) => {
                                current_element =
                                    String::from_utf8_lossy(e.name().as_ref()).to_string();
                            }
                            Ok(quick_xml::events::Event::Text(e)) => {
                                let text = e.unescape().unwrap_or_default();
                                match current_element.as_str() {
                                    "dc:title" => {
                                        metadata.insert("title".to_string(), text.to_string());
                                    }
                                    "dc:creator" => {
                                        metadata.insert("author".to_string(), text.to_string());
                                    }
                                    "dc:subject" => {
                                        metadata.insert("subject".to_string(), text.to_string());
                                    }
                                    "dc:description" => {
                                        metadata
                                            .insert("description".to_string(), text.to_string());
                                    }
                                    _ => {}
                                }
                            }
                            Ok(quick_xml::events::Event::Eof) => break,
                            _ => {}
                        }
                        buf.clear();
                    }
                }
            }
        }

        metadata.insert("size".to_string(), data.len().to_string());
        metadata
    }
}

/// DOCX document handler
#[cfg(feature = "content-processing")]
pub struct DocxHandler;

#[cfg(feature = "content-processing")]
impl OfficeDocumentHandler for DocxHandler {}

#[cfg(feature = "content-processing")]
impl FormatHandler for DocxHandler {
    fn extract_content(
        &self,
        data: &[u8],
        _config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent> {
        let text = self.extract_from_zip(data, "word/document.xml")?;
        let metadata = self.extract_metadata_from_zip(data);
        let title = metadata.get("title").cloned();

        // Extract headings (would need style analysis for proper heading detection)
        let headings = self.extract_docx_headings(&text);

        Ok(ExtractedContent {
            format: DocumentFormat::Docx,
            text,
            metadata,
            images: Vec::new(), // Would require parsing word/media folder
            tables: Vec::new(), // Would require parsing table XML structures
            links: Vec::new(),  // Would require parsing hyperlink relationships
            structure: DocumentStructure {
                title,
                headings,
                page_count: 1, // Would need to analyze page breaks
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
        if data.len() < 4 {
            return false;
        }

        // Check for ZIP signature
        if &data[0..4] != &[0x50, 0x4B, 0x03, 0x04] && &data[0..4] != &[0x50, 0x4B, 0x05, 0x06] {
            return false;
        }

        // Check if it contains DOCX-specific files
        let cursor = std::io::Cursor::new(data);
        if let Ok(mut archive) = zip::ZipArchive::new(cursor) {
            archive.by_name("word/document.xml").is_ok()
                && archive.by_name("[Content_Types].xml").is_ok()
        } else {
            false
        }
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["docx"]
    }
}

#[cfg(feature = "content-processing")]
impl DocxHandler {
    fn extract_docx_headings(&self, text: &str) -> Vec<Heading> {
        let mut headings = Vec::new();

        // Simple heuristic for headings in extracted text
        for (i, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.len() > 3 && trimmed.len() < 100 {
                // Check if line looks like a heading
                let words: Vec<&str> = trimmed.split_whitespace().collect();
                if words.len() <= 8 && words.len() > 0 {
                    let first_char = trimmed.chars().next().unwrap_or(' ');
                    if first_char.is_uppercase() {
                        headings.push(Heading {
                            level: 1, // Would need style information for proper level detection
                            text: trimmed.to_string(),
                            location: ContentLocation {
                                page: None,
                                section: None,
                                char_offset: None,
                                line: Some(i as u32 + 1),
                                column: None,
                            },
                        });
                    }
                }
            }
        }

        headings
    }
}

/// PPTX document handler
#[cfg(feature = "content-processing")]
pub struct PptxHandler;

#[cfg(feature = "content-processing")]
impl OfficeDocumentHandler for PptxHandler {}

#[cfg(feature = "content-processing")]
impl FormatHandler for PptxHandler {
    fn extract_content(
        &self,
        data: &[u8],
        _config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent> {
        // Extract text from all slides
        let mut all_text = Vec::new();
        let cursor = std::io::Cursor::new(data);
        let mut archive = zip::ZipArchive::new(cursor)
            .map_err(|e| anyhow!("Failed to open PPTX archive: {}", e))?;

        // Find all slide files
        let file_names: Vec<String> = (0..archive.len())
            .filter_map(|i| {
                archive
                    .by_index(i)
                    .ok()
                    .map(|file| file.name().to_string())
                    .filter(|name| name.starts_with("ppt/slides/slide") && name.ends_with(".xml"))
            })
            .collect();

        for slide_name in file_names {
            if let Ok(file) = archive.by_name(&slide_name) {
                if let Ok(content) = std::io::read_to_string(file) {
                    if let Ok(slide_text) = self.extract_text_from_xml(&content) {
                        all_text.push(slide_text);
                    }
                }
            }
        }

        let text = all_text.join("\n\n");
        let metadata = self.extract_metadata_from_zip(data);
        let title = metadata.get("title").cloned();

        Ok(ExtractedContent {
            format: DocumentFormat::Pptx,
            text,
            metadata,
            images: Vec::new(),
            tables: Vec::new(),
            links: Vec::new(),
            structure: DocumentStructure {
                title,
                headings: Vec::new(), // Would extract slide titles as headings
                page_count: all_text.len(), // Each slide is a "page"
                section_count: all_text.len(),
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
        if data.len() < 4 {
            return false;
        }

        // Check for ZIP signature
        if &data[0..4] != &[0x50, 0x4B, 0x03, 0x04] && &data[0..4] != &[0x50, 0x4B, 0x05, 0x06] {
            return false;
        }

        // Check if it contains PPTX-specific files
        let cursor = std::io::Cursor::new(data);
        if let Ok(mut archive) = zip::ZipArchive::new(cursor) {
            archive.by_name("ppt/presentation.xml").is_ok()
                && archive.by_name("[Content_Types].xml").is_ok()
        } else {
            false
        }
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["pptx"]
    }
}

/// XLSX document handler
#[cfg(feature = "content-processing")]
pub struct XlsxHandler;

#[cfg(feature = "content-processing")]
impl OfficeDocumentHandler for XlsxHandler {}

#[cfg(feature = "content-processing")]
impl FormatHandler for XlsxHandler {
    fn extract_content(
        &self,
        data: &[u8],
        config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent> {
        let cursor = std::io::Cursor::new(data);
        let mut archive = zip::ZipArchive::new(cursor)
            .map_err(|e| anyhow!("Failed to open XLSX archive: {}", e))?;

        // Extract shared strings first
        let shared_strings = self.extract_shared_strings(&mut archive)?;

        // Extract worksheet content
        let (text, tables) = self.extract_worksheets(&mut archive, &shared_strings, config)?;
        let metadata = self.extract_metadata_from_zip(data);
        let title = metadata.get("title").cloned();

        Ok(ExtractedContent {
            format: DocumentFormat::Xlsx,
            text,
            metadata,
            images: Vec::new(),
            tables,
            links: Vec::new(),
            structure: DocumentStructure {
                title,
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
        if data.len() < 4 {
            return false;
        }

        // Check for ZIP signature
        if &data[0..4] != &[0x50, 0x4B, 0x03, 0x04] && &data[0..4] != &[0x50, 0x4B, 0x05, 0x06] {
            return false;
        }

        // Check if it contains XLSX-specific files
        let cursor = std::io::Cursor::new(data);
        if let Ok(mut archive) = zip::ZipArchive::new(cursor) {
            archive.by_name("xl/workbook.xml").is_ok()
                && archive.by_name("[Content_Types].xml").is_ok()
        } else {
            false
        }
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["xlsx"]
    }
}

#[cfg(feature = "content-processing")]
impl XlsxHandler {
    fn extract_shared_strings(
        &self,
        archive: &mut zip::ZipArchive<std::io::Cursor<&[u8]>>,
    ) -> Result<Vec<String>> {
        let mut shared_strings = Vec::new();

        if let Ok(file) = archive.by_name("xl/sharedStrings.xml") {
            let content = std::io::read_to_string(file)
                .map_err(|e| anyhow!("Failed to read shared strings: {}", e))?;

            let mut reader = quick_xml::Reader::from_str(&content);
            let mut buf = Vec::new();
            let mut in_text = false;
            let mut current_string = String::new();

            loop {
                match reader.read_event_into(&mut buf) {
                    Ok(quick_xml::events::Event::Start(ref e)) => {
                        if e.name().as_ref() == b"t" {
                            in_text = true;
                            current_string.clear();
                        }
                    }
                    Ok(quick_xml::events::Event::End(ref e)) => {
                        if e.name().as_ref() == b"t" {
                            in_text = false;
                        } else if e.name().as_ref() == b"si" {
                            shared_strings.push(current_string.clone());
                            current_string.clear();
                        }
                    }
                    Ok(quick_xml::events::Event::Text(e)) if in_text => {
                        let text = e.unescape().unwrap_or_default();
                        current_string.push_str(&text);
                    }
                    Ok(quick_xml::events::Event::Eof) => break,
                    Err(e) => return Err(anyhow!("XML parsing error: {}", e)),
                    _ => {}
                }
                buf.clear();
            }
        }

        Ok(shared_strings)
    }

    fn extract_worksheets(
        &self,
        archive: &mut zip::ZipArchive<std::io::Cursor<&[u8]>>,
        shared_strings: &[String],
        config: &ContentExtractionConfig,
    ) -> Result<(String, Vec<ExtractedTable>)> {
        let mut all_text = Vec::new();
        let mut tables = Vec::new();

        // Find all worksheet files
        let file_names: Vec<String> = (0..archive.len())
            .filter_map(|i| {
                archive
                    .by_index(i)
                    .ok()
                    .map(|file| file.name().to_string())
                    .filter(|name| {
                        name.starts_with("xl/worksheets/sheet") && name.ends_with(".xml")
                    })
            })
            .collect();

        for (sheet_index, sheet_name) in file_names.iter().enumerate() {
            if let Ok(file) = archive.by_name(sheet_name) {
                if let Ok(content) = std::io::read_to_string(file) {
                    let (sheet_text, sheet_table) =
                        self.extract_sheet_content(&content, shared_strings)?;
                    all_text.push(sheet_text);

                    if config.extract_tables && !sheet_table.rows.is_empty() {
                        let mut table = sheet_table;
                        table.caption = Some(format!("Sheet {}", sheet_index + 1));
                        tables.push(table);
                    }
                }
            }
        }

        Ok((all_text.join("\n\n"), tables))
    }

    fn extract_sheet_content(
        &self,
        xml: &str,
        shared_strings: &[String],
    ) -> Result<(String, ExtractedTable)> {
        let mut reader = quick_xml::Reader::from_str(xml);
        let mut buf = Vec::new();
        let mut cells = Vec::new();
        let mut current_cell = (0, 0, String::new()); // (row, col, value)
        let mut in_value = false;
        let mut cell_type_owned = String::from("str"); // Default to string
        let mut row_index = 0;
        let mut col_index = 0;

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(quick_xml::events::Event::Start(ref e)) => {
                    match e.name().as_ref() {
                        b"c" => {
                            // Cell
                            // Parse cell reference and type
                            for attr in e.attributes() {
                                if let Ok(attr) = attr {
                                    match attr.key.as_ref() {
                                        b"r" => {
                                            // Parse cell reference like "A1", "B2", etc.
                                            let cell_ref = String::from_utf8_lossy(&attr.value);
                                            (col_index, row_index) =
                                                self.parse_cell_reference(&cell_ref);
                                        }
                                        b"t" => {
                                            cell_type_owned =
                                                String::from_utf8_lossy(&attr.value).to_string();
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                        b"v" => {
                            // Cell value
                            in_value = true;
                            current_cell = (row_index, col_index, String::new());
                        }
                        _ => {}
                    }
                }
                Ok(quick_xml::events::Event::End(ref e)) => {
                    match e.name().as_ref() {
                        b"c" => {
                            if !current_cell.2.is_empty() {
                                cells.push(current_cell.clone());
                            }
                            // Reset for next cell
                            cell_type_owned = String::from("str");
                        }
                        b"v" => {
                            in_value = false;
                        }
                        _ => {}
                    }
                }
                Ok(quick_xml::events::Event::Text(e)) if in_value => {
                    let text = e.unescape().unwrap_or_default();
                    if cell_type_owned == "s" {
                        // Shared string reference
                        if let Ok(index) = text.parse::<usize>() {
                            if index < shared_strings.len() {
                                current_cell.2 = shared_strings[index].clone();
                            }
                        }
                    } else {
                        current_cell.2 = text.to_string();
                    }
                }
                Ok(quick_xml::events::Event::Eof) => break,
                Err(e) => return Err(anyhow!("XML parsing error: {}", e)),
                _ => {}
            }
            buf.clear();
        }

        // Convert cells to table format
        let (text, table) = self.cells_to_table(cells);
        Ok((text, table))
    }

    fn parse_cell_reference(&self, cell_ref: &str) -> (usize, usize) {
        let mut col = 0;
        let mut row = 0;
        let mut i = 0;

        // Parse column letters
        for ch in cell_ref.chars() {
            if ch.is_alphabetic() {
                col = col * 26 + (ch.to_ascii_uppercase() as u8 - b'A') as usize + 1;
                i += 1;
            } else {
                break;
            }
        }

        // Parse row number
        if let Ok(row_num) = cell_ref[i..].parse::<usize>() {
            row = row_num;
        }

        (col.saturating_sub(1), row.saturating_sub(1))
    }

    fn cells_to_table(&self, cells: Vec<(usize, usize, String)>) -> (String, ExtractedTable) {
        if cells.is_empty() {
            return (
                String::new(),
                ExtractedTable {
                    headers: Vec::new(),
                    rows: Vec::new(),
                    caption: None,
                    location: ContentLocation {
                        page: Some(1),
                        section: None,
                        char_offset: None,
                        line: None,
                        column: None,
                    },
                },
            );
        }

        // Find dimensions
        let max_row = cells.iter().map(|(r, _, _)| *r).max().unwrap_or(0);
        let max_col = cells.iter().map(|(_, c, _)| *c).max().unwrap_or(0);

        // Create grid
        let mut grid = vec![vec![String::new(); max_col + 1]; max_row + 1];
        for (row, col, value) in cells {
            if row <= max_row && col <= max_col {
                grid[row][col] = value;
            }
        }

        // Extract headers (first row) and data rows
        let headers = if !grid.is_empty() {
            grid[0].clone()
        } else {
            Vec::new()
        };

        let rows = if grid.len() > 1 {
            grid[1..].to_vec()
        } else {
            Vec::new()
        };

        // Create text representation
        let mut text_parts = Vec::new();
        for row in &grid {
            let row_text = row
                .iter()
                .filter(|cell| !cell.is_empty())
                .cloned()
                .collect::<Vec<_>>()
                .join(" | ");
            if !row_text.is_empty() {
                text_parts.push(row_text);
            }
        }
        let text = text_parts.join("\n");

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

        (text, table)
    }
}
