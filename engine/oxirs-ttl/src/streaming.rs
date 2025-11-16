//! Streaming support for memory-efficient parsing of large RDF files

use crate::error::{TurtleParseError, TurtleResult};
use oxirs_core::model::Triple;
use std::io::{BufRead, BufReader, Read};

/// Configuration for streaming parser
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Number of triples to buffer before yielding
    pub batch_size: usize,
    /// Whether to continue parsing after errors
    pub lenient: bool,
    /// Maximum memory to use for buffering (bytes)
    pub max_buffer_size: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            batch_size: 10_000,
            lenient: false,
            max_buffer_size: 100 * 1024 * 1024, // 100 MB
        }
    }
}

impl StreamingConfig {
    /// Create a new streaming configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Enable lenient mode (continue on errors)
    pub fn lenient(mut self, lenient: bool) -> Self {
        self.lenient = lenient;
        self
    }

    /// Set maximum buffer size
    pub fn with_max_buffer_size(mut self, size: usize) -> Self {
        self.max_buffer_size = size;
        self
    }
}

/// Streaming parser that yields batches of triples
pub struct StreamingParser<R: BufRead> {
    reader: R,
    config: StreamingConfig,
    buffer: String,
    triples_parsed: usize,
    bytes_read: usize,
    /// Accumulated prefix declarations to preserve across batches
    prefix_declarations: String,
}

impl<R: Read> StreamingParser<BufReader<R>> {
    /// Create a new streaming parser from a reader
    pub fn new(reader: R) -> Self {
        Self::with_config(reader, StreamingConfig::default())
    }

    /// Create a streaming parser with custom configuration
    pub fn with_config(reader: R, config: StreamingConfig) -> Self {
        Self {
            reader: BufReader::new(reader),
            config,
            buffer: String::new(),
            triples_parsed: 0,
            bytes_read: 0,
            prefix_declarations: String::new(),
        }
    }
}

impl<R: BufRead> StreamingParser<R> {
    /// Create from an existing BufRead
    pub fn from_buf_reader(reader: R, config: StreamingConfig) -> Self {
        Self {
            reader,
            config,
            buffer: String::new(),
            triples_parsed: 0,
            bytes_read: 0,
            prefix_declarations: String::new(),
        }
    }

    /// Get the number of triples parsed so far
    pub fn triples_parsed(&self) -> usize {
        self.triples_parsed
    }

    /// Get the number of bytes read so far
    pub fn bytes_read(&self) -> usize {
        self.bytes_read
    }

    /// Parse the next batch of triples
    pub fn next_batch(&mut self) -> TurtleResult<Option<Vec<Triple>>> {
        use crate::formats::trig::TriGParser;
        use crate::toolkit::Parser;
        use crate::turtle::TurtleParser;
        use oxirs_core::model::Quad;

        // Read up to batch_size lines or until buffer limit
        // But always complete the current statement (read until we see a '.' or '}')
        self.buffer.clear();
        let mut lines_read = 0;
        let target_lines = self.config.batch_size / 10; // Rough estimate: ~10 triples per line
        let mut in_multiline_string = false;
        let mut last_line_ended_statement = false;

        while lines_read < target_lines && self.buffer.len() < self.config.max_buffer_size {
            let mut line = String::new();
            match self.reader.read_line(&mut line) {
                Ok(0) => break, // EOF
                Ok(n) => {
                    self.bytes_read += n;
                    self.buffer.push_str(&line);
                    lines_read += 1;

                    // Track multiline strings
                    let triple_quotes =
                        line.matches("\"\"\"").count() + line.matches("'''").count();
                    if triple_quotes % 2 == 1 {
                        in_multiline_string = !in_multiline_string;
                    }

                    // Check if this line ends a statement (only if not in multiline string)
                    let trimmed = line.trim();
                    if !in_multiline_string && (trimmed.ends_with('.') || trimmed == "}") {
                        last_line_ended_statement = true;
                        // If we've read enough lines and found a statement boundary, stop here
                        if lines_read >= target_lines {
                            break;
                        }
                    } else {
                        last_line_ended_statement = false;
                    }
                }
                Err(e) => return Err(TurtleParseError::io(e)),
            }
        }

        // Continue reading until we complete the current statement
        // (unless we're at EOF or hit the buffer limit)
        while !last_line_ended_statement
            && self.buffer.len() < self.config.max_buffer_size
            && !in_multiline_string
        {
            let mut line = String::new();
            match self.reader.read_line(&mut line) {
                Ok(0) => break, // EOF
                Ok(n) => {
                    self.bytes_read += n;
                    self.buffer.push_str(&line);

                    // Track multiline strings
                    let triple_quotes =
                        line.matches("\"\"\"").count() + line.matches("'''").count();
                    if triple_quotes % 2 == 1 {
                        in_multiline_string = !in_multiline_string;
                    }

                    let trimmed = line.trim();
                    if !in_multiline_string && (trimmed.ends_with('.') || trimmed == "}") {
                        break;
                    }
                }
                Err(e) => return Err(TurtleParseError::io(e)),
            }
        }

        if self.buffer.is_empty() {
            return Ok(None); // EOF
        }

        // Extract prefix and base declarations from this batch
        for line in self.buffer.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("@prefix") || trimmed.starts_with("@base") {
                // Save prefix/base declarations for future batches
                if !self.prefix_declarations.contains(trimmed) {
                    self.prefix_declarations.push_str(trimmed);
                    self.prefix_declarations.push('\n');
                }
            }
        }

        // Prepend accumulated prefix declarations to this batch
        let document = format!("{}{}", self.prefix_declarations, self.buffer);

        // Detect if this is TriG (contains named graphs) or Turtle
        let is_trig = document.contains('{') || document.contains("GRAPH");

        if is_trig {
            // For TriG format with named graphs, read the entire document
            // (proper streaming would require graph-aware state management)
            let mut complete_document = document.clone();
            loop {
                let mut line = String::new();
                match self.reader.read_line(&mut line) {
                    Ok(0) => break, // EOF
                    Ok(n) => {
                        self.bytes_read += n;
                        complete_document.push_str(&line);
                    }
                    Err(e) => return Err(TurtleParseError::io(e)),
                }
            }

            // Use TriG parser for documents with named graphs
            let mut parser = TriGParser::new();
            if self.config.lenient {
                parser.lenient = true;
            }

            match parser.parse(complete_document.as_bytes()) {
                Ok(quads) => {
                    // Extract triples from quads
                    let triples: Vec<Triple> = quads
                        .into_iter()
                        .map(|q: Quad| {
                            Triple::new(
                                q.subject().clone(),
                                q.predicate().clone(),
                                q.object().clone(),
                            )
                        })
                        .collect();
                    self.triples_parsed += triples.len();
                    // Clear buffer to signal EOF on next call
                    self.buffer.clear();
                    Ok(Some(triples))
                }
                Err(_e) if self.config.lenient => {
                    // In lenient mode, return empty batch on error
                    // Clear buffer to signal EOF on next call
                    self.buffer.clear();
                    Ok(Some(Vec::new()))
                }
                Err(e) => Err(e),
            }
        } else {
            // Use Turtle parser for plain triples
            let parser = if self.config.lenient {
                TurtleParser::new_lenient()
            } else {
                TurtleParser::new()
            };

            match parser.parse_document(&document) {
                Ok(triples) => {
                    self.triples_parsed += triples.len();
                    Ok(Some(triples))
                }
                Err(_e) if self.config.lenient => {
                    // In lenient mode, return empty batch on error
                    Ok(Some(Vec::new()))
                }
                Err(e) => Err(e),
            }
        }
    }

    /// Get an iterator over batches
    pub fn batches(self) -> StreamingBatchIterator<R> {
        StreamingBatchIterator { parser: self }
    }

    /// Get an iterator over individual triples
    pub fn triples(self) -> StreamingTripleIterator<R> {
        StreamingTripleIterator {
            parser: self,
            current_batch: Vec::new(),
            batch_index: 0,
        }
    }
}

/// Iterator over batches of triples
pub struct StreamingBatchIterator<R: BufRead> {
    parser: StreamingParser<R>,
}

impl<R: BufRead> Iterator for StreamingBatchIterator<R> {
    type Item = TurtleResult<Vec<Triple>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.parser.next_batch() {
            Ok(Some(batch)) => Some(Ok(batch)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Iterator over individual triples
pub struct StreamingTripleIterator<R: BufRead> {
    parser: StreamingParser<R>,
    current_batch: Vec<Triple>,
    batch_index: usize,
}

impl<R: BufRead> Iterator for StreamingTripleIterator<R> {
    type Item = TurtleResult<Triple>;

    fn next(&mut self) -> Option<Self::Item> {
        // If we have triples in the current batch, return the next one
        if self.batch_index < self.current_batch.len() {
            let triple = self.current_batch[self.batch_index].clone();
            self.batch_index += 1;
            return Some(Ok(triple));
        }

        // Need to load next batch
        match self.parser.next_batch() {
            Ok(Some(batch)) => {
                self.current_batch = batch;
                self.batch_index = 0;
                self.next() // Recursively get first item from new batch
            }
            Ok(None) => None, // EOF
            Err(e) => Some(Err(e)),
        }
    }
}

/// Progress callback for streaming operations
pub trait ProgressCallback: Send {
    /// Called when a batch is parsed
    fn on_batch(&mut self, triples_count: usize, bytes_read: usize);

    /// Called when an error occurs (in lenient mode)
    fn on_error(&mut self, error: &TurtleParseError);
}

/// Simple progress printer
pub struct PrintProgress {
    last_report: usize,
    report_interval: usize,
}

impl PrintProgress {
    /// Create a new progress printer
    pub fn new(report_interval: usize) -> Self {
        Self {
            last_report: 0,
            report_interval,
        }
    }
}

impl Default for PrintProgress {
    fn default() -> Self {
        Self::new(10_000)
    }
}

impl ProgressCallback for PrintProgress {
    fn on_batch(&mut self, triples_count: usize, bytes_read: usize) {
        if triples_count - self.last_report >= self.report_interval {
            eprintln!(
                "Parsed {} triples ({:.2} MB)",
                triples_count,
                bytes_read as f64 / 1_024_000.0
            );
            self.last_report = triples_count;
        }
    }

    fn on_error(&mut self, error: &TurtleParseError) {
        eprintln!("Warning: {}", error);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_streaming_parser_basic() {
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            ex:alice ex:name "Alice" .
            ex:bob ex:name "Bob" .
            ex:charlie ex:name "Charlie" .
        "#;

        let reader = Cursor::new(turtle);
        let mut parser = StreamingParser::new(reader);

        let batch = parser.next_batch().unwrap();
        assert!(batch.is_some());

        let triples = batch.unwrap();
        assert_eq!(triples.len(), 3);
    }

    #[test]
    fn test_batch_iterator() {
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            ex:alice ex:name "Alice" .
            ex:bob ex:name "Bob" .
        "#;

        let reader = Cursor::new(turtle);
        let parser = StreamingParser::new(reader);

        let batches: Vec<_> = parser.batches().collect();
        assert_eq!(batches.len(), 1);
        assert!(batches[0].is_ok());
    }

    #[test]
    fn test_triple_iterator() {
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            ex:alice ex:name "Alice" .
            ex:bob ex:name "Bob" .
        "#;

        let reader = Cursor::new(turtle);
        let parser = StreamingParser::new(reader);

        let triples: Result<Vec<_>, _> = parser.triples().collect();
        assert!(triples.is_ok());
        assert_eq!(triples.unwrap().len(), 2);
    }

    #[test]
    fn test_large_document_streaming() {
        // Generate a large document
        let mut turtle = String::from("@prefix ex: <http://example.org/> .\n");
        for i in 0..1000 {
            turtle.push_str(&format!("ex:subject{} ex:predicate \"object{}\" .\n", i, i));
        }

        let reader = Cursor::new(turtle);
        let config = StreamingConfig::default().with_batch_size(100);
        let mut parser = StreamingParser::with_config(reader, config);

        let mut total_triples = 0;
        while let Some(batch) = parser.next_batch().unwrap() {
            total_triples += batch.len();
        }

        assert_eq!(total_triples, 1000);
    }

    #[test]
    fn test_lenient_mode() {
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            ex:alice ex:name "Alice" .
            invalid syntax here
            ex:bob ex:name "Bob" .
        "#;

        let reader = Cursor::new(turtle);
        let config = StreamingConfig::default().lenient(true);
        let parser = StreamingParser::with_config(reader, config);

        // Should not panic in lenient mode
        let _triples: Vec<_> = parser.triples().collect();
    }

    #[test]
    fn test_progress_tracking() {
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            ex:alice ex:name "Alice" .
            ex:bob ex:name "Bob" .
        "#;

        let reader = Cursor::new(turtle);
        let mut parser = StreamingParser::new(reader);

        let _ = parser.next_batch();

        assert!(parser.triples_parsed() > 0);
        assert!(parser.bytes_read() > 0);
    }
}
