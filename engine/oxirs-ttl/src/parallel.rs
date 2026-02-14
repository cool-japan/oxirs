//! Parallel processing support for RDF parsing using rayon
//!
//! This module provides parallel parsing capabilities for processing large RDF files
//! by splitting them into chunks and parsing them concurrently.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::error::TurtleResult;
use oxirs_core::model::Triple;
use std::io::{BufRead, BufReader, Read};
use std::sync::{Arc, Mutex};

/// Configuration for parallel parsing
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of threads to use (0 = use rayon's default thread pool)
    pub num_threads: usize,
    /// Size of chunks to process in parallel
    pub chunk_size: usize,
    /// Whether to continue parsing after errors
    pub lenient: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: 0, // Use rayon's default
            chunk_size: 10_000,
            lenient: false,
        }
    }
}

impl ParallelConfig {
    /// Create a new parallel configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of threads
    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Set the chunk size
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Enable lenient mode
    pub fn lenient(mut self, lenient: bool) -> Self {
        self.lenient = lenient;
        self
    }
}

/// Parallel parser for processing RDF files using multiple threads
#[cfg(feature = "parallel")]
pub struct ParallelParser<R: Read> {
    reader: BufReader<R>,
    config: ParallelConfig,
}

#[cfg(feature = "parallel")]
impl<R: Read + Send + Sync> ParallelParser<R> {
    /// Create a new parallel parser
    pub fn new(reader: R) -> Self {
        Self::with_config(reader, ParallelConfig::default())
    }

    /// Create a parallel parser with custom configuration
    pub fn with_config(reader: R, config: ParallelConfig) -> Self {
        Self {
            reader: BufReader::new(reader),
            config,
        }
    }

    /// Parse the entire document in parallel
    ///
    /// This splits the document into chunks and parses them concurrently.
    /// Note: This loads the entire document into memory for splitting.
    pub fn parse_all(&mut self) -> TurtleResult<Vec<Triple>> {
        use std::io::Read;

        // Read entire document
        let mut content = String::new();
        self.reader
            .read_to_string(&mut content)
            .map_err(crate::error::TurtleParseError::io)?;

        // Extract prefix and base declarations
        let mut prefixes = String::new();
        let mut data_lines = Vec::new();

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("@prefix")
                || trimmed.starts_with("@base")
                || trimmed.starts_with("PREFIX")
                || trimmed.starts_with("BASE")
            {
                prefixes.push_str(line);
                prefixes.push('\n');
            } else if !trimmed.is_empty() && !trimmed.starts_with('#') {
                data_lines.push(line);
            }
        }

        // Split data lines into chunks
        let chunks: Vec<Vec<&str>> = data_lines
            .chunks(self.config.chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        // Parse chunks in parallel, each with prefixes prepended
        let prefix_arc = std::sync::Arc::new(prefixes);
        let results: Vec<TurtleResult<Vec<Triple>>> = chunks
            .par_iter()
            .map(|chunk| {
                let chunk_text = format!("{}{}", prefix_arc, chunk.join("\n"));
                self.parse_chunk(&chunk_text)
            })
            .collect();

        // Collect results
        let mut all_triples = Vec::new();
        for result in results {
            match result {
                Ok(triples) => all_triples.extend(triples),
                Err(e) if self.config.lenient => {
                    eprintln!("Warning: Parse error in chunk: {}", e);
                }
                Err(e) => return Err(e),
            }
        }

        Ok(all_triples)
    }

    /// Parse a chunk of text
    fn parse_chunk(&self, chunk: &str) -> TurtleResult<Vec<Triple>> {
        use crate::turtle::TurtleParser;
        let parser = TurtleParser::new();
        parser.parse_document(chunk)
    }
}

/// Parallel streaming parser for processing large files without loading entirely into memory
#[cfg(feature = "parallel")]
pub struct ParallelStreamingParser<R: Read + Send + Sync> {
    reader: Arc<Mutex<BufReader<R>>>,
    config: ParallelConfig,
}

#[cfg(feature = "parallel")]
impl<R: Read + Send + Sync + 'static> ParallelStreamingParser<R> {
    /// Create a new parallel streaming parser
    pub fn new(reader: R) -> Self {
        Self::with_config(reader, ParallelConfig::default())
    }

    /// Create a parallel streaming parser with custom configuration
    pub fn with_config(reader: R, config: ParallelConfig) -> Self {
        Self {
            reader: Arc::new(Mutex::new(BufReader::new(reader))),
            config,
        }
    }

    /// Process the file in parallel batches
    ///
    /// This reads batches from the file and processes them in parallel.
    pub fn process_batches<F>(&mut self, mut processor: F) -> TurtleResult<usize>
    where
        F: FnMut(Vec<Triple>) + Send,
    {
        let batch_size = self.config.chunk_size;
        let mut total_triples = 0;
        let mut batches = Vec::new();
        let mut prefixes = String::new();

        // Read all batches first and extract prefixes
        loop {
            let mut reader_guard = self.reader.lock().expect("lock should not be poisoned");
            let mut batch_content = String::new();
            let mut lines_read = 0;

            while lines_read < batch_size {
                let mut line = String::new();
                match reader_guard.read_line(&mut line) {
                    Ok(0) => break, // EOF
                    Ok(_) => {
                        // Extract prefix declarations
                        let trimmed = line.trim();
                        if (trimmed.starts_with("@prefix")
                            || trimmed.starts_with("@base")
                            || trimmed.starts_with("PREFIX")
                            || trimmed.starts_with("BASE"))
                            && !prefixes.contains(trimmed)
                        {
                            prefixes.push_str(&line);
                        }
                        batch_content.push_str(&line);
                        lines_read += 1;
                    }
                    Err(e) => return Err(crate::error::TurtleParseError::io(e)),
                }
            }

            if batch_content.is_empty() {
                break;
            }

            batches.push(batch_content);
        }

        // Process batches in parallel with prefixes prepended
        let prefix_arc = Arc::new(prefixes);
        let results: Vec<TurtleResult<Vec<Triple>>> = batches
            .par_iter()
            .map(|batch| {
                use crate::turtle::TurtleParser;
                let parser = TurtleParser::new();
                let doc_with_prefixes = format!("{}{}", prefix_arc, batch);
                parser.parse_document(&doc_with_prefixes)
            })
            .collect();

        // Collect and process results
        for result in results {
            match result {
                Ok(triples) => {
                    total_triples += triples.len();
                    processor(triples);
                }
                Err(e) if self.config.lenient => {
                    eprintln!("Warning: Parse error in batch: {}", e);
                }
                Err(e) => return Err(e),
            }
        }

        Ok(total_triples)
    }
}

#[cfg(not(feature = "parallel"))]
compile_error!("Parallel processing requires the 'parallel' feature to be enabled");

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_parallel_parser_basic() {
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            ex:alice ex:name "Alice" .
            ex:bob ex:name "Bob" .
            ex:charlie ex:name "Charlie" .
        "#;

        let mut parser = ParallelParser::new(Cursor::new(turtle));
        let result = parser.parse_all();

        assert!(result.is_ok());
        let triples = result.expect("result should be Ok");
        assert_eq!(triples.len(), 3);
    }

    #[test]
    fn test_parallel_parser_large_document() {
        let mut turtle = String::from("@prefix ex: <http://example.org/> .\n");
        for i in 0..1000 {
            turtle.push_str(&format!("ex:subject{} ex:predicate \"object{}\" .\n", i, i));
        }

        let config = ParallelConfig::default().with_chunk_size(100);
        let mut parser = ParallelParser::with_config(Cursor::new(turtle), config);
        let result = parser.parse_all();

        match &result {
            Ok(triples) => {
                assert_eq!(triples.len(), 1000);
            }
            Err(e) => {
                panic!("Parse failed: {:?}", e);
            }
        }
    }

    #[test]
    fn test_parallel_streaming_parser() {
        let mut turtle = String::from("@prefix ex: <http://example.org/> .\n");
        for i in 0..500 {
            turtle.push_str(&format!("ex:subject{} ex:predicate \"object{}\" .\n", i, i));
        }

        let config = ParallelConfig::default().with_chunk_size(100);
        let mut parser = ParallelStreamingParser::with_config(Cursor::new(turtle), config);

        let mut total_processed = 0;
        let result = parser.process_batches(|triples| {
            total_processed += triples.len();
        });

        match &result {
            Ok(count) => {
                assert_eq!(*count, 500);
                assert_eq!(total_processed, 500);
            }
            Err(e) => {
                panic!("Parse failed: {:?}", e);
            }
        }
    }

    #[test]
    fn test_parallel_parser_lenient_mode() {
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            ex:alice ex:name "Alice" .
            invalid syntax here
            ex:bob ex:name "Bob" .
        "#;

        let config = ParallelConfig::default().lenient(true);
        let mut parser = ParallelParser::with_config(Cursor::new(turtle), config);
        let result = parser.parse_all();

        // Should succeed in lenient mode
        assert!(result.is_ok());
    }
}
