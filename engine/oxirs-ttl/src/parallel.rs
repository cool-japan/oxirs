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
    /// Target number of complete statements to group into each parallel
    /// chunk. Chunks are always cut on complete-statement boundaries (a
    /// statement is never split across two chunks), so the actual statement
    /// count in a given chunk may occasionally exceed this when a single
    /// statement is unusually large.
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

/// Outcome of a parallel parse.
///
/// In strict mode a per-chunk parse error aborts the whole parse (returned as
/// `Err` from [`ParallelParser::parse_all`]) exactly as before; `errors` is
/// therefore always empty in that mode. In lenient mode, per-chunk parse
/// errors are collected here instead of being silently printed to stderr, so
/// the caller can inspect (or count) how much data failed to parse.
#[derive(Debug, Default)]
pub struct ParallelParseOutcome {
    /// Triples successfully parsed from every chunk.
    pub triples: Vec<Triple>,
    /// Errors collected from chunks that failed to parse (lenient mode only).
    pub errors: Vec<crate::error::TurtleParseError>,
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
    /// Chunks are cut on complete-statement boundaries (see
    /// [`crate::statement_boundary`]) rather than raw line boundaries, so a
    /// statement that spans multiple lines (a pretty-printed predicate/object
    /// list, or a triple-quoted string containing embedded newlines) is
    /// always kept whole in a single chunk instead of being corrupted or
    /// silently dropped at a chunk boundary.
    ///
    /// `@prefix`/`@base`/`PREFIX`/`BASE` declarations are extracted (as
    /// complete statements, so a rare multi-line prefix declaration is still
    /// captured whole) and prepended to every data chunk so prefixed names
    /// resolve correctly no matter which chunk the declaration appeared in.
    ///
    /// In lenient mode, per-chunk parse errors are collected into the
    /// returned [`ParallelParseOutcome::errors`] instead of being silently
    /// printed to stderr.
    ///
    /// Note: This loads the entire document into memory for splitting, since
    /// finding a statement boundary may require looking arbitrarily far ahead
    /// past a string literal.
    pub fn parse_all(&mut self) -> TurtleResult<ParallelParseOutcome> {
        use std::io::Read;

        // Read entire document
        let mut content = String::new();
        self.reader
            .read_to_string(&mut content)
            .map_err(crate::error::TurtleParseError::io)?;

        // Split into complete statements first (rather than by raw line), then
        // separate prefix/base declarations from data statements.
        let boundaries = crate::statement_boundary::statement_boundaries(&content);
        let mut prefix_header = String::new();
        let mut data_statements = String::new();
        let mut start = 0usize;
        for &end in &boundaries {
            let statement = &content[start..end];
            let trimmed = statement.trim_start();
            if trimmed.starts_with("@prefix")
                || trimmed.starts_with("@base")
                || trimmed.starts_with("PREFIX")
                || trimmed.starts_with("prefix")
                || trimmed.starts_with("BASE")
                || trimmed.starts_with("base")
            {
                prefix_header.push_str(statement);
            } else {
                data_statements.push_str(statement);
            }
            start = end;
        }
        // Any trailing bytes after the last complete statement (a truncated
        // or malformed final statement, or trailing whitespace/comments) are
        // still handed to the parser so they can surface a proper syntax
        // error instead of being silently dropped.
        if start < content.len() {
            data_statements.push_str(&content[start..]);
        }

        // Split data statements into chunks, each ending on a complete
        // statement boundary.
        let chunks = crate::statement_boundary::split_into_statement_chunks(
            &data_statements,
            self.config.chunk_size,
        );

        // Parse chunks in parallel, each with the collected prefixes prepended
        let prefix_arc = std::sync::Arc::new(prefix_header);
        let results: Vec<TurtleResult<Vec<Triple>>> = chunks
            .par_iter()
            .map(|chunk| {
                let chunk_text = format!("{prefix_arc}{chunk}");
                self.parse_chunk(&chunk_text)
            })
            .collect();

        // Collect results
        let mut all_triples = Vec::new();
        let mut errors = Vec::new();
        for result in results {
            match result {
                Ok(triples) => all_triples.extend(triples),
                Err(e) if self.config.lenient => {
                    errors.push(e);
                }
                Err(e) => return Err(e),
            }
        }

        Ok(ParallelParseOutcome {
            triples: all_triples,
            errors,
        })
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
        let outcome = result.expect("result should be Ok");
        assert_eq!(outcome.triples.len(), 3);
        assert!(outcome.errors.is_empty());
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
            Ok(outcome) => {
                assert_eq!(outcome.triples.len(), 1000);
                assert!(outcome.errors.is_empty());
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

        // Should succeed in lenient mode, and the chunk-level error must be
        // surfaced to the caller rather than only printed to stderr.
        assert!(result.is_ok());
        let outcome = result.expect("result should be Ok");
        assert!(
            !outcome.errors.is_empty(),
            "lenient mode should report the chunk parse error instead of silently discarding it"
        );
    }

    #[test]
    fn test_parallel_parser_does_not_split_multiline_statement() {
        // Regression test: a statement pretty-printed across multiple lines
        // (semicolon-separated predicate/object list) must survive parallel
        // chunking intact even with a chunk size of 1 statement per chunk,
        // instead of being corrupted or silently dropped because it landed on
        // a raw line-based chunk boundary.
        let turtle = concat!(
            "@prefix ex: <http://example.org/> .\n",
            "ex:alice\n",
            "  ex:name \"Alice\" ;\n",
            "  ex:age \"30\" ;\n",
            "  ex:email \"alice@example.org\" .\n",
            "ex:bob ex:name \"Bob\" .\n",
        );

        let config = ParallelConfig::default().with_chunk_size(1);
        let mut parser = ParallelParser::with_config(Cursor::new(turtle), config);
        let result = parser.parse_all();

        let outcome = result.expect("parsing should succeed");
        assert!(outcome.errors.is_empty());
        // 3 triples for ex:alice (name/age/email) + 1 for ex:bob
        assert_eq!(outcome.triples.len(), 4);
    }
}
