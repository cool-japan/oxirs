//! Convenience functions for common RDF parsing operations
//!
//! This module provides high-level convenience functions that simplify common
//! parsing tasks without requiring manual setup of parsers and readers.
//!
//! # Features
//!
//! - **File parsing**: Parse RDF files directly from paths
//! - **Auto-detection**: Automatically detect format from file extension
//! - **Batch processing**: Process large files in batches with callbacks
//! - **Statistics**: Collect parsing statistics automatically
//!
//! # Examples
//!
//! ## Parse a Turtle file
//!
//! ```rust,no_run
//! use oxirs_ttl::convenience::parse_turtle_file;
//!
//! let triples = parse_turtle_file("data.ttl")?;
//! println!("Parsed {} triples", triples.len());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Parse with auto-detection
//!
//! ```rust,no_run
//! use oxirs_ttl::convenience::parse_rdf_file;
//!
//! let triples = parse_rdf_file("data.ttl")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Process large files in batches
//!
//! ```rust,no_run
//! use oxirs_ttl::convenience::process_rdf_file_in_batches;
//!
//! let mut total = 0;
//! process_rdf_file_in_batches("large_data.ttl", 10000, |batch| {
//!     total += batch.len();
//!     println!("Processed batch of {} triples (total: {})", batch.len(), total);
//!     Ok(())
//! })?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::error::{TurtleParseError, TurtleResult};
use crate::formats::nquads::NQuadsParser;
use crate::formats::ntriples::NTriplesParser;
use crate::formats::trig::TriGParser;
use crate::formats::turtle::TurtleParser;
use crate::streaming::{StreamingConfig, StreamingParser};
use crate::toolkit::{FormatDetector, Parser, RdfFormat};
use oxirs_core::model::{Quad, Triple};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Parse a Turtle file from a path
///
/// This is a convenience function that opens the file, creates a parser,
/// and returns all parsed triples.
///
/// # Example
///
/// ```rust,no_run
/// use oxirs_ttl::convenience::parse_turtle_file;
///
/// let triples = parse_turtle_file("data.ttl")?;
/// println!("Parsed {} triples", triples.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn parse_turtle_file<P: AsRef<Path>>(path: P) -> TurtleResult<Vec<Triple>> {
    let file = File::open(path).map_err(TurtleParseError::io)?;
    let reader = BufReader::new(file);
    TurtleParser::new().parse(reader)
}

/// Parse an N-Triples file from a path
///
/// # Example
///
/// ```rust,no_run
/// use oxirs_ttl::convenience::parse_ntriples_file;
///
/// let triples = parse_ntriples_file("data.nt")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn parse_ntriples_file<P: AsRef<Path>>(path: P) -> TurtleResult<Vec<Triple>> {
    let file = File::open(path).map_err(TurtleParseError::io)?;
    let reader = BufReader::new(file);
    NTriplesParser::new().parse(reader)
}

/// Parse an N-Quads file from a path
///
/// # Example
///
/// ```rust,no_run
/// use oxirs_ttl::convenience::parse_nquads_file;
///
/// let quads = parse_nquads_file("data.nq")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn parse_nquads_file<P: AsRef<Path>>(path: P) -> TurtleResult<Vec<Quad>> {
    let file = File::open(path).map_err(TurtleParseError::io)?;
    let reader = BufReader::new(file);
    NQuadsParser::new().parse(reader)
}

/// Parse a TriG file from a path
///
/// # Example
///
/// ```rust,no_run
/// use oxirs_ttl::convenience::parse_trig_file;
///
/// let quads = parse_trig_file("data.trig")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn parse_trig_file<P: AsRef<Path>>(path: P) -> TurtleResult<Vec<Quad>> {
    let file = File::open(path).map_err(TurtleParseError::io)?;
    let reader = BufReader::new(file);
    TriGParser::new().parse(reader)
}

/// Parse an RDF file with automatic format detection
///
/// The format is detected based on the file extension. Supported extensions:
/// - `.ttl` - Turtle
/// - `.nt` - N-Triples
/// - `.nq` - N-Quads
/// - `.trig` - TriG
///
/// # Example
///
/// ```rust,no_run
/// use oxirs_ttl::convenience::parse_rdf_file;
///
/// let triples = parse_rdf_file("data.ttl")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn parse_rdf_file<P: AsRef<Path>>(path: P) -> TurtleResult<Vec<Triple>> {
    let path_ref = path.as_ref();
    let detector = FormatDetector::new();
    let format = detector
        .detect_from_path(path_ref)
        .ok_or_else(|| {
            TurtleParseError::io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Could not detect format for file: {:?}", path_ref),
            ))
        })?
        .format;

    match format {
        RdfFormat::Turtle => parse_turtle_file(path),
        RdfFormat::NTriples => parse_ntriples_file(path),
        _ => Err(TurtleParseError::io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "Format {:?} returns quads, not triples. Use parse_rdf_file_quads instead.",
                format
            ),
        ))),
    }
}

/// Parse an RDF file with automatic format detection (returns quads)
///
/// This function is similar to `parse_rdf_file` but returns quads,
/// allowing it to handle all supported formats including TriG and N-Quads.
///
/// # Example
///
/// ```rust,no_run
/// use oxirs_ttl::convenience::parse_rdf_file_quads;
///
/// let quads = parse_rdf_file_quads("data.trig")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn parse_rdf_file_quads<P: AsRef<Path>>(path: P) -> TurtleResult<Vec<Quad>> {
    let path_ref = path.as_ref();
    let detector = FormatDetector::new();
    let format = detector
        .detect_from_path(path_ref)
        .ok_or_else(|| {
            TurtleParseError::io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Could not detect format for file: {:?}", path_ref),
            ))
        })?
        .format;

    match format {
        RdfFormat::TriG => parse_trig_file(path),
        RdfFormat::NQuads => parse_nquads_file(path),
        RdfFormat::Turtle => {
            let triples = parse_turtle_file(path)?;
            Ok(triples
                .into_iter()
                .map(|t| {
                    Quad::new(
                        t.subject().clone(),
                        t.predicate().clone(),
                        t.object().clone(),
                        oxirs_core::model::GraphName::DefaultGraph,
                    )
                })
                .collect())
        }
        RdfFormat::NTriples => {
            let triples = parse_ntriples_file(path)?;
            Ok(triples
                .into_iter()
                .map(|t| {
                    Quad::new(
                        t.subject().clone(),
                        t.predicate().clone(),
                        t.object().clone(),
                        oxirs_core::model::GraphName::DefaultGraph,
                    )
                })
                .collect())
        }
    }
}

/// Process a large RDF file in batches
///
/// This function is designed for processing very large files that don't fit in memory.
/// It parses the file in batches and calls the provided callback for each batch.
///
/// # Arguments
///
/// * `path` - Path to the RDF file
/// * `batch_size` - Number of triples per batch
/// * `callback` - Function to call for each batch
///
/// # Example
///
/// ```rust,no_run
/// use oxirs_ttl::convenience::process_rdf_file_in_batches;
///
/// let mut total = 0;
/// process_rdf_file_in_batches("large_data.ttl", 10000, |batch| {
///     total += batch.len();
///     println!("Processed batch of {} triples (total: {})", batch.len(), total);
///     // Insert into database, etc.
///     Ok(())
/// })?;
/// println!("Total: {} triples", total);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn process_rdf_file_in_batches<P, F>(
    path: P,
    batch_size: usize,
    mut callback: F,
) -> TurtleResult<()>
where
    P: AsRef<Path>,
    F: FnMut(&[Triple]) -> TurtleResult<()>,
{
    let file = File::open(path).map_err(TurtleParseError::io)?;
    let config = StreamingConfig::default().with_batch_size(batch_size);
    let parser = StreamingParser::with_config(file, config);

    for batch_result in parser.batches() {
        let batch = batch_result?;
        callback(&batch)?;
    }

    Ok(())
}

/// Statistics collected during parsing
#[derive(Debug, Clone, Default)]
pub struct ParsingStatistics {
    /// Total number of triples/quads parsed
    pub total_items: usize,
    /// Number of batches processed
    pub batches_processed: usize,
    /// Total bytes read
    pub bytes_read: usize,
    /// Number of errors encountered (in lenient mode)
    pub errors: usize,
}

impl ParsingStatistics {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the average batch size
    pub fn avg_batch_size(&self) -> f64 {
        if self.batches_processed == 0 {
            0.0
        } else {
            self.total_items as f64 / self.batches_processed as f64
        }
    }

    /// Get formatted statistics report
    pub fn report(&self) -> String {
        format!(
            "Parsing Statistics:\n\
             - Total items: {}\n\
             - Batches: {}\n\
             - Avg batch size: {:.1}\n\
             - Bytes read: {}\n\
             - Errors: {}",
            self.total_items,
            self.batches_processed,
            self.avg_batch_size(),
            self.bytes_read,
            self.errors
        )
    }
}

/// Process a large RDF file in batches with statistics tracking
///
/// Similar to `process_rdf_file_in_batches` but also collects parsing statistics.
///
/// # Example
///
/// ```rust,no_run
/// use oxirs_ttl::convenience::process_rdf_file_with_stats;
///
/// let stats = process_rdf_file_with_stats("large_data.ttl", 10000, |batch| {
///     // Process batch
///     Ok(())
/// })?;
///
/// println!("{}", stats.report());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn process_rdf_file_with_stats<P, F>(
    path: P,
    batch_size: usize,
    mut callback: F,
) -> TurtleResult<ParsingStatistics>
where
    P: AsRef<Path>,
    F: FnMut(&[Triple]) -> TurtleResult<()>,
{
    let file = File::open(path).map_err(TurtleParseError::io)?;
    let config = StreamingConfig::default().with_batch_size(batch_size);
    let parser = StreamingParser::with_config(file, config);

    let mut stats = ParsingStatistics::new();

    // Note: batches() consumes the parser, so we can't read bytes_read after.
    // This is a limitation of the current API design.
    for batch_result in parser.batches() {
        let batch = batch_result?;
        stats.total_items += batch.len();
        stats.batches_processed += 1;

        callback(&batch)?;
    }

    // Cannot read bytes_read after batches() as it consumes the parser
    // stats.bytes_read would need to be tracked differently

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_turtle_file() -> TurtleResult<()> {
        let mut file = NamedTempFile::new().map_err(TurtleParseError::io)?;
        writeln!(
            file,
            "@prefix ex: <http://example.org/> .\nex:s ex:p ex:o ."
        )
        .map_err(TurtleParseError::io)?;

        let triples = parse_turtle_file(file.path())?;
        assert_eq!(triples.len(), 1);
        Ok(())
    }

    #[test]
    fn test_parse_ntriples_file() -> TurtleResult<()> {
        let mut file = NamedTempFile::new().map_err(TurtleParseError::io)?;
        writeln!(file, "<http://s> <http://p> <http://o> .").map_err(TurtleParseError::io)?;

        let triples = parse_ntriples_file(file.path())?;
        assert_eq!(triples.len(), 1);
        Ok(())
    }

    #[test]
    fn test_process_in_batches() -> TurtleResult<()> {
        let mut file = NamedTempFile::new().map_err(TurtleParseError::io)?;
        for i in 0..100 {
            writeln!(file, "<http://s{}> <http://p> <http://o{}> .", i, i)
                .map_err(TurtleParseError::io)?;
        }

        let mut total = 0;
        let mut batch_count = 0;
        process_rdf_file_in_batches(file.path(), 10, |batch| {
            total += batch.len();
            batch_count += 1;
            assert!(batch.len() <= 10);
            Ok(())
        })?;

        assert_eq!(total, 100);
        assert!(batch_count >= 10); // At least 10 batches for 100 triples
        Ok(())
    }

    #[test]
    fn test_parsing_statistics() -> TurtleResult<()> {
        let mut file = NamedTempFile::new().map_err(TurtleParseError::io)?;
        for i in 0..50 {
            writeln!(file, "<http://s{}> <http://p> <http://o{}> .", i, i)
                .map_err(TurtleParseError::io)?;
        }

        let stats = process_rdf_file_with_stats(file.path(), 10, |_batch| Ok(()))?;

        assert_eq!(stats.total_items, 50);
        assert!(stats.batches_processed >= 5);
        assert!(stats.avg_batch_size() > 0.0);
        // Note: bytes_read is not tracked in current implementation
        // due to API limitations

        let report = stats.report();
        assert!(report.contains("Total items: 50"));
        Ok(())
    }
}
