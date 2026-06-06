//! CSV on the Web (CSVW) parser and RDF converter.
//!
//! Implements a subset of the W3C CSVW Recommendation:
//! <https://www.w3.org/TR/tabular-data-primer/>
//!
//! # Overview
//!
//! A CSVW document pairs a tabular CSV file with a JSON-LD metadata file that
//! annotates the columns with datatypes, predicate IRIs and URI templates.
//! This module provides:
//!
//! - **[`schema`]** — typed deserialization of the JSON metadata.
//! - **[`reader`]** — a pure-Rust RFC 4180 CSV reader (no external `csv` crate).
//! - **[`converter`]** — maps parsed CSV rows to [`RdfStatement`] triples.
//!
//! # Quick example
//!
//! ```rust
//! use oxirs_ttl::csvw::{parse_csv, CsvwMetadata, CsvwConverter, CsvwConverterConfig};
//!
//! let csv = "id,name\n1,Alice\n2,Bob\n";
//! let meta_json = r#"{"tableSchema":{"columns":[{"name":"id"},{"name":"name"}]}}"#;
//!
//! let (headers, records) = parse_csv(csv)?;
//! let meta = CsvwMetadata::from_json(meta_json)?;
//! let converter = CsvwConverter::new(CsvwConverterConfig::default());
//! let stmts = converter.convert(&headers, &records, &meta)?;
//! println!("Generated {} RDF statements", stmts.len());
//! # Ok::<(), oxirs_ttl::csvw::CsvwError>(())
//! ```

pub mod converter;
pub mod reader;
pub mod schema;

mod tests;

pub use converter::{CsvwConverter, CsvwConverterConfig, RdfStatement};
pub use reader::{parse_csv, CsvReader, CsvRecord};
pub use schema::{ColumnDef, CsvwMetadata, TableSchema};

// ────────────────────────────────────────────────────────────────────────────
// Module-level error type
// ────────────────────────────────────────────────────────────────────────────

/// All errors that can be produced by the CSVW parser and converter.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CsvwError {
    /// A JSON metadata file could not be parsed.
    #[error("JSON parse error: {0}")]
    JsonError(String),

    /// A CSV file could not be parsed (includes the 1-based line number).
    #[error("CSV parse error at line {line}: {msg}")]
    CsvError {
        /// 1-based line number where the error occurred.
        line: usize,
        /// Human-readable description.
        msg: String,
    },

    /// The CSVW schema is logically inconsistent.
    #[error("Schema error: {0}")]
    SchemaError(String),

    /// A CSV row could not be converted to RDF.
    #[error("Conversion error: {0}")]
    ConversionError(String),

    /// An underlying I/O operation failed.
    #[error("IO error: {0}")]
    IoError(String),
}

impl From<serde_json::Error> for CsvwError {
    fn from(e: serde_json::Error) -> Self {
        CsvwError::JsonError(e.to_string())
    }
}

impl From<std::io::Error> for CsvwError {
    fn from(e: std::io::Error) -> Self {
        CsvwError::IoError(e.to_string())
    }
}
