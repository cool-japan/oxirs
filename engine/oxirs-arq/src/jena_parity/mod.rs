//! Apache Jena feature parity tracking for `oxirs-arq`.
//!
//! This module provides a structured catalogue of every documented Apache Jena
//! feature with its current implementation status in OxiRS, along with a
//! markdown report generator. OxiRS is a JVM-free Jena/Fuseki alternative
//! written in pure Rust.
//!
//! # Quick start
//!
//! ```rust
//! use oxirs_arq::jena_parity::{load_catalog, generate_jena_report};
//!
//! let matrix = load_catalog().expect("catalog should parse");
//! let report = generate_jena_report(&matrix);
//! // `report` is a markdown string summarising parity status across all categories
//! # let _ = report;
//! ```

pub mod matrix;
pub mod report;

pub use matrix::{JenaCategory, JenaEntry, JenaParityMatrix, JenaStatus};
pub use report::generate_jena_report;

/// Load the hand-curated Apache Jena feature catalog embedded at compile time
/// from `src/jena_parity/jena_catalog.toml`.
///
/// # Errors
///
/// Returns an error if the embedded TOML is malformed (should never happen
/// unless the source file is corrupted).
pub fn load_catalog() -> Result<JenaParityMatrix, Box<dyn std::error::Error>> {
    let toml_str = include_str!("jena_catalog.toml");
    matrix::parse_catalog(toml_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_catalog_succeeds() {
        let matrix = load_catalog().expect("embedded catalog should parse without error");
        assert!(
            !matrix.is_empty(),
            "catalog must have at least one category"
        );
    }

    #[test]
    fn test_generate_report_from_catalog() {
        let matrix = load_catalog().expect("catalog");
        let report = generate_jena_report(&matrix);
        assert!(report.len() > 100, "report should be non-trivial");
    }
}
