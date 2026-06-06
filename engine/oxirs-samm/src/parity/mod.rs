//! ESMF SDK 2.x feature parity tracking for `oxirs-samm`.
//!
//! This module provides a structured catalogue of every documented ESMF SDK 2.x
//! feature with its current implementation status in `oxirs-samm`, along with a
//! markdown report generator.
//!
//! # Quick start
//!
//! ```rust
//! use oxirs_samm::parity::{load_catalog, generate_report};
//!
//! let matrix = load_catalog().expect("catalog should parse");
//! let report = generate_report(&matrix);
//! // `report` is a markdown string ready for writing to docs/esmf_parity.md
//! # let _ = report;
//! ```

pub mod matrix;
pub mod report;

pub use matrix::{FeatureCategory, FeatureEntry, FeatureStatus, ImplStatus, ParityMatrix};
pub use report::generate_report;

/// Load the hand-curated ESMF SDK 2.x feature catalog that is embedded at
/// compile time from `src/parity/esmf_catalog.toml`.
///
/// # Errors
///
/// Returns an error if the embedded TOML is malformed (should never happen
/// unless the source file is corrupted).
pub fn load_catalog() -> Result<ParityMatrix, Box<dyn std::error::Error>> {
    let toml_str = include_str!("esmf_catalog.toml");
    Ok(matrix::ParityMatrix::from_toml(toml_str)?)
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
        let report = generate_report(&matrix);
        assert!(report.len() > 100, "report should be non-trivial");
    }

    #[test]
    fn test_overall_coverage_positive() {
        let matrix = load_catalog().expect("catalog");
        let cov = matrix.overall_coverage();
        assert!(cov > 0.0, "coverage should be positive");
        assert!(cov <= 100.0, "coverage must be at most 100%");
    }
}
