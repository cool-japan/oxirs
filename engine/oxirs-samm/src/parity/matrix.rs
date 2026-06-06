//! ESMF SDK 2.x parity matrix data types and TOML catalog parser.
//!
//! The [`ParityMatrix`] is a structured catalogue mapping [`FeatureCategory`]
//! to a list of [`FeatureEntry`] records, each representing one documented
//! ESMF SDK 2.x feature and its current implementation status in `oxirs-samm`.
//!
//! # Building a matrix
//!
//! ```rust
//! use oxirs_samm::parity::{ParityMatrix, FeatureCategory, FeatureEntry, FeatureStatus};
//!
//! let mut matrix = ParityMatrix::new();
//! let entry = FeatureEntry {
//!     name: "Aspect definition".to_string(),
//!     description: "Core Aspect model".to_string(),
//!     status: FeatureStatus::Done,
//!     oxirs_module: Some("oxirs_samm::metamodel::aspect".to_string()),
//!     notes: None,
//! };
//! matrix.add_entry(FeatureCategory::AspectModeling, entry).expect("unique name");
//! assert!(matrix.coverage_percent(&FeatureCategory::AspectModeling) > 0.0);
//! ```

use crate::error::SammError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// High-level grouping that mirrors ESMF SDK 2.x documentation chapters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FeatureCategory {
    /// Features related to Aspect, Property, Characteristic, Entity, Operation, Event.
    AspectModeling,
    /// Model validation — syntax, SHACL shape conformance, semantic checks.
    Validation,
    /// Source-code generation (Java, TypeScript, Python, Scala, SQL, …).
    CodeGeneration,
    /// OpenAPI 3.x specification emission from aspect models.
    OpenApiEmission,
    /// JSON-LD context and profile support.
    JsonLdProfiles,
    /// URN / file / HTTP model resolution and loading.
    ModelResolution,
    /// CLI sub-commands (validate, generate, aspect, …).
    CommandLineTooling,
    /// Any feature category that does not fit the named variants.
    Other(String),
}

/// Implementation status of a single ESMF SDK feature in `oxirs-samm`.
///
/// `FeatureStatus` is the canonical name; `ImplStatus` is kept as an alias for
/// backwards compatibility.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FeatureStatus {
    /// The feature is fully implemented and tested.
    Done,
    /// The feature is partially implemented; known gaps exist.
    Partial,
    /// The feature is not yet implemented.
    Missing,
}

/// Backwards-compatible alias for [`FeatureStatus`].
///
/// The pre-struct API used `ImplStatus` with variant `Implemented`.
/// Migration: replace `ImplStatus::Implemented` → `FeatureStatus::Done`.
pub type ImplStatus = FeatureStatus;

impl FeatureStatus {
    /// Returns `true` if this status counts as "done" (fully implemented).
    #[inline]
    pub fn is_done(&self) -> bool {
        matches!(self, FeatureStatus::Done)
    }

    /// Returns `true` if this status contributes fractionally to coverage.
    #[inline]
    pub fn is_partial(&self) -> bool {
        matches!(self, FeatureStatus::Partial)
    }

    /// Returns `true` if this feature is not yet implemented.
    #[inline]
    pub fn is_missing(&self) -> bool {
        matches!(self, FeatureStatus::Missing)
    }
}

/// One row in the parity matrix, describing a single ESMF SDK 2.x feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEntry {
    /// Human-readable feature name (must be unique within its category).
    pub name: String,
    /// One-sentence clarification or scope note describing what is covered.
    pub description: String,
    /// Current implementation status in `oxirs-samm`.
    pub status: FeatureStatus,
    /// Rust module path inside this crate that implements the feature (if any).
    /// e.g. `"oxirs_samm::aspect"`.
    pub oxirs_module: Option<String>,
    /// Additional notes — gaps, caveats, or ESMF reference URLs.
    pub notes: Option<String>,
}

/// The full parity matrix: a map from [`FeatureCategory`] to ordered feature entries.
///
/// Build via [`ParityMatrix::new`], [`ParityMatrix::from_toml`], or
/// [`super::load_catalog`].
///
/// # Coverage semantics
///
/// `Done` counts as 1.0, `Partial` counts as 0.5, `Missing` counts as 0.0
/// towards the coverage fraction.
#[derive(Debug, Default)]
pub struct ParityMatrix {
    entries: HashMap<FeatureCategory, Vec<FeatureEntry>>,
}

impl ParityMatrix {
    // ── Constructors ────────────────────────────────────────────────────────

    /// Create an empty parity matrix.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Parse a TOML catalog string into a [`ParityMatrix`].
    ///
    /// The TOML must use `snake_case` table-array names matching
    /// [`FeatureCategory`] variants (e.g. `[[aspect_modeling]]`).
    ///
    /// # Errors
    ///
    /// Returns `SammError::ParseError` if the TOML is malformed or contains
    /// unexpected types.
    pub fn from_toml(content: &str) -> Result<Self, SammError> {
        parse_catalog_inner(content)
    }

    // ── Mutation ────────────────────────────────────────────────────────────

    /// Add a [`FeatureEntry`] to the given category.
    ///
    /// # Errors
    ///
    /// Returns `SammError::ValidationError` if an entry with the same `name`
    /// already exists in that category (deduplication by name).
    pub fn add_entry(
        &mut self,
        cat: FeatureCategory,
        entry: FeatureEntry,
    ) -> Result<(), SammError> {
        let bucket = self.entries.entry(cat).or_default();
        if bucket.iter().any(|e| e.name == entry.name) {
            return Err(SammError::ValidationError(format!(
                "parity entry '{}' already exists in this category",
                entry.name
            )));
        }
        bucket.push(entry);
        Ok(())
    }

    // ── Queries ─────────────────────────────────────────────────────────────

    /// Return the entries for the given category, or `None` if the category
    /// has no entries.
    pub fn get(&self, cat: &FeatureCategory) -> Option<&[FeatureEntry]> {
        self.entries.get(cat).map(|v| v.as_slice())
    }

    /// Return `true` if the matrix has no categories.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Coverage percentage for a single category.
    ///
    /// `Done` contributes 1.0, `Partial` contributes 0.5, `Missing`
    /// contributes 0.0.  Returns `0.0` for an unknown or empty category.
    pub fn coverage_percent(&self, cat: &FeatureCategory) -> f64 {
        let entries = match self.entries.get(cat) {
            Some(v) if !v.is_empty() => v,
            _ => return 0.0,
        };
        let score: f64 = entries
            .iter()
            .map(|e| match e.status {
                FeatureStatus::Done => 1.0,
                FeatureStatus::Partial => 0.5,
                FeatureStatus::Missing => 0.0,
            })
            .sum();
        score / entries.len() as f64 * 100.0
    }

    /// Overall coverage percentage across all categories.
    ///
    /// Returns `0.0` for an empty matrix.
    pub fn overall_coverage(&self) -> f64 {
        let all_entries: Vec<&FeatureEntry> = self.entries.values().flatten().collect();
        if all_entries.is_empty() {
            return 0.0;
        }
        let score: f64 = all_entries
            .iter()
            .map(|e| match e.status {
                FeatureStatus::Done => 1.0,
                FeatureStatus::Partial => 0.5,
                FeatureStatus::Missing => 0.0,
            })
            .sum();
        score / all_entries.len() as f64 * 100.0
    }

    /// Collect all entries whose status is [`FeatureStatus::Missing`].
    ///
    /// The returned slice is ordered by category insertion order, then by
    /// entry position within each category.
    pub fn missing_entries(&self) -> Vec<(&FeatureCategory, &FeatureEntry)> {
        let mut result = Vec::new();
        for (cat, entries) in &self.entries {
            for entry in entries {
                if entry.status == FeatureStatus::Missing {
                    result.push((cat, entry));
                }
            }
        }
        result
    }

    /// Return the first `n` missing entries across all categories.
    ///
    /// If there are fewer than `n` missing entries, returns all of them.
    /// Order follows [`missing_entries`](Self::missing_entries).
    pub fn top_missing(&self, n: usize) -> Vec<(&FeatureCategory, &FeatureEntry)> {
        self.missing_entries().into_iter().take(n).collect()
    }

    /// Iterate over all (category, entries) pairs (exposes internal map for
    /// report generation).
    pub fn iter(&self) -> impl Iterator<Item = (&FeatureCategory, &[FeatureEntry])> {
        self.entries.iter().map(|(k, v)| (k, v.as_slice()))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal TOML parser helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Internal raw entry for TOML deserialization.
#[derive(Deserialize)]
struct RawEntry {
    name: String,
    /// Optional description field; falls back to the `notes` field if absent.
    description: Option<String>,
    status: String,
    oxirs_module: Option<String>,
    /// ESMF canonical URL — stored in `notes` if no separate description exists.
    esmf_reference: Option<String>,
    notes: Option<String>,
}

/// Internal TOML catalog file structure.
#[derive(Deserialize)]
struct CatalogFile {
    aspect_modeling: Option<Vec<RawEntry>>,
    validation: Option<Vec<RawEntry>>,
    code_generation: Option<Vec<RawEntry>>,
    open_api_emission: Option<Vec<RawEntry>>,
    json_ld_profiles: Option<Vec<RawEntry>>,
    model_resolution: Option<Vec<RawEntry>>,
    command_line_tooling: Option<Vec<RawEntry>>,
}

fn parse_status(s: &str) -> FeatureStatus {
    match s {
        "implemented" | "done" | "Done" | "Implemented" => FeatureStatus::Done,
        "partial" | "Partial" => FeatureStatus::Partial,
        _ => FeatureStatus::Missing,
    }
}

fn convert_entries(raw: Vec<RawEntry>) -> Vec<FeatureEntry> {
    raw.into_iter()
        .map(|r| {
            // Build a useful description: prefer explicit `description`, then
            // fall back to the `esmf_reference` URL so that the field is never empty.
            let description = r
                .description
                .or_else(|| r.esmf_reference.clone())
                .unwrap_or_else(|| r.name.clone());

            // Merge esmf_reference into notes if present
            let notes = match (r.notes, r.esmf_reference) {
                (Some(n), Some(url)) => Some(format!("{n} Reference: {url}")),
                (Some(n), None) => Some(n),
                (None, Some(url)) => Some(url),
                (None, None) => None,
            };

            FeatureEntry {
                name: r.name,
                description,
                status: parse_status(&r.status),
                oxirs_module: r.oxirs_module,
                notes,
            }
        })
        .collect()
}

fn parse_catalog_inner(toml_str: &str) -> Result<ParityMatrix, SammError> {
    let file: CatalogFile = toml::from_str(toml_str)
        .map_err(|e| SammError::ParseError(format!("catalog TOML parse error: {e}")))?;

    let mut matrix = ParityMatrix::new();

    macro_rules! insert_cat {
        ($field:expr, $variant:expr) => {
            if let Some(entries) = $field {
                matrix.entries.insert($variant, convert_entries(entries));
            }
        };
    }

    insert_cat!(file.aspect_modeling, FeatureCategory::AspectModeling);
    insert_cat!(file.validation, FeatureCategory::Validation);
    insert_cat!(file.code_generation, FeatureCategory::CodeGeneration);
    insert_cat!(file.open_api_emission, FeatureCategory::OpenApiEmission);
    insert_cat!(file.json_ld_profiles, FeatureCategory::JsonLdProfiles);
    insert_cat!(file.model_resolution, FeatureCategory::ModelResolution);
    insert_cat!(
        file.command_line_tooling,
        FeatureCategory::CommandLineTooling
    );

    Ok(matrix)
}

/// Parse a TOML catalog string into a [`ParityMatrix`] (free function for
/// backwards compatibility with pre-struct API).
///
/// Prefer [`ParityMatrix::from_toml`] for new code.
///
/// # Errors
///
/// Returns `Box<dyn Error>` if TOML is malformed.
pub fn parse_catalog(toml_str: &str) -> Result<ParityMatrix, Box<dyn std::error::Error>> {
    Ok(parse_catalog_inner(toml_str)?)
}

// ──────────────────────────────────────────────────────────────────────────────
// Unit tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_TOML: &str = r#"
[[aspect_modeling]]
name = "Aspect definition"
status = "implemented"
oxirs_module = "metamodel::aspect"
esmf_reference = "https://example.com/aspect"
notes = "Core aspect."

[[aspect_modeling]]
name = "Entity"
status = "partial"
esmf_reference = "https://example.com/entity"
notes = "Partial."

[[aspect_modeling]]
name = "Either characteristic"
status = "missing"
esmf_reference = "https://example.com/either"
notes = "Not yet done."

[[validation]]
name = "SHACL validation"
status = "missing"
esmf_reference = "https://example.com/missing"
notes = "Not yet done."
"#;

    #[test]
    fn test_parse_minimal_catalog() {
        let matrix = ParityMatrix::from_toml(MINIMAL_TOML).expect("should parse");
        let aspect = matrix
            .get(&FeatureCategory::AspectModeling)
            .expect("aspect modeling");
        assert_eq!(aspect.len(), 3);
        assert_eq!(aspect[0].status, FeatureStatus::Done);
        assert!(aspect[0].oxirs_module.is_some());
    }

    #[test]
    fn test_parse_status_variants() {
        let matrix = ParityMatrix::from_toml(MINIMAL_TOML).expect("should parse");
        let validation = matrix
            .get(&FeatureCategory::Validation)
            .expect("validation");
        assert_eq!(validation[0].status, FeatureStatus::Missing);
    }

    #[test]
    fn test_unknown_status_becomes_missing() {
        let toml = r#"
[[aspect_modeling]]
name = "X"
status = "unknown_value"
esmf_reference = "https://example.com"
notes = "test"
"#;
        let matrix = ParityMatrix::from_toml(toml).expect("should parse");
        let entries = matrix
            .get(&FeatureCategory::AspectModeling)
            .expect("aspect");
        assert_eq!(entries[0].status, FeatureStatus::Missing);
    }

    #[test]
    fn test_invalid_toml_returns_error() {
        let result = ParityMatrix::from_toml("this is not valid toml ][}{");
        assert!(result.is_err());
    }

    #[test]
    fn test_coverage_percent_mixed() {
        // 1 Done (1.0), 1 Partial (0.5), 1 Missing (0.0) = 1.5 / 3 * 100 = 50.0
        let matrix = ParityMatrix::from_toml(MINIMAL_TOML).expect("should parse");
        let pct = matrix.coverage_percent(&FeatureCategory::AspectModeling);
        let expected = (1.0_f64 + 0.5 + 0.0) / 3.0 * 100.0;
        assert!(
            (pct - expected).abs() < 0.01,
            "expected {expected:.2} got {pct:.2}"
        );
    }

    #[test]
    fn test_coverage_percent_empty_category() {
        let matrix = ParityMatrix::new();
        assert_eq!(
            matrix.coverage_percent(&FeatureCategory::Validation),
            0.0,
            "empty category should have 0% coverage"
        );
    }

    #[test]
    fn test_overall_coverage_empty_matrix() {
        let matrix = ParityMatrix::new();
        assert_eq!(matrix.overall_coverage(), 0.0);
    }

    #[test]
    fn test_add_entry_dedup() {
        let mut matrix = ParityMatrix::new();
        let entry = FeatureEntry {
            name: "Aspect definition".to_string(),
            description: "Core Aspect".to_string(),
            status: FeatureStatus::Done,
            oxirs_module: Some("metamodel::aspect".to_string()),
            notes: None,
        };
        let entry2 = FeatureEntry {
            name: "Aspect definition".to_string(), // duplicate name
            description: "Duplicate".to_string(),
            status: FeatureStatus::Missing,
            oxirs_module: None,
            notes: None,
        };
        matrix
            .add_entry(FeatureCategory::AspectModeling, entry)
            .expect("first insert should succeed");
        let result = matrix.add_entry(FeatureCategory::AspectModeling, entry2);
        assert!(result.is_err(), "duplicate name should return error");
    }

    #[test]
    fn test_missing_entries_filter() {
        let matrix = ParityMatrix::from_toml(MINIMAL_TOML).expect("should parse");
        let missing = matrix.missing_entries();
        assert!(
            missing
                .iter()
                .all(|(_, e)| e.status == FeatureStatus::Missing),
            "missing_entries must only contain Missing status"
        );
        assert!(!missing.is_empty(), "should have some missing entries");
    }

    #[test]
    fn test_top_missing_bounded() {
        let matrix = ParityMatrix::from_toml(MINIMAL_TOML).expect("should parse");
        let top3 = matrix.top_missing(3);
        assert!(top3.len() <= 3, "top_missing(3) must return at most 3");
    }

    #[test]
    fn test_get_unknown_category() {
        let matrix = ParityMatrix::new();
        assert!(
            matrix.get(&FeatureCategory::Validation).is_none(),
            "unknown category must return None"
        );
    }

    #[test]
    fn test_is_empty() {
        let matrix = ParityMatrix::new();
        assert!(matrix.is_empty());
        let matrix2 = ParityMatrix::from_toml(MINIMAL_TOML).expect("should parse");
        assert!(!matrix2.is_empty());
    }

    #[test]
    fn test_parse_catalog_free_fn_compat() {
        // backwards-compat free function
        let matrix = parse_catalog(MINIMAL_TOML).expect("should parse");
        assert!(!matrix.is_empty());
    }
}
