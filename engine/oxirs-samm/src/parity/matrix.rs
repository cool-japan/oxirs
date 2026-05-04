//! ESMF SDK 2.x parity matrix data types and TOML catalog parser.
//!
//! The [`ParityMatrix`] is a map from [`FeatureCategory`] to a list of
//! [`FeatureEntry`] records, each representing one documented ESMF SDK 2.x
//! feature and its current implementation status in `oxirs-samm`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ImplStatus {
    /// The feature is fully implemented and tested.
    Implemented,
    /// The feature is partially implemented; known gaps exist.
    Partial,
    /// The feature is not yet implemented.
    Missing,
}

/// One row in the parity matrix, describing a single ESMF SDK 2.x feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEntry {
    /// Human-readable feature name.
    pub name: String,
    /// Current implementation status in `oxirs-samm`.
    pub status: ImplStatus,
    /// Rust module path inside this crate that implements the feature (if any).
    pub oxirs_module: Option<String>,
    /// Canonical URL into the ESMF / SAMM 2.x specification.
    pub esmf_reference: String,
    /// One-sentence clarification or scope note.
    pub notes: String,
}

/// The full parity matrix: a mapping from feature category to feature entries.
///
/// Populate with [`parse_catalog`] or [`super::load_catalog`].
pub type ParityMatrix = HashMap<FeatureCategory, Vec<FeatureEntry>>;

/// Parse a TOML catalog string into a [`ParityMatrix`].
///
/// The TOML structure uses snake_case section names matching
/// [`FeatureCategory`] variants (e.g. `[[aspect_modeling]]`).
///
/// # Errors
///
/// Returns an error if the TOML is malformed or contains unexpected types.
pub fn parse_catalog(toml_str: &str) -> Result<ParityMatrix, Box<dyn std::error::Error>> {
    // ── Internal deserialization helpers ────────────────────────────────

    #[derive(Deserialize)]
    struct RawEntry {
        name: String,
        status: String,
        oxirs_module: Option<String>,
        esmf_reference: String,
        notes: String,
    }

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

    let file: CatalogFile = toml::from_str(toml_str)?;
    let mut matrix = ParityMatrix::new();

    fn parse_status(s: &str) -> ImplStatus {
        match s {
            "implemented" => ImplStatus::Implemented,
            "partial" => ImplStatus::Partial,
            _ => ImplStatus::Missing,
        }
    }

    fn convert(raw: Vec<RawEntry>) -> Vec<FeatureEntry> {
        raw.into_iter()
            .map(|r| FeatureEntry {
                name: r.name,
                status: parse_status(&r.status),
                oxirs_module: r.oxirs_module,
                esmf_reference: r.esmf_reference,
                notes: r.notes,
            })
            .collect()
    }

    if let Some(entries) = file.aspect_modeling {
        matrix.insert(FeatureCategory::AspectModeling, convert(entries));
    }
    if let Some(entries) = file.validation {
        matrix.insert(FeatureCategory::Validation, convert(entries));
    }
    if let Some(entries) = file.code_generation {
        matrix.insert(FeatureCategory::CodeGeneration, convert(entries));
    }
    if let Some(entries) = file.open_api_emission {
        matrix.insert(FeatureCategory::OpenApiEmission, convert(entries));
    }
    if let Some(entries) = file.json_ld_profiles {
        matrix.insert(FeatureCategory::JsonLdProfiles, convert(entries));
    }
    if let Some(entries) = file.model_resolution {
        matrix.insert(FeatureCategory::ModelResolution, convert(entries));
    }
    if let Some(entries) = file.command_line_tooling {
        matrix.insert(FeatureCategory::CommandLineTooling, convert(entries));
    }

    Ok(matrix)
}

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

[[validation]]
name = "Missing feature"
status = "missing"
esmf_reference = "https://example.com/missing"
notes = "Not yet done."
"#;

    #[test]
    fn test_parse_minimal_catalog() {
        let matrix = parse_catalog(MINIMAL_TOML).expect("should parse");
        assert_eq!(matrix.len(), 2);
        let aspect = matrix
            .get(&FeatureCategory::AspectModeling)
            .expect("aspect modeling");
        assert_eq!(aspect.len(), 1);
        assert_eq!(aspect[0].status, ImplStatus::Implemented);
        assert!(aspect[0].oxirs_module.is_some());
    }

    #[test]
    fn test_parse_status_variants() {
        let matrix = parse_catalog(MINIMAL_TOML).expect("should parse");
        let validation = matrix
            .get(&FeatureCategory::Validation)
            .expect("validation");
        assert_eq!(validation[0].status, ImplStatus::Missing);
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
        let matrix = parse_catalog(toml).expect("should parse");
        let entries = matrix
            .get(&FeatureCategory::AspectModeling)
            .expect("aspect");
        assert_eq!(entries[0].status, ImplStatus::Missing);
    }

    #[test]
    fn test_invalid_toml_returns_error() {
        let result = parse_catalog("this is not valid toml ][}{");
        assert!(result.is_err());
    }
}
