//! Model manifest: TOML-deserializable metadata for zoo entries.

use serde::{Deserialize, Serialize};

/// Metadata describing a single pretrained model in the model zoo.
///
/// Each entry is stored as a `.toml` file (either shipped with the crate or
/// loaded from a user-supplied directory at runtime).  Entries that ship as
/// part of the crate describe *synthetic seed checkpoints* — small,
/// randomly-initialised weights for testing — **not** actual research weights.
/// The `notes` field always documents this distinction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Human-readable registry key (must be unique within a zoo).
    pub name: String,

    /// Model architecture type.
    ///
    /// Must match one of the strings that `persistence::ModelRepository`
    /// dispatches on: `TransE`, `DistMult`, `ComplEx`, `RotatE`, `HoLE`,
    /// `GNNEmbedding`.
    pub model_type: String,

    /// Training dataset identifier (e.g. `"FB15k-237"`, `"WN18RR"`).
    pub dataset: String,

    /// Embedding dimensionality.
    pub dimensions: usize,

    /// Number of entities in the vocabulary.
    pub entities: usize,

    /// Number of relation types.
    pub relations: usize,

    /// Hex-encoded SHA-256 digest of the checkpoint file at `source`.
    ///
    /// The loader verifies this before deserialising.  Use
    /// `"PLACEHOLDER"` for catalog entries whose checkpoint does not ship
    /// inside the crate.
    pub sha256: String,

    /// Location of the checkpoint.
    ///
    /// Supported schemes:
    /// - `file:///absolute/path/to/file.ckpt` — resolved by the loader
    ///   relative to `base_dir` when the `file:///` root does not exist
    ///   literally on the filesystem.
    /// - `https://...` — requires the optional `download` feature (OFF by
    ///   default).
    pub source: String,

    /// SPDX license expression (e.g. `"Apache-2.0"`, `"CC-BY-4.0"`).
    pub license: String,

    /// Bibliographic citation for the model's origin paper.
    pub citation: String,

    /// Semantic version of this manifest record.
    pub version: String,

    /// ISO 8601 creation date of this manifest entry.
    pub created: String,

    /// Optional free-text notes (provenance warnings, dataset caveats, etc.).
    pub notes: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest() -> ModelManifest {
        ModelManifest {
            name: "test-model".to_string(),
            model_type: "TransE".to_string(),
            dataset: "FB15k-237".to_string(),
            dimensions: 200,
            entities: 14541,
            relations: 237,
            sha256: "abc123def456".to_string(),
            source: "file:///seeds/test.ckpt".to_string(),
            license: "Apache-2.0".to_string(),
            citation: "Test Citation".to_string(),
            version: "1.0.0".to_string(),
            created: "2026-05-01".to_string(),
            notes: Some("Test notes".to_string()),
        }
    }

    #[test]
    fn test_manifest_serialize_deserialize_toml() {
        let m = sample_manifest();
        let toml_str = toml::to_string(&m).expect("serialize to TOML");
        let back: ModelManifest = toml::from_str(&toml_str).expect("deserialize from TOML");
        assert_eq!(m, back);
    }

    #[test]
    fn test_manifest_serialize_deserialize_json() {
        let m = sample_manifest();
        let json_str = serde_json::to_string(&m).expect("serialize to JSON");
        let back: ModelManifest = serde_json::from_str(&json_str).expect("deserialize from JSON");
        assert_eq!(m, back);
    }

    #[test]
    fn test_manifest_notes_optional() {
        let toml_str = r#"
name = "minimal"
model_type = "DistMult"
dataset = "WN18RR"
dimensions = 100
entities = 40943
relations = 11
sha256 = "deadbeef"
source = "file:///seeds/minimal.ckpt"
license = "MIT"
citation = "Nobody 2026"
version = "0.1.0"
created = "2026-05-01"
"#;
        let m: ModelManifest = toml::from_str(toml_str).expect("parse without notes");
        assert!(m.notes.is_none());
        assert_eq!(m.model_type, "DistMult");
    }
}
