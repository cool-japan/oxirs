//! Manifest descriptor for shape-learning model checkpoints.
//!
//! A [`ShapeModelManifest`] is a TOML document that describes a pretrained
//! SHACL shape-learning model: its architecture hyper-parameters, the dataset it
//! was trained on, the location of the weight file, and a SHA-256 checksum used
//! to verify checkpoint integrity before loading.

use serde::{Deserialize, Serialize};

/// Manifest descriptor for a pretrained SHACL shape-learning model checkpoint.
///
/// Serialises to / deserialises from TOML.  The fields deliberately mirror the
/// layout used in the bundled `*.toml` manifests under `src/model_zoo/manifests/`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ShapeModelManifest {
    /// Unique registry name, e.g. `"gat-shacl-base"`.
    pub name: String,

    /// Architecture family — one of `"GAT"`, `"GraphSAGE"`, `"Graphormer"`,
    /// or `"GraphTransformer"`.
    pub model_type: String,

    /// Dataset used during training, e.g. `"LUBM-SHACL-synthetic"`.
    pub dataset: String,

    /// Dimensionality of input node features.
    pub input_dim: usize,

    /// Width of hidden layers / attention embedding dimension.
    pub hidden_dim: usize,

    /// Dimensionality of the output (number of constraint classes).
    pub output_dim: usize,

    /// Number of attention heads.
    pub num_heads: usize,

    /// Number of transformer / message-passing layers.
    pub num_layers: usize,

    /// Lowercase hex SHA-256 of the weight file byte content.
    pub sha256: String,

    /// Location of the weight file.  Only `file://` scheme is supported in
    /// default features.  The path after stripping the `file://` prefix is
    /// joined to the loader's `base_dir` if it is relative.
    pub source: String,

    /// SPDX license identifier (e.g. `"Apache-2.0"`).
    pub license: String,

    /// Free-text citation / attribution for the model.
    pub citation: String,

    /// Semantic version of this checkpoint, e.g. `"1.0.0"`.
    pub version: String,

    /// ISO-8601 creation date (`"YYYY-MM-DD"`).
    pub created: String,

    /// Optional human-readable notes.
    pub notes: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Confirms that round-tripping a manifest through TOML is lossless.
    #[test]
    fn test_manifest_toml_roundtrip() {
        let original = ShapeModelManifest {
            name: "test-model".to_string(),
            model_type: "GAT".to_string(),
            dataset: "test-dataset".to_string(),
            input_dim: 8,
            hidden_dim: 32,
            output_dim: 4,
            num_heads: 2,
            num_layers: 2,
            sha256: "deadbeef".to_string(),
            source: "file:///tmp/test.ckpt".to_string(),
            license: "Apache-2.0".to_string(),
            citation: "test citation".to_string(),
            version: "0.1.0".to_string(),
            created: "2026-05-01".to_string(),
            notes: Some("test notes".to_string()),
        };

        let toml_str = toml::to_string(&original).expect("serialization must succeed");
        let recovered: ShapeModelManifest =
            toml::from_str(&toml_str).expect("deserialization must succeed");

        assert_eq!(original, recovered);
    }
}
