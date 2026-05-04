//! Integration tests for the shape-model zoo.
//!
//! All file I/O uses [`std::env::temp_dir()`] so that these tests are
//! hermetic and do not leave state in the repository tree.

use sha2::{Digest, Sha256};

use oxirs_shacl_ai::model_zoo::{
    ShapeModelManifest, ShapeModelZoo, ShapeModelZooError, ShapeModelZooLoader,
};

// ---------------------------------------------------------------------------
// Registry tests
// ---------------------------------------------------------------------------

#[test]
fn test_registry_has_four_entries() {
    assert_eq!(
        ShapeModelZoo::registry().list().len(),
        4,
        "there must be exactly 4 built-in manifests"
    );
}

#[test]
fn test_registry_get_existing() {
    let m = ShapeModelZoo::registry().get("gat-shacl-base");
    assert!(m.is_some(), "gat-shacl-base must be in the registry");
    assert_eq!(m.expect("just checked").model_type, "GAT");
}

#[test]
fn test_registry_get_missing() {
    assert!(
        ShapeModelZoo::registry().get("nonexistent").is_none(),
        "missing key must return None"
    );
}

#[test]
fn test_search_by_substring() {
    let results = ShapeModelZoo::registry().search("shacl");
    assert!(
        !results.is_empty(),
        "search('shacl') must return ≥ 1 result"
    );
}

#[test]
fn test_by_model_type() {
    let results = ShapeModelZoo::registry().by_model_type("GAT");
    assert!(
        !results.is_empty(),
        "by_model_type('GAT') must return ≥ 1 result"
    );
}

// ---------------------------------------------------------------------------
// Manifest serde round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_manifest_serde_roundtrip() {
    let original = ShapeModelManifest {
        name: "roundtrip-test".to_string(),
        model_type: "GraphSAGE".to_string(),
        dataset: "LUBM-synthetic".to_string(),
        input_dim: 32,
        hidden_dim: 128,
        output_dim: 16,
        num_heads: 8,
        num_layers: 4,
        sha256: "cafebabe".to_string(),
        source: "file:///tmp/roundtrip.ckpt".to_string(),
        license: "MIT".to_string(),
        citation: "test citation".to_string(),
        version: "2.0.0".to_string(),
        created: "2026-05-01".to_string(),
        notes: Some("test notes".to_string()),
    };

    let toml_str = toml::to_string(&original).expect("serialization must succeed");
    let recovered: ShapeModelManifest =
        toml::from_str(&toml_str).expect("deserialization must succeed");

    assert_eq!(original, recovered);
}

// ---------------------------------------------------------------------------
// SHA-256 verification
// ---------------------------------------------------------------------------

#[test]
fn test_sha256_verify_ok() {
    let tmp = std::env::temp_dir().join("model_zoo_test_sha256_ok.ckpt");
    let content = b"synthetic-seed-GAT-v1";
    std::fs::write(&tmp, content).expect("write must succeed");
    let data = std::fs::read(&tmp).expect("read must succeed");

    ShapeModelZooLoader::verify_sha256(
        &data,
        "f5166a14c94cb4e8fb3ca7db76954312fe6a2859709de22026228c98db9f0a2d",
    )
    .expect("correct SHA-256 must pass");
}

#[test]
fn test_sha256_verify_fail() {
    let tmp = std::env::temp_dir().join("model_zoo_test_sha256_fail.ckpt");
    let mut content = b"synthetic-seed-GAT-v1".to_vec();
    content[0] ^= 0xFF; // corrupt one byte
    std::fs::write(&tmp, &content).expect("write must succeed");
    let data = std::fs::read(&tmp).expect("read must succeed");

    let err = ShapeModelZooLoader::verify_sha256(
        &data,
        "f5166a14c94cb4e8fb3ca7db76954312fe6a2859709de22026228c98db9f0a2d",
    )
    .expect_err("wrong SHA-256 must fail");

    assert!(
        matches!(err, ShapeModelZooError::ChecksumMismatch { .. }),
        "must be ChecksumMismatch, got: {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Loader tests
// ---------------------------------------------------------------------------

#[test]
fn test_loader_license_refusal() {
    let manifest = ShapeModelManifest {
        name: "proprietary-model".to_string(),
        model_type: "GAT".to_string(),
        dataset: "secret".to_string(),
        input_dim: 16,
        hidden_dim: 64,
        output_dim: 8,
        num_heads: 4,
        num_layers: 2,
        sha256: "deadbeef".to_string(),
        source: "file:///nonexistent/model.ckpt".to_string(),
        license: "Proprietary".to_string(),
        citation: "none".to_string(),
        version: "1.0.0".to_string(),
        created: "2026-05-01".to_string(),
        notes: None,
    };

    let zoo = ShapeModelZoo::with_single_entry(manifest);
    let tmp_dir = std::env::temp_dir();
    let loader = ShapeModelZooLoader::with_zoo(zoo, &tmp_dir);

    let err = loader
        .load("proprietary-model")
        .expect_err("proprietary license without accept_license() must fail");

    assert!(
        matches!(err, ShapeModelZooError::LicenseNotAccepted { .. }),
        "must be LicenseNotAccepted, got: {err:?}"
    );
}

#[test]
fn test_loader_load_ok() {
    let tmp_dir = std::env::temp_dir();
    let ckpt_name = "model_zoo_test_load_ok.ckpt";
    let ckpt_path = tmp_dir.join(ckpt_name);

    let content = b"synthetic-seed-GAT-v1";
    std::fs::write(&ckpt_path, content).expect("write must succeed");

    // Compute the real checksum from the content we just wrote.
    let mut hasher = Sha256::new();
    hasher.update(content);
    let sha256 = hex::encode(hasher.finalize());

    let manifest = ShapeModelManifest {
        name: "gat-load-ok".to_string(),
        model_type: "GAT".to_string(),
        dataset: "LUBM-SHACL-synthetic".to_string(),
        input_dim: 16,
        hidden_dim: 64,
        output_dim: 8,
        num_heads: 4,
        num_layers: 2,
        sha256,
        // relative path — loader resolves relative to base_dir (tmp_dir)
        source: format!("file://{ckpt_name}"),
        license: "Apache-2.0".to_string(),
        citation: "test".to_string(),
        version: "1.0.0".to_string(),
        created: "2026-05-01".to_string(),
        notes: None,
    };

    let zoo = ShapeModelZoo::with_single_entry(manifest);
    let loader = ShapeModelZooLoader::with_zoo(zoo, &tmp_dir);

    let loaded = loader.load("gat-load-ok").expect("load must succeed");
    assert_eq!(
        loaded.weights, content,
        "loaded bytes must match written bytes"
    );
    assert_eq!(loaded.manifest.name, "gat-load-ok");
    assert_eq!(loaded.manifest.model_type, "GAT");
}
