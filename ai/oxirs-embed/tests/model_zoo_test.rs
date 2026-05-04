//! Integration tests for the model zoo module.
//!
//! All file I/O uses `std::env::temp_dir()` — no persistent test artifacts.

use oxirs_embed::model_zoo::{sha256_hex, ModelManifest, ModelZoo, ModelZooError, ModelZooLoader};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn unique_tmp(suffix: &str) -> PathBuf {
    std::env::temp_dir().join(format!("oxirs_zoo_{suffix}_{}", std::process::id()))
}

/// Write a minimal valid manifest into `dir` and return the TOML text.
fn write_manifest(dir: &std::path::Path, manifest: &ModelManifest) -> String {
    std::fs::create_dir_all(dir).expect("create manifest dir");
    let toml_str = toml::to_string(manifest).expect("serialize manifest");
    std::fs::write(dir.join(format!("{}.toml", manifest.name)), &toml_str)
        .expect("write manifest toml");
    toml_str
}

/// Create a small synthetic TransE "checkpoint" file and return its path.
///
/// The content is a minimal JSON blob so `ModelRepository::load_model` does
/// not fail even though `TransE::load` is a no-op stub.
fn create_synthetic_ckpt(dir: &std::path::Path, filename: &str) -> PathBuf {
    std::fs::create_dir_all(dir).expect("create ckpt dir");
    let path = dir.join(filename);
    // Write deterministic JSON content
    std::fs::write(&path, br#"{"type":"TransE","entities":5,"relations":2}"#).expect("write ckpt");
    path
}

/// Build a [`ModelManifest`] that points to a local `file:///` checkpoint and
/// carries the correct SHA-256 digest of that file.
fn manifest_with_real_sha(name: &str, ckpt_path: &std::path::Path) -> ModelManifest {
    let bytes = std::fs::read(ckpt_path).expect("read ckpt for sha");
    let sha = sha256_hex(&bytes);
    ModelManifest {
        name: name.to_string(),
        model_type: "TransE".to_string(),
        dataset: "TestDS".to_string(),
        dimensions: 10,
        entities: 5,
        relations: 2,
        sha256: sha,
        source: format!("file:///{}", ckpt_path.to_string_lossy()),
        license: "Apache-2.0".to_string(),
        citation: "Test 2026".to_string(),
        version: "1.0.0".to_string(),
        created: "2026-05-01".to_string(),
        notes: Some("Synthetic seed for testing".to_string()),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_registry_parse() {
    let zoo = ModelZoo::registry();
    assert_eq!(
        zoo.len(),
        5,
        "Built-in registry must have exactly 5 entries"
    );
}

#[test]
fn test_manifest_serde_roundtrip() {
    let original = ModelManifest {
        name: "roundtrip-model".to_string(),
        model_type: "DistMult".to_string(),
        dataset: "WN18RR".to_string(),
        dimensions: 100,
        entities: 40943,
        relations: 11,
        sha256: "deadbeef".to_string(),
        source: "file:///seeds/roundtrip.ckpt".to_string(),
        license: "Apache-2.0".to_string(),
        citation: "Test Citation 2026".to_string(),
        version: "1.0.0".to_string(),
        created: "2026-05-01".to_string(),
        notes: Some("This is a test".to_string()),
    };

    let toml_str = toml::to_string(&original).expect("serialize to TOML");
    let deserialized: ModelManifest = toml::from_str(&toml_str).expect("deserialize from TOML");

    assert_eq!(
        original, deserialized,
        "Roundtrip should preserve all fields"
    );
}

#[test]
fn test_sha256_mismatch() {
    let tmp = unique_tmp("sha_mismatch");
    let manifest_dir = tmp.join("manifests");
    let ckpt_dir = tmp.join("ckpt");

    // Create a real checkpoint file
    let ckpt_path = create_synthetic_ckpt(&ckpt_dir, "mismatch.ckpt");

    // Build manifest with a WRONG sha256
    let manifest = ModelManifest {
        name: "sha-mismatch-model".to_string(),
        model_type: "TransE".to_string(),
        dataset: "TestDS".to_string(),
        dimensions: 10,
        entities: 5,
        relations: 2,
        sha256: "0".repeat(64), // wrong!
        source: format!("file:///{}", ckpt_path.to_string_lossy()),
        license: "Apache-2.0".to_string(),
        citation: "Test".to_string(),
        version: "1.0.0".to_string(),
        created: "2026-05-01".to_string(),
        notes: None,
    };

    write_manifest(&manifest_dir, &manifest);

    let zoo = ModelZoo::with_manifest_dir(&manifest_dir).expect("build zoo");
    let zoo_ref: &'static ModelZoo = Box::leak(Box::new(zoo));

    let loader = ModelZooLoader::with_zoo(zoo_ref, tmp.join("repo")).accept_license();
    let result = loader.load("sha-mismatch-model");

    let is_mismatch = matches!(result, Err(ModelZooError::ChecksumMismatch { .. }));
    assert!(is_mismatch, "Expected ChecksumMismatch error");

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn test_sha256_ok() {
    let tmp = unique_tmp("sha_ok");
    let manifest_dir = tmp.join("manifests");
    let ckpt_dir = tmp.join("ckpt");

    // Create a real checkpoint file
    let ckpt_path = create_synthetic_ckpt(&ckpt_dir, "ok.ckpt");

    // Build manifest with the CORRECT sha256
    let manifest = manifest_with_real_sha("sha-ok-model", &ckpt_path);
    write_manifest(&manifest_dir, &manifest);

    let zoo = ModelZoo::with_manifest_dir(&manifest_dir).expect("build zoo");
    let zoo_ref: &'static ModelZoo = Box::leak(Box::new(zoo));

    let repo_dir = tmp.join("repo");
    let loader = ModelZooLoader::with_zoo(zoo_ref, &repo_dir).accept_license();
    let result = loader.load("sha-ok-model");

    assert!(result.is_ok(), "Load with correct SHA should succeed");

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn test_license_refusal() {
    let tmp = unique_tmp("license_refusal");
    let manifest_dir = tmp.join("manifests");

    let manifest = ModelManifest {
        name: "restricted-model".to_string(),
        model_type: "TransE".to_string(),
        dataset: "TestDS".to_string(),
        dimensions: 10,
        entities: 5,
        relations: 2,
        sha256: "PLACEHOLDER".to_string(),
        source: "file:///nonexistent.ckpt".to_string(),
        license: "CC-BY-NC-4.0".to_string(), // non-permissive
        citation: "Test".to_string(),
        version: "1.0.0".to_string(),
        created: "2026-05-01".to_string(),
        notes: None,
    };

    write_manifest(&manifest_dir, &manifest);

    let zoo = ModelZoo::with_manifest_dir(&manifest_dir).expect("build zoo");
    let zoo_ref: &'static ModelZoo = Box::leak(Box::new(zoo));

    // Loader WITHOUT accept_license
    let loader = ModelZooLoader::with_zoo(zoo_ref, std::env::temp_dir());
    let result = loader.load("restricted-model");

    let is_refused = matches!(result, Err(ModelZooError::LicenseNotAccepted { .. }));
    assert!(is_refused, "Expected LicenseNotAccepted error");

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn test_missing_entry() {
    let zoo = ModelZoo::registry();
    assert!(
        zoo.get("nonexistent").is_none(),
        "Non-existent model should return None"
    );
}

#[test]
fn test_search() {
    let zoo = ModelZoo::registry();

    // FB15k-237 has 3 models
    let fb_results = zoo.search("FB15k");
    assert_eq!(
        fb_results.len(),
        3,
        "FB15k-237 has 3 models: {fb_results:?}"
    );

    // WN18RR has 2 models
    let wn_results = zoo.search("WN18RR");
    assert_eq!(wn_results.len(), 2, "WN18RR has 2 models");

    // Case-insensitive
    let upper = zoo.search("ROTATE");
    let lower = zoo.search("rotate");
    assert_eq!(upper.len(), lower.len());
    assert!(!lower.is_empty());
}

#[test]
fn test_list() {
    let zoo = ModelZoo::registry();
    let list = zoo.list();
    assert_eq!(list.len(), 5, "Must list all 5 built-in entries");
}

#[test]
fn test_unknown_model_type() {
    let tmp = unique_tmp("unknown_type");
    let manifest_dir = tmp.join("manifests");
    let ckpt_dir = tmp.join("ckpt");

    let ckpt_path = create_synthetic_ckpt(&ckpt_dir, "bogus.ckpt");
    let bytes = std::fs::read(&ckpt_path).expect("read ckpt");
    let sha = sha256_hex(&bytes);

    let manifest = ModelManifest {
        name: "bogus-model".to_string(),
        model_type: "Bogus".to_string(), // unsupported!
        dataset: "TestDS".to_string(),
        dimensions: 10,
        entities: 5,
        relations: 2,
        sha256: sha,
        source: format!("file:///{}", ckpt_path.to_string_lossy()),
        license: "Apache-2.0".to_string(),
        citation: "Test".to_string(),
        version: "1.0.0".to_string(),
        created: "2026-05-01".to_string(),
        notes: None,
    };

    write_manifest(&manifest_dir, &manifest);

    let zoo = ModelZoo::with_manifest_dir(&manifest_dir).expect("build zoo");
    let zoo_ref: &'static ModelZoo = Box::leak(Box::new(zoo));

    let loader = ModelZooLoader::with_zoo(zoo_ref, tmp.join("repo")).accept_license();
    let result = loader.load("bogus-model");

    let is_unsupported = matches!(result, Err(ModelZooError::UnsupportedModelType(_)));
    assert!(is_unsupported, "Expected UnsupportedModelType error");

    std::fs::remove_dir_all(&tmp).ok();
}
