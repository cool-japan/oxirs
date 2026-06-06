//! Integration tests for the GGUF model loader (feature = "gguf-loader").

#![cfg(feature = "gguf-loader")]

use std::path::PathBuf;

use oxirs_graphrag::model_loader::{
    GgufMetadata, GgufModelArch, GgufParseError, GgufParser, GgufTensorInfo, GgufValue,
    ModelRegistry,
};

// ── GGUF binary builder helpers ───────────────────────────────────────────────

/// Construct a minimal GGUF buffer for tests.
///
/// Layout: magic (4) + version (4) + n_tensors (8) + n_kv (8) = 24 bytes minimum.
fn minimal_gguf(version: u32, n_tensors: u64, n_kv: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    // magic "GGUF"
    buf.extend_from_slice(&[0x47, 0x47, 0x55, 0x46]);
    buf.extend_from_slice(&version.to_le_bytes());
    buf.extend_from_slice(&n_tensors.to_le_bytes());
    buf.extend_from_slice(&n_kv.to_le_bytes());
    buf
}

/// Append a GGUF string (u64 length + bytes) to `buf`.
fn push_gguf_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

/// Append a uint32 KV entry.
fn push_kv_u32(buf: &mut Vec<u8>, key: &str, val: u32) {
    push_gguf_str(buf, key);
    buf.extend_from_slice(&4u32.to_le_bytes()); // type = UINT32
    buf.extend_from_slice(&val.to_le_bytes());
}

/// Append a string KV entry.
fn push_kv_str(buf: &mut Vec<u8>, key: &str, val: &str) {
    push_gguf_str(buf, key);
    buf.extend_from_slice(&8u32.to_le_bytes()); // type = STRING
    push_gguf_str(buf, val);
}

/// Append one 2D tensor info record.
fn push_tensor_2d(buf: &mut Vec<u8>, name: &str, rows: u64, cols: u64, data_type: u32) {
    push_gguf_str(buf, name);
    buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims = 2
    buf.extend_from_slice(&rows.to_le_bytes());
    buf.extend_from_slice(&cols.to_le_bytes());
    buf.extend_from_slice(&data_type.to_le_bytes()); // data_type
    buf.extend_from_slice(&0u64.to_le_bytes()); // offset = 0
}

// ── Test 1: minimal valid GGUF v3 (no tensors, no kv) ────────────────────────

#[test]
fn test_parse_minimal_v3() {
    let buf = minimal_gguf(3, 0, 0);
    let meta = GgufParser::parse_bytes(&buf).expect("parse ok");
    assert_eq!(meta.version, 3, "version should be 3");
    assert_eq!(meta.n_tensors, 0, "n_tensors should be 0");
    assert!(meta.kv.is_empty(), "kv should be empty");
    assert!(meta.tensors.is_empty(), "tensors should be empty");
}

// ── Test 2: invalid magic → GgufParseError::InvalidMagic ─────────────────────

#[test]
fn test_parse_invalid_magic() {
    let mut buf = minimal_gguf(3, 0, 0);
    // Corrupt the magic bytes.
    buf[0] = 0xFF;
    let err = GgufParser::parse_bytes(&buf).expect_err("should fail with invalid magic");
    assert!(
        matches!(err, GgufParseError::InvalidMagic),
        "expected InvalidMagic, got {err:?}"
    );
}

// ── Test 3: unsupported version → GgufParseError::UnsupportedVersion ──────────

#[test]
fn test_parse_unsupported_version_1() {
    let buf = minimal_gguf(1, 0, 0);
    let err = GgufParser::parse_bytes(&buf).expect_err("version 1 should fail");
    assert!(
        matches!(err, GgufParseError::UnsupportedVersion(1)),
        "expected UnsupportedVersion(1), got {err:?}"
    );
}

#[test]
fn test_parse_unsupported_version_99() {
    let buf = minimal_gguf(99, 0, 0);
    let err = GgufParser::parse_bytes(&buf).expect_err("version 99 should fail");
    assert!(
        matches!(err, GgufParseError::UnsupportedVersion(99)),
        "expected UnsupportedVersion(99), got {err:?}"
    );
}

// ── Test 4: parse one uint32 kv entry ────────────────────────────────────────

#[test]
fn test_parse_kv_uint32() {
    let mut buf = minimal_gguf(3, 0, 1);
    push_kv_u32(&mut buf, "test.key", 42);
    let meta = GgufParser::parse_bytes(&buf).expect("parse ok");
    let val = meta.kv.get("test.key").expect("key must exist");
    assert!(
        matches!(val, GgufValue::U32(42)),
        "expected U32(42), got {val:?}"
    );
}

// ── Test 5: parse one string kv entry ────────────────────────────────────────

#[test]
fn test_parse_kv_string() {
    let mut buf = minimal_gguf(3, 0, 1);
    push_kv_str(&mut buf, "general.architecture", "llama");
    let meta = GgufParser::parse_bytes(&buf).expect("parse ok");
    let val = meta.kv.get("general.architecture").expect("key must exist");
    assert!(
        matches!(val, GgufValue::Str(s) if s == "llama"),
        "expected Str(\"llama\"), got {val:?}"
    );
}

// ── Test 6: parse one 2D tensor info record ───────────────────────────────────

#[test]
fn test_parse_tensor_info_2d() {
    let mut buf = minimal_gguf(3, 1, 0); // n_tensors = 1
    push_tensor_2d(&mut buf, "blk.0.attn_q.weight", 128, 64, 0); // F32
    let meta = GgufParser::parse_bytes(&buf).expect("parse ok");
    assert_eq!(meta.tensors.len(), 1);
    let t = &meta.tensors[0];
    assert_eq!(t.name, "blk.0.attn_q.weight");
    assert_eq!(t.dims, vec![128, 64]);
    assert_eq!(t.data_type, 0);
    assert_eq!(t.offset, 0);
    assert_eq!(t.param_count, 128 * 64, "param_count = product of dims");
}

// ── Test 7: GgufMetadata::total_params counts correctly ──────────────────────

#[test]
fn test_total_params_two_tensors() {
    let mut buf = minimal_gguf(3, 2, 0);
    push_tensor_2d(&mut buf, "layer.weight", 4, 8, 0);
    push_tensor_2d(&mut buf, "layer.bias", 1, 8, 0);
    let meta = GgufParser::parse_bytes(&buf).expect("parse ok");
    assert_eq!(meta.n_tensors, 2);
    assert_eq!(meta.total_params(), 32 + 8, "total_params = 40");
}

// ── Test 8: ModelRegistry::register_with_metadata + get by handle ─────────────

#[test]
fn test_registry_register_and_get() {
    let registry = ModelRegistry::new();
    let meta = GgufParser::parse_bytes(&minimal_gguf(3, 0, 0)).expect("parse ok");
    let handle = registry
        .register_with_metadata(
            "mymodel",
            std::env::temp_dir().join(format!("oxirs_mymodel_{}.gguf", std::process::id())),
            meta,
        )
        .expect("register ok");
    assert_eq!(handle.name(), "mymodel");
    let info = registry.get(&handle).expect("model must be found");
    assert_eq!(info.handle.name(), "mymodel");
    assert_eq!(registry.len(), 1);
}

// ── Test 9: duplicate registration → RegistryError::AlreadyRegistered ─────────

#[test]
fn test_registry_duplicate_error() {
    use oxirs_graphrag::model_loader::RegistryError;
    let registry = ModelRegistry::new();
    let meta1 = GgufParser::parse_bytes(&minimal_gguf(3, 0, 0)).expect("parse ok");
    let meta2 = GgufParser::parse_bytes(&minimal_gguf(3, 0, 0)).expect("parse ok");
    registry
        .register_with_metadata("same-name", PathBuf::from("/a.gguf"), meta1)
        .expect("first register ok");
    let err = registry
        .register_with_metadata("same-name", PathBuf::from("/b.gguf"), meta2)
        .expect_err("second register must fail");
    assert!(
        matches!(err, RegistryError::AlreadyRegistered(_)),
        "expected AlreadyRegistered, got {err:?}"
    );
}

// ── Test 10: get_by_name for unknown name → None ──────────────────────────────

#[test]
fn test_registry_get_by_name_not_found() {
    let registry = ModelRegistry::new();
    assert!(
        registry.get_by_name("nonexistent").is_none(),
        "unknown model must return None"
    );
}

// ── Test 11: GgufValue::as_u64 conversions ────────────────────────────────────

#[test]
fn test_gguf_value_as_u64() {
    assert_eq!(GgufValue::U8(200).as_u64(), Some(200));
    assert_eq!(GgufValue::U16(1000).as_u64(), Some(1000));
    assert_eq!(GgufValue::U32(70000).as_u64(), Some(70000));
    assert_eq!(GgufValue::U64(u64::MAX).as_u64(), Some(u64::MAX));
    assert_eq!(GgufValue::I32(42).as_u64(), Some(42));
    assert_eq!(GgufValue::I32(-1).as_u64(), None);
    assert_eq!(GgufValue::F32(1.0).as_u64(), None);
}

// ── Test 12: GgufValue::as_str conversions ────────────────────────────────────

#[test]
fn test_gguf_value_as_str() {
    assert_eq!(GgufValue::Str("hello".to_owned()).as_str(), Some("hello"));
    assert_eq!(GgufValue::U32(1).as_str(), None);
    assert_eq!(GgufValue::Bool(true).as_str(), None);
}

// ── Test 13: registry::remove works correctly ─────────────────────────────────

#[test]
fn test_registry_remove() {
    let registry = ModelRegistry::new();
    let meta = GgufParser::parse_bytes(&minimal_gguf(3, 0, 0)).expect("parse ok");
    let handle = registry
        .register_with_metadata("to-remove", PathBuf::from("/r.gguf"), meta)
        .expect("register ok");
    assert_eq!(registry.len(), 1);
    let removed = registry.remove(&handle);
    assert!(removed, "remove must return true for existing model");
    assert_eq!(registry.len(), 0);
    assert!(registry.get(&handle).is_none(), "model gone after removal");
}

// ── Test 14: architecture extraction from kv ──────────────────────────────────

#[test]
fn test_arch_extraction() {
    let mut buf = minimal_gguf(3, 0, 2);
    push_kv_str(&mut buf, "general.architecture", "mistral");
    push_kv_u32(&mut buf, "mistral.context_length", 32768);
    let meta = GgufParser::parse_bytes(&buf).expect("parse ok");
    assert_eq!(meta.arch.architecture.as_deref(), Some("mistral"));
    assert_eq!(meta.arch.context_length, Some(32768));
}

// ── Test 15: truncated file → Truncated error ─────────────────────────────────

#[test]
fn test_parse_truncated_file() {
    // Only 3 bytes — not even the full magic.
    let buf = vec![0x47, 0x47, 0x55];
    let err = GgufParser::parse_bytes(&buf).expect_err("should fail");
    // Could be InvalidMagic (if we read only 3 bytes instead of 4) or Truncated.
    // The exact variant depends on whether read_exact fails before or after magic check.
    // Accept either.
    assert!(
        matches!(
            err,
            GgufParseError::Truncated | GgufParseError::InvalidMagic
        ),
        "expected Truncated or InvalidMagic, got {err:?}"
    );
}
