//! Integration tests for the compression codec system.
//!
//! These tests exercise the public API exposed by `oxirs_cluster::compression`.

use oxirs_cluster::compression::{
    codecs::{Compressor, IdentityCodec, Lz4Codec, RleCodec, ZstdCodec},
    registry::CodecRegistry,
    CompressionError,
};

// ---------------------------------------------------------------------------
// IdentityCodec
// ---------------------------------------------------------------------------

#[test]
fn identity_codec_round_trip_empty() {
    let codec = IdentityCodec;
    let data: Vec<u8> = vec![];
    let compressed = codec.compress(&data).unwrap();
    let decompressed = codec.decompress(&compressed).unwrap();
    assert_eq!(decompressed, data);
}

#[test]
fn identity_codec_round_trip_all_bytes() {
    let codec = IdentityCodec;
    let data: Vec<u8> = (0u8..=255).collect();
    let compressed = codec.compress(&data).unwrap();
    let decompressed = codec.decompress(&compressed).unwrap();
    assert_eq!(decompressed, data);
}

#[test]
fn identity_codec_is_noop() {
    let codec = IdentityCodec;
    let data = b"hello world";
    let compressed = codec.compress(data).unwrap();
    assert_eq!(&compressed, data); // identity: no change
}

// ---------------------------------------------------------------------------
// RleCodec
// ---------------------------------------------------------------------------

#[test]
fn rle_codec_round_trip_empty() {
    let codec = RleCodec;
    let data: Vec<u8> = vec![];
    let compressed = codec.compress(&data).unwrap();
    let decompressed = codec.decompress(&compressed).unwrap();
    assert_eq!(decompressed, data);
}

#[test]
fn rle_codec_round_trip_repetitive() {
    let codec = RleCodec;
    let data = vec![0u8; 100];
    let compressed = codec.compress(&data).unwrap();
    assert!(
        compressed.len() < data.len(),
        "RLE should compress 100 zeros; got {} bytes for {} input",
        compressed.len(),
        data.len()
    );
    let decompressed = codec.decompress(&compressed).unwrap();
    assert_eq!(decompressed, data);
}

#[test]
fn rle_codec_round_trip_all_unique() {
    let codec = RleCodec;
    let data: Vec<u8> = (0u8..=127).collect();
    let compressed = codec.compress(&data).unwrap();
    let decompressed = codec.decompress(&compressed).unwrap();
    assert_eq!(decompressed, data);
}

#[test]
fn rle_codec_decompression_odd_input_error() {
    let codec = RleCodec;
    let result = codec.decompress(&[0xABu8]);
    assert!(
        matches!(result, Err(CompressionError::DecompressFailed(_))),
        "expected DecompressFailed for odd-length input"
    );
}

#[test]
fn rle_codec_long_run_split() {
    // 300 bytes of the same value — requires two RLE pairs (255 + 45)
    let codec = RleCodec;
    let data = vec![7u8; 300];
    let compressed = codec.compress(&data).unwrap();
    assert_eq!(
        compressed.len(),
        4,
        "300-byte run should produce 2 pairs (4 bytes)"
    );
    let decompressed = codec.decompress(&compressed).unwrap();
    assert_eq!(decompressed, data);
}

// ---------------------------------------------------------------------------
// Lz4Codec
// ---------------------------------------------------------------------------

#[test]
fn lz4_codec_round_trip_empty() {
    let codec = Lz4Codec;
    let compressed = codec.compress(&[]).unwrap();
    let decompressed = codec.decompress(&compressed).unwrap();
    assert_eq!(decompressed, Vec::<u8>::new());
}

#[test]
fn lz4_codec_round_trip_repetitive() {
    let codec = Lz4Codec;
    let data = b"abcdef".repeat(500);
    let compressed = codec.compress(&data).unwrap();
    assert!(
        compressed.len() < data.len(),
        "LZ4 should compress repetitive data"
    );
    let decompressed = codec.decompress(&compressed).unwrap();
    assert_eq!(decompressed, data);
}

#[test]
fn lz4_codec_round_trip_all_zeros() {
    let codec = Lz4Codec;
    let data = vec![0u8; 4096];
    let decompressed = codec.decompress(&codec.compress(&data).unwrap()).unwrap();
    assert_eq!(decompressed, data);
}

// ---------------------------------------------------------------------------
// ZstdCodec
// ---------------------------------------------------------------------------

#[test]
fn zstd_codec_round_trip_empty() {
    let codec = ZstdCodec::default_level();
    let compressed = codec.compress(&[]).unwrap();
    let decompressed = codec.decompress(&compressed).unwrap();
    assert_eq!(decompressed, Vec::<u8>::new());
}

#[test]
fn zstd_codec_round_trip_repetitive() {
    let codec = ZstdCodec::new(3);
    let data = b"oxirs cluster zstd test ".repeat(500);
    let compressed = codec.compress(&data).unwrap();
    assert!(
        compressed.len() < data.len(),
        "Zstd should compress repetitive data"
    );
    let decompressed = codec.decompress(&compressed).unwrap();
    assert_eq!(decompressed, data);
}

#[test]
fn zstd_codec_round_trip_all_bytes() {
    let codec = ZstdCodec::default_level();
    let data: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
    let decompressed = codec.decompress(&codec.compress(&data).unwrap()).unwrap();
    assert_eq!(decompressed, data);
}

// ---------------------------------------------------------------------------
// CodecRegistry
// ---------------------------------------------------------------------------

#[test]
fn codec_registry_has_all_four_codecs() {
    let registry = CodecRegistry::default();
    let names = registry.available_codecs();
    for expected in &["identity", "rle", "lz4", "zstd"] {
        assert!(
            names.contains(expected),
            "missing codec '{expected}'; available: {names:?}"
        );
    }
}

#[test]
fn codec_registry_default_codec_is_identity() {
    let registry = CodecRegistry::default();
    let default = registry.default_codec();
    assert_eq!(default.name(), "identity");
}

#[test]
fn codec_registry_get_each_codec_by_name() {
    let registry = CodecRegistry::default();
    for name in &["identity", "rle", "lz4", "zstd"] {
        let codec = registry
            .get(name)
            .unwrap_or_else(|_| panic!("codec '{name}' not found"));
        assert_eq!(codec.name(), *name);
    }
}

#[test]
fn codec_registry_unknown_codec_returns_error() {
    let registry = CodecRegistry::default();
    assert!(registry.get("nonexistent").is_err());
}

#[test]
fn codec_registry_round_trip_all_codecs() {
    let registry = CodecRegistry::default();
    let data = b"The quick brown fox jumps over the lazy dog".repeat(100);
    for name in registry.available_codecs() {
        let codec = registry.get(name).unwrap();
        let enc = codec
            .compress(&data)
            .unwrap_or_else(|e| panic!("{name}: compress error: {e}"));
        let dec = codec
            .decompress(&enc)
            .unwrap_or_else(|e| panic!("{name}: decompress error: {e}"));
        assert_eq!(dec, data.as_slice(), "round-trip failed for codec '{name}'");
    }
}

#[test]
fn codec_registry_set_default_changes_default() {
    let mut registry = CodecRegistry::default();
    registry.set_default("zstd").unwrap();
    assert_eq!(registry.default_codec().name(), "zstd");
}

#[test]
fn codec_registry_set_unknown_default_errors() {
    let mut registry = CodecRegistry::default();
    assert!(registry.set_default("bogus").is_err());
}
