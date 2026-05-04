//! Codec registry for dynamic codec selection by name.
//!
//! `CodecRegistry` holds named `Compressor` implementations and provides
//! look-up by name.  A default codec (`"identity"`) is always present so
//! existing callers that don't configure a codec see no behaviour change.

use std::collections::HashMap;
use std::sync::Arc;

use super::codecs::{CompressionError, Compressor, IdentityCodec, Lz4Codec, RleCodec, ZstdCodec};

/// Thread-safe registry of compression codecs indexed by name.
///
/// # Default state
/// When constructed via [`CodecRegistry::default`] the registry contains:
/// - `"identity"` — no-op, default
/// - `"rle"`      — pure-Rust run-length encoding
/// - `"lz4"`      — LZ4 via oxiarc-lz4
/// - `"zstd"`     — Zstandard via oxiarc-zstd
pub struct CodecRegistry {
    codecs: HashMap<String, Arc<dyn Compressor>>,
    default_name: String,
}

impl Default for CodecRegistry {
    fn default() -> Self {
        let mut reg = CodecRegistry {
            codecs: HashMap::new(),
            default_name: "identity".to_owned(),
        };
        reg.register(Arc::new(IdentityCodec));
        reg.register(Arc::new(RleCodec));
        reg.register(Arc::new(Lz4Codec));
        reg.register(Arc::new(ZstdCodec::default_level()));
        reg
    }
}

impl CodecRegistry {
    /// Register a new codec.  An existing entry with the same name is replaced.
    pub fn register(&mut self, codec: Arc<dyn Compressor>) {
        self.codecs.insert(codec.name().to_owned(), codec);
    }

    /// Set the name of the default codec.
    ///
    /// Returns an error if the name is not registered.
    pub fn set_default(&mut self, name: &str) -> Result<(), CompressionError> {
        if !self.codecs.contains_key(name) {
            return Err(CompressionError::UnknownCodec(name.to_owned()));
        }
        self.default_name = name.to_owned();
        Ok(())
    }

    /// Retrieve a codec by name.
    pub fn get(&self, name: &str) -> Result<Arc<dyn Compressor>, CompressionError> {
        self.codecs
            .get(name)
            .cloned()
            .ok_or_else(|| CompressionError::UnknownCodec(name.to_owned()))
    }

    /// Return the default codec.
    pub fn default_codec(&self) -> Arc<dyn Compressor> {
        self.codecs[&self.default_name].clone()
    }

    /// List the names of all registered codecs (order unspecified).
    pub fn available_codecs(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.codecs.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_registry_has_all_codecs() {
        let reg = CodecRegistry::default();
        let names = reg.available_codecs();
        assert!(names.contains(&"identity"), "identity missing: {names:?}");
        assert!(names.contains(&"rle"), "rle missing: {names:?}");
        assert!(names.contains(&"lz4"), "lz4 missing: {names:?}");
        assert!(names.contains(&"zstd"), "zstd missing: {names:?}");
    }

    #[test]
    fn default_codec_is_identity() {
        let reg = CodecRegistry::default();
        let c = reg.default_codec();
        assert_eq!(c.name(), "identity");
    }

    #[test]
    fn get_known_codec() {
        let reg = CodecRegistry::default();
        assert_eq!(reg.get("rle").unwrap().name(), "rle");
        assert_eq!(reg.get("lz4").unwrap().name(), "lz4");
        assert_eq!(reg.get("zstd").unwrap().name(), "zstd");
    }

    #[test]
    fn get_unknown_codec_errors() {
        let reg = CodecRegistry::default();
        assert!(reg.get("nonexistent").is_err());
    }

    #[test]
    fn set_default_changes_default() {
        let mut reg = CodecRegistry::default();
        reg.set_default("zstd").unwrap();
        assert_eq!(reg.default_codec().name(), "zstd");
    }

    #[test]
    fn set_default_unknown_errors() {
        let mut reg = CodecRegistry::default();
        assert!(reg.set_default("unknown_codec").is_err());
    }

    #[test]
    fn register_custom_codec_is_retrievable() {
        struct NoopCodec;
        impl super::super::codecs::Compressor for NoopCodec {
            fn name(&self) -> &'static str {
                "noop"
            }
            fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
                Ok(data.to_vec())
            }
            fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
                Ok(data.to_vec())
            }
        }

        let mut reg = CodecRegistry::default();
        reg.register(Arc::new(NoopCodec));
        assert!(reg.get("noop").is_ok());
    }

    #[test]
    fn round_trip_via_registry_each_codec() {
        let reg = CodecRegistry::default();
        let data = b"The quick brown fox jumps over the lazy dog".repeat(50);
        for name in reg.available_codecs() {
            let codec = reg.get(name).unwrap();
            let enc = codec.compress(&data).expect(name);
            let dec = codec.decompress(&enc).expect(name);
            assert_eq!(dec, data.as_slice(), "round-trip failed for codec: {name}");
        }
    }
}
