//! Advanced compression codec system for oxirs-cluster.
//!
//! This module provides a `Compressor` trait with four built-in implementations:
//! - `IdentityCodec` — no compression (default; backward-compatible)
//! - `RleCodec`      — pure-Rust run-length encoding for repetitive byte streams
//! - `Lz4Codec`      — fast LZ4 via oxiarc-lz4 (Pure Rust)
//! - `ZstdCodec`     — Zstandard via oxiarc-zstd at configurable level (Pure Rust)
//!
//! A `CodecRegistry` maps codec names to `Arc<dyn Compressor>` so callers
//! can select a codec at runtime (e.g. per-shard or per-tenant).
//!
//! # Quick start
//!
//! ```rust
//! use oxirs_cluster::compression::{CodecRegistry, Compressor};
//!
//! let registry = CodecRegistry::default();
//! let codec = registry.default_codec(); // "identity" by default
//! let data = b"example payload";
//! let compressed = codec.compress(data).unwrap();
//! let decompressed = codec.decompress(&compressed).unwrap();
//! assert_eq!(decompressed, data);
//! ```

pub mod codecs;
pub mod registry;

pub use codecs::{CompressionError, Compressor, IdentityCodec, Lz4Codec, RleCodec, ZstdCodec};
pub use registry::CodecRegistry;
