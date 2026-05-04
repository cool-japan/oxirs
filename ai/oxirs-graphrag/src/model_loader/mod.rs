//! Pure-Rust GGUF model metadata loader and thread-safe model registry.
//!
//! This module is enabled by the `gguf-loader` Cargo feature.
//!
//! # Overview
//!
//! - [`GgufParser`] — reads the GGUF v2/v3 file header (magic, version,
//!   key-value metadata, tensor shape/offset records) **without** loading
//!   tensor weights into RAM.
//! - [`ModelRegistry`] — a thread-safe `RwLock`-protected store of
//!   [`ModelInfo`] records indexed by name-based [`ModelHandle`]s.
//!
//! Together these components provide the "model loading strategy" required by
//! the LoRA phase-d fine-tuning scaffold: you can enumerate model architecture
//! metadata (embedding dimension, vocab size, layer count, …) before committing
//! to loading any weights, and the registry lets multiple threads share that
//! metadata safely.
//!
//! # Example
//!
//! ```rust
//! # #[cfg(feature = "gguf-loader")]
//! # {
//! use oxirs_graphrag::model_loader::{GgufParser, ModelRegistry};
//!
//! // Parse a minimal in-memory GGUF buffer (v3, 0 tensors, 0 kv entries):
//! let mut buf = Vec::new();
//! buf.extend_from_slice(&[0x47, 0x47, 0x55, 0x46]); // magic
//! buf.extend_from_slice(&3u32.to_le_bytes());         // version 3
//! buf.extend_from_slice(&0u64.to_le_bytes());         // n_tensors
//! buf.extend_from_slice(&0u64.to_le_bytes());         // n_kv
//!
//! let meta = GgufParser::parse_bytes(&buf).expect("parse ok");
//! assert_eq!(meta.version, 3);
//! assert_eq!(meta.n_tensors, 0);
//! assert_eq!(meta.total_params(), 0);
//!
//! let registry = ModelRegistry::new();
//! assert!(registry.is_empty());
//! # }
//! ```

pub mod gguf_parser;
pub mod registry;

pub use gguf_parser::{
    GgufMetadata, GgufModelArch, GgufParseError, GgufParser, GgufTensorInfo, GgufValue,
};
pub use registry::{ModelHandle, ModelInfo, ModelRegistry, RegistryError};
