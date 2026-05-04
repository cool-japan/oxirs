//! Thread-safe model registry for GGUF model metadata.
//!
//! The registry stores [`ModelInfo`] records (path + parsed metadata) keyed by
//! name-based [`ModelHandle`]s.  No tensor weights are held in memory.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;

use super::gguf_parser::{GgufMetadata, GgufParseError, GgufParser};

// ─── ModelHandle ─────────────────────────────────────────────────────────────

/// An opaque handle to a registered model, identified by name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelHandle(String);

impl ModelHandle {
    /// Return the model name.
    pub fn name(&self) -> &str {
        &self.0
    }
}

// ─── ModelInfo ───────────────────────────────────────────────────────────────

/// Lightweight model record held by the registry.
///
/// Contains only the file path and the parsed GGUF metadata header; tensor
/// weight data is never loaded.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Opaque handle for this model.
    pub handle: ModelHandle,
    /// Absolute path to the `.gguf` file.
    pub path: PathBuf,
    /// Parsed GGUF header metadata.
    pub metadata: GgufMetadata,
}

impl ModelInfo {
    /// Return the model's hidden / embedding dimension, if declared.
    pub fn embedding_dim(&self) -> Option<usize> {
        self.metadata.arch.embedding_length.map(|v| v as usize)
    }

    /// Return the vocabulary size, if declared.
    pub fn vocab_size(&self) -> Option<usize> {
        self.metadata.arch.vocab_size.map(|v| v as usize)
    }
}

// ─── RegistryError ───────────────────────────────────────────────────────────

/// Errors produced by the model registry.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    /// A model with this name is already registered.
    #[error("model '{0}' already registered")]
    AlreadyRegistered(String),
    /// No model with this name was found.
    #[error("model '{0}' not found")]
    NotFound(String),
    /// Failed to parse the GGUF file.
    #[error("GGUF parse error: {0}")]
    ParseError(#[from] GgufParseError),
}

// ─── ModelRegistry ───────────────────────────────────────────────────────────

/// Thread-safe registry of loaded model metadata.
///
/// Models are keyed by name; each registration parses (or accepts) GGUF
/// metadata without loading tensor weights.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "gguf-loader")]
/// # {
/// use oxirs_graphrag::model_loader::{ModelRegistry, GgufMetadata, GgufParser};
/// use std::path::PathBuf;
///
/// let registry = ModelRegistry::new();
/// // In real usage you would supply a path to a .gguf file:
/// // let handle = registry.register("llama-3-8b", PathBuf::from("model.gguf")).unwrap();
/// assert!(registry.is_empty());
/// # }
/// ```
#[derive(Default)]
pub struct ModelRegistry {
    models: RwLock<HashMap<String, ModelInfo>>,
}

impl ModelRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a model by parsing its GGUF file.
    ///
    /// Returns an error if `name` is already registered or the file cannot be
    /// parsed.
    pub fn register(&self, name: &str, path: PathBuf) -> Result<ModelHandle, RegistryError> {
        let metadata = GgufParser::parse_file(&path)?;
        self.register_with_metadata(name, path, metadata)
    }

    /// Register a model with pre-parsed metadata.
    ///
    /// Useful for tests that construct synthetic metadata without a real file.
    pub fn register_with_metadata(
        &self,
        name: &str,
        path: PathBuf,
        metadata: GgufMetadata,
    ) -> Result<ModelHandle, RegistryError> {
        let mut map = self.models.write().expect("registry lock poisoned");
        if map.contains_key(name) {
            return Err(RegistryError::AlreadyRegistered(name.to_owned()));
        }
        let handle = ModelHandle(name.to_owned());
        let info = ModelInfo {
            handle: handle.clone(),
            path,
            metadata,
        };
        map.insert(name.to_owned(), info);
        Ok(handle)
    }

    /// Look up a model by its handle.
    pub fn get(&self, handle: &ModelHandle) -> Option<ModelInfo> {
        let map = self.models.read().expect("registry lock poisoned");
        map.get(handle.name()).cloned()
    }

    /// Look up a model by name.
    pub fn get_by_name(&self, name: &str) -> Option<ModelInfo> {
        let map = self.models.read().expect("registry lock poisoned");
        map.get(name).cloned()
    }

    /// List all registered model handles.
    pub fn list(&self) -> Vec<ModelHandle> {
        let map = self.models.read().expect("registry lock poisoned");
        map.values().map(|info| info.handle.clone()).collect()
    }

    /// Remove a model from the registry.
    ///
    /// Returns `true` if the model existed and was removed.
    pub fn remove(&self, handle: &ModelHandle) -> bool {
        let mut map = self.models.write().expect("registry lock poisoned");
        map.remove(handle.name()).is_some()
    }

    /// Return the number of registered models.
    pub fn len(&self) -> usize {
        let map = self.models.read().expect("registry lock poisoned");
        map.len()
    }

    /// Return `true` if no models are registered.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
