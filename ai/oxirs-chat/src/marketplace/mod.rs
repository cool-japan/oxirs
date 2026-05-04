//! # Model Marketplace
//!
//! Unified model discovery and management across multiple AI model sources:
//!
//! | Source | Registry | Notes |
//! |---|---|---|
//! | HuggingFace Hub | [`HuggingFaceRegistry`] | Offline catalogue; production wires to `api.huggingface.co` |
//! | Ollama | [`OllamaRegistry`] | Lists models from a local Ollama server |
//! | Local GGUF | [`LocalFileRegistry`] | Discovers `*.gguf` files on disk |
//!
//! ## Quick start
//!
//! ```rust
//! use oxirs_chat::marketplace::{
//!     ModelMarketplace, ModelType,
//!     HuggingFaceRegistry, OllamaRegistry,
//! };
//!
//! let marketplace = ModelMarketplace::new()
//!     .with_registry(Box::new(HuggingFaceRegistry::new()))
//!     .with_registry(Box::new(OllamaRegistry::default()));
//!
//! let all_models = marketplace.list_all().expect("list_all should succeed");
//! let chat_models = marketplace.filter_by_type(ModelType::Chat)
//!     .expect("filter_by_type should succeed");
//! assert!(!chat_models.is_empty());
//! ```

pub mod huggingface;
pub mod local;
pub mod ollama;
pub mod registry;

pub use huggingface::HuggingFaceRegistry;
pub use local::LocalFileRegistry;
pub use ollama::OllamaRegistry;
pub use registry::{ModelEntry, ModelRegistry, ModelSource, ModelType};

use thiserror::Error;

/// Errors that can occur when interacting with the model marketplace.
#[derive(Debug, Error)]
pub enum MarketplaceError {
    /// A filesystem I/O error occurred.
    #[error("IO error: {0}")]
    Io(String),
    /// The provided path is not a valid model location.
    #[error("invalid model path: {0}")]
    InvalidPath(String),
    /// A requested model could not be found in any registry.
    #[error("model not found: {0}")]
    NotFound(String),
    /// A serialization or deserialization error occurred.
    #[error("serialization error: {0}")]
    Serialization(String),
}

/// Aggregates multiple [`ModelRegistry`] sources into a single discovery API.
///
/// Use the builder pattern to compose registries at construction time, or
/// mutate the registry list later via [`ModelMarketplace::add_registry`].
pub struct ModelMarketplace {
    registries: Vec<Box<dyn ModelRegistry>>,
}

impl ModelMarketplace {
    /// Create an empty marketplace with no registries.
    ///
    /// Add registries with [`ModelMarketplace::with_registry`] or
    /// [`ModelMarketplace::add_registry`].
    pub fn new() -> Self {
        Self {
            registries: Vec::new(),
        }
    }

    /// Attach a registry and return `self` for method chaining.
    pub fn with_registry(mut self, registry: Box<dyn ModelRegistry>) -> Self {
        self.registries.push(registry);
        self
    }

    /// Attach a registry in-place.
    pub fn add_registry(&mut self, registry: Box<dyn ModelRegistry>) {
        self.registries.push(registry);
    }

    /// List all models from every registered registry.
    ///
    /// Results are concatenated in the order registries were added.  If a
    /// registry returns an error, that error is propagated immediately.
    pub fn list_all(&self) -> Result<Vec<ModelEntry>, MarketplaceError> {
        let mut all = Vec::new();
        for registry in &self.registries {
            let mut entries = registry.list_models()?;
            all.append(&mut entries);
        }
        Ok(all)
    }

    /// Search across all registries and return de-duplicated results.
    ///
    /// Deduplication is performed on the model `id` field; the first occurrence
    /// wins when the same `id` appears in multiple registries.
    pub fn search_all(&self, query: &str) -> Result<Vec<ModelEntry>, MarketplaceError> {
        let mut seen_ids = std::collections::HashSet::new();
        let mut results = Vec::new();

        for registry in &self.registries {
            for entry in registry.search(query)? {
                if seen_ids.insert(entry.id.clone()) {
                    results.push(entry);
                }
            }
        }

        Ok(results)
    }

    /// Find a model by `id` in any registry, returning the first match.
    ///
    /// Returns `Ok(None)` when no registry contains a model with that `id`.
    pub fn find(&self, id: &str) -> Result<Option<ModelEntry>, MarketplaceError> {
        for registry in &self.registries {
            if let Some(entry) = registry.get_model(id)? {
                return Ok(Some(entry));
            }
        }
        Ok(None)
    }

    /// Return only models whose [`ModelType`] matches `model_type`.
    pub fn filter_by_type(
        &self,
        model_type: ModelType,
    ) -> Result<Vec<ModelEntry>, MarketplaceError> {
        Ok(self
            .list_all()?
            .into_iter()
            .filter(|e| e.model_type == model_type)
            .collect())
    }

    /// The number of registries currently attached to this marketplace.
    pub fn registry_count(&self) -> usize {
        self.registries.len()
    }
}

impl Default for ModelMarketplace {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use super::*;

    // ── HuggingFace Hub tests ────────────────────────────────────────────────

    #[test]
    fn test_huggingface_registry_nonempty() {
        let registry = HuggingFaceRegistry::new();
        let models = registry.list_models().expect("list_models should succeed");
        assert!(
            !models.is_empty(),
            "HuggingFace catalogue should be non-empty"
        );
    }

    #[test]
    fn test_huggingface_search_llama() {
        let registry = HuggingFaceRegistry::new();
        let results = registry.search("llama").expect("search should succeed");
        assert!(
            !results.is_empty(),
            "search('llama') should return at least one result"
        );
    }

    #[test]
    fn test_huggingface_get_by_id() {
        let registry = HuggingFaceRegistry::new();
        let model = registry
            .get_model("meta-llama/Llama-2-7b-chat-hf")
            .expect("get_model should succeed");
        assert!(model.is_some(), "known model id should be found");
        let entry = model.expect("model exists");
        assert_eq!(entry.id, "meta-llama/Llama-2-7b-chat-hf");
    }

    // ── Ollama tests ─────────────────────────────────────────────────────────

    fn sample_ollama_models(url: &str) -> Vec<ModelEntry> {
        vec![
            ModelEntry {
                id: "ollama::llama2".to_string(),
                name: "llama2".to_string(),
                source: ModelSource::Ollama {
                    server_url: url.to_string(),
                    name: "llama2".to_string(),
                },
                model_type: ModelType::Chat,
                size_bytes: None,
                tags: vec!["chat".to_string(), "llama".to_string()],
                description: "Meta Llama 2 via Ollama.".to_string(),
                download_url: None,
            },
            ModelEntry {
                id: "ollama::nomic-embed-text".to_string(),
                name: "nomic-embed-text".to_string(),
                source: ModelSource::Ollama {
                    server_url: url.to_string(),
                    name: "nomic-embed-text".to_string(),
                },
                model_type: ModelType::Embedding,
                size_bytes: None,
                tags: vec!["embedding".to_string()],
                description: "Nomic embedding model via Ollama.".to_string(),
                download_url: None,
            },
        ]
    }

    #[test]
    fn test_ollama_registry_with_models() {
        let url = "http://localhost:11434";
        let registry = OllamaRegistry::with_models(url, sample_ollama_models(url));
        let models = registry.list_models().expect("list_models should succeed");
        assert_eq!(models.len(), 2);
    }

    #[test]
    fn test_ollama_search_by_tag() {
        let url = "http://localhost:11434";
        let registry = OllamaRegistry::with_models(url, sample_ollama_models(url));
        let results = registry.search("chat").expect("search should succeed");
        assert!(
            !results.is_empty(),
            "search('chat') should return tagged models"
        );
    }

    // ── Local filesystem tests ────────────────────────────────────────────────

    #[test]
    fn test_local_registry_scan_empty_dir() {
        let dir = tempfile::tempdir().expect("tempdir creation should succeed");
        let registry = LocalFileRegistry::new(dir.path());
        let models = registry
            .list_models()
            .expect("list_models should succeed on empty dir");
        assert!(models.is_empty(), "empty directory should yield no models");
    }

    #[test]
    fn test_local_registry_scan_gguf_files() {
        let dir = tempfile::tempdir().expect("tempdir creation should succeed");

        File::create(dir.path().join("llama.Q4_K_M.gguf")).expect("create first gguf file");
        File::create(dir.path().join("mistral.Q4_K_M.gguf")).expect("create second gguf file");
        // Non-GGUF file — should be ignored.
        File::create(dir.path().join("README.md")).expect("create non-gguf file");

        let registry = LocalFileRegistry::new(dir.path());
        let models = registry.list_models().expect("list_models should succeed");

        assert_eq!(models.len(), 2, "only .gguf files should be registered");
    }

    // ── Marketplace aggregation tests ─────────────────────────────────────────

    #[test]
    fn test_marketplace_aggregates_multiple_registries() {
        let url = "http://localhost:11434";

        let hf_registry = HuggingFaceRegistry::new();
        let hf_count = hf_registry.list_models().expect("HF list_models").len();

        let ollama_models = sample_ollama_models(url);
        let ollama_count = ollama_models.len();
        let ollama_registry = OllamaRegistry::with_models(url, ollama_models);

        let marketplace = ModelMarketplace::new()
            .with_registry(Box::new(hf_registry))
            .with_registry(Box::new(ollama_registry));

        let all = marketplace.list_all().expect("list_all should succeed");
        assert_eq!(
            all.len(),
            hf_count + ollama_count,
            "list_all should aggregate both registries"
        );
    }

    #[test]
    fn test_marketplace_search_all() {
        let url = "http://localhost:11434";

        let hf_registry = HuggingFaceRegistry::new();
        let ollama_registry = OllamaRegistry::with_models(url, sample_ollama_models(url));

        let marketplace = ModelMarketplace::new()
            .with_registry(Box::new(hf_registry))
            .with_registry(Box::new(ollama_registry));

        // "llama" exists in both HF (Llama-2-*) and Ollama (llama2).
        let results = marketplace
            .search_all("llama")
            .expect("search_all should succeed");
        assert!(
            results.len() >= 2,
            "search_all should find llama in multiple registries"
        );
    }

    #[test]
    fn test_marketplace_find_by_id() {
        let url = "http://localhost:11434";
        let marketplace = ModelMarketplace::new()
            .with_registry(Box::new(HuggingFaceRegistry::new()))
            .with_registry(Box::new(OllamaRegistry::with_models(
                url,
                sample_ollama_models(url),
            )));

        let found = marketplace
            .find("meta-llama/Llama-2-7b-chat-hf")
            .expect("find should not error");
        assert!(found.is_some(), "should locate HF model by id");

        let not_found = marketplace
            .find("no-such-model-xyz-999")
            .expect("find should not error for unknown id");
        assert!(not_found.is_none(), "unknown id should return None");
    }

    #[test]
    fn test_marketplace_filter_by_type() {
        let url = "http://localhost:11434";
        let marketplace = ModelMarketplace::new()
            .with_registry(Box::new(HuggingFaceRegistry::new()))
            .with_registry(Box::new(OllamaRegistry::with_models(
                url,
                sample_ollama_models(url),
            )));

        let chat_models = marketplace
            .filter_by_type(ModelType::Chat)
            .expect("filter_by_type should succeed");

        assert!(
            !chat_models.is_empty(),
            "there should be at least one Chat model"
        );
        for entry in &chat_models {
            assert_eq!(
                entry.model_type,
                ModelType::Chat,
                "filter_by_type returned a non-Chat entry: {}",
                entry.id
            );
        }
    }

    #[test]
    fn test_model_entry_serialization() {
        let entry = ModelEntry {
            id: "test/serialization-check".to_string(),
            name: "Serialization Check".to_string(),
            source: ModelSource::HuggingFaceHub {
                repo_id: "test/serialization-check".to_string(),
            },
            model_type: ModelType::Embedding,
            size_bytes: Some(1_234_567),
            tags: vec!["test".to_string(), "serialization".to_string()],
            description: "A model entry used to test serde round-trip.".to_string(),
            download_url: Some("https://example.com/model".to_string()),
        };

        let json = serde_json::to_string(&entry).expect("serialization should succeed");
        let decoded: ModelEntry =
            serde_json::from_str(&json).expect("deserialization should succeed");

        assert_eq!(entry.id, decoded.id);
        assert_eq!(entry.name, decoded.name);
        assert_eq!(entry.source, decoded.source);
        assert_eq!(entry.model_type, decoded.model_type);
        assert_eq!(entry.size_bytes, decoded.size_bytes);
        assert_eq!(entry.tags, decoded.tags);
        assert_eq!(entry.description, decoded.description);
        assert_eq!(entry.download_url, decoded.download_url);
    }

    #[test]
    fn test_marketplace_registry_count() {
        let mut mp = ModelMarketplace::new();
        assert_eq!(mp.registry_count(), 0);
        mp.add_registry(Box::new(HuggingFaceRegistry::new()));
        assert_eq!(mp.registry_count(), 1);
        mp.add_registry(Box::new(OllamaRegistry::default()));
        assert_eq!(mp.registry_count(), 2);
    }
}
