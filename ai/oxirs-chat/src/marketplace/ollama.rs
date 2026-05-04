//! Ollama local server model registry.
//!
//! In production, `load_from_server` would call the Ollama `/api/tags`
//! endpoint.  The current implementation uses an in-memory catalogue so the
//! registry works without a running Ollama instance.

use crate::marketplace::{MarketplaceError, ModelEntry, ModelRegistry};

/// A model registry backed by an Ollama local server.
///
/// The `catalogue` is populated either via [`OllamaRegistry::with_models`]
/// (for tests / offline use) or by calling [`OllamaRegistry::load_from_server`].
pub struct OllamaRegistry {
    /// Base URL of the Ollama server, e.g. `"http://localhost:11434"`.
    server_url: String,
    /// Current in-memory model catalogue.
    catalogue: Vec<ModelEntry>,
}

impl OllamaRegistry {
    /// Create an empty registry pointing at the given Ollama server.
    ///
    /// Call [`OllamaRegistry::load_from_server`] to populate the catalogue
    /// from the live `/api/tags` endpoint.
    pub fn new(server_url: impl Into<String>) -> Self {
        Self {
            server_url: server_url.into(),
            catalogue: Vec::new(),
        }
    }

    /// Create a registry pre-populated with `models` (useful for tests).
    pub fn with_models(server_url: impl Into<String>, models: Vec<ModelEntry>) -> Self {
        Self {
            server_url: server_url.into(),
            catalogue: models,
        }
    }

    /// Attempt to populate the catalogue from the Ollama `/api/tags` endpoint.
    ///
    /// Returns the number of models loaded on success, or `Ok(0)` when the
    /// Ollama server is unreachable (so callers can continue gracefully).
    ///
    /// In the current offline implementation this always returns `Ok(0)`.
    /// A production implementation would call:
    /// ```text
    /// GET {server_url}/api/tags  →  { "models": [{ "name": "…", … }] }
    /// ```
    pub fn load_from_server(&mut self) -> Result<usize, MarketplaceError> {
        // Real implementation would use an HTTP client here.
        // We return Ok(0) rather than an error so callers remain functional
        // without a running Ollama instance.
        Ok(0)
    }

    /// Base URL configured for this registry.
    pub fn server_url(&self) -> &str {
        &self.server_url
    }
}

impl ModelRegistry for OllamaRegistry {
    fn list_models(&self) -> Result<Vec<ModelEntry>, MarketplaceError> {
        Ok(self.catalogue.clone())
    }

    fn search(&self, query: &str) -> Result<Vec<ModelEntry>, MarketplaceError> {
        let q = query.to_lowercase();
        let results = self
            .catalogue
            .iter()
            .filter(|entry| {
                entry.id.to_lowercase().contains(&q)
                    || entry.name.to_lowercase().contains(&q)
                    || entry.tags.iter().any(|t| t.to_lowercase().contains(&q))
                    || entry.description.to_lowercase().contains(&q)
            })
            .cloned()
            .collect();
        Ok(results)
    }

    fn get_model(&self, id: &str) -> Result<Option<ModelEntry>, MarketplaceError> {
        Ok(self.catalogue.iter().find(|e| e.id == id).cloned())
    }

    fn source_name(&self) -> &'static str {
        "Ollama"
    }
}

/// Constructs a default `OllamaRegistry` pointing at `http://localhost:11434`
/// with an empty catalogue.
impl Default for OllamaRegistry {
    fn default() -> Self {
        Self::new("http://localhost:11434")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marketplace::{ModelSource, ModelType};

    fn make_test_entry(
        server_url: &str,
        name: &str,
        model_type: ModelType,
        tags: Vec<String>,
        description: &str,
    ) -> ModelEntry {
        ModelEntry {
            id: format!("ollama::{name}"),
            name: name.to_string(),
            source: ModelSource::Ollama {
                server_url: server_url.to_string(),
                name: name.to_string(),
            },
            model_type,
            size_bytes: None,
            tags,
            description: description.to_string(),
            download_url: None,
        }
    }

    fn sample_models(server_url: &str) -> Vec<ModelEntry> {
        vec![
            make_test_entry(
                server_url,
                "llama2",
                ModelType::Chat,
                vec!["llama".to_string(), "chat".to_string(), "7b".to_string()],
                "Meta Llama 2 7B chat model via Ollama.",
            ),
            make_test_entry(
                server_url,
                "mistral",
                ModelType::Chat,
                vec!["mistral".to_string(), "chat".to_string(), "7b".to_string()],
                "Mistral 7B instruct model via Ollama.",
            ),
            make_test_entry(
                server_url,
                "nomic-embed-text",
                ModelType::Embedding,
                vec!["embedding".to_string(), "nomic".to_string()],
                "Nomic text embedding model via Ollama.",
            ),
        ]
    }

    #[test]
    fn test_ollama_registry_with_models() {
        let url = "http://localhost:11434";
        let models = sample_models(url);
        let registry = OllamaRegistry::with_models(url, models);
        let listed = registry.list_models().expect("list_models should succeed");
        assert_eq!(listed.len(), 3);
    }

    #[test]
    fn test_ollama_search_by_tag() {
        let url = "http://localhost:11434";
        let models = sample_models(url);
        let registry = OllamaRegistry::with_models(url, models);
        let results = registry.search("chat").expect("search should succeed");
        assert!(
            !results.is_empty(),
            "search('chat') should return tagged models"
        );
        // All returned models should have "chat" in their tags or name
        for entry in &results {
            let has_chat = entry.tags.iter().any(|t| t.contains("chat"))
                || entry.name.contains("chat")
                || entry.id.contains("chat")
                || entry.description.to_lowercase().contains("chat");
            assert!(
                has_chat,
                "entry '{}' matched chat search but lacks chat marker",
                entry.id
            );
        }
    }

    #[test]
    fn test_ollama_load_from_server_returns_ok() {
        let mut registry = OllamaRegistry::new("http://localhost:11434");
        let count = registry
            .load_from_server()
            .expect("load_from_server should not error");
        assert_eq!(count, 0, "offline load_from_server returns 0");
    }

    #[test]
    fn test_ollama_source_name() {
        let registry = OllamaRegistry::new("http://localhost:11434");
        assert_eq!(registry.source_name(), "Ollama");
    }

    #[test]
    fn test_ollama_server_url_accessor() {
        let registry = OllamaRegistry::new("http://custom:9999");
        assert_eq!(registry.server_url(), "http://custom:9999");
    }
}
