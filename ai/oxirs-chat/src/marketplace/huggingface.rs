//! HuggingFace Hub model registry (offline catalogue).
//!
//! In production this would call `api.huggingface.co`; the current
//! implementation ships a curated offline catalogue so the registry is fully
//! usable without network access.

use crate::marketplace::{MarketplaceError, ModelEntry, ModelRegistry, ModelSource, ModelType};

/// A model registry seeded from a curated HuggingFace Hub catalogue.
///
/// The catalogue is populated at construction time and is never mutated
/// afterwards, so `HuggingFaceRegistry` is cheaply cloneable and share-safe.
pub struct HuggingFaceRegistry {
    catalogue: Vec<ModelEntry>,
}

impl HuggingFaceRegistry {
    /// Create a new registry pre-seeded with well-known HuggingFace models.
    pub fn new() -> Self {
        Self {
            catalogue: Self::default_catalogue(),
        }
    }

    /// Create a registry with a custom model list (useful for testing).
    pub fn with_models(models: Vec<ModelEntry>) -> Self {
        Self { catalogue: models }
    }

    /// Returns a curated offline catalogue of popular HuggingFace models.
    fn default_catalogue() -> Vec<ModelEntry> {
        vec![
            ModelEntry {
                id: "meta-llama/Llama-2-7b-chat-hf".to_string(),
                name: "Llama 2 7B Chat".to_string(),
                source: ModelSource::HuggingFaceHub {
                    repo_id: "meta-llama/Llama-2-7b-chat-hf".to_string(),
                },
                model_type: ModelType::Chat,
                size_bytes: Some(13_476_839_424),
                tags: vec![
                    "llama".to_string(),
                    "chat".to_string(),
                    "7b".to_string(),
                    "meta".to_string(),
                ],
                description: "Meta's Llama 2 7B fine-tuned for dialogue use cases.".to_string(),
                download_url: Some(
                    "https://huggingface.co/meta-llama/Llama-2-7b-chat-hf".to_string(),
                ),
            },
            ModelEntry {
                id: "meta-llama/Llama-2-13b-chat-hf".to_string(),
                name: "Llama 2 13B Chat".to_string(),
                source: ModelSource::HuggingFaceHub {
                    repo_id: "meta-llama/Llama-2-13b-chat-hf".to_string(),
                },
                model_type: ModelType::Chat,
                size_bytes: Some(26_033_864_704),
                tags: vec![
                    "llama".to_string(),
                    "chat".to_string(),
                    "13b".to_string(),
                    "meta".to_string(),
                ],
                description: "Meta's Llama 2 13B fine-tuned for dialogue use cases.".to_string(),
                download_url: Some(
                    "https://huggingface.co/meta-llama/Llama-2-13b-chat-hf".to_string(),
                ),
            },
            ModelEntry {
                id: "mistralai/Mistral-7B-v0.1".to_string(),
                name: "Mistral 7B v0.1".to_string(),
                source: ModelSource::HuggingFaceHub {
                    repo_id: "mistralai/Mistral-7B-v0.1".to_string(),
                },
                model_type: ModelType::Completion,
                size_bytes: Some(14_484_664_320),
                tags: vec![
                    "mistral".to_string(),
                    "7b".to_string(),
                    "completion".to_string(),
                ],
                description: "Mistral AI's 7B parameter base model.".to_string(),
                download_url: Some("https://huggingface.co/mistralai/Mistral-7B-v0.1".to_string()),
            },
            ModelEntry {
                id: "mistralai/Mistral-7B-Instruct-v0.2".to_string(),
                name: "Mistral 7B Instruct v0.2".to_string(),
                source: ModelSource::HuggingFaceHub {
                    repo_id: "mistralai/Mistral-7B-Instruct-v0.2".to_string(),
                },
                model_type: ModelType::Chat,
                size_bytes: Some(14_484_664_320),
                tags: vec![
                    "mistral".to_string(),
                    "chat".to_string(),
                    "instruct".to_string(),
                    "7b".to_string(),
                ],
                description: "Mistral AI's 7B instruct-tuned model (v0.2).".to_string(),
                download_url: Some(
                    "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2".to_string(),
                ),
            },
            ModelEntry {
                id: "tiiuae/falcon-7b".to_string(),
                name: "Falcon 7B".to_string(),
                source: ModelSource::HuggingFaceHub {
                    repo_id: "tiiuae/falcon-7b".to_string(),
                },
                model_type: ModelType::Completion,
                size_bytes: Some(14_175_354_880),
                tags: vec!["falcon".to_string(), "7b".to_string(), "tii".to_string()],
                description: "Technology Innovation Institute's Falcon 7B base model.".to_string(),
                download_url: Some("https://huggingface.co/tiiuae/falcon-7b".to_string()),
            },
            ModelEntry {
                id: "google/flan-t5-large".to_string(),
                name: "Flan-T5 Large".to_string(),
                source: ModelSource::HuggingFaceHub {
                    repo_id: "google/flan-t5-large".to_string(),
                },
                model_type: ModelType::Chat,
                size_bytes: Some(1_538_000_000),
                tags: vec![
                    "flan".to_string(),
                    "t5".to_string(),
                    "google".to_string(),
                    "instruction".to_string(),
                ],
                description: "Google's instruction-tuned Flan-T5 large model.".to_string(),
                download_url: Some("https://huggingface.co/google/flan-t5-large".to_string()),
            },
            ModelEntry {
                id: "BAAI/bge-large-en-v1.5".to_string(),
                name: "BGE Large EN v1.5".to_string(),
                source: ModelSource::HuggingFaceHub {
                    repo_id: "BAAI/bge-large-en-v1.5".to_string(),
                },
                model_type: ModelType::Embedding,
                size_bytes: Some(670_000_000),
                tags: vec![
                    "embedding".to_string(),
                    "bge".to_string(),
                    "baai".to_string(),
                    "english".to_string(),
                ],
                description: "BAAI's BGE large English embedding model v1.5.".to_string(),
                download_url: Some("https://huggingface.co/BAAI/bge-large-en-v1.5".to_string()),
            },
            ModelEntry {
                id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                name: "all-MiniLM-L6-v2".to_string(),
                source: ModelSource::HuggingFaceHub {
                    repo_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                },
                model_type: ModelType::Embedding,
                size_bytes: Some(91_000_000),
                tags: vec![
                    "embedding".to_string(),
                    "sentence-transformers".to_string(),
                    "minilm".to_string(),
                ],
                description: "Compact sentence embedding model, great for semantic search."
                    .to_string(),
                download_url: Some(
                    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2".to_string(),
                ),
            },
            ModelEntry {
                id: "bigscience/bloom-7b1".to_string(),
                name: "BLOOM 7B1".to_string(),
                source: ModelSource::HuggingFaceHub {
                    repo_id: "bigscience/bloom-7b1".to_string(),
                },
                model_type: ModelType::Completion,
                size_bytes: Some(13_960_000_000),
                tags: vec![
                    "bloom".to_string(),
                    "bigscience".to_string(),
                    "multilingual".to_string(),
                    "7b".to_string(),
                ],
                description: "BigScience BLOOM 7.1B multilingual language model.".to_string(),
                download_url: Some("https://huggingface.co/bigscience/bloom-7b1".to_string()),
            },
            ModelEntry {
                id: "EleutherAI/gpt-j-6b".to_string(),
                name: "GPT-J 6B".to_string(),
                source: ModelSource::HuggingFaceHub {
                    repo_id: "EleutherAI/gpt-j-6b".to_string(),
                },
                model_type: ModelType::Completion,
                size_bytes: Some(12_062_613_504),
                tags: vec![
                    "gpt-j".to_string(),
                    "eleutherai".to_string(),
                    "6b".to_string(),
                ],
                description: "EleutherAI's GPT-J 6B open-source language model.".to_string(),
                download_url: Some("https://huggingface.co/EleutherAI/gpt-j-6b".to_string()),
            },
            ModelEntry {
                id: "microsoft/phi-2".to_string(),
                name: "Phi-2".to_string(),
                source: ModelSource::HuggingFaceHub {
                    repo_id: "microsoft/phi-2".to_string(),
                },
                model_type: ModelType::Completion,
                size_bytes: Some(5_559_296_000),
                tags: vec![
                    "phi".to_string(),
                    "microsoft".to_string(),
                    "2.7b".to_string(),
                    "small".to_string(),
                ],
                description: "Microsoft's Phi-2 2.7B parameter language model.".to_string(),
                download_url: Some("https://huggingface.co/microsoft/phi-2".to_string()),
            },
            ModelEntry {
                id: "openchat/openchat-3.5-0106".to_string(),
                name: "OpenChat 3.5".to_string(),
                source: ModelSource::HuggingFaceHub {
                    repo_id: "openchat/openchat-3.5-0106".to_string(),
                },
                model_type: ModelType::Chat,
                size_bytes: Some(14_484_664_320),
                tags: vec![
                    "openchat".to_string(),
                    "chat".to_string(),
                    "7b".to_string(),
                    "mistral-based".to_string(),
                ],
                description: "OpenChat 3.5: open-source 7B chat model with C-RLFT training."
                    .to_string(),
                download_url: Some("https://huggingface.co/openchat/openchat-3.5-0106".to_string()),
            },
        ]
    }
}

impl Default for HuggingFaceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelRegistry for HuggingFaceRegistry {
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
        "HuggingFace Hub"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huggingface_registry_nonempty() {
        let registry = HuggingFaceRegistry::new();
        let models = registry.list_models().expect("list_models should succeed");
        assert!(
            !models.is_empty(),
            "HuggingFace catalogue should be non-empty"
        );
        assert!(models.len() >= 10, "expected at least 10 curated models");
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
        assert_eq!(
            model.expect("model exists").id,
            "meta-llama/Llama-2-7b-chat-hf"
        );
    }

    #[test]
    fn test_huggingface_get_unknown_id_returns_none() {
        let registry = HuggingFaceRegistry::new();
        let model = registry
            .get_model("no-such/model-xyz")
            .expect("get_model should not error");
        assert!(model.is_none(), "unknown id should return None");
    }

    #[test]
    fn test_huggingface_source_name() {
        let registry = HuggingFaceRegistry::new();
        assert_eq!(registry.source_name(), "HuggingFace Hub");
    }

    #[test]
    fn test_huggingface_with_models_custom() {
        let custom = vec![ModelEntry {
            id: "test/custom-model".to_string(),
            name: "Custom Test Model".to_string(),
            source: ModelSource::HuggingFaceHub {
                repo_id: "test/custom-model".to_string(),
            },
            model_type: ModelType::Chat,
            size_bytes: None,
            tags: vec!["test".to_string()],
            description: "A test model.".to_string(),
            download_url: None,
        }];
        let registry = HuggingFaceRegistry::with_models(custom);
        let models = registry.list_models().expect("list_models should succeed");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "test/custom-model");
    }
}
