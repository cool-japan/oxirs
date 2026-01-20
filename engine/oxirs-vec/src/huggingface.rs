//! HuggingFace Transformers integration for embedding generation

use crate::{EmbeddableContent, EmbeddingConfig, Vector};
use anyhow::{anyhow, Result};
use scirs2_core::random::Random;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// HuggingFace model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceConfig {
    pub model_name: String,
    pub cache_dir: Option<String>,
    pub device: String,
    pub batch_size: usize,
    pub max_length: usize,
    pub pooling_strategy: PoolingStrategy,
    pub trust_remote_code: bool,
}

/// Pooling strategies for transformer outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PoolingStrategy {
    /// Use `[CLS]` token embedding
    Cls,
    /// Mean pooling of all token embeddings
    Mean,
    /// Max pooling of all token embeddings
    Max,
    /// Weighted mean pooling based on attention weights
    AttentionWeighted,
}

impl Default for HuggingFaceConfig {
    fn default() -> Self {
        Self {
            model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            cache_dir: None,
            device: "cpu".to_string(),
            batch_size: 32,
            max_length: 512,
            pooling_strategy: PoolingStrategy::Mean,
            trust_remote_code: false,
        }
    }
}

/// HuggingFace transformer model for embedding generation
#[derive(Debug)]
pub struct HuggingFaceEmbedder {
    config: HuggingFaceConfig,
    model_cache: HashMap<String, ModelInfo>,
}

/// Model information and metadata
#[derive(Debug, Clone)]
struct ModelInfo {
    dimensions: usize,
    max_sequence_length: usize,
    model_type: String,
    loaded: bool,
}

impl HuggingFaceEmbedder {
    /// Create a new HuggingFace embedder
    pub fn new(config: HuggingFaceConfig) -> Result<Self> {
        Ok(Self {
            config,
            model_cache: HashMap::new(),
        })
    }

    /// Create embedder with default configuration
    pub fn with_default_config() -> Result<Self> {
        Self::new(HuggingFaceConfig::default())
    }

    /// Load a model and prepare it for inference
    pub async fn load_model(&mut self, model_name: &str) -> Result<()> {
        if self.model_cache.contains_key(model_name) {
            return Ok(());
        }

        // Check if model exists in cache directory
        let model_info = self.get_model_info(model_name).await?;
        self.model_cache.insert(model_name.to_string(), model_info);

        tracing::info!("Loaded HuggingFace model: {}", model_name);
        Ok(())
    }

    /// Get model information from HuggingFace Hub
    async fn get_model_info(&self, model_name: &str) -> Result<ModelInfo> {
        // Simulate fetching model info from HuggingFace Hub
        // In a real implementation, this would use the HuggingFace API
        let dimensions = match model_name {
            "sentence-transformers/all-MiniLM-L6-v2" => 384,
            "sentence-transformers/all-mpnet-base-v2" => 768,
            "microsoft/DialoGPT-medium" => 1024,
            "bert-base-uncased" => 768,
            "distilbert-base-uncased" => 768,
            _ => 768, // Default dimension
        };

        Ok(ModelInfo {
            dimensions,
            max_sequence_length: self.config.max_length,
            model_type: "transformer".to_string(),
            loaded: true,
        })
    }

    /// Generate embeddings for a batch of content
    pub async fn embed_batch(&mut self, contents: &[EmbeddableContent]) -> Result<Vec<Vector>> {
        if contents.is_empty() {
            return Ok(vec![]);
        }

        // Load model if not already loaded
        let model_name = self.config.model_name.clone();
        self.load_model(&model_name).await?;

        let model_info = self
            .model_cache
            .get(&self.config.model_name)
            .ok_or_else(|| anyhow!("Model not loaded: {}", self.config.model_name))?;

        let mut embeddings = Vec::with_capacity(contents.len());

        // Process in batches
        for chunk in contents.chunks(self.config.batch_size) {
            let texts: Vec<String> = chunk
                .iter()
                .map(|content| self.content_to_text(content))
                .collect();

            let batch_embeddings = self.generate_embeddings(&texts, model_info).await?;
            embeddings.extend(batch_embeddings);
        }

        Ok(embeddings)
    }

    /// Generate a single embedding
    pub async fn embed(&mut self, content: &EmbeddableContent) -> Result<Vector> {
        let embeddings = self.embed_batch(std::slice::from_ref(content)).await?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("Failed to generate embedding"))
    }

    /// Convert embeddable content to text
    fn content_to_text(&self, content: &EmbeddableContent) -> String {
        match content {
            EmbeddableContent::Text(text) => text.clone(),
            EmbeddableContent::RdfResource {
                uri,
                label,
                description,
                properties,
            } => {
                let mut text_parts = vec![uri.clone()];

                if let Some(label) = label {
                    text_parts.push(label.clone());
                }

                if let Some(desc) = description {
                    text_parts.push(desc.clone());
                }

                for (prop, values) in properties {
                    text_parts.push(format!("{}: {}", prop, values.join(", ")));
                }

                text_parts.join(" ")
            }
            EmbeddableContent::SparqlQuery(query) => query.clone(),
            EmbeddableContent::GraphPattern(pattern) => pattern.clone(),
        }
    }

    /// Generate embeddings using transformer model
    async fn generate_embeddings(
        &self,
        texts: &[String],
        model_info: &ModelInfo,
    ) -> Result<Vec<Vector>> {
        // In a real implementation, this would use actual HuggingFace transformers
        // For now, simulate embedding generation
        let mut embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            let embedding = self.simulate_embedding(text, model_info.dimensions)?;
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    /// Simulate embedding generation (placeholder for actual transformer inference)
    fn simulate_embedding(&self, text: &str, dimensions: usize) -> Result<Vector> {
        // Simple hash-based embedding simulation
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let seed = hasher.finish();

        let mut rng = Random::seed(seed);

        let mut embedding = vec![0.0f32; dimensions];
        for value in embedding.iter_mut().take(dimensions) {
            *value = rng.gen_range(-1.0..1.0); // Random values between -1 and 1
        }

        // Normalize if required
        if matches!(self.config.pooling_strategy, PoolingStrategy::Mean) {
            let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut embedding {
                    *x /= norm;
                }
            }
        }

        Ok(Vector::new(embedding))
    }

    /// Get available models from cache
    pub fn get_cached_models(&self) -> Vec<String> {
        self.model_cache.keys().cloned().collect()
    }

    /// Clear model cache
    pub fn clear_cache(&mut self) {
        self.model_cache.clear();
    }

    /// Get model dimensions
    pub fn get_model_dimensions(&self, model_name: &str) -> Option<usize> {
        self.model_cache.get(model_name).map(|info| info.dimensions)
    }
}

/// HuggingFace model manager for multiple models
#[derive(Debug)]
pub struct HuggingFaceModelManager {
    embedders: HashMap<String, HuggingFaceEmbedder>,
    default_model: String,
}

impl HuggingFaceModelManager {
    /// Create a new model manager
    pub fn new(default_model: String) -> Self {
        Self {
            embedders: HashMap::new(),
            default_model,
        }
    }

    /// Add a model to the manager
    pub fn add_model(&mut self, name: String, config: HuggingFaceConfig) -> Result<()> {
        let embedder = HuggingFaceEmbedder::new(config)?;
        self.embedders.insert(name, embedder);
        Ok(())
    }

    /// Get embeddings using specified model
    pub async fn embed_with_model(
        &mut self,
        model_name: &str,
        content: &EmbeddableContent,
    ) -> Result<Vector> {
        let embedder = self
            .embedders
            .get_mut(model_name)
            .ok_or_else(|| anyhow!("Model not found: {}", model_name))?;
        embedder.embed(content).await
    }

    /// Get embeddings using default model
    pub async fn embed(&mut self, content: &EmbeddableContent) -> Result<Vector> {
        self.embed_with_model(&self.default_model.clone(), content)
            .await
    }

    /// List available models
    pub fn list_models(&self) -> Vec<String> {
        self.embedders.keys().cloned().collect()
    }
}

/// Integration with existing embedding config
impl From<EmbeddingConfig> for HuggingFaceConfig {
    fn from(config: EmbeddingConfig) -> Self {
        Self {
            model_name: config.model_name,
            cache_dir: None,
            device: "cpu".to_string(),
            batch_size: 32,
            max_length: config.max_sequence_length,
            pooling_strategy: if config.normalize {
                PoolingStrategy::Mean
            } else {
                PoolingStrategy::Cls
            },
            trust_remote_code: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_huggingface_embedder_creation() {
        let embedder = HuggingFaceEmbedder::with_default_config();
        assert!(embedder.is_ok());
    }

    #[tokio::test]
    async fn test_model_loading() {
        let mut embedder = HuggingFaceEmbedder::with_default_config().unwrap();
        let result = embedder
            .load_model("sentence-transformers/all-MiniLM-L6-v2")
            .await;
        assert!(result.is_ok());

        let dimensions = embedder.get_model_dimensions("sentence-transformers/all-MiniLM-L6-v2");
        assert_eq!(dimensions, Some(384));
    }

    #[tokio::test]
    async fn test_text_embedding() {
        let mut embedder = HuggingFaceEmbedder::with_default_config().unwrap();
        let content = EmbeddableContent::Text("Hello, world!".to_string());

        let result = embedder.embed(&content).await;
        assert!(result.is_ok());

        let embedding = result.unwrap();
        assert_eq!(embedding.dimensions, 384);
    }

    #[tokio::test]
    async fn test_rdf_resource_embedding() {
        let mut embedder = HuggingFaceEmbedder::with_default_config().unwrap();
        let mut properties = HashMap::new();
        properties.insert("type".to_string(), vec!["Person".to_string()]);

        let content = EmbeddableContent::RdfResource {
            uri: "http://example.org/person/1".to_string(),
            label: Some("John Doe".to_string()),
            description: Some("A person in the knowledge graph".to_string()),
            properties,
        };

        let result = embedder.embed(&content).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_batch_embedding() {
        let mut embedder = HuggingFaceEmbedder::with_default_config().unwrap();
        let contents = vec![
            EmbeddableContent::Text("First text".to_string()),
            EmbeddableContent::Text("Second text".to_string()),
            EmbeddableContent::Text("Third text".to_string()),
        ];

        let result = embedder.embed_batch(&contents).await;
        assert!(result.is_ok());

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 3);
    }

    #[tokio::test]
    async fn test_model_manager() {
        let mut manager = HuggingFaceModelManager::new("default".to_string());
        let config = HuggingFaceConfig::default();

        let result = manager.add_model("default".to_string(), config);
        assert!(result.is_ok());

        let models = manager.list_models();
        assert!(models.contains(&"default".to_string()));
    }

    #[test]
    fn test_config_conversion() {
        let embedding_config = EmbeddingConfig {
            model_name: "test-model".to_string(),
            dimensions: 768,
            max_sequence_length: 512,
            normalize: true,
        };

        let hf_config: HuggingFaceConfig = embedding_config.into();
        assert_eq!(hf_config.model_name, "test-model");
        assert_eq!(hf_config.max_length, 512);
        assert!(matches!(hf_config.pooling_strategy, PoolingStrategy::Mean));
    }

    #[test]
    fn test_pooling_strategies() {
        let strategies = vec![
            PoolingStrategy::Cls,
            PoolingStrategy::Mean,
            PoolingStrategy::Max,
            PoolingStrategy::AttentionWeighted,
        ];

        for strategy in strategies {
            let config = HuggingFaceConfig {
                pooling_strategy: strategy,
                ..Default::default()
            };
            assert!(matches!(
                config.pooling_strategy,
                PoolingStrategy::Cls
                    | PoolingStrategy::Mean
                    | PoolingStrategy::Max
                    | PoolingStrategy::AttentionWeighted
            ));
        }
    }
}
