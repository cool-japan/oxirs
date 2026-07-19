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
    /// HuggingFace Inference API token (https://huggingface.co/settings/tokens).
    /// When set and [`HuggingFaceConfig::use_inference_api`] is `true`,
    /// [`HuggingFaceEmbedder`] calls the real HuggingFace Inference API over
    /// HTTP instead of falling back to [`HuggingFaceEmbedder::deterministic_mock_embedding`].
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub api_token: Option<String>,
    /// Whether to call the real HuggingFace Inference API (requires
    /// `api_token` and network access). Defaults to `false`: without an
    /// explicit opt-in, this type never masquerades as real HF inference —
    /// it uses an honestly-labeled deterministic offline mock instead.
    #[serde(default)]
    pub use_inference_api: bool,
    /// Base URL for the HuggingFace Inference API (overridable for testing
    /// against a mock server).
    #[serde(default = "default_inference_api_base_url")]
    pub inference_api_base_url: String,
}

fn default_inference_api_base_url() -> String {
    "https://api-inference.huggingface.co/models".to_string()
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
            api_token: None,
            use_inference_api: false,
            inference_api_base_url: default_inference_api_base_url(),
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

    /// Generate embeddings using the configured backend.
    ///
    /// Calls the real HuggingFace Inference API over HTTP when
    /// [`HuggingFaceConfig::use_inference_api`] is `true` (and an
    /// `api_token` is configured); otherwise falls back to
    /// [`Self::deterministic_mock_embedding`] with a logged warning, since
    /// that fallback is NOT derived from any real transformer inference.
    async fn generate_embeddings(
        &self,
        texts: &[String],
        model_info: &ModelInfo,
    ) -> Result<Vec<Vector>> {
        if self.config.use_inference_api {
            let token = self.config.api_token.as_ref().ok_or_else(|| {
                anyhow!(
                    "HuggingFaceConfig.use_inference_api is true but no api_token is \
                     configured; set api_token (see https://huggingface.co/settings/tokens) \
                     to call the real HuggingFace Inference API"
                )
            })?;
            return self.call_inference_api(texts, token).await;
        }

        tracing::warn!(
            "HuggingFaceEmbedder: use_inference_api is false, so '{}' embeddings are \
             produced by deterministic_mock_embedding — a hash-seeded pseudo-random \
             vector that is NOT related to text semantics and NOT real HuggingFace \
             inference. Set HuggingFaceConfig::use_inference_api=true with an api_token \
             for real embeddings.",
            self.config.model_name
        );

        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            let embedding = self.deterministic_mock_embedding(text, model_info.dimensions)?;
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    /// Call the real HuggingFace Inference API's feature-extraction endpoint.
    async fn call_inference_api(&self, texts: &[String], token: &str) -> Result<Vec<Vector>> {
        let url = format!(
            "{}/{}",
            self.config.inference_api_base_url, self.config.model_name
        );

        let client = reqwest::Client::new();
        let response = client
            .post(&url)
            .bearer_auth(token)
            .json(&serde_json::json!({
                "inputs": texts,
                "options": { "wait_for_model": true },
            }))
            .send()
            .await
            .map_err(|e| anyhow!("HuggingFace Inference API request to {} failed: {}", url, e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!(
                "HuggingFace Inference API returned {} for {}: {}",
                status,
                url,
                body
            ));
        }

        let raw: serde_json::Value = response.json().await.map_err(|e| {
            anyhow!(
                "Failed to parse HuggingFace Inference API response as JSON: {}",
                e
            )
        })?;

        Self::parse_inference_response(&raw, texts.len())
    }

    /// Parse a HuggingFace feature-extraction response into per-text
    /// vectors. Handles both already-pooled (`[[f32; d]; n]`) and
    /// token-level (`[[[f32; d]; tokens]; n]`, mean-pooled here) response
    /// shapes, since different models' pipeline tags return either.
    fn parse_inference_response(
        raw: &serde_json::Value,
        expected_count: usize,
    ) -> Result<Vec<Vector>> {
        let items = raw.as_array().ok_or_else(|| {
            anyhow!("Unexpected HuggingFace Inference API response shape: expected a JSON array")
        })?;

        let vectors = items
            .iter()
            .map(Self::parse_single_embedding)
            .collect::<Result<Vec<Vector>>>()?;

        if vectors.len() != expected_count {
            return Err(anyhow!(
                "HuggingFace Inference API returned {} embeddings for {} input texts",
                vectors.len(),
                expected_count
            ));
        }

        Ok(vectors)
    }

    /// Parse one response item, which is either an already-pooled flat
    /// array of numbers, or a nested array of per-token vectors (mean-pooled
    /// here into a single sentence vector).
    fn parse_single_embedding(value: &serde_json::Value) -> Result<Vector> {
        let arr = value
            .as_array()
            .ok_or_else(|| anyhow!("Unexpected embedding item shape in HF response"))?;

        if arr.iter().all(|v| v.is_number()) {
            let values: Vec<f32> = arr
                .iter()
                .map(|n| n.as_f64().unwrap_or(0.0) as f32)
                .collect();
            return Ok(Vector::new(values));
        }

        // Token-level response: mean-pool across tokens.
        let mut sum: Option<Vec<f32>> = None;
        let mut count = 0usize;
        for token in arr {
            let token_values: Vec<f32> = token
                .as_array()
                .ok_or_else(|| anyhow!("Unexpected token embedding shape in HF response"))?
                .iter()
                .map(|n| n.as_f64().unwrap_or(0.0) as f32)
                .collect();
            sum = Some(match sum {
                None => token_values,
                Some(mut acc) => {
                    for (a, b) in acc.iter_mut().zip(token_values.iter()) {
                        *a += b;
                    }
                    acc
                }
            });
            count += 1;
        }

        let mut pooled =
            sum.ok_or_else(|| anyhow!("Empty token embedding array in HF response"))?;
        if count > 0 {
            for v in pooled.iter_mut() {
                *v /= count as f32;
            }
        }
        Ok(Vector::new(pooled))
    }

    /// Deterministic, hash-seeded pseudo-random embedding.
    ///
    /// **This is NOT real HuggingFace inference.** It is an offline mock
    /// used only when [`HuggingFaceConfig::use_inference_api`] is `false`
    /// (the default), so tests and offline development don't require
    /// network access. The output is unrelated to the text's semantics —
    /// callers that need real embeddings must set `use_inference_api = true`
    /// with a valid `api_token`.
    fn deterministic_mock_embedding(&self, text: &str, dimensions: usize) -> Result<Vector> {
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
            api_token: None,
            use_inference_api: false,
            inference_api_base_url: default_inference_api_base_url(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    #[tokio::test]
    async fn test_huggingface_embedder_creation() {
        let embedder = HuggingFaceEmbedder::with_default_config();
        assert!(embedder.is_ok());
    }

    #[tokio::test]
    async fn test_model_loading() -> Result<()> {
        let mut embedder = HuggingFaceEmbedder::with_default_config()?;
        let result = embedder
            .load_model("sentence-transformers/all-MiniLM-L6-v2")
            .await;
        assert!(result.is_ok());

        let dimensions = embedder.get_model_dimensions("sentence-transformers/all-MiniLM-L6-v2");
        assert_eq!(dimensions, Some(384));
        Ok(())
    }

    #[tokio::test]
    async fn test_text_embedding() -> Result<()> {
        let mut embedder = HuggingFaceEmbedder::with_default_config()?;
        let content = EmbeddableContent::Text("Hello, world!".to_string());

        let result = embedder.embed(&content).await;
        assert!(result.is_ok());

        let embedding = result?;
        assert_eq!(embedding.dimensions, 384);
        Ok(())
    }

    #[tokio::test]
    async fn test_rdf_resource_embedding() -> Result<()> {
        let mut embedder = HuggingFaceEmbedder::with_default_config()?;
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
        Ok(())
    }

    #[tokio::test]
    async fn test_batch_embedding() -> Result<()> {
        let mut embedder = HuggingFaceEmbedder::with_default_config()?;
        let contents = vec![
            EmbeddableContent::Text("First text".to_string()),
            EmbeddableContent::Text("Second text".to_string()),
            EmbeddableContent::Text("Third text".to_string()),
        ];

        let result = embedder.embed_batch(&contents).await;
        assert!(result.is_ok());

        let embeddings = result?;
        assert_eq!(embeddings.len(), 3);
        Ok(())
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

    /// Regression test for the P1 finding: `use_inference_api` requires an
    /// `api_token`; without one it must fail loudly instead of silently
    /// falling back to the mock while `use_inference_api` is `true`.
    #[tokio::test]
    async fn test_use_inference_api_without_token_errors() {
        let config = HuggingFaceConfig {
            use_inference_api: true,
            api_token: None,
            ..Default::default()
        };
        let mut embedder = HuggingFaceEmbedder::new(config).expect("embedder should construct");
        let content = EmbeddableContent::Text("hello".to_string());
        let result = embedder.embed(&content).await;
        assert!(result.is_err(), "missing api_token must be a hard error");
    }

    /// Regression test for the P1 finding: `HuggingFaceEmbedder` used to
    /// silently return hash-noise labeled as if it were a transformer
    /// embedding. The default config (no `use_inference_api`) must still
    /// work offline via the explicitly-named `deterministic_mock_embedding`,
    /// but must never be mistaken for real inference by identical inputs
    /// producing identical (deterministic) — not text-semantic — output.
    #[tokio::test]
    async fn test_default_config_uses_deterministic_mock_not_real_api() -> Result<()> {
        let mut embedder = HuggingFaceEmbedder::with_default_config()?;
        assert!(!embedder.config.use_inference_api);

        let a = embedder
            .embed(&EmbeddableContent::Text("hello world".to_string()))
            .await?;
        let b = embedder
            .embed(&EmbeddableContent::Text("hello world".to_string()))
            .await?;
        // Deterministic (same hash seed) but NOT claiming semantic meaning.
        assert_eq!(a.as_f32(), b.as_f32());
        Ok(())
    }

    /// `parse_single_embedding` must correctly handle both the pooled
    /// (`[f32; d]`) and token-level (`[[f32; d]; tokens]`, mean-pooled)
    /// response shapes returned by different HF feature-extraction models.
    #[test]
    fn test_parse_single_embedding_pooled_and_token_level() -> Result<()> {
        let pooled = serde_json::json!([0.1, 0.2, 0.3]);
        let pooled_vec = HuggingFaceEmbedder::parse_single_embedding(&pooled)?;
        assert_eq!(pooled_vec.as_f32(), vec![0.1, 0.2, 0.3]);

        // Two tokens: [1.0, 1.0] and [3.0, 3.0] -> mean = [2.0, 2.0]
        let token_level = serde_json::json!([[1.0, 1.0], [3.0, 3.0]]);
        let pooled_from_tokens = HuggingFaceEmbedder::parse_single_embedding(&token_level)?;
        assert_eq!(pooled_from_tokens.as_f32(), vec![2.0, 2.0]);
        Ok(())
    }

    /// `parse_inference_response` must error (not silently truncate/pad) on
    /// a count mismatch between the number of input texts and returned
    /// embeddings.
    #[test]
    fn test_parse_inference_response_count_mismatch_errors() {
        let raw = serde_json::json!([[0.1, 0.2], [0.3, 0.4]]);
        let result = HuggingFaceEmbedder::parse_inference_response(&raw, 3);
        assert!(result.is_err());
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
