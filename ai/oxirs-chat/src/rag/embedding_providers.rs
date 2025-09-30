//! Enhanced embedding models and providers
//!
//! Support for multiple embedding providers including OpenAI, HuggingFace,
//! and local models with caching and batch processing.

use super::*;
use reqwest::Client;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Enhanced embedding model that supports multiple providers and caching
pub struct EnhancedEmbeddingModel {
    provider: EmbeddingProvider,
    dimension: usize,
    cache: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    config: EmbeddingConfig,
}

#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub provider_type: EmbeddingProviderType,
    pub model_name: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub cache_size: usize,
    pub batch_size: usize,
    pub timeout_seconds: u64,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider_type: EmbeddingProviderType::Local,
            model_name: "all-MiniLM-L6-v2".to_string(),
            api_key: None,
            base_url: None,
            cache_size: 10000,
            batch_size: 32,
            timeout_seconds: 30,
        }
    }
}

#[derive(Debug, Clone)]
pub enum EmbeddingProviderType {
    OpenAI,
    HuggingFace,
    SentenceTransformers,
    Local,
}

enum EmbeddingProvider {
    OpenAI(OpenAIEmbeddingProvider),
    HuggingFace(HuggingFaceEmbeddingProvider),
    SentenceTransformers(SentenceEmbeddingProvider),
    Local(LocalEmbeddingProvider),
}

impl EnhancedEmbeddingModel {
    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        let dimension = Self::get_model_dimension(&config.model_name);
        let provider = match config.provider_type {
            EmbeddingProviderType::OpenAI => {
                EmbeddingProvider::OpenAI(OpenAIEmbeddingProvider::new(config.clone())?)
            }
            EmbeddingProviderType::HuggingFace => {
                EmbeddingProvider::HuggingFace(HuggingFaceEmbeddingProvider::new(config.clone())?)
            }
            EmbeddingProviderType::SentenceTransformers => EmbeddingProvider::SentenceTransformers(
                SentenceEmbeddingProvider::new(config.clone())?,
            ),
            EmbeddingProviderType::Local => {
                EmbeddingProvider::Local(LocalEmbeddingProvider::new(config.clone())?)
            }
        };

        Ok(Self {
            provider,
            dimension,
            cache: Arc::new(RwLock::new(HashMap::new())),
            config,
        })
    }

    pub async fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::new();
        let mut uncached_texts = Vec::new();
        let mut cache_indices = Vec::new();

        // Check cache first
        {
            let cache = self.cache.read().await;
            for (i, text) in texts.iter().enumerate() {
                if let Some(embedding) = cache.get(text) {
                    results.push((i, embedding.clone()));
                } else {
                    cache_indices.push(i);
                    uncached_texts.push(text.clone());
                }
            }
        }

        // Process uncached texts
        if !uncached_texts.is_empty() {
            let new_embeddings = match &self.provider {
                EmbeddingProvider::OpenAI(provider) => {
                    provider.encode_batch(&uncached_texts).await?
                }
                EmbeddingProvider::HuggingFace(provider) => {
                    provider.encode_batch(&uncached_texts).await?
                }
                EmbeddingProvider::SentenceTransformers(provider) => {
                    provider.encode_batch(&uncached_texts).await?
                }
                EmbeddingProvider::Local(provider) => {
                    provider.encode_batch(&uncached_texts).await?
                }
            };

            // Update cache
            {
                let mut cache = self.cache.write().await;
                for (text, embedding) in uncached_texts.iter().zip(new_embeddings.iter()) {
                    cache.insert(text.clone(), embedding.clone());
                }
            }

            // Add new results
            for (i, embedding) in cache_indices.into_iter().zip(new_embeddings.into_iter()) {
                results.push((i, embedding));
            }
        }

        // Sort by original order and extract embeddings
        results.sort_by_key(|(i, _)| *i);
        Ok(results
            .into_iter()
            .map(|(_, embedding)| embedding)
            .collect())
    }

    fn get_model_dimension(model_name: &str) -> usize {
        match model_name {
            "text-embedding-ada-002" => 1536,
            "all-MiniLM-L6-v2" => 384,
            "all-mpnet-base-v2" => 768,
            "sentence-transformers/all-MiniLM-L6-v2" => 384,
            _ => 384, // Default dimension
        }
    }
}

/// OpenAI embedding provider
struct OpenAIEmbeddingProvider {
    client: Client,
    api_key: String,
    model: String,
}

impl OpenAIEmbeddingProvider {
    fn new(config: EmbeddingConfig) -> Result<Self> {
        let api_key = config
            .api_key
            .ok_or_else(|| anyhow!("OpenAI API key required"))?;

        Ok(Self {
            client: Client::new(),
            api_key,
            model: config.model_name,
        })
    }

    async fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let request_body = serde_json::json!({
            "input": texts,
            "model": self.model
        });

        let response = self
            .client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        let response_json: serde_json::Value = response.json().await?;
        let data = response_json["data"]
            .as_array()
            .ok_or_else(|| anyhow!("Invalid response format"))?;

        let mut embeddings = Vec::new();
        for item in data {
            let embedding = item["embedding"]
                .as_array()
                .ok_or_else(|| anyhow!("Invalid embedding format"))?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }
}

/// HuggingFace embedding provider
struct HuggingFaceEmbeddingProvider {
    client: Client,
    api_key: Option<String>,
    model: String,
}

impl HuggingFaceEmbeddingProvider {
    fn new(config: EmbeddingConfig) -> Result<Self> {
        Ok(Self {
            client: Client::new(),
            api_key: config.api_key,
            model: config.model_name,
        })
    }

    async fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();

        for text in texts {
            let request_body = serde_json::json!({
                "inputs": text
            });

            let mut request_builder = self
                .client
                .post(format!(
                    "https://api-inference.huggingface.co/pipeline/feature-extraction/{}",
                    self.model
                ))
                .header("Content-Type", "application/json")
                .json(&request_body);

            if let Some(ref api_key) = self.api_key {
                request_builder =
                    request_builder.header("Authorization", format!("Bearer {api_key}"));
            }

            let response = request_builder.send().await?;
            let status = response.status();

            if !status.is_success() {
                let error_text = response.text().await?;
                return Err(anyhow!(
                    "HuggingFace API error: {} - {}",
                    status,
                    error_text
                ));
            }

            let embedding: Vec<f32> = response.json().await?;
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }
}

/// Sentence transformers provider (for local models)
struct SentenceEmbeddingProvider {
    model_name: String,
}

impl SentenceEmbeddingProvider {
    fn new(config: EmbeddingConfig) -> Result<Self> {
        Ok(Self {
            model_name: config.model_name,
        })
    }

    async fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // For now, fall back to simple embedding until we have proper sentence-transformers integration
        info!(
            "SentenceEmbeddingProvider: Using fallback implementation for model {}",
            self.model_name
        );

        let simple_model = SimpleEmbeddingModel::new(384); // MiniLM dimension
        let mut embeddings = Vec::new();

        for text in texts {
            let embedding = simple_model.text_to_embedding(text);
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }
}

/// Local embedding provider (fallback)
struct LocalEmbeddingProvider {
    model: SimpleEmbeddingModel,
}

impl LocalEmbeddingProvider {
    fn new(config: EmbeddingConfig) -> Result<Self> {
        let dimension = EnhancedEmbeddingModel::get_model_dimension(&config.model_name);
        Ok(Self {
            model: SimpleEmbeddingModel::new(dimension),
        })
    }

    async fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();
        for text in texts {
            embeddings.push(self.model.text_to_embedding(text));
        }
        Ok(embeddings)
    }
}

/// Simple embedding model implementation (fallback)
pub struct SimpleEmbeddingModel {
    dimension: usize,
}

impl SimpleEmbeddingModel {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Convenience method to embed a single text string
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.text_to_embedding(text))
    }

    fn text_to_embedding(&self, text: &str) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut embedding = vec![0.0; self.dimension];

        // Improved hash-based embedding with TF-IDF-like weighting
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = text_lower
            .split_whitespace()
            .filter(|w| w.len() > 2) // Filter out very short words
            .collect();

        if words.is_empty() {
            return embedding;
        }

        // Simple TF-IDF-like scoring
        let mut word_counts = HashMap::new();
        for word in &words {
            *word_counts.entry(*word).or_insert(0) += 1;
        }

        for (word, count) in word_counts {
            let mut hasher = DefaultHasher::new();
            word.hash(&mut hasher);
            let hash = hasher.finish();

            // Use multiple hash functions to distribute across dimensions
            for i in 0..3 {
                let mut hasher = DefaultHasher::new();
                (word.to_string() + &i.to_string()).hash(&mut hasher);
                let dimension_hash = hasher.finish();
                let dimension_index = (dimension_hash as usize) % self.dimension;

                // TF-IDF-like weighting: frequent words get lower weights
                let tf = count as f32 / words.len() as f32;
                let idf = 1.0 / (1.0 + tf); // Simplified IDF
                let weight = tf * idf;

                // Use hash to determine positive/negative contribution
                let sign = if (hash >> i) & 1 == 0 { 1.0 } else { -1.0 };
                embedding[dimension_index] += sign * weight;
            }
        }

        // Normalize the embedding vector
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in embedding.iter_mut() {
                *x /= norm;
            }
        }

        embedding
    }
}

#[async_trait::async_trait]
impl EmbeddingModel for SimpleEmbeddingModel {
    fn config(&self) -> &ModelConfig {
        // Create a default config since SimpleEmbeddingModel doesn't store one
        use std::sync::OnceLock;
        static DEFAULT_CONFIG: OnceLock<ModelConfig> = OnceLock::new();

        DEFAULT_CONFIG.get_or_init(|| ModelConfig {
            dimensions: 384,
            learning_rate: 0.01,
            l2_reg: 0.0001,
            max_epochs: 1000,
            batch_size: 1000,
            negative_samples: 10,
            seed: None,
            use_gpu: false,
            model_params: HashMap::new(),
        })
    }

    fn model_id(&self) -> &Uuid {
        static DEFAULT_ID: Uuid = Uuid::from_u128(0);
        &DEFAULT_ID
    }

    fn model_type(&self) -> &'static str {
        "SimpleEmbedding"
    }

    fn add_triple(&mut self, _triple: EmbedTriple) -> Result<()> {
        // Simple model doesn't learn from triples
        Ok(())
    }

    async fn train(&mut self, _epochs: Option<usize>) -> Result<TrainingStats> {
        // Simple model doesn't require training
        Ok(TrainingStats {
            epochs_completed: 0,
            final_loss: 0.0,
            training_time_seconds: 0.0,
            convergence_achieved: true,
            loss_history: vec![],
        })
    }

    fn get_entity_embedding(&self, entity: &str) -> Result<EmbedVector> {
        let embedding = self.text_to_embedding(entity);
        Ok(EmbedVector::new(embedding))
    }

    fn getrelation_embedding(&self, relation: &str) -> Result<EmbedVector> {
        let embedding = self.text_to_embedding(relation);
        Ok(EmbedVector::new(embedding))
    }

    fn score_triple(&self, subject: &str, predicate: &str, object: &str) -> Result<f64> {
        let subject_emb = self.text_to_embedding(subject);
        let predicate_emb = self.text_to_embedding(predicate);
        let object_emb = self.text_to_embedding(object);

        // Simple scoring: average similarity
        let score = (similarity::cosine_similarity(&subject_emb, &object_emb)
            + similarity::cosine_similarity(&predicate_emb, &object_emb))
            / 2.0;

        Ok(score as f64)
    }

    fn predict_objects(
        &self,
        subject: &str,
        predicate: &str,
        _k: usize,
    ) -> Result<Vec<(String, f64)>> {
        // Simple implementation: return dummy predictions
        let score = self.score_triple(subject, predicate, "dummy_object")?;
        Ok(vec![("dummy_object".to_string(), score)])
    }

    fn predict_subjects(
        &self,
        predicate: &str,
        object: &str,
        _k: usize,
    ) -> Result<Vec<(String, f64)>> {
        // Simple implementation: return dummy predictions
        let score = self.score_triple("dummy_subject", predicate, object)?;
        Ok(vec![("dummy_subject".to_string(), score)])
    }

    fn predict_relations(
        &self,
        subject: &str,
        object: &str,
        _k: usize,
    ) -> Result<Vec<(String, f64)>> {
        // Simple implementation: return dummy predictions
        let score = self.score_triple(subject, "dummy_predicate", object)?;
        Ok(vec![("dummy_predicate".to_string(), score)])
    }

    fn get_entities(&self) -> Vec<String> {
        // Simple model doesn't track entities
        vec![]
    }

    fn get_relations(&self) -> Vec<String> {
        // Simple model doesn't track relations
        vec![]
    }

    fn get_stats(&self) -> ModelStats {
        ModelStats {
            num_entities: 0,
            num_relations: 0,
            num_triples: 0,
            dimensions: self.dimension,
            is_trained: true,
            model_type: "SimpleEmbedding".to_string(),
            creation_time: chrono::Utc::now(),
            last_training_time: None,
        }
    }

    fn save(&self, _path: &str) -> Result<()> {
        // Simple model doesn't require saving
        Ok(())
    }

    fn load(&mut self, _path: &str) -> Result<()> {
        // Simple model doesn't require loading
        Ok(())
    }

    fn clear(&mut self) {
        // Nothing to clear for simple model
    }

    fn is_trained(&self) -> bool {
        true
    }

    async fn encode(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let embeddings: Vec<Vec<f32>> = texts
            .iter()
            .map(|text| self.text_to_embedding(text))
            .collect();
        Ok(embeddings)
    }
}
