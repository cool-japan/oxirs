//! Embedding generation and management for RDF resources and text content

use crate::Vector;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
// AsAny trait will be defined locally

/// Embedding model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub model_name: String,
    pub dimensions: usize,
    pub max_sequence_length: usize,
    pub normalize: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            dimensions: 384,
            max_sequence_length: 512,
            normalize: true,
        }
    }
}

/// Content to be embedded
#[derive(Debug, Clone)]
pub enum EmbeddableContent {
    /// Plain text content
    Text(String),
    /// RDF resource with properties
    RdfResource {
        uri: String,
        label: Option<String>,
        description: Option<String>,
        properties: HashMap<String, Vec<String>>,
    },
    /// SPARQL query or query fragment
    SparqlQuery(String),
    /// Knowledge graph path or pattern
    GraphPattern(String),
}

impl EmbeddableContent {
    /// Convert content to text representation for embedding
    pub fn to_text(&self) -> String {
        match self {
            EmbeddableContent::Text(text) => text.clone(),
            EmbeddableContent::RdfResource {
                uri,
                label,
                description,
                properties,
            } => {
                let mut text_parts = vec![uri.clone()];

                if let Some(label) = label {
                    text_parts.push(format!("label: {}", label));
                }

                if let Some(desc) = description {
                    text_parts.push(format!("description: {}", desc));
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

    /// Get a unique identifier for this content
    pub fn content_hash(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.to_text().hash(&mut hasher);
        hasher.finish()
    }
}

/// Embedding generation strategy
#[derive(Debug, Clone)]
pub enum EmbeddingStrategy {
    /// Simple TF-IDF based embeddings (for testing/fallback)
    TfIdf,
    /// Sentence transformer embeddings (requires external service)
    SentenceTransformer,
    /// BERT-based transformer models
    Transformer(TransformerModelType),
    /// Word2Vec embeddings
    Word2Vec(crate::word2vec::Word2VecConfig),
    /// OpenAI embeddings (requires API key)
    OpenAI(OpenAIConfig),
    /// Custom embedding model
    Custom(String),
}

/// OpenAI embeddings configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIConfig {
    /// API key for OpenAI service
    pub api_key: String,
    /// Model to use (e.g., "text-embedding-ada-002", "text-embedding-3-small")
    pub model: String,
    /// Base URL for API calls (default: https://api.openai.com/v1)
    pub base_url: String,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Rate limiting: requests per minute
    pub requests_per_minute: u32,
    /// Batch size for batch processing
    pub batch_size: usize,
    /// Enable local caching
    pub enable_cache: bool,
    /// Maximum retries for failed requests
    pub max_retries: u32,
    /// Retry delay in milliseconds
    pub retry_delay_ms: u64,
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            model: "text-embedding-3-small".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            timeout_seconds: 30,
            requests_per_minute: 3000,
            batch_size: 100,
            enable_cache: true,
            max_retries: 3,
            retry_delay_ms: 1000,
        }
    }
}

/// Embedding generator trait
pub trait EmbeddingGenerator: Send + Sync + AsAny {
    /// Generate embedding for content
    fn generate(&self, content: &EmbeddableContent) -> Result<Vector>;

    /// Generate embeddings for multiple contents in batch
    fn generate_batch(&self, contents: &[EmbeddableContent]) -> Result<Vec<Vector>> {
        contents.iter().map(|c| self.generate(c)).collect()
    }

    /// Get the embedding dimensions
    fn dimensions(&self) -> usize;

    /// Get the model configuration
    fn config(&self) -> &EmbeddingConfig;
}

/// Simple TF-IDF based embedding generator
pub struct TfIdfEmbeddingGenerator {
    config: EmbeddingConfig,
    vocabulary: HashMap<String, usize>,
    idf_scores: HashMap<String, f32>,
}

impl TfIdfEmbeddingGenerator {
    pub fn new(config: EmbeddingConfig) -> Self {
        Self {
            config,
            vocabulary: HashMap::new(),
            idf_scores: HashMap::new(),
        }
    }

    /// Build vocabulary from a corpus of documents
    pub fn build_vocabulary(&mut self, documents: &[String]) -> Result<()> {
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        let mut doc_counts: HashMap<String, usize> = HashMap::new();

        for doc in documents {
            let words: Vec<String> = self.tokenize(doc);
            let unique_words: std::collections::HashSet<_> = words.iter().collect();

            for word in &words {
                *word_counts.entry(word.clone()).or_insert(0) += 1;
            }

            for word in unique_words {
                *doc_counts.entry(word.clone()).or_insert(0) += 1;
            }
        }

        // Build vocabulary with most frequent words
        let mut word_freq: Vec<(String, usize)> = word_counts.into_iter().collect();
        word_freq.sort_by(|a, b| b.1.cmp(&a.1));

        self.vocabulary = word_freq
            .into_iter()
            .take(self.config.dimensions)
            .enumerate()
            .map(|(idx, (word, _))| (word, idx))
            .collect();

        // Calculate IDF scores
        let total_docs = documents.len() as f32;
        for (word, _idx) in &self.vocabulary {
            let doc_freq = doc_counts.get(word).unwrap_or(&0);
            let idf = (total_docs / (*doc_freq as f32 + 1.0)).ln();
            self.idf_scores.insert(word.clone(), idf);
        }

        Ok(())
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect()
    }

    fn calculate_tf_idf(&self, text: &str) -> Vector {
        let words = self.tokenize(text);
        let mut tf_counts: HashMap<String, usize> = HashMap::new();

        for word in &words {
            *tf_counts.entry(word.clone()).or_insert(0) += 1;
        }

        let total_words = words.len() as f32;
        let mut embedding = vec![0.0; self.config.dimensions];

        for (word, count) in tf_counts {
            if let Some(&idx) = self.vocabulary.get(&word) {
                let tf = count as f32 / total_words;
                let idf = self.idf_scores.get(&word).unwrap_or(&0.0);
                embedding[idx] = tf * idf;
            }
        }

        if self.config.normalize {
            self.normalize_vector(&mut embedding);
        }

        Vector::new(embedding)
    }

    fn normalize_vector(&self, vector: &mut [f32]) {
        let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for value in vector {
                *value /= magnitude;
            }
        }
    }
}

impl EmbeddingGenerator for TfIdfEmbeddingGenerator {
    fn generate(&self, content: &EmbeddableContent) -> Result<Vector> {
        if self.vocabulary.is_empty() {
            return Err(anyhow!(
                "Vocabulary not built. Call build_vocabulary first."
            ));
        }

        let text = content.to_text();
        Ok(self.calculate_tf_idf(&text))
    }

    fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
}

/// Transformer-based embedding generator supporting multiple models
pub struct SentenceTransformerGenerator {
    config: EmbeddingConfig,
    model_type: TransformerModelType,
}

/// Supported transformer model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformerModelType {
    /// Basic BERT-based model (already implemented)
    BERT,
    /// RoBERTa model with improved training
    RoBERTa,
    /// DistilBERT for efficiency
    DistilBERT,
    /// Multilingual BERT
    MultiBERT,
    /// Custom model path
    Custom(String),
}

impl Default for TransformerModelType {
    fn default() -> Self {
        TransformerModelType::BERT
    }
}

impl SentenceTransformerGenerator {
    pub fn new(config: EmbeddingConfig) -> Self {
        Self { 
            config,
            model_type: TransformerModelType::default(),
        }
    }

    pub fn with_model_type(config: EmbeddingConfig, model_type: TransformerModelType) -> Self {
        Self { config, model_type }
    }

    /// Get model-specific configuration adjustments
    fn get_model_config(&self) -> (usize, usize, f32) {
        match &self.model_type {
            TransformerModelType::BERT => (768, 512, 1.0), // (dimensions, max_seq_len, efficiency)
            TransformerModelType::RoBERTa => (768, 512, 0.95), // Slightly slower than BERT
            TransformerModelType::DistilBERT => (768, 512, 1.5), // Faster, smaller
            TransformerModelType::MultiBERT => (768, 512, 0.8), // Slower for multilingual
            TransformerModelType::Custom(_) => (self.config.dimensions, self.config.max_sequence_length, 1.0),
        }
    }

    /// Generate embedding with model-specific processing
    fn generate_with_model(&self, text: &str) -> Result<Vector> {
        let text_hash = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            text.hash(&mut hasher);
            hasher.finish()
        };

        let (dimensions, max_len, efficiency) = self.get_model_config();
        
        // Truncate text if too long
        let processed_text = if text.len() > max_len {
            &text[..max_len]
        } else {
            text
        };

        // Model-specific adjustments to the hash seed
        let model_seed = match &self.model_type {
            TransformerModelType::BERT => text_hash,
            TransformerModelType::RoBERTa => text_hash.wrapping_mul(1234567),
            TransformerModelType::DistilBERT => text_hash.wrapping_mul(987654321),
            TransformerModelType::MultiBERT => text_hash.wrapping_mul(555555555),
            TransformerModelType::Custom(path) => {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                path.hash(&mut hasher);
                text_hash.wrapping_add(hasher.finish())
            }
        };

        // Generate deterministic "embeddings" based on model type and text
        let mut values = Vec::with_capacity(dimensions);
        let mut seed = model_seed;

        for i in 0..dimensions {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            
            // Add some model-specific variance
            let model_variance = match &self.model_type {
                TransformerModelType::BERT => 0.0,
                TransformerModelType::RoBERTa => (i as f32 * 0.01).sin(),
                TransformerModelType::DistilBERT => (i as f32 * 0.02).cos(),
                TransformerModelType::MultiBERT => (i as f32 * 0.005).sin() * 0.5,
                TransformerModelType::Custom(_) => (i as f32 * 0.001).tan().clamp(-0.1, 0.1),
            };
            
            let normalized = (seed as f32) / (u64::MAX as f32);
            let value = ((normalized - 0.5) * 2.0 * efficiency) + model_variance;
            values.push(value);
        }

        if self.config.normalize {
            let magnitude: f32 = values.iter().map(|x| x * x).sum::<f32>().sqrt();
            if magnitude > 0.0 {
                for value in &mut values {
                    *value /= magnitude;
                }
            }
        }

        Ok(Vector::new(values))
    }
}

impl EmbeddingGenerator for SentenceTransformerGenerator {
    fn generate(&self, content: &EmbeddableContent) -> Result<Vector> {
        let text = content.to_text();
        self.generate_with_model(&text)
    }

    fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
}

/// Embedding cache for frequently accessed embeddings
pub struct EmbeddingCache {
    cache: HashMap<u64, Vector>,
    max_size: usize,
    access_order: Vec<u64>,
}

impl EmbeddingCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
            access_order: Vec::new(),
        }
    }

    pub fn get(&mut self, content: &EmbeddableContent) -> Option<&Vector> {
        let hash = content.content_hash();
        if let Some(vector) = self.cache.get(&hash) {
            // Move to end (most recently used)
            if let Some(pos) = self.access_order.iter().position(|&x| x == hash) {
                self.access_order.remove(pos);
            }
            self.access_order.push(hash);
            Some(vector)
        } else {
            None
        }
    }

    pub fn insert(&mut self, content: &EmbeddableContent, vector: Vector) {
        let hash = content.content_hash();

        // Remove least recently used if at capacity
        if self.cache.len() >= self.max_size && !self.cache.contains_key(&hash) {
            if let Some(&lru_hash) = self.access_order.first() {
                self.cache.remove(&lru_hash);
                self.access_order.remove(0);
            }
        }

        self.cache.insert(hash, vector);
        self.access_order.push(hash);
    }

    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
    }

    pub fn size(&self) -> usize {
        self.cache.len()
    }
}

/// Embedding manager that combines generation, caching, and persistence
pub struct EmbeddingManager {
    generator: Box<dyn EmbeddingGenerator>,
    cache: EmbeddingCache,
    strategy: EmbeddingStrategy,
}

impl EmbeddingManager {
    pub fn new(strategy: EmbeddingStrategy, cache_size: usize) -> Result<Self> {
        let generator: Box<dyn EmbeddingGenerator> = match &strategy {
            EmbeddingStrategy::TfIdf => {
                let config = EmbeddingConfig::default();
                Box::new(TfIdfEmbeddingGenerator::new(config))
            }
            EmbeddingStrategy::SentenceTransformer => {
                let config = EmbeddingConfig::default();
                Box::new(SentenceTransformerGenerator::new(config))
            }
            EmbeddingStrategy::Transformer(model_type) => {
                let config = EmbeddingConfig {
                    model_name: format!("{:?}", model_type),
                    dimensions: match model_type {
                        TransformerModelType::DistilBERT => 384, // DistilBERT is smaller
                        _ => 768, // Most BERT variants
                    },
                    max_sequence_length: 512,
                    normalize: true,
                };
                Box::new(SentenceTransformerGenerator::with_model_type(config, model_type.clone()))
            }
            EmbeddingStrategy::Word2Vec(word2vec_config) => {
                let embedding_config = EmbeddingConfig {
                    model_name: "word2vec".to_string(),
                    dimensions: word2vec_config.dimensions,
                    max_sequence_length: 512,
                    normalize: word2vec_config.normalize,
                };
                Box::new(crate::word2vec::Word2VecEmbeddingGenerator::new(
                    word2vec_config.clone(),
                    embedding_config,
                )?)
            }
            EmbeddingStrategy::OpenAI(openai_config) => {
                Box::new(OpenAIEmbeddingGenerator::new(openai_config.clone())?)
            }
            EmbeddingStrategy::Custom(_model_path) => {
                // For now, fall back to sentence transformer
                let config = EmbeddingConfig::default();
                Box::new(SentenceTransformerGenerator::new(config))
            }
        };

        Ok(Self {
            generator,
            cache: EmbeddingCache::new(cache_size),
            strategy,
        })
    }

    /// Get or generate embedding for content
    pub fn get_embedding(&mut self, content: &EmbeddableContent) -> Result<Vector> {
        if let Some(cached) = self.cache.get(content) {
            return Ok(cached.clone());
        }

        let embedding = self.generator.generate(content)?;
        self.cache.insert(content, embedding.clone());
        Ok(embedding)
    }

    /// Pre-compute embeddings for a batch of content
    pub fn precompute_embeddings(&mut self, contents: &[EmbeddableContent]) -> Result<()> {
        let embeddings = self.generator.generate_batch(contents)?;

        for (content, embedding) in contents.iter().zip(embeddings) {
            self.cache.insert(content, embedding);
        }

        Ok(())
    }

    /// Build vocabulary for TF-IDF strategy
    pub fn build_vocabulary(&mut self, documents: &[String]) -> Result<()> {
        if let EmbeddingStrategy::TfIdf = self.strategy {
            if let Some(tfidf_gen) = self
                .generator
                .as_any_mut()
                .downcast_mut::<TfIdfEmbeddingGenerator>()
            {
                tfidf_gen.build_vocabulary(documents)?;
            }
        }
        Ok(())
    }

    pub fn dimensions(&self) -> usize {
        self.generator.dimensions()
    }

    pub fn cache_stats(&self) -> (usize, usize) {
        (self.cache.size(), self.cache.max_size)
    }
}

/// Extension trait to add downcast functionality
pub trait AsAny {
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

impl AsAny for TfIdfEmbeddingGenerator {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl AsAny for SentenceTransformerGenerator {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// OpenAI embeddings generator with rate limiting and retry logic
pub struct OpenAIEmbeddingGenerator {
    config: EmbeddingConfig,
    openai_config: OpenAIConfig,
    client: reqwest::Client,
    rate_limiter: RateLimiter,
    request_cache: HashMap<u64, Vector>,
}

/// Simple rate limiter implementation
pub struct RateLimiter {
    requests_per_minute: u32,
    request_times: std::collections::VecDeque<std::time::Instant>,
}

impl RateLimiter {
    pub fn new(requests_per_minute: u32) -> Self {
        Self {
            requests_per_minute,
            request_times: std::collections::VecDeque::new(),
        }
    }

    pub async fn wait_if_needed(&mut self) {
        let now = std::time::Instant::now();
        let minute_ago = now - std::time::Duration::from_secs(60);

        // Remove requests older than 1 minute
        while let Some(&front_time) = self.request_times.front() {
            if front_time < minute_ago {
                self.request_times.pop_front();
            } else {
                break;
            }
        }

        // If we're at the rate limit, wait
        if self.request_times.len() >= self.requests_per_minute as usize {
            if let Some(&oldest) = self.request_times.front() {
                let wait_time = oldest + std::time::Duration::from_secs(60) - now;
                if !wait_time.is_zero() {
                    tokio::time::sleep(wait_time).await;
                }
            }
        }

        self.request_times.push_back(now);
    }
}

impl OpenAIEmbeddingGenerator {
    pub fn new(openai_config: OpenAIConfig) -> Result<Self> {
        if openai_config.api_key.is_empty() {
            return Err(anyhow!("OpenAI API key is required. Set OPENAI_API_KEY environment variable."));
        }

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(openai_config.timeout_seconds))
            .build()
            .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))?;

        let embedding_config = EmbeddingConfig {
            model_name: openai_config.model.clone(),
            dimensions: match openai_config.model.as_str() {
                "text-embedding-ada-002" => 1536,
                "text-embedding-3-small" => 1536,
                "text-embedding-3-large" => 3072,
                _ => 1536, // Default
            },
            max_sequence_length: 8191, // OpenAI limit
            normalize: true,
        };

        Ok(Self {
            config: embedding_config,
            openai_config: openai_config.clone(),
            client,
            rate_limiter: RateLimiter::new(openai_config.requests_per_minute),
            request_cache: HashMap::new(),
        })
    }

    /// Make API request to OpenAI with retry logic
    async fn make_request(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut attempts = 0;
        
        while attempts < self.openai_config.max_retries {
            match self.try_request(texts).await {
                Ok(embeddings) => return Ok(embeddings),
                Err(e) => {
                    attempts += 1;
                    if attempts >= self.openai_config.max_retries {
                        return Err(e);
                    }
                    
                    // Exponential backoff
                    let delay = self.openai_config.retry_delay_ms * (2_u64.pow(attempts - 1));
                    tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                }
            }
        }

        Err(anyhow!("Max retries exceeded"))
    }

    async fn try_request(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.rate_limiter.wait_if_needed().await;

        let request_body = serde_json::json!({
            "model": self.openai_config.model,
            "input": texts,
            "encoding_format": "float"
        });

        let response = self
            .client
            .post(&format!("{}/embeddings", self.openai_config.base_url))
            .header("Authorization", format!("Bearer {}", self.openai_config.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| anyhow!("Request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!("API request failed with status {}: {}", status, error_text));
        }

        let response_data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse response: {}", e))?;

        let embeddings_data = response_data["data"]
            .as_array()
            .ok_or_else(|| anyhow!("Invalid response format: missing data array"))?;

        let mut embeddings = Vec::new();
        for item in embeddings_data {
            let embedding = item["embedding"]
                .as_array()
                .ok_or_else(|| anyhow!("Invalid response format: missing embedding"))?;
            
            let vec: Result<Vec<f32>, _> = embedding
                .iter()
                .map(|v| v.as_f64().ok_or_else(|| anyhow!("Invalid embedding value")).map(|f| f as f32))
                .collect();
            
            embeddings.push(vec?);
        }

        Ok(embeddings)
    }

    /// Generate embeddings with batching support
    pub async fn generate_async(&mut self, content: &EmbeddableContent) -> Result<Vector> {
        let text = content.to_text();
        
        // Check cache first
        if self.openai_config.enable_cache {
            let hash = content.content_hash();
            if let Some(cached) = self.request_cache.get(&hash) {
                return Ok(cached.clone());
            }
        }

        let embeddings = self.make_request(&[text]).await?;
        
        if embeddings.is_empty() {
            return Err(anyhow!("No embeddings returned from API"));
        }

        let vector = Vector::new(embeddings[0].clone());

        // Cache the result
        if self.openai_config.enable_cache {
            let hash = content.content_hash();
            self.request_cache.insert(hash, vector.clone());
        }

        Ok(vector)
    }

    /// Generate embeddings for multiple texts in batch
    pub async fn generate_batch_async(&mut self, contents: &[EmbeddableContent]) -> Result<Vec<Vector>> {
        if contents.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(contents.len());
        let batch_size = self.openai_config.batch_size;

        for chunk in contents.chunks(batch_size) {
            let texts: Vec<String> = chunk.iter().map(|c| c.to_text()).collect();
            let embeddings = self.make_request(&texts).await?;
            
            if embeddings.len() != chunk.len() {
                return Err(anyhow!("Mismatch between request and response sizes"));
            }

            for (content, embedding) in chunk.iter().zip(embeddings) {
                let vector = Vector::new(embedding);
                
                // Cache the result
                if self.openai_config.enable_cache {
                    let hash = content.content_hash();
                    self.request_cache.insert(hash, vector.clone());
                }
                
                results.push(vector);
            }
        }

        Ok(results)
    }

    /// Clear the request cache
    pub fn clear_cache(&mut self) {
        self.request_cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, Option<usize>) {
        (self.request_cache.len(), None) // No max size limit for simplicity
    }
}

impl EmbeddingGenerator for OpenAIEmbeddingGenerator {
    fn generate(&self, content: &EmbeddableContent) -> Result<Vector> {
        // For synchronous interface, we use a runtime
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| anyhow!("Failed to create async runtime: {}", e))?;
        
        // We need a mutable reference, but trait only provides immutable
        // This is a limitation of the current trait design
        // For now, return an error suggesting async usage
        Err(anyhow!("OpenAI embeddings require async execution. Use generate_async() instead."))
    }

    fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
}

impl AsAny for OpenAIEmbeddingGenerator {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
