//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::Vector;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use super::functions::EmbeddingGenerator;
use super::openaiembeddinggenerator_type::OpenAIEmbeddingGenerator;
use super::sentencetransformergenerator_type::SentenceTransformerGenerator;

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
/// Detailed information about a transformer model
#[derive(Debug, Clone)]
pub struct ModelDetails {
    pub vocab_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub supports_languages: Vec<String>,
    pub model_size_mb: usize,
    pub typical_inference_time_ms: u64,
}
/// Retry strategy for failed requests
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RetryStrategy {
    /// Fixed delay between retries
    Fixed,
    /// Exponential backoff with jitter
    ExponentialBackoff,
    /// Linear backoff
    LinearBackoff,
}
/// Embedding model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub model_name: String,
    pub dimensions: usize,
    pub max_sequence_length: usize,
    pub normalize: bool,
}
/// Mock embedding generator for testing
#[cfg(test)]
pub struct MockEmbeddingGenerator {
    pub(super) config: EmbeddingConfig,
}
#[cfg(test)]
impl MockEmbeddingGenerator {
    pub fn new() -> Self {
        Self {
            config: EmbeddingConfig {
                dimensions: 128,
                ..Default::default()
            },
        }
    }
    pub fn with_dimensions(dimensions: usize) -> Self {
        Self {
            config: EmbeddingConfig {
                dimensions,
                ..Default::default()
            },
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
                    text_parts.push(format!("label: {label}"));
                }
                if let Some(desc) = description {
                    text_parts.push(format!("description: {desc}"));
                }
                for (prop, values) in properties {
                    text_parts.push(format!("{prop}: {}", values.join(", ")));
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
                    model_name: format!("{model_type:?}"),
                    dimensions: match model_type {
                        TransformerModelType::DistilBERT => 384,
                        _ => 768,
                    },
                    max_sequence_length: 512,
                    normalize: true,
                };
                Box::new(SentenceTransformerGenerator::with_model_type(
                    config,
                    model_type.clone(),
                ))
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
/// Supported transformer model types
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum TransformerModelType {
    /// Basic BERT-based model (already implemented)
    #[default]
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
/// OpenAI embeddings configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIConfig {
    /// API key for OpenAI service
    pub api_key: String,
    /// Model to use (e.g., "text-embedding-ada-002", "text-embedding-3-small")
    pub model: String,
    /// Base URL for API calls (default: `https://api.openai.com/v1`)
    pub base_url: String,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Rate limiting: requests per minute
    pub requests_per_minute: u32,
    /// Batch size for batch processing
    pub batch_size: usize,
    /// Enable local caching
    pub enable_cache: bool,
    /// Cache size (number of embeddings to cache)
    pub cache_size: usize,
    /// Cache TTL in seconds (0 for no expiration)
    pub cache_ttl_seconds: u64,
    /// Maximum retries for failed requests
    pub max_retries: u32,
    /// Retry delay in milliseconds
    pub retry_delay_ms: u64,
    /// Retry strategy
    pub retry_strategy: RetryStrategy,
    /// Enable cost tracking
    pub track_costs: bool,
    /// Enable detailed metrics
    pub enable_metrics: bool,
    /// User agent for requests
    pub user_agent: String,
}
impl OpenAIConfig {
    /// Create config for production use
    pub fn production() -> Self {
        Self {
            requests_per_minute: 1000,
            cache_size: 50000,
            cache_ttl_seconds: 7200,
            max_retries: 5,
            retry_strategy: RetryStrategy::ExponentialBackoff,
            ..Default::default()
        }
    }
    /// Create config for development/testing
    pub fn development() -> Self {
        Self {
            requests_per_minute: 100,
            cache_size: 1000,
            cache_ttl_seconds: 300,
            max_retries: 2,
            ..Default::default()
        }
    }
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.api_key.is_empty() {
            return Err(anyhow!("OpenAI API key is required"));
        }
        if self.requests_per_minute == 0 {
            return Err(anyhow!("requests_per_minute must be greater than 0"));
        }
        if self.batch_size == 0 {
            return Err(anyhow!("batch_size must be greater than 0"));
        }
        if self.timeout_seconds == 0 {
            return Err(anyhow!("timeout_seconds must be greater than 0"));
        }
        Ok(())
    }
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
        while let Some(&front_time) = self.request_times.front() {
            if front_time < minute_ago {
                self.request_times.pop_front();
            } else {
                break;
            }
        }
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
/// Metrics for OpenAI API usage
#[derive(Debug, Clone, Default)]
pub struct OpenAIMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_tokens_processed: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_cost_usd: f64,
    pub retry_count: u64,
    pub rate_limit_waits: u64,
    pub average_response_time_ms: f64,
    pub last_request_time: Option<std::time::SystemTime>,
    pub requests_by_model: HashMap<String, u64>,
    pub errors_by_type: HashMap<String, u64>,
}
impl OpenAIMetrics {
    /// Calculate cache hit ratio
    pub fn cache_hit_ratio(&self) -> f64 {
        if self.cache_hits + self.cache_misses == 0 {
            0.0
        } else {
            self.cache_hits as f64 / (self.cache_hits + self.cache_misses) as f64
        }
    }
    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.successful_requests as f64 / self.total_requests as f64
        }
    }
    /// Calculate average cost per request
    pub fn average_cost_per_request(&self) -> f64 {
        if self.successful_requests == 0 {
            0.0
        } else {
            self.total_cost_usd / self.successful_requests as f64
        }
    }
    /// Get formatted metrics report
    pub fn report(&self) -> String {
        format!(
            "OpenAI Metrics Report:\n\
            Total Requests: {}\n\
            Success Rate: {:.2}%\n\
            Cache Hit Ratio: {:.2}%\n\
            Total Cost: ${:.4}\n\
            Avg Cost/Request: ${:.6}\n\
            Avg Response Time: {:.2}ms\n\
            Retries: {}\n\
            Rate Limit Waits: {}",
            self.total_requests,
            self.success_rate() * 100.0,
            self.cache_hit_ratio() * 100.0,
            self.total_cost_usd,
            self.average_cost_per_request(),
            self.average_response_time_ms,
            self.retry_count,
            self.rate_limit_waits
        )
    }
}
/// Cached embedding with metadata
#[derive(Debug, Clone)]
pub struct CachedEmbedding {
    pub vector: Vector,
    pub cached_at: std::time::SystemTime,
    pub model: String,
    pub cost_usd: f64,
}
/// Simple TF-IDF based embedding generator
pub struct TfIdfEmbeddingGenerator {
    pub(super) config: EmbeddingConfig,
    pub(super) vocabulary: HashMap<String, usize>,
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
        let mut word_freq: Vec<(String, usize)> = word_counts.into_iter().collect();
        word_freq.sort_by(|a, b| b.1.cmp(&a.1));
        self.vocabulary = word_freq
            .into_iter()
            .take(self.config.dimensions)
            .enumerate()
            .map(|(idx, (word, _))| (word, idx))
            .collect();
        let total_docs = documents.len() as f32;
        for word in self.vocabulary.keys() {
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
    pub(super) fn calculate_tf_idf(&self, text: &str) -> Vector {
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
