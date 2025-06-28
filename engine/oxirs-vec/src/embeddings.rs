//! Embedding generation and management for RDF resources and text content

use crate::Vector;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::Duration;
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
            cache_size: 10000,
            cache_ttl_seconds: 3600, // 1 hour
            max_retries: 3,
            retry_delay_ms: 1000,
            retry_strategy: RetryStrategy::ExponentialBackoff,
            track_costs: true,
            enable_metrics: true,
            user_agent: "oxirs-vec/0.1.0".to_string(),
        }
    }
}

impl OpenAIConfig {
    /// Create config for production use
    pub fn production() -> Self {
        Self {
            requests_per_minute: 1000, // More conservative for production
            cache_size: 50000,
            cache_ttl_seconds: 7200, // 2 hours
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
            cache_ttl_seconds: 300, // 5 minutes
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

    /// Create a new RoBERTa model generator
    pub fn roberta(config: EmbeddingConfig) -> Self {
        Self::with_model_type(config, TransformerModelType::RoBERTa)
    }

    /// Create a new DistilBERT model generator
    pub fn distilbert(config: EmbeddingConfig) -> Self {
        let adjusted_config = EmbeddingConfig {
            dimensions: 384, // DistilBERT has smaller dimensions
            ..config
        };
        Self::with_model_type(adjusted_config, TransformerModelType::DistilBERT)
    }

    /// Create a new multilingual BERT model generator
    pub fn multilingual_bert(config: EmbeddingConfig) -> Self {
        Self::with_model_type(config, TransformerModelType::MultiBERT)
    }

    /// Get the current model type
    pub fn model_type(&self) -> &TransformerModelType {
        &self.model_type
    }

    /// Get detailed information about the current model
    pub fn model_details(&self) -> ModelDetails {
        self.get_model_details()
    }

    /// Check if the model supports a specific language
    pub fn supports_language(&self, language_code: &str) -> bool {
        let details = self.get_model_details();
        details
            .supports_languages
            .contains(&language_code.to_string())
    }

    /// Get the estimated inference time for a given text length
    pub fn estimate_inference_time(&self, text_length: usize) -> u64 {
        let details = self.get_model_details();
        let base_time = details.typical_inference_time_ms;

        // Rough estimation: longer texts take more time
        let length_factor = (text_length as f64 / 100.0).sqrt().max(1.0);
        (base_time as f64 * length_factor) as u64
    }

    /// Get the memory footprint of the model in MB
    pub fn model_size_mb(&self) -> usize {
        self.get_model_details().model_size_mb
    }

    /// Get efficiency rating (higher is better/faster)
    pub fn efficiency_rating(&self) -> f32 {
        match &self.model_type {
            TransformerModelType::DistilBERT => 1.5, // Fastest
            TransformerModelType::BERT => 1.0,       // Baseline
            TransformerModelType::RoBERTa => 0.95,   // Slightly slower
            TransformerModelType::MultiBERT => 0.8,  // Slowest due to multilingual complexity
            TransformerModelType::Custom(_) => 1.0,  // Unknown, assume baseline
        }
    }

    /// Get model-specific configuration adjustments
    fn get_model_config(&self) -> (usize, usize, f32) {
        match &self.model_type {
            TransformerModelType::BERT => (768, 512, 1.0), // (dimensions, max_seq_len, efficiency)
            TransformerModelType::RoBERTa => (768, 514, 0.95), // RoBERTa supports slightly longer sequences
            TransformerModelType::DistilBERT => (384, 512, 1.5), // DistilBERT is smaller and faster
            TransformerModelType::MultiBERT => (768, 512, 0.8), // Multilingual BERT, slower but broader language support
            TransformerModelType::Custom(_) => {
                (self.config.dimensions, self.config.max_sequence_length, 1.0)
            }
        }
    }

    /// Get model-specific vocabulary size and training details
    fn get_model_details(&self) -> ModelDetails {
        match &self.model_type {
            TransformerModelType::BERT => ModelDetails {
                vocab_size: 30522,
                num_layers: 12,
                num_attention_heads: 12,
                hidden_size: 768,
                intermediate_size: 3072,
                max_position_embeddings: 512,
                supports_languages: vec!["en".to_string()],
                model_size_mb: 440,
                typical_inference_time_ms: 50,
            },
            TransformerModelType::RoBERTa => ModelDetails {
                vocab_size: 50265,
                num_layers: 12,
                num_attention_heads: 12,
                hidden_size: 768,
                intermediate_size: 3072,
                max_position_embeddings: 514,
                supports_languages: vec!["en".to_string()],
                model_size_mb: 470,
                typical_inference_time_ms: 55, // Slightly slower due to different training
            },
            TransformerModelType::DistilBERT => ModelDetails {
                vocab_size: 30522,
                num_layers: 6, // Half the layers of BERT
                num_attention_heads: 12,
                hidden_size: 384, // Smaller hidden size
                intermediate_size: 1536,
                max_position_embeddings: 512,
                supports_languages: vec!["en".to_string()],
                model_size_mb: 250,            // Much smaller
                typical_inference_time_ms: 25, // Much faster
            },
            TransformerModelType::MultiBERT => ModelDetails {
                vocab_size: 120000, // Larger vocabulary for multilingual support
                num_layers: 12,
                num_attention_heads: 12,
                hidden_size: 768,
                intermediate_size: 3072,
                max_position_embeddings: 512,
                supports_languages: vec![
                    "en".to_string(),
                    "de".to_string(),
                    "fr".to_string(),
                    "es".to_string(),
                    "it".to_string(),
                    "pt".to_string(),
                    "ru".to_string(),
                    "zh".to_string(),
                    "ja".to_string(),
                    "ko".to_string(),
                    "ar".to_string(),
                    "hi".to_string(),
                    "th".to_string(),
                    "tr".to_string(),
                    "pl".to_string(),
                    "nl".to_string(),
                    "sv".to_string(),
                    "da".to_string(),
                    "no".to_string(),
                    "fi".to_string(),
                ], // Top 20 languages supported
                model_size_mb: 670, // Larger due to multilingual vocabulary
                typical_inference_time_ms: 70, // Slower due to larger vocabulary
            },
            TransformerModelType::Custom(path) => ModelDetails {
                vocab_size: 50000, // Default assumption
                num_layers: 12,
                num_attention_heads: 12,
                hidden_size: self.config.dimensions,
                intermediate_size: self.config.dimensions * 4,
                max_position_embeddings: self.config.max_sequence_length,
                supports_languages: vec!["unknown".to_string()],
                model_size_mb: 500, // Estimate
                typical_inference_time_ms: 60,
            },
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
        let model_details = self.get_model_details();

        // Apply model-specific text preprocessing
        let processed_text = self.preprocess_text_for_model(text, max_len)?;

        // Simulate tokenization differences between models
        let token_ids = self.simulate_tokenization(&processed_text, &model_details);

        // Generate model-specific embeddings
        let values =
            self.generate_embeddings_from_tokens(&token_ids, dimensions, &model_details)?;

        if self.config.normalize {
            let magnitude: f32 = values.iter().map(|x| x * x).sum::<f32>().sqrt();
            if magnitude > 0.0 {
                let mut normalized_values = values;
                for value in &mut normalized_values {
                    *value /= magnitude;
                }
                return Ok(Vector::new(normalized_values));
            }
        }

        Ok(Vector::new(values))
    }

    /// Preprocess text according to model-specific requirements
    fn preprocess_text_for_model(&self, text: &str, max_len: usize) -> Result<String> {
        let processed = match &self.model_type {
            TransformerModelType::BERT => {
                // BERT uses [CLS] and [SEP] tokens
                let truncated = if text.len() > max_len - 20 {
                    // Reserve space for special tokens
                    &text[..max_len - 20]
                } else {
                    text
                };
                format!("[CLS] {} [SEP]", truncated.to_lowercase())
            }
            TransformerModelType::RoBERTa => {
                // RoBERTa uses <s> and </s> tokens and preserves case better
                let truncated = if text.len() > max_len - 10 {
                    &text[..max_len - 10]
                } else {
                    text
                };
                format!("<s>{}</s>", truncated) // RoBERTa preserves case
            }
            TransformerModelType::DistilBERT => {
                // DistilBERT is similar to BERT but more aggressive truncation due to efficiency
                let truncated = if text.len() > max_len - 20 {
                    &text[..max_len - 20]
                } else {
                    text
                };
                format!("[CLS] {} [SEP]", truncated.to_lowercase())
            }
            TransformerModelType::MultiBERT => {
                // Multilingual BERT handles multiple languages, no case conversion for non-Latin scripts
                let truncated = if text.len() > max_len - 20 {
                    &text[..max_len - 20]
                } else {
                    text
                };
                // Detect if text contains non-Latin characters
                let has_non_latin = text.chars().any(|c| !c.is_ascii());
                if has_non_latin {
                    format!("[CLS] {} [SEP]", truncated) // Preserve case for non-Latin
                } else {
                    format!("[CLS] {} [SEP]", truncated.to_lowercase()) // Lowercase for Latin
                }
            }
            TransformerModelType::Custom(_) => {
                // Basic preprocessing for custom models
                let truncated = if text.len() > max_len {
                    &text[..max_len]
                } else {
                    text
                };
                truncated.to_string()
            }
        };

        Ok(processed)
    }

    /// Simulate tokenization process for different models
    fn simulate_tokenization(&self, text: &str, model_details: &ModelDetails) -> Vec<u32> {
        let mut token_ids = Vec::new();

        // Simple word-based tokenization simulation
        let words: Vec<&str> = text.split_whitespace().collect();

        for word in words {
            // Simulate subword tokenization
            let subwords = match &self.model_type {
                TransformerModelType::RoBERTa => {
                    // RoBERTa uses byte-pair encoding, tends to create more subwords
                    self.simulate_bpe_tokenization(word, model_details.vocab_size)
                }
                TransformerModelType::DistilBERT | TransformerModelType::BERT => {
                    // BERT uses WordPiece tokenization
                    self.simulate_wordpiece_tokenization(word, model_details.vocab_size)
                }
                TransformerModelType::MultiBERT => {
                    // Multilingual BERT has larger vocabulary, fewer subwords
                    self.simulate_multilingual_tokenization(word, model_details.vocab_size)
                }
                TransformerModelType::Custom(_) => {
                    // Simple tokenization for custom models
                    vec![self.word_to_token_id(word, model_details.vocab_size)]
                }
            };

            token_ids.extend(subwords);
        }

        // Truncate to max sequence length
        token_ids.truncate(model_details.max_position_embeddings - 2); // Reserve space for special tokens
        token_ids
    }

    /// Simulate BPE tokenization (used by RoBERTa)
    fn simulate_bpe_tokenization(&self, word: &str, vocab_size: usize) -> Vec<u32> {
        let mut tokens = Vec::new();
        let mut remaining = word;

        while !remaining.is_empty() {
            let chunk_size = if remaining.len() > 4 {
                4
            } else {
                remaining.len()
            };
            let chunk = &remaining[..chunk_size];
            tokens.push(self.word_to_token_id(chunk, vocab_size));
            remaining = &remaining[chunk_size..];
        }

        tokens
    }

    /// Simulate WordPiece tokenization (used by BERT)
    fn simulate_wordpiece_tokenization(&self, word: &str, vocab_size: usize) -> Vec<u32> {
        if word.len() <= 6 {
            vec![self.word_to_token_id(word, vocab_size)]
        } else {
            let mid = word.len() / 2;
            vec![
                self.word_to_token_id(&word[..mid], vocab_size),
                self.word_to_token_id(&format!("##{}", &word[mid..]), vocab_size), // ## prefix for subwords
            ]
        }
    }

    /// Simulate multilingual tokenization (larger vocab = fewer subwords)
    fn simulate_multilingual_tokenization(&self, word: &str, vocab_size: usize) -> Vec<u32> {
        // Multilingual models have larger vocabularies, so less subword splitting
        if word.len() <= 10 {
            vec![self.word_to_token_id(word, vocab_size)]
        } else {
            let mid = word.len() / 2;
            vec![
                self.word_to_token_id(&word[..mid], vocab_size),
                self.word_to_token_id(&word[mid..], vocab_size),
            ]
        }
    }

    /// Convert word to token ID
    fn word_to_token_id(&self, word: &str, vocab_size: usize) -> u32 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        word.hash(&mut hasher);
        (hasher.finish() % vocab_size as u64) as u32
    }

    /// Generate embeddings from token IDs using model-specific patterns
    fn generate_embeddings_from_tokens(
        &self,
        token_ids: &[u32],
        dimensions: usize,
        model_details: &ModelDetails,
    ) -> Result<Vec<f32>> {
        let mut values = vec![0.0; dimensions];

        // Model-specific embedding generation
        match &self.model_type {
            TransformerModelType::BERT => {
                self.generate_bert_style_embeddings(token_ids, &mut values, model_details)
            }
            TransformerModelType::RoBERTa => {
                self.generate_roberta_style_embeddings(token_ids, &mut values, model_details)
            }
            TransformerModelType::DistilBERT => {
                self.generate_distilbert_style_embeddings(token_ids, &mut values, model_details)
            }
            TransformerModelType::MultiBERT => {
                self.generate_multibert_style_embeddings(token_ids, &mut values, model_details)
            }
            TransformerModelType::Custom(_) => {
                self.generate_custom_style_embeddings(token_ids, &mut values, model_details)
            }
        }

        Ok(values)
    }

    /// Generate BERT-style embeddings
    fn generate_bert_style_embeddings(
        &self,
        token_ids: &[u32],
        values: &mut [f32],
        model_details: &ModelDetails,
    ) {
        for (i, &token_id) in token_ids.iter().enumerate() {
            let mut seed = token_id as u64;
            for j in 0..values.len() {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let normalized = (seed as f32) / (u64::MAX as f32);
                let position_encoding =
                    ((i as f32 / 512.0) * 2.0 * std::f32::consts::PI).sin() * 0.1;
                values[j] += ((normalized - 0.5) * 2.0) + position_encoding;
            }
        }

        // Average the contributions from all tokens
        if !token_ids.is_empty() {
            for value in values.iter_mut() {
                *value /= token_ids.len() as f32;
            }
        }
    }

    /// Generate RoBERTa-style embeddings (no segment embeddings, different position encoding)
    fn generate_roberta_style_embeddings(
        &self,
        token_ids: &[u32],
        values: &mut [f32],
        model_details: &ModelDetails,
    ) {
        for (i, &token_id) in token_ids.iter().enumerate() {
            let mut seed = token_id.wrapping_mul(31415927); // Different seed pattern
            for j in 0..values.len() {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let normalized = (seed as f32) / (u64::MAX as f32);
                // RoBERTa uses learned position embeddings starting from index 2
                let position_encoding =
                    ((i as f32 + 2.0) / 514.0 * 2.0 * std::f32::consts::PI).cos() * 0.1;
                values[j] += ((normalized - 0.5) * 2.0) + position_encoding;
            }
        }

        if !token_ids.is_empty() {
            for value in values.iter_mut() {
                *value /= token_ids.len() as f32;
            }
        }
    }

    /// Generate DistilBERT-style embeddings (simpler, faster)
    fn generate_distilbert_style_embeddings(
        &self,
        token_ids: &[u32],
        values: &mut [f32],
        model_details: &ModelDetails,
    ) {
        // DistilBERT has fewer layers and smaller hidden size
        for (i, &token_id) in token_ids.iter().enumerate() {
            let mut seed = token_id as u64;
            for j in 0..values.len() {
                seed = seed.wrapping_mul(982451653).wrapping_add(12345); // Faster computation
                let normalized = (seed as f32) / (u64::MAX as f32);
                // Simpler position encoding
                let position_encoding = (i as f32 / 512.0).sin() * 0.05;
                values[j] += ((normalized - 0.5) * 1.5) + position_encoding; // Slightly different scale
            }
        }

        if !token_ids.is_empty() {
            for value in values.iter_mut() {
                *value /= token_ids.len() as f32;
            }
        }
    }

    /// Generate multilingual BERT-style embeddings
    fn generate_multibert_style_embeddings(
        &self,
        token_ids: &[u32],
        values: &mut [f32],
        model_details: &ModelDetails,
    ) {
        for (i, &token_id) in token_ids.iter().enumerate() {
            // Multilingual models have different patterns due to cross-lingual training
            let mut seed = token_id.wrapping_mul(2654435761); // Different multiplier for multilingual
            for j in 0..values.len() {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let normalized = (seed as f32) / (u64::MAX as f32);
                let position_encoding =
                    ((i as f32 / 512.0) * 2.0 * std::f32::consts::PI).sin() * 0.08;
                // Add language-agnostic patterns
                let cross_lingual_bias =
                    (j as f32 / values.len() as f32 * std::f32::consts::PI).cos() * 0.05;
                values[j] += ((normalized - 0.5) * 1.8) + position_encoding + cross_lingual_bias;
            }
        }

        if !token_ids.is_empty() {
            for value in values.iter_mut() {
                *value /= token_ids.len() as f32;
            }
        }
    }

    /// Generate custom model embeddings
    fn generate_custom_style_embeddings(
        &self,
        token_ids: &[u32],
        values: &mut [f32],
        model_details: &ModelDetails,
    ) {
        // Simple approach for custom models
        for &token_id in token_ids {
            let mut seed = token_id as u64;
            for value in values.iter_mut() {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let normalized = (seed as f32) / (u64::MAX as f32);
                *value += (normalized - 0.5) * 2.0;
            }
        }

        if !token_ids.is_empty() {
            for value in values.iter_mut() {
                *value /= token_ids.len() as f32;
            }
        }
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
                        _ => 768,                                // Most BERT variants
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
    request_cache: std::sync::Arc<std::sync::Mutex<lru::LruCache<u64, CachedEmbedding>>>,
    metrics: OpenAIMetrics,
}

/// Cached embedding with metadata
#[derive(Debug, Clone)]
pub struct CachedEmbedding {
    pub vector: Vector,
    pub cached_at: std::time::SystemTime,
    pub model: String,
    pub cost_usd: f64,
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
        openai_config.validate()?;

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(
                openai_config.timeout_seconds,
            ))
            .user_agent(&openai_config.user_agent)
            .build()
            .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))?;

        let embedding_config = EmbeddingConfig {
            model_name: openai_config.model.clone(),
            dimensions: Self::get_model_dimensions(&openai_config.model),
            max_sequence_length: 8191, // OpenAI limit
            normalize: true,
        };

        let cache_size = if openai_config.enable_cache {
            std::num::NonZeroUsize::new(openai_config.cache_size)
                .unwrap_or(std::num::NonZeroUsize::new(1000).unwrap())
        } else {
            std::num::NonZeroUsize::new(1).unwrap()
        };

        Ok(Self {
            config: embedding_config,
            openai_config: openai_config.clone(),
            client,
            rate_limiter: RateLimiter::new(openai_config.requests_per_minute),
            request_cache: std::sync::Arc::new(std::sync::Mutex::new(lru::LruCache::new(
                cache_size,
            ))),
            metrics: OpenAIMetrics::default(),
        })
    }

    /// Get dimensions for different OpenAI models
    fn get_model_dimensions(model: &str) -> usize {
        match model {
            "text-embedding-ada-002" => 1536,
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-004" => 1536,
            _ => 1536, // Default
        }
    }

    /// Get cost per 1k tokens for different models (in USD)
    fn get_model_cost_per_1k_tokens(model: &str) -> f64 {
        match model {
            "text-embedding-ada-002" => 0.0001,
            "text-embedding-3-small" => 0.00002,
            "text-embedding-3-large" => 0.00013,
            "text-embedding-004" => 0.00002,
            _ => 0.0001, // Conservative default
        }
    }

    /// Calculate cost for processing texts
    fn calculate_cost(&self, texts: &[String]) -> f64 {
        if !self.openai_config.track_costs {
            return 0.0;
        }

        let total_tokens: usize = texts.iter().map(|t| t.len() / 4).sum(); // Rough token estimation
        let cost_per_1k = Self::get_model_cost_per_1k_tokens(&self.openai_config.model);
        (total_tokens as f64 / 1000.0) * cost_per_1k
    }

    /// Check if cached embedding is still valid
    fn is_cache_valid(&self, cached: &CachedEmbedding) -> bool {
        if self.openai_config.cache_ttl_seconds == 0 {
            return true; // No expiration
        }

        let elapsed = cached
            .cached_at
            .elapsed()
            .unwrap_or(std::time::Duration::from_secs(u64::MAX));

        elapsed.as_secs() < self.openai_config.cache_ttl_seconds
    }

    /// Make API request to OpenAI with retry logic
    async fn make_request(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let start_time = std::time::Instant::now();
        let mut attempts = 0;

        while attempts < self.openai_config.max_retries {
            match self.try_request(texts).await {
                Ok(embeddings) => {
                    // Update metrics
                    if self.openai_config.enable_metrics {
                        let response_time = start_time.elapsed().as_millis() as f64;
                        self.update_response_time(response_time);

                        let cost = self.calculate_cost(texts);
                        self.metrics.total_cost_usd += cost;

                        *self
                            .metrics
                            .requests_by_model
                            .entry(self.openai_config.model.clone())
                            .or_insert(0) += 1;
                    }

                    return Ok(embeddings);
                }
                Err(e) => {
                    attempts += 1;
                    self.metrics.retry_count += 1;

                    // Track error types
                    let error_type = if e.to_string().contains("rate_limit") {
                        "rate_limit"
                    } else if e.to_string().contains("timeout") {
                        "timeout"
                    } else if e.to_string().contains("401") {
                        "unauthorized"
                    } else if e.to_string().contains("400") {
                        "bad_request"
                    } else {
                        "other"
                    };

                    *self
                        .metrics
                        .errors_by_type
                        .entry(error_type.to_string())
                        .or_insert(0) += 1;

                    if attempts >= self.openai_config.max_retries {
                        return Err(e);
                    }

                    // Calculate delay based on retry strategy
                    let delay = self.calculate_retry_delay(attempts);
                    tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                }
            }
        }

        Err(anyhow!("Max retries exceeded"))
    }

    /// Calculate retry delay based on strategy
    fn calculate_retry_delay(&self, attempt: u32) -> u64 {
        let base_delay = self.openai_config.retry_delay_ms;

        match self.openai_config.retry_strategy {
            RetryStrategy::Fixed => base_delay,
            RetryStrategy::LinearBackoff => base_delay * attempt as u64,
            RetryStrategy::ExponentialBackoff => {
                let delay = base_delay * (2_u64.pow(attempt - 1));
                // Add jitter (±25%)
                let jitter = (delay as f64 * 0.25 * (rand::random::<f64>() - 0.5)) as u64;
                delay.saturating_add(jitter).min(30000) // Max 30 seconds
            }
        }
    }

    /// Update response time metrics
    fn update_response_time(&mut self, response_time_ms: f64) {
        if self.metrics.successful_requests == 0 {
            self.metrics.average_response_time_ms = response_time_ms;
        } else {
            // Running average
            let total =
                self.metrics.average_response_time_ms * self.metrics.successful_requests as f64;
            self.metrics.average_response_time_ms =
                (total + response_time_ms) / (self.metrics.successful_requests + 1) as f64;
        }
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
            .header(
                "Authorization",
                format!("Bearer {}", self.openai_config.api_key),
            )
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| anyhow!("Request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!(
                "API request failed with status {}: {}",
                status,
                error_text
            ));
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
                .map(|v| {
                    v.as_f64()
                        .ok_or_else(|| anyhow!("Invalid embedding value"))
                        .map(|f| f as f32)
                })
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

            // Check if cached entry exists and is valid
            let cached_vector = if let Ok(mut cache) = self.request_cache.lock() {
                if let Some(cached) = cache.get(&hash) {
                    let is_valid = cached.cached_at.elapsed().unwrap_or_default()
                        < Duration::from_secs(self.openai_config.cache_ttl_seconds);
                    if is_valid {
                        Some(cached.vector.clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(result) = cached_vector {
                self.update_cache_hit();
                return Ok(result);
            } else {
                // Remove expired entry if it exists
                if let Ok(mut cache) = self.request_cache.lock() {
                    cache.pop(&hash);
                }
                self.update_cache_miss();
            }
        }

        let embeddings = match self.make_request(&[text.clone()]).await {
            Ok(embeddings) => {
                self.update_metrics_success(&[text.clone()]);
                embeddings
            }
            Err(e) => {
                self.update_metrics_failure();
                return Err(e);
            }
        };

        if embeddings.is_empty() {
            self.update_metrics_failure();
            return Err(anyhow!("No embeddings returned from API"));
        }

        let vector = Vector::new(embeddings[0].clone());

        // Cache the result
        if self.openai_config.enable_cache {
            let hash = content.content_hash();
            let cost = self.calculate_cost(&[text.clone()]);
            let cached_embedding = CachedEmbedding {
                vector: vector.clone(),
                cached_at: std::time::SystemTime::now(),
                model: self.openai_config.model.clone(),
                cost_usd: cost,
            };
            if let Ok(mut cache) = self.request_cache.lock() {
                cache.put(hash, cached_embedding);
            }
        }

        Ok(vector)
    }

    /// Generate embeddings for multiple texts in batch
    pub async fn generate_batch_async(
        &mut self,
        contents: &[EmbeddableContent],
    ) -> Result<Vec<Vector>> {
        if contents.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(contents.len());
        let batch_size = self.openai_config.batch_size;

        for chunk in contents.chunks(batch_size) {
            let texts: Vec<String> = chunk.iter().map(|c| c.to_text()).collect();

            let embeddings = match self.make_request(&texts).await {
                Ok(embeddings) => {
                    self.update_metrics_success(&texts);
                    embeddings
                }
                Err(e) => {
                    self.update_metrics_failure();
                    return Err(e);
                }
            };

            if embeddings.len() != chunk.len() {
                self.update_metrics_failure();
                return Err(anyhow!("Mismatch between request and response sizes"));
            }

            let batch_cost = self.calculate_cost(&texts) / chunk.len() as f64;

            for (content, embedding) in chunk.iter().zip(embeddings) {
                let vector = Vector::new(embedding);

                // Cache the result
                if self.openai_config.enable_cache {
                    let hash = content.content_hash();
                    let cached_embedding = CachedEmbedding {
                        vector: vector.clone(),
                        cached_at: std::time::SystemTime::now(),
                        model: self.openai_config.model.clone(),
                        cost_usd: batch_cost,
                    };
                    if let Ok(mut cache) = self.request_cache.lock() {
                        cache.put(hash, cached_embedding);
                    }
                }

                results.push(vector);
            }
        }

        Ok(results)
    }

    /// Clear the request cache
    pub fn clear_cache(&mut self) {
        if let Ok(mut cache) = self.request_cache.lock() {
            cache.clear();
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, Option<usize>) {
        if let Ok(cache) = self.request_cache.lock() {
            (cache.len(), Some(cache.cap().into()))
        } else {
            (0, None)
        }
    }

    /// Get total cache cost
    pub fn get_cache_cost(&self) -> f64 {
        if let Ok(cache) = self.request_cache.lock() {
            cache.iter().map(|(_, cached)| cached.cost_usd).sum()
        } else {
            0.0
        }
    }

    /// Get API usage metrics
    pub fn get_metrics(&self) -> &OpenAIMetrics {
        &self.metrics
    }

    /// Reset metrics
    pub fn reset_metrics(&mut self) {
        self.metrics = OpenAIMetrics::default();
    }

    /// Estimate token count for text (approximate)
    fn estimate_tokens(&self, text: &str) -> u64 {
        // Rough estimation: ~4 characters per token on average
        // This is an approximation - actual tokenization depends on the model
        (text.len() / 4).max(1) as u64
    }

    /// Calculate cost for embeddings request
    fn calculate_cost_from_tokens(&self, total_tokens: u64) -> f64 {
        // OpenAI pricing (as of 2024) - these should be configurable
        let cost_per_1k_tokens = match self.openai_config.model.as_str() {
            "text-embedding-ada-002" => 0.0001,  // $0.0001 per 1K tokens
            "text-embedding-3-small" => 0.00002, // $0.00002 per 1K tokens
            "text-embedding-3-large" => 0.00013, // $0.00013 per 1K tokens
            _ => 0.0001,                         // Default to ada-002 pricing
        };

        (total_tokens as f64 / 1000.0) * cost_per_1k_tokens
    }

    /// Update metrics after successful request
    fn update_metrics_success(&mut self, texts: &[String]) {
        self.metrics.total_requests += 1;
        self.metrics.successful_requests += 1;

        let total_tokens: u64 = texts.iter().map(|text| self.estimate_tokens(text)).sum();

        self.metrics.total_tokens_processed += total_tokens;
        self.metrics.total_cost_usd += self.calculate_cost_from_tokens(total_tokens);
    }

    /// Update metrics after failed request
    fn update_metrics_failure(&mut self) {
        self.metrics.total_requests += 1;
        self.metrics.failed_requests += 1;
    }

    /// Update cache metrics
    fn update_cache_hit(&mut self) {
        self.metrics.cache_hits += 1;
    }

    fn update_cache_miss(&mut self) {
        self.metrics.cache_misses += 1;
    }
}

impl EmbeddingGenerator for OpenAIEmbeddingGenerator {
    fn generate(&self, content: &EmbeddableContent) -> Result<Vector> {
        // Check cache first (readonly access is fine)
        if self.openai_config.enable_cache {
            let hash = content.content_hash();
            if let Ok(mut cache) = self.request_cache.lock() {
                if let Some(cached) = cache.get(&hash) {
                    return Ok(cached.vector.clone());
                }
            }
        }

        // For synchronous interface with API calls, we need to use async runtime
        // This is a workaround for the trait design limitation
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| anyhow!("Failed to create async runtime: {}", e))?;

        // Create a temporary mutable copy for the async operation
        let mut temp_generator = OpenAIEmbeddingGenerator {
            config: self.config.clone(),
            openai_config: self.openai_config.clone(),
            client: self.client.clone(),
            rate_limiter: RateLimiter::new(self.openai_config.requests_per_minute),
            request_cache: self.request_cache.clone(),
            metrics: self.metrics.clone(),
        };

        rt.block_on(temp_generator.generate_async(content))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_model_types() {
        let config = EmbeddingConfig::default();

        // Test BERT
        let bert = SentenceTransformerGenerator::new(config.clone());
        assert!(matches!(bert.model_type(), TransformerModelType::BERT));
        assert_eq!(bert.dimensions(), 384); // Default config dimensions

        // Test RoBERTa
        let roberta = SentenceTransformerGenerator::roberta(config.clone());
        assert!(matches!(
            roberta.model_type(),
            TransformerModelType::RoBERTa
        ));

        // Test DistilBERT
        let distilbert = SentenceTransformerGenerator::distilbert(config.clone());
        assert!(matches!(
            distilbert.model_type(),
            TransformerModelType::DistilBERT
        ));
        assert_eq!(distilbert.dimensions(), 384); // DistilBERT uses smaller dimensions

        // Test Multilingual BERT
        let multibert = SentenceTransformerGenerator::multilingual_bert(config.clone());
        assert!(matches!(
            multibert.model_type(),
            TransformerModelType::MultiBERT
        ));
    }

    #[test]
    fn test_model_details() {
        let config = EmbeddingConfig::default();

        // Test BERT details
        let bert = SentenceTransformerGenerator::new(config.clone());
        let bert_details = bert.model_details();
        assert_eq!(bert_details.vocab_size, 30522);
        assert_eq!(bert_details.num_layers, 12);
        assert_eq!(bert_details.hidden_size, 768);
        assert!(bert_details.supports_languages.contains(&"en".to_string()));

        // Test RoBERTa details
        let roberta = SentenceTransformerGenerator::roberta(config.clone());
        let roberta_details = roberta.model_details();
        assert_eq!(roberta_details.vocab_size, 50265); // Larger vocab than BERT
        assert_eq!(roberta_details.max_position_embeddings, 514); // RoBERTa supports longer sequences

        // Test DistilBERT details
        let distilbert = SentenceTransformerGenerator::distilbert(config.clone());
        let distilbert_details = distilbert.model_details();
        assert_eq!(distilbert_details.num_layers, 6); // Half the layers of BERT
        assert_eq!(distilbert_details.hidden_size, 384); // Smaller hidden size
        assert!(distilbert_details.model_size_mb < bert_details.model_size_mb); // Smaller model
        assert!(
            distilbert_details.typical_inference_time_ms < bert_details.typical_inference_time_ms
        ); // Faster

        // Test Multilingual BERT details
        let multibert = SentenceTransformerGenerator::multilingual_bert(config.clone());
        let multibert_details = multibert.model_details();
        assert_eq!(multibert_details.vocab_size, 120000); // Much larger vocabulary
        assert!(multibert_details.supports_languages.len() > 10); // Supports many languages
        assert!(multibert_details
            .supports_languages
            .contains(&"zh".to_string())); // Chinese
        assert!(multibert_details
            .supports_languages
            .contains(&"de".to_string())); // German
    }

    #[test]
    fn test_language_support() {
        let config = EmbeddingConfig::default();

        // BERT and DistilBERT only support English
        let bert = SentenceTransformerGenerator::new(config.clone());
        assert!(bert.supports_language("en"));
        assert!(!bert.supports_language("zh"));
        assert!(!bert.supports_language("de"));

        let distilbert = SentenceTransformerGenerator::distilbert(config.clone());
        assert!(distilbert.supports_language("en"));
        assert!(!distilbert.supports_language("zh"));

        // Multilingual BERT supports many languages
        let multibert = SentenceTransformerGenerator::multilingual_bert(config.clone());
        assert!(multibert.supports_language("en"));
        assert!(multibert.supports_language("zh"));
        assert!(multibert.supports_language("de"));
        assert!(multibert.supports_language("fr"));
        assert!(multibert.supports_language("es"));
        assert!(!multibert.supports_language("unknown_lang"));
    }

    #[test]
    fn test_efficiency_ratings() {
        let config = EmbeddingConfig::default();

        let bert = SentenceTransformerGenerator::new(config.clone());
        let roberta = SentenceTransformerGenerator::roberta(config.clone());
        let distilbert = SentenceTransformerGenerator::distilbert(config.clone());
        let multibert = SentenceTransformerGenerator::multilingual_bert(config.clone());

        // DistilBERT should be the most efficient
        assert!(distilbert.efficiency_rating() > bert.efficiency_rating());
        assert!(distilbert.efficiency_rating() > roberta.efficiency_rating());
        assert!(distilbert.efficiency_rating() > multibert.efficiency_rating());

        // RoBERTa should be slightly less efficient than BERT
        assert!(bert.efficiency_rating() > roberta.efficiency_rating());

        // Multilingual BERT should be the least efficient
        assert!(bert.efficiency_rating() > multibert.efficiency_rating());
        assert!(roberta.efficiency_rating() > multibert.efficiency_rating());
    }

    #[test]
    fn test_inference_time_estimation() {
        let config = EmbeddingConfig::default();
        let distilbert = SentenceTransformerGenerator::distilbert(config.clone());
        let bert = SentenceTransformerGenerator::new(config.clone());

        // Short text
        let short_time_distilbert = distilbert.estimate_inference_time(50);
        let short_time_bert = bert.estimate_inference_time(50);

        // Long text
        let long_time_distilbert = distilbert.estimate_inference_time(500);
        let long_time_bert = bert.estimate_inference_time(500);

        // DistilBERT should be faster for both short and long texts
        assert!(short_time_distilbert < short_time_bert);
        assert!(long_time_distilbert < long_time_bert);

        // Longer texts should take more time
        assert!(long_time_distilbert > short_time_distilbert);
        assert!(long_time_bert > short_time_bert);
    }

    #[test]
    fn test_model_specific_text_preprocessing() {
        let config = EmbeddingConfig::default();

        let bert = SentenceTransformerGenerator::new(config.clone());
        let roberta = SentenceTransformerGenerator::roberta(config.clone());
        let multibert = SentenceTransformerGenerator::multilingual_bert(config.clone());

        let text = "Hello World";

        // BERT should use [CLS] and [SEP] tokens and lowercase
        let bert_processed = bert.preprocess_text_for_model(text, 512).unwrap();
        assert!(bert_processed.contains("[CLS]"));
        assert!(bert_processed.contains("[SEP]"));
        assert!(bert_processed.contains("hello world")); // Should be lowercase

        // RoBERTa should use <s> and </s> tokens and preserve case
        let roberta_processed = roberta.preprocess_text_for_model(text, 512).unwrap();
        assert!(roberta_processed.contains("<s>"));
        assert!(roberta_processed.contains("</s>"));
        assert!(roberta_processed.contains("Hello World")); // Should preserve case

        // Multilingual BERT should handle different scripts appropriately
        let latin_text = "Hello World";
        let chinese_text = "你好世界";

        let latin_processed = multibert
            .preprocess_text_for_model(latin_text, 512)
            .unwrap();
        let chinese_processed = multibert
            .preprocess_text_for_model(chinese_text, 512)
            .unwrap();

        assert!(latin_processed.contains("hello world")); // Latin should be lowercase
        assert!(chinese_processed.contains("你好世界")); // Chinese should preserve characters
    }

    #[test]
    fn test_embedding_generation_differences() {
        let config = EmbeddingConfig::default();

        let bert = SentenceTransformerGenerator::new(config.clone());
        let roberta = SentenceTransformerGenerator::roberta(config.clone());
        let distilbert = SentenceTransformerGenerator::distilbert(config.clone());

        let content = EmbeddableContent::Text("This is a test sentence".to_string());

        let bert_embedding = bert.generate(&content).unwrap();
        let roberta_embedding = roberta.generate(&content).unwrap();
        let distilbert_embedding = distilbert.generate(&content).unwrap();

        // Embeddings should be different between models
        assert_ne!(bert_embedding.as_f32(), roberta_embedding.as_f32());
        assert_ne!(bert_embedding.as_f32(), distilbert_embedding.as_f32());
        assert_ne!(roberta_embedding.as_f32(), distilbert_embedding.as_f32());

        // DistilBERT should have smaller dimensions
        assert_eq!(distilbert_embedding.dimensions, 384);
        assert_eq!(bert_embedding.dimensions, 384); // Using default config dimensions
        assert_eq!(roberta_embedding.dimensions, 384);

        // All embeddings should be normalized if config specifies it
        if config.normalize {
            let bert_magnitude: f32 = bert_embedding
                .as_f32()
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            let roberta_magnitude: f32 = roberta_embedding
                .as_f32()
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            let distilbert_magnitude: f32 = distilbert_embedding
                .as_f32()
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();

            assert!((bert_magnitude - 1.0).abs() < 0.1);
            assert!((roberta_magnitude - 1.0).abs() < 0.1);
            assert!((distilbert_magnitude - 1.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_tokenization_differences() {
        let config = EmbeddingConfig::default();

        let bert = SentenceTransformerGenerator::new(config.clone());
        let roberta = SentenceTransformerGenerator::roberta(config.clone());
        let multibert = SentenceTransformerGenerator::multilingual_bert(config.clone());

        let model_details_bert = bert.get_model_details();
        let model_details_roberta = roberta.get_model_details();
        let model_details_multibert = multibert.get_model_details();

        let complex_word = "preprocessing";

        // Test different tokenization approaches
        let bert_tokens =
            bert.simulate_wordpiece_tokenization(complex_word, model_details_bert.vocab_size);
        let roberta_tokens =
            roberta.simulate_bpe_tokenization(complex_word, model_details_roberta.vocab_size);
        let multibert_tokens = multibert
            .simulate_multilingual_tokenization(complex_word, model_details_multibert.vocab_size);

        // RoBERTa BPE should create more subword tokens for complex words
        assert!(roberta_tokens.len() >= bert_tokens.len());

        // Multilingual BERT should create fewer subwords due to larger vocabulary
        assert!(multibert_tokens.len() <= bert_tokens.len());

        // All tokenizations should produce valid token IDs
        for token in &bert_tokens {
            assert!(*token < model_details_bert.vocab_size as u32);
        }
        for token in &roberta_tokens {
            assert!(*token < model_details_roberta.vocab_size as u32);
        }
        for token in &multibert_tokens {
            assert!(*token < model_details_multibert.vocab_size as u32);
        }
    }

    #[test]
    fn test_model_size_comparisons() {
        let config = EmbeddingConfig::default();

        let bert = SentenceTransformerGenerator::new(config.clone());
        let roberta = SentenceTransformerGenerator::roberta(config.clone());
        let distilbert = SentenceTransformerGenerator::distilbert(config.clone());
        let multibert = SentenceTransformerGenerator::multilingual_bert(config.clone());

        let bert_size = bert.model_size_mb();
        let roberta_size = roberta.model_size_mb();
        let distilbert_size = distilbert.model_size_mb();
        let multibert_size = multibert.model_size_mb();

        // DistilBERT should be the smallest
        assert!(distilbert_size < bert_size);
        assert!(distilbert_size < roberta_size);
        assert!(distilbert_size < multibert_size);

        // Multilingual BERT should be the largest
        assert!(multibert_size > bert_size);
        assert!(multibert_size > roberta_size);
        assert!(multibert_size > distilbert_size);

        // RoBERTa should be slightly larger than BERT due to larger vocabulary
        assert!(roberta_size > bert_size);
    }
}
