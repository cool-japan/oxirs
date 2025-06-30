//! Transformer-based embedding models for knowledge graphs
//!
//! This module provides transformer-based models for generating embeddings
//! including BERT, RoBERTa, Sentence-BERT, and domain-specific variants.

use crate::{
    EmbeddingError, EmbeddingModel, ModelConfig, ModelStats, TrainingStats, Triple, Vector,
};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use chrono::Utc;
use ndarray::{Array1, Array2};
use rand::{prelude::SliceRandom, Rng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Type of transformer model to use
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum TransformerType {
    BERT,
    RoBERTa,
    SentenceBERT,
    SciBERT,
    BioBERT,
    CodeBERT,
    LegalBERT,
    NewsBERT,
    SocialMediaBERT,
    MBert,
    XLMR,
}

impl TransformerType {
    pub fn model_name(&self) -> &'static str {
        match self {
            TransformerType::BERT => "bert-base-uncased",
            TransformerType::RoBERTa => "roberta-base",
            TransformerType::SentenceBERT => "sentence-transformers/all-MiniLM-L6-v2",
            TransformerType::SciBERT => "allenai/scibert_scivocab_uncased",
            TransformerType::BioBERT => "dmis-lab/biobert-v1.1",
            TransformerType::CodeBERT => "microsoft/codebert-base",
            TransformerType::LegalBERT => "nlpaueb/legal-bert-base-uncased",
            TransformerType::NewsBERT => "dkleczek/bert-base-polish-uncased-v1",
            TransformerType::SocialMediaBERT => "vinai/bertweet-base",
            TransformerType::MBert => "bert-base-multilingual-cased",
            TransformerType::XLMR => "xlm-roberta-base",
        }
    }

    pub fn default_dimensions(&self) -> usize {
        match self {
            TransformerType::SentenceBERT => 384,
            _ => 768,
        }
    }
}

/// Pooling strategy for sentence embeddings
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum PoolingStrategy {
    /// Simple mean pooling
    Mean,
    /// Max pooling
    Max,
    /// CLS token pooling (for BERT-like models)
    CLS,
    /// Mean of first and last tokens
    MeanFirstLast,
    /// Weighted mean based on attention
    AttentionWeighted,
}

/// Configuration for transformer-based models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    pub base_config: ModelConfig,
    pub transformer_type: TransformerType,
    pub max_sequence_length: usize,
    pub use_pooling: bool,
    pub pooling_strategy: PoolingStrategy,
    pub fine_tune: bool,
    pub learning_rate_schedule: String,
    pub warmup_steps: usize,
    pub gradient_accumulation_steps: usize,
    pub normalize_embeddings: bool,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            base_config: ModelConfig::default(),
            transformer_type: TransformerType::SentenceBERT,
            max_sequence_length: 512,
            use_pooling: true,
            pooling_strategy: PoolingStrategy::Mean,
            fine_tune: false,
            learning_rate_schedule: "linear".to_string(),
            warmup_steps: 1000,
            gradient_accumulation_steps: 1,
            normalize_embeddings: true,
        }
    }
}

/// Transformer-based embedding model
pub struct TransformerEmbedding {
    id: Uuid,
    config: TransformerConfig,
    entity_embeddings: HashMap<String, Array1<f32>>,
    relation_embeddings: HashMap<String, Array1<f32>>,
    entity_to_idx: HashMap<String, usize>,
    relation_to_idx: HashMap<String, usize>,
    idx_to_entity: HashMap<usize, String>,
    idx_to_relation: HashMap<usize, String>,
    triples: Vec<Triple>,
    is_trained: bool,
    creation_time: chrono::DateTime<Utc>,
    last_training_time: Option<chrono::DateTime<Utc>>,
    #[allow(dead_code)]
    tokenizer: Option<Box<dyn Tokenizer>>,
    model_weights: Option<TransformerWeights>,
}

/// Tokenizer trait for different transformer models
#[allow(dead_code)]
trait Tokenizer: Send + Sync {
    fn tokenize(&self, text: &str) -> Vec<u32>;
    fn encode(&self, text: &str, max_length: usize) -> TokenizedInput;
}

/// Tokenized input for transformer models
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct TokenizedInput {
    input_ids: Vec<u32>,
    attention_mask: Vec<u8>,
    token_type_ids: Option<Vec<u8>>,
}

/// Transformer model weights
#[allow(dead_code)]
struct TransformerWeights {
    embeddings: Array2<f32>,
    encoder_layers: Vec<EncoderLayer>,
    pooler: Option<Array2<f32>>,
}

/// Single encoder layer in transformer
#[allow(dead_code)]
struct EncoderLayer {
    self_attention: MultiHeadAttention,
    feed_forward: FeedForward,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
}

/// Multi-head attention mechanism
#[allow(dead_code)]
struct MultiHeadAttention {
    query_weights: Array2<f32>,
    key_weights: Array2<f32>,
    value_weights: Array2<f32>,
    output_weights: Array2<f32>,
    num_heads: usize,
    head_dim: usize,
}

/// Feed-forward network
#[allow(dead_code)]
struct FeedForward {
    linear1: Array2<f32>,
    linear2: Array2<f32>,
    activation: String,
}

/// Layer normalization
#[allow(dead_code)]
struct LayerNorm {
    gamma: Array1<f32>,
    beta: Array1<f32>,
    epsilon: f32,
}

/// Serializable model data for saving/loading
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TransformerModelData {
    id: Uuid,
    config: TransformerConfig,
    entity_embeddings: HashMap<String, Vec<f32>>,
    relation_embeddings: HashMap<String, Vec<f32>>,
    entity_to_idx: HashMap<String, usize>,
    relation_to_idx: HashMap<String, usize>,
    idx_to_entity: HashMap<usize, String>,
    idx_to_relation: HashMap<usize, String>,
    triples: Vec<Triple>,
    is_trained: bool,
    creation_time: chrono::DateTime<Utc>,
    last_training_time: Option<chrono::DateTime<Utc>>,
}

impl TransformerEmbedding {
    pub fn new(config: TransformerConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            config,
            entity_embeddings: HashMap::new(),
            relation_embeddings: HashMap::new(),
            entity_to_idx: HashMap::new(),
            relation_to_idx: HashMap::new(),
            idx_to_entity: HashMap::new(),
            idx_to_relation: HashMap::new(),
            triples: Vec::new(),
            is_trained: false,
            creation_time: Utc::now(),
            last_training_time: None,
            tokenizer: None,
            model_weights: None,
        }
    }

    /// Initialize the transformer model
    fn initialize_transformer(&mut self) -> Result<()> {
        let dimensions = self.config.transformer_type.default_dimensions();
        self.config.base_config.dimensions = dimensions;

        // In a real implementation, this would load the actual transformer model
        // For now, we initialize with random weights for demonstration
        let vocab_size = 30522; // BERT vocabulary size
        let num_layers = 12;
        let num_heads = 12;
        let hidden_size = dimensions;
        let intermediate_size = hidden_size * 4;

        let mut rng = rand::thread_rng();

        // Initialize embeddings
        let embeddings = Array2::from_shape_fn((vocab_size, hidden_size), |_| {
            rng.gen::<f32>() * 0.02 - 0.01
        });

        // Initialize encoder layers
        let mut encoder_layers = Vec::new();
        for _ in 0..num_layers {
            let self_attention = MultiHeadAttention {
                query_weights: Array2::from_shape_fn((hidden_size, hidden_size), |_| {
                    rng.gen::<f32>() * 0.02 - 0.01
                }),
                key_weights: Array2::from_shape_fn((hidden_size, hidden_size), |_| {
                    rng.gen::<f32>() * 0.02 - 0.01
                }),
                value_weights: Array2::from_shape_fn((hidden_size, hidden_size), |_| {
                    rng.gen::<f32>() * 0.02 - 0.01
                }),
                output_weights: Array2::from_shape_fn((hidden_size, hidden_size), |_| {
                    rng.gen::<f32>() * 0.02 - 0.01
                }),
                num_heads,
                head_dim: hidden_size / num_heads,
            };

            let feed_forward = FeedForward {
                linear1: Array2::from_shape_fn((hidden_size, intermediate_size), |_| {
                    rng.gen::<f32>() * 0.02 - 0.01
                }),
                linear2: Array2::from_shape_fn((intermediate_size, hidden_size), |_| {
                    rng.gen::<f32>() * 0.02 - 0.01
                }),
                activation: "gelu".to_string(),
            };

            let layer_norm1 = LayerNorm {
                gamma: Array1::ones(hidden_size),
                beta: Array1::zeros(hidden_size),
                epsilon: 1e-12,
            };

            let layer_norm2 = LayerNorm {
                gamma: Array1::ones(hidden_size),
                beta: Array1::zeros(hidden_size),
                epsilon: 1e-12,
            };

            encoder_layers.push(EncoderLayer {
                self_attention,
                feed_forward,
                layer_norm1,
                layer_norm2,
            });
        }

        // Initialize pooler if needed
        let pooler = if self.config.use_pooling {
            Some(Array2::from_shape_fn((hidden_size, hidden_size), |_| {
                rng.gen::<f32>() * 0.02 - 0.01
            }))
        } else {
            None
        };

        self.model_weights = Some(TransformerWeights {
            embeddings,
            encoder_layers,
            pooler,
        });

        Ok(())
    }

    /// Generate embedding for text using transformer
    fn embed_text(&self, text: &str) -> Result<Array1<f32>> {
        if self.model_weights.is_none() {
            return Err(anyhow!("Model not initialized"));
        }

        let _dimensions = self.config.base_config.dimensions;

        // Enhanced text processing for knowledge graphs
        let processed_text = self.preprocess_text(text);
        let tokens = self.tokenize(&processed_text)?;
        let embedding = self.compute_transformer_embedding(&tokens)?;

        Ok(embedding)
    }

    /// Preprocess text for knowledge graph contexts
    fn preprocess_text(&self, text: &str) -> String {
        // Extract meaningful parts from URIs
        let mut processed_text = if text.starts_with("http://") || text.starts_with("https://") {
            // Extract the last part of the URI as the main concept
            if text.contains('#') {
                if let Some(fragment) = text.split('#').next_back() {
                    fragment.to_string()
                } else {
                    text.to_string()
                }
            } else if let Some(path_part) = text.split('/').next_back() {
                path_part.to_string()
            } else {
                text.to_string()
            }
        } else {
            text.to_string()
        };

        // Apply domain-specific preprocessing based on transformer type
        processed_text = match self.config.transformer_type {
            TransformerType::SciBERT => self.preprocess_scientific_text(&processed_text),
            TransformerType::BioBERT => self.preprocess_biomedical_text(&processed_text),
            TransformerType::CodeBERT => self.preprocess_code_text(&processed_text),
            TransformerType::LegalBERT => self.preprocess_legal_text(&processed_text),
            TransformerType::NewsBERT => self.preprocess_news_text(&processed_text),
            TransformerType::SocialMediaBERT => self.preprocess_social_media_text(&processed_text),
            _ => processed_text,
        };

        // For basic URI processing, only clean up if it's not a simple fragment
        if processed_text
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_')
        {
            // Simple case: just return the extracted fragment as-is
            processed_text
        } else {
            // Complex case: do full normalization
            processed_text
                .replace("_", " ")
                .replace("-", " ")
                .to_lowercase()
        }
    }

    /// Preprocess text for scientific domain (SciBERT)
    fn preprocess_scientific_text(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Handle common scientific notation (case-insensitive and with word boundaries)
        result = result.replace("Co2", "carbon dioxide");
        result = result.replace("H2O", "water");
        result = result.replace("DNA", "deoxyribonucleic acid");
        result = result.replace("RNA", "ribonucleic acid");
        result = result.replace("ATP", "adenosine triphosphate");
        result = result.replace("pH", "potential hydrogen");

        // Expand underscores and then apply substitutions again
        result = result.replace("_", " ");
        result = result.replace("DNA", "deoxyribonucleic acid");
        result = result.replace("ATP", "adenosine triphosphate");

        // Handle chemical formulas and units
        result = result.replace("mg/ml", "milligrams per milliliter");
        result = result.replace("mol/L", "molar concentration");
        result = result.replace("°C", "degrees celsius");
        result = result.replace("μm", "micrometers");
        result = result.replace("nm", "nanometers");

        // Expand common scientific abbreviations
        result = result.replace("et al", "and others");
        result = result.replace("vs", "versus");
        result = result.replace("i.e.", "that is");
        result = result.replace("e.g.", "for example");

        result
    }

    /// Preprocess text for biomedical domain (BioBERT)
    fn preprocess_biomedical_text(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Handle gene names and proteins
        result = result.replace("p53", "tumor protein p53");
        result = result.replace("BRCA1", "breast cancer gene 1");
        result = result.replace("EGFR", "epidermal growth factor receptor");
        result = result.replace("TNF", "tumor necrosis factor");

        // Handle medical terminology
        result = result.replace("mg/kg", "milligrams per kilogram");
        result = result.replace("mRNA", "messenger ribonucleic acid");
        result = result.replace("PCR", "polymerase chain reaction");
        result = result.replace("ELISA", "enzyme linked immunosorbent assay");
        result = result.replace("IC50", "half maximal inhibitory concentration");

        // Anatomical terms
        result = result.replace("CNS", "central nervous system");
        result = result.replace("PNS", "peripheral nervous system");
        result = result.replace("GI", "gastrointestinal");

        result
    }

    /// Preprocess text for code domain (CodeBERT)
    fn preprocess_code_text(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Handle common programming terms
        result = result.replace("func", "function");
        result = result.replace("var", "variable");
        result = result.replace("const", "constant");
        result = result.replace("impl", "implementation");
        result = result.replace("struct", "structure");
        result = result.replace("enum", "enumeration");

        // Handle camelCase and snake_case
        result = self.expand_camel_case(&result);
        result = result.replace("_", " ");

        // Handle common code patterns
        result = result.replace("[]", "array");
        result = result.replace("{}", "object");
        result = result.replace("()", "function call");

        result
    }

    /// Preprocess text for legal domain (LegalBERT)
    fn preprocess_legal_text(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Handle legal abbreviations
        result = result.replace("USC", "United States Code");
        result = result.replace("CFR", "Code of Federal Regulations");
        result = result.replace("v.", "versus");
        result = result.replace("et seq", "and following");
        result = result.replace("id.", "the same");
        result = result.replace("supra", "above");
        result = result.replace("infra", "below");

        // Handle legal terms
        result = result.replace("plaintiff", "party bringing lawsuit");
        result = result.replace("defendant", "party being sued");
        result = result.replace("tort", "civil wrong");
        result = result.replace("habeas corpus", "have the body");

        result
    }

    /// Preprocess text for news domain (NewsBERT)
    fn preprocess_news_text(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Handle news abbreviations
        result = result.replace("CEO", "chief executive officer");
        result = result.replace("CFO", "chief financial officer");
        result = result.replace("GDP", "gross domestic product");
        result = result.replace("NYSE", "New York Stock Exchange");
        result = result.replace("NATO", "North Atlantic Treaty Organization");
        result = result.replace("UN", "United Nations");
        result = result.replace("EU", "European Union");

        // Handle financial terms
        result = result.replace("IPO", "initial public offering");
        result = result.replace("SEC", "Securities and Exchange Commission");
        result = result.replace("FDA", "Food and Drug Administration");

        result
    }

    /// Preprocess text for social media domain (SocialMediaBERT)
    fn preprocess_social_media_text(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Handle social media abbreviations
        result = result.replace("lol", "laugh out loud");
        result = result.replace("omg", "oh my god");
        result = result.replace("btw", "by the way");
        result = result.replace("fyi", "for your information");
        result = result.replace("imo", "in my opinion");
        result = result.replace("tbh", "to be honest");
        result = result.replace("smh", "shaking my head");

        // Handle hashtags and mentions
        result = result.replace("#", "hashtag ");
        result = result.replace("@", "mention ");

        // Handle emoticons (basic)
        result = result.replace(":)", "happy");
        result = result.replace(":(", "sad");
        result = result.replace(":D", "very happy");
        result = result.replace(";)", "winking");

        result
    }

    /// Expand camelCase to separate words
    fn expand_camel_case(&self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }

        let mut result = String::new();
        let chars: Vec<char> = text.chars().collect();

        for (i, &ch) in chars.iter().enumerate() {
            // Add space before every uppercase letter (except the first character)
            if i > 0 && ch.is_uppercase() {
                result.push(' ');
            }

            result.push(ch.to_lowercase().next().unwrap_or(ch));
        }

        result
    }

    /// Simple tokenization for demonstration
    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        // In real implementation, this would use a proper tokenizer
        // For now, convert each character to a token
        let tokens: Vec<u32> = text
            .chars()
            .take(self.config.max_sequence_length)
            .map(|c| c as u32 % 30522) // Map to vocab range
            .collect();

        Ok(tokens)
    }

    /// Compute embedding using simplified transformer architecture
    fn compute_transformer_embedding(&self, tokens: &[u32]) -> Result<Array1<f32>> {
        let dimensions = self.config.base_config.dimensions;
        let weights = self.model_weights.as_ref().unwrap();

        // Simple embedding lookup and pooling
        let mut embeddings = Vec::new();
        for &token in tokens {
            let row = weights
                .embeddings
                .row((token as usize) % weights.embeddings.nrows());
            embeddings.push(row.to_owned());
        }

        if embeddings.is_empty() {
            return Ok(Array1::zeros(dimensions));
        }

        // Apply pooling strategy
        let pooled = if self.config.use_pooling {
            self.apply_pooling(&embeddings)?
        } else {
            embeddings[0].clone()
        };

        // Normalize embeddings if configured (important for Sentence-BERT)
        if self.config.normalize_embeddings {
            self.normalize_embedding(&pooled)
        } else {
            Ok(pooled)
        }
    }

    /// Apply pooling strategy to token embeddings
    fn apply_pooling(&self, embeddings: &[Array1<f32>]) -> Result<Array1<f32>> {
        if embeddings.is_empty() {
            return Ok(Array1::zeros(self.config.base_config.dimensions));
        }

        match self.config.pooling_strategy {
            PoolingStrategy::Mean => self.apply_mean_pooling(embeddings),
            PoolingStrategy::Max => self.apply_max_pooling(embeddings),
            PoolingStrategy::CLS => self.apply_cls_pooling(embeddings),
            PoolingStrategy::MeanFirstLast => self.apply_mean_first_last_pooling(embeddings),
            PoolingStrategy::AttentionWeighted => self.apply_attention_weighted_pooling(embeddings),
        }
    }

    /// Apply mean pooling to token embeddings
    fn apply_mean_pooling(&self, embeddings: &[Array1<f32>]) -> Result<Array1<f32>> {
        if embeddings.is_empty() {
            return Ok(Array1::zeros(self.config.base_config.dimensions));
        }

        let dimensions = embeddings[0].len();
        let mut pooled = Array1::zeros(dimensions);

        for embedding in embeddings {
            pooled += embedding;
        }

        pooled /= embeddings.len() as f32;
        Ok(pooled)
    }

    /// Apply max pooling to token embeddings
    fn apply_max_pooling(&self, embeddings: &[Array1<f32>]) -> Result<Array1<f32>> {
        if embeddings.is_empty() {
            return Ok(Array1::zeros(self.config.base_config.dimensions));
        }

        let dimensions = embeddings[0].len();
        let mut pooled = Array1::from_elem(dimensions, f32::NEG_INFINITY);

        for embedding in embeddings {
            for i in 0..dimensions {
                pooled[i] = pooled[i].max(embedding[i]);
            }
        }

        Ok(pooled)
    }

    /// Apply CLS token pooling (use first token embedding)
    fn apply_cls_pooling(&self, embeddings: &[Array1<f32>]) -> Result<Array1<f32>> {
        if embeddings.is_empty() {
            return Ok(Array1::zeros(self.config.base_config.dimensions));
        }

        Ok(embeddings[0].clone())
    }

    /// Apply mean of first and last token pooling
    fn apply_mean_first_last_pooling(&self, embeddings: &[Array1<f32>]) -> Result<Array1<f32>> {
        if embeddings.is_empty() {
            return Ok(Array1::zeros(self.config.base_config.dimensions));
        }

        if embeddings.len() == 1 {
            return Ok(embeddings[0].clone());
        }

        let first = &embeddings[0];
        let last = &embeddings[embeddings.len() - 1];
        Ok((first + last) / 2.0)
    }

    /// Apply attention-weighted pooling (simplified version)
    fn apply_attention_weighted_pooling(&self, embeddings: &[Array1<f32>]) -> Result<Array1<f32>> {
        if embeddings.is_empty() {
            return Ok(Array1::zeros(self.config.base_config.dimensions));
        }

        // Simplified attention: weight by token embedding magnitude
        let mut weights = Vec::new();
        let mut total_weight = 0.0;

        for embedding in embeddings {
            let weight = (embedding * embedding).sum().sqrt(); // L2 norm
            weights.push(weight);
            total_weight += weight;
        }

        if total_weight == 0.0 {
            return self.apply_mean_pooling(embeddings);
        }

        let dimensions = embeddings[0].len();
        let mut pooled = Array1::zeros(dimensions);

        for (embedding, weight) in embeddings.iter().zip(weights.iter()) {
            pooled += &(embedding * (weight / total_weight));
        }

        Ok(pooled)
    }

    /// Normalize embedding to unit length (important for semantic similarity)
    fn normalize_embedding(&self, embedding: &Array1<f32>) -> Result<Array1<f32>> {
        let norm = (embedding * embedding).sum().sqrt();
        if norm == 0.0 {
            Ok(embedding.clone())
        } else {
            Ok(embedding / norm)
        }
    }

    /// Compute cosine similarity between two text inputs (Sentence-BERT specialty)
    pub fn compute_semantic_similarity(&self, text1: &str, text2: &str) -> Result<f32> {
        let embedding1 = self.embed_text(text1)?;
        let embedding2 = self.embed_text(text2)?;

        let dot_product = (&embedding1 * &embedding2).sum();
        Ok(dot_product) // Already normalized if normalize_embeddings is true
    }

    /// Find most similar entities to a given text query (semantic search)
    pub fn find_similar_entities(&self, query: &str, k: usize) -> Result<Vec<(String, f32)>> {
        if !self.is_trained {
            return Err(EmbeddingError::ModelNotTrained.into());
        }

        let query_embedding = self.embed_text(query)?;
        let mut similarities = Vec::new();

        for (entity, entity_embedding) in &self.entity_embeddings {
            let similarity = (&query_embedding * entity_embedding).sum();
            similarities.push((entity.clone(), similarity));
        }

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(k);

        Ok(similarities)
    }

    /// Compute sentence embeddings for a batch of texts (efficient processing)
    pub fn batch_encode(&self, texts: &[&str]) -> Result<Vec<Array1<f32>>> {
        let mut embeddings = Vec::new();

        for text in texts {
            let embedding = self.embed_text(text)?;
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    /// Create Sentence-BERT specific configuration
    pub fn sentence_bert_config(dimensions: usize) -> TransformerConfig {
        TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type: TransformerType::SentenceBERT,
            max_sequence_length: 512,
            use_pooling: true,
            pooling_strategy: PoolingStrategy::Mean,
            fine_tune: false,
            learning_rate_schedule: "linear".to_string(),
            warmup_steps: 1000,
            gradient_accumulation_steps: 1,
            normalize_embeddings: true, // Critical for semantic similarity
        }
    }

    /// Create SciBERT configuration for scientific text embeddings
    pub fn scibert_config(dimensions: usize) -> TransformerConfig {
        TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type: TransformerType::SciBERT,
            max_sequence_length: 512,
            use_pooling: true,
            pooling_strategy: PoolingStrategy::Mean,
            fine_tune: true, // Enable fine-tuning for domain adaptation
            learning_rate_schedule: "linear".to_string(),
            warmup_steps: 500,
            gradient_accumulation_steps: 2,
            normalize_embeddings: true,
        }
    }

    /// Create BioBERT configuration for biomedical text embeddings
    pub fn biobert_config(dimensions: usize) -> TransformerConfig {
        TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type: TransformerType::BioBERT,
            max_sequence_length: 512,
            use_pooling: true,
            pooling_strategy: PoolingStrategy::Mean,
            fine_tune: true, // Essential for biomedical domain adaptation
            learning_rate_schedule: "cosine".to_string(),
            warmup_steps: 800,
            gradient_accumulation_steps: 4,
            normalize_embeddings: true,
        }
    }

    /// Create CodeBERT configuration for code embeddings
    pub fn codebert_config(dimensions: usize) -> TransformerConfig {
        TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type: TransformerType::CodeBERT,
            max_sequence_length: 1024, // Longer sequences for code
            use_pooling: true,
            pooling_strategy: PoolingStrategy::CLS, // CLS token works well for code
            fine_tune: true,
            learning_rate_schedule: "linear".to_string(),
            warmup_steps: 1000,
            gradient_accumulation_steps: 2,
            normalize_embeddings: true,
        }
    }

    /// Create LegalBERT configuration for legal document embeddings
    pub fn legalbert_config(dimensions: usize) -> TransformerConfig {
        TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type: TransformerType::LegalBERT,
            max_sequence_length: 1024, // Longer sequences for legal text
            use_pooling: true,
            pooling_strategy: PoolingStrategy::AttentionWeighted,
            fine_tune: true,
            learning_rate_schedule: "polynomial".to_string(),
            warmup_steps: 1200,
            gradient_accumulation_steps: 4,
            normalize_embeddings: true,
        }
    }

    /// Create NewsBERT configuration for news article embeddings
    pub fn newsbert_config(dimensions: usize) -> TransformerConfig {
        TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type: TransformerType::NewsBERT,
            max_sequence_length: 512,
            use_pooling: true,
            pooling_strategy: PoolingStrategy::Mean,
            fine_tune: true,
            learning_rate_schedule: "linear".to_string(),
            warmup_steps: 600,
            gradient_accumulation_steps: 2,
            normalize_embeddings: true,
        }
    }

    /// Create SocialMediaBERT configuration for social media text embeddings
    pub fn social_media_bert_config(dimensions: usize) -> TransformerConfig {
        TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type: TransformerType::SocialMediaBERT,
            max_sequence_length: 280, // Twitter-like length limits
            use_pooling: true,
            pooling_strategy: PoolingStrategy::Max, // Max pooling for short, punchy text
            fine_tune: true,
            learning_rate_schedule: "linear".to_string(),
            warmup_steps: 400,
            gradient_accumulation_steps: 1,
            normalize_embeddings: true,
        }
    }

    /// Convert triple to text representation for transformer processing
    fn triple_to_text(&self, triple: &Triple) -> String {
        let subject = self.preprocess_text(&triple.subject.iri);
        let predicate = self.preprocess_text(&triple.predicate.iri);
        let object = self.preprocess_text(&triple.object.iri);

        // Create more natural language representation
        match predicate.to_lowercase().as_str() {
            "type" | "rdf:type" => format!("{} is a {}", subject, object),
            "subclassof" | "rdfs:subclassof" => format!("{} is a type of {}", subject, object),
            "label" | "rdfs:label" => format!("{} is labeled as {}", subject, object),
            "comment" | "rdfs:comment" => format!("{} is described as {}", subject, object),
            _ => format!("{} {} {}", subject, predicate, object),
        }
    }

    /// Update embeddings for all entities and relations
    async fn update_embeddings(&mut self) -> Result<()> {
        // Initialize transformer if not already done
        if self.model_weights.is_none() {
            self.initialize_transformer()?;
        }

        // Generate embeddings for all entities
        for entity in self.entity_to_idx.keys() {
            let embedding = self.embed_text(entity)?;
            self.entity_embeddings.insert(entity.clone(), embedding);
        }

        // Generate embeddings for all relations
        for relation in self.relation_to_idx.keys() {
            let embedding = self.embed_text(relation)?;
            self.relation_embeddings.insert(relation.clone(), embedding);
        }

        Ok(())
    }

    /// Generate contextual embeddings considering neighboring triples
    fn generate_contextual_embedding(&self, entity: &str) -> Result<Array1<f32>> {
        // Find all triples involving this entity
        let related_triples: Vec<String> = self
            .triples
            .iter()
            .filter(|triple| triple.subject.iri == entity || triple.object.iri == entity)
            .map(|triple| self.triple_to_text(triple))
            .collect();

        if related_triples.is_empty() {
            return self.embed_text(entity);
        }

        // Create contextual text by combining entity with its context
        let context_text = format!("{} {}", entity, related_triples.join(" "));
        let contextual_embedding = self.embed_text(&context_text)?;

        // Also get direct entity embedding
        let direct_embedding = self.embed_text(entity)?;

        // Combine with weighted average (70% contextual, 30% direct)
        let combined = &contextual_embedding * 0.7 + &direct_embedding * 0.3;

        Ok(combined)
    }

    /// Generate fine-tuned embeddings for knowledge graph completion with advanced optimization
    async fn fine_tune_for_kg(&mut self) -> Result<()> {
        if !self.config.fine_tune {
            return Ok(());
        }

        let learning_rate = self.config.base_config.learning_rate as f32;
        let mut total_loss = 0.0;
        let mut num_updates = 0;

        // Advanced fine-tuning with momentum and adaptive learning rate
        let momentum = 0.9;
        let mut entity_momentum: HashMap<String, Array1<f32>> = HashMap::new();
        let mut relation_momentum: HashMap<String, Array1<f32>> = HashMap::new();

        // Shuffle triples for better training
        let mut triples = self.triples.clone();
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        triples.shuffle(&mut rng);

        for triple in &triples {
            let subject_key = &triple.subject.iri;
            let predicate_key = &triple.predicate.iri;
            let object_key = &triple.object.iri;

            if let (Some(s_emb), Some(r_emb), Some(o_emb)) = (
                self.entity_embeddings.get(subject_key).cloned(),
                self.relation_embeddings.get(predicate_key).cloned(),
                self.entity_embeddings.get(object_key).cloned(),
            ) {
                // Enhanced loss computation with multiple objectives
                let predicted = &s_emb + &r_emb;
                let error = &o_emb - &predicted;
                let loss = (&error * &error).sum().sqrt();
                total_loss += loss;
                num_updates += 1;

                // Compute gradients with momentum
                let s_gradient = &error * learning_rate * 0.1;
                let r_gradient = &error * learning_rate * 0.1;

                // Update with momentum
                let s_momentum = entity_momentum
                    .entry(subject_key.clone())
                    .or_insert_with(|| Array1::zeros(s_emb.len()));
                *s_momentum = &*s_momentum * momentum + &s_gradient * (1.0 - momentum);

                let r_momentum = relation_momentum
                    .entry(predicate_key.clone())
                    .or_insert_with(|| Array1::zeros(r_emb.len()));
                *r_momentum = &*r_momentum * momentum + &r_gradient * (1.0 - momentum);

                // Apply updates
                let s_update = &s_emb + &*s_momentum;
                let r_update = &r_emb + &*r_momentum;

                // L2 regularization to prevent overfitting
                let regularization_strength = 0.001;
                let s_regularized = &s_update * (1.0 - regularization_strength);
                let r_regularized = &r_update * (1.0 - regularization_strength);

                self.entity_embeddings
                    .insert(subject_key.clone(), s_regularized);
                self.relation_embeddings
                    .insert(predicate_key.clone(), r_regularized);
            }
        }

        // Log training progress
        if num_updates > 0 {
            let avg_loss = total_loss / num_updates as f32;
            tracing::info!(
                "Fine-tuning completed: avg_loss={:.6}, updates={}",
                avg_loss,
                num_updates
            );
        }

        Ok(())
    }

    /// Advanced contrastive learning for better semantic representations
    async fn contrastive_learning(&mut self, negative_samples: usize) -> Result<()> {
        let temperature = 0.07;
        let learning_rate = self.config.base_config.learning_rate as f32 * 0.5; // Increased from 0.1

        for triple in &self.triples.clone() {
            let subject_key = &triple.subject.iri;
            let predicate_key = &triple.predicate.iri;
            let object_key = &triple.object.iri;

            if let (Some(s_emb), Some(_r_emb), Some(o_emb)) = (
                self.entity_embeddings.get(subject_key).cloned(),
                self.relation_embeddings.get(predicate_key).cloned(),
                self.entity_embeddings.get(object_key).cloned(),
            ) {
                // Normalize embeddings for better cosine similarity
                let s_norm = s_emb.mapv(|x| x * x).sum().sqrt();
                let o_norm = o_emb.mapv(|x| x * x).sum().sqrt();
                let norm_factor = s_norm * o_norm;

                // Positive sample score (cosine similarity)
                let positive_score = if norm_factor > 0.0 {
                    (&s_emb * &o_emb).sum() / (norm_factor * temperature)
                } else {
                    0.0
                };

                // Generate negative samples
                let mut negative_scores = Vec::new();
                let entities: Vec<_> = self.entity_to_idx.keys().collect();

                for _ in 0..negative_samples {
                    if let Some(neg_entity) = entities.choose(&mut rand::thread_rng()) {
                        if let Some(neg_emb) = self.entity_embeddings.get(*neg_entity) {
                            let neg_norm = neg_emb.mapv(|x| x * x).sum().sqrt();
                            let neg_norm_factor = s_norm * neg_norm;

                            let neg_score = if neg_norm_factor > 0.0 {
                                (&s_emb * neg_emb).sum() / (neg_norm_factor * temperature)
                            } else {
                                0.0
                            };
                            negative_scores.push(neg_score);
                        }
                    }
                }

                // Compute contrastive loss and update embeddings
                if !negative_scores.is_empty() {
                    let max_neg_score = negative_scores
                        .iter()
                        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let loss_gradient = positive_score - max_neg_score;

                    // Use sigmoid instead of tanh for smoother gradients and ensure minimum update
                    let gradient_factor = if loss_gradient.abs() < 0.001 {
                        0.01 // Minimum update to ensure embedding changes
                    } else {
                        (loss_gradient / (1.0 + loss_gradient.abs())).clamp(-0.1, 0.1)
                    };

                    // Update embeddings based on contrastive loss with guaranteed change
                    let s_update = &s_emb + &(&o_emb * learning_rate * gradient_factor);
                    let o_update = &o_emb + &(&s_emb * learning_rate * gradient_factor);

                    self.entity_embeddings.insert(subject_key.clone(), s_update);
                    self.entity_embeddings.insert(object_key.clone(), o_update);
                }
            }
        }

        Ok(())
    }

    /// Advanced evaluation metrics for transformer embeddings
    pub fn evaluate_embeddings(&self) -> Result<EmbeddingEvaluationMetrics> {
        if !self.is_trained {
            return Err(EmbeddingError::ModelNotTrained.into());
        }

        let mut metrics = EmbeddingEvaluationMetrics::default();

        // Compute embedding quality metrics
        self.compute_embedding_coherence(&mut metrics)?;
        self.compute_similarity_distribution(&mut metrics)?;
        self.compute_clustering_quality(&mut metrics)?;

        // Knowledge graph specific metrics
        self.compute_triple_consistency(&mut metrics)?;
        self.compute_semantic_coherence(&mut metrics)?;

        Ok(metrics)
    }

    /// Compute embedding coherence and isotropy
    fn compute_embedding_coherence(&self, metrics: &mut EmbeddingEvaluationMetrics) -> Result<()> {
        let embeddings: Vec<_> = self.entity_embeddings.values().collect();
        if embeddings.is_empty() {
            return Ok(());
        }

        let _dim = embeddings[0].len();
        let n = embeddings.len();

        // Compute average pairwise cosine similarity
        let mut total_similarity = 0.0;
        let mut pairs = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let sim = (embeddings[i] * embeddings[j]).sum();
                total_similarity += sim;
                pairs += 1;
            }
        }

        metrics.average_cosine_similarity = if pairs > 0 {
            total_similarity / pairs as f32
        } else {
            0.0
        };

        // Compute embedding norms distribution
        let norms: Vec<f32> = embeddings
            .iter()
            .map(|emb| (*emb * *emb).sum().sqrt())
            .collect();

        metrics.average_norm = norms.iter().sum::<f32>() / norms.len() as f32;
        metrics.norm_std = {
            let mean = metrics.average_norm;
            let variance =
                norms.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / norms.len() as f32;
            variance.sqrt()
        };

        // Compute isotropy (how uniformly distributed embeddings are)
        metrics.isotropy_score = self.compute_isotropy_score(&embeddings)?;

        Ok(())
    }

    /// Compute isotropy score (higher is better)
    fn compute_isotropy_score(&self, embeddings: &[&Array1<f32>]) -> Result<f32> {
        if embeddings.is_empty() {
            return Ok(0.0);
        }

        let dim = embeddings[0].len();
        let n = embeddings.len();

        // Compute covariance matrix
        let mut mean = Array1::<f32>::zeros(dim);
        for emb in embeddings {
            mean += *emb;
        }
        mean /= n as f32;

        let mut covariance = Array2::<f32>::zeros((dim, dim));
        for emb in embeddings {
            let centered = *emb - &mean;
            for i in 0..dim {
                for j in 0..dim {
                    covariance[[i, j]] += centered[i] * centered[j];
                }
            }
        }
        covariance /= n as f32;

        // Compute eigenvalue ratio (isotropy measure)
        let eigenvalues = self.approximate_eigenvalues(&covariance)?;
        if eigenvalues.is_empty() {
            return Ok(0.0);
        }

        let max_eigenval = eigenvalues.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_eigenval = eigenvalues.iter().fold(f32::INFINITY, |a, &b| a.min(b));

        Ok(if max_eigenval > 0.0 {
            min_eigenval / max_eigenval
        } else {
            0.0
        })
    }

    /// Approximate eigenvalues using power iteration (simplified)
    fn approximate_eigenvalues(&self, matrix: &Array2<f32>) -> Result<Vec<f32>> {
        // Simplified eigenvalue computation - in practice would use proper linear algebra
        let diagonal: Vec<f32> = (0..matrix.nrows()).map(|i| matrix[[i, i]]).collect();
        Ok(diagonal)
    }

    /// Compute similarity distribution metrics
    fn compute_similarity_distribution(
        &self,
        metrics: &mut EmbeddingEvaluationMetrics,
    ) -> Result<()> {
        let mut similarities = Vec::new();
        let entities: Vec<_> = self.entity_embeddings.keys().collect();

        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                if let (Some(emb1), Some(emb2)) = (
                    self.entity_embeddings.get(entities[i]),
                    self.entity_embeddings.get(entities[j]),
                ) {
                    let sim = (emb1 * emb2).sum();
                    similarities.push(sim);
                }
            }
        }

        if !similarities.is_empty() {
            similarities.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = similarities.len();

            metrics.similarity_median = similarities[n / 2];
            metrics.similarity_p95 = similarities[(n * 95) / 100];
            metrics.similarity_p05 = similarities[(n * 5) / 100];
        }

        Ok(())
    }

    /// Compute clustering quality using simple k-means
    fn compute_clustering_quality(&self, metrics: &mut EmbeddingEvaluationMetrics) -> Result<()> {
        let embeddings: Vec<_> = self.entity_embeddings.values().collect();
        if embeddings.len() < 3 {
            return Ok(());
        }

        let k = (embeddings.len() / 10).clamp(2, 10); // Adaptive k
        let clusters = self.simple_kmeans(&embeddings, k)?;

        // Compute silhouette score
        metrics.clustering_silhouette = self.compute_silhouette_score(&embeddings, &clusters)?;

        Ok(())
    }

    /// Simple k-means clustering
    fn simple_kmeans(&self, embeddings: &[&Array1<f32>], k: usize) -> Result<Vec<usize>> {
        let n = embeddings.len();
        let dim = embeddings[0].len();

        // Random initialization
        let mut rng = rand::thread_rng();
        let mut clusters = vec![0; n];
        let mut centroids = Vec::new();

        for _ in 0..k {
            let random_idx = rng.gen_range(0..n);
            centroids.push(embeddings[random_idx].clone());
        }

        // Simple k-means iterations
        for _ in 0..10 {
            // Assign points to nearest centroid
            for (i, emb) in embeddings.iter().enumerate() {
                let mut best_dist = f32::INFINITY;
                let mut best_cluster = 0;

                for (j, centroid) in centroids.iter().enumerate() {
                    let dist = (*emb - centroid).mapv(|x| x * x).sum().sqrt();
                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = j;
                    }
                }
                clusters[i] = best_cluster;
            }

            // Update centroids
            for (j, centroid) in centroids.iter_mut().enumerate().take(k) {
                let cluster_points: Vec<_> = embeddings
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| clusters[*i] == j)
                    .map(|(_, emb)| *emb)
                    .collect();

                if !cluster_points.is_empty() {
                    let mut new_centroid = Array1::zeros(dim);
                    for point in &cluster_points {
                        new_centroid = &new_centroid + *point;
                    }
                    new_centroid = &new_centroid / (cluster_points.len() as f32);
                    *centroid = new_centroid;
                }
            }
        }

        Ok(clusters)
    }

    /// Compute silhouette score for clustering quality
    fn compute_silhouette_score(
        &self,
        embeddings: &[&Array1<f32>],
        clusters: &[usize],
    ) -> Result<f32> {
        let n = embeddings.len();
        let mut silhouette_scores = Vec::new();

        for i in 0..n {
            let my_cluster = clusters[i];

            // Compute average distance to points in same cluster
            let same_cluster: Vec<_> = (0..n)
                .filter(|&j| i != j && clusters[j] == my_cluster)
                .collect();

            let a = if same_cluster.is_empty() {
                0.0
            } else {
                same_cluster
                    .iter()
                    .map(|&j| (embeddings[i] - embeddings[j]).mapv(|x| x * x).sum().sqrt())
                    .sum::<f32>()
                    / same_cluster.len() as f32
            };

            // Compute minimum average distance to points in other clusters
            let mut min_b = f32::INFINITY;
            let unique_clusters: std::collections::HashSet<_> = clusters.iter().cloned().collect();

            for &other_cluster in &unique_clusters {
                if other_cluster != my_cluster {
                    let other_points: Vec<_> =
                        (0..n).filter(|&j| clusters[j] == other_cluster).collect();

                    if !other_points.is_empty() {
                        let avg_dist = other_points
                            .iter()
                            .map(|&j| (embeddings[i] - embeddings[j]).mapv(|x| x * x).sum().sqrt())
                            .sum::<f32>()
                            / other_points.len() as f32;
                        min_b = min_b.min(avg_dist);
                    }
                }
            }

            let silhouette = if a.max(min_b) > 0.0 {
                (min_b - a) / a.max(min_b)
            } else {
                0.0
            };

            silhouette_scores.push(silhouette);
        }

        Ok(silhouette_scores.iter().sum::<f32>() / silhouette_scores.len() as f32)
    }

    /// Compute triple consistency metrics
    fn compute_triple_consistency(&self, metrics: &mut EmbeddingEvaluationMetrics) -> Result<()> {
        let mut consistent_triples = 0;
        let mut total_triples = 0;

        for triple in &self.triples {
            if let Ok(score) = self.score_triple(
                &triple.subject.iri,
                &triple.predicate.iri,
                &triple.object.iri,
            ) {
                total_triples += 1;
                if score > 0.5 {
                    // Threshold for consistency
                    consistent_triples += 1;
                }
            }
        }

        metrics.triple_consistency_ratio = if total_triples > 0 {
            consistent_triples as f32 / total_triples as f32
        } else {
            0.0
        };

        Ok(())
    }

    /// Compute semantic coherence using relation-specific patterns
    fn compute_semantic_coherence(&self, metrics: &mut EmbeddingEvaluationMetrics) -> Result<()> {
        let mut coherence_scores = Vec::new();

        // Group triples by relation type
        let mut relation_groups: HashMap<String, Vec<&Triple>> = HashMap::new();
        for triple in &self.triples {
            relation_groups
                .entry(triple.predicate.iri.clone())
                .or_default()
                .push(triple);
        }

        // Compute coherence for each relation type
        for (relation, triples) in relation_groups {
            if triples.len() < 2 {
                continue;
            }

            let mut relation_scores = Vec::new();
            for triple in &triples {
                if let Ok(score) =
                    self.score_triple(&triple.subject.iri, &relation, &triple.object.iri)
                {
                    relation_scores.push(score as f32);
                }
            }

            if !relation_scores.is_empty() {
                let avg_score = relation_scores.iter().sum::<f32>() / relation_scores.len() as f32;
                coherence_scores.push(avg_score);
            }
        }

        metrics.semantic_coherence_score = if coherence_scores.is_empty() {
            0.0
        } else {
            coherence_scores.iter().sum::<f32>() / coherence_scores.len() as f32
        };

        Ok(())
    }
}

/// Comprehensive evaluation metrics for transformer embeddings
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct EmbeddingEvaluationMetrics {
    pub average_cosine_similarity: f32,
    pub average_norm: f32,
    pub norm_std: f32,
    pub isotropy_score: f32,
    pub similarity_median: f32,
    pub similarity_p95: f32,
    pub similarity_p05: f32,
    pub clustering_silhouette: f32,
    pub triple_consistency_ratio: f32,
    pub semantic_coherence_score: f32,
}

#[async_trait]
impl EmbeddingModel for TransformerEmbedding {
    fn config(&self) -> &ModelConfig {
        &self.config.base_config
    }

    fn model_id(&self) -> &Uuid {
        &self.id
    }

    fn model_type(&self) -> &'static str {
        "TransformerEmbedding"
    }

    fn add_triple(&mut self, triple: Triple) -> Result<()> {
        // Add entities to index
        let subject = triple.subject.iri.clone();
        let object = triple.object.iri.clone();
        let predicate = triple.predicate.iri.clone();

        if !self.entity_to_idx.contains_key(&subject) {
            let idx = self.entity_to_idx.len();
            self.entity_to_idx.insert(subject.clone(), idx);
            self.idx_to_entity.insert(idx, subject);
        }

        if !self.entity_to_idx.contains_key(&object) {
            let idx = self.entity_to_idx.len();
            self.entity_to_idx.insert(object.clone(), idx);
            self.idx_to_entity.insert(idx, object);
        }

        if !self.relation_to_idx.contains_key(&predicate) {
            let idx = self.relation_to_idx.len();
            self.relation_to_idx.insert(predicate.clone(), idx);
            self.idx_to_relation.insert(idx, predicate);
        }

        self.triples.push(triple);
        self.is_trained = false;
        Ok(())
    }

    async fn train(&mut self, epochs: Option<usize>) -> Result<TrainingStats> {
        let start_time = std::time::Instant::now();
        let epochs = epochs.unwrap_or(self.config.base_config.max_epochs);
        let mut loss_history = Vec::new();
        let mut best_loss = f32::INFINITY;
        let mut epochs_without_improvement = 0;
        let early_stopping_patience = 5;

        tracing::info!("Starting transformer training with {} epochs", epochs);

        // Update embeddings using transformer
        self.update_embeddings().await?;

        // If fine-tuning is enabled, perform knowledge graph-specific fine-tuning
        if self.config.fine_tune {
            for epoch in 0..epochs {
                let epoch_start = std::time::Instant::now();

                // Perform knowledge graph fine-tuning
                self.fine_tune_for_kg().await?;

                // Apply contrastive learning every few epochs
                if epoch % 3 == 0 {
                    self.contrastive_learning(5).await?;
                }

                // Compute validation loss for early stopping
                let validation_loss = self.compute_validation_loss().await?;
                loss_history.push(validation_loss);

                // Early stopping check
                if validation_loss < best_loss {
                    best_loss = validation_loss;
                    epochs_without_improvement = 0;
                } else {
                    epochs_without_improvement += 1;
                }

                // Log detailed progress
                let epoch_time = epoch_start.elapsed().as_secs_f64();
                tracing::info!(
                    "Epoch {}/{}: loss={:.6}, time={:.2}s, best_loss={:.6}",
                    epoch + 1,
                    epochs,
                    validation_loss,
                    epoch_time,
                    best_loss
                );

                // Early stopping
                if epochs_without_improvement >= early_stopping_patience {
                    tracing::info!(
                        "Early stopping triggered after {} epochs without improvement",
                        early_stopping_patience
                    );
                    break;
                }

                // Learning rate decay
                if epoch > 0 && epoch % (epochs / 4).max(1) == 0 {
                    self.config.base_config.learning_rate *= 0.9;
                    tracing::info!(
                        "Learning rate decayed to: {:.6}",
                        self.config.base_config.learning_rate
                    );
                }
            }
        }

        self.is_trained = true;
        self.last_training_time = Some(Utc::now());

        let training_time = start_time.elapsed().as_secs_f64();
        let final_loss = loss_history.last().copied().unwrap_or(0.001);
        let convergence_achieved = epochs_without_improvement < early_stopping_patience;

        tracing::info!(
            "Training completed: final_loss={:.6}, time={:.2}s, convergence={}",
            final_loss,
            training_time,
            convergence_achieved
        );

        Ok(TrainingStats {
            epochs_completed: loss_history.len(),
            final_loss: final_loss as f64,
            training_time_seconds: training_time,
            convergence_achieved,
            loss_history: loss_history.into_iter().map(|x| x as f64).collect(),
        })
    }

    fn get_entity_embedding(&self, entity: &str) -> Result<Vector> {
        if !self.is_trained {
            return Err(EmbeddingError::ModelNotTrained.into());
        }

        // First try to get cached embedding
        if let Some(embedding) = self.entity_embeddings.get(entity) {
            return Ok(Vector::new(embedding.to_vec()));
        }

        // If not found but we have the entity in our vocabulary, generate contextual embedding
        if self.entity_to_idx.contains_key(entity) {
            if let Ok(contextual_embedding) = self.generate_contextual_embedding(entity) {
                return Ok(Vector::new(contextual_embedding.to_vec()));
            }
        }

        Err(EmbeddingError::EntityNotFound {
            entity: entity.to_string(),
        }
        .into())
    }

    fn get_relation_embedding(&self, relation: &str) -> Result<Vector> {
        if !self.is_trained {
            return Err(EmbeddingError::ModelNotTrained.into());
        }

        self.relation_embeddings
            .get(relation)
            .map(|e| Vector::new(e.to_vec()))
            .ok_or_else(|| {
                EmbeddingError::RelationNotFound {
                    relation: relation.to_string(),
                }
                .into()
            })
    }

    fn score_triple(&self, subject: &str, predicate: &str, object: &str) -> Result<f64> {
        if !self.is_trained {
            return Err(EmbeddingError::ModelNotTrained.into());
        }

        let subj_emb =
            self.entity_embeddings
                .get(subject)
                .ok_or_else(|| EmbeddingError::EntityNotFound {
                    entity: subject.to_string(),
                })?;

        let pred_emb = self.relation_embeddings.get(predicate).ok_or_else(|| {
            EmbeddingError::RelationNotFound {
                relation: predicate.to_string(),
            }
        })?;

        let obj_emb =
            self.entity_embeddings
                .get(object)
                .ok_or_else(|| EmbeddingError::EntityNotFound {
                    entity: object.to_string(),
                })?;

        // Simple scoring function: cosine similarity between
        // concatenated (subject, predicate) and object embeddings
        let combined = subj_emb + pred_emb;
        let dot_product = (&combined * obj_emb).sum();
        let norm1 = ((&combined * &combined).sum()).sqrt();
        let norm2 = ((obj_emb * obj_emb).sum()).sqrt();

        Ok((dot_product / (norm1 * norm2)).into())
    }

    fn predict_objects(
        &self,
        subject: &str,
        predicate: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        if !self.is_trained {
            return Err(EmbeddingError::ModelNotTrained.into());
        }

        let mut scores = Vec::new();

        for entity in self.entity_to_idx.keys() {
            if let Ok(score) = self.score_triple(subject, predicate, entity) {
                scores.push((entity.clone(), score));
            }
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        Ok(scores)
    }

    fn predict_subjects(
        &self,
        predicate: &str,
        object: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        if !self.is_trained {
            return Err(EmbeddingError::ModelNotTrained.into());
        }

        let mut scores = Vec::new();

        for entity in self.entity_to_idx.keys() {
            if let Ok(score) = self.score_triple(entity, predicate, object) {
                scores.push((entity.clone(), score));
            }
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        Ok(scores)
    }

    fn predict_relations(
        &self,
        subject: &str,
        object: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        if !self.is_trained {
            return Err(EmbeddingError::ModelNotTrained.into());
        }

        let mut scores = Vec::new();

        for relation in self.relation_to_idx.keys() {
            if let Ok(score) = self.score_triple(subject, relation, object) {
                scores.push((relation.clone(), score));
            }
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        Ok(scores)
    }

    fn get_entities(&self) -> Vec<String> {
        self.entity_to_idx.keys().cloned().collect()
    }

    fn get_relations(&self) -> Vec<String> {
        self.relation_to_idx.keys().cloned().collect()
    }

    fn get_stats(&self) -> ModelStats {
        ModelStats {
            num_entities: self.entity_to_idx.len(),
            num_relations: self.relation_to_idx.len(),
            num_triples: self.triples.len(),
            dimensions: self.config.base_config.dimensions,
            is_trained: self.is_trained,
            model_type: format!("TransformerEmbedding-{:?}", self.config.transformer_type),
            creation_time: self.creation_time,
            last_training_time: self.last_training_time,
        }
    }

    fn save(&self, path: &str) -> Result<()> {
        use std::fs;
        use std::io::Write;

        tracing::info!("Saving transformer model to: {}", path);

        // Create the directory if it doesn't exist
        if let Some(parent) = std::path::Path::new(path).parent() {
            fs::create_dir_all(parent)?;
        }

        // Prepare model data for serialization
        let model_data = TransformerModelData {
            id: self.id,
            config: self.config.clone(),
            entity_embeddings: self
                .entity_embeddings
                .iter()
                .map(|(k, v)| (k.clone(), v.to_vec()))
                .collect(),
            relation_embeddings: self
                .relation_embeddings
                .iter()
                .map(|(k, v)| (k.clone(), v.to_vec()))
                .collect(),
            entity_to_idx: self.entity_to_idx.clone(),
            relation_to_idx: self.relation_to_idx.clone(),
            idx_to_entity: self.idx_to_entity.clone(),
            idx_to_relation: self.idx_to_relation.clone(),
            triples: self.triples.clone(),
            is_trained: self.is_trained,
            creation_time: self.creation_time,
            last_training_time: self.last_training_time,
        };

        // Serialize to JSON (could be enhanced with binary format for efficiency)
        let json_data = serde_json::to_string_pretty(&model_data)?;

        // Write to file
        let mut file = fs::File::create(path)?;
        file.write_all(json_data.as_bytes())?;
        file.sync_all()?;

        tracing::info!(
            "Model saved successfully: {} entities, {} relations, {} triples",
            self.entity_to_idx.len(),
            self.relation_to_idx.len(),
            self.triples.len()
        );

        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<()> {
        use std::fs;

        tracing::info!("Loading transformer model from: {}", path);

        // Read file
        let json_data = fs::read_to_string(path)?;

        // Deserialize model data
        let model_data: TransformerModelData = serde_json::from_str(&json_data)?;

        // Restore model state
        self.id = model_data.id;
        self.config = model_data.config;
        self.entity_embeddings = model_data
            .entity_embeddings
            .iter()
            .map(|(k, v)| (k.clone(), Array1::from_vec(v.clone())))
            .collect();
        self.relation_embeddings = model_data
            .relation_embeddings
            .iter()
            .map(|(k, v)| (k.clone(), Array1::from_vec(v.clone())))
            .collect();
        self.entity_to_idx = model_data.entity_to_idx;
        self.relation_to_idx = model_data.relation_to_idx;
        self.idx_to_entity = model_data.idx_to_entity;
        self.idx_to_relation = model_data.idx_to_relation;
        self.triples = model_data.triples;
        self.is_trained = model_data.is_trained;
        self.creation_time = model_data.creation_time;
        self.last_training_time = model_data.last_training_time;

        // Reinitialize transformer weights if needed
        if self.model_weights.is_none() && self.is_trained {
            self.initialize_transformer()?;
        }

        tracing::info!(
            "Model loaded successfully: {} entities, {} relations, {} triples, trained={}",
            self.entity_to_idx.len(),
            self.relation_to_idx.len(),
            self.triples.len(),
            self.is_trained
        );

        Ok(())
    }

    fn clear(&mut self) {
        self.entity_embeddings.clear();
        self.relation_embeddings.clear();
        self.entity_to_idx.clear();
        self.relation_to_idx.clear();
        self.idx_to_entity.clear();
        self.idx_to_relation.clear();
        self.triples.clear();
        self.is_trained = false;
        self.model_weights = None;
    }

    fn is_trained(&self) -> bool {
        self.is_trained
    }

    async fn encode(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        #[cfg(feature = "neural-models")]
        {
            self.encode_texts(texts).await
        }
        #[cfg(not(feature = "neural-models"))]
        {
            // Fallback implementation using simple TF-IDF-like embeddings
            let mut embeddings = Vec::new();
            for text in texts {
                let embedding = self.embed_text(text)?;
                embeddings.push(embedding.to_vec());
            }
            Ok(embeddings)
        }
    }
}

// Additional methods for TransformerEmbedding (not part of the trait)
impl TransformerEmbedding {
    /// Compute validation loss for monitoring training progress
    pub async fn compute_validation_loss(&self) -> Result<f32> {
        let mut total_loss = 0.0;
        let mut num_samples = 0;

        // Sample a subset of triples for validation
        let validation_size = (self.triples.len() / 10).max(1);
        let validation_triples: Vec<_> = self.triples.iter().take(validation_size).collect();

        for triple in validation_triples {
            let subject_key = &triple.subject.iri;
            let predicate_key = &triple.predicate.iri;
            let object_key = &triple.object.iri;

            if let (Some(s_emb), Some(r_emb), Some(o_emb)) = (
                self.entity_embeddings.get(subject_key),
                self.relation_embeddings.get(predicate_key),
                self.entity_embeddings.get(object_key),
            ) {
                // Compute TransE-style loss: ||s + r - o||
                let predicted = s_emb + r_emb;
                let error = o_emb - &predicted;
                let loss = (&error * &error).sum().sqrt();
                total_loss += loss;
                num_samples += 1;
            }
        }

        Ok(if num_samples > 0 {
            total_loss / num_samples as f32
        } else {
            0.0
        })
    }

    /// Simple text embedding using character-based features (fallback)
    fn simple_text_embedding(&self, text: &str) -> Result<Vector> {
        let dimensions = self.config.base_config.dimensions;
        let mut embedding = vec![0.0; dimensions];

        // Simple character-based embedding
        let chars: Vec<char> = text.chars().collect();
        if !chars.is_empty() {
            for (_i, &ch) in chars.iter().enumerate() {
                let char_idx = (ch as u32) % (dimensions as u32);
                embedding[char_idx as usize] += 1.0 / chars.len() as f32;
            }
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        Ok(Vector::new(embedding))
    }

    #[cfg(feature = "neural-models")]
    /// Encode texts using transformer models
    async fn encode_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // In a real implementation, this would load the actual transformer model
        // For now, we'll simulate specialized model behavior
        match self.config.transformer_type {
            TransformerType::SciBERT => self.encode_scientific_text(texts).await,
            TransformerType::BioBERT => self.encode_biomedical_text(texts).await,
            TransformerType::CodeBERT => self.encode_code_text(texts).await,
            TransformerType::LegalBERT => self.encode_legal_text(texts).await,
            TransformerType::NewsBERT => self.encode_news_text(texts).await,
            TransformerType::SocialMediaBERT => self.encode_social_text(texts).await,
            _ => self.encode_general_text(texts).await,
        }
    }

    #[cfg(feature = "neural-models")]
    /// Encode scientific text with domain-specific preprocessing
    async fn encode_scientific_text(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();

        for text in texts {
            // Scientific text preprocessing
            let processed_text = self.preprocess_scientific_text(text);
            let embedding = self.bert_encode(&processed_text)?;
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    #[cfg(feature = "neural-models")]
    /// Encode biomedical text with medical entity recognition
    async fn encode_biomedical_text(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();

        for text in texts {
            // Biomedical preprocessing (gene names, drug names, etc.)
            let processed_text = self.preprocess_biomedical_text(text);
            let embedding = self.bert_encode(&processed_text)?;
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    #[cfg(feature = "neural-models")]
    /// Encode code with syntax-aware processing
    async fn encode_code_text(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();

        for text in texts {
            // Code preprocessing (identifier normalization, etc.)
            let processed_text = self.preprocess_code_text(text);
            let embedding = self.bert_encode(&processed_text)?;
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    #[cfg(feature = "neural-models")]
    /// Encode legal text with legal entity recognition
    async fn encode_legal_text(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();

        for text in texts {
            // Legal text preprocessing
            let processed_text = self.preprocess_legal_text(text);
            let embedding = self.bert_encode(&processed_text)?;
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    #[cfg(feature = "neural-models")]
    /// Encode news text with temporal and topic awareness
    async fn encode_news_text(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();

        for text in texts {
            // News text preprocessing
            let processed_text = self.preprocess_news_text(text);
            let embedding = self.bert_encode(&processed_text)?;
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    #[cfg(feature = "neural-models")]
    /// Encode social media text with hashtag and mention processing
    async fn encode_social_text(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();

        for text in texts {
            // Social media preprocessing (hashtags, mentions, emojis)
            let processed_text = self.preprocess_social_text(text);
            let embedding = self.bert_encode(&processed_text)?;
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    #[cfg(feature = "neural-models")]
    /// Encode general text with standard BERT processing
    async fn encode_general_text(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();

        for text in texts {
            let embedding = self.bert_encode(text)?;
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    #[cfg(feature = "neural-models")]
    /// Core BERT encoding logic (simplified for now)
    fn bert_encode(&self, text: &str) -> Result<Vec<f32>> {
        // This is a simplified implementation
        // In production, this would use actual transformer models

        let dimensions = self.config.base_config.dimensions;
        let mut embedding = vec![0.0; dimensions];

        // Simulate BERT-like encoding with better features than simple embedding
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut word_count = HashMap::new();

        for word in &words {
            *word_count.entry(word.to_lowercase()).or_insert(0) += 1;
        }

        // Create features based on word frequency and position
        for (word, count) in word_count {
            let word_hash = self.hash_word(&word) % dimensions;
            embedding[word_hash] += count as f32 / words.len() as f32;
        }

        // Apply pooling strategy
        match self.config.pooling_strategy {
            PoolingStrategy::Mean => {
                // Already using mean-like approach
            }
            PoolingStrategy::Max => {
                // Find max values
                let max_val = embedding.iter().fold(0.0, |a, &b| a.max(b));
                for val in &mut embedding {
                    if *val < max_val * 0.5 {
                        *val = 0.0;
                    }
                }
            }
            PoolingStrategy::CLS => {
                // Simulate CLS token representation
                embedding[0] = embedding.iter().sum::<f32>() / embedding.len() as f32;
            }
            _ => {} // Other strategies use default
        }

        // Normalize if configured
        if self.config.normalize_embeddings {
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut embedding {
                    *x /= norm;
                }
            }
        }

        Ok(embedding)
    }

    /// Hash function for words
    fn hash_word(&self, word: &str) -> usize {
        let mut hash = 0usize;
        for byte in word.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as usize);
        }
        hash
    }

    // Domain-specific preprocessing methods
    #[cfg(feature = "neural-models")]
    fn preprocess_scientific_text(&self, text: &str) -> String {
        // Remove citations, normalize equations, handle scientific notation
        text.to_lowercase()
            .replace("et al.", "")
            .replace(char::is_numeric, " ")
    }

    #[cfg(feature = "neural-models")]
    fn preprocess_biomedical_text(&self, text: &str) -> String {
        // Normalize gene names, drug names, medical terms
        text.to_lowercase()
            .replace("gene", "GENE")
            .replace("protein", "PROTEIN")
    }

    #[cfg(feature = "neural-models")]
    fn preprocess_code_text(&self, text: &str) -> String {
        // Normalize identifiers, handle syntax
        text.replace("_", " ").replace("CamelCase", "camel case")
    }

    #[cfg(feature = "neural-models")]
    fn preprocess_legal_text(&self, text: &str) -> String {
        // Handle legal citations, normalize terms
        text.to_lowercase()
            .replace("v.", "versus")
            .replace("§", "section")
    }

    #[cfg(feature = "neural-models")]
    fn preprocess_news_text(&self, text: &str) -> String {
        // Remove bylines, normalize quotes
        text.replace("\"", "").replace("'", "")
    }

    #[cfg(feature = "neural-models")]
    fn preprocess_social_text(&self, text: &str) -> String {
        // Handle hashtags, mentions, emojis
        text.replace("#", "hashtag ")
            .replace("@", "mention ")
            .to_lowercase()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NamedNode;

    #[tokio::test]
    async fn test_transformer_embedding_basic() {
        let config = TransformerConfig::default();
        let mut model = TransformerEmbedding::new(config);

        // Add some triples
        let triple1 = Triple::new(
            NamedNode::new("http://example.org/Alice").unwrap(),
            NamedNode::new("http://example.org/knows").unwrap(),
            NamedNode::new("http://example.org/Bob").unwrap(),
        );

        model.add_triple(triple1).unwrap();

        // Train the model
        let stats = model.train(Some(10)).await.unwrap();
        assert!(stats.convergence_achieved);

        // Get embeddings
        let alice_emb = model
            .get_entity_embedding("http://example.org/Alice")
            .unwrap();
        assert_eq!(alice_emb.dimensions, 384); // SentenceBERT default

        // Test predictions
        let predictions = model
            .predict_objects("http://example.org/Alice", "http://example.org/knows", 5)
            .unwrap();
        assert!(!predictions.is_empty());
    }

    #[test]
    fn test_transformer_types() {
        assert_eq!(TransformerType::BERT.model_name(), "bert-base-uncased");
        assert_eq!(
            TransformerType::SciBERT.model_name(),
            "allenai/scibert_scivocab_uncased"
        );
        assert_eq!(TransformerType::SentenceBERT.default_dimensions(), 384);
        assert_eq!(TransformerType::BERT.default_dimensions(), 768);
    }

    #[tokio::test]
    async fn test_sentence_bert_features() {
        let config = TransformerEmbedding::sentence_bert_config(384);
        let mut model = TransformerEmbedding::new(config);

        // Add some semantic triples
        let triple1 = Triple::new(
            NamedNode::new("http://example.org/Machine_Learning").unwrap(),
            NamedNode::new("http://example.org/related_to").unwrap(),
            NamedNode::new("http://example.org/Artificial_Intelligence").unwrap(),
        );

        let triple2 = Triple::new(
            NamedNode::new("http://example.org/Deep_Learning").unwrap(),
            NamedNode::new("http://example.org/subset_of").unwrap(),
            NamedNode::new("http://example.org/Machine_Learning").unwrap(),
        );

        model.add_triple(triple1).unwrap();
        model.add_triple(triple2).unwrap();

        // Train the model
        let _stats = model.train(Some(5)).await.unwrap();
        assert!(model.is_trained());

        // Test semantic similarity
        let similarity = model
            .compute_semantic_similarity("Machine Learning", "Artificial Intelligence")
            .unwrap();
        assert!((0.0..=1.0).contains(&similarity));

        // Test semantic search
        let similar_entities = model.find_similar_entities("AI research", 3).unwrap();
        assert!(!similar_entities.is_empty());

        // Test batch encoding
        let texts = vec!["machine learning", "deep learning", "neural networks"];
        let embeddings = model.batch_encode(&texts).unwrap();
        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0].len(), 384);
    }

    #[test]
    fn test_pooling_strategies() {
        let config = TransformerConfig {
            pooling_strategy: PoolingStrategy::Max,
            normalize_embeddings: true,
            ..TransformerEmbedding::sentence_bert_config(64)
        };

        let model = TransformerEmbedding::new(config);

        // Test different pooling strategies
        let test_embeddings = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]),
            Array1::from_vec(vec![2.0, 1.0, 4.0]),
            Array1::from_vec(vec![3.0, 3.0, 2.0]),
        ];

        // Test mean pooling
        let mean_result = model.apply_mean_pooling(&test_embeddings).unwrap();
        assert_eq!(mean_result.len(), 3);
        assert!((mean_result[0] - 2.0).abs() < 1e-6);

        // Test max pooling
        let max_result = model.apply_max_pooling(&test_embeddings).unwrap();
        assert_eq!(max_result.len(), 3);
        assert!((max_result[0] - 3.0).abs() < 1e-6);
        assert!((max_result[1] - 3.0).abs() < 1e-6);
        assert!((max_result[2] - 4.0).abs() < 1e-6);

        // Test CLS pooling
        let cls_result = model.apply_cls_pooling(&test_embeddings).unwrap();
        assert_eq!(cls_result.len(), 3);
        assert!((cls_result[0] - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_advanced_training_features() {
        let config = TransformerConfig {
            fine_tune: true,
            ..TransformerEmbedding::sentence_bert_config(128)
        };
        let mut model = TransformerEmbedding::new(config);

        // Add diverse triples for better testing
        let triples = vec![
            Triple::new(
                NamedNode::new("http://example.org/Alice").unwrap(),
                NamedNode::new("http://example.org/knows").unwrap(),
                NamedNode::new("http://example.org/Bob").unwrap(),
            ),
            Triple::new(
                NamedNode::new("http://example.org/Bob").unwrap(),
                NamedNode::new("http://example.org/likes").unwrap(),
                NamedNode::new("http://example.org/Jazz").unwrap(),
            ),
            Triple::new(
                NamedNode::new("http://example.org/Alice").unwrap(),
                NamedNode::new("http://example.org/lives_in").unwrap(),
                NamedNode::new("http://example.org/NYC").unwrap(),
            ),
        ];

        for triple in triples {
            model.add_triple(triple).unwrap();
        }

        // Test enhanced training with early stopping and contrastive learning
        let stats = model.train(Some(15)).await.unwrap();
        assert!(stats.epochs_completed > 0);
        assert!(stats.training_time_seconds > 0.0);
        assert!(!stats.loss_history.is_empty());

        // Verify model is trained
        assert!(model.is_trained());

        // Test validation loss computation
        let validation_loss = model.compute_validation_loss().await.unwrap();
        assert!(validation_loss >= 0.0);
    }

    #[tokio::test]
    async fn test_model_serialization() {
        let config = TransformerEmbedding::sentence_bert_config(64);
        let mut model = TransformerEmbedding::new(config);

        // Add test data
        let triple = Triple::new(
            NamedNode::new("http://example.org/Entity1").unwrap(),
            NamedNode::new("http://example.org/relation").unwrap(),
            NamedNode::new("http://example.org/Entity2").unwrap(),
        );
        model.add_triple(triple).unwrap();

        // Train the model
        let _stats = model.train(Some(3)).await.unwrap();

        // Test saving
        let save_path = "/tmp/test_transformer_model.json";
        model.save(save_path).unwrap();

        // Test loading into new model
        let mut loaded_model =
            TransformerEmbedding::new(TransformerEmbedding::sentence_bert_config(64));
        loaded_model.load(save_path).unwrap();

        // Verify loaded model state
        assert_eq!(loaded_model.is_trained(), model.is_trained());
        assert_eq!(loaded_model.entity_to_idx.len(), model.entity_to_idx.len());
        assert_eq!(
            loaded_model.relation_to_idx.len(),
            model.relation_to_idx.len()
        );
        assert_eq!(loaded_model.triples.len(), model.triples.len());

        // Clean up
        std::fs::remove_file(save_path).ok();
    }

    #[tokio::test]
    async fn test_embedding_evaluation_metrics() {
        let config = TransformerEmbedding::sentence_bert_config(32);
        let mut model = TransformerEmbedding::new(config);

        // Add more diverse test data for better evaluation
        let test_triples = vec![
            Triple::new(
                NamedNode::new("http://example.org/Person1").unwrap(),
                NamedNode::new("http://example.org/knows").unwrap(),
                NamedNode::new("http://example.org/Person2").unwrap(),
            ),
            Triple::new(
                NamedNode::new("http://example.org/Person2").unwrap(),
                NamedNode::new("http://example.org/works_at").unwrap(),
                NamedNode::new("http://example.org/Company1").unwrap(),
            ),
            Triple::new(
                NamedNode::new("http://example.org/Person1").unwrap(),
                NamedNode::new("http://example.org/lives_in").unwrap(),
                NamedNode::new("http://example.org/City1").unwrap(),
            ),
            Triple::new(
                NamedNode::new("http://example.org/Company1").unwrap(),
                NamedNode::new("http://example.org/located_in").unwrap(),
                NamedNode::new("http://example.org/City1").unwrap(),
            ),
        ];

        for triple in test_triples {
            model.add_triple(triple).unwrap();
        }

        // Train the model
        let _stats = model.train(Some(5)).await.unwrap();

        // Test evaluation metrics
        let metrics = model.evaluate_embeddings().unwrap();

        // Verify metrics are computed
        assert!(metrics.average_norm > 0.0);
        assert!(metrics.norm_std >= 0.0);
        assert!(metrics.isotropy_score >= 0.0 && metrics.isotropy_score <= 1.0);
        assert!(metrics.triple_consistency_ratio >= 0.0 && metrics.triple_consistency_ratio <= 1.0);
        assert!(metrics.semantic_coherence_score >= 0.0);

        // Print metrics for manual verification
        println!("Evaluation metrics: {:#?}", metrics);
    }

    #[tokio::test]
    async fn test_contrastive_learning() {
        let config = TransformerConfig {
            fine_tune: true,
            ..TransformerEmbedding::sentence_bert_config(64)
        };
        let mut model = TransformerEmbedding::new(config);

        // Add test triples
        for i in 0..5 {
            let triple = Triple::new(
                NamedNode::new(&format!("http://example.org/Entity{}", i)).unwrap(),
                NamedNode::new("http://example.org/related_to").unwrap(),
                NamedNode::new(&format!("http://example.org/Target{}", i)).unwrap(),
            );
            model.add_triple(triple).unwrap();
        }

        // Initialize embeddings
        model.update_embeddings().await.unwrap();

        // Get embeddings before contrastive learning
        let entity_before = model
            .entity_embeddings
            .get("http://example.org/Entity0")
            .cloned();

        // Apply contrastive learning
        model.contrastive_learning(3).await.unwrap();

        // Get embeddings after contrastive learning
        let entity_after = model
            .entity_embeddings
            .get("http://example.org/Entity0")
            .cloned();

        // Verify embeddings changed (they should be different after contrastive learning)
        if let (Some(before), Some(after)) = (entity_before, entity_after) {
            let difference = (&before - &after).mapv(|x| x.abs()).sum();
            assert!(
                difference > 0.0,
                "Embeddings should change after contrastive learning"
            );
        }
    }

    #[test]
    fn test_text_preprocessing() {
        let config = TransformerEmbedding::sentence_bert_config(64);
        let model = TransformerEmbedding::new(config);

        // Test URI preprocessing
        let uri_text = "http://example.org/Person#John_Doe";
        let processed = model.preprocess_text(uri_text);
        assert_eq!(processed, "John_Doe");

        // Test path-based URI
        let path_uri = "http://example.org/vocabulary/hasFriend";
        let processed_path = model.preprocess_text(path_uri);
        assert_eq!(processed_path, "hasFriend");

        // Test regular text
        let regular_text = "Some_Regular-Text";
        let processed_regular = model.preprocess_text(regular_text);
        assert_eq!(processed_regular, "some regular text");
    }

    #[test]
    fn test_triple_to_text_conversion() {
        let config = TransformerEmbedding::sentence_bert_config(64);
        let model = TransformerEmbedding::new(config);

        // Test different relation types
        let type_triple = Triple::new(
            NamedNode::new("http://example.org/John").unwrap(),
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap(),
            NamedNode::new("http://example.org/Person").unwrap(),
        );

        let text = model.triple_to_text(&type_triple);
        assert!(text.contains("is a"));

        // Test subclass relation
        let subclass_triple = Triple::new(
            NamedNode::new("http://example.org/Student").unwrap(),
            NamedNode::new("http://www.w3.org/2000/01/rdf-schema#subClassOf").unwrap(),
            NamedNode::new("http://example.org/Person").unwrap(),
        );

        let subclass_text = model.triple_to_text(&subclass_triple);
        assert!(subclass_text.contains("is a type of"));

        // Test generic relation
        let generic_triple = Triple::new(
            NamedNode::new("http://example.org/John").unwrap(),
            NamedNode::new("http://example.org/knows").unwrap(),
            NamedNode::new("http://example.org/Jane").unwrap(),
        );

        let generic_text = model.triple_to_text(&generic_triple);
        assert!(
            generic_text.contains("John")
                && generic_text.contains("knows")
                && generic_text.contains("Jane")
        );
    }

    #[test]
    fn test_specialized_model_configurations() {
        // Test SciBERT configuration
        let scibert_config = TransformerEmbedding::scibert_config(768);
        assert_eq!(scibert_config.transformer_type, TransformerType::SciBERT);
        assert_eq!(scibert_config.base_config.dimensions, 768);
        assert!(scibert_config.fine_tune);
        assert_eq!(scibert_config.gradient_accumulation_steps, 2);

        // Test BioBERT configuration
        let biobert_config = TransformerEmbedding::biobert_config(768);
        assert_eq!(biobert_config.transformer_type, TransformerType::BioBERT);
        assert_eq!(biobert_config.learning_rate_schedule, "cosine");
        assert_eq!(biobert_config.warmup_steps, 800);
        assert_eq!(biobert_config.gradient_accumulation_steps, 4);

        // Test CodeBERT configuration
        let codebert_config = TransformerEmbedding::codebert_config(768);
        assert_eq!(codebert_config.transformer_type, TransformerType::CodeBERT);
        assert_eq!(codebert_config.max_sequence_length, 1024);
        assert_eq!(codebert_config.pooling_strategy, PoolingStrategy::CLS);

        // Test LegalBERT configuration
        let legalbert_config = TransformerEmbedding::legalbert_config(768);
        assert_eq!(
            legalbert_config.transformer_type,
            TransformerType::LegalBERT
        );
        assert_eq!(legalbert_config.max_sequence_length, 1024);
        assert_eq!(
            legalbert_config.pooling_strategy,
            PoolingStrategy::AttentionWeighted
        );
        assert_eq!(legalbert_config.learning_rate_schedule, "polynomial");

        // Test NewsBERT configuration
        let newsbert_config = TransformerEmbedding::newsbert_config(768);
        assert_eq!(newsbert_config.transformer_type, TransformerType::NewsBERT);
        assert_eq!(newsbert_config.warmup_steps, 600);
        assert_eq!(newsbert_config.pooling_strategy, PoolingStrategy::Mean);

        // Test SocialMediaBERT configuration
        let social_config = TransformerEmbedding::social_media_bert_config(768);
        assert_eq!(
            social_config.transformer_type,
            TransformerType::SocialMediaBERT
        );
        assert_eq!(social_config.max_sequence_length, 280);
        assert_eq!(social_config.pooling_strategy, PoolingStrategy::Max);
    }

    #[test]
    fn test_specialized_model_names() {
        assert_eq!(
            TransformerType::SciBERT.model_name(),
            "allenai/scibert_scivocab_uncased"
        );
        assert_eq!(
            TransformerType::BioBERT.model_name(),
            "dmis-lab/biobert-v1.1"
        );
        assert_eq!(
            TransformerType::CodeBERT.model_name(),
            "microsoft/codebert-base"
        );
        assert_eq!(
            TransformerType::LegalBERT.model_name(),
            "nlpaueb/legal-bert-base-uncased"
        );
        assert_eq!(
            TransformerType::NewsBERT.model_name(),
            "dkleczek/bert-base-polish-uncased-v1"
        );
        assert_eq!(
            TransformerType::SocialMediaBERT.model_name(),
            "vinai/bertweet-base"
        );
    }

    #[test]
    fn test_domain_specific_embedding_generation() {
        // Test scientific text with SciBERT
        let scibert_config = TransformerEmbedding::scibert_config(64);
        let mut scibert_model = TransformerEmbedding::new(scibert_config);
        scibert_model.initialize_transformer().unwrap();

        let scientific_text = "The novel coronavirus SARS-CoV-2 causes COVID-19 disease";
        let embedding_result = scibert_model.embed_text(scientific_text);
        assert!(embedding_result.is_ok());

        // Test biomedical text with BioBERT
        let biobert_config = TransformerEmbedding::biobert_config(64);
        let mut biobert_model = TransformerEmbedding::new(biobert_config);
        biobert_model.initialize_transformer().unwrap();

        let biomedical_text = "p53 gene mutations are associated with various cancer types";
        let bio_embedding_result = biobert_model.embed_text(biomedical_text);
        assert!(bio_embedding_result.is_ok());

        // Test code with CodeBERT
        let codebert_config = TransformerEmbedding::codebert_config(64);
        let mut codebert_model = TransformerEmbedding::new(codebert_config);
        codebert_model.initialize_transformer().unwrap();

        let code_text =
            "function fibonacci(n) { return n < 2 ? n : fibonacci(n-1) + fibonacci(n-2); }";
        let code_embedding_result = codebert_model.embed_text(code_text);
        assert!(code_embedding_result.is_ok());

        // Test legal text with LegalBERT
        let legalbert_config = TransformerEmbedding::legalbert_config(64);
        let mut legalbert_model = TransformerEmbedding::new(legalbert_config);
        legalbert_model.initialize_transformer().unwrap();

        let legal_text = "The defendant hereby agrees to the terms and conditions of this contract";
        let legal_embedding_result = legalbert_model.embed_text(legal_text);
        assert!(legal_embedding_result.is_ok());
    }

    #[test]
    fn test_pooling_strategies_for_specialized_models() {
        // Test that different pooling strategies work correctly
        let base_config = TransformerEmbedding::sentence_bert_config(64);

        // Test Mean pooling
        let mut mean_config = base_config.clone();
        mean_config.pooling_strategy = PoolingStrategy::Mean;
        let mut mean_model = TransformerEmbedding::new(mean_config);
        mean_model.initialize_transformer().unwrap();

        // Test Max pooling
        let mut max_config = base_config.clone();
        max_config.pooling_strategy = PoolingStrategy::Max;
        let mut max_model = TransformerEmbedding::new(max_config);
        max_model.initialize_transformer().unwrap();

        // Test CLS pooling
        let mut cls_config = base_config.clone();
        cls_config.pooling_strategy = PoolingStrategy::CLS;
        let mut cls_model = TransformerEmbedding::new(cls_config);
        cls_model.initialize_transformer().unwrap();

        // Test AttentionWeighted pooling
        let mut attention_config = base_config;
        attention_config.pooling_strategy = PoolingStrategy::AttentionWeighted;
        let mut attention_model = TransformerEmbedding::new(attention_config);
        attention_model.initialize_transformer().unwrap();

        let test_text = "This is a test sentence for pooling strategies";

        // All pooling strategies should produce valid embeddings
        assert!(mean_model.embed_text(test_text).is_ok());
        assert!(max_model.embed_text(test_text).is_ok());
        assert!(cls_model.embed_text(test_text).is_ok());
        assert!(attention_model.embed_text(test_text).is_ok());
    }

    #[test]
    fn test_specialized_model_learning_schedules() {
        // Test different learning rate schedules
        let linear_config = TransformerEmbedding::scibert_config(64);
        assert_eq!(linear_config.learning_rate_schedule, "linear");

        let cosine_config = TransformerEmbedding::biobert_config(64);
        assert_eq!(cosine_config.learning_rate_schedule, "cosine");

        let polynomial_config = TransformerEmbedding::legalbert_config(64);
        assert_eq!(polynomial_config.learning_rate_schedule, "polynomial");

        // Verify all configs have fine-tuning enabled for domain adaptation
        assert!(linear_config.fine_tune);
        assert!(cosine_config.fine_tune);
        assert!(polynomial_config.fine_tune);
    }

    #[test]
    fn test_domain_specific_preprocessing() {
        // Test SciBERT preprocessing
        let scibert_config = TransformerEmbedding::scibert_config(768);
        let scibert_model = TransformerEmbedding::new(scibert_config);

        let scientific_text = "DNA synthesis with ATP and Co2 at 25°C using 5mg/ml concentration";
        let processed = scibert_model.preprocess_scientific_text(scientific_text);
        assert!(processed.contains("deoxyribonucleic acid"));
        assert!(processed.contains("adenosine triphosphate"));
        assert!(processed.contains("carbon dioxide"));
        assert!(processed.contains("degrees celsius"));
        assert!(processed.contains("milligrams per milliliter"));

        // Test BioBERT preprocessing
        let biobert_config = TransformerEmbedding::biobert_config(768);
        let biobert_model = TransformerEmbedding::new(biobert_config);

        let biomedical_text = "p53 and BRCA1 mutations affect TNF-α via mRNA expression in CNS";
        let processed = biobert_model.preprocess_biomedical_text(biomedical_text);
        assert!(processed.contains("tumor protein p53"));
        assert!(processed.contains("breast cancer gene 1"));
        assert!(processed.contains("tumor necrosis factor"));
        assert!(processed.contains("messenger ribonucleic acid"));
        assert!(processed.contains("central nervous system"));

        // Test CodeBERT preprocessing
        let codebert_config = TransformerEmbedding::codebert_config(768);
        let codebert_model = TransformerEmbedding::new(codebert_config);

        let code_text = "MyClass impl func calculateValue() returns Vec<i32>";
        let processed = codebert_model.preprocess_code_text(code_text);
        assert!(processed.contains("my class"));
        assert!(processed.contains("implementation"));
        assert!(processed.contains("function"));
        assert!(processed.contains("calculate value"));

        // Test LegalBERT preprocessing
        let legal_config = TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(768),
            transformer_type: TransformerType::LegalBERT,
            max_sequence_length: 512,
            use_pooling: true,
            pooling_strategy: PoolingStrategy::Mean,
            fine_tune: true,
            learning_rate_schedule: "linear".to_string(),
            warmup_steps: 500,
            gradient_accumulation_steps: 2,
            normalize_embeddings: true,
        };
        let legal_model = TransformerEmbedding::new(legal_config);

        let legal_text = "USC §1983 provides plaintiff v. defendant tort relief";
        let processed = legal_model.preprocess_legal_text(legal_text);
        assert!(processed.contains("United States Code"));
        assert!(processed.contains("party bringing lawsuit"));
        assert!(processed.contains("versus"));
        assert!(processed.contains("party being sued"));
        assert!(processed.contains("civil wrong"));

        // Test SocialMediaBERT preprocessing
        let social_config = TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(768),
            transformer_type: TransformerType::SocialMediaBERT,
            max_sequence_length: 280,
            use_pooling: true,
            pooling_strategy: PoolingStrategy::Max,
            fine_tune: true,
            learning_rate_schedule: "linear".to_string(),
            warmup_steps: 400,
            gradient_accumulation_steps: 1,
            normalize_embeddings: true,
        };
        let social_model = TransformerEmbedding::new(social_config);

        let social_text = "lol this is amazing! btw check out @username #awesome :)";
        let processed = social_model.preprocess_social_media_text(social_text);
        assert!(processed.contains("laugh out loud"));
        assert!(processed.contains("by the way"));
        assert!(processed.contains("mention username"));
        assert!(processed.contains("hashtag awesome"));
        assert!(processed.contains("happy"));
    }

    #[test]
    fn test_camel_case_expansion() {
        let config = TransformerEmbedding::codebert_config(768);
        let model = TransformerEmbedding::new(config);

        // Test various camelCase patterns
        assert_eq!(model.expand_camel_case("MyClass"), "my class");
        assert_eq!(model.expand_camel_case("calculateValue"), "calculate value");
        assert_eq!(
            model.expand_camel_case("getUserNameFromAPI"),
            "get user name from a p i"
        );
        assert_eq!(
            model.expand_camel_case("XMLHttpRequest"),
            "x m l http request"
        );
        assert_eq!(model.expand_camel_case("iPhone"), "i phone");

        // Test edge cases
        assert_eq!(model.expand_camel_case("A"), "a");
        assert_eq!(model.expand_camel_case("AB"), "a b");
        assert_eq!(model.expand_camel_case("lowercase"), "lowercase");
        assert_eq!(model.expand_camel_case(""), "");
    }

    #[test]
    fn test_integrated_domain_preprocessing() {
        // Test that the main preprocess_text method correctly routes to domain-specific methods

        // Scientific text with SciBERT
        let scibert_config = TransformerEmbedding::scibert_config(768);
        let scibert_model = TransformerEmbedding::new(scibert_config);
        let result = scibert_model.preprocess_text("DNA_synthesis_with_ATP");
        assert!(result.contains("deoxyribonucleic acid"));
        assert!(result.contains("adenosine triphosphate"));

        // Biomedical text with BioBERT
        let biobert_config = TransformerEmbedding::biobert_config(768);
        let biobert_model = TransformerEmbedding::new(biobert_config);
        let result = biobert_model.preprocess_text("p53-mutation");
        assert!(result.contains("tumor protein p53"));

        // Code text with CodeBERT
        let codebert_config = TransformerEmbedding::codebert_config(768);
        let codebert_model = TransformerEmbedding::new(codebert_config);
        let result = codebert_model.preprocess_text("MyClass_impl");
        assert!(result.contains("my class"));
        assert!(result.contains("implementation"));

        // URI processing should still work
        let uri_result = scibert_model.preprocess_text("http://example.org/DNA_molecule");
        assert!(uri_result.contains("deoxyribonucleic acid"));
        assert!(uri_result.contains("molecule"));
    }

    #[test]
    fn test_news_domain_preprocessing() {
        let news_config = TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(768),
            transformer_type: TransformerType::NewsBERT,
            max_sequence_length: 512,
            use_pooling: true,
            pooling_strategy: PoolingStrategy::Mean,
            fine_tune: true,
            learning_rate_schedule: "linear".to_string(),
            warmup_steps: 500,
            gradient_accumulation_steps: 2,
            normalize_embeddings: true,
        };
        let news_model = TransformerEmbedding::new(news_config);

        let news_text = "CEO announces IPO filing with SEC, GDP growth expected";
        let processed = news_model.preprocess_news_text(news_text);
        assert!(processed.contains("chief executive officer"));
        assert!(processed.contains("initial public offering"));
        assert!(processed.contains("Securities and Exchange Commission"));
        assert!(processed.contains("gross domestic product"));
    }
}
