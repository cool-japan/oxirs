//! Neural Transformer Pattern Integration for Advanced Query Optimization
//!
//! This module integrates transformer-style attention mechanisms with pattern optimization
//! to provide state-of-the-art neural enhancement for SPARQL query planning and execution.

use crate::{
    neural_patterns::{NeuralPatternConfig, NeuralPatternRecognizer},
    Result, ShaclAiError,
};

use oxirs_core::{
    model::Variable,
    query::{
        algebra::{AlgebraTriplePattern, TermPattern as AlgebraTermPattern},
        pattern_optimizer::{IndexType, OptimizedPatternPlan, PatternStrategy},
    },
};
use scirs2_core::ndarray_ext::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Neural transformer pattern integration engine for advanced query optimization
#[derive(Debug)]
pub struct NeuralTransformerPatternIntegration {
    /// Multi-head attention mechanism
    attention_mechanism: Arc<Mutex<MultiHeadAttention>>,

    /// Transformer encoder layers
    transformer_encoder: Arc<Mutex<TransformerEncoder>>,

    /// Pattern embedding engine
    pattern_embedder: Arc<Mutex<PatternEmbedder>>,

    /// Attention-based cost predictor
    attention_cost_predictor: Arc<Mutex<AttentionCostPredictor>>,

    /// Neural pattern recognizer integration
    neural_recognizer: Arc<Mutex<NeuralPatternRecognizer>>,

    /// Position encoding for sequential patterns
    positional_encoder: Arc<Mutex<PositionalEncoder>>,

    /// Memory bank for pattern history
    pattern_memory: Arc<Mutex<PatternMemoryBank>>,

    /// Configuration
    config: NeuralTransformerConfig,

    /// Performance statistics
    stats: NeuralTransformerStats,
}

/// Configuration for neural transformer integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralTransformerConfig {
    /// Model dimension for embeddings
    pub model_dim: usize,

    /// Number of attention heads
    pub num_heads: usize,

    /// Number of encoder layers
    pub num_layers: usize,

    /// Feed-forward network dimension
    pub ffn_dim: usize,

    /// Dropout rate for regularization
    pub dropout_rate: f64,

    /// Maximum sequence length for patterns
    pub max_sequence_length: usize,

    /// Enable residual connections
    pub enable_residual_connections: bool,

    /// Enable layer normalization
    pub enable_layer_norm: bool,

    /// Enable position encoding
    pub enable_position_encoding: bool,

    /// Enable memory attention
    pub enable_memory_attention: bool,

    /// Memory bank size
    pub memory_bank_size: usize,

    /// Learning rate for fine-tuning
    pub learning_rate: f64,

    /// Attention temperature
    pub attention_temperature: f64,

    /// Enable causal attention (for autoregressive patterns)
    pub enable_causal_attention: bool,

    /// Enable cross-attention between patterns
    pub enable_cross_attention: bool,

    /// Enable hierarchical attention
    pub enable_hierarchical_attention: bool,

    /// Hierarchical attention levels
    pub hierarchy_levels: usize,
}

impl Default for NeuralTransformerConfig {
    fn default() -> Self {
        Self {
            model_dim: 512,
            num_heads: 8,
            num_layers: 6,
            ffn_dim: 2048,
            dropout_rate: 0.1,
            max_sequence_length: 1024,
            enable_residual_connections: true,
            enable_layer_norm: true,
            enable_position_encoding: true,
            enable_memory_attention: true,
            memory_bank_size: 10000,
            learning_rate: 0.0001,
            attention_temperature: 1.0,
            enable_causal_attention: false,
            enable_cross_attention: true,
            enable_hierarchical_attention: true,
            hierarchy_levels: 3,
        }
    }
}

/// Multi-head attention mechanism for pattern analysis
#[derive(Debug)]
pub struct MultiHeadAttention {
    /// Query projection weights
    query_weights: Array3<f64>, // [head, input_dim, head_dim]

    /// Key projection weights
    key_weights: Array3<f64>,

    /// Value projection weights
    value_weights: Array3<f64>,

    /// Output projection weights
    output_weights: Array2<f64>, // [heads * head_dim, output_dim]

    /// Configuration
    config: NeuralTransformerConfig,

    /// Attention patterns cache
    attention_cache: HashMap<String, Array3<f64>>,
}

impl MultiHeadAttention {
    pub fn new(config: NeuralTransformerConfig) -> Self {
        let head_dim = config.model_dim / config.num_heads;

        let query_weights = Array3::zeros((config.num_heads, config.model_dim, head_dim));
        let key_weights = Array3::zeros((config.num_heads, config.model_dim, head_dim));
        let value_weights = Array3::zeros((config.num_heads, config.model_dim, head_dim));
        let output_weights = Array2::zeros((config.num_heads * head_dim, config.model_dim));

        Self {
            query_weights,
            key_weights,
            value_weights,
            output_weights,
            config,
            attention_cache: HashMap::new(),
        }
    }

    /// Compute multi-head attention
    pub fn forward(
        &mut self,
        input: &Array2<f64>,
        mask: Option<&Array2<bool>>,
    ) -> Result<(Array2<f64>, Array3<f64>)> {
        let seq_len = input.shape()[0]; // First dimension is sequence length
        let model_dim = input.shape()[1]; // Second dimension is model dimension
        let head_dim = self.config.model_dim / self.config.num_heads;

        let mut attention_outputs = Vec::new();
        let mut attention_weights = Array3::zeros((self.config.num_heads, seq_len, seq_len));

        // Process each attention head
        for head in 0..self.config.num_heads {
            // Extract head-specific weights manually
            let query_head_weights = self.extract_head_weights(&self.query_weights, head);
            let key_head_weights = self.extract_head_weights(&self.key_weights, head);
            let value_head_weights = self.extract_head_weights(&self.value_weights, head);

            let q = self.project_to_head(input, &query_head_weights, head)?;
            let k = self.project_to_head(input, &key_head_weights, head)?;
            let v = self.project_to_head(input, &value_head_weights, head)?;

            let (head_output, head_attention) =
                self.scaled_dot_product_attention(&q, &k, &v, mask)?;
            attention_outputs.push(head_output);

            // Safely assign attention weights for this head
            for i in 0..seq_len.min(head_attention.nrows()) {
                for j in 0..seq_len.min(head_attention.ncols()) {
                    attention_weights[[head, i, j]] = head_attention[[i, j]];
                }
            }
        }

        // Concatenate all heads
        let concatenated = self.concatenate_heads(&attention_outputs)?;

        // Apply output projection
        let output = concatenated.dot(&self.output_weights);

        Ok((output, attention_weights))
    }

    /// Extract head-specific weights from 3D weight matrix
    fn extract_head_weights(&self, weights: &Array3<f64>, head: usize) -> Array2<f64> {
        let model_dim = weights.shape()[1];
        let head_dim = weights.shape()[2];
        let mut head_weights = Array2::zeros((model_dim, head_dim));

        // Copy weights for this head
        for i in 0..model_dim {
            for j in 0..head_dim {
                head_weights[[i, j]] = weights[[head, i, j]];
            }
        }

        head_weights
    }

    /// Project input to attention head space
    fn project_to_head(
        &self,
        input: &Array2<f64>,
        weights: &Array2<f64>,
        head: usize,
    ) -> Result<Array2<f64>> {
        Ok(input.dot(weights))
    }

    /// Scaled dot-product attention
    fn scaled_dot_product_attention(
        &self,
        q: &Array2<f64>,
        k: &Array2<f64>,
        v: &Array2<f64>,
        mask: Option<&Array2<bool>>,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let d_k = k.shape()[1] as f64;
        let scale = 1.0 / d_k.sqrt();

        // Compute attention scores
        let scores = q.dot(&k.t()) * scale / self.config.attention_temperature;

        // Apply mask if provided
        let masked_scores = if let Some(mask) = mask {
            self.apply_attention_mask(&scores, mask)?
        } else {
            scores
        };

        // Apply softmax
        let attention_weights = self.softmax(&masked_scores)?;

        // Apply attention to values
        let output = attention_weights.dot(v);

        Ok((output, attention_weights))
    }

    /// Apply attention mask
    fn apply_attention_mask(
        &self,
        scores: &Array2<f64>,
        mask: &Array2<bool>,
    ) -> Result<Array2<f64>> {
        let mut masked_scores = scores.clone();

        for i in 0..scores.shape()[0] {
            for j in 0..scores.shape()[1] {
                if !mask[[i, j]] {
                    masked_scores[[i, j]] = f64::NEG_INFINITY;
                }
            }
        }

        Ok(masked_scores)
    }

    /// Softmax activation
    fn softmax(&self, input: &Array2<f64>) -> Result<Array2<f64>> {
        let mut output = Array2::zeros(input.raw_dim());

        for i in 0..input.shape()[0] {
            let row = input.row(i);
            let max_val = row.fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            let exp_row: Array1<f64> = row.mapv(|x| (x - max_val).exp());
            let sum_exp = exp_row.sum();

            if sum_exp > 0.0 {
                output.row_mut(i).assign(&(exp_row / sum_exp));
            }
        }

        Ok(output)
    }

    /// Concatenate attention heads
    fn concatenate_heads(&self, heads: &[Array2<f64>]) -> Result<Array2<f64>> {
        if heads.is_empty() {
            return Err(ShaclAiError::DataProcessing(
                "No attention heads provided".to_string(),
            ));
        }

        let seq_len = heads[0].shape()[0];
        let total_dim = heads.len() * heads[0].shape()[1];

        let mut concatenated = Array2::zeros((seq_len, total_dim));

        for (head_idx, head_output) in heads.iter().enumerate() {
            let start_dim = head_idx * head_output.shape()[1];
            let head_dim = head_output.shape()[1];

            // Safely assign head output to concatenated result
            for i in 0..head_output.shape()[0].min(concatenated.shape()[0]) {
                for j in 0..head_dim.min(concatenated.shape()[1] - start_dim) {
                    concatenated[[i, start_dim + j]] = head_output[[i, j]];
                }
            }
        }

        Ok(concatenated)
    }
}

/// Transformer encoder layer
#[derive(Debug)]
pub struct TransformerEncoderLayer {
    /// Multi-head attention
    attention: MultiHeadAttention,

    /// Feed-forward network weights
    ffn_weights1: Array2<f64>,
    ffn_weights2: Array2<f64>,
    ffn_bias1: Array1<f64>,
    ffn_bias2: Array1<f64>,

    /// Layer normalization parameters
    layer_norm1_weight: Array1<f64>,
    layer_norm1_bias: Array1<f64>,
    layer_norm2_weight: Array1<f64>,
    layer_norm2_bias: Array1<f64>,

    /// Configuration
    config: NeuralTransformerConfig,
}

impl TransformerEncoderLayer {
    pub fn new(config: NeuralTransformerConfig) -> Self {
        let attention = MultiHeadAttention::new(config.clone());

        let ffn_weights1 = Array2::zeros((config.model_dim, config.ffn_dim));
        let ffn_weights2 = Array2::zeros((config.ffn_dim, config.model_dim));
        let ffn_bias1 = Array1::zeros(config.ffn_dim);
        let ffn_bias2 = Array1::zeros(config.model_dim);

        let layer_norm1_weight = Array1::ones(config.model_dim);
        let layer_norm1_bias = Array1::zeros(config.model_dim);
        let layer_norm2_weight = Array1::ones(config.model_dim);
        let layer_norm2_bias = Array1::zeros(config.model_dim);

        Self {
            attention,
            ffn_weights1,
            ffn_weights2,
            ffn_bias1,
            ffn_bias2,
            layer_norm1_weight,
            layer_norm1_bias,
            layer_norm2_weight,
            layer_norm2_bias,
            config,
        }
    }

    /// Forward pass through encoder layer
    pub fn forward(
        &mut self,
        input: &Array2<f64>,
        mask: Option<&Array2<bool>>,
    ) -> Result<Array2<f64>> {
        // Multi-head attention with residual connection
        let (attention_output, _) = self.attention.forward(input, mask)?;
        let attention_residual = if self.config.enable_residual_connections {
            &attention_output + input
        } else {
            attention_output
        };

        // Layer normalization
        let norm1_output = if self.config.enable_layer_norm {
            self.layer_norm(
                &attention_residual,
                &self.layer_norm1_weight,
                &self.layer_norm1_bias,
            )?
        } else {
            attention_residual
        };

        // Feed-forward network
        let ffn_output = self.feed_forward(&norm1_output)?;

        // Residual connection
        let ffn_residual = if self.config.enable_residual_connections {
            &ffn_output + &norm1_output
        } else {
            ffn_output
        };

        // Final layer normalization
        let output = if self.config.enable_layer_norm {
            self.layer_norm(
                &ffn_residual,
                &self.layer_norm2_weight,
                &self.layer_norm2_bias,
            )?
        } else {
            ffn_residual
        };

        Ok(output)
    }

    /// Feed-forward network
    fn feed_forward(&self, input: &Array2<f64>) -> Result<Array2<f64>> {
        // First layer with ReLU activation
        let hidden = input.dot(&self.ffn_weights1) + &self.ffn_bias1;
        let activated = hidden.mapv(|x| x.max(0.0)); // ReLU

        // Second layer
        let output = activated.dot(&self.ffn_weights2) + &self.ffn_bias2;

        Ok(output)
    }

    /// Layer normalization
    fn layer_norm(
        &self,
        input: &Array2<f64>,
        weight: &Array1<f64>,
        bias: &Array1<f64>,
    ) -> Result<Array2<f64>> {
        let epsilon = 1e-5;
        let mut output = Array2::zeros(input.raw_dim());

        for i in 0..input.shape()[0] {
            let row = input.row(i);
            let mean = row.mean().unwrap_or(0.0);
            let variance = row.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);
            let std = (variance + epsilon).sqrt();

            let normalized = row.mapv(|x| (x - mean) / std);
            let scaled = &normalized * weight + bias;

            output.row_mut(i).assign(&scaled);
        }

        Ok(output)
    }
}

/// Full transformer encoder
#[derive(Debug)]
pub struct TransformerEncoder {
    /// Encoder layers
    layers: Vec<TransformerEncoderLayer>,

    /// Configuration
    config: NeuralTransformerConfig,
}

impl TransformerEncoder {
    pub fn new(config: NeuralTransformerConfig) -> Self {
        let mut layers = Vec::new();

        for _ in 0..config.num_layers {
            layers.push(TransformerEncoderLayer::new(config.clone()));
        }

        Self { layers, config }
    }

    /// Forward pass through all encoder layers
    pub fn forward(
        &mut self,
        input: &Array2<f64>,
        mask: Option<&Array2<bool>>,
    ) -> Result<Array2<f64>> {
        let mut current_input = input.clone();

        for layer in &mut self.layers {
            current_input = layer.forward(&current_input, mask)?;
        }

        Ok(current_input)
    }
}

/// Pattern embedder for converting patterns to vector representations
#[derive(Debug)]
pub struct PatternEmbedder {
    /// Embedding lookup table
    embedding_table: Array2<f64>, // [vocab_size, embedding_dim]

    /// Pattern vocabulary
    pattern_vocab: HashMap<String, usize>,

    /// Reverse vocabulary
    vocab_reverse: HashMap<usize, String>,

    /// Configuration
    config: NeuralTransformerConfig,
}

impl PatternEmbedder {
    pub fn new(config: NeuralTransformerConfig) -> Self {
        let vocab_size = 10000; // Large vocabulary for patterns
        let embedding_table = Array2::zeros((vocab_size, config.model_dim));

        Self {
            embedding_table,
            pattern_vocab: HashMap::new(),
            vocab_reverse: HashMap::new(),
            config,
        }
    }

    /// Convert pattern to embedding
    pub fn embed_pattern(&mut self, pattern: &AlgebraTriplePattern) -> Result<Array1<f64>> {
        let pattern_tokens = self.tokenize_pattern(pattern)?;
        let mut embedding = Array1::zeros(self.config.model_dim);

        for token in &pattern_tokens {
            let token_id = self.get_or_create_token_id(token);
            let token_embedding = self.embedding_table.row(token_id);
            embedding = embedding + token_embedding;
        }

        // Normalize by number of tokens
        if !pattern_tokens.is_empty() {
            embedding /= pattern_tokens.len() as f64;
        }

        Ok(embedding)
    }

    /// Convert patterns to sequence of embeddings
    pub fn embed_pattern_sequence(
        &mut self,
        patterns: &[AlgebraTriplePattern],
    ) -> Result<Array2<f64>> {
        let seq_len = patterns.len().min(self.config.max_sequence_length);
        let mut embeddings = Array2::zeros((seq_len, self.config.model_dim));

        for (i, pattern) in patterns.iter().take(seq_len).enumerate() {
            let pattern_embedding = self.embed_pattern(pattern)?;
            embeddings.row_mut(i).assign(&pattern_embedding);
        }

        Ok(embeddings)
    }

    /// Tokenize pattern into components
    fn tokenize_pattern(&self, pattern: &AlgebraTriplePattern) -> Result<Vec<String>> {
        let tokens = vec![
            self.term_to_token(&pattern.subject),
            self.term_to_token(&pattern.predicate),
            self.term_to_token(&pattern.object),
        ];

        Ok(tokens)
    }

    /// Convert term to token string
    fn term_to_token(&self, term: &AlgebraTermPattern) -> String {
        match term {
            AlgebraTermPattern::Variable(v) => format!("VAR_{}", v.as_str()),
            AlgebraTermPattern::NamedNode(n) => format!("NODE_{}", n.as_str()),
            AlgebraTermPattern::Literal(l) => format!("LIT_{}", l.value()),
            AlgebraTermPattern::BlankNode(b) => format!("BLANK_{}", b.as_str()),
            AlgebraTermPattern::QuotedTriple(triple) => {
                // RDF-star quoted triple - convert to string representation
                format!("QUOTED_TRIPLE_{:?}", triple)
            }
        }
    }

    /// Get or create token ID
    fn get_or_create_token_id(&mut self, token: &str) -> usize {
        if let Some(&id) = self.pattern_vocab.get(token) {
            id
        } else {
            let id = self.pattern_vocab.len();
            self.pattern_vocab.insert(token.to_string(), id);
            self.vocab_reverse.insert(id, token.to_string());
            id
        }
    }
}

/// Attention-based cost predictor
#[derive(Debug)]
pub struct AttentionCostPredictor {
    /// Pattern attention weights
    pattern_attention: Array2<f64>,

    /// Cost prediction head
    cost_head: Array2<f64>,

    /// Historical cost patterns
    cost_history: Vec<(Array1<f64>, f64)>, // (pattern_embedding, cost)

    /// Configuration
    config: NeuralTransformerConfig,
}

impl AttentionCostPredictor {
    pub fn new(config: NeuralTransformerConfig) -> Self {
        let pattern_attention = Array2::zeros((config.model_dim, config.model_dim));
        let cost_head = Array2::zeros((config.model_dim, 1));

        Self {
            pattern_attention,
            cost_head,
            cost_history: Vec::new(),
            config,
        }
    }

    /// Predict cost using attention mechanism
    pub fn predict_cost(
        &self,
        pattern_embedding: &Array1<f64>,
        context_embeddings: &[Array1<f64>],
    ) -> Result<f64> {
        // Compute attention weights over context
        let mut attention_weights = Vec::new();

        for context_emb in context_embeddings {
            let attention_score = pattern_embedding.dot(&self.pattern_attention.dot(context_emb));
            attention_weights.push(attention_score);
        }

        // Softmax normalization
        let max_weight = attention_weights
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_weights: Vec<f64> = attention_weights
            .iter()
            .map(|&w| (w - max_weight).exp())
            .collect();
        let sum_exp: f64 = exp_weights.iter().sum();

        let normalized_weights: Vec<f64> = if sum_exp > 0.0 {
            exp_weights.iter().map(|&w| w / sum_exp).collect()
        } else {
            vec![1.0 / context_embeddings.len() as f64; context_embeddings.len()]
        };

        // Weighted context vector
        let mut context_vector: Array1<f64> = Array1::zeros(self.config.model_dim);
        for (weight, context_emb) in normalized_weights.iter().zip(context_embeddings.iter()) {
            context_vector += &(context_emb * *weight);
        }

        // Combine pattern and context
        let combined = pattern_embedding + &context_vector;

        // Predict cost using first column of cost_head
        let cost_logits = combined.dot(&self.cost_head.column(0));
        let cost: f64 = cost_logits.max(0.1); // Ensure positive cost

        Ok(cost)
    }

    /// Update cost predictor with new data
    pub fn update(&mut self, pattern_embedding: Array1<f64>, actual_cost: f64) {
        self.cost_history.push((pattern_embedding, actual_cost));

        // Keep limited history
        if self.cost_history.len() > 1000 {
            self.cost_history.drain(0..100);
        }

        // Simple gradient update (in practice, would use more sophisticated optimization)
        if self.cost_history.len() > 10 {
            self.simple_gradient_update();
        }
    }

    /// Simple gradient update for cost prediction
    fn simple_gradient_update(&mut self) {
        let lr = self.config.learning_rate;

        // Update based on recent samples
        for (pattern_emb, actual_cost) in self.cost_history.iter().rev().take(10) {
            let predicted_cost = pattern_emb.dot(&self.cost_head.column(0));
            let error = actual_cost - predicted_cost;

            // Update cost head weights
            let gradient = pattern_emb * error * lr;
            self.cost_head.column_mut(0).scaled_add(1.0, &gradient);
        }
    }
}

/// Positional encoder for sequential pattern information
#[derive(Debug)]
pub struct PositionalEncoder {
    /// Positional encoding matrix
    position_encodings: Array2<f64>,

    /// Configuration
    config: NeuralTransformerConfig,
}

impl PositionalEncoder {
    pub fn new(config: NeuralTransformerConfig) -> Self {
        let mut position_encodings = Array2::zeros((config.max_sequence_length, config.model_dim));

        // Generate sinusoidal position encodings
        for pos in 0..config.max_sequence_length {
            for i in 0..config.model_dim {
                if i % 2 == 0 {
                    let angle = pos as f64 / 10000_f64.powf(i as f64 / config.model_dim as f64);
                    position_encodings[[pos, i]] = angle.sin();
                } else {
                    let angle =
                        pos as f64 / 10000_f64.powf((i - 1) as f64 / config.model_dim as f64);
                    position_encodings[[pos, i]] = angle.cos();
                }
            }
        }

        Self {
            position_encodings,
            config,
        }
    }

    /// Add positional encoding to embeddings
    pub fn encode(&self, embeddings: &Array2<f64>) -> Result<Array2<f64>> {
        let seq_len = embeddings.shape()[0].min(self.config.max_sequence_length);
        let mut encoded = embeddings.clone();

        for i in 0..seq_len {
            let pos_encoding = self.position_encodings.row(i);
            encoded.row_mut(i).scaled_add(1.0, &pos_encoding);
        }

        Ok(encoded)
    }
}

/// Memory bank for storing and retrieving pattern information
#[derive(Debug, Clone)]
pub struct PatternMemoryBank {
    /// Memory storage
    memory: Vec<PatternMemoryEntry>,

    /// Memory access patterns
    access_patterns: HashMap<String, usize>,

    /// Configuration
    config: NeuralTransformerConfig,
}

#[derive(Debug, Clone)]
pub struct PatternMemoryEntry {
    pub pattern: AlgebraTriplePattern,
    pub embedding: Array1<f64>,
    pub cost: f64,
    pub frequency: usize,
    pub last_accessed: std::time::Instant,
}

impl PatternMemoryBank {
    pub fn new(config: NeuralTransformerConfig) -> Self {
        Self {
            memory: Vec::new(),
            access_patterns: HashMap::new(),
            config,
        }
    }

    /// Store pattern in memory
    pub fn store(&mut self, pattern: AlgebraTriplePattern, embedding: Array1<f64>, cost: f64) {
        let pattern_key = format!("{pattern:?}");

        // Check if pattern already exists
        if let Some(entry) = self
            .memory
            .iter_mut()
            .find(|e| format!("{:?}", e.pattern) == pattern_key)
        {
            entry.frequency += 1;
            entry.last_accessed = std::time::Instant::now();
            entry.cost = (entry.cost + cost) / 2.0; // Running average
        } else {
            // Add new entry
            self.memory.push(PatternMemoryEntry {
                pattern,
                embedding,
                cost,
                frequency: 1,
                last_accessed: std::time::Instant::now(),
            });

            // Maintain memory size limit
            if self.memory.len() > self.config.memory_bank_size {
                self.evict_oldest();
            }
        }

        self.access_patterns
            .insert(pattern_key, self.memory.len() - 1);
    }

    /// Retrieve similar patterns from memory
    pub fn retrieve_similar(
        &self,
        query_embedding: &Array1<f64>,
        top_k: usize,
    ) -> Vec<&PatternMemoryEntry> {
        let mut similarities: Vec<(f64, &PatternMemoryEntry)> = self
            .memory
            .iter()
            .map(|entry| {
                let similarity = self.cosine_similarity(query_embedding, &entry.embedding);
                (similarity, entry)
            })
            .collect();

        similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        similarities
            .into_iter()
            .take(top_k)
            .map(|(_, entry)| entry)
            .collect()
    }

    /// Calculate cosine similarity
    fn cosine_similarity(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        let dot_product = a.dot(b);
        let norm_a = a.dot(a).sqrt();
        let norm_b = b.dot(b).sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    /// Evict oldest entry
    fn evict_oldest(&mut self) {
        if let Some(oldest_idx) = self
            .memory
            .iter()
            .enumerate()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(idx, _)| idx)
        {
            self.memory.remove(oldest_idx);
        }
    }
}

/// Performance statistics for neural transformer integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralTransformerStats {
    pub total_pattern_optimizations: usize,
    pub attention_computations: usize,
    pub memory_retrievals: usize,
    pub cost_predictions: usize,
    pub average_attention_accuracy: f64,
    pub memory_hit_rate: f64,
    pub cost_prediction_error: f64,
    pub transformer_inference_time: std::time::Duration,
    pub memory_usage_mb: f64,
}

impl Default for NeuralTransformerStats {
    fn default() -> Self {
        Self {
            total_pattern_optimizations: 0,
            attention_computations: 0,
            memory_retrievals: 0,
            cost_predictions: 0,
            average_attention_accuracy: 0.0,
            memory_hit_rate: 0.0,
            cost_prediction_error: 0.0,
            transformer_inference_time: std::time::Duration::from_secs(0),
            memory_usage_mb: 0.0,
        }
    }
}

impl NeuralTransformerPatternIntegration {
    /// Create new neural transformer pattern integration
    pub fn new(config: NeuralTransformerConfig) -> Result<Self> {
        let attention_mechanism = Arc::new(Mutex::new(MultiHeadAttention::new(config.clone())));
        let transformer_encoder = Arc::new(Mutex::new(TransformerEncoder::new(config.clone())));
        let pattern_embedder = Arc::new(Mutex::new(PatternEmbedder::new(config.clone())));
        let attention_cost_predictor =
            Arc::new(Mutex::new(AttentionCostPredictor::new(config.clone())));
        let neural_recognizer = Arc::new(Mutex::new(NeuralPatternRecognizer::new(
            NeuralPatternConfig::default(),
        )));
        let positional_encoder = Arc::new(Mutex::new(PositionalEncoder::new(config.clone())));
        let pattern_memory = Arc::new(Mutex::new(PatternMemoryBank::new(config.clone())));

        Ok(Self {
            attention_mechanism,
            transformer_encoder,
            pattern_embedder,
            attention_cost_predictor,
            neural_recognizer,
            positional_encoder,
            pattern_memory,
            config,
            stats: NeuralTransformerStats::default(),
        })
    }

    /// Optimize patterns using neural transformer integration
    pub fn optimize_patterns_with_attention(
        &mut self,
        patterns: &[AlgebraTriplePattern],
    ) -> Result<OptimizedPatternPlan> {
        let start_time = Instant::now();

        // Embed patterns
        let pattern_embeddings = self.embed_patterns(patterns)?;

        // Apply positional encoding
        let positioned_embeddings = self.apply_positional_encoding(&pattern_embeddings)?;

        // Transformer encoding with attention
        let attention_enhanced_embeddings = self.transformer_encode(&positioned_embeddings)?;

        // Predict costs using attention
        let predicted_costs =
            self.predict_costs_with_attention(&attention_enhanced_embeddings, patterns)?;

        // Generate optimal plan
        let optimized_plan = self.generate_attention_based_plan(
            patterns,
            &predicted_costs,
            &attention_enhanced_embeddings,
        )?;

        // Update memory and statistics
        self.update_memory_and_stats(patterns, &attention_enhanced_embeddings, &predicted_costs)?;

        self.stats.transformer_inference_time += start_time.elapsed();
        self.stats.total_pattern_optimizations += 1;

        Ok(optimized_plan)
    }

    /// Embed patterns using pattern embedder
    fn embed_patterns(&mut self, patterns: &[AlgebraTriplePattern]) -> Result<Array2<f64>> {
        match self.pattern_embedder.lock() {
            Ok(mut embedder) => embedder.embed_pattern_sequence(patterns),
            _ => Err(ShaclAiError::DataProcessing(
                "Failed to lock pattern embedder".to_string(),
            )),
        }
    }

    /// Apply positional encoding
    fn apply_positional_encoding(&self, embeddings: &Array2<f64>) -> Result<Array2<f64>> {
        match self.positional_encoder.lock() {
            Ok(encoder) => encoder.encode(embeddings),
            _ => Err(ShaclAiError::DataProcessing(
                "Failed to lock positional encoder".to_string(),
            )),
        }
    }

    /// Transform embeddings using transformer encoder
    fn transformer_encode(&mut self, embeddings: &Array2<f64>) -> Result<Array2<f64>> {
        match self.transformer_encoder.lock() {
            Ok(mut encoder) => encoder.forward(embeddings, None),
            _ => Err(ShaclAiError::DataProcessing(
                "Failed to lock transformer encoder".to_string(),
            )),
        }
    }

    /// Predict costs using attention mechanism
    fn predict_costs_with_attention(
        &mut self,
        embeddings: &Array2<f64>,
        patterns: &[AlgebraTriplePattern],
    ) -> Result<Vec<f64>> {
        let mut costs = Vec::new();

        match self.attention_cost_predictor.lock() {
            Ok(predictor) => {
                for i in 0..patterns.len() {
                    let pattern_emb = embeddings.row(i).to_owned();

                    // Use other patterns as context
                    let context_embeddings: Vec<Array1<f64>> = (0..patterns.len())
                        .filter(|&j| j != i)
                        .map(|j| embeddings.row(j).to_owned())
                        .collect();

                    let cost = predictor.predict_cost(&pattern_emb, &context_embeddings)?;
                    costs.push(cost);
                }

                self.stats.cost_predictions += patterns.len();
            }
            _ => {
                return Err(ShaclAiError::DataProcessing(
                    "Failed to lock attention cost predictor".to_string(),
                ));
            }
        }

        Ok(costs)
    }

    /// Generate optimized plan based on attention analysis
    fn generate_attention_based_plan(
        &self,
        patterns: &[AlgebraTriplePattern],
        costs: &[f64],
        embeddings: &Array2<f64>,
    ) -> Result<OptimizedPatternPlan> {
        let mut pattern_strategies = Vec::new();
        let mut total_cost = 0.0;
        let mut binding_order = Vec::new();
        let mut bound_vars = HashSet::new();

        // Create pattern-cost pairs and sort by cost
        let mut indexed_costs: Vec<(usize, f64)> = costs
            .iter()
            .enumerate()
            .map(|(i, &cost)| (i, cost))
            .collect();

        indexed_costs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Build optimized plan
        for (pattern_idx, cost) in indexed_costs {
            let pattern = &patterns[pattern_idx];
            let pattern_embedding = embeddings.row(pattern_idx);

            // Determine optimal index using attention weights
            let optimal_index =
                self.determine_optimal_index_with_attention(pattern, &pattern_embedding)?;

            let strategy = PatternStrategy {
                index_type: optimal_index,
                estimated_cost: cost,
                selectivity: self.estimate_selectivity_with_attention(&pattern_embedding)?,
                bound_vars: self.extract_variables(pattern),
                pushdown_filters: Vec::new(),
            };

            pattern_strategies.push((pattern.clone(), strategy.clone()));
            total_cost += cost;
            bound_vars.extend(strategy.bound_vars.clone());
            binding_order.push(bound_vars.clone());
        }

        Ok(OptimizedPatternPlan {
            patterns: pattern_strategies,
            total_cost,
            binding_order,
        })
    }

    /// Determine optimal index using attention weights
    fn determine_optimal_index_with_attention(
        &self,
        pattern: &AlgebraTriplePattern,
        embedding: &scirs2_core::ndarray_ext::ArrayView1<f64>,
    ) -> Result<IndexType> {
        // Simple heuristic based on pattern structure
        let s_bound = !matches!(pattern.subject, AlgebraTermPattern::Variable(_));
        let p_bound = !matches!(pattern.predicate, AlgebraTermPattern::Variable(_));
        let o_bound = !matches!(pattern.object, AlgebraTermPattern::Variable(_));

        // In a full implementation, this would use attention weights to make the decision
        let optimal_index = match (s_bound, p_bound, o_bound) {
            (true, true, _) => IndexType::SPO,
            (true, false, true) => IndexType::SPO,
            (false, true, true) => IndexType::POS,
            (true, false, false) => IndexType::SPO,
            (false, true, false) => IndexType::POS,
            (false, false, true) => IndexType::OSP,
            (false, false, false) => IndexType::SPO,
        };

        Ok(optimal_index)
    }

    /// Estimate selectivity using attention mechanism
    fn estimate_selectivity_with_attention(
        &self,
        embedding: &scirs2_core::ndarray_ext::ArrayView1<f64>,
    ) -> Result<f64> {
        // Simple selectivity estimation based on embedding magnitude
        let embedding_norm = embedding.dot(embedding).sqrt();
        let selectivity = (1.0 / (1.0 + embedding_norm)).clamp(0.001, 0.9);

        Ok(selectivity)
    }

    /// Extract variables from pattern
    fn extract_variables(&self, pattern: &AlgebraTriplePattern) -> HashSet<Variable> {
        let mut vars = HashSet::new();

        if let AlgebraTermPattern::Variable(v) = &pattern.subject {
            vars.insert(v.clone());
        }
        if let AlgebraTermPattern::Variable(v) = &pattern.predicate {
            vars.insert(v.clone());
        }
        if let AlgebraTermPattern::Variable(v) = &pattern.object {
            vars.insert(v.clone());
        }

        vars
    }

    /// Update memory bank and statistics
    fn update_memory_and_stats(
        &mut self,
        patterns: &[AlgebraTriplePattern],
        embeddings: &Array2<f64>,
        costs: &[f64],
    ) -> Result<()> {
        if let Ok(mut memory) = self.pattern_memory.lock() {
            for (i, pattern) in patterns.iter().enumerate() {
                let embedding = embeddings.row(i).to_owned();
                let cost = costs[i];

                memory.store(pattern.clone(), embedding, cost);
            }

            self.stats.memory_retrievals += patterns.len();
        }

        Ok(())
    }

    /// Process pattern embeddings through the transformer
    ///
    /// This method takes raw pattern embeddings and processes them through
    /// the transformer architecture to produce attention-enhanced embeddings.
    pub fn process_pattern_embeddings(&mut self, embeddings: &Array2<f64>) -> Result<Array2<f64>> {
        // Apply positional encoding
        let positioned_embeddings = self.apply_positional_encoding(embeddings)?;

        // Process through transformer encoder
        let enhanced_embeddings = self.transformer_encode(&positioned_embeddings)?;

        // Update statistics
        self.stats.attention_computations += embeddings.shape()[0];

        Ok(enhanced_embeddings)
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> NeuralTransformerStats {
        self.stats.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::{NamedNode, Variable};

    #[test]
    fn test_multi_head_attention() {
        let config = NeuralTransformerConfig::default();
        let mut attention = MultiHeadAttention::new(config.clone());

        let input = Array2::zeros((10, config.model_dim)); // 10 sequence length, model_dim
        let result = attention.forward(&input, None);

        assert!(result.is_ok());
        let (output, attention_weights) = result.unwrap();
        assert_eq!(output.shape(), [10, config.model_dim]);
        assert_eq!(attention_weights.shape(), [config.num_heads, 10, 10]); // heads x seq_len x seq_len attention
    }

    #[test]
    fn test_pattern_embedder() {
        let config = NeuralTransformerConfig::default();
        let mut embedder = PatternEmbedder::new(config);

        let pattern = AlgebraTriplePattern::new(
            AlgebraTermPattern::Variable(Variable::new("s").unwrap()),
            AlgebraTermPattern::NamedNode(NamedNode::new("http://example.org/p").unwrap()),
            AlgebraTermPattern::Variable(Variable::new("o").unwrap()),
        );

        let embedding = embedder.embed_pattern(&pattern);
        assert!(embedding.is_ok());
        assert_eq!(embedding.unwrap().len(), 512);
    }

    #[test]
    fn test_attention_cost_predictor() {
        let config = NeuralTransformerConfig::default();
        let predictor = AttentionCostPredictor::new(config);

        let pattern_emb = Array1::zeros(512);
        let context_embs = vec![Array1::zeros(512), Array1::ones(512)];

        let cost = predictor.predict_cost(&pattern_emb, &context_embs);
        assert!(cost.is_ok());
        assert!(cost.unwrap() > 0.0);
    }

    #[test]
    fn test_positional_encoder() {
        let config = NeuralTransformerConfig::default();
        let encoder = PositionalEncoder::new(config);

        let embeddings = Array2::zeros((10, 512));
        let encoded = encoder.encode(&embeddings);

        assert!(encoded.is_ok());
        assert_eq!(encoded.unwrap().shape(), [10, 512]);
    }

    #[test]
    fn test_pattern_memory_bank() {
        let config = NeuralTransformerConfig::default();
        let mut memory = PatternMemoryBank::new(config);

        let pattern = AlgebraTriplePattern::new(
            AlgebraTermPattern::Variable(Variable::new("s").unwrap()),
            AlgebraTermPattern::NamedNode(NamedNode::new("http://example.org/p").unwrap()),
            AlgebraTermPattern::Variable(Variable::new("o").unwrap()),
        );

        let embedding = Array1::zeros(512);
        memory.store(pattern, embedding.clone(), 10.0);

        let similar = memory.retrieve_similar(&embedding, 1);
        assert_eq!(similar.len(), 1);
    }

    #[test]
    fn test_neural_transformer_integration() {
        let config = NeuralTransformerConfig::default();
        let mut integration = NeuralTransformerPatternIntegration::new(config).unwrap();

        let patterns = vec![AlgebraTriplePattern::new(
            AlgebraTermPattern::Variable(Variable::new("s").unwrap()),
            AlgebraTermPattern::NamedNode(NamedNode::new("http://example.org/p").unwrap()),
            AlgebraTermPattern::Variable(Variable::new("o").unwrap()),
        )];

        let result = integration.optimize_patterns_with_attention(&patterns);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert_eq!(plan.patterns.len(), 1);
        assert!(plan.total_cost > 0.0);
    }
}
