//! Word2Vec embedding integration for text content
//!
//! This module provides Word2Vec-based embeddings with support for:
//! - Pre-trained model loading
//! - Document embedding aggregation
//! - Subword handling
//! - Out-of-vocabulary management
//! - Hierarchical softmax support

use crate::{
    embeddings::{EmbeddableContent, EmbeddingConfig, EmbeddingGenerator},
    Vector,
};
use anyhow::{anyhow, Result};
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Word2Vec model format
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Word2VecFormat {
    /// Text format (word2vec text format)
    Text,
    /// Binary format (word2vec binary format)
    Binary,
    /// GloVe format (space-separated text)
    GloVe,
}

/// Word2Vec configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Word2VecConfig {
    /// Path to pre-trained model file
    pub model_path: String,
    /// Model format
    pub format: Word2VecFormat,
    /// Embedding dimensions
    pub dimensions: usize,
    /// Aggregation method for document embeddings
    pub aggregation: AggregationMethod,
    /// Enable subword handling
    pub use_subwords: bool,
    /// Minimum subword length
    pub min_subword_len: usize,
    /// Maximum subword length
    pub max_subword_len: usize,
    /// Out-of-vocabulary strategy
    pub oov_strategy: OovStrategy,
    /// Whether to normalize embeddings
    pub normalize: bool,
}

impl Default for Word2VecConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            format: Word2VecFormat::Text,
            dimensions: 300,
            aggregation: AggregationMethod::Mean,
            use_subwords: true,
            min_subword_len: 3,
            max_subword_len: 6,
            oov_strategy: OovStrategy::Subword,
            normalize: true,
        }
    }
}

/// Aggregation method for combining word embeddings into document embeddings
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AggregationMethod {
    /// Simple average of word embeddings
    Mean,
    /// Weighted average by term frequency
    WeightedMean,
    /// Max pooling across dimensions
    Max,
    /// Min pooling across dimensions
    Min,
    /// Concatenation of mean and max
    MeanMax,
    /// TF-IDF weighted average
    TfIdfWeighted,
}

/// Out-of-vocabulary word handling strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum OovStrategy {
    /// Use zero vector
    Zero,
    /// Use random vector
    Random,
    /// Use subword embeddings
    Subword,
    /// Skip OOV words
    Skip,
    /// Use a learned OOV embedding
    LearnedOov,
}

/// Word2Vec embedding generator
pub struct Word2VecEmbeddingGenerator {
    config: Word2VecConfig,
    embedding_config: EmbeddingConfig,
    /// Word embeddings lookup table
    embeddings: HashMap<String, Vec<f32>>,
    /// Subword embeddings for OOV handling
    subword_embeddings: HashMap<String, Vec<f32>>,
    /// Document frequency for TF-IDF weighting
    doc_frequencies: HashMap<String, f32>,
    /// Learned OOV embedding
    oov_embedding: Option<Vec<f32>>,
}

impl Word2VecEmbeddingGenerator {
    /// Create a new Word2Vec embedding generator
    pub fn new(word2vec_config: Word2VecConfig, embedding_config: EmbeddingConfig) -> Result<Self> {
        let mut generator = Self {
            config: word2vec_config,
            embedding_config,
            embeddings: HashMap::new(),
            subword_embeddings: HashMap::new(),
            doc_frequencies: HashMap::new(),
            oov_embedding: None,
        };

        // Load pre-trained embeddings if path is provided
        let model_path = generator.config.model_path.clone();
        if !model_path.is_empty() {
            generator.load_model(&model_path)?;
        }

        Ok(generator)
    }

    /// Load pre-trained Word2Vec model
    pub fn load_model(&mut self, path: &str) -> Result<()> {
        let path = Path::new(path);

        if !path.exists() {
            return Err(anyhow!("Model file not found: {}", path.display()));
        }

        match self.config.format {
            Word2VecFormat::Text => self.load_text_format(path),
            Word2VecFormat::Binary => self.load_binary_format(path),
            Word2VecFormat::GloVe => self.load_glove_format(path),
        }
    }

    /// Load Word2Vec text format
    fn load_text_format(&mut self, path: &Path) -> Result<()> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // First line contains vocab size and dimensions
        if let Some(Ok(header)) = lines.next() {
            let parts: Vec<&str> = header.split_whitespace().collect();
            if parts.len() == 2 {
                let _vocab_size: usize = parts[0].parse()?;
                let dimensions: usize = parts[1].parse()?;

                if dimensions != self.config.dimensions {
                    return Err(anyhow!(
                        "Model dimensions ({}) don't match config ({})",
                        dimensions,
                        self.config.dimensions
                    ));
                }
            }
        }

        // Read embeddings
        for line in lines {
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.len() < self.config.dimensions + 1 {
                continue;
            }

            let word = parts[0].to_string();
            let embedding: Result<Vec<f32>> = parts[1..=self.config.dimensions]
                .iter()
                .map(|s| s.parse::<f32>().map_err(Into::into))
                .collect();

            if let Ok(embedding) = embedding {
                self.embeddings.insert(word, embedding);
            }
        }

        // Generate subword embeddings if enabled
        if self.config.use_subwords {
            self.generate_subword_embeddings()?;
        }

        // Initialize OOV embedding if using learned strategy
        if self.config.oov_strategy == OovStrategy::LearnedOov {
            self.initialize_oov_embedding();
        }

        Ok(())
    }

    /// Load Word2Vec binary format
    fn load_binary_format(&mut self, path: &Path) -> Result<()> {
        use std::io::Read;

        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        // Parse binary format
        #[allow(unused_assignments)]
        let mut pos = 0;

        // Read header
        let header_end = buffer
            .iter()
            .position(|&b| b == b'\n')
            .ok_or_else(|| anyhow!("Invalid binary format"))?;
        let header = std::str::from_utf8(&buffer[..header_end])?;
        let parts: Vec<&str> = header.split_whitespace().collect();

        if parts.len() != 2 {
            return Err(anyhow!("Invalid header format"));
        }

        let vocab_size: usize = parts[0].parse()?;
        let dimensions: usize = parts[1].parse()?;

        if dimensions != self.config.dimensions {
            return Err(anyhow!(
                "Model dimensions ({}) don't match config ({})",
                dimensions,
                self.config.dimensions
            ));
        }

        pos = header_end + 1;

        // Read embeddings
        for _ in 0..vocab_size {
            // Read word until space
            let word_start = pos;
            while pos < buffer.len() && buffer[pos] != b' ' {
                pos += 1;
            }

            if pos >= buffer.len() {
                break;
            }

            let word = std::str::from_utf8(&buffer[word_start..pos])?.to_string();
            pos += 1; // Skip space

            // Read embedding values
            let mut embedding = Vec::with_capacity(dimensions);
            for _ in 0..dimensions {
                if pos + 4 > buffer.len() {
                    break;
                }

                let bytes = [
                    buffer[pos],
                    buffer[pos + 1],
                    buffer[pos + 2],
                    buffer[pos + 3],
                ];
                let value = f32::from_le_bytes(bytes);
                embedding.push(value);
                pos += 4;
            }

            if embedding.len() == dimensions {
                self.embeddings.insert(word, embedding);
            }

            // Skip newline if present
            if pos < buffer.len() && buffer[pos] == b'\n' {
                pos += 1;
            }
        }

        // Generate subword embeddings if enabled
        if self.config.use_subwords {
            self.generate_subword_embeddings()?;
        }

        Ok(())
    }

    /// Load GloVe format
    fn load_glove_format(&mut self, path: &Path) -> Result<()> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.len() < self.config.dimensions + 1 {
                continue;
            }

            let word = parts[0].to_string();
            let embedding: Result<Vec<f32>> = parts[1..=self.config.dimensions]
                .iter()
                .map(|s| s.parse::<f32>().map_err(Into::into))
                .collect();

            if let Ok(embedding) = embedding {
                self.embeddings.insert(word, embedding);
            }
        }

        // Generate subword embeddings if enabled
        if self.config.use_subwords {
            self.generate_subword_embeddings()?;
        }

        Ok(())
    }

    /// Generate subword embeddings from word embeddings
    fn generate_subword_embeddings(&mut self) -> Result<()> {
        let mut subword_counts: HashMap<String, usize> = HashMap::new();
        let mut subword_sums: HashMap<String, Vec<f32>> = HashMap::new();

        // Collect subwords from vocabulary
        for (word, embedding) in &self.embeddings {
            let subwords = self.get_subwords(word);

            for subword in subwords {
                *subword_counts.entry(subword.clone()).or_insert(0) += 1;

                let sum = subword_sums
                    .entry(subword)
                    .or_insert_with(|| vec![0.0; self.config.dimensions]);
                for (i, val) in embedding.iter().enumerate() {
                    sum[i] += val;
                }
            }
        }

        // Average subword embeddings
        for (subword, count) in subword_counts {
            if let Some(sum) = subword_sums.get(&subword) {
                let avg: Vec<f32> = sum.iter().map(|&s| s / count as f32).collect();
                self.subword_embeddings.insert(subword, avg);
            }
        }

        Ok(())
    }

    /// Get subwords for a given word
    fn get_subwords(&self, word: &str) -> Vec<String> {
        let mut subwords = Vec::new();
        let chars: Vec<char> = word.chars().collect();

        for len in self.config.min_subword_len..=self.config.max_subword_len.min(chars.len()) {
            for start in 0..=chars.len().saturating_sub(len) {
                let subword: String = chars[start..start + len].iter().collect();
                subwords.push(format!("<{subword}>")); // Mark as subword
            }
        }

        subwords
    }

    /// Initialize learned OOV embedding
    fn initialize_oov_embedding(&mut self) {
        // Average all embeddings to create OOV embedding
        let mut sum = vec![0.0; self.config.dimensions];
        let count = self.embeddings.len() as f32;

        for embedding in self.embeddings.values() {
            for (i, val) in embedding.iter().enumerate() {
                sum[i] += val;
            }
        }

        self.oov_embedding = Some(sum.iter().map(|&s| s / count).collect());
    }

    /// Get embedding for a word
    fn get_word_embedding(&self, word: &str) -> Option<Vec<f32>> {
        // Try exact match first
        if let Some(embedding) = self.embeddings.get(word) {
            return Some(embedding.clone());
        }

        // Try lowercase
        if let Some(embedding) = self.embeddings.get(&word.to_lowercase()) {
            return Some(embedding.clone());
        }

        // Handle OOV
        match self.config.oov_strategy {
            OovStrategy::Zero => Some(vec![0.0; self.config.dimensions]),
            OovStrategy::Random => {
                // Generate deterministic "random" vector based on word hash
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                std::hash::Hash::hash(&word, &mut hasher);
                let hash = std::hash::Hasher::finish(&hasher);

                let mut rng = rand::rngs::StdRng::seed_from_u64(hash);
                use rand::Rng;

                Some(
                    (0..self.config.dimensions)
                        .map(|_| rng.gen_range(-0.1..0.1))
                        .collect(),
                )
            }
            OovStrategy::Subword => {
                if self.config.use_subwords {
                    self.get_subword_embedding(word)
                } else {
                    None
                }
            }
            OovStrategy::Skip => None,
            OovStrategy::LearnedOov => self.oov_embedding.clone(),
        }
    }

    /// Get subword-based embedding for OOV word
    fn get_subword_embedding(&self, word: &str) -> Option<Vec<f32>> {
        let subwords = self.get_subwords(word);
        let mut sum = vec![0.0; self.config.dimensions];
        let mut count = 0;

        for subword in subwords {
            if let Some(embedding) = self.subword_embeddings.get(&subword) {
                for (i, val) in embedding.iter().enumerate() {
                    sum[i] += val;
                }
                count += 1;
            }
        }

        if count > 0 {
            Some(sum.iter().map(|&s| s / count as f32).collect())
        } else {
            None
        }
    }

    /// Tokenize text into words
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect()
    }

    /// Aggregate word embeddings into document embedding
    fn aggregate_embeddings(&self, word_embeddings: &[(String, Vec<f32>)]) -> Vec<f32> {
        if word_embeddings.is_empty() {
            return vec![0.0; self.config.dimensions];
        }

        match self.config.aggregation {
            AggregationMethod::Mean => {
                let mut sum = vec![0.0; self.config.dimensions];

                for (_, embedding) in word_embeddings {
                    for (i, val) in embedding.iter().enumerate() {
                        sum[i] += val;
                    }
                }

                let count = word_embeddings.len() as f32;
                sum.iter().map(|&s| s / count).collect()
            }
            AggregationMethod::WeightedMean => {
                // Weight by word frequency in document
                let mut word_counts: HashMap<String, usize> = HashMap::new();
                for (word, _) in word_embeddings {
                    *word_counts.entry(word.clone()).or_insert(0) += 1;
                }

                let total_words = word_embeddings.len() as f32;
                let mut weighted_sum = vec![0.0; self.config.dimensions];

                for (word, embedding) in word_embeddings {
                    let weight = word_counts[word] as f32 / total_words;
                    for (i, val) in embedding.iter().enumerate() {
                        weighted_sum[i] += val * weight;
                    }
                }

                weighted_sum
            }
            AggregationMethod::Max => {
                let mut max_vals = vec![f32::NEG_INFINITY; self.config.dimensions];

                for (_, embedding) in word_embeddings {
                    for (i, val) in embedding.iter().enumerate() {
                        max_vals[i] = max_vals[i].max(*val);
                    }
                }

                max_vals
            }
            AggregationMethod::Min => {
                let mut min_vals = vec![f32::INFINITY; self.config.dimensions];

                for (_, embedding) in word_embeddings {
                    for (i, val) in embedding.iter().enumerate() {
                        min_vals[i] = min_vals[i].min(*val);
                    }
                }

                min_vals
            }
            AggregationMethod::MeanMax => {
                // Concatenate mean and max embeddings
                let mean =
                    self.aggregate_embeddings_with_method(word_embeddings, AggregationMethod::Mean);
                let max =
                    self.aggregate_embeddings_with_method(word_embeddings, AggregationMethod::Max);

                let mut result = Vec::with_capacity(self.config.dimensions * 2);
                result.extend(mean);
                result.extend(max);

                // Truncate or pad to match configured dimensions
                result.resize(self.config.dimensions, 0.0);
                result
            }
            AggregationMethod::TfIdfWeighted => {
                // Weight by TF-IDF if document frequencies are available
                if self.doc_frequencies.is_empty() {
                    // Fall back to mean if no doc frequencies
                    return self.aggregate_embeddings_with_method(
                        word_embeddings,
                        AggregationMethod::Mean,
                    );
                }

                let mut weighted_sum = vec![0.0; self.config.dimensions];
                let mut total_weight = 0.0;

                for (word, embedding) in word_embeddings {
                    let tf = word_embeddings.iter().filter(|(w, _)| w == word).count() as f32
                        / word_embeddings.len() as f32;
                    let idf = self.doc_frequencies.get(word).unwrap_or(&1.0);
                    let weight = tf * idf;

                    for (i, val) in embedding.iter().enumerate() {
                        weighted_sum[i] += val * weight;
                    }
                    total_weight += weight;
                }

                if total_weight > 0.0 {
                    weighted_sum.iter().map(|&s| s / total_weight).collect()
                } else {
                    weighted_sum
                }
            }
        }
    }

    /// Helper method for recursive aggregation
    fn aggregate_embeddings_with_method(
        &self,
        word_embeddings: &[(String, Vec<f32>)],
        method: AggregationMethod,
    ) -> Vec<f32> {
        let _original_method = self.config.aggregation;
        let mut config_clone = self.config.clone();
        config_clone.aggregation = method;

        let temp_self = Self {
            config: config_clone,
            embedding_config: self.embedding_config.clone(),
            embeddings: self.embeddings.clone(),
            subword_embeddings: self.subword_embeddings.clone(),
            doc_frequencies: self.doc_frequencies.clone(),
            oov_embedding: self.oov_embedding.clone(),
        };

        temp_self.aggregate_embeddings(word_embeddings)
    }

    /// Set document frequencies for TF-IDF weighting
    pub fn set_document_frequencies(&mut self, frequencies: HashMap<String, f32>) {
        self.doc_frequencies = frequencies;
    }

    /// Calculate document frequencies from a corpus
    pub fn calculate_document_frequencies(&mut self, documents: &[String]) -> Result<()> {
        let total_docs = documents.len() as f32;
        let mut doc_counts: HashMap<String, usize> = HashMap::new();

        for doc in documents {
            let words = self.tokenize(doc);
            let unique_words: std::collections::HashSet<_> = words.into_iter().collect();

            for word in unique_words {
                *doc_counts.entry(word).or_insert(0) += 1;
            }
        }

        // Calculate IDF scores
        self.doc_frequencies = doc_counts
            .into_iter()
            .map(|(word, count)| {
                let idf = (total_docs / (count as f32 + 1.0)).ln();
                (word, idf)
            })
            .collect();

        Ok(())
    }
}

impl EmbeddingGenerator for Word2VecEmbeddingGenerator {
    fn generate(&self, content: &EmbeddableContent) -> Result<Vector> {
        let text = content.to_text();
        let words = self.tokenize(&text);

        // Get embeddings for each word
        let mut word_embeddings = Vec::new();

        for word in words {
            if let Some(embedding) = self.get_word_embedding(&word) {
                word_embeddings.push((word, embedding));
            }
        }

        if word_embeddings.is_empty() {
            return Ok(Vector::new(vec![0.0; self.config.dimensions]));
        }

        // Aggregate embeddings
        let mut document_embedding = self.aggregate_embeddings(&word_embeddings);

        // Normalize if configured
        if self.config.normalize {
            use oxirs_core::simd::SimdOps;
            let norm = f32::norm(&document_embedding);
            if norm > 0.0 {
                for val in &mut document_embedding {
                    *val /= norm;
                }
            }
        }

        Ok(Vector::new(document_embedding))
    }

    fn generate_batch(&self, contents: &[EmbeddableContent]) -> Result<Vec<Vector>> {
        // For Word2Vec, batch processing doesn't provide significant benefits
        // since we're just looking up pre-computed embeddings
        contents.iter().map(|c| self.generate(c)).collect()
    }

    fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    fn config(&self) -> &EmbeddingConfig {
        &self.embedding_config
    }
}

impl crate::embeddings::AsAny for Word2VecEmbeddingGenerator {
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
    fn test_word2vec_generator() {
        let config = Word2VecConfig {
            dimensions: 100,
            ..Default::default()
        };

        let embedding_config = EmbeddingConfig {
            model_name: "word2vec-test".to_string(),
            dimensions: 100,
            max_sequence_length: 512,
            normalize: true,
        };

        let mut generator = Word2VecEmbeddingGenerator::new(config, embedding_config).unwrap();

        // Add some test embeddings
        generator
            .embeddings
            .insert("hello".to_string(), vec![0.1; 100]);
        generator
            .embeddings
            .insert("world".to_string(), vec![0.2; 100]);

        // Test embedding generation
        let content = EmbeddableContent::Text("hello world".to_string());
        let embedding = generator.generate(&content).unwrap();

        assert_eq!(embedding.dimensions, 100);
    }

    #[test]
    fn test_subword_generation() {
        let config = Word2VecConfig::default();
        let generator =
            Word2VecEmbeddingGenerator::new(config, EmbeddingConfig::default()).unwrap();

        let subwords = generator.get_subwords("hello");
        assert!(subwords.contains(&"<hel>".to_string()));
        assert!(subwords.contains(&"<ell>".to_string()));
        assert!(subwords.contains(&"<llo>".to_string()));
    }

    #[test]
    fn test_aggregation_methods() {
        let mut config = Word2VecConfig {
            dimensions: 3,
            normalize: false,
            ..Default::default()
        };

        let embedding_config = EmbeddingConfig {
            model_name: "test".to_string(),
            dimensions: 3,
            max_sequence_length: 512,
            normalize: false,
        };

        // Test different aggregation methods
        for method in [
            AggregationMethod::Mean,
            AggregationMethod::Max,
            AggregationMethod::Min,
        ] {
            config.aggregation = method;
            let mut generator =
                Word2VecEmbeddingGenerator::new(config.clone(), embedding_config.clone()).unwrap();

            generator
                .embeddings
                .insert("a".to_string(), vec![1.0, 2.0, 3.0]);
            generator
                .embeddings
                .insert("b".to_string(), vec![4.0, 5.0, 6.0]);

            let content = EmbeddableContent::Text("a b".to_string());
            let embedding = generator.generate(&content).unwrap();

            match method {
                AggregationMethod::Mean => {
                    assert_eq!(embedding.as_f32(), vec![2.5, 3.5, 4.5]);
                }
                AggregationMethod::Max => {
                    assert_eq!(embedding.as_f32(), vec![4.0, 5.0, 6.0]);
                }
                AggregationMethod::Min => {
                    assert_eq!(embedding.as_f32(), vec![1.0, 2.0, 3.0]);
                }
                _ => {}
            }
        }
    }
}
