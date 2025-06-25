//! Embedding generation and management for RDF resources and text content

use crate::Vector;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

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
    /// Custom embedding model
    Custom(String),
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

/// Dummy sentence transformer embedding generator (placeholder for actual implementation)
pub struct SentenceTransformerGenerator {
    config: EmbeddingConfig,
}

impl SentenceTransformerGenerator {
    pub fn new(config: EmbeddingConfig) -> Self {
        Self { config }
    }
}

impl EmbeddingGenerator for SentenceTransformerGenerator {
    fn generate(&self, content: &EmbeddableContent) -> Result<Vector> {
        let _text = content.to_text();
        let text_hash = content.content_hash();

        // Generate deterministic "embeddings" based on text hash for now
        // In a real implementation, this would call an actual sentence transformer model
        let mut values = Vec::with_capacity(self.config.dimensions);
        let mut seed = text_hash;

        for _ in 0..self.config.dimensions {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let normalized = (seed as f32) / (u64::MAX as f32);
            values.push((normalized - 0.5) * 2.0); // Range: -1.0 to 1.0
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
