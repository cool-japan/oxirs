//! RAG (Retrieval-Augmented Generation) System for OxiRS Chat
//!
//! Implements multi-stage retrieval with semantic search, graph traversal,
//! and intelligent context assembly for knowledge graph exploration.

use anyhow::{anyhow, Result};
use oxirs_core::{
    model::{quad::Quad, term::Term, triple::Triple, NamedNode, Subject, Object},
    Store,
};
// Vector search integration (temporarily disabled)
// use oxirs_vec::{
//     embeddings::{EmbeddingModel, EmbeddingProvider},
//     index::VectorIndex,
//     similarity::SimilaritySearch,
// };
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

// Vector search integration types
pub trait EmbeddingModel: Send + Sync {
    fn encode<'a>(
        &'a self,
        texts: &'a [String],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>, anyhow::Error>> + Send + 'a>>;
}

/// Simple in-memory vector index for semantic search
pub struct VectorIndex {
    vectors: Vec<IndexedVector>,
    dimension: usize,
}

#[derive(Debug, Clone)]
struct IndexedVector {
    id: String,
    vector: Vec<f32>,
    triple: Triple,
    metadata: HashMap<String, String>,
}

impl VectorIndex {
    pub fn new(dimension: usize) -> Self {
        Self {
            vectors: Vec::new(),
            dimension,
        }
    }

    /// Add a vector to the index
    pub fn add(&mut self, id: String, vector: Vec<f32>, triple: Triple, metadata: HashMap<String, String>) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(anyhow!("Vector dimension mismatch: expected {}, got {}", self.dimension, vector.len()));
        }

        let indexed_vector = IndexedVector {
            id,
            vector,
            triple,
            metadata,
        };

        self.vectors.push(indexed_vector);
        Ok(())
    }

    /// Search for similar vectors using cosine similarity
    pub fn search(
        &self,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<SearchDocument>, anyhow::Error> {
        if query.len() != self.dimension {
            return Err(anyhow!("Query vector dimension mismatch: expected {}, got {}", self.dimension, query.len()));
        }

        let mut results: Vec<(f32, &IndexedVector)> = self
            .vectors
            .iter()
            .map(|indexed_vector| {
                let similarity = cosine_similarity(query, &indexed_vector.vector);
                (similarity, indexed_vector)
            })
            .collect();

        // Sort by similarity score (highest first)
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top results and convert to SearchDocument
        let search_results = results
            .into_iter()
            .take(limit)
            .map(|(score, indexed_vector)| SearchDocument {
                document: indexed_vector.triple.clone(),
                score,
            })
            .collect();

        Ok(search_results)
    }

    /// Get the number of vectors in the index
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

pub struct SearchDocument {
    pub document: Triple,
    pub score: f32,
}

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
            provider_type: EmbeddingProviderType::OpenAI,
            model_name: "text-embedding-ada-002".to_string(),
            api_key: std::env::var("OPENAI_API_KEY").ok(),
            base_url: None,
            cache_size: 10000,
            batch_size: 100,
            timeout_seconds: 30,
        }
    }
}

#[derive(Debug, Clone)]
pub enum EmbeddingProviderType {
    OpenAI,
    HuggingFace,
    Sentence,
    Local,
}

enum EmbeddingProvider {
    OpenAI(OpenAIEmbeddingProvider),
    HuggingFace(HuggingFaceEmbeddingProvider),
    Sentence(SentenceEmbeddingProvider),
    Local(LocalEmbeddingProvider),
}

impl EnhancedEmbeddingModel {
    pub async fn new(config: EmbeddingConfig) -> Result<Self> {
        let dimension = Self::get_model_dimension(&config.model_name);
        let provider = Self::create_provider(&config).await?;
        let cache = Arc::new(RwLock::new(HashMap::new()));

        Ok(Self {
            provider,
            dimension,
            cache,
            config,
        })
    }

    fn get_model_dimension(model_name: &str) -> usize {
        match model_name {
            "text-embedding-ada-002" => 1536,
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "sentence-transformers/all-MiniLM-L6-v2" => 384,
            "sentence-transformers/all-mpnet-base-v2" => 768,
            "sentence-transformers/paraphrase-MiniLM-L6-v2" => 384,
            _ => 768, // Default dimension
        }
    }

    async fn create_provider(config: &EmbeddingConfig) -> Result<EmbeddingProvider> {
        match config.provider_type {
            EmbeddingProviderType::OpenAI => {
                Ok(EmbeddingProvider::OpenAI(OpenAIEmbeddingProvider::new(config.clone())?))
            }
            EmbeddingProviderType::HuggingFace => {
                Ok(EmbeddingProvider::HuggingFace(HuggingFaceEmbeddingProvider::new(config.clone())?))
            }
            EmbeddingProviderType::Sentence => {
                Ok(EmbeddingProvider::Sentence(SentenceEmbeddingProvider::new(config.clone())?))
            }
            EmbeddingProviderType::Local => {
                Ok(EmbeddingProvider::Local(LocalEmbeddingProvider::new(config.clone())?))
            }
        }
    }

    async fn get_cached_embedding(&self, text: &str) -> Option<Vec<f32>> {
        let cache = self.cache.read().await;
        cache.get(text).cloned()
    }

    async fn cache_embedding(&self, text: String, embedding: Vec<f32>) {
        let mut cache = self.cache.write().await;
        
        // Simple LRU-like cache management
        if cache.len() >= self.config.cache_size {
            // Remove 10% of entries randomly when cache is full
            let keys_to_remove: Vec<String> = cache.keys()
                .take(self.config.cache_size / 10)
                .cloned()
                .collect();
            
            for key in keys_to_remove {
                cache.remove(&key);
            }
        }
        
        cache.insert(text, embedding);
    }
}

impl EmbeddingModel for EnhancedEmbeddingModel {
    fn encode<'a>(
        &'a self,
        texts: &'a [String],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>, anyhow::Error>> + Send + 'a>> {
        Box::pin(async move {
            let mut results = Vec::with_capacity(texts.len());
            let mut cache_misses = Vec::new();
            let mut cache_miss_indices = Vec::new();

            // Check cache first
            for (i, text) in texts.iter().enumerate() {
                if let Some(cached) = self.get_cached_embedding(text).await {
                    results.push(Some(cached));
                } else {
                    results.push(None);
                    cache_misses.push(text.clone());
                    cache_miss_indices.push(i);
                }
            }

            // Generate embeddings for cache misses
            if !cache_misses.is_empty() {
                let new_embeddings = match &self.provider {
                    EmbeddingProvider::OpenAI(provider) => provider.encode_batch(&cache_misses).await?,
                    EmbeddingProvider::HuggingFace(provider) => provider.encode_batch(&cache_misses).await?,
                    EmbeddingProvider::Sentence(provider) => provider.encode_batch(&cache_misses).await?,
                    EmbeddingProvider::Local(provider) => provider.encode_batch(&cache_misses).await?,
                };

                // Fill in results and cache new embeddings
                for (i, embedding) in new_embeddings.into_iter().enumerate() {
                    let result_index = cache_miss_indices[i];
                    let text = &cache_misses[i];
                    
                    results[result_index] = Some(embedding.clone());
                    self.cache_embedding(text.clone(), embedding).await;
                }
            }

            // Convert Option<Vec<f32>> to Vec<f32>
            let final_results: Vec<Vec<f32>> = results
                .into_iter()
                .map(|opt| opt.expect("All embeddings should be filled"))
                .collect();

            Ok(final_results)
        })
    }
}

/// OpenAI embedding provider
struct OpenAIEmbeddingProvider {
    client: reqwest::Client,
    api_key: String,
    model_name: String,
    base_url: String,
}

impl OpenAIEmbeddingProvider {
    fn new(config: EmbeddingConfig) -> Result<Self> {
        let api_key = config.api_key
            .ok_or_else(|| anyhow!("OpenAI API key is required"))?;
        
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()?;

        Ok(Self {
            client,
            api_key,
            model_name: config.model_name,
            base_url: config.base_url.unwrap_or_else(|| "https://api.openai.com".to_string()),
        })
    }

    async fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let request_body = serde_json::json!({
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float"
        });

        let response = self.client
            .post(&format!("{}/v1/embeddings", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        let status = response.status();
        let response_text = response.text().await?;
        
        if !status.is_success() {
            return Err(anyhow!("OpenAI API error: {} - {}", status, response_text));
        }

        let response_json: serde_json::Value = serde_json::from_str(&response_text)?;
        
        let embeddings = response_json
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| anyhow!("Invalid response format from OpenAI"))?;

        let mut results = Vec::new();
        for embedding_obj in embeddings {
            let embedding_vec = embedding_obj
                .get("embedding")
                .and_then(|e| e.as_array())
                .ok_or_else(|| anyhow!("Invalid embedding format"))?;
            
            let embedding: Vec<f32> = embedding_vec
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();
            
            results.push(embedding);
        }

        Ok(results)
    }
}

/// HuggingFace embedding provider
struct HuggingFaceEmbeddingProvider {
    client: reqwest::Client,
    api_key: Option<String>,
    model_name: String,
    base_url: String,
}

impl HuggingFaceEmbeddingProvider {
    fn new(config: EmbeddingConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()?;

        Ok(Self {
            client,
            api_key: config.api_key,
            model_name: config.model_name,
            base_url: config.base_url.unwrap_or_else(|| "https://api-inference.huggingface.co".to_string()),
        })
    }

    async fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();
        
        // HuggingFace Inference API typically processes one text at a time
        for text in texts {
            let request_body = serde_json::json!({
                "inputs": text
            });

            let mut request = self.client
                .post(&format!("{}/pipeline/feature-extraction/{}", self.base_url, self.model_name))
                .header("Content-Type", "application/json")
                .json(&request_body);

            if let Some(ref api_key) = self.api_key {
                request = request.header("Authorization", format!("Bearer {}", api_key));
            }

            let response = request.send().await?;
            let status = response.status();
            
            if !status.is_success() {
                let error_text = response.text().await?;
                return Err(anyhow!("HuggingFace API error: {} - {}", status, error_text));
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
        info!("SentenceEmbeddingProvider: Using fallback implementation for model {}", self.model_name);
        
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
        
        let word_count = words.len() as f32;
        let mut word_freqs: HashMap<&str, f32> = HashMap::new();
        
        // Calculate word frequencies
        for word in &words {
            *word_freqs.entry(word).or_insert(0.0) += 1.0;
        }
        
        // Generate embedding with frequency weighting
        for (word, freq) in word_freqs {
            let mut hasher = DefaultHasher::new();
            word.hash(&mut hasher);
            let hash = hasher.finish();
            
            // Use multiple hash functions for better distribution
            for i in 0..3 {
                let mut shifted_hasher = DefaultHasher::new();
                (hash.wrapping_add(i * 1299827)).hash(&mut shifted_hasher);
                let shifted_hash = shifted_hasher.finish();
                
                let idx = (shifted_hash as usize) % self.dimension;
                let tf_weight = freq / word_count; // Term frequency
                let idf_weight = (1.0 + word.len() as f32).ln(); // Simple IDF approximation
                
                embedding[idx] += tf_weight * idf_weight * ((hash % 1000) as f32 / 1000.0);
            }
        }

        // Add semantic features
        let sentence_length_feature = (text.len() as f32 / 1000.0).min(1.0);
        let word_count_feature = (word_count / 50.0).min(1.0);
        let avg_word_length = words.iter().map(|w| w.len()).sum::<usize>() as f32 / word_count.max(1.0) / 10.0;
        
        if self.dimension > 3 {
            embedding[0] += sentence_length_feature;
            embedding[1] += word_count_feature;
            embedding[2] += avg_word_length;
        }

        // Normalize the embedding vector
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }

        embedding
    }
}

impl EmbeddingModel for SimpleEmbeddingModel {
    fn encode<'a>(
        &'a self,
        texts: &'a [String],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>, anyhow::Error>> + Send + 'a>> {
        let embeddings: Vec<Vec<f32>> = texts
            .iter()
            .map(|text| self.text_to_embedding(text))
            .collect();

        Box::pin(async move { Ok(embeddings) })
    }
}

/// RAG system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RAGConfig {
    pub retrieval: RetrievalConfig,
    pub context: ContextConfig,
    pub ranking: RankingConfig,
    pub filtering: FilteringConfig,
}

impl Default for RAGConfig {
    fn default() -> Self {
        Self {
            retrieval: RetrievalConfig::default(),
            context: ContextConfig::default(),
            ranking: RankingConfig::default(),
            filtering: FilteringConfig::default(),
        }
    }
}

/// Retrieval stage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    pub max_results: usize,
    pub similarity_threshold: f32,
    pub use_hybrid_search: bool,
    pub bm25_weight: f32,
    pub semantic_weight: f32,
    pub graph_traversal_depth: usize,
    pub enable_entity_expansion: bool,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            max_results: 50,
            similarity_threshold: 0.7,
            use_hybrid_search: true,
            bm25_weight: 0.3,
            semantic_weight: 0.7,
            graph_traversal_depth: 2,
            enable_entity_expansion: true,
        }
    }
}

/// Context assembly configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    pub max_context_length: usize,
    pub max_triples: usize,
    pub include_schema: bool,
    pub include_examples: bool,
    pub context_window_strategy: ContextWindowStrategy,
    pub redundancy_threshold: f32,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_context_length: 4000,
            max_triples: 100,
            include_schema: true,
            include_examples: true,
            context_window_strategy: ContextWindowStrategy::Sliding,
            redundancy_threshold: 0.8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextWindowStrategy {
    Sliding,
    Important,
    Recent,
    Balanced,
}

/// Ranking configuration for relevance scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingConfig {
    pub semantic_weight: f32,
    pub graph_distance_weight: f32,
    pub frequency_weight: f32,
    pub recency_weight: f32,
    pub diversity_penalty: f32,
}

impl Default for RankingConfig {
    fn default() -> Self {
        Self {
            semantic_weight: 0.4,
            graph_distance_weight: 0.3,
            frequency_weight: 0.2,
            recency_weight: 0.1,
            diversity_penalty: 0.1,
        }
    }
}

/// Filtering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilteringConfig {
    pub min_quality_score: f32,
    pub max_age_hours: Option<usize>,
    pub allowed_predicates: Option<Vec<String>>,
    pub blocked_predicates: Option<Vec<String>>,
    pub entity_type_filters: Option<Vec<String>>,
}

impl Default for FilteringConfig {
    fn default() -> Self {
        Self {
            min_quality_score: 0.5,
            max_age_hours: None,
            allowed_predicates: None,
            blocked_predicates: None,
            entity_type_filters: None,
        }
    }
}

/// Query context for retrieval
#[derive(Debug, Clone)]
pub struct QueryContext {
    pub query: String,
    pub intent: QueryIntent,
    pub entities: Vec<ExtractedEntity>,
    pub relationships: Vec<ExtractedRelationship>,
    pub constraints: Vec<QueryConstraint>,
    pub conversation_history: Vec<String>,
}

/// Query intent classification
#[derive(Debug, Clone, PartialEq)]
pub enum QueryIntent {
    FactualLookup,
    Relationship,
    Comparison,
    Aggregation,
    Exploration,
    Definition,
    ListQuery,
    Complex,
}

/// Extracted entity from query
#[derive(Debug, Clone)]
pub struct ExtractedEntity {
    pub text: String,
    pub entity_type: Option<String>,
    pub confidence: f32,
    pub iri: Option<String>,
    pub aliases: Vec<String>,
}

/// Extracted relationship from query
#[derive(Debug, Clone)]
pub struct ExtractedRelationship {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f32,
}

/// Query constraint
#[derive(Debug, Clone)]
pub struct QueryConstraint {
    pub constraint_type: ConstraintType,
    pub value: String,
    pub operator: String,
}

#[derive(Debug, Clone, Copy)]
pub enum ConstraintType {
    Type,
    Property,
    Value,
    Temporal,
    Spatial,
}

/// Retrieved knowledge item
#[derive(Debug, Clone)]
pub struct RetrievedKnowledge {
    pub triples: Vec<Triple>,
    pub entities: Vec<EntityInfo>,
    pub schema_info: Vec<SchemaInfo>,
    pub graph_paths: Vec<GraphPath>,
    pub relevance_scores: HashMap<String, f32>,
    pub metadata: RetrievalMetadata,
}

/// Entity information
#[derive(Debug, Clone)]
pub struct EntityInfo {
    pub iri: String,
    pub label: Option<String>,
    pub description: Option<String>,
    pub entity_type: Option<String>,
    pub properties: Vec<PropertyInfo>,
    pub related_entities: Vec<String>,
}

/// Property information
#[derive(Debug, Clone)]
pub struct PropertyInfo {
    pub property: String,
    pub value: Term,
    pub confidence: f32,
}

/// Schema information
#[derive(Debug, Clone)]
pub struct SchemaInfo {
    pub class_hierarchy: Vec<String>,
    pub property_domains: Vec<String>,
    pub property_ranges: Vec<String>,
    pub constraints: Vec<String>,
}

/// Graph path information
#[derive(Debug, Clone)]
pub struct GraphPath {
    pub path: Vec<String>,
    pub path_type: PathType,
    pub strength: f32,
    pub explanation: String,
}

#[derive(Debug, Clone)]
pub enum PathType {
    Direct,
    Hierarchical,
    Related,
    Inferred,
}

/// Retrieval metadata
#[derive(Debug, Clone)]
pub struct RetrievalMetadata {
    pub retrieval_time_ms: u64,
    pub total_candidates: usize,
    pub filtered_results: usize,
    pub search_strategy: String,
    pub quality_score: f32,
}

/// Context assembly result
#[derive(Debug, Clone)]
pub struct AssembledContext {
    pub context_text: String,
    pub structured_context: StructuredContext,
    pub token_count: usize,
    pub quality_score: f32,
    pub coverage_score: f32,
}

/// Structured context for LLM
#[derive(Debug, Clone)]
pub struct StructuredContext {
    pub entities: Vec<EntityInfo>,
    pub relationships: Vec<String>,
    pub facts: Vec<String>,
    pub schema: Vec<String>,
    pub examples: Vec<String>,
}

/// Main RAG system
pub struct RAGSystem {
    config: RAGConfig,
    store: Arc<Store>,
    vector_index: Option<Arc<VectorIndex>>,
    embedding_model: Option<Box<dyn EmbeddingModel + Send + Sync>>,
    entity_extractor: EntityExtractor,
    context_assembler: ContextAssembler,
}

impl RAGSystem {
    pub fn new(
        config: RAGConfig,
        store: Arc<Store>,
        vector_index: Option<Arc<VectorIndex>>,
        embedding_model: Option<Box<dyn EmbeddingModel + Send + Sync>>,
    ) -> Self {
        Self {
            config: config.clone(),
            store,
            vector_index,
            embedding_model,
            entity_extractor: EntityExtractor::new(),
            context_assembler: ContextAssembler::new(config.context),
        }
    }

    /// Create a new RAGSystem with a vector index built from the RDF store
    pub async fn with_vector_index(
        config: RAGConfig,
        store: Arc<Store>,
        embedding_dimension: usize,
    ) -> Result<Self> {
        let embedding_model = Box::new(SimpleEmbeddingModel::new(embedding_dimension));
        let mut vector_index = VectorIndex::new(embedding_dimension);

        // Build vector index from RDF store
        Self::populate_vector_index(&mut vector_index, &*store, &*embedding_model).await?;

        let rag_system = Self {
            config: config.clone(),
            store,
            vector_index: Some(Arc::new(vector_index)),
            embedding_model: Some(embedding_model),
            entity_extractor: EntityExtractor::new(),
            context_assembler: ContextAssembler::new(config.context),
        };

        Ok(rag_system)
    }

    /// Create a new RAGSystem with enhanced embedding model and vector index
    pub async fn with_enhanced_embeddings(
        config: RAGConfig,
        store: Arc<Store>,
        embedding_config: EmbeddingConfig,
    ) -> Result<Self> {
        let enhanced_model = EnhancedEmbeddingModel::new(embedding_config).await?;
        let dimension = enhanced_model.dimension;
        let mut vector_index = VectorIndex::new(dimension);

        // Build vector index from RDF store with enhanced embeddings
        Self::populate_vector_index(&mut vector_index, &*store, &enhanced_model).await?;

        let rag_system = Self {
            config: config.clone(),
            store,
            vector_index: Some(Arc::new(vector_index)),
            embedding_model: Some(Box::new(enhanced_model)),
            entity_extractor: EntityExtractor::new(),
            context_assembler: ContextAssembler::new(config.context),
        };

        Ok(rag_system)
    }

    /// Populate a vector index with embeddings from RDF triples
    async fn populate_vector_index(
        vector_index: &mut VectorIndex,
        store: &Store,
        embedding_model: &dyn EmbeddingModel,
    ) -> Result<usize> {
        info!("Starting to populate vector index from RDF store");

        // Get all triples from the store
        let triples = store.triples().map_err(|e| anyhow!("Failed to get triples: {}", e))?;
        let mut indexed_count = 0;

        info!("Found {} triples to index", triples.len());

        // Process triples in batches for better performance
        const BATCH_SIZE: usize = 100;
        for (batch_idx, triple_batch) in triples.chunks(BATCH_SIZE).enumerate() {
            let mut texts = Vec::new();
            let mut triple_refs = Vec::new();

            // Prepare batch of texts for embedding
            for triple in triple_batch {
                let text = Self::triple_to_text(triple);
                texts.push(text);
                triple_refs.push(triple.clone());
            }

            // Generate embeddings for the batch
            match embedding_model.encode(&texts).await {
                Ok(embeddings) => {
                    // Add each embedding to the vector index
                    for (i, embedding) in embeddings.into_iter().enumerate() {
                        let triple = &triple_refs[i];
                        let id = format!("triple_{}_{}", batch_idx, i);
                        let metadata = Self::create_triple_metadata(triple);

                        if let Err(e) = vector_index.add(id, embedding, triple.clone(), metadata) {
                            warn!("Failed to add triple to vector index: {}", e);
                        } else {
                            indexed_count += 1;
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to generate embeddings for batch {}: {}", batch_idx, e);
                }
            }

            // Log progress
            if (batch_idx + 1) % 10 == 0 {
                info!("Processed {} batches, indexed {} triples", batch_idx + 1, indexed_count);
            }
        }

        info!("Vector index populated with {} embeddings", indexed_count);
        Ok(indexed_count)
    }

    /// Convert a triple to a text representation for embedding
    fn triple_to_text(triple: &Triple) -> String {
        // Create a meaningful text representation of the triple
        let subject_text = Self::term_to_text(triple.subject());
        let predicate_text = Self::term_to_text(triple.predicate());
        let object_text = Self::term_to_text(triple.object());

        format!("{} {} {}", subject_text, predicate_text, object_text)
    }

    /// Convert a term to a readable text representation
    fn term_to_text<T: std::fmt::Display>(term: &T) -> String {
        let term_str = term.to_string();
        
        // Extract meaningful parts from IRIs
        if term_str.starts_with('<') && term_str.ends_with('>') {
            let iri = &term_str[1..term_str.len()-1]; // Remove < >
            
            // Extract local name from IRI
            if let Some(fragment_pos) = iri.rfind('#') {
                return iri[fragment_pos + 1..].to_string();
            } else if let Some(slash_pos) = iri.rfind('/') {
                return iri[slash_pos + 1..].to_string();
            }
        }
        
        // For literals, remove quotes and type information
        if term_str.starts_with('"') {
            if let Some(quote_end) = term_str[1..].find('"') {
                return term_str[1..quote_end + 1].to_string();
            }
        }

        term_str
    }

    /// Create metadata for a triple
    fn create_triple_metadata(triple: &Triple) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        
        metadata.insert("subject".to_string(), triple.subject().to_string());
        metadata.insert("predicate".to_string(), triple.predicate().to_string());
        metadata.insert("object".to_string(), triple.object().to_string());
        
        // Add type information
        if triple.predicate().to_string().contains("type") {
            metadata.insert("is_type_statement".to_string(), "true".to_string());
        }
        
        metadata
    }

    /// Retrieve relevant knowledge for a query
    pub async fn retrieve_knowledge(
        &self,
        query_context: &QueryContext,
    ) -> Result<RetrievedKnowledge> {
        let start_time = std::time::Instant::now();

        info!(
            "Starting knowledge retrieval for query: {}",
            query_context.query
        );

        // Stage 1: Entity and relationship extraction
        let extracted_info = self.extract_query_components(query_context).await?;
        debug!(
            "Extracted {} entities and {} relationships",
            extracted_info.entities.len(),
            extracted_info.relationships.len()
        );

        // Stage 2: Enhanced retrieval (hybrid or semantic search)
        let search_results = if let Some(ref vector_index) = self.vector_index {
            if self.config.retrieval.use_hybrid_search {
                self.hybrid_search(&query_context.query, vector_index)
                    .await?
            } else {
                self.semantic_search(&query_context.query, vector_index)
                    .await?
            }
        } else {
            // Fallback to keyword search only if no vector index available
            self.keyword_search(&query_context.query).await?
        };

        // Stage 3: Graph traversal
        let graph_results = self.graph_traversal(&extracted_info.entities).await?;

        // Stage 4: Hybrid ranking and combination
        let combined_results =
            self.combine_and_rank_results(search_results, graph_results, &query_context.intent)?;

        // Stage 5: Context filtering and assembly
        let filtered_results = self.filter_results(combined_results)?;

        let retrieval_time = start_time.elapsed();
        let metadata = RetrievalMetadata {
            retrieval_time_ms: retrieval_time.as_millis() as u64,
            total_candidates: filtered_results.triples.len(),
            filtered_results: filtered_results.triples.len(),
            search_strategy: "hybrid".to_string(),
            quality_score: 0.8, // TODO: Calculate actual quality score
        };

        Ok(RetrievedKnowledge {
            triples: filtered_results.triples,
            entities: filtered_results.entities,
            schema_info: filtered_results.schema_info,
            graph_paths: filtered_results.graph_paths,
            relevance_scores: filtered_results.relevance_scores,
            metadata,
        })
    }

    /// Assemble context for LLM
    pub async fn assemble_context(
        &self,
        knowledge: &RetrievedKnowledge,
        query_context: &QueryContext,
    ) -> Result<AssembledContext> {
        self.context_assembler
            .assemble(knowledge, query_context)
            .await
    }


    /// Enhanced semantic search with hybrid approach
    async fn semantic_search(
        &self,
        query: &str,
        vector_index: &VectorIndex,
    ) -> Result<Vec<SearchResult>> {
        if let Some(ref embedding_model) = self.embedding_model {
            let query_embedding = embedding_model.encode(&[query.to_string()]).await?;
            let results =
                vector_index.search(&query_embedding[0], self.config.retrieval.max_results)?;

            Ok(results
                .into_iter()
                .map(|r| SearchResult {
                    triple: r.document, // Assuming document is a triple
                    score: r.score,
                    search_type: SearchType::Semantic,
                })
                .collect())
        } else {
            Ok(Vec::new())
        }
    }

    /// Hybrid search combining semantic and BM25-like text search
    async fn hybrid_search(
        &self,
        query: &str,
        vector_index: &VectorIndex,
    ) -> Result<Vec<SearchResult>> {
        // Semantic search
        let semantic_results = self.semantic_search(query, vector_index).await?;
        
        // BM25-like keyword search
        let keyword_results = self.keyword_search(query).await?;
        
        // Combine and rerank results
        let hybrid_results = self.combine_search_results(
            semantic_results,
            keyword_results,
            &self.config.retrieval,
        )?;

        Ok(hybrid_results)
    }

    /// BM25-inspired keyword search
    async fn keyword_search(&self, query: &str) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();
        let query_lower = query.to_lowercase();
        let query_terms: Vec<&str> = query_lower
            .split_whitespace()
            .filter(|term| term.len() > 2)
            .collect();

        if query_terms.is_empty() {
            return Ok(results);
        }

        // Get all triples from store
        let all_triples = self.store.triples()
            .map_err(|e| anyhow!("Failed to get triples for keyword search: {}", e))?;

        for triple in all_triples {
            let triple_text = format!("{} {} {}", 
                triple.subject(), 
                triple.predicate(), 
                triple.object()
            ).to_lowercase();

            let mut score = 0.0f32;
            let mut matched_terms = 0;

            for term in &query_terms {
                let term_frequency = triple_text.matches(term).count() as f32;
                if term_frequency > 0.0 {
                    matched_terms += 1;
                    // BM25-like scoring: TF * IDF approximation
                    let tf_component = term_frequency / (term_frequency + 1.2);
                    let idf_component = (query_terms.len() as f32).ln() + 1.0;
                    score += tf_component * idf_component;
                }
            }

            // Only include results that match at least one term
            if matched_terms > 0 {
                // Boost score for multiple term matches
                let coverage_boost = matched_terms as f32 / query_terms.len() as f32;
                score *= 1.0 + coverage_boost;
                
                results.push(SearchResult {
                    triple,
                    score,
                    search_type: SearchType::BM25,
                });
            }
        }

        // Sort by score and limit results
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(self.config.retrieval.max_results);

        Ok(results)
    }

    /// Combine semantic and keyword search results with hybrid scoring
    fn combine_search_results(
        &self,
        semantic_results: Vec<SearchResult>,
        keyword_results: Vec<SearchResult>,
        config: &RetrievalConfig,
    ) -> Result<Vec<SearchResult>> {
        let mut combined_map: HashMap<String, SearchResult> = HashMap::new();

        // Add semantic results
        for result in semantic_results {
            let key = format!("{:?}", result.triple);
            let weighted_score = result.score * config.semantic_weight;
            combined_map.insert(key, SearchResult {
                score: weighted_score,
                search_type: SearchType::Hybrid,
                ..result
            });
        }

        // Add keyword results, combining scores if triple already exists
        for result in keyword_results {
            let key = format!("{:?}", result.triple);
            let weighted_score = result.score * config.bm25_weight;
            
            if let Some(existing) = combined_map.get_mut(&key) {
                existing.score += weighted_score;
                existing.search_type = SearchType::Hybrid;
            } else {
                combined_map.insert(key, SearchResult {
                    score: weighted_score,
                    search_type: SearchType::Hybrid,
                    ..result
                });
            }
        }

        // Convert to vec and sort
        let mut final_results: Vec<SearchResult> = combined_map.into_values().collect();
        final_results.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply similarity threshold
        final_results.retain(|r| r.score >= config.similarity_threshold);
        final_results.truncate(config.max_results);

        Ok(final_results)
    }

    /// Enhanced graph traversal with multiple strategies
    async fn graph_traversal(&self, entities: &[ExtractedEntity]) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();
        let mut visited_entities = HashSet::new();

        for entity in entities {
            if let Some(ref iri) = entity.iri {
                if visited_entities.contains(iri) {
                    continue;
                }
                visited_entities.insert(iri.clone());

                // Basic entity traversal
                let entity_triples = self
                    .find_entity_triples(iri, self.config.retrieval.graph_traversal_depth)
                    .await?;
                
                for triple in entity_triples {
                    results.push(SearchResult {
                        triple,
                        score: entity.confidence * 0.8, // Slightly lower score than direct matches
                        search_type: SearchType::GraphTraversal,
                    });
                }

                // Enhanced entity expansion
                if self.config.retrieval.enable_entity_expansion {
                    let expanded_results = self.expand_entity_context(iri, entity.confidence).await?;
                    results.extend(expanded_results);
                }
            }
        }

        // Remove duplicates and apply graph-specific ranking
        self.deduplicate_and_rank_graph_results(results)
    }

    /// Expand entity context with related entities and properties
    async fn expand_entity_context(&self, entity_iri: &str, base_confidence: f32) -> Result<Vec<SearchResult>> {
        let mut expanded_results = Vec::new();

        // Find type information
        let type_triples = self.find_entity_types(entity_iri).await?;
        for triple in type_triples {
            expanded_results.push(SearchResult {
                triple,
                score: base_confidence * 0.9, // High score for type information
                search_type: SearchType::GraphTraversal,
            });
        }

        // Find same-type entities (for entity recommendation)
        let same_type_entities = self.find_same_type_entities(entity_iri, 5).await?;
        for triple in same_type_entities {
            expanded_results.push(SearchResult {
                triple,
                score: base_confidence * 0.6, // Lower score for related entities
                search_type: SearchType::GraphTraversal,
            });
        }

        // Find property domains and ranges
        let property_context = self.find_property_context(entity_iri).await?;
        for triple in property_context {
            expanded_results.push(SearchResult {
                triple,
                score: base_confidence * 0.7,
                search_type: SearchType::GraphTraversal,
            });
        }

        Ok(expanded_results)
    }

    /// Find type information for an entity
    async fn find_entity_types(&self, entity_iri: &str) -> Result<Vec<Triple>> {
        let type_predicate = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
        let mut type_triples = Vec::new();

        if let Ok(subject_triples) = self.find_triples_with_subject(entity_iri).await {
            for triple in subject_triples {
                if triple.predicate().to_string().contains(type_predicate) {
                    type_triples.push(triple);
                }
            }
        }

        Ok(type_triples)
    }

    /// Find entities of the same type
    async fn find_same_type_entities(&self, entity_iri: &str, limit: usize) -> Result<Vec<Triple>> {
        let mut same_type_triples = Vec::new();

        // First, get the types of the input entity
        let entity_types = self.find_entity_types(entity_iri).await?;
        
        if entity_types.is_empty() {
            return Ok(same_type_triples);
        }

        // For each type, find other entities of the same type
        for type_triple in entity_types.iter().take(2) { // Limit to first 2 types
            let entity_type = type_triple.object().to_string();
            
            // Find other entities with this type
            if let Ok(type_instances) = self.find_triples_with_object(&entity_type).await {
                for instance_triple in type_instances.iter().take(limit) {
                    // Skip the original entity
                    if instance_triple.subject().to_string() != entity_iri {
                        same_type_triples.push(instance_triple.clone());
                    }
                }
            }
        }

        same_type_triples.truncate(limit);
        Ok(same_type_triples)
    }

    /// Find property context (domains and ranges)
    async fn find_property_context(&self, entity_iri: &str) -> Result<Vec<Triple>> {
        let mut context_triples = Vec::new();

        // Find properties where this entity is used
        if let Ok(subject_triples) = self.find_triples_with_subject(entity_iri).await {
            for triple in subject_triples.iter().take(10) { // Limit for performance
                let property_iri = triple.predicate().to_string();
                
                // Find domain and range information for this property
                let domain_range_triples = self.find_property_domain_range(&property_iri).await?;
                context_triples.extend(domain_range_triples);
            }
        }

        context_triples.truncate(20); // Limit total context triples
        Ok(context_triples)
    }

    /// Find domain and range information for a property
    async fn find_property_domain_range(&self, property_iri: &str) -> Result<Vec<Triple>> {
        let mut domain_range_triples = Vec::new();

        // Look for rdfs:domain and rdfs:range triples
        let domain_predicate = "http://www.w3.org/2000/01/rdf-schema#domain";
        let range_predicate = "http://www.w3.org/2000/01/rdf-schema#range";

        if let Ok(property_triples) = self.find_triples_with_subject(property_iri).await {
            for triple in property_triples {
                let predicate_str = triple.predicate().to_string();
                if predicate_str.contains(domain_predicate) || predicate_str.contains(range_predicate) {
                    domain_range_triples.push(triple);
                }
            }
        }

        Ok(domain_range_triples)
    }

    /// Remove duplicates and apply graph-specific ranking
    fn deduplicate_and_rank_graph_results(&self, results: Vec<SearchResult>) -> Result<Vec<SearchResult>> {
        let mut unique_results: HashMap<String, SearchResult> = HashMap::new();

        for result in results {
            let key = format!("{:?}", result.triple);
            if let Some(existing) = unique_results.get_mut(&key) {
                // If duplicate, keep the higher score
                if result.score > existing.score {
                    existing.score = result.score;
                }
            } else {
                unique_results.insert(key, result);
            }
        }

        let mut final_results: Vec<SearchResult> = unique_results.into_values().collect();

        // Apply graph-specific ranking factors
        for result in &mut final_results {
            // Boost scores based on triple patterns
            let triple_text = format!("{} {} {}", 
                result.triple.subject(), 
                result.triple.predicate(), 
                result.triple.object()
            ).to_lowercase();

            // Boost type statements
            if triple_text.contains("type") {
                result.score *= 1.2;
            }
            
            // Boost label and name properties
            if triple_text.contains("label") || triple_text.contains("name") {
                result.score *= 1.3;
            }
            
            // Boost description properties
            if triple_text.contains("description") || triple_text.contains("comment") {
                result.score *= 1.1;
            }
        }

        // Sort by score
        final_results.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply graph traversal limits
        final_results.truncate(self.config.retrieval.max_results / 2); // Reserve half for graph results

        Ok(final_results)
    }

    async fn find_entity_triples(&self, entity_iri: &str, depth: usize) -> Result<Vec<Triple>> {
        let mut visited = HashSet::new();
        let mut result_triples = Vec::new();
        let mut queue = vec![(entity_iri.to_string(), 0)];

        while let Some((current_entity, current_depth)) = queue.pop() {
            if current_depth >= depth || visited.contains(&current_entity) {
                continue;
            }

            visited.insert(current_entity.clone());

            // Find all triples where current entity is subject
            if let Ok(subject_triples) = self.find_triples_with_subject(&current_entity).await {
                for triple in subject_triples {
                    result_triples.push(triple.clone());

                    // Add object to queue for further traversal
                    if current_depth + 1 < depth {
                        let object_str = format!("{}", triple.object());
                        if !visited.contains(&object_str) {
                            queue.push((object_str, current_depth + 1));
                        }
                    }
                }
            }

            // Find all triples where current entity is object
            if let Ok(object_triples) = self.find_triples_with_object(&current_entity).await {
                for triple in object_triples {
                    result_triples.push(triple.clone());

                    // Add subject to queue for further traversal
                    if current_depth + 1 < depth {
                        let subject_str = format!("{}", triple.subject());
                        if !visited.contains(&subject_str) {
                            queue.push((subject_str, current_depth + 1));
                        }
                    }
                }
            }
        }

        // Remove duplicates
        result_triples.sort_by(|a, b| {
            format!("{} {} {}", a.subject(), a.predicate(), a.object())
                .cmp(&format!("{} {} {}", b.subject(), b.predicate(), b.object()))
        });
        result_triples.dedup_by(|a, b| {
            format!("{} {} {}", a.subject(), a.predicate(), a.object())
                == format!("{} {} {}", b.subject(), b.predicate(), b.object())
        });

        Ok(result_triples)
    }

    async fn find_triples_with_subject(&self, subject: &str) -> Result<Vec<Triple>> {
        use oxirs_core::model::{iri::NamedNode, term::Term};
        
        let mut results = Vec::new();
        
        // Try to parse subject as IRI
        if let Ok(subject_node) = NamedNode::new(subject) {
            let subject_term = Subject::NamedNode(subject_node);
            
            // Query the store for triples with this subject
            if let Ok(quads) = self.store.query_quads(Some(&subject_term), None, None, None) {
                for quad in quads {
                    let triple = Triple::new(
                        quad.subject().clone(),
                        quad.predicate().clone(),
                        quad.object().clone()
                    );
                    results.push(triple);
                }
            }
        }
        
        Ok(results)
    }

    async fn find_triples_with_object(&self, object: &str) -> Result<Vec<Triple>> {
        use oxirs_core::model::{iri::NamedNode, term::Term};
        
        let mut results = Vec::new();
        
        // Try to parse object as IRI
        if let Ok(object_node) = NamedNode::new(object) {
            let object_term = Object::NamedNode(object_node);
            
            // Query the store for triples with this object
            if let Ok(quads) = self.store.query_quads(None, None, Some(&object_term), None) {
                for quad in quads {
                    let triple = Triple::new(
                        quad.subject().clone(),
                        quad.predicate().clone(),
                        quad.object().clone()
                    );
                    results.push(triple);
                }
            }
        }
        
        Ok(results)
    }

    fn combine_and_rank_results(
        &self,
        semantic_results: Vec<SearchResult>,
        graph_results: Vec<SearchResult>,
        intent: &QueryIntent,
    ) -> Result<Vec<SearchResult>> {
        let mut all_results = Vec::new();
        all_results.extend(semantic_results);
        all_results.extend(graph_results);

        // Remove duplicates and compute hybrid scores
        let mut unique_results: HashMap<String, SearchResult> = HashMap::new();

        for result in all_results {
            let key = format!("{:?}", result.triple); // Simple serialization as key
            if let Some(existing) = unique_results.get_mut(&key) {
                // Combine scores based on search type
                existing.score = self.combine_scores(
                    existing.score,
                    result.score,
                    &existing.search_type,
                    &result.search_type,
                );
            } else {
                unique_results.insert(key, result);
            }
        }

        let mut final_results: Vec<SearchResult> = unique_results.into_values().collect();

        // Sort by relevance score
        final_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top results
        final_results.truncate(self.config.retrieval.max_results);

        Ok(final_results)
    }

    fn combine_scores(
        &self,
        score1: f32,
        score2: f32,
        type1: &SearchType,
        type2: &SearchType,
    ) -> f32 {
        match (type1, type2) {
            (SearchType::Semantic, SearchType::GraphTraversal)
            | (SearchType::GraphTraversal, SearchType::Semantic) => {
                self.config.retrieval.semantic_weight * score1.max(score2)
                    + self.config.retrieval.bm25_weight * score1.min(score2)
            }
            _ => score1.max(score2),
        }
    }

    fn filter_results(&self, results: Vec<SearchResult>) -> Result<FilteredResults> {
        let mut filtered_triples = Vec::new();
        let mut entities = Vec::new();
        let mut schema_info = Vec::new();
        let mut graph_paths = Vec::new();
        let mut relevance_scores = HashMap::new();

        for result in results {
            if result.score >= self.config.filtering.min_quality_score {
                filtered_triples.push(result.triple.clone());
                relevance_scores.insert(format!("{:?}", result.triple), result.score);
            }
        }

        Ok(FilteredResults {
            triples: filtered_triples,
            entities,
            schema_info,
            graph_paths,
            relevance_scores,
        })
    }
}

/// Search result from different retrieval methods
#[derive(Debug, Clone)]
struct SearchResult {
    triple: Triple,
    score: f32,
    search_type: SearchType,
}

#[derive(Debug, Clone)]
enum SearchType {
    Semantic,
    GraphTraversal,
    BM25,
    Hybrid,
}

/// Extracted query information
#[derive(Debug, Clone)]
struct ExtractedQueryInfo {
    entities: Vec<ExtractedEntity>,
    relationships: Vec<ExtractedRelationship>,
    intent: QueryIntent,
}

/// Filtered results from retrieval
#[derive(Debug, Clone)]
struct FilteredResults {
    triples: Vec<Triple>,
    entities: Vec<EntityInfo>,
    schema_info: Vec<SchemaInfo>,
    graph_paths: Vec<GraphPath>,
    relevance_scores: HashMap<String, f32>,
}

/// Entity extraction component with multiple strategies
pub struct EntityExtractor {
    patterns: HashMap<String, regex::Regex>,
    entity_dict: HashMap<String, Vec<String>>,
}

impl EntityExtractor {
    pub fn new() -> Self {
        let mut patterns = HashMap::new();

        // Common entity patterns
        patterns.insert(
            "person".to_string(),
            regex::Regex::new(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b").unwrap(),
        );
        patterns.insert(
            "location".to_string(),
            regex::Regex::new(r"\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b").unwrap(),
        );
        patterns.insert(
            "organization".to_string(),
            regex::Regex::new(r"\b[A-Z][A-Za-z]*\s+(?:Inc|Corp|Ltd|Company|University)\b").unwrap(),
        );
        patterns.insert(
            "date".to_string(),
            regex::Regex::new(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b").unwrap()
        );

        // Build entity dictionary for known entities
        let mut entity_dict = HashMap::new();
        entity_dict.insert(
            "cities".to_string(),
            vec![
                "New York".to_string(),
                "London".to_string(),
                "Tokyo".to_string(),
                "Paris".to_string(),
                "Berlin".to_string(),
                "Sydney".to_string(),
            ],
        );
        entity_dict.insert(
            "countries".to_string(),
            vec![
                "United States".to_string(),
                "United Kingdom".to_string(),
                "Japan".to_string(),
                "France".to_string(),
                "Germany".to_string(),
                "Australia".to_string(),
            ],
        );

        Self {
            patterns,
            entity_dict,
        }
    }

    pub async fn extract(&self, query: &str) -> Result<ExtractedQueryInfo> {
        let mut entities = Vec::new();
        let mut relationships = Vec::new();

        // Extract entities using patterns
        for (entity_type, pattern) in &self.patterns {
            for cap in pattern.captures_iter(query) {
                if let Some(matched) = cap.get(0) {
                    entities.push(ExtractedEntity {
                        text: matched.as_str().to_string(),
                        entity_type: Some(entity_type.clone()),
                        confidence: 0.8,
                        iri: None, // TODO: Link to knowledge graph IRIs
                        aliases: Vec::new(),
                    });
                }
            }
        }

        // Extract entities from dictionary
        for (category, entity_list) in &self.entity_dict {
            for entity in entity_list {
                if query.to_lowercase().contains(&entity.to_lowercase()) {
                    entities.push(ExtractedEntity {
                        text: entity.clone(),
                        entity_type: Some(category.clone()),
                        confidence: 0.9,
                        iri: None,
                        aliases: Vec::new(),
                    });
                }
            }
        }

        // Extract relationships using pattern matching
        let relationship_patterns = vec![
            (r"(.+?)\s+(?:is|was)\s+(?:a|an|the)?\s*(.+)", "is_a"),
            (r"(.+?)\s+(?:has|have)\s+(?:a|an|the)?\s*(.+)", "has"),
            (r"(.+?)\s+(?:lives|lived)\s+in\s+(.+)", "lives_in"),
            (r"(.+?)\s+(?:works|worked)\s+(?:at|for)\s+(.+)", "works_at"),
            (r"(.+?)\s+(?:born|was born)\s+in\s+(.+)", "born_in"),
        ];

        for (pattern_str, relation_type) in relationship_patterns {
            if let Ok(pattern) = regex::Regex::new(pattern_str) {
                for cap in pattern.captures_iter(query) {
                    if let (Some(subject), Some(object)) = (cap.get(1), cap.get(2)) {
                        relationships.push(ExtractedRelationship {
                            subject: subject.as_str().trim().to_string(),
                            predicate: relation_type.to_string(),
                            object: object.as_str().trim().to_string(),
                            confidence: 0.7,
                        });
                    }
                }
            }
        }

        // Classify intent based on query patterns
        let intent = self.classify_intent(query);

        Ok(ExtractedQueryInfo {
            entities,
            relationships,
            intent,
        })
    }

    fn classify_intent(&self, query: &str) -> QueryIntent {
        let query_lower = query.to_lowercase();

        if query_lower.contains("what is")
            || query_lower.contains("who is")
            || query_lower.contains("define")
        {
            QueryIntent::FactualLookup
        } else if query_lower.contains("how")
            && (query_lower.contains("related") || query_lower.contains("connected"))
        {
            QueryIntent::Relationship
        } else if query_lower.contains("list")
            || query_lower.contains("show me all")
            || query_lower.contains("what are")
        {
            QueryIntent::ListQuery
        } else if query_lower.contains("compare")
            || query_lower.contains("difference")
            || query_lower.contains("vs")
        {
            QueryIntent::Comparison
        } else if query_lower.contains("count")
            || query_lower.contains("how many")
            || query_lower.contains("number of")
        {
            QueryIntent::Aggregation
        } else if query_lower.contains("mean") || query_lower.contains("definition") {
            QueryIntent::Definition
        } else if query.len() > 100 || query_lower.matches("and").count() > 2 {
            QueryIntent::Complex
        } else {
            QueryIntent::Exploration
        }
    }
}

/// Context assembly component
pub struct ContextAssembler {
    config: ContextConfig,
}

impl ContextAssembler {
    pub fn new(config: ContextConfig) -> Self {
        Self { config }
    }

    pub async fn assemble(
        &self,
        knowledge: &RetrievedKnowledge,
        query_context: &QueryContext,
    ) -> Result<AssembledContext> {
        // Build structured context
        let structured_context = self.build_structured_context(knowledge)?;

        // Generate context text
        let context_text = self.generate_context_text(&structured_context, query_context)?;

        // Calculate metrics
        let token_count = self.estimate_token_count(&context_text);
        let quality_score = self.calculate_quality_score(&structured_context);
        let coverage_score = self.calculate_coverage_score(&structured_context, query_context);

        Ok(AssembledContext {
            context_text,
            structured_context,
            token_count,
            quality_score,
            coverage_score,
        })
    }

    fn build_structured_context(
        &self,
        knowledge: &RetrievedKnowledge,
    ) -> Result<StructuredContext> {
        let entities = knowledge.entities.clone();

        let relationships: Vec<String> = knowledge
            .triples
            .iter()
            .map(|t| format!("{} {} {}", t.subject(), t.predicate(), t.object()))
            .collect();

        let facts: Vec<String> = knowledge
            .triples
            .iter()
            .take(self.config.max_triples)
            .map(|t| format!("{} {} {}", t.subject(), t.predicate(), t.object()))
            .collect();

        let schema: Vec<String> = if self.config.include_schema {
            knowledge
                .schema_info
                .iter()
                .flat_map(|s| s.class_hierarchy.clone())
                .collect()
        } else {
            Vec::new()
        };

        let examples: Vec<String> = if self.config.include_examples {
            // TODO: Generate examples from the knowledge
            Vec::new()
        } else {
            Vec::new()
        };

        Ok(StructuredContext {
            entities,
            relationships,
            facts,
            schema,
            examples,
        })
    }

    fn generate_context_text(
        &self,
        structured_context: &StructuredContext,
        query_context: &QueryContext,
    ) -> Result<String> {
        let mut context_parts = Vec::new();

        // Add query context
        context_parts.push(format!("Query: {}", query_context.query));

        // Add relevant entities
        if !structured_context.entities.is_empty() {
            context_parts.push("Relevant Entities:".to_string());
            for entity in &structured_context.entities {
                if let Some(ref label) = entity.label {
                    context_parts.push(format!("- {} ({})", label, entity.iri));
                } else {
                    context_parts.push(format!("- {}", entity.iri));
                }
            }
        }

        // Add facts
        if !structured_context.facts.is_empty() {
            context_parts.push("Relevant Facts:".to_string());
            for fact in structured_context.facts.iter().take(20) {
                // Limit facts
                context_parts.push(format!("- {}", fact));
            }
        }

        // Add schema information
        if !structured_context.schema.is_empty() {
            context_parts.push("Schema Information:".to_string());
            for schema_item in &structured_context.schema {
                context_parts.push(format!("- {}", schema_item));
            }
        }

        let full_context = context_parts.join("\n");

        // Truncate if too long
        if full_context.len() > self.config.max_context_length {
            Ok(full_context[..self.config.max_context_length].to_string())
        } else {
            Ok(full_context)
        }
    }

    fn estimate_token_count(&self, text: &str) -> usize {
        // Rough estimation: 1 token ≈ 4 characters
        text.len() / 4
    }

    fn calculate_quality_score(&self, structured_context: &StructuredContext) -> f32 {
        let mut quality_factors = Vec::new();
        
        // Factor 1: Entity completeness (0.0-1.0)
        let entity_completeness = if structured_context.entities.is_empty() {
            0.0
        } else {
            let entities_with_labels = structured_context.entities.iter()
                .filter(|e| e.label.is_some())
                .count() as f32;
            entities_with_labels / structured_context.entities.len() as f32
        };
        quality_factors.push(("entity_completeness", entity_completeness, 0.3));
        
        // Factor 2: Fact density (0.0-1.0)
        let fact_density = if structured_context.facts.is_empty() {
            0.0
        } else {
            // Normalize by expected fact count (assume 10 is good baseline)
            (structured_context.facts.len() as f32 / 10.0).min(1.0)
        };
        quality_factors.push(("fact_density", fact_density, 0.3));
        
        // Factor 3: Relationship richness (0.0-1.0)
        let relationship_richness = if structured_context.relationships.is_empty() {
            0.0
        } else {
            // Normalize by fact count ratio
            let rel_to_fact_ratio = structured_context.relationships.len() as f32 / 
                (structured_context.facts.len() as f32).max(1.0);
            rel_to_fact_ratio.min(1.0)
        };
        quality_factors.push(("relationship_richness", relationship_richness, 0.2));
        
        // Factor 4: Schema presence (0.0-1.0)
        let schema_presence = if structured_context.schema.is_empty() {
            0.5 // Neutral - schema not always required
        } else {
            0.8 // Bonus for having schema information
        };
        quality_factors.push(("schema_presence", schema_presence, 0.1));
        
        // Factor 5: Content diversity (0.0-1.0)
        let content_diversity = {
            let mut unique_predicates = std::collections::HashSet::new();
            for fact in &structured_context.facts {
                if let Some(predicate) = fact.split_whitespace().nth(1) {
                    unique_predicates.insert(predicate);
                }
            }
            (unique_predicates.len() as f32 / 5.0).min(1.0) // Normalize by 5 unique predicates
        };
        quality_factors.push(("content_diversity", content_diversity, 0.1));
        
        // Calculate weighted average
        let total_score: f32 = quality_factors.iter()
            .map(|(name, score, weight)| {
                debug!("Quality factor {}: {} (weight: {})", name, score, weight);
                score * weight
            })
            .sum();
        
        let final_score = total_score.max(0.0).min(1.0);
        debug!("Calculated quality score: {}", final_score);
        
        final_score
    }

    fn calculate_coverage_score(
        &self,
        structured_context: &StructuredContext,
        query_context: &QueryContext,
    ) -> f32 {
        let mut coverage_factors = Vec::new();
        
        // Factor 1: Entity coverage (0.0-1.0)
        let entity_coverage = if query_context.entities.is_empty() {
            1.0 // No entities to cover
        } else {
            let query_entity_iris: std::collections::HashSet<_> = query_context.entities.iter()
                .filter_map(|e| e.iri.as_ref())
                .collect();
            
            if query_entity_iris.is_empty() {
                0.5 // Can't determine entity coverage
            } else {
                let context_entity_iris: std::collections::HashSet<_> = structured_context.entities.iter()
                    .map(|e| &e.iri)
                    .collect();
                
                let covered_entities = query_entity_iris.iter()
                    .filter(|iri| context_entity_iris.contains(*iri))
                    .count();
                
                covered_entities as f32 / query_entity_iris.len() as f32
            }
        };
        coverage_factors.push(("entity_coverage", entity_coverage, 0.4));
        
        // Factor 2: Keyword coverage (0.0-1.0)
        let keyword_coverage = {
            let query_lowercase = query_context.query.to_lowercase();
            let query_words: std::collections::HashSet<_> = query_lowercase
                .split_whitespace()
                .filter(|w| w.len() > 3) // Only consider significant words
                .collect();
            
            if query_words.is_empty() {
                0.5
            } else {
                let context_text = structured_context.facts.join(" ").to_lowercase();
                let covered_words = query_words.iter()
                    .filter(|word| context_text.contains(*word))
                    .count();
                
                covered_words as f32 / query_words.len() as f32
            }
        };
        coverage_factors.push(("keyword_coverage", keyword_coverage, 0.3));
        
        // Factor 3: Intent-specific coverage (0.0-1.0)
        let intent_coverage = match query_context.intent {
            QueryIntent::FactualLookup => {
                // Check if we have direct facts about the queried entity
                if structured_context.facts.len() > 2 { 0.8 } else { 0.4 }
            },
            QueryIntent::Relationship => {
                // Check if we have relationship information
                if structured_context.relationships.len() > 1 { 0.9 } else { 0.3 }
            },
            QueryIntent::ListQuery => {
                // Check if we have multiple entities/facts
                if structured_context.entities.len() > 3 { 0.8 } else { 0.4 }
            },
            QueryIntent::Aggregation => {
                // Check if we have sufficient data for aggregation
                if structured_context.facts.len() > 5 { 0.7 } else { 0.3 }
            },
            _ => 0.6, // Default coverage for other intents
        };
        coverage_factors.push(("intent_coverage", intent_coverage, 0.2));
        
        // Factor 4: Completeness (0.0-1.0)
        let completeness = {
            let has_entities = !structured_context.entities.is_empty();
            let has_facts = !structured_context.facts.is_empty();
            let has_relationships = !structured_context.relationships.is_empty();
            
            match (has_entities, has_facts, has_relationships) {
                (true, true, true) => 1.0,
                (true, true, false) => 0.8,
                (true, false, true) => 0.6,
                (false, true, true) => 0.7,
                (true, false, false) => 0.4,
                (false, true, false) => 0.5,
                (false, false, true) => 0.3,
                (false, false, false) => 0.0,
            }
        };
        coverage_factors.push(("completeness", completeness, 0.1));
        
        // Calculate weighted average
        let total_score: f32 = coverage_factors.iter()
            .map(|(name, score, weight)| {
                debug!("Coverage factor {}: {} (weight: {})", name, score, weight);
                score * weight
            })
            .sum();
        
        let final_score = total_score.max(0.0).min(1.0);
        debug!("Calculated coverage score: {}", final_score);
        
        final_score
    }
}

impl RAGSystem {
    /// Extract entities and relationships from query using LLM and knowledge graph linking
    async fn extract_query_components(&self, query_context: &QueryContext) -> Result<QueryContext> {
        info!("Extracting entities and relationships from query: {}", query_context.query);
        
        // Try LLM-powered extraction first, fallback to rule-based
        let (entities, relationships) = match self.llm_extract_entities(&query_context.query).await {
            Ok((llm_entities, llm_relationships)) => {
                info!("LLM extraction successful: {} entities, {} relationships", 
                     llm_entities.len(), llm_relationships.len());
                (llm_entities, llm_relationships)
            }
            Err(e) => {
                warn!("LLM extraction failed ({}), using fallback extraction", e);
                let entities = self.rule_based_entity_extraction(&query_context.query).await?;
                let relationships = self.rule_based_relationship_extraction(&query_context.query, &entities).await?;
                (entities, relationships)
            }
        };
        
        // Enhance entities with knowledge graph linking
        let enhanced_entities = self.link_entities_to_knowledge_graph(entities).await?;
        
        // Extract constraints from query
        let constraints = self.extract_constraints(&query_context.query, &enhanced_entities).await?;
        
        Ok(QueryContext {
            query: query_context.query.clone(),
            intent: query_context.intent.clone(),
            entities: enhanced_entities,
            relationships,
            constraints,
            conversation_history: query_context.conversation_history.clone(),
        })
    }
    
    /// LLM-powered entity and relationship extraction
    async fn llm_extract_entities(&self, query: &str) -> Result<(Vec<ExtractedEntity>, Vec<ExtractedRelationship>)> {
        use crate::llm::{ChatMessage, ChatRole, LLMConfig, LLMManager, LLMRequest, Priority, UseCase};
        
        // Create extraction prompt
        let prompt = format!(
            r#"Extract entities and relationships from the following query. Return a JSON response with the following structure:

{{
  "entities": [
    {{
      "text": "entity name",
      "type": "Person|Organization|Location|Concept|Other",
      "confidence": 0.95
    }}
  ],
  "relationships": [
    {{
      "subject": "entity1",
      "predicate": "relationship type",
      "object": "entity2",
      "confidence": 0.85
    }}
  ]
}}

Query: "{}"

Focus on:
- Named entities (people, places, organizations, concepts)
- Implicit relationships between entities
- Technical terms and domain-specific concepts
- Only extract explicit entities mentioned in the query

JSON Response:"#,
            query
        );
        
        // Initialize LLM manager
        let llm_config = LLMConfig::default();
        let llm_manager = LLMManager::new(llm_config)?;
        
        let chat_messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: "You are an expert at extracting entities and relationships from text. Always respond with valid JSON only.".to_string(),
                metadata: None,
            },
            ChatMessage {
                role: ChatRole::User,
                content: prompt,
                metadata: None,
            },
        ];
        
        let request = LLMRequest {
            messages: chat_messages,
            system_prompt: Some("Extract entities and relationships as JSON.".to_string()),
            use_case: UseCase::SimpleQuery,
            priority: Priority::Normal,
            max_tokens: Some(500),
            temperature: 0.1f32, // Low temperature for consistent extraction
            timeout: Some(std::time::Duration::from_secs(15)),
        };
        
        let response = llm_manager.generate_response(request).await?;
        
        // Parse JSON response
        self.parse_extraction_response(&response.content)
    }
    
    /// Parse LLM extraction response
    fn parse_extraction_response(&self, response: &str) -> Result<(Vec<ExtractedEntity>, Vec<ExtractedRelationship>)> {
        // Clean response (remove markdown formatting if present)
        let cleaned_response = response
            .trim()
            .strip_prefix("```json")
            .unwrap_or(response)
            .strip_suffix("```")
            .unwrap_or(response)
            .trim();
        
        let parsed: serde_json::Value = serde_json::from_str(cleaned_response)?;
        
        let mut entities = Vec::new();
        let mut relationships = Vec::new();
        
        // Parse entities
        if let Some(entity_array) = parsed.get("entities").and_then(|e| e.as_array()) {
            for entity_obj in entity_array {
                if let (Some(text), Some(entity_type), Some(confidence)) = (
                    entity_obj.get("text").and_then(|t| t.as_str()),
                    entity_obj.get("type").and_then(|t| t.as_str()),
                    entity_obj.get("confidence").and_then(|c| c.as_f64()),
                ) {
                    entities.push(ExtractedEntity {
                        text: text.to_string(),
                        entity_type: Some(entity_type.to_string()),
                        confidence: confidence as f32,
                        iri: None, // Will be filled by knowledge graph linking
                        aliases: Vec::new(),
                    });
                }
            }
        }
        
        // Parse relationships
        if let Some(rel_array) = parsed.get("relationships").and_then(|r| r.as_array()) {
            for rel_obj in rel_array {
                if let (Some(subject), Some(predicate), Some(object), Some(confidence)) = (
                    rel_obj.get("subject").and_then(|s| s.as_str()),
                    rel_obj.get("predicate").and_then(|p| p.as_str()),
                    rel_obj.get("object").and_then(|o| o.as_str()),
                    rel_obj.get("confidence").and_then(|c| c.as_f64()),
                ) {
                    relationships.push(ExtractedRelationship {
                        subject: subject.to_string(),
                        predicate: predicate.to_string(),
                        object: object.to_string(),
                        confidence: confidence as f32,
                    });
                }
            }
        }
        
        Ok((entities, relationships))
    }
    
    /// Rule-based entity extraction fallback
    async fn rule_based_entity_extraction(&self, query: &str) -> Result<Vec<ExtractedEntity>> {
        let mut entities = Vec::new();
        let query_lower = query.to_lowercase();
        
        // Pattern 1: Proper nouns (capitalized words)
        let proper_noun_regex = regex::Regex::new(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")?;
        for cap in proper_noun_regex.captures_iter(query) {
            if let Some(text) = cap.get(0) {
                let entity_text = text.as_str().to_string();
                if !self.is_stop_word(&entity_text.to_lowercase()) && entity_text.len() > 2 {
                    entities.push(ExtractedEntity {
                        text: entity_text,
                        entity_type: Some("Unknown".to_string()),
                        confidence: 0.6,
                        iri: None,
                        aliases: Vec::new(),
                    });
                }
            }
        }
        
        // Pattern 2: Technical terms and domain concepts
        let technical_patterns = [
            (r"\b(?:class|property|relationship|entity|triple|graph|ontology|schema)\b", "Concept"),
            (r"\b(?:person|people|individual|user|author|creator)\b", "Person"),
            (r"\b(?:organization|company|institution|university|group)\b", "Organization"),
            (r"\b(?:location|place|city|country|address|region)\b", "Location"),
            (r"\b(?:time|date|year|month|day|period|duration)\b", "Temporal"),
        ];
        
        for (pattern, entity_type) in technical_patterns {
            let regex = regex::Regex::new(pattern)?;
            for cap in regex.captures_iter(&query_lower) {
                if let Some(text) = cap.get(0) {
                    entities.push(ExtractedEntity {
                        text: text.as_str().to_string(),
                        entity_type: Some(entity_type.to_string()),
                        confidence: 0.7,
                        iri: None,
                        aliases: Vec::new(),
                    });
                }
            }
        }
        
        // Pattern 3: Quoted strings (explicit mentions)
        let quoted_regex = regex::Regex::new(r#""([^"]+)""#)?;
        for cap in quoted_regex.captures_iter(query) {
            if let Some(text) = cap.get(1) {
                entities.push(ExtractedEntity {
                    text: text.as_str().to_string(),
                    entity_type: Some("Literal".to_string()),
                    confidence: 0.8,
                    iri: None,
                    aliases: Vec::new(),
                });
            }
        }
        
        // Remove duplicates and low-confidence entities
        entities.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        entities.dedup_by(|a, b| a.text.to_lowercase() == b.text.to_lowercase());
        entities.retain(|e| e.confidence > 0.5);
        
        debug!("Rule-based extraction found {} entities", entities.len());
        Ok(entities)
    }
    
    /// Rule-based relationship extraction
    async fn rule_based_relationship_extraction(
        &self, 
        query: &str, 
        entities: &[ExtractedEntity]
    ) -> Result<Vec<ExtractedRelationship>> {
        let mut relationships = Vec::new();
        let query_lower = query.to_lowercase();
        
        // Pattern 1: Direct relationship indicators
        let relationship_patterns = [
            (r"(\w+)\s+(?:is|are)\s+(?:a|an|the)?\s*(\w+)", "type"),
            (r"(\w+)\s+(?:has|have|owns|contains)\s+(\w+)", "has"),
            (r"(\w+)\s+(?:works for|employed by|part of)\s+(\w+)", "worksFor"),
            (r"(\w+)\s+(?:knows|related to|connected to)\s+(\w+)", "relatedTo"),
            (r"(\w+)\s+(?:created|authored|made)\s+(\w+)", "created"),
            (r"(\w+)\s+(?:located in|from|based in)\s+(\w+)", "locatedIn"),
        ];
        
        for (pattern, relation_type) in relationship_patterns {
            let regex = regex::Regex::new(pattern)?;
            for cap in regex.captures_iter(&query_lower) {
                if let (Some(subject), Some(object)) = (cap.get(1), cap.get(2)) {
                    relationships.push(ExtractedRelationship {
                        subject: subject.as_str().to_string(),
                        predicate: relation_type.to_string(),
                        object: object.as_str().to_string(),
                        confidence: 0.7,
                    });
                }
            }
        }
        
        // Pattern 2: Inferred relationships from entity proximity
        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                let entity1 = &entities[i];
                let entity2 = &entities[j];
                
                // Check if entities appear close together in query
                let text1_pos = query_lower.find(&entity1.text.to_lowercase());
                let text2_pos = query_lower.find(&entity2.text.to_lowercase());
                
                if let (Some(pos1), Some(pos2)) = (text1_pos, text2_pos) {
                    let distance = (pos1 as i32 - pos2 as i32).abs();
                    if distance < 50 { // Within 50 characters
                        relationships.push(ExtractedRelationship {
                            subject: entity1.text.clone(),
                            predicate: "relatedTo".to_string(),
                            object: entity2.text.clone(),
                            confidence: 0.5,
                        });
                    }
                }
            }
        }
        
        // Remove duplicates
        relationships.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        relationships.dedup_by(|a, b| {
            a.subject == b.subject && a.predicate == b.predicate && a.object == b.object
        });
        
        debug!("Rule-based extraction found {} relationships", relationships.len());
        Ok(relationships)
    }
    
    /// Link extracted entities to knowledge graph IRIs
    async fn link_entities_to_knowledge_graph(&self, entities: Vec<ExtractedEntity>) -> Result<Vec<ExtractedEntity>> {
        let mut enhanced_entities = Vec::new();
        
        for mut entity in entities {
            // Search for matching entities in the knowledge graph
            let matching_iris = self.find_matching_iris(&entity.text).await?;
            
            if !matching_iris.is_empty() {
                // Select best matching IRI based on confidence
                entity.iri = Some(matching_iris[0].clone());
                entity.confidence = (entity.confidence * 1.2).min(1.0); // Boost confidence for linked entities
                
                // Add aliases from knowledge graph
                entity.aliases = self.get_entity_aliases(&entity.iri.as_ref().unwrap()).await?;
            }
            
            enhanced_entities.push(entity);
        }
        
        debug!("Linked {} entities to knowledge graph IRIs", 
               enhanced_entities.iter().filter(|e| e.iri.is_some()).count());
        
        Ok(enhanced_entities)
    }
    
    /// Find matching IRIs in knowledge graph for entity text
    async fn find_matching_iris(&self, entity_text: &str) -> Result<Vec<String>> {
        let mut matching_iris = Vec::new();
        
        // Search through available triples for label matches
        // This is a simplified implementation - in production you'd want more sophisticated matching
        let search_patterns = [
            entity_text,
            &entity_text.to_lowercase(),
            &format!("\"{}\"", entity_text),
        ];
        
        for pattern in search_patterns {
            // Search in literals (labels, names, etc.)
            if let Ok(quads) = self.store.query_quads(None, None, None, None) {
                for quad in quads.into_iter().take(1000) { // Limit search for performance
                    if let oxirs_core::model::term::Object::Literal(literal) = quad.object() {
                        if literal.value().to_lowercase().contains(&pattern.to_lowercase()) {
                            matching_iris.push(quad.subject().to_string());
                            break; // Found a match, stop searching
                        }
                    }
                }
            }
            
            if !matching_iris.is_empty() {
                break; // Found matches with this pattern
            }
        }
        
        Ok(matching_iris)
    }
    
    /// Get aliases for an entity IRI
    async fn get_entity_aliases(&self, iri: &str) -> Result<Vec<String>> {
        let mut aliases = Vec::new();
        
        // Look for rdfs:label, foaf:name, skos:prefLabel, etc.
        let label_properties = [
            "http://www.w3.org/2000/01/rdf-schema#label",
            "http://xmlns.com/foaf/0.1/name",
            "http://www.w3.org/2004/02/skos/core#prefLabel",
            "http://purl.org/dc/terms/title",
        ];
        
        for prop in label_properties {
            if let Ok(property_node) = oxirs_core::model::iri::NamedNode::new(prop) {
                if let Ok(iri_node) = oxirs_core::model::iri::NamedNode::new(iri) {
                    let subject = oxirs_core::model::term::Subject::NamedNode(iri_node);
                    let predicate = oxirs_core::model::term::Predicate::NamedNode(property_node);
                    
                    if let Ok(quads) = self.store.query_quads(Some(&subject), Some(&predicate), None, None) {
                        for quad in quads.into_iter().take(5) { // Limit aliases
                            if let oxirs_core::model::term::Object::Literal(literal) = quad.object() {
                                aliases.push(literal.value().to_string());
                            }
                        }
                    }
                }
            }
        }
        
        Ok(aliases)
    }
    
    /// Extract query constraints
    async fn extract_constraints(&self, query: &str, entities: &[ExtractedEntity]) -> Result<Vec<QueryConstraint>> {
        let mut constraints = Vec::new();
        let query_lower = query.to_lowercase();
        
        // Temporal constraints
        let temporal_patterns = [
            (r"(?:in|during|from|since|before|after)\s+(\d{4})", ConstraintType::Temporal, "year"),
            (r"(?:today|yesterday|tomorrow|now|recent)", ConstraintType::Temporal, "relative_time"),
        ];
        
        for (pattern, constraint_type, operator) in temporal_patterns {
            let regex = regex::Regex::new(pattern)?;
            for cap in regex.captures_iter(&query_lower) {
                if let Some(value) = cap.get(1) {
                    constraints.push(QueryConstraint {
                        constraint_type,
                        value: value.as_str().to_string(),
                        operator: operator.to_string(),
                    });
                } else if cap.get(0).is_some() {
                    constraints.push(QueryConstraint {
                        constraint_type,
                        value: cap.get(0).unwrap().as_str().to_string(),
                        operator: operator.to_string(),
                    });
                }
            }
        }
        
        // Type constraints
        if query_lower.contains("type") || query_lower.contains("kind") || query_lower.contains("class") {
            constraints.push(QueryConstraint {
                constraint_type: ConstraintType::Type,
                value: "type_constraint".to_string(),
                operator: "equals".to_string(),
            });
        }
        
        // Value constraints (numeric, comparison)
        let value_patterns = [
            (r"(?:greater than|more than|>\s*)(\d+)", "greater_than"),
            (r"(?:less than|fewer than|<\s*)(\d+)", "less_than"),
            (r"(?:equals?|is|=\s*)(\d+)", "equals"),
        ];
        
        for (pattern, operator) in value_patterns {
            let regex = regex::Regex::new(pattern)?;
            for cap in regex.captures_iter(&query_lower) {
                if let Some(value) = cap.get(1) {
                    constraints.push(QueryConstraint {
                        constraint_type: ConstraintType::Value,
                        value: value.as_str().to_string(),
                        operator: operator.to_string(),
                    });
                }
            }
        }
        
        debug!("Extracted {} constraints from query", constraints.len());
        Ok(constraints)
    }
    
    /// Check if a word is a stop word (for entity extraction)
    fn is_stop_word(&self, word: &str) -> bool {
        matches!(word, "the" | "and" | "or" | "but" | "in" | "on" | "at" | "to" | "for" | 
                       "of" | "with" | "by" | "from" | "up" | "about" | "into" | "through" | 
                       "during" | "before" | "after" | "above" | "below" | "between" | "among" |
                       "this" | "that" | "these" | "those" | "i" | "you" | "he" | "she" | "it" |
                       "we" | "they" | "me" | "him" | "her" | "us" | "them" | "my" | "your" |
                       "his" | "its" | "our" | "their" | "am" | "is" | "are" | "was" | "were" |
                       "be" | "been" | "being" | "have" | "has" | "had" | "do" | "does" | "did" |
                       "will" | "would" | "could" | "should" | "may" | "might" | "must" | "can" |
                       "what" | "when" | "where" | "who" | "why" | "how" | "which")
    }
}
