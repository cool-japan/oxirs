//! Vector search integration for semantic SPARQL queries
//!
//! This module provides advanced vector-based similarity search capabilities
//! integrated with SPARQL queries for semantic and knowledge graph applications.

use crate::{
    error::{FusekiError, FusekiResult},
    store::Store,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, instrument};

/// Vector search engine with multiple embedding models
#[derive(Clone)]
pub struct VectorSearchEngine {
    embeddings: Arc<RwLock<HashMap<String, VectorEmbedding>>>,
    models: Arc<RwLock<HashMap<String, EmbeddingModel>>>,
    indices: Arc<RwLock<HashMap<String, VectorIndex>>>,
    config: VectorSearchConfig,
}

/// Vector embedding data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEmbedding {
    pub id: String,
    pub resource_uri: String,
    pub vector: Vec<f32>,
    pub model_name: String,
    pub created_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

/// Embedding model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModel {
    pub name: String,
    pub model_type: ModelType,
    pub dimensions: usize,
    pub description: String,
    pub endpoint_url: Option<String>,
    pub api_key: Option<String>,
    pub preprocessing_steps: Vec<String>,
    pub max_input_length: usize,
}

/// Types of embedding models
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelType {
    Sentence,   // Sentence transformers
    Word,       // Word embeddings (Word2Vec, GloVe)
    Document,   // Document embeddings
    Knowledge,  // Knowledge graph embeddings
    Multimodal, // Text + image embeddings
    Custom,     // Custom model
}

/// Vector index for efficient similarity search
#[derive(Debug, Clone)]
pub struct VectorIndex {
    pub name: String,
    pub index_type: IndexType,
    pub model_name: String,
    pub embeddings_count: usize,
    pub created_at: DateTime<Utc>,
    pub index_parameters: IndexParameters,
}

/// Vector index types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexType {
    Exact, // Brute force exact search
    Hnsw,  // Hierarchical Navigable Small World
    Ivf,   // Inverted File Index
    Lsh,   // Locality Sensitive Hashing
    Annoy, // Approximate Nearest Neighbors Oh Yeah
}

/// Index configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexParameters {
    pub distance_metric: DistanceMetric,
    pub ef_construction: Option<usize>, // HNSW parameter
    pub ef_search: Option<usize>,       // HNSW parameter
    pub m: Option<usize>,               // HNSW parameter
    pub num_trees: Option<usize>,       // Annoy parameter
    pub num_probes: Option<usize>,      // IVF parameter
}

/// Distance metrics for similarity calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    Manhattan,
    Dot,
    Jaccard,
}

/// Vector search configuration
#[derive(Debug, Clone)]
pub struct VectorSearchConfig {
    pub default_model: String,
    pub max_results: usize,
    pub similarity_threshold: f32,
    pub enable_caching: bool,
    pub cache_ttl_seconds: u64,
    pub batch_size: usize,
}

/// Vector search query
#[derive(Debug, Clone, Deserialize)]
pub struct VectorSearchQuery {
    pub query_text: Option<String>,
    pub query_vector: Option<Vec<f32>>,
    pub model_name: Option<String>,
    pub top_k: Option<usize>,
    pub similarity_threshold: Option<f32>,
    pub filters: Option<HashMap<String, String>>,
    pub include_metadata: Option<bool>,
}

/// Vector search result
#[derive(Debug, Clone, Serialize)]
pub struct VectorSearchResult {
    pub results: Vec<SimilarityResult>,
    pub query_time_ms: u64,
    pub total_candidates: usize,
    pub model_used: String,
}

/// Individual similarity result
#[derive(Debug, Clone, Serialize)]
pub struct SimilarityResult {
    pub resource_uri: String,
    pub similarity_score: f32,
    pub embedding_id: String,
    pub metadata: HashMap<String, String>,
    pub vector: Option<Vec<f32>>,
}

/// Semantic SPARQL query extension
#[derive(Debug, Clone)]
pub struct SemanticQuery {
    pub sparql_query: String,
    pub vector_clauses: Vec<VectorClause>,
    pub hybrid_scoring: Option<HybridScoring>,
}

/// Vector-based clause in SPARQL query
#[derive(Debug, Clone)]
pub struct VectorClause {
    pub variable: String,
    pub search_query: VectorSearchQuery,
    pub weight: f32,
}

/// Hybrid scoring combining SPARQL and vector results
#[derive(Debug, Clone)]
pub struct HybridScoring {
    pub sparql_weight: f32,
    pub vector_weight: f32,
    pub aggregation_method: AggregationMethod,
}

/// Methods for aggregating hybrid scores
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AggregationMethod {
    WeightedSum,
    Product,
    Maximum,
    Minimum,
    RankFusion,
}

impl VectorSearchEngine {
    /// Create new vector search engine
    pub fn new(config: VectorSearchConfig) -> Self {
        VectorSearchEngine {
            embeddings: Arc::new(RwLock::new(HashMap::new())),
            models: Arc::new(RwLock::new(HashMap::new())),
            indices: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Register an embedding model
    pub async fn register_model(&self, model: EmbeddingModel) -> FusekiResult<()> {
        let mut models = self.models.write().await;

        info!(
            "Registering embedding model: {} ({})",
            model.name,
            model.model_type.to_string()
        );
        models.insert(model.name.clone(), model);

        Ok(())
    }

    /// Generate embeddings for text using specified model
    #[instrument(skip(self, texts))]
    pub async fn generate_embeddings(
        &self,
        texts: &[String],
        model_name: &str,
    ) -> FusekiResult<Vec<Vec<f32>>> {
        debug!(
            "Generating embeddings for {} texts using model: {}",
            texts.len(),
            model_name
        );

        let models = self.models.read().await;
        let model = models
            .get(model_name)
            .ok_or_else(|| FusekiError::not_found(format!("Model not found: {model_name}")))?;

        match &model.model_type {
            ModelType::Sentence => self.generate_sentence_embeddings(texts, model).await,
            ModelType::Document => self.generate_document_embeddings(texts, model).await,
            ModelType::Knowledge => self.generate_knowledge_embeddings(texts, model).await,
            _ => self.generate_generic_embeddings(texts, model).await,
        }
    }

    /// Add embedding to the vector store
    pub async fn add_embedding(
        &self,
        resource_uri: String,
        text: String,
        model_name: String,
        metadata: HashMap<String, String>,
    ) -> FusekiResult<String> {
        // Generate embedding
        let embeddings = self.generate_embeddings(&[text], &model_name).await?;
        let vector = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| FusekiError::internal("Failed to generate embedding"))?;

        // Create embedding record
        let embedding_id = uuid::Uuid::new_v4().to_string();
        let embedding = VectorEmbedding {
            id: embedding_id.clone(),
            resource_uri: resource_uri.clone(),
            vector,
            model_name,
            created_at: Utc::now(),
            metadata,
        };

        // Store embedding
        let mut embeddings_store = self.embeddings.write().await;
        embeddings_store.insert(embedding_id.clone(), embedding);

        info!("Added embedding for resource: {}", resource_uri);
        Ok(embedding_id)
    }

    /// Perform vector similarity search
    #[instrument(skip(self, query))]
    pub async fn vector_search(
        &self,
        query: VectorSearchQuery,
    ) -> FusekiResult<VectorSearchResult> {
        let start_time = std::time::Instant::now();

        // Determine model to use
        let model_name = query
            .model_name
            .as_ref()
            .unwrap_or(&self.config.default_model)
            .clone();

        // Get query vector
        let query_vector = if let Some(vector) = query.query_vector {
            vector
        } else if let Some(text) = &query.query_text {
            let embeddings = self
                .generate_embeddings(&[text.clone()], &model_name)
                .await?;
            embeddings
                .into_iter()
                .next()
                .ok_or_else(|| FusekiError::internal("Failed to generate query embedding"))?
        } else {
            return Err(FusekiError::bad_request(
                "Either query_text or query_vector must be provided",
            ));
        };

        // Perform similarity search
        let results = self
            .find_similar_vectors(
                &query_vector,
                &model_name,
                query.top_k.unwrap_or(self.config.max_results),
                query
                    .similarity_threshold
                    .unwrap_or(self.config.similarity_threshold),
                &query.filters.unwrap_or_default(),
            )
            .await?;

        let query_time = start_time.elapsed();

        let result = VectorSearchResult {
            results,
            query_time_ms: query_time.as_millis() as u64,
            total_candidates: self.embeddings.read().await.len(),
            model_used: model_name,
        };

        info!(
            "Vector search completed in {:?}, {} results",
            query_time,
            result.results.len()
        );
        Ok(result)
    }

    /// Execute semantic SPARQL query combining vector and graph search
    #[instrument(skip(self, semantic_query, store))]
    pub async fn execute_semantic_sparql(
        &self,
        semantic_query: SemanticQuery,
        store: &Store,
    ) -> FusekiResult<serde_json::Value> {
        debug!(
            "Executing semantic SPARQL query with {} vector clauses",
            semantic_query.vector_clauses.len()
        );

        // Execute vector searches for each clause
        let mut vector_results = HashMap::new();
        for clause in &semantic_query.vector_clauses {
            let search_result = self.vector_search(clause.search_query.clone()).await?;
            vector_results.insert(clause.variable.clone(), search_result);
        }

        // Modify SPARQL query to include vector results as VALUES clauses
        let enhanced_sparql = self
            .enhance_sparql_with_vector_results(&semantic_query.sparql_query, &vector_results)
            .await?;

        debug!("Enhanced SPARQL query: {}", enhanced_sparql);

        // Execute enhanced SPARQL query
        let sparql_result = store.query(&enhanced_sparql)?;

        // Combine results if hybrid scoring is enabled
        if let Some(hybrid_scoring) = &semantic_query.hybrid_scoring {
            self.apply_hybrid_scoring(sparql_result, &vector_results, hybrid_scoring)
                .await
        } else {
            // Convert SPARQL result to JSON
            sparql_result
                .to_json()
                .map(|json_str| serde_json::from_str(&json_str).unwrap_or_default())
                .map_err(|e| FusekiError::internal(format!("Failed to convert result: {e}")))
        }
    }

    /// Create vector index for faster search
    pub async fn create_index(
        &self,
        name: String,
        model_name: String,
        index_type: IndexType,
        parameters: IndexParameters,
    ) -> FusekiResult<()> {
        let index = VectorIndex {
            name: name.clone(),
            index_type,
            model_name,
            embeddings_count: self.embeddings.read().await.len(),
            created_at: Utc::now(),
            index_parameters: parameters,
        };

        let mut indices = self.indices.write().await;
        indices.insert(name.clone(), index);

        info!("Created vector index: {}", name);
        Ok(())
    }

    /// Get embedding statistics
    pub async fn get_embedding_statistics(&self) -> EmbeddingStatistics {
        let embeddings = self.embeddings.read().await;
        let models = self.models.read().await;
        let indices = self.indices.read().await;

        let mut model_counts = HashMap::new();
        for embedding in embeddings.values() {
            *model_counts
                .entry(embedding.model_name.clone())
                .or_insert(0) += 1;
        }

        EmbeddingStatistics {
            total_embeddings: embeddings.len(),
            total_models: models.len(),
            total_indices: indices.len(),
            embeddings_by_model: model_counts,
            memory_usage_mb: self.estimate_memory_usage(&embeddings).await,
        }
    }

    // Private implementation methods

    async fn generate_sentence_embeddings(
        &self,
        texts: &[String],
        model: &EmbeddingModel,
    ) -> FusekiResult<Vec<Vec<f32>>> {
        // Mock implementation for sentence embeddings
        debug!("Generating sentence embeddings using: {}", model.name);

        let mut embeddings = Vec::new();
        for text in texts {
            // Simulate sentence embedding generation
            let mut vector = vec![0.0; model.dimensions];

            // Simple hash-based mock embedding
            for (i, byte) in text.bytes().enumerate() {
                if i < model.dimensions {
                    vector[i] = (byte as f32) / 255.0;
                }
            }

            // Normalize vector
            let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut vector {
                    *x /= norm;
                }
            }

            embeddings.push(vector);
        }

        Ok(embeddings)
    }

    async fn generate_document_embeddings(
        &self,
        texts: &[String],
        model: &EmbeddingModel,
    ) -> FusekiResult<Vec<Vec<f32>>> {
        // Mock implementation for document embeddings
        debug!("Generating document embeddings using: {}", model.name);

        let mut embeddings = Vec::new();
        for text in texts {
            // Simulate document-level features
            let word_count = text.split_whitespace().count();
            let char_count = text.len();
            let avg_word_length = if word_count > 0 {
                char_count as f32 / word_count as f32
            } else {
                0.0
            };

            let mut vector = vec![0.0; model.dimensions];
            vector[0] = (word_count as f32).ln();
            vector[1] = (char_count as f32).ln();
            vector[2] = avg_word_length;

            // Fill remaining dimensions with text-based features
            for (i, byte) in text.bytes().enumerate() {
                if i + 3 < model.dimensions {
                    vector[i + 3] = (byte as f32) / 255.0;
                }
            }

            embeddings.push(vector);
        }

        Ok(embeddings)
    }

    async fn generate_knowledge_embeddings(
        &self,
        texts: &[String],
        model: &EmbeddingModel,
    ) -> FusekiResult<Vec<Vec<f32>>> {
        // Mock implementation for knowledge graph embeddings
        debug!(
            "Generating knowledge graph embeddings using: {}",
            model.name
        );

        // This would typically use entity linking and relation extraction
        self.generate_sentence_embeddings(texts, model).await
    }

    async fn generate_generic_embeddings(
        &self,
        texts: &[String],
        model: &EmbeddingModel,
    ) -> FusekiResult<Vec<Vec<f32>>> {
        // Generic fallback implementation
        self.generate_sentence_embeddings(texts, model).await
    }

    async fn find_similar_vectors(
        &self,
        query_vector: &[f32],
        model_name: &str,
        top_k: usize,
        threshold: f32,
        filters: &HashMap<String, String>,
    ) -> FusekiResult<Vec<SimilarityResult>> {
        let embeddings = self.embeddings.read().await;
        let mut candidates = Vec::new();

        // Find embeddings from the same model
        for embedding in embeddings.values() {
            if embedding.model_name != model_name {
                continue;
            }

            // Apply filters
            let mut matches_filters = true;
            for (key, value) in filters {
                if embedding.metadata.get(key) != Some(value) {
                    matches_filters = false;
                    break;
                }
            }

            if !matches_filters {
                continue;
            }

            // Calculate similarity
            let similarity = self.calculate_cosine_similarity(query_vector, &embedding.vector);

            if similarity >= threshold {
                candidates.push(SimilarityResult {
                    resource_uri: embedding.resource_uri.clone(),
                    similarity_score: similarity,
                    embedding_id: embedding.id.clone(),
                    metadata: embedding.metadata.clone(),
                    vector: None, // Don't include vector by default
                });
            }
        }

        // Sort by similarity (descending) and take top_k
        candidates.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap());
        candidates.truncate(top_k);

        Ok(candidates)
    }

    fn calculate_cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    async fn enhance_sparql_with_vector_results(
        &self,
        sparql_query: &str,
        vector_results: &HashMap<String, VectorSearchResult>,
    ) -> FusekiResult<String> {
        let mut enhanced_query = sparql_query.to_string();

        // Add VALUES clauses for each vector search result
        for (variable, results) in vector_results {
            let mut values_clause = format!("VALUES ?{variable} {{\n");

            for result in &results.results {
                values_clause.push_str(&format!("  <{}>\n", result.resource_uri));
            }

            values_clause.push_str("}\n");

            // Insert VALUES clause before the closing brace of WHERE clause
            if let Some(pos) = enhanced_query.rfind('}') {
                enhanced_query.insert_str(pos, &values_clause);
            }
        }

        Ok(enhanced_query)
    }

    async fn apply_hybrid_scoring(
        &self,
        sparql_result: crate::store::QueryResult,
        vector_results: &HashMap<String, VectorSearchResult>,
        hybrid_scoring: &HybridScoring,
    ) -> FusekiResult<serde_json::Value> {
        // This is a simplified implementation of hybrid scoring
        // In a full implementation, this would combine SPARQL and vector scores

        let sparql_json = sparql_result.to_json()?;
        let mut combined_result: serde_json::Value = serde_json::from_str(&sparql_json)?;

        // Add vector similarity scores to results
        if let Some(results_obj) = combined_result.get_mut("results") {
            if let Some(bindings) = results_obj.get_mut("bindings") {
                if let Some(bindings_array) = bindings.as_array_mut() {
                    for binding in bindings_array {
                        // Add similarity scores from vector results
                        for (variable, vector_result) in vector_results {
                            if let Some(uri_binding) = binding.get(variable) {
                                if let Some(uri_value) = uri_binding.get("value") {
                                    if let Some(uri_str) = uri_value.as_str() {
                                        // Find matching similarity score
                                        for sim_result in &vector_result.results {
                                            if sim_result.resource_uri == uri_str {
                                                let score_key = format!("{variable}_similarity");
                                                binding.as_object_mut().unwrap().insert(
                                                    score_key,
                                                    serde_json::json!({
                                                        "type": "literal",
                                                        "datatype": "http://www.w3.org/2001/XMLSchema#float",
                                                        "value": sim_result.similarity_score.to_string()
                                                    })
                                                );
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(combined_result)
    }

    async fn estimate_memory_usage(&self, embeddings: &HashMap<String, VectorEmbedding>) -> f64 {
        let mut total_bytes = 0;

        for embedding in embeddings.values() {
            total_bytes += embedding.vector.len() * std::mem::size_of::<f32>();
            total_bytes += embedding.resource_uri.len();
            total_bytes += embedding.id.len();
        }

        total_bytes as f64 / (1024.0 * 1024.0) // Convert to MB
    }
}

/// Embedding statistics
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingStatistics {
    pub total_embeddings: usize,
    pub total_models: usize,
    pub total_indices: usize,
    pub embeddings_by_model: HashMap<String, usize>,
    pub memory_usage_mb: f64,
}

impl Default for VectorSearchConfig {
    fn default() -> Self {
        VectorSearchConfig {
            default_model: "sentence-transformer".to_string(),
            max_results: 100,
            similarity_threshold: 0.7,
            enable_caching: true,
            cache_ttl_seconds: 3600,
            batch_size: 32,
        }
    }
}

impl Default for IndexParameters {
    fn default() -> Self {
        IndexParameters {
            distance_metric: DistanceMetric::Cosine,
            ef_construction: Some(200),
            ef_search: Some(50),
            m: Some(16),
            num_trees: Some(10),
            num_probes: Some(1),
        }
    }
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::Sentence => write!(f, "sentence"),
            ModelType::Word => write!(f, "word"),
            ModelType::Document => write!(f, "document"),
            ModelType::Knowledge => write!(f, "knowledge"),
            ModelType::Multimodal => write!(f, "multimodal"),
            ModelType::Custom => write!(f, "custom"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_vector_search_engine_creation() {
        let config = VectorSearchConfig::default();
        let engine = VectorSearchEngine::new(config);

        let stats = engine.get_embedding_statistics().await;
        assert_eq!(stats.total_embeddings, 0);
        assert_eq!(stats.total_models, 0);
    }

    #[tokio::test]
    async fn test_model_registration() {
        let config = VectorSearchConfig::default();
        let engine = VectorSearchEngine::new(config);

        let model = EmbeddingModel {
            name: "test-model".to_string(),
            model_type: ModelType::Sentence,
            dimensions: 384,
            description: "Test model".to_string(),
            endpoint_url: None,
            api_key: None,
            preprocessing_steps: vec![],
            max_input_length: 512,
        };

        engine.register_model(model).await.unwrap();

        let stats = engine.get_embedding_statistics().await;
        assert_eq!(stats.total_models, 1);
    }

    #[tokio::test]
    async fn test_embedding_generation() {
        let config = VectorSearchConfig::default();
        let engine = VectorSearchEngine::new(config);

        let model = EmbeddingModel {
            name: "test-model".to_string(),
            model_type: ModelType::Sentence,
            dimensions: 384,
            description: "Test model".to_string(),
            endpoint_url: None,
            api_key: None,
            preprocessing_steps: vec![],
            max_input_length: 512,
        };

        engine.register_model(model).await.unwrap();

        let texts = vec!["Hello world".to_string(), "Testing embeddings".to_string()];
        let embeddings = engine
            .generate_embeddings(&texts, "test-model")
            .await
            .unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 384);
        assert_eq!(embeddings[1].len(), 384);
    }

    #[test]
    fn test_cosine_similarity() {
        let engine = VectorSearchEngine::new(VectorSearchConfig::default());

        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![0.0, 1.0, 0.0];
        let vec3 = vec![1.0, 0.0, 0.0];

        assert_eq!(engine.calculate_cosine_similarity(&vec1, &vec2), 0.0);
        assert_eq!(engine.calculate_cosine_similarity(&vec1, &vec3), 1.0);
    }
}
