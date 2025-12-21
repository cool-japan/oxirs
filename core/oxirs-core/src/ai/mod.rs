//! AI/ML Integration Platform for OxiRS
//!
//! This module provides comprehensive AI and machine learning capabilities for RDF graphs,
//! including Graph Neural Networks, knowledge graph embeddings, entity resolution,
//! and automated reasoning.

pub mod embeddings;
pub mod entity_resolution;
pub mod gnn;
pub mod gpu_monitor;
pub mod neural;
pub mod relation_extraction;
pub mod temporal_reasoning;
pub mod training;
pub mod vector_store;

use crate::model::Triple;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

pub use embeddings::{
    create_embedding_model, ComplEx, DistMult, EmbeddingConfig, EmbeddingModelType,
    KnowledgeGraphEmbedding, TransE,
};
pub use gnn::{
    Aggregation, GnnArchitecture, GnnConfig, GraphNeuralNetwork, LayerType, MessagePassingType,
};
pub use training::{
    DefaultTrainer, LossFunction, Optimizer, Trainer, TrainingConfig, TrainingMetrics,
};
pub use vector_store::{SimilarityMetric, VectorIndex, VectorQuery, VectorStore};

/// AI configuration for the OxiRS platform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiConfig {
    /// Enable Graph Neural Networks
    pub enable_gnn: bool,

    /// Knowledge graph embedding configuration
    pub embedding_config: EmbeddingConfig,

    /// Vector store configuration
    pub vector_store_config: VectorStoreConfig,

    /// Training configuration
    pub training_config: TrainingConfig,

    /// GPU acceleration settings
    pub gpu_config: GpuConfig,

    /// Model cache settings
    pub cache_config: CacheConfig,
}

impl Default for AiConfig {
    fn default() -> Self {
        Self {
            enable_gnn: true,
            embedding_config: EmbeddingConfig::default(),
            vector_store_config: VectorStoreConfig::default(),
            training_config: TrainingConfig::default(),
            gpu_config: GpuConfig::default(),
            cache_config: CacheConfig::default(),
        }
    }
}

/// Vector store configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreConfig {
    /// Vector dimension
    pub dimension: usize,

    /// Distance metric for similarity search
    pub metric: SimilarityMetric,

    /// Index type for nearest neighbor search
    pub index_type: IndexType,

    /// Maximum number of vectors in memory
    pub max_vectors: usize,

    /// Enable approximate nearest neighbor search
    pub enable_ann: bool,

    /// Number of neighbors for ANN
    pub ann_neighbors: usize,
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            metric: SimilarityMetric::Cosine,
            index_type: IndexType::HierarchicalNavigableSmallWorld,
            max_vectors: 10_000_000,
            enable_ann: true,
            ann_neighbors: 16,
        }
    }
}

/// Index types for vector search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    /// Flat index (exact search)
    Flat,
    /// IVF (Inverted File) index
    InvertedFile { clusters: usize },
    /// LSH (Locality-Sensitive Hashing)
    LocalitySensitiveHashing {
        hash_tables: usize,
        hash_length: usize,
    },
    /// HNSW (Hierarchical Navigable Small World)
    HierarchicalNavigableSmallWorld,
    /// Product Quantization
    ProductQuantization { subquantizers: usize, bits: usize },
}

/// GPU acceleration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Enable GPU acceleration
    pub enabled: bool,

    /// GPU device ID
    pub device_id: u32,

    /// Memory pool size in MB
    pub memory_pool_mb: usize,

    /// Batch size for GPU operations
    pub batch_size: usize,

    /// Enable mixed precision training
    pub mixed_precision: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            device_id: 0,
            memory_pool_mb: 4096,
            batch_size: 1024,
            mixed_precision: true,
        }
    }
}

/// Model cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable model caching
    pub enabled: bool,

    /// Cache directory path
    pub cache_dir: String,

    /// Maximum cache size in MB
    pub max_size_mb: usize,

    /// Cache TTL in seconds
    pub ttl_seconds: u64,

    /// Enable compression for cached models
    pub compression: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cache_dir: "/tmp/oxirs/ai_cache".to_string(),
            max_size_mb: 10240, // 10GB
            ttl_seconds: 86400, // 24 hours
            compression: true,
        }
    }
}

/// AI-powered RDF processing engine
pub struct AiEngine {
    /// Configuration
    #[allow(dead_code)]
    config: AiConfig,

    /// Graph Neural Network
    gnn: Option<Arc<dyn GraphNeuralNetwork>>,

    /// Knowledge graph embeddings
    embeddings: HashMap<String, Arc<dyn KnowledgeGraphEmbedding>>,

    /// Vector store for similarity search
    vector_store: Arc<dyn VectorStore>,

    /// Training engine
    trainer: Arc<Mutex<Box<dyn Trainer>>>,

    /// Entity resolution module
    entity_resolver: Arc<entity_resolution::EntityResolver>,

    /// Relation extraction module
    relation_extractor: Arc<relation_extraction::RelationExtractor>,

    /// Temporal reasoning module
    temporal_reasoner: Arc<temporal_reasoning::TemporalReasoner>,
}

impl AiEngine {
    /// Create a new AI engine
    pub fn new(config: AiConfig) -> Result<Self> {
        let vs_config = vector_store::VectorStoreConfig {
            dimension: config.vector_store_config.dimension,
            default_metric: config.vector_store_config.metric,
            index_type: match config.vector_store_config.index_type {
                IndexType::Flat => vector_store::IndexType::Flat,
                IndexType::HierarchicalNavigableSmallWorld => vector_store::IndexType::HNSW {
                    max_connections: 16,
                    ef_construction: 200,
                    ef_search: 50,
                },
                IndexType::InvertedFile { clusters } => vector_store::IndexType::IVF {
                    num_clusters: clusters,
                    num_probes: 8,
                },
                IndexType::LocalitySensitiveHashing {
                    hash_tables,
                    hash_length,
                } => vector_store::IndexType::LSH {
                    num_tables: hash_tables,
                    hash_length,
                },
                IndexType::ProductQuantization {
                    subquantizers,
                    bits,
                } => vector_store::IndexType::PQ {
                    num_subquantizers: subquantizers,
                    bits_per_subquantizer: bits,
                },
            },
            enable_cache: config.vector_store_config.enable_ann,
            cache_size: if config.vector_store_config.max_vectors > 10000 {
                10000
            } else {
                config.vector_store_config.max_vectors
            },
            cache_ttl: 3600,
            batch_size: 1000,
        };
        let vector_store = vector_store::create_vector_store(&vs_config)?;
        // Use tokio::sync::Mutex for async-aware locking
        let trainer = Arc::new(Mutex::new(Box::new(training::DefaultTrainer::new(
            config.training_config.clone(),
        )) as Box<dyn Trainer>));
        let entity_resolver = Arc::new(entity_resolution::EntityResolver::new(&config)?);
        let relation_extractor = Arc::new(relation_extraction::RelationExtractor::new(&config)?);
        let temporal_reasoner = Arc::new(temporal_reasoning::TemporalReasoner::new(&config)?);

        Ok(Self {
            config,
            gnn: None,
            embeddings: HashMap::new(),
            vector_store,
            trainer,
            entity_resolver,
            relation_extractor,
            temporal_reasoner,
        })
    }

    /// Initialize Graph Neural Network
    pub async fn initialize_gnn(&mut self, gnn_config: GnnConfig) -> Result<()> {
        let gnn = gnn::create_gnn(gnn_config)?;
        self.gnn = Some(gnn);
        Ok(())
    }

    /// Add knowledge graph embedding model
    pub async fn add_embedding_model(
        &mut self,
        name: String,
        model: Arc<dyn KnowledgeGraphEmbedding>,
    ) -> Result<()> {
        self.embeddings.insert(name, model);
        Ok(())
    }

    /// Generate embeddings for RDF graph
    pub async fn generate_embeddings(
        &self,
        model_name: &str,
        triples: &[Triple],
    ) -> Result<Vec<Vec<f32>>> {
        let model = self
            .embeddings
            .get(model_name)
            .ok_or_else(|| anyhow!("Embedding model not found: {}", model_name))?;

        model.generate_embeddings(triples).await
    }

    /// Find similar entities using vector similarity
    pub async fn find_similar_entities(
        &self,
        entity_vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<(String, f32)>> {
        let query = VectorQuery {
            vector: entity_vector.to_vec(),
            k: top_k,
            include_metadata: true,
            metric: None,
            filters: None,
            min_similarity: None,
        };

        self.vector_store.search(&query).await
    }

    /// Predict missing links in knowledge graph
    pub async fn predict_links(
        &self,
        model_name: &str,
        entities: &[String],
        relations: &[String],
    ) -> Result<Vec<(String, String, String, f32)>> {
        let model = self
            .embeddings
            .get(model_name)
            .ok_or_else(|| anyhow!("Embedding model not found: {}", model_name))?;

        model.predict_links(entities, relations).await
    }

    /// Resolve entity identity across different sources
    pub async fn resolve_entities(
        &self,
        entities: &[Triple],
    ) -> Result<Vec<entity_resolution::EntityCluster>> {
        self.entity_resolver.resolve_entities(entities).await
    }

    /// Extract relations from text using NLP
    pub async fn extract_relations_from_text(
        &self,
        text: &str,
    ) -> Result<Vec<relation_extraction::ExtractedRelation>> {
        self.relation_extractor.extract_relations(text).await
    }

    /// Perform temporal reasoning on knowledge graph
    pub async fn temporal_reasoning(
        &self,
        query: &temporal_reasoning::TemporalQuery,
    ) -> Result<temporal_reasoning::TemporalResult> {
        self.temporal_reasoner.reason(query).await
    }

    /// Train embedding model on knowledge graph
    pub async fn train_embedding_model(
        &self,
        model_name: &str,
        training_data: &[Triple],
        validation_data: &[Triple],
    ) -> Result<TrainingMetrics> {
        let model = self
            .embeddings
            .get(model_name)
            .ok_or_else(|| anyhow!("Embedding model not found: {}", model_name))?;

        // Clone references for async operation
        let trainer = self.trainer.clone();
        let model = model.clone();
        let training_data = training_data.to_vec();
        let validation_data = validation_data.to_vec();

        // Use async-aware mutex (tokio::sync::Mutex) - safe to hold across await
        let mut trainer_guard = trainer.lock().await;
        trainer_guard
            .train_embedding_model(model, &training_data, &validation_data)
            .await
    }

    /// Evaluate model performance
    pub async fn evaluate_model(
        &self,
        model_name: &str,
        test_data: &[Triple],
    ) -> Result<EvaluationMetrics> {
        let model = self
            .embeddings
            .get(model_name)
            .ok_or_else(|| anyhow!("Embedding model not found: {}", model_name))?;

        EvaluationMetrics::evaluate(model.as_ref(), test_data).await
    }

    /// Get AI engine statistics
    pub async fn get_statistics(&self) -> Result<AiStatistics> {
        // Get vector store statistics for cache hit rate
        let vs_stats = self.vector_store.get_statistics().await?;

        // Get GPU utilization from global GPU monitor
        let gpu_monitor = gpu_monitor::GpuMonitor::global();
        let gpu_utilization = gpu_monitor
            .lock()
            .map(|monitor| monitor.get_utilization())
            .unwrap_or(0.0);

        Ok(AiStatistics {
            gnn_enabled: self.gnn.is_some(),
            embedding_models: self.embeddings.len(),
            vector_store_size: self.vector_store.size(),
            cache_hit_rate: vs_stats.cache_hit_rate,
            gpu_utilization,
        })
    }
}

/// Evaluation metrics for AI models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    /// Mean Reciprocal Rank
    pub mrr: f32,

    /// Hits at K (K=1,3,10)
    pub hits_at_1: f32,
    pub hits_at_3: f32,
    pub hits_at_10: f32,

    /// Link prediction accuracy
    pub link_prediction_accuracy: f32,

    /// Entity resolution F1 score
    pub entity_resolution_f1: f32,

    /// Relation extraction precision/recall
    pub relation_extraction_precision: f32,
    pub relation_extraction_recall: f32,
}

impl EvaluationMetrics {
    /// Evaluate model performance on test data
    pub async fn evaluate(
        model: &dyn KnowledgeGraphEmbedding,
        test_data: &[Triple],
    ) -> Result<Self> {
        // Convert test data to string tuples for evaluation
        let test_triples: Vec<(String, String, String)> = test_data
            .iter()
            .map(|t| {
                (
                    t.subject().to_string(),
                    t.predicate().to_string(),
                    t.object().to_string(),
                )
            })
            .collect();

        // Use test_triples as all_triples for filtered setting (simplified)
        // In production, this should include training triples too
        let all_triples = test_triples.clone();

        // Define k values for Hits@K metrics
        let k_values = vec![1, 3, 10];

        // Compute comprehensive knowledge graph metrics using the embeddings evaluation module
        let kg_metrics = embeddings::evaluation::compute_kg_metrics(
            model,
            &test_triples,
            &all_triples,
            &k_values,
        )
        .await?;

        // Compute link prediction accuracy (simplified)
        let link_prediction_accuracy =
            Self::compute_link_prediction_accuracy(model, &test_triples).await?;

        // Extract key metrics from kg_metrics
        let mrr = kg_metrics.mrr_filtered;
        let hits_at_1 = *kg_metrics.hits_at_k_filtered.get(&1).unwrap_or(&0.0);
        let hits_at_3 = *kg_metrics.hits_at_k_filtered.get(&3).unwrap_or(&0.0);
        let hits_at_10 = *kg_metrics.hits_at_k_filtered.get(&10).unwrap_or(&0.0);

        // Entity resolution and relation extraction metrics would require additional data
        // For now, set them to 0.0 (these are specialized tasks beyond standard link prediction)
        let entity_resolution_f1 = 0.0;
        let relation_extraction_precision = 0.0;
        let relation_extraction_recall = 0.0;

        Ok(Self {
            mrr,
            hits_at_1,
            hits_at_3,
            hits_at_10,
            link_prediction_accuracy,
            entity_resolution_f1,
            relation_extraction_precision,
            relation_extraction_recall,
        })
    }

    /// Compute link prediction accuracy using negative sampling
    async fn compute_link_prediction_accuracy(
        model: &dyn KnowledgeGraphEmbedding,
        test_triples: &[(String, String, String)],
    ) -> Result<f32> {
        if test_triples.is_empty() {
            return Ok(0.0);
        }

        // Sample up to 100 triples for efficiency
        let sample_size = test_triples.len().min(100);
        let mut correct = 0;

        // Collect all entities for negative sampling
        let entities: std::collections::HashSet<String> = test_triples
            .iter()
            .flat_map(|(h, _, t)| vec![h.clone(), t.clone()])
            .collect();
        let entity_vec: Vec<String> = entities.into_iter().collect();

        if entity_vec.len() < 2 {
            return Ok(0.0);
        }

        for triple in test_triples.iter().take(sample_size) {
            let positive_score = model.score_triple(&triple.0, &triple.1, &triple.2).await?;

            // Generate a random negative sample by corrupting head or tail
            let corrupt_idx = {
                use scirs2_core::random::Random;
                let mut rng = Random::default();
                rng.random_range(0, entity_vec.len())
            };
            let corrupt_entity = &entity_vec[corrupt_idx];

            let negative_score = {
                use scirs2_core::random::Random;
                let mut rng = Random::default();
                if rng.random_bool_with_chance(0.5) {
                    // Corrupt head
                    model
                        .score_triple(corrupt_entity, &triple.1, &triple.2)
                        .await?
                } else {
                    // Corrupt tail
                    model
                        .score_triple(&triple.0, &triple.1, corrupt_entity)
                        .await?
                }
            };

            // For most models, positive triples should have better scores than negatives
            // TransE uses distance (lower is better), DistMult/ComplEx use similarity (higher is better)
            // We'll use a simple heuristic: if scores are significantly different, count as correct
            if (positive_score - negative_score).abs() > 0.01 {
                correct += 1;
            }
        }

        Ok(correct as f32 / sample_size as f32)
    }
}

/// AI engine statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiStatistics {
    /// Whether GNN is enabled
    pub gnn_enabled: bool,

    /// Number of embedding models loaded
    pub embedding_models: usize,

    /// Vector store size
    pub vector_store_size: usize,

    /// Cache hit rate
    pub cache_hit_rate: f32,

    /// GPU utilization percentage
    pub gpu_utilization: f32,
}

/// AI-powered query enhancement
pub trait AiQueryEnhancement {
    /// Enhance SPARQL query with AI insights
    fn enhance_query(&self, query: &str) -> Result<String>;

    /// Suggest related entities
    fn suggest_entities(&self, entity: &str) -> Result<Vec<String>>;

    /// Expand query with related concepts
    fn expand_query(&self, query: &str) -> Result<Vec<String>>;
}

/// AI-powered data validation
pub trait AiDataValidation {
    /// Detect anomalies in RDF data
    fn detect_anomalies(&self, triples: &[Triple]) -> Result<Vec<Anomaly>>;

    /// Suggest data quality improvements
    fn suggest_improvements(&self, triples: &[Triple]) -> Result<Vec<Improvement>>;

    /// Validate data consistency
    fn validate_consistency(&self, triples: &[Triple]) -> Result<Vec<InconsistencyError>>;
}

/// Data anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    /// Anomaly type
    pub anomaly_type: AnomalyType,

    /// Affected triple
    pub triple: Triple,

    /// Confidence score
    pub confidence: f32,

    /// Description
    pub description: String,
}

/// Types of data anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Outlier value
    Outlier,

    /// Missing relation
    MissingRelation,

    /// Inconsistent type
    InconsistentType,

    /// Duplicate entity
    DuplicateEntity,

    /// Invalid format
    InvalidFormat,
}

/// Data improvement suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Improvement {
    /// Improvement type
    pub improvement_type: ImprovementType,

    /// Target triple or pattern
    pub target: String,

    /// Suggested action
    pub suggestion: String,

    /// Impact score
    pub impact: f32,
}

/// Types of data improvements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImprovementType {
    /// Add missing relation
    AddRelation,

    /// Merge duplicate entities
    MergeEntities,

    /// Correct data type
    CorrectType,

    /// Add validation constraint
    AddConstraint,

    /// Normalize format
    NormalizeFormat,
}

/// Data consistency error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InconsistencyError {
    /// Error type
    pub error_type: InconsistencyType,

    /// Conflicting triples
    pub triples: Vec<Triple>,

    /// Severity level
    pub severity: Severity,

    /// Error message
    pub message: String,
}

/// Types of data inconsistencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InconsistencyType {
    /// Logical contradiction
    LogicalContradiction,

    /// Type violation
    TypeViolation,

    /// Cardinality violation
    CardinalityViolation,

    /// Domain/range violation
    DomainRangeViolation,
}

/// Severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ai_engine_creation() {
        let config = AiConfig::default();
        let engine = AiEngine::new(config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_config_serialization() {
        let config = AiConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: AiConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(config.enable_gnn, deserialized.enable_gnn);
    }
}
