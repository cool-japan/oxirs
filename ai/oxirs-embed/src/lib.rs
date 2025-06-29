//! # OxiRS Embed: Advanced Knowledge Graph Embeddings
//!
//! This crate provides state-of-the-art knowledge graph embedding methods
//! including TransE, DistMult, ComplEx, and RotatE models, enhanced with
//! biomedical AI capabilities, GPU acceleration, and specialized text processing.
//!
//! ## Key Features
//!
//! ### 🧬 Biomedical AI
//! - Specialized biomedical knowledge graph embeddings
//! - Gene-disease association prediction
//! - Drug-target interaction modeling
//! - Pathway analysis and protein interactions
//! - Domain-specific text embeddings (SciBERT, BioBERT, etc.)
//!
//! ### 🚀 GPU Acceleration
//! - Advanced GPU memory pooling and management
//! - Intelligent tensor caching
//! - Mixed precision training and inference
//! - Multi-stream parallel processing
//! - Pipeline parallelism for large-scale training
//!
//! ### 🤖 Advanced Models
//! - Traditional KG embeddings (TransE, DistMult, ComplEx, RotatE, etc.)
//! - Graph Neural Networks (GCN, GraphSAGE, GAT)
//! - Transformer-based embeddings with fine-tuning
//! - Ontology-aware embeddings with reasoning
//!
//! ### 📊 Production-Ready
//! - Comprehensive evaluation and benchmarking
//! - Model registry and version management
//! - Intelligent caching and optimization
//! - API server for deployment
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use oxirs_embed::{TransE, ModelConfig, Triple, NamedNode};
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create a knowledge graph embedding model
//! let config = ModelConfig::default().with_dimensions(128);
//! let mut model = TransE::new(config);
//!
//! // Add knowledge triples
//! let triple = Triple::new(
//!     NamedNode::new("http://example.org/alice")?,
//!     NamedNode::new("http://example.org/knows")?,
//!     NamedNode::new("http://example.org/bob")?,
//! );
//! model.add_triple(triple)?;
//!
//! // Train the model
//! let stats = model.train(Some(100)).await?;
//! println!("Training completed: {:?}", stats);
//! # Ok(())
//! # }
//! ```
//!
//! ## Biomedical Example
//!
//! ```rust,no_run
//! use oxirs_embed::{BiomedicalEmbedding, BiomedicalEmbeddingConfig};
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create biomedical embedding model
//! let config = BiomedicalEmbeddingConfig::default();
//! let mut model = BiomedicalEmbedding::new(config);
//!
//! // Add biomedical knowledge
//! model.add_gene_disease_association("BRCA1", "breast_cancer", 0.95);
//! model.add_drug_target_interaction("aspirin", "COX1", 0.92);
//!
//! // Train and predict
//! model.train(Some(100)).await?;
//! let predictions = model.predict_gene_disease_associations("BRCA1", 5)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## GPU Acceleration Example
//!
//! ```rust,no_run
//! use oxirs_embed::{GpuAccelerationConfig, GpuAccelerationManager};
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Configure GPU acceleration
//! let config = GpuAccelerationConfig {
//!     enabled: true,
//!     mixed_precision: true,
//!     tensor_caching: true,
//!     multi_stream: true,
//!     num_streams: 4,
//!     ..Default::default()
//! };
//!
//! let mut gpu_manager = GpuAccelerationManager::new(config);
//!
//! // Use accelerated embedding generation
//! let entities = vec!["entity1".to_string(), "entity2".to_string()];
//! let embeddings = gpu_manager.accelerated_embedding_generation(
//!     entities,
//!     |entity| { /* compute embedding */ vec![0.0; 128].into() }
//! ).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Examples
//!
//! See the `examples/` directory for comprehensive demonstrations:
//! - `biomedical_embedding_demo.rs` - Biomedical AI capabilities
//! - `gpu_acceleration_demo.rs` - GPU acceleration features  
//! - `integrated_ai_platform_demo.rs` - Complete AI platform showcase

#[cfg(feature = "api-server")]
pub mod api;
pub mod batch_processing;
pub mod biomedical_embeddings;
pub mod caching;
pub mod cloud_integration;
pub mod compression;
pub mod delta;
pub mod enterprise_knowledge;
pub mod evaluation;
pub mod gpu_acceleration;
pub mod graphql_api;
pub mod inference;
pub mod integration;
pub mod model_registry;
pub mod models;
pub mod monitoring;
pub mod multimodal;
pub mod persistence;
pub mod research_networks;
pub mod training;
pub mod utils;

// Local type definitions (normally would import from oxirs-core and oxirs-vec)
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Vector for embeddings
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    pub values: Vec<f32>,
    pub dimensions: usize,
}

impl Vector {
    pub fn new(values: Vec<f32>) -> Self {
        let dimensions = values.len();
        Self { values, dimensions }
    }

    /// Create from ndarray Array1
    pub fn from_array1(array: &ndarray::Array1<f32>) -> Self {
        Self::new(array.to_vec())
    }

    /// Convert to ndarray Array1
    pub fn to_array1(&self) -> ndarray::Array1<f32> {
        ndarray::Array1::from_vec(self.values.clone())
    }

    /// Element-wise mapping
    pub fn mapv<F>(&self, f: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        Self::new(self.values.iter().map(|&x| f(x)).collect())
    }

    /// Sum of all elements
    pub fn sum(&self) -> f32 {
        self.values.iter().sum()
    }

    /// Square root of the sum
    pub fn sqrt(&self) -> f32 {
        self.sum().sqrt()
    }
}

// Arithmetic operations for Vector
use std::ops::{Add, Sub};

impl Add for &Vector {
    type Output = Vector;

    fn add(self, other: &Vector) -> Vector {
        assert_eq!(
            self.dimensions, other.dimensions,
            "Vector dimensions must match"
        );
        let values: Vec<f32> = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a + b)
            .collect();
        Vector::new(values)
    }
}

impl Sub for &Vector {
    type Output = Vector;

    fn sub(self, other: &Vector) -> Vector {
        assert_eq!(
            self.dimensions, other.dimensions,
            "Vector dimensions must match"
        );
        let values: Vec<f32> = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a - b)
            .collect();
        Vector::new(values)
    }
}

/// Triple structure for RDF triples
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Triple {
    pub subject: NamedNode,
    pub predicate: NamedNode,
    pub object: NamedNode,
}

impl Triple {
    pub fn new(subject: NamedNode, predicate: NamedNode, object: NamedNode) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }
}

/// Named node for RDF resources
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NamedNode {
    pub iri: String,
}

impl NamedNode {
    pub fn new(iri: &str) -> Result<Self> {
        Ok(Self {
            iri: iri.to_string(),
        })
    }
}

impl std::fmt::Display for NamedNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.iri)
    }
}

/// Configuration for embedding models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub dimensions: usize,
    pub learning_rate: f64,
    pub l2_reg: f64,
    pub max_epochs: usize,
    pub batch_size: usize,
    pub negative_samples: usize,
    pub seed: Option<u64>,
    pub use_gpu: bool,
    pub model_params: HashMap<String, f64>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            dimensions: 100,
            learning_rate: 0.01,
            l2_reg: 0.0001,
            max_epochs: 1000,
            batch_size: 1000,
            negative_samples: 10,
            seed: None,
            use_gpu: false,
            model_params: HashMap::new(),
        }
    }
}

impl ModelConfig {
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = dimensions;
        self
    }

    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn with_max_epochs(mut self, max_epochs: usize) -> Self {
        self.max_epochs = max_epochs;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
}

/// Training statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStats {
    pub epochs_completed: usize,
    pub final_loss: f64,
    pub training_time_seconds: f64,
    pub convergence_achieved: bool,
    pub loss_history: Vec<f64>,
}

/// Model statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStats {
    pub num_entities: usize,
    pub num_relations: usize,
    pub num_triples: usize,
    pub dimensions: usize,
    pub is_trained: bool,
    pub model_type: String,
    pub creation_time: DateTime<Utc>,
    pub last_training_time: Option<DateTime<Utc>>,
}

/// Embedding errors
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("Model not trained")]
    ModelNotTrained,
    #[error("Entity not found: {entity}")]
    EntityNotFound { entity: String },
    #[error("Relation not found: {relation}")]
    RelationNotFound { relation: String },
    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

/// Basic embedding model trait
#[async_trait::async_trait]
pub trait EmbeddingModel: Send + Sync {
    fn config(&self) -> &ModelConfig;
    fn model_id(&self) -> &Uuid;
    fn model_type(&self) -> &'static str;
    fn add_triple(&mut self, triple: Triple) -> Result<()>;
    async fn train(&mut self, epochs: Option<usize>) -> Result<TrainingStats>;
    fn get_entity_embedding(&self, entity: &str) -> Result<Vector>;
    fn get_relation_embedding(&self, relation: &str) -> Result<Vector>;
    fn score_triple(&self, subject: &str, predicate: &str, object: &str) -> Result<f64>;
    fn predict_objects(
        &self,
        subject: &str,
        predicate: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>>;
    fn predict_subjects(
        &self,
        predicate: &str,
        object: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>>;
    fn predict_relations(
        &self,
        subject: &str,
        object: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>>;
    fn get_entities(&self) -> Vec<String>;
    fn get_relations(&self) -> Vec<String>;
    fn get_stats(&self) -> ModelStats;
    fn save(&self, path: &str) -> Result<()>;
    fn load(&mut self, path: &str) -> Result<()>;
    fn clear(&mut self);
    fn is_trained(&self) -> bool;

    /// Encode text strings into embeddings
    async fn encode(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

// Re-export main types
#[cfg(feature = "api-server")]
pub use api::{start_server, ApiConfig, ApiState};
pub use batch_processing::{
    BatchJob, BatchProcessingConfig, BatchProcessingManager, BatchProcessingResult,
    BatchProcessingStats, IncrementalConfig, JobProgress, JobStatus, OutputFormat,
    PartitioningStrategy, RetryConfig,
};
pub use biomedical_embeddings::{
    BiomedicalEmbedding, BiomedicalEmbeddingConfig, BiomedicalEntityType, BiomedicalRelationType,
    FineTuningConfig, PreprocessingRule, SpecializedTextConfig, SpecializedTextEmbedding,
    SpecializedTextModel,
};
pub use caching::{CacheConfig, CacheManager, CachedEmbeddingModel};
pub use cloud_integration::{
    AWSSageMakerService, AutoScalingConfig, AzureMLService, BackupConfig, CloudIntegrationConfig,
    CloudIntegrationManager, CloudProvider, CloudService, ClusterStatus, CostEstimate,
    CostOptimizationResult, CostOptimizationStrategy, DeploymentConfig, DeploymentResult,
    DeploymentStatus, EndpointInfo, FunctionInvocationResult, GPUClusterConfig, GPUClusterResult,
    LifecyclePolicy, OptimizationAction, PerformanceTier, ReplicationType,
    ServerlessDeploymentResult, ServerlessFunctionConfig, ServerlessStatus, StorageConfig,
    StorageResult, StorageStatus, StorageType,
};
pub use compression::{
    CompressedModel, CompressionStats, CompressionTarget, DistillationConfig,
    ModelCompressionManager, NASConfig, OptimizationTarget, PruningConfig, PruningMethod,
    QuantizationConfig, QuantizationMethod,
};
pub use delta::{
    ChangeRecord, ChangeStatistics, ChangeType, DeltaConfig, DeltaManager, DeltaResult, DeltaStats,
    IncrementalStrategy,
};
pub use enterprise_knowledge::{
    BehaviorMetrics, CareerPredictions, Category, CategoryHierarchy, CategoryPerformance,
    ColdStartStrategy, CommunicationFrequency, CommunicationPreferences, CustomerEmbedding,
    CustomerPreferences, CustomerRatings, CustomerSegment, Department, DepartmentPerformance,
    EmployeeEmbedding, EnterpriseConfig, EnterpriseKnowledgeAnalyzer, EnterpriseMetrics,
    ExperienceLevel, FeatureType, MarketAnalysis, OrganizationalStructure,
    PerformanceMetrics as EnterprisePerformanceMetrics, ProductAvailability, ProductEmbedding,
    ProductFeature, ProductRecommendation, Project, ProjectOutcome, ProjectParticipation,
    ProjectPerformance, ProjectStatus, Purchase, PurchaseChannel, RecommendationConfig,
    RecommendationEngine, RecommendationEngineType, RecommendationPerformance,
    RecommendationReason, SalesMetrics, Skill, SkillCategory, Team, TeamPerformance,
};
pub use evaluation::{
    BenchmarkReport,
    BenchmarkSuite,
    ConsoleAlertHandler,
    DriftAlert,
    DriftDetector,
    DriftDetectorConfig,
    DriftThresholds,
    // Outlier detection exports
    EmbeddingSnapshot,
    EvaluationConfig,
    EvaluationMetric,
    EvaluationResults,
    EvaluationSuite,
    ModelComparison,
    OutlierDetailedAnalysis,
    OutlierDetectionConfig,
    OutlierDetectionMethod,
    OutlierDetectionResults,
    OutlierDetectionStats,
    OutlierDetector,
    OutlierInstance,
    OutlierScoreDistribution,
    OutlierSummary,
    OutlierThresholds,
    OutlierType,
    QualityMonitor,
    QualityMonitorConfig,
    QualitySnapshot,
    RootCause,
    TemporalPattern,
    TripleEvaluationResult,
};
pub use gpu_acceleration::{
    GpuAccelerationConfig, GpuAccelerationManager, GpuMemoryPool, GpuPerformanceStats,
    MixedPrecisionProcessor, MultiStreamProcessor, TensorCache,
};
pub use graphql_api::{
    create_schema, BatchEmbeddingInput, BatchEmbeddingResult, BatchStatus, DistanceMetric,
    EmbeddingFormat, EmbeddingQueryInput, EmbeddingResult, EmbeddingSchema, GraphQLContext,
    ModelInfo, ModelType, SimilarityResult, SimilaritySearchInput,
};
pub use models::{
    AggregationType, ComplEx, DistMult, GNNConfig, GNNEmbedding, GNNType, PoolingStrategy, RotatE,
    TransE, TransformerConfig, TransformerEmbedding, TransformerType,
};
pub use monitoring::{
    Alert, AlertSeverity, AlertThresholds, AlertType, CacheMetrics, ConsoleAlertHandler,
    DriftMetrics, ErrorEvent, ErrorMetrics, ErrorSeverity, LatencyMetrics, MonitoringConfig,
    PerformanceMetrics as MonitoringPerformanceMetrics, PerformanceMonitor, QualityAssessment,
    QualityMetrics, ResourceMetrics, SlackAlertHandler, ThroughputMetrics,
};
pub use multimodal::{
    AlignmentNetwork, AlignmentObjective, ContrastiveConfig, CrossDomainConfig, CrossModalConfig,
    KGEncoder, MultiModalEmbedding, MultiModalStats, TextEncoder,
};
pub use research_networks::{
    AuthorEmbedding, Citation, CitationNetwork, CitationType, Collaboration, CollaborationNetwork,
    NetworkMetrics, PaperSection, PublicationEmbedding, PublicationType, ResearchCommunity,
    ResearchNetworkAnalyzer, ResearchNetworkConfig, TopicModel, TopicModelingConfig,
};

#[cfg(feature = "tucker")]
pub use models::TuckER;

#[cfg(feature = "quatd")]
pub use models::QuatD;

// Re-export model registry types
pub use crate::model_registry::{
    ModelRegistry, ModelVersion, ResourceAllocation as ModelResourceAllocation,
};
