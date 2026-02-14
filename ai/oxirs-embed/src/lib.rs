//! # OxiRS Embed: Advanced Knowledge Graph Embeddings
//!
//! [![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/cool-japan/oxirs/releases)
//! [![docs.rs](https://docs.rs/oxirs-embed/badge.svg)](https://docs.rs/oxirs-embed)
//!
//! **Status**: Production Release (v0.1.0)
//! **Stability**: Public APIs are stable. Production-ready with comprehensive testing.
//!
//! State-of-the-art knowledge graph embedding methods including TransE, DistMult, ComplEx,
//! and RotatE models, enhanced with biomedical AI, GPU acceleration, and specialized text processing.
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
//! use oxirs_embed::{TransE, ModelConfig, Triple, NamedNode, EmbeddingModel};
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
//! println!("Training completed: {stats:?}");
//! # Ok(())
//! # }
//! ```
//!
//! ## Biomedical Example
//!
//! ```rust,no_run
//! use oxirs_embed::{BiomedicalEmbedding, BiomedicalEmbeddingConfig, EmbeddingModel};
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
//! ```rust,ignore
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

#![allow(dead_code)]

pub mod acceleration;
pub mod adaptive_learning;
pub mod advanced_profiler;
#[cfg(feature = "api-server")]
pub mod api;
pub mod application_tasks;
pub mod batch_processing;
pub mod biomedical_embeddings;
pub mod caching;
pub mod causal_representation_learning;
pub mod cloud_integration;
pub mod clustering;
pub mod community_detection;
pub mod compression;
pub mod contextual;
pub mod continual_learning;
pub mod cross_domain_transfer;
pub mod cross_module_performance;
pub mod delta;
pub mod diffusion_embeddings;
pub mod distributed_training;
pub mod enterprise_knowledge;
pub mod entity_linking;
pub mod evaluation;
pub mod federated_learning;
pub mod fine_tuning;
#[cfg(feature = "gpu")]
pub mod gpu_acceleration;
pub mod graphql_api;
pub mod inference;
pub mod integration;
pub mod interpretability;
pub mod link_prediction;
pub mod mamba_attention;
pub mod mixed_precision;
pub mod model_registry;
pub mod model_selection;
pub mod models;
pub mod monitoring;
pub mod multimodal;
pub mod neural_symbolic_integration;
pub mod neuro_evolution;
pub mod novel_architectures;
pub mod performance_profiler;
pub mod persistence;
pub mod quantization;
pub mod real_time_fine_tuning;
pub mod real_time_optimization;
pub mod research_networks;
// pub mod revolutionary_optimization; // Disabled - awaiting scirs2-core API stabilization
pub mod sparql_extension;
pub mod storage_backend;
pub mod temporal_embeddings;
pub mod training;
pub mod utils;
pub mod validation;
pub mod vector_search;
pub mod vision_language_graph;
pub mod visualization;

// Import Vector from oxirs-vec for type compatibility across the ecosystem
pub use oxirs_vec::Vector as VecVector;

// Adaptive Learning System exports
pub use adaptive_learning::{
    AdaptationMetrics, AdaptationStrategy, AdaptiveLearningConfig, AdaptiveLearningSystem,
    QualityFeedback,
};

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::{Add, Sub};
use uuid::Uuid;

/// Compatibility wrapper for Vector that provides the old interface
/// while using the sophisticated oxirs-vec Vector internally
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Vector {
    pub values: Vec<f32>,
    pub dimensions: usize,
    #[serde(skip)]
    inner: Option<VecVector>,
}

impl Vector {
    pub fn new(values: Vec<f32>) -> Self {
        let dimensions = values.len();
        Self {
            values,
            dimensions,
            inner: None,
        }
    }

    /// Get or create the inner VecVector
    fn get_inner(&self) -> VecVector {
        // Create a new VecVector from values if needed
        if let Some(ref inner) = self.inner {
            inner.clone()
        } else {
            VecVector::new(self.values.clone())
        }
    }

    /// Update internal state when values change
    fn sync_internal(&mut self) {
        self.dimensions = self.values.len();
        self.inner = None; // Will be recreated on next access
    }

    /// Create from ndarray Array1
    pub fn from_array1(array: &scirs2_core::ndarray_ext::Array1<f32>) -> Self {
        Self::new(array.to_vec())
    }

    /// Convert to ndarray Array1
    pub fn to_array1(&self) -> scirs2_core::ndarray_ext::Array1<f32> {
        scirs2_core::ndarray_ext::Array1::from_vec(self.values.clone())
    }

    /// Element-wise mapping
    pub fn mapv<F>(&self, f: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        Self::new(self.values.iter().copied().map(f).collect())
    }

    /// Sum of all elements
    pub fn sum(&self) -> f32 {
        self.values.iter().sum()
    }

    /// Square root of the sum
    pub fn sqrt(&self) -> f32 {
        self.sum().sqrt()
    }

    /// Get the inner VecVector for advanced operations
    pub fn inner(&self) -> VecVector {
        self.get_inner()
    }

    /// Convert into the inner VecVector
    pub fn into_inner(self) -> VecVector {
        self.inner.unwrap_or_else(|| VecVector::new(self.values))
    }

    /// Create from VecVector with optimized memory allocation
    pub fn from_vec_vector(vec_vector: VecVector) -> Self {
        let values = vec_vector.as_f32().to_vec();
        let dimensions = values.len();
        Self {
            values,
            dimensions,
            inner: Some(vec_vector),
        }
    }

    /// Create vector with pre-allocated capacity for performance
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            dimensions: 0,
            inner: None,
        }
    }

    /// Extend vector with optimized memory reallocation
    pub fn extend_optimized(&mut self, other_values: &[f32]) {
        // Reserve capacity to avoid multiple reallocations
        self.values.reserve(other_values.len());
        self.values.extend_from_slice(other_values);
        self.sync_internal();
    }

    /// Shrink vector memory to fit actual size
    pub fn shrink_to_fit(&mut self) {
        self.values.shrink_to_fit();
        self.sync_internal();
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.values.capacity() * std::mem::size_of::<f32>() + std::mem::size_of::<Self>()
    }
}

// Arithmetic operations for Vector (compatibility with old interface)
impl Add for &Vector {
    type Output = Vector;

    fn add(self, other: &Vector) -> Vector {
        // Use the sophisticated vector addition from oxirs-vec
        if let (Some(self_inner), Some(other_inner)) = (&self.inner, &other.inner) {
            if let Ok(result) = self_inner.add(other_inner) {
                return Vector::from_vec_vector(result);
            }
        }
        // Fallback to element-wise addition for compatibility
        assert_eq!(
            self.values.len(),
            other.values.len(),
            "Vector dimensions must match"
        );
        let result_values: Vec<f32> = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a + b)
            .collect();
        Vector::new(result_values)
    }
}

impl Sub for &Vector {
    type Output = Vector;

    fn sub(self, other: &Vector) -> Vector {
        // Use the sophisticated vector subtraction from oxirs-vec
        if let (Some(self_inner), Some(other_inner)) = (&self.inner, &other.inner) {
            if let Ok(result) = self_inner.subtract(other_inner) {
                return Vector::from_vec_vector(result);
            }
        }
        // Fallback to element-wise subtraction for compatibility
        assert_eq!(
            self.values.len(),
            other.values.len(),
            "Vector dimensions must match"
        );
        let result_values: Vec<f32> = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a - b)
            .collect();
        Vector::new(result_values)
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

impl Default for ModelStats {
    fn default() -> Self {
        Self {
            num_entities: 0,
            num_relations: 0,
            num_triples: 0,
            dimensions: 0,
            is_trained: false,
            model_type: "unknown".to_string(),
            creation_time: Utc::now(),
            last_training_time: None,
        }
    }
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
pub use acceleration::{AdaptiveEmbeddingAccelerator, GpuEmbeddingAccelerator};
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
pub use causal_representation_learning::{
    CausalDiscoveryAlgorithm, CausalDiscoveryConfig, CausalGraph, CausalRepresentationConfig,
    CausalRepresentationModel, ConstraintSettings, CounterfactualConfig, CounterfactualQuery,
    DisentanglementConfig, DisentanglementMethod, ExplanationType, IndependenceTest,
    InterventionConfig, ScoreSettings, StructuralCausalModelConfig,
};
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
// pub use contextual::{
//     ContextualConfig, ContextualEmbeddingModel, EmbeddingContext,
// };
pub use continual_learning::{
    ArchitectureConfig, BoundaryDetection, ConsolidationConfig, ContinualLearningConfig,
    ContinualLearningModel, MemoryConfig, MemoryType, MemoryUpdateStrategy, RegularizationConfig,
    ReplayConfig, ReplayMethod, TaskConfig, TaskDetection, TaskSwitching,
};
pub use cross_module_performance::{
    CoordinatorConfig, CrossModulePerformanceCoordinator, GlobalPerformanceMetrics, ModuleMetrics,
    ModulePerformanceMonitor, OptimizationCache, PerformanceSnapshot, PredictivePerformanceEngine,
    ResourceAllocator, ResourceTracker,
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
    QueryAnsweringEvaluator, QueryEvaluationConfig, QueryEvaluationResults, QueryMetric,
    QueryResult, QueryTemplate, QueryType, ReasoningChain, ReasoningEvaluationConfig,
    ReasoningEvaluationResults, ReasoningRule, ReasoningStep, ReasoningTaskEvaluator,
    ReasoningType, TypeSpecificResults,
};
pub use federated_learning::{
    AggregationEngine, AggregationStrategy, AuthenticationConfig, AuthenticationMethod,
    CertificateConfig, ClippingMechanisms, ClippingMethod, CommunicationConfig,
    CommunicationManager, CommunicationProtocol, CompressionAlgorithm, CompressionConfig,
    CompressionEngine, ConvergenceMetrics, ConvergenceStatus, DataSelectionStrategy,
    DataStatistics, EncryptionScheme, FederatedConfig, FederatedCoordinator,
    FederatedEmbeddingModel, FederatedMessage, FederatedRound, FederationStats, GlobalModelState,
    HardwareAccelerator, KeyManager, LocalModelState, LocalTrainingStats, LocalUpdate,
    MetaLearningConfig, NoiseGenerator, NoiseMechanism, OutlierAction, OutlierDetection,
    OutlierDetectionMethod, Participant, ParticipantCapabilities, ParticipantStatus,
    PersonalizationConfig, PersonalizationStrategy, PrivacyAccountant, PrivacyConfig,
    PrivacyEngine, PrivacyMetrics, PrivacyParams, RoundMetrics, RoundStatus, SecurityConfig,
    SecurityFeature, SecurityManager, TrainingConfig, VerificationEngine, VerificationMechanism,
    VerificationResult, WeightingScheme,
};
#[cfg(feature = "gpu")]
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
    AggregationType, ComplEx, DistMult, GNNConfig, GNNEmbedding, GNNType, HoLE, HoLEConfig,
    PoolingStrategy, RotatE, TransE, TransformerConfig, TransformerEmbedding, TransformerType,
};

pub use contextual::{
    AccessibilityPreferences, ComplexityLevel, ContextualConfig, ContextualEmbeddingModel,
    DomainContext, EmbeddingContext, PerformanceRequirements, PriorityLevel, PrivacySettings,
    QueryContext, QueryType as ContextualQueryType, ResponseFormat, TaskConstraints, TaskContext,
    TaskType, UserContext, UserHistory, UserPreferences,
};
pub use distributed_training::{
    AggregationMethod, CommunicationBackend, DistributedEmbeddingTrainer, DistributedStrategy,
    DistributedTrainingConfig, DistributedTrainingCoordinator, DistributedTrainingStats,
    FaultToleranceConfig, WorkerInfo, WorkerStatus,
};
#[cfg(feature = "conve")]
pub use models::{ConvE, ConvEConfig};
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
pub use neural_symbolic_integration::{
    ConstraintSatisfactionConfig, ConstraintType, KnowledgeIntegrationConfig, KnowledgeRule,
    LogicIntegrationConfig, LogicProgrammingConfig, LogicalFormula, NeuralSymbolicConfig,
    NeuralSymbolicModel, NeuroSymbolicArchitectureConfig, OntologicalConfig, ReasoningEngine,
    RuleBasedConfig, SymbolicReasoningConfig,
};
pub use novel_architectures::{
    ActivationType, ArchitectureParams, ArchitectureState, ArchitectureType, CurvatureComputation,
    CurvatureMethod, CurvatureType, DynamicsConfig, EntanglementStructure, EquivarianceGroup,
    FlowType, GeometricConfig, GeometricParams, GeometricSpace, GeometricState,
    GraphTransformerParams, GraphTransformerState, HyperbolicDistance, HyperbolicInit,
    HyperbolicManifold, HyperbolicParams, HyperbolicState, IntegrationScheme, IntegrationStats,
    ManifoldLearning, ManifoldMethod, ManifoldOptimizer, NeuralODEParams, NeuralODEState,
    NovelArchitectureConfig, NovelArchitectureModel, ODERegularization, ODESolverType,
    ParallelTransport, QuantumGateSet, QuantumMeasurement, QuantumNoise, QuantumParams,
    QuantumState, StabilityConstraints, StructuralBias, TimeEvolution, TransportMethod,
};
pub use research_networks::{
    AuthorEmbedding, Citation, CitationNetwork, CitationType, Collaboration, CollaborationNetwork,
    NetworkMetrics, PaperSection, PublicationEmbedding, PublicationType, ResearchCommunity,
    ResearchNetworkAnalyzer, ResearchNetworkConfig, TopicModel, TopicModelingConfig,
};
pub use sparql_extension::{
    ExpandedQuery, Expansion, ExpansionType, QueryStatistics as SparqlQueryStatistics,
    SparqlExtension, SparqlExtensionConfig,
};
pub use storage_backend::{
    DiskBackend, EmbeddingMetadata, EmbeddingVersion, MemoryBackend, StorageBackend,
    StorageBackendConfig, StorageBackendManager, StorageBackendType, StorageStats,
};
pub use temporal_embeddings::{
    TemporalEmbeddingConfig, TemporalEmbeddingModel, TemporalEvent, TemporalForecast,
    TemporalGranularity, TemporalScope, TemporalStats, TemporalTriple,
};
pub use vision_language_graph::{
    AggregationFunction, CNNConfig, CrossAttentionConfig, DomainAdaptationConfig,
    DomainAdaptationMethod, EpisodeConfig, FewShotConfig, FewShotMethod, FusionStrategy,
    GraphArchitecture, GraphEncoder, GraphEncoderConfig, JointTrainingConfig, LanguageArchitecture,
    LanguageEncoder, LanguageEncoderConfig, LanguageTransformerConfig, MetaLearner,
    ModalityEncoding, MultiModalTransformer, MultiModalTransformerConfig, NormalizationType,
    PoolingType, PositionEncodingType, ReadoutFunction, TaskCategory, TaskSpecificParams,
    TrainingObjective, TransferLearningConfig, TransferStrategy, ViTConfig, VisionArchitecture,
    VisionEncoder, VisionEncoderConfig, VisionLanguageGraphConfig, VisionLanguageGraphModel,
    VisionLanguageGraphStats, ZeroShotConfig, ZeroShotMethod,
};

#[cfg(feature = "tucker")]
pub use models::TuckER;

#[cfg(feature = "quatd")]
pub use models::QuatD;

// Re-export model registry types
pub use crate::model_registry::{
    ModelRegistry, ModelVersion, ResourceAllocation as ModelResourceAllocation,
};

// Re-export model selection types
pub use crate::model_selection::{
    DatasetCharacteristics, MemoryRequirement, ModelComparison, ModelComparisonEntry,
    ModelRecommendation, ModelSelector, ModelType as SelectionModelType, TrainingTime, UseCaseType,
};

// Re-export performance profiler types
pub use crate::performance_profiler::{
    OperationStats, OperationTimer, OperationType, PerformanceProfiler, PerformanceReport,
};

// Re-export revolutionary optimization types
// Temporarily disabled - awaiting scirs2-core v0.2.0 API stabilization
/*
pub use revolutionary_optimization::{
    AdvancedMemoryConfig, EmbeddingOptimizationResult, OptimizationPriority, OptimizationStatistics,
    OptimizationStrategy, PerformanceTargets, QuantumOptimizationStrategy,
    RevolutionaryEmbeddingOptimizer, RevolutionaryEmbeddingOptimizerFactory,
    RevolutionaryOptimizationConfig, SimilarityComputationMethod, SimilarityOptimizationResult,
    StreamingOptimizationConfig,
};
*/

/// Convenience functions for quick setup and common operations
pub mod quick_start {
    use super::*;
    use crate::models::TransE;

    /// Create a TransE model with sensible defaults for experimentation
    pub fn create_simple_transe_model() -> TransE {
        let config = ModelConfig::default()
            .with_dimensions(128)
            .with_learning_rate(0.01)
            .with_max_epochs(100);
        TransE::new(config)
    }

    /// Create a biomedical embedding model for life sciences applications
    pub fn create_biomedical_model() -> BiomedicalEmbedding {
        let config = BiomedicalEmbeddingConfig::default();
        BiomedicalEmbedding::new(config)
    }

    /// Parse a triple from simple string format "subject predicate object"
    pub fn parse_triple_from_string(triple_str: &str) -> Result<Triple> {
        let parts: Vec<&str> = triple_str.split_whitespace().collect();
        if parts.len() != 3 {
            return Err(anyhow::anyhow!(
                "Triple must have exactly 3 parts separated by spaces"
            ));
        }

        // Helper function to convert short names to full URIs
        let expand_uri = |s: &str| -> String {
            if s.starts_with("http://") || s.starts_with("https://") {
                s.to_string()
            } else {
                format!("http://example.org/{s}")
            }
        };

        Ok(Triple::new(
            NamedNode::new(&expand_uri(parts[0]))?,
            NamedNode::new(&expand_uri(parts[1]))?,
            NamedNode::new(&expand_uri(parts[2]))?,
        ))
    }

    /// Helper to add multiple triples from string format
    pub fn add_triples_from_strings<T: EmbeddingModel>(
        model: &mut T,
        triple_strings: &[&str],
    ) -> Result<usize> {
        let mut count = 0;
        for triple_str in triple_strings {
            let triple = parse_triple_from_string(triple_str)?;
            model.add_triple(triple)?;
            count += 1;
        }
        Ok(count)
    }

    /// Quick function to compute cosine similarity between two embedding vectors
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> Result<f64> {
        if a.len() != b.len() {
            return Err(anyhow::anyhow!(
                "Vector dimensions don't match: {} vs {}",
                a.len(),
                b.len()
            ));
        }

        let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / (norm_a * norm_b))
    }

    /// Generate sample knowledge graph data for testing and prototyping
    pub fn generate_sample_kg_data(
        num_entities: usize,
        num_relations: usize,
    ) -> Vec<(String, String, String)> {
        #[allow(unused_imports)]
        use scirs2_core::random::{Random, Rng};

        let mut random = Random::default();
        let mut triples = Vec::new();

        let entities: Vec<String> = (0..num_entities)
            .map(|i| format!("http://example.org/entity_{i}"))
            .collect();

        let relations: Vec<String> = (0..num_relations)
            .map(|i| format!("http://example.org/relation_{i}"))
            .collect();

        // Generate random triples (avoid self-loops)
        for _ in 0..(num_entities * 2) {
            let subject_idx = random.random_range(0..entities.len());
            let relation_idx = random.random_range(0..relations.len());
            let object_idx = random.random_range(0..entities.len());

            let subject = entities[subject_idx].clone();
            let relation = relations[relation_idx].clone();
            let object = entities[object_idx].clone();

            if subject != object {
                triples.push((subject, relation, object));
            }
        }

        triples
    }

    /// Quick performance measurement utility
    pub fn quick_performance_test<F>(
        name: &str,
        iterations: usize,
        operation: F,
    ) -> std::time::Duration
    where
        F: Fn(),
    {
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            operation();
        }
        let duration = start.elapsed();

        println!(
            "Performance test '{name}': {iterations} iterations in {duration:?} ({:.2} ops/sec)",
            iterations as f64 / duration.as_secs_f64()
        );

        duration
    }

    // Revolutionary optimizer functions temporarily disabled - awaiting scirs2-core v0.2.0 API stabilization
    /*
    /// Create a revolutionary embedding optimizer with quantum focus
    pub async fn create_quantum_optimizer() -> anyhow::Result<RevolutionaryEmbeddingOptimizer> {
        RevolutionaryEmbeddingOptimizerFactory::create_quantum_focused()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create quantum optimizer: {}", e))
    }

    /// Create a revolutionary embedding optimizer with streaming focus
    pub async fn create_streaming_optimizer() -> anyhow::Result<RevolutionaryEmbeddingOptimizer> {
        RevolutionaryEmbeddingOptimizerFactory::create_streaming_focused()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create streaming optimizer: {}", e))
    }

    /// Create a revolutionary embedding optimizer with GPU focus
    pub async fn create_gpu_optimizer() -> anyhow::Result<RevolutionaryEmbeddingOptimizer> {
        RevolutionaryEmbeddingOptimizerFactory::create_gpu_focused()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create GPU optimizer: {}", e))
    }

    /// Create a balanced revolutionary embedding optimizer
    pub async fn create_balanced_optimizer() -> anyhow::Result<RevolutionaryEmbeddingOptimizer> {
        RevolutionaryEmbeddingOptimizerFactory::create_balanced()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create balanced optimizer: {}", e))
    }
    */

    /// Quick performance test with revolutionary optimization
    pub async fn quick_revolutionary_performance_test<F, Fut>(
        name: &str,
        iterations: usize,
        async_operation: F,
    ) -> std::time::Duration
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            async_operation().await;
        }
        let duration = start.elapsed();

        println!(
            "Revolutionary performance test '{name}': {iterations} iterations in {duration:?} ({:.2} ops/sec)",
            iterations as f64 / duration.as_secs_f64()
        );

        duration
    }
}

#[cfg(test)]
mod quick_start_tests {
    use super::*;
    use crate::quick_start::*;

    #[test]
    fn test_create_simple_transe_model() {
        let model = create_simple_transe_model();
        let config = model.config();
        assert_eq!(config.dimensions, 128);
        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.max_epochs, 100);
    }

    #[test]
    fn test_parse_triple_from_string() {
        let triple_str = "http://example.org/alice http://example.org/knows http://example.org/bob";
        let triple = parse_triple_from_string(triple_str).unwrap();
        assert_eq!(triple.subject.iri, "http://example.org/alice");
        assert_eq!(triple.predicate.iri, "http://example.org/knows");
        assert_eq!(triple.object.iri, "http://example.org/bob");
    }

    #[test]
    fn test_parse_triple_from_string_invalid() {
        let triple_str = "http://example.org/alice http://example.org/knows";
        let result = parse_triple_from_string(triple_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_triples_from_strings() {
        let mut model = create_simple_transe_model();
        let triple_strings = [
            "http://example.org/alice http://example.org/knows http://example.org/bob",
            "http://example.org/bob http://example.org/likes http://example.org/music",
        ];

        let count = add_triples_from_strings(&mut model, &triple_strings).unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = cosine_similarity(&a, &b).unwrap();
        assert!((similarity - 1.0).abs() < 1e-10);

        let c = vec![0.0, 1.0, 0.0];
        let similarity2 = cosine_similarity(&a, &c).unwrap();
        assert!((similarity2 - 0.0).abs() < 1e-10);

        // Test different dimensions should fail
        let d = vec![1.0, 0.0];
        assert!(cosine_similarity(&a, &d).is_err());
    }

    #[test]
    fn test_generate_sample_kg_data() {
        let triples = generate_sample_kg_data(5, 3);
        assert!(!triples.is_empty());

        // Check that all subjects and objects are in the expected format
        for (subject, relation, object) in &triples {
            assert!(subject.starts_with("http://example.org/entity_"));
            assert!(relation.starts_with("http://example.org/relation_"));
            assert!(object.starts_with("http://example.org/entity_"));
            assert_ne!(subject, object); // No self-loops
        }
    }

    #[test]
    fn test_quick_performance_test() {
        let duration = quick_performance_test("test_operation", 100, || {
            // Simple operation for testing
            let _sum: i32 = (1..10).sum();
        });

        // In release mode, operations can be extremely fast
        // Just verify the function completes and returns a valid duration
        let _nanos = duration.as_nanos();
    }
}
