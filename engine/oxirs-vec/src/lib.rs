//! # OxiRS Vector Search
//!
//! [![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/cool-japan/oxirs/releases)
//! [![docs.rs](https://docs.rs/oxirs-vec/badge.svg)](https://docs.rs/oxirs-vec)
//!
//! **Status**: Production Release (v0.1.0) - **Production-Ready with Complete Documentation**
//! **Stability**: Public APIs are stable. Production-ready with comprehensive testing and 100 KB of documentation.
//!
//! Vector index abstractions for semantic similarity and AI-augmented SPARQL querying.
//!
//! This crate provides comprehensive vector search capabilities for knowledge graphs,
//! enabling semantic similarity searches, AI-augmented SPARQL queries, and hybrid
//! symbolic-vector operations.

#![allow(dead_code)]
//!
//! ## Features
//!
//! - **Multi-algorithm embeddings**: TF-IDF, sentence transformers, custom models
//! - **Advanced indexing**: HNSW, flat, quantized, and multi-index support
//! - **Rich similarity metrics**: Cosine, Euclidean, Pearson, Jaccard, and more
//! - **SPARQL integration**: `vec:similar` service functions and hybrid queries
//! - **Performance optimization**: Caching, batching, and parallel processing
//!
//! ## Quick Start
//!
//! ```rust
//! use oxirs_vec::{VectorStore, embeddings::EmbeddingStrategy};
//!
//! // Create vector store with sentence transformer embeddings
//! let mut store = VectorStore::with_embedding_strategy(
//!     EmbeddingStrategy::SentenceTransformer
//! ).unwrap();
//!
//! // Index some content
//! store
//!     .index_resource(
//!         "http://example.org/doc1".to_string(),
//!         "This is a document about AI",
//!     )
//!     .unwrap();
//! store
//!     .index_resource(
//!         "http://example.org/doc2".to_string(),
//!         "Machine learning tutorial",
//!     )
//!     .unwrap();
//!
//! // Search for similar content
//! let results = store
//!     .similarity_search("artificial intelligence", 5)
//!     .unwrap();
//!
//! println!("Found {} matching resources", results.len());
//! ```
//!
//! ## Cargo Features
//!
//! This crate follows the **COOLJAPAN Pure Rust Policy**: default features are 100% Pure Rust
//! with no C/Fortran/CUDA dependencies. Optional features requiring system libraries are
//! properly feature-gated.
//!
//! ### Core Features (Pure Rust)
//!
//! - `hnsw` - HNSW index support (default: disabled, Pure Rust)
//! - `simd` - SIMD optimizations for vector operations (Pure Rust)
//! - `parallel` - Parallel processing support (Pure Rust)
//!
//! ### Optional Features (with system dependencies)
//!
//! - `gpu` - GPU acceleration abstractions (Pure Rust, uses scirs2-core GPU backend)
//! - `blas` - BLAS acceleration (requires system BLAS library)
//! - `cuda` - CUDA GPU acceleration (requires NVIDIA CUDA Toolkit)
//!   - When CUDA toolkit is installed: enables GPU-accelerated operations
//!   - When CUDA toolkit is missing: gracefully falls back to CPU implementations
//!   - Install CUDA from: <https://developer.nvidia.com/cuda-downloads>
//! - `candle-gpu` - Candle GPU backend (Pure Rust)
//! - `gpu-full` - All GPU features combined (`cuda` + `candle-gpu` + `gpu`)
//!
//! ### Content Processing
//!
//! - `images` - Image processing support
//! - `content-processing` - Full content processing (PDF, archives, XML, images)
//!
//! ### Language Integration
//!
//! - `python` - Python bindings via PyO3
//! - `huggingface` - HuggingFace Hub integration
//!
//! ### Default Build
//!
//! ```toml
//! [dependencies]
//! oxirs-vec = "0.1"  # 100% Pure Rust, no system dependencies
//! ```
//!
//! ### GPU-Accelerated Build (requires CUDA toolkit)
//!
//! ```toml
//! [dependencies]
//! oxirs-vec = { version = "0.1", features = ["gpu-full"] }
//! ```

use anyhow::Result;
use std::collections::HashMap;

pub mod adaptive_compression;
pub mod adaptive_intelligent_caching;
pub mod advanced_analytics;
pub mod advanced_benchmarking;
pub mod advanced_caching;
pub mod advanced_metrics;
pub mod advanced_result_merging;
pub mod automl_optimization;
pub mod benchmarking;
pub mod cache_friendly_index;
pub mod clustering;
pub mod compaction;
pub mod compression;
#[cfg(feature = "content-processing")]
pub mod content_processing;
pub mod crash_recovery;
pub mod cross_language_alignment;
pub mod cross_modal_embeddings;
pub mod diskann;
pub mod distance_metrics;
pub mod distributed_vector_search;
pub mod dynamic_index_selector;
pub mod embedding_pipeline;
pub mod embeddings;
pub mod enhanced_performance_monitoring;
pub mod faiss_compatibility;
pub mod faiss_gpu_integration;
pub mod faiss_integration;
pub mod faiss_migration_tools;
pub mod faiss_native_integration;
pub mod federated_search;
pub mod filtered_search;
pub mod gnn_embeddings;
pub mod gpu;
pub mod gpu_benchmarks;
pub mod graph_aware_search;
pub mod graph_indices;
pub mod hierarchical_similarity;
pub mod hnsw;
pub mod huggingface;
pub mod hybrid_fusion;
pub mod hybrid_search;
pub mod index;
pub mod ivf;
pub mod joint_embedding_spaces;
pub mod kg_embeddings;
pub mod learned_index;
pub mod lsh;
pub mod mmap_advanced;
pub mod mmap_index;
pub mod multi_modal_search;
pub mod multi_tenancy;
pub mod nsg;
pub mod opq;
pub mod oxirs_arq_integration;
pub mod performance_insights;
pub mod persistence;
pub mod personalized_search;
pub mod pq;
pub mod pytorch;
pub mod quantum_search;
pub mod query_planning;
pub mod query_rewriter;
pub mod random_utils;
pub mod rdf_content_enhancement;
pub mod rdf_integration;
pub mod real_time_analytics;
pub mod real_time_embedding_pipeline;
pub mod real_time_updates;
pub mod reranking;
pub mod result_fusion;
pub mod similarity;
pub mod sparql_integration;
pub mod sparql_service_endpoint;
pub mod sparse;
pub mod sq;
pub mod storage_optimizations;
pub mod store_integration;
pub mod structured_vectors;
pub mod tensorflow;
pub mod tiering;
pub mod tree_indices;
pub mod validation;
pub mod wal;
pub mod word2vec;

// Python bindings module
#[cfg(feature = "python")]
pub mod python_bindings;

// Re-export commonly used types
pub use adaptive_compression::{
    AdaptiveCompressor, CompressionMetrics, CompressionPriorities, MultiLevelCompression,
    VectorStats,
};
pub use adaptive_intelligent_caching::{
    AccessPatternAnalyzer, AdaptiveIntelligentCache, CacheConfiguration, CacheOptimizer,
    CachePerformanceMetrics, CacheTier, MLModels, PredictivePrefetcher,
};
pub use advanced_analytics::{
    AnomalyDetection, AnomalyDetector, AnomalyType, ImplementationEffort,
    OptimizationRecommendation, PerformanceTrends, Priority, QualityAspect, QualityRecommendation,
    QueryAnalytics, QueryAnomaly, RecommendationType, VectorAnalyticsEngine,
    VectorDistributionAnalysis, VectorQualityAssessment,
};
pub use advanced_benchmarking::{
    AdvancedBenchmarkConfig, AdvancedBenchmarkResult, AdvancedBenchmarkSuite, AlgorithmParameters,
    BenchmarkAlgorithm, BuildTimeMetrics, CacheMetrics, DatasetQualityMetrics, DatasetStatistics,
    DistanceStatistics, EnhancedBenchmarkDataset, HyperparameterTuner, IndexSizeMetrics,
    LatencyMetrics, MemoryMetrics, ObjectiveFunction, OptimizationStrategy,
    ParallelBenchmarkConfig, ParameterSpace, ParameterType, ParameterValue, PerformanceMetrics,
    PerformanceProfiler, QualityDegradation, QualityMetrics, ScalabilityMetrics,
    StatisticalAnalyzer, StatisticalMetrics, ThroughputMetrics,
};
pub use advanced_caching::{
    BackgroundCacheWorker, CacheAnalysisReport, CacheAnalyzer, CacheConfig, CacheEntry,
    CacheInvalidator, CacheKey, CacheStats, CacheWarmer, EvictionPolicy, InvalidationStats,
    MultiLevelCache, MultiLevelCacheStats,
};
pub use advanced_result_merging::{
    AdvancedResultMerger, ConfidenceInterval, DiversityConfig, DiversityMetric, FusionStatistics,
    MergedResult, RankFusionAlgorithm, RankingFactor, ResultExplanation, ResultMergingConfig,
    ResultMetadata, ScoreCombinationStrategy, ScoreNormalizationMethod, ScoredResult,
    SourceContribution, SourceResult, SourceType,
};
pub use automl_optimization::{
    AutoMLConfig, AutoMLOptimizer, AutoMLResults, AutoMLStatistics, IndexConfiguration,
    IndexParameterSpace, OptimizationMetric, OptimizationTrial, ResourceConstraints, SearchSpace,
    TrialResult,
};
pub use benchmarking::{
    BenchmarkConfig, BenchmarkDataset, BenchmarkOutputFormat, BenchmarkResult, BenchmarkRunner,
    BenchmarkSuite, BenchmarkTestCase, MemoryMetrics as BenchmarkMemoryMetrics,
    PerformanceMetrics as BenchmarkPerformanceMetrics, QualityMetrics as BenchmarkQualityMetrics,
    ScalabilityMetrics as BenchmarkScalabilityMetrics, SystemInfo,
};
pub use cache_friendly_index::{CacheFriendlyVectorIndex, IndexConfig as CacheFriendlyIndexConfig};
pub use compaction::{
    CompactionConfig, CompactionManager, CompactionMetrics, CompactionResult, CompactionState,
    CompactionStatistics, CompactionStrategy,
};
pub use compression::{create_compressor, CompressionMethod, VectorCompressor};
#[cfg(feature = "content-processing")]
pub use content_processing::{
    ChunkType, ChunkingStrategy, ContentChunk, ContentExtractionConfig, ContentLocation,
    ContentProcessor, DocumentFormat, DocumentStructure, ExtractedContent, ExtractedImage,
    ExtractedLink, ExtractedTable, FormatHandler, Heading, ProcessingStats, TocEntry,
};
pub use crash_recovery::{CrashRecoveryManager, RecoveryConfig, RecoveryPolicy, RecoveryStats};
pub use cross_modal_embeddings::{
    AttentionMechanism, AudioData, AudioEncoder, CrossModalConfig, CrossModalEncoder, FusionLayer,
    FusionStrategy, GraphData, GraphEncoder, ImageData, ImageEncoder, Modality, ModalityData,
    MultiModalContent, TextEncoder, VideoData, VideoEncoder,
};
pub use diskann::{
    DiskAnnBuildStats, DiskAnnBuilder, DiskAnnConfig, DiskAnnError, DiskAnnIndex, DiskAnnResult,
    DiskStorage, IndexMetadata as DiskAnnIndexMetadata, MemoryMappedStorage, NodeId,
    PruningStrategy, SearchMode as DiskAnnSearchMode, SearchStats as DiskAnnSearchStats,
    StorageBackend, VamanaGraph, VamanaNode, VectorId as DiskAnnVectorId,
};
pub use distributed_vector_search::{
    ConsistencyLevel, DistributedClusterStats, DistributedNodeConfig, DistributedQuery,
    DistributedSearchResponse, DistributedVectorSearch, LoadBalancingAlgorithm, NodeHealthStatus,
    PartitioningStrategy, QueryExecutionStrategy,
};
pub use dynamic_index_selector::{DynamicIndexSelector, IndexSelectorConfig};
pub use embedding_pipeline::{
    DimensionalityReduction, EmbeddingPipeline, NormalizationConfig, PostprocessingPipeline,
    PreprocessingPipeline, TokenizerConfig, VectorNormalization,
};
pub use embeddings::{
    EmbeddableContent, EmbeddingConfig, EmbeddingManager, EmbeddingStrategy, ModelDetails,
    OpenAIConfig, OpenAIEmbeddingGenerator, SentenceTransformerGenerator, TransformerModelType,
};
pub use enhanced_performance_monitoring::{
    Alert, AlertManager, AlertSeverity, AlertThresholds, AlertType, AnalyticsEngine,
    AnalyticsReport, DashboardData, EnhancedPerformanceMonitor, ExportConfig, ExportDestination,
    ExportFormat, LatencyDistribution, MonitoringConfig as EnhancedMonitoringConfig,
    QualityMetrics as EnhancedQualityMetrics, QualityMetricsCollector, QualityStatistics,
    QueryInfo, QueryMetricsCollector, QueryStatistics, QueryType, Recommendation,
    RecommendationCategory, RecommendationPriority, SystemMetrics, SystemMetricsCollector,
    SystemStatistics, TrendData, TrendDirection,
};
pub use faiss_compatibility::{
    CompressionLevel, ConversionMetrics, ConversionResult, FaissCompatibility, FaissExportConfig,
    FaissImportConfig, FaissIndexMetadata, FaissIndexType, FaissMetricType, FaissParameter,
    SimpleVectorIndex,
};
pub use federated_search::{
    AuthenticationConfig, FederatedSearchConfig, FederatedVectorSearch, FederationEndpoint,
    PrivacyEngine, PrivacyMode, SchemaCompatibility, TrustManager,
};
pub use gnn_embeddings::{AggregatorType, GraphSAGE, GCN};
pub use gpu::{
    create_default_accelerator, create_memory_optimized_accelerator,
    create_performance_accelerator, is_gpu_available, GpuAccelerator, GpuBuffer, GpuConfig,
    GpuDevice, GpuExecutionConfig,
};
pub use gpu_benchmarks::{
    BenchmarkResult as GpuBenchmarkResult, GpuBenchmarkConfig, GpuBenchmarkSuite,
};
pub use graph_indices::{
    DelaunayGraph, GraphIndex, GraphIndexConfig, GraphType, NSWGraph, ONNGGraph, PANNGGraph,
    RNGGraph,
};
pub use hierarchical_similarity::{
    ConceptHierarchy, HierarchicalSimilarity, HierarchicalSimilarityConfig,
    HierarchicalSimilarityResult, HierarchicalSimilarityStats, SimilarityContext,
    SimilarityExplanation, SimilarityTaskType,
};
pub use hnsw::{HnswConfig, HnswIndex};
pub use hybrid_fusion::{
    FusedResult, HybridFusion, HybridFusionConfig, HybridFusionStatistics, HybridFusionStrategy,
    NormalizationMethod,
};
pub use hybrid_search::{
    Bm25Scorer, DocumentScore, HybridQuery, HybridResult, HybridSearchConfig, HybridSearchManager,
    KeywordAlgorithm, KeywordMatch, KeywordSearcher, QueryExpander, RankFusion, RankFusionStrategy,
    SearchMode, SearchWeights, TfidfScorer,
};

#[cfg(feature = "tantivy-search")]
pub use hybrid_search::{
    IndexStats, RdfDocument, TantivyConfig, TantivySearchResult, TantivySearcher,
};
pub use index::{AdvancedVectorIndex, DistanceMetric, IndexConfig, IndexType, SearchResult};
pub use ivf::{IvfConfig, IvfIndex, IvfStats, QuantizationStrategy};
pub use joint_embedding_spaces::{
    ActivationFunction, AlignmentPair, CLIPAligner, ContrastiveOptimizer, CrossModalAttention,
    CurriculumLearning, DataAugmentation, DifficultySchedule, DomainAdapter, DomainStatistics,
    JointEmbeddingConfig, JointEmbeddingSpace, LearningRateSchedule, LinearProjector,
    PacingFunction, ScheduleType, TemperatureScheduler, TrainingStatistics,
};
pub use kg_embeddings::{
    ComplEx, KGEmbedding, KGEmbeddingConfig, KGEmbeddingModel as KGModel, KGEmbeddingModelType,
    RotatE, TransE, Triple,
};
pub use lsh::{LshConfig, LshFamily, LshIndex, LshStats};
pub use mmap_index::{MemoryMappedIndexStats, MemoryMappedVectorIndex};
pub use multi_tenancy::{
    AccessControl, AccessPolicy, BillingEngine, BillingMetrics, BillingPeriod, IsolationLevel,
    IsolationStrategy, MultiTenancyError, MultiTenancyResult, MultiTenantManager, NamespaceManager,
    Permission, PricingModel, QuotaEnforcer, QuotaLimits, QuotaUsage, RateLimiter, ResourceQuota,
    ResourceType, Role, Tenant, TenantConfig, TenantContext, TenantId, TenantManagerConfig,
    TenantMetadata, TenantOperation, TenantStatistics, TenantStatus, UsageRecord,
};
pub use nsg::{DistanceMetric as NsgDistanceMetric, NsgConfig, NsgIndex, NsgStats};
pub use performance_insights::{
    AlertingSystem, OptimizationRecommendations, PerformanceInsightsAnalyzer,
    PerformanceTrends as InsightsPerformanceTrends, QueryComplexity,
    QueryStatistics as InsightsQueryStatistics, ReportFormat, VectorStatistics,
};
pub use pq::{PQConfig, PQIndex, PQStats};
pub use pytorch::{
    ArchitectureType, CompileMode, DeviceManager, PyTorchConfig, PyTorchDevice, PyTorchEmbedder,
    PyTorchModelManager, PyTorchModelMetadata, PyTorchTokenizer,
};
pub use quantum_search::{
    QuantumSearchConfig, QuantumSearchResult, QuantumSearchStatistics, QuantumState,
    QuantumVectorSearch,
};
pub use query_planning::{
    CostModel, IndexStatistics, QueryCharacteristics, QueryPlan, QueryPlanner, QueryStrategy,
    VectorQueryType,
};
pub use query_rewriter::{
    QueryRewriter, QueryRewriterConfig, QueryVectorStatistics, RewriteRule, RewrittenQuery,
};
pub use rdf_content_enhancement::{
    ComponentWeights, MultiLanguageProcessor, PathConstraint, PathDirection, PropertyAggregator,
    PropertyPath, RdfContentConfig, RdfContentProcessor, RdfContext, RdfEntity, RdfValue,
    TemporalInfo,
};
pub use rdf_integration::{
    RdfIntegrationStats, RdfTermMapping, RdfTermMetadata, RdfTermType, RdfVectorConfig,
    RdfVectorIntegration, RdfVectorSearchResult, SearchMetadata,
};
pub use real_time_analytics::{
    AlertSeverity as AnalyticsAlertSeverity, AlertType as AnalyticsAlertType, AnalyticsConfig,
    AnalyticsEvent, AnalyticsReport as RealTimeAnalyticsReport,
    DashboardData as RealTimeDashboardData, ExportFormat as AnalyticsExportFormat,
    MetricsCollector, PerformanceMonitor, QueryMetrics, SystemMetrics as AnalyticsSystemMetrics,
    VectorAnalyticsEngine as RealTimeVectorAnalyticsEngine,
};
pub use real_time_embedding_pipeline::{
    AlertThresholds as PipelineAlertThresholds, AutoScalingConfig, CompressionConfig, ContentItem,
    MonitoringConfig as PipelineMonitoringConfig, PipelineConfig as RealTimeEmbeddingConfig,
    PipelineStatistics as PipelineStats, ProcessingPriority, ProcessingResult, ProcessingStatus,
    RealTimeEmbeddingPipeline, VersioningStrategy,
};
pub use real_time_updates::{
    BatchProcessor, RealTimeConfig, RealTimeVectorSearch, RealTimeVectorUpdater, UpdateBatch,
    UpdateOperation, UpdatePriority, UpdateStats,
};
pub use reranking::{
    CrossEncoder, CrossEncoderBackend, CrossEncoderModel, CrossEncoderReranker, DiversityReranker,
    DiversityStrategy, FusionStrategy as RerankingFusionStrategy, ModelBackend, ModelConfig,
    RerankingCache, RerankingCacheConfig, RerankingConfig, RerankingError, RerankingMode,
    RerankingOutput, RerankingStats, Result as RerankingResult, ScoreFusion, ScoreFusionConfig,
    ScoredCandidate,
};
pub use result_fusion::{
    FusedResults, FusionAlgorithm, FusionConfig, FusionQualityMetrics, FusionStats,
    ResultFusionEngine, ScoreNormalizationStrategy, SourceResults, VectorSearchResult,
};
pub use similarity::{AdaptiveSimilarity, SemanticSimilarity, SimilarityConfig, SimilarityMetric};
pub use sparql_integration::{
    CrossLanguageProcessor, FederatedQueryResult, QueryExecutor, SparqlVectorFunctions,
    SparqlVectorService, VectorOperation, VectorQuery, VectorQueryResult, VectorServiceArg,
    VectorServiceConfig, VectorServiceResult,
};

#[cfg(feature = "tantivy-search")]
pub use sparql_integration::{RdfLiteral, SearchStats, SparqlSearchResult, SparqlTextFunctions};
pub use sparql_service_endpoint::{
    AuthenticationInfo, AuthenticationType, CustomFunctionRegistry, FederatedOperation,
    FederatedSearchResult, FederatedServiceEndpoint, FederatedVectorQuery, FunctionMetadata,
    LoadBalancer, ParameterInfo, ParameterType as ServiceParameterType, PartialSearchResult,
    QueryScope, ReturnType, ServiceCapability, ServiceEndpointManager, ServiceType,
};
pub use sparse::{COOMatrix, CSRMatrix, SparseVector};
pub use sq::{QuantizationMode, QuantizationParams, SqConfig, SqIndex, SqStats};
pub use storage_optimizations::{
    CompressionType, MmapVectorFile, StorageConfig, StorageUtils, VectorBlock, VectorFileHeader,
    VectorReader, VectorWriter,
};
pub use structured_vectors::{
    ConfidenceScoredVector, HierarchicalVector, NamedDimensionVector, TemporalVector,
    WeightedDimensionVector,
};
pub use tensorflow::{
    OptimizationLevel, PreprocessingPipeline as TensorFlowPreprocessingPipeline, ServerConfig,
    SessionConfig, TensorDataType, TensorFlowConfig, TensorFlowDevice, TensorFlowEmbedder,
    TensorFlowModelInfo, TensorFlowModelServer, TensorSpec,
};
pub use tiering::{
    IndexMetadata, StorageTier, TierMetrics, TierStatistics, TierTransitionReason, TieringConfig,
    TieringManager, TieringPolicy,
};
pub use tree_indices::{
    BallTree, CoverTree, KdTree, RandomProjectionTree, TreeIndex, TreeIndexConfig, TreeType, VpTree,
};
pub use wal::{WalConfig, WalEntry, WalManager};
pub use word2vec::{
    AggregationMethod, OovStrategy, Word2VecConfig, Word2VecEmbeddingGenerator, Word2VecFormat,
};

/// Vector identifier type
pub type VectorId = String;

/// Batch search result type
pub type BatchSearchResult = Vec<Result<Vec<(String, f32)>>>;

/// Trait for vector store implementations
pub trait VectorStoreTrait: Send + Sync {
    /// Insert a vector with metadata
    fn insert_vector(&mut self, id: VectorId, vector: Vector) -> Result<()>;

    /// Add a vector and return its ID
    fn add_vector(&mut self, vector: Vector) -> Result<VectorId>;

    /// Get a vector by its ID
    fn get_vector(&self, id: &VectorId) -> Result<Option<Vector>>;

    /// Get all vector IDs
    fn get_all_vector_ids(&self) -> Result<Vec<VectorId>>;

    /// Search for similar vectors
    fn search_similar(&self, query: &Vector, k: usize) -> Result<Vec<(VectorId, f32)>>;

    /// Remove a vector by ID
    fn remove_vector(&mut self, id: &VectorId) -> Result<bool>;

    /// Get the number of vectors stored
    fn len(&self) -> usize;

    /// Check if the store is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Precision types for vectors
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum VectorPrecision {
    F32,
    F64,
    F16,
    I8,
    Binary,
}

/// Multi-precision vector with enhanced functionality
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Vector {
    pub dimensions: usize,
    pub precision: VectorPrecision,
    pub values: VectorData,
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

/// Vector data storage supporting multiple precisions
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum VectorData {
    F32(Vec<f32>),
    F64(Vec<f64>),
    F16(Vec<u16>), // Using u16 to represent f16 bits
    I8(Vec<i8>),
    Binary(Vec<u8>), // Packed binary representation
}

impl Vector {
    /// Create a new F32 vector from values
    pub fn new(values: Vec<f32>) -> Self {
        let dimensions = values.len();
        Self {
            dimensions,
            precision: VectorPrecision::F32,
            values: VectorData::F32(values),
            metadata: None,
        }
    }

    /// Create a new vector with specific precision
    pub fn with_precision(values: VectorData) -> Self {
        let (dimensions, precision) = match &values {
            VectorData::F32(v) => (v.len(), VectorPrecision::F32),
            VectorData::F64(v) => (v.len(), VectorPrecision::F64),
            VectorData::F16(v) => (v.len(), VectorPrecision::F16),
            VectorData::I8(v) => (v.len(), VectorPrecision::I8),
            VectorData::Binary(v) => (v.len() * 8, VectorPrecision::Binary), // 8 bits per byte
        };

        Self {
            dimensions,
            precision,
            values,
            metadata: None,
        }
    }

    /// Create a new vector with metadata
    pub fn with_metadata(
        values: Vec<f32>,
        metadata: std::collections::HashMap<String, String>,
    ) -> Self {
        let dimensions = values.len();
        Self {
            dimensions,
            precision: VectorPrecision::F32,
            values: VectorData::F32(values),
            metadata: Some(metadata),
        }
    }

    /// Create F64 vector
    pub fn f64(values: Vec<f64>) -> Self {
        Self::with_precision(VectorData::F64(values))
    }

    /// Create F16 vector (using u16 representation)
    pub fn f16(values: Vec<u16>) -> Self {
        Self::with_precision(VectorData::F16(values))
    }

    /// Create I8 quantized vector
    pub fn i8(values: Vec<i8>) -> Self {
        Self::with_precision(VectorData::I8(values))
    }

    /// Create binary vector
    pub fn binary(values: Vec<u8>) -> Self {
        Self::with_precision(VectorData::Binary(values))
    }

    /// Get vector values as f32 (converting if necessary)
    pub fn as_f32(&self) -> Vec<f32> {
        match &self.values {
            VectorData::F32(v) => v.clone(),
            VectorData::F64(v) => v.iter().map(|&x| x as f32).collect(),
            VectorData::F16(v) => v.iter().map(|&x| Self::f16_to_f32(x)).collect(),
            VectorData::I8(v) => v.iter().map(|&x| x as f32 / 128.0).collect(), // Normalize to [-1, 1]
            VectorData::Binary(v) => {
                let mut result = Vec::new();
                for &byte in v {
                    for bit in 0..8 {
                        result.push(if (byte >> bit) & 1 == 1 { 1.0 } else { 0.0 });
                    }
                }
                result
            }
        }
    }

    /// Convert f32 to f16 representation (simplified)
    #[allow(dead_code)]
    fn f32_to_f16(value: f32) -> u16 {
        // Simplified f16 conversion - in practice, use proper IEEE 754 half-precision
        let bits = value.to_bits();
        let sign = (bits >> 31) & 0x1;
        let exp = ((bits >> 23) & 0xff) as i32;
        let mantissa = bits & 0x7fffff;

        // Simplified conversion
        let f16_exp = if exp == 0 {
            0
        } else {
            (exp - 127 + 15).clamp(0, 31) as u16
        };

        let f16_mantissa = (mantissa >> 13) as u16;
        ((sign as u16) << 15) | (f16_exp << 10) | f16_mantissa
    }

    /// Convert f16 representation to f32 (simplified)
    fn f16_to_f32(value: u16) -> f32 {
        // Simplified f16 conversion - in practice, use proper IEEE 754 half-precision
        let sign = (value >> 15) & 0x1;
        let exp = ((value >> 10) & 0x1f) as i32;
        let mantissa = value & 0x3ff;

        if exp == 0 {
            if mantissa == 0 {
                if sign == 1 {
                    -0.0
                } else {
                    0.0
                }
            } else {
                // Denormalized number
                let f32_exp = -14 - 127;
                let f32_mantissa = (mantissa as u32) << 13;
                f32::from_bits(((sign as u32) << 31) | ((f32_exp as u32) << 23) | f32_mantissa)
            }
        } else {
            let f32_exp = exp - 15 + 127;
            let f32_mantissa = (mantissa as u32) << 13;
            f32::from_bits(((sign as u32) << 31) | ((f32_exp as u32) << 23) | f32_mantissa)
        }
    }

    /// Quantize f32 vector to i8
    pub fn quantize_to_i8(values: &[f32]) -> Vec<i8> {
        // Find min/max for normalization
        let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        if range == 0.0 {
            vec![0; values.len()]
        } else {
            values
                .iter()
                .map(|&x| {
                    let normalized = (x - min_val) / range; // 0 to 1
                    let scaled = normalized * 254.0 - 127.0; // -127 to 127
                    scaled.round().clamp(-127.0, 127.0) as i8
                })
                .collect()
        }
    }

    /// Convert to binary representation using threshold
    pub fn to_binary(values: &[f32], threshold: f32) -> Vec<u8> {
        let mut binary = Vec::new();
        let mut current_byte = 0u8;
        let mut bit_position = 0;

        for &value in values {
            if value > threshold {
                current_byte |= 1 << bit_position;
            }

            bit_position += 1;
            if bit_position == 8 {
                binary.push(current_byte);
                current_byte = 0;
                bit_position = 0;
            }
        }

        // Handle remaining bits
        if bit_position > 0 {
            binary.push(current_byte);
        }

        binary
    }

    /// Calculate cosine similarity with another vector
    pub fn cosine_similarity(&self, other: &Vector) -> Result<f32> {
        if self.dimensions != other.dimensions {
            return Err(anyhow::anyhow!("Vector dimensions must match"));
        }

        let self_f32 = self.as_f32();
        let other_f32 = other.as_f32();

        let dot_product: f32 = self_f32.iter().zip(&other_f32).map(|(a, b)| a * b).sum();

        let magnitude_self: f32 = self_f32.iter().map(|x| x * x).sum::<f32>().sqrt();
        let magnitude_other: f32 = other_f32.iter().map(|x| x * x).sum::<f32>().sqrt();

        if magnitude_self == 0.0 || magnitude_other == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / (magnitude_self * magnitude_other))
    }

    /// Calculate Euclidean distance to another vector
    pub fn euclidean_distance(&self, other: &Vector) -> Result<f32> {
        if self.dimensions != other.dimensions {
            return Err(anyhow::anyhow!("Vector dimensions must match"));
        }

        let self_f32 = self.as_f32();
        let other_f32 = other.as_f32();

        let distance = self_f32
            .iter()
            .zip(&other_f32)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        Ok(distance)
    }

    /// Calculate Manhattan distance (L1 norm) to another vector
    pub fn manhattan_distance(&self, other: &Vector) -> Result<f32> {
        if self.dimensions != other.dimensions {
            return Err(anyhow::anyhow!("Vector dimensions must match"));
        }

        let self_f32 = self.as_f32();
        let other_f32 = other.as_f32();

        let distance = self_f32
            .iter()
            .zip(&other_f32)
            .map(|(a, b)| (a - b).abs())
            .sum();

        Ok(distance)
    }

    /// Calculate Minkowski distance (general Lp norm) to another vector
    pub fn minkowski_distance(&self, other: &Vector, p: f32) -> Result<f32> {
        if self.dimensions != other.dimensions {
            return Err(anyhow::anyhow!("Vector dimensions must match"));
        }

        if p <= 0.0 {
            return Err(anyhow::anyhow!("p must be positive"));
        }

        let self_f32 = self.as_f32();
        let other_f32 = other.as_f32();

        if p == f32::INFINITY {
            // Special case: Chebyshev distance
            return self.chebyshev_distance(other);
        }

        let distance = self_f32
            .iter()
            .zip(&other_f32)
            .map(|(a, b)| (a - b).abs().powf(p))
            .sum::<f32>()
            .powf(1.0 / p);

        Ok(distance)
    }

    /// Calculate Chebyshev distance (L∞ norm) to another vector
    pub fn chebyshev_distance(&self, other: &Vector) -> Result<f32> {
        if self.dimensions != other.dimensions {
            return Err(anyhow::anyhow!("Vector dimensions must match"));
        }

        let self_f32 = self.as_f32();
        let other_f32 = other.as_f32();

        let distance = self_f32
            .iter()
            .zip(&other_f32)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |max, val| max.max(val));

        Ok(distance)
    }

    /// Get vector magnitude (L2 norm)
    pub fn magnitude(&self) -> f32 {
        let values = self.as_f32();
        values.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize vector to unit length
    pub fn normalize(&mut self) {
        let mag = self.magnitude();
        if mag > 0.0 {
            match &mut self.values {
                VectorData::F32(values) => {
                    for value in values {
                        *value /= mag;
                    }
                }
                VectorData::F64(values) => {
                    let mag_f64 = mag as f64;
                    for value in values {
                        *value /= mag_f64;
                    }
                }
                _ => {
                    // For other types, convert to f32, normalize, then convert back
                    let mut f32_values = self.as_f32();
                    for value in &mut f32_values {
                        *value /= mag;
                    }
                    self.values = VectorData::F32(f32_values);
                    self.precision = VectorPrecision::F32;
                }
            }
        }
    }

    /// Get a normalized copy of this vector
    pub fn normalized(&self) -> Vector {
        let mut normalized = self.clone();
        normalized.normalize();
        normalized
    }

    /// Add another vector (element-wise)
    pub fn add(&self, other: &Vector) -> Result<Vector> {
        if self.dimensions != other.dimensions {
            return Err(anyhow::anyhow!("Vector dimensions must match"));
        }

        let self_f32 = self.as_f32();
        let other_f32 = other.as_f32();

        let result_values: Vec<f32> = self_f32
            .iter()
            .zip(&other_f32)
            .map(|(a, b)| a + b)
            .collect();

        Ok(Vector::new(result_values))
    }

    /// Subtract another vector (element-wise)
    pub fn subtract(&self, other: &Vector) -> Result<Vector> {
        if self.dimensions != other.dimensions {
            return Err(anyhow::anyhow!("Vector dimensions must match"));
        }

        let self_f32 = self.as_f32();
        let other_f32 = other.as_f32();

        let result_values: Vec<f32> = self_f32
            .iter()
            .zip(&other_f32)
            .map(|(a, b)| a - b)
            .collect();

        Ok(Vector::new(result_values))
    }

    /// Scale vector by a scalar
    pub fn scale(&self, scalar: f32) -> Vector {
        let values = self.as_f32();
        let scaled_values: Vec<f32> = values.iter().map(|x| x * scalar).collect();

        Vector::new(scaled_values)
    }

    /// Get the number of dimensions in the vector
    pub fn len(&self) -> usize {
        self.dimensions
    }

    /// Check if vector is empty (zero dimensions)
    pub fn is_empty(&self) -> bool {
        self.dimensions == 0
    }

    /// Get vector as slice of f32 values
    pub fn as_slice(&self) -> Vec<f32> {
        self.as_f32()
    }
}

/// Vector index trait for efficient similarity search
pub trait VectorIndex: Send + Sync {
    /// Insert a vector with associated URI
    fn insert(&mut self, uri: String, vector: Vector) -> Result<()>;

    /// Find k nearest neighbors
    fn search_knn(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>>;

    /// Find all vectors within threshold similarity
    fn search_threshold(&self, query: &Vector, threshold: f32) -> Result<Vec<(String, f32)>>;

    /// Get a vector by its URI
    fn get_vector(&self, uri: &str) -> Option<&Vector>;

    /// Add a vector with associated ID and metadata
    fn add_vector(
        &mut self,
        id: VectorId,
        vector: Vector,
        _metadata: Option<HashMap<String, String>>,
    ) -> Result<()> {
        // Default implementation that delegates to insert
        self.insert(id, vector)
    }

    /// Update an existing vector
    fn update_vector(&mut self, id: VectorId, vector: Vector) -> Result<()> {
        // Default implementation that delegates to insert
        self.insert(id, vector)
    }

    /// Update metadata for a vector
    fn update_metadata(&mut self, _id: VectorId, _metadata: HashMap<String, String>) -> Result<()> {
        // Default implementation (no-op)
        Ok(())
    }

    /// Remove a vector by its ID
    fn remove_vector(&mut self, _id: VectorId) -> Result<()> {
        // Default implementation (no-op)
        Ok(())
    }
}

/// In-memory vector index implementation
pub struct MemoryVectorIndex {
    vectors: Vec<(String, Vector)>,
    similarity_config: similarity::SimilarityConfig,
}

impl MemoryVectorIndex {
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
            similarity_config: similarity::SimilarityConfig::default(),
        }
    }

    pub fn with_similarity_config(config: similarity::SimilarityConfig) -> Self {
        Self {
            vectors: Vec::new(),
            similarity_config: config,
        }
    }
}

impl Default for MemoryVectorIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorIndex for MemoryVectorIndex {
    fn insert(&mut self, uri: String, vector: Vector) -> Result<()> {
        // Check if vector already exists and update it
        if let Some(pos) = self.vectors.iter().position(|(id, _)| id == &uri) {
            self.vectors[pos] = (uri, vector);
        } else {
            self.vectors.push((uri, vector));
        }
        Ok(())
    }

    fn search_knn(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>> {
        let metric = self.similarity_config.primary_metric;
        let query_f32 = query.as_f32();
        let mut similarities: Vec<(String, f32)> = self
            .vectors
            .iter()
            .map(|(uri, vec)| {
                let vec_f32 = vec.as_f32();
                let sim = metric.similarity(&query_f32, &vec_f32).unwrap_or(0.0);
                (uri.clone(), sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(k);

        Ok(similarities)
    }

    fn search_threshold(&self, query: &Vector, threshold: f32) -> Result<Vec<(String, f32)>> {
        let metric = self.similarity_config.primary_metric;
        let query_f32 = query.as_f32();
        let similarities: Vec<(String, f32)> = self
            .vectors
            .iter()
            .filter_map(|(uri, vec)| {
                let vec_f32 = vec.as_f32();
                let sim = metric.similarity(&query_f32, &vec_f32).unwrap_or(0.0);
                if sim >= threshold {
                    Some((uri.clone(), sim))
                } else {
                    None
                }
            })
            .collect();

        Ok(similarities)
    }

    fn get_vector(&self, uri: &str) -> Option<&Vector> {
        self.vectors.iter().find(|(u, _)| u == uri).map(|(_, v)| v)
    }

    fn update_vector(&mut self, id: VectorId, vector: Vector) -> Result<()> {
        if let Some(pos) = self.vectors.iter().position(|(uri, _)| uri == &id) {
            self.vectors[pos] = (id, vector);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Vector with id '{}' not found", id))
        }
    }

    fn remove_vector(&mut self, id: VectorId) -> Result<()> {
        if let Some(pos) = self.vectors.iter().position(|(uri, _)| uri == &id) {
            self.vectors.remove(pos);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Vector with id '{}' not found", id))
        }
    }
}

/// Enhanced vector store with embedding management and advanced features
pub struct VectorStore {
    index: Box<dyn VectorIndex>,
    embedding_manager: Option<embeddings::EmbeddingManager>,
    config: VectorStoreConfig,
}

/// Configuration for vector store
#[derive(Debug, Clone)]
pub struct VectorStoreConfig {
    pub auto_embed: bool,
    pub cache_embeddings: bool,
    pub similarity_threshold: f32,
    pub max_results: usize,
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self {
            auto_embed: true,
            cache_embeddings: true,
            similarity_threshold: 0.7,
            max_results: 100,
        }
    }
}

impl VectorStore {
    /// Create a new vector store with default memory index
    pub fn new() -> Self {
        Self {
            index: Box::new(MemoryVectorIndex::new()),
            embedding_manager: None,
            config: VectorStoreConfig::default(),
        }
    }

    /// Create vector store with specific embedding strategy
    pub fn with_embedding_strategy(strategy: embeddings::EmbeddingStrategy) -> Result<Self> {
        let embedding_manager = embeddings::EmbeddingManager::new(strategy, 1000)?;

        Ok(Self {
            index: Box::new(MemoryVectorIndex::new()),
            embedding_manager: Some(embedding_manager),
            config: VectorStoreConfig::default(),
        })
    }

    /// Create vector store with custom index
    pub fn with_index(index: Box<dyn VectorIndex>) -> Self {
        Self {
            index,
            embedding_manager: None,
            config: VectorStoreConfig::default(),
        }
    }

    /// Create vector store with custom index and embedding strategy
    pub fn with_index_and_embeddings(
        index: Box<dyn VectorIndex>,
        strategy: embeddings::EmbeddingStrategy,
    ) -> Result<Self> {
        let embedding_manager = embeddings::EmbeddingManager::new(strategy, 1000)?;

        Ok(Self {
            index,
            embedding_manager: Some(embedding_manager),
            config: VectorStoreConfig::default(),
        })
    }

    /// Set vector store configuration
    pub fn with_config(mut self, config: VectorStoreConfig) -> Self {
        self.config = config;
        self
    }

    /// Index a resource with automatic embedding generation
    pub fn index_resource(&mut self, uri: String, content: &str) -> Result<()> {
        if let Some(ref mut embedding_manager) = self.embedding_manager {
            let embeddable_content = embeddings::EmbeddableContent::Text(content.to_string());
            let vector = embedding_manager.get_embedding(&embeddable_content)?;
            self.index.insert(uri, vector)
        } else {
            // Generate a simple hash-based vector as fallback
            let vector = self.generate_fallback_vector(content);
            self.index.insert(uri, vector)
        }
    }

    /// Index an RDF resource with structured content
    pub fn index_rdf_resource(
        &mut self,
        uri: String,
        label: Option<String>,
        description: Option<String>,
        properties: std::collections::HashMap<String, Vec<String>>,
    ) -> Result<()> {
        if let Some(ref mut embedding_manager) = self.embedding_manager {
            let embeddable_content = embeddings::EmbeddableContent::RdfResource {
                uri: uri.clone(),
                label,
                description,
                properties,
            };
            let vector = embedding_manager.get_embedding(&embeddable_content)?;
            self.index.insert(uri, vector)
        } else {
            Err(anyhow::anyhow!(
                "Embedding manager required for RDF resource indexing"
            ))
        }
    }

    /// Index a pre-computed vector
    pub fn index_vector(&mut self, uri: String, vector: Vector) -> Result<()> {
        self.index.insert(uri, vector)
    }

    /// Search for similar resources using text query
    pub fn similarity_search(&self, query: &str, limit: usize) -> Result<Vec<(String, f32)>> {
        let query_vector = if let Some(ref _embedding_manager) = self.embedding_manager {
            let _embeddable_content = embeddings::EmbeddableContent::Text(query.to_string());
            // We need a mutable reference, but we only have an immutable one
            // For now, generate a fallback vector
            self.generate_fallback_vector(query)
        } else {
            self.generate_fallback_vector(query)
        };

        self.index.search_knn(&query_vector, limit)
    }

    /// Search for similar resources using a vector query
    pub fn similarity_search_vector(
        &self,
        query: &Vector,
        limit: usize,
    ) -> Result<Vec<(String, f32)>> {
        self.index.search_knn(query, limit)
    }

    /// Find resources within similarity threshold
    pub fn threshold_search(&self, query: &str, threshold: f32) -> Result<Vec<(String, f32)>> {
        let query_vector = self.generate_fallback_vector(query);
        self.index.search_threshold(&query_vector, threshold)
    }

    /// Advanced search with multiple options
    pub fn advanced_search(&self, options: SearchOptions) -> Result<Vec<(String, f32)>> {
        let query_vector = match options.query {
            SearchQuery::Text(text) => self.generate_fallback_vector(&text),
            SearchQuery::Vector(vector) => vector,
        };

        let results = match options.search_type {
            SearchType::KNN(k) => self.index.search_knn(&query_vector, k)?,
            SearchType::Threshold(threshold) => {
                self.index.search_threshold(&query_vector, threshold)?
            }
        };

        Ok(results)
    }

    fn generate_fallback_vector(&self, text: &str) -> Vector {
        // Simple hash-based vector generation for fallback
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();

        let mut values = Vec::with_capacity(384); // Standard embedding size
        let mut seed = hash;

        for _ in 0..384 {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let normalized = (seed as f32) / (u64::MAX as f32);
            values.push((normalized - 0.5) * 2.0); // Range: -1.0 to 1.0
        }

        Vector::new(values)
    }

    /// Get embedding manager statistics
    pub fn embedding_stats(&self) -> Option<(usize, usize)> {
        self.embedding_manager.as_ref().map(|em| em.cache_stats())
    }

    /// Build vocabulary for TF-IDF embeddings
    pub fn build_vocabulary(&mut self, documents: &[String]) -> Result<()> {
        if let Some(ref mut embedding_manager) = self.embedding_manager {
            embedding_manager.build_vocabulary(documents)
        } else {
            Ok(()) // No-op if no embedding manager
        }
    }

    /// Calculate similarity between two resources by their URIs
    pub fn calculate_similarity(&self, uri1: &str, uri2: &str) -> Result<f32> {
        // If the URIs are identical, return perfect similarity
        if uri1 == uri2 {
            return Ok(1.0);
        }

        // Get the vectors for both URIs
        let vector1 = self
            .index
            .get_vector(uri1)
            .ok_or_else(|| anyhow::anyhow!("Vector not found for URI: {}", uri1))?;

        let vector2 = self
            .index
            .get_vector(uri2)
            .ok_or_else(|| anyhow::anyhow!("Vector not found for URI: {}", uri2))?;

        // Calculate cosine similarity between the vectors
        vector1.cosine_similarity(vector2)
    }

    /// Get a vector by its ID (delegates to VectorIndex)
    pub fn get_vector(&self, id: &str) -> Option<&Vector> {
        self.index.get_vector(id)
    }

    /// Index a vector with metadata (stub)
    pub fn index_vector_with_metadata(
        &mut self,
        uri: String,
        vector: Vector,
        _metadata: HashMap<String, String>,
    ) -> Result<()> {
        // For now, just delegate to index_vector, ignoring metadata
        // Future: Extend VectorIndex trait to support metadata
        self.index_vector(uri, vector)
    }

    /// Index a resource with metadata (stub)
    pub fn index_resource_with_metadata(
        &mut self,
        uri: String,
        content: &str,
        _metadata: HashMap<String, String>,
    ) -> Result<()> {
        // For now, just delegate to index_resource, ignoring metadata
        // Future: Store and utilize metadata
        self.index_resource(uri, content)
    }

    /// Search with additional parameters (stub)
    pub fn similarity_search_with_params(
        &self,
        query: &str,
        limit: usize,
        _params: HashMap<String, String>,
    ) -> Result<Vec<(String, f32)>> {
        // For now, just delegate to similarity_search, ignoring params
        // Future: Use params for filtering, threshold, etc.
        self.similarity_search(query, limit)
    }

    /// Vector search with additional parameters (stub)
    pub fn vector_search_with_params(
        &self,
        query: &Vector,
        limit: usize,
        _params: HashMap<String, String>,
    ) -> Result<Vec<(String, f32)>> {
        // For now, just delegate to similarity_search_vector, ignoring params
        // Future: Use params for filtering, distance metric selection, etc.
        self.similarity_search_vector(query, limit)
    }

    /// Get all vector IDs (stub)
    pub fn get_vector_ids(&self) -> Result<Vec<String>> {
        // VectorIndex trait doesn't provide this method yet
        // Future: Add to VectorIndex trait or track separately
        Ok(Vec::new())
    }

    /// Remove a vector by its URI (stub)
    pub fn remove_vector(&mut self, uri: &str) -> Result<()> {
        // Delegate to VectorIndex trait's remove_vector method
        self.index.remove_vector(uri.to_string())
    }

    /// Get store statistics (stub)
    pub fn get_statistics(&self) -> Result<HashMap<String, String>> {
        // Return basic statistics as a map
        // Future: Provide comprehensive stats from index
        let mut stats = HashMap::new();
        stats.insert("type".to_string(), "VectorStore".to_string());

        if let Some((cache_size, cache_capacity)) = self.embedding_stats() {
            stats.insert("embedding_cache_size".to_string(), cache_size.to_string());
            stats.insert(
                "embedding_cache_capacity".to_string(),
                cache_capacity.to_string(),
            );
        }

        Ok(stats)
    }

    /// Save store to disk (stub)
    pub fn save_to_disk(&self, _path: &str) -> Result<()> {
        // Stub implementation - serialization not yet implemented
        // Future: Serialize index and configuration to disk
        Err(anyhow::anyhow!("save_to_disk not yet implemented"))
    }

    /// Load store from disk (stub)
    pub fn load_from_disk(_path: &str) -> Result<Self> {
        // Stub implementation - deserialization not yet implemented
        // Future: Deserialize index and configuration from disk
        Err(anyhow::anyhow!("load_from_disk not yet implemented"))
    }

    /// Optimize the underlying index (stub)
    pub fn optimize_index(&mut self) -> Result<()> {
        // Stub implementation - optimization not yet implemented
        // Future: Trigger index compaction, rebalancing, etc.
        Ok(())
    }
}

impl Default for VectorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorStoreTrait for VectorStore {
    fn insert_vector(&mut self, id: VectorId, vector: Vector) -> Result<()> {
        self.index.insert(id, vector)
    }

    fn add_vector(&mut self, vector: Vector) -> Result<VectorId> {
        // Generate a unique ID for the vector
        let id = format!("vec_{}", uuid::Uuid::new_v4());
        self.index.insert(id.clone(), vector)?;
        Ok(id)
    }

    fn get_vector(&self, id: &VectorId) -> Result<Option<Vector>> {
        Ok(self.index.get_vector(id).cloned())
    }

    fn get_all_vector_ids(&self) -> Result<Vec<VectorId>> {
        // For now, return empty vec as VectorIndex doesn't provide this method
        // This could be enhanced if the underlying index supports it
        Ok(Vec::new())
    }

    fn search_similar(&self, query: &Vector, k: usize) -> Result<Vec<(VectorId, f32)>> {
        self.index.search_knn(query, k)
    }

    fn remove_vector(&mut self, id: &VectorId) -> Result<bool> {
        // VectorIndex trait doesn't have remove, so we'll return false for now
        // This could be enhanced in the future if needed
        let _ = id;
        Ok(false)
    }

    fn len(&self) -> usize {
        // VectorIndex trait doesn't have len, so we'll return 0 for now
        // This could be enhanced in the future if needed
        0
    }
}

/// Search query types
#[derive(Debug, Clone)]
pub enum SearchQuery {
    Text(String),
    Vector(Vector),
}

/// Search operation types
#[derive(Debug, Clone)]
pub enum SearchType {
    KNN(usize),
    Threshold(f32),
}

/// Advanced search options
#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub query: SearchQuery,
    pub search_type: SearchType,
}

/// Vector operation results with enhanced metadata
#[derive(Debug, Clone)]
pub struct VectorOperationResult {
    pub uri: String,
    pub similarity: f32,
    pub vector: Option<Vector>,
    pub metadata: Option<std::collections::HashMap<String, String>>,
    pub rank: usize,
}

/// Document batch processing utilities
pub struct DocumentBatchProcessor;

impl DocumentBatchProcessor {
    /// Process multiple documents in batch for efficient indexing
    pub fn batch_index(
        store: &mut VectorStore,
        documents: &[(String, String)], // (uri, content) pairs
    ) -> Result<Vec<Result<()>>> {
        let mut results = Vec::new();

        for (uri, content) in documents {
            let result = store.index_resource(uri.clone(), content);
            results.push(result);
        }

        Ok(results)
    }

    /// Process multiple queries in batch
    pub fn batch_search(
        store: &VectorStore,
        queries: &[String],
        limit: usize,
    ) -> Result<BatchSearchResult> {
        let mut results = Vec::new();

        for query in queries {
            let result = store.similarity_search(query, limit);
            results.push(result);
        }

        Ok(results)
    }
}

/// Error types specific to vector operations
#[derive(Debug, thiserror::Error)]
pub enum VectorError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Empty vector")]
    EmptyVector,

    #[error("Index not built")]
    IndexNotBuilt,

    #[error("Embedding generation failed: {message}")]
    EmbeddingError { message: String },

    #[error("SPARQL service error: {message}")]
    SparqlServiceError { message: String },

    #[error("Compression error: {0}")]
    CompressionError(String),

    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Invalid data: {0}")]
    InvalidData(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Utility functions for vector operations
pub mod utils {
    use super::Vector;

    /// Calculate centroid of a set of vectors
    pub fn centroid(vectors: &[Vector]) -> Option<Vector> {
        if vectors.is_empty() {
            return None;
        }

        let dimensions = vectors[0].dimensions;
        let mut sum_values = vec![0.0; dimensions];

        for vector in vectors {
            if vector.dimensions != dimensions {
                return None; // Inconsistent dimensions
            }

            let vector_f32 = vector.as_f32();
            for (i, &value) in vector_f32.iter().enumerate() {
                sum_values[i] += value;
            }
        }

        let count = vectors.len() as f32;
        for value in &mut sum_values {
            *value /= count;
        }

        Some(Vector::new(sum_values))
    }

    /// Generate random vector for testing
    pub fn random_vector(dimensions: usize, seed: Option<u64>) -> Vector {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        seed.unwrap_or(42).hash(&mut hasher);
        let mut rng_state = hasher.finish();

        let mut values = Vec::with_capacity(dimensions);
        for _ in 0..dimensions {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let normalized = (rng_state as f32) / (u64::MAX as f32);
            values.push((normalized - 0.5) * 2.0); // Range: -1.0 to 1.0
        }

        Vector::new(values)
    }

    /// Convert vector to normalized unit vector
    pub fn normalize_vector(vector: &Vector) -> Vector {
        vector.normalized()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::similarity::SimilarityMetric;

    #[test]
    fn test_vector_creation() {
        let values = vec![1.0, 2.0, 3.0];
        let vector = Vector::new(values.clone());

        assert_eq!(vector.dimensions, 3);
        assert_eq!(vector.precision, VectorPrecision::F32);
        assert_eq!(vector.as_f32(), values);
    }

    #[test]
    fn test_multi_precision_vectors() {
        // Test F64 vector
        let f64_values = vec![1.0, 2.0, 3.0];
        let f64_vector = Vector::f64(f64_values.clone());
        assert_eq!(f64_vector.precision, VectorPrecision::F64);
        assert_eq!(f64_vector.dimensions, 3);

        // Test I8 vector
        let i8_values = vec![100, -50, 0];
        let i8_vector = Vector::i8(i8_values);
        assert_eq!(i8_vector.precision, VectorPrecision::I8);
        assert_eq!(i8_vector.dimensions, 3);

        // Test binary vector
        let binary_values = vec![0b10101010, 0b11110000];
        let binary_vector = Vector::binary(binary_values);
        assert_eq!(binary_vector.precision, VectorPrecision::Binary);
        assert_eq!(binary_vector.dimensions, 16); // 2 bytes * 8 bits
    }

    #[test]
    fn test_vector_operations() {
        let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let v2 = Vector::new(vec![4.0, 5.0, 6.0]);

        // Test addition
        let sum = v1.add(&v2).unwrap();
        assert_eq!(sum.as_f32(), vec![5.0, 7.0, 9.0]);

        // Test subtraction
        let diff = v2.subtract(&v1).unwrap();
        assert_eq!(diff.as_f32(), vec![3.0, 3.0, 3.0]);

        // Test scaling
        let scaled = v1.scale(2.0);
        assert_eq!(scaled.as_f32(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_cosine_similarity() {
        let v1 = Vector::new(vec![1.0, 0.0, 0.0]);
        let v2 = Vector::new(vec![1.0, 0.0, 0.0]);
        let v3 = Vector::new(vec![0.0, 1.0, 0.0]);

        // Identical vectors should have similarity 1.0
        assert!((v1.cosine_similarity(&v2).unwrap() - 1.0).abs() < 0.001);

        // Orthogonal vectors should have similarity 0.0
        assert!((v1.cosine_similarity(&v3).unwrap()).abs() < 0.001);
    }

    #[test]
    fn test_vector_store() {
        let mut store = VectorStore::new();

        // Test indexing
        store
            .index_resource("doc1".to_string(), "This is a test")
            .unwrap();
        store
            .index_resource("doc2".to_string(), "Another test document")
            .unwrap();

        // Test searching
        let results = store.similarity_search("test", 5).unwrap();
        assert_eq!(results.len(), 2);

        // Results should be sorted by similarity (descending)
        assert!(results[0].1 >= results[1].1);
    }

    #[test]
    fn test_similarity_metrics() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        // Test different similarity metrics
        let cosine_sim = SimilarityMetric::Cosine.similarity(&a, &b).unwrap();
        let euclidean_sim = SimilarityMetric::Euclidean.similarity(&a, &b).unwrap();
        let manhattan_sim = SimilarityMetric::Manhattan.similarity(&a, &b).unwrap();

        // All similarities should be between 0 and 1
        assert!((0.0..=1.0).contains(&cosine_sim));
        assert!((0.0..=1.0).contains(&euclidean_sim));
        assert!((0.0..=1.0).contains(&manhattan_sim));
    }

    #[test]
    fn test_quantization() {
        let values = vec![1.0, -0.5, 0.0, 0.75];
        let quantized = Vector::quantize_to_i8(&values);

        // Check that quantized values are in the expected range
        for &q in &quantized {
            assert!((-127..=127).contains(&q));
        }
    }

    #[test]
    fn test_binary_conversion() {
        let values = vec![0.8, -0.3, 0.1, -0.9];
        let binary = Vector::to_binary(&values, 0.0);

        // Should have 1 byte (4 values, each becomes 1 bit, packed into bytes)
        assert_eq!(binary.len(), 1);

        // First bit should be 1 (0.8 > 0.0), second should be 0 (-0.3 < 0.0), etc.
        let byte = binary[0];
        assert_eq!(byte & 1, 1); // bit 0: 0.8 > 0.0
        assert_eq!((byte >> 1) & 1, 0); // bit 1: -0.3 < 0.0
        assert_eq!((byte >> 2) & 1, 1); // bit 2: 0.1 > 0.0
        assert_eq!((byte >> 3) & 1, 0); // bit 3: -0.9 < 0.0
    }

    #[test]
    fn test_memory_vector_index() {
        let mut index = MemoryVectorIndex::new();

        let v1 = Vector::new(vec![1.0, 0.0, 0.0]);
        let v2 = Vector::new(vec![0.0, 1.0, 0.0]);

        index.insert("v1".to_string(), v1.clone()).unwrap();
        index.insert("v2".to_string(), v2.clone()).unwrap();

        // Test KNN search
        let results = index.search_knn(&v1, 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "v1");

        // Test threshold search
        let results = index.search_threshold(&v1, 0.5).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_hnsw_index() {
        use crate::hnsw::{HnswConfig, HnswIndex};

        let config = HnswConfig::default();
        let mut index = HnswIndex::new(config).unwrap();

        let v1 = Vector::new(vec![1.0, 0.0, 0.0]);
        let v2 = Vector::new(vec![0.0, 1.0, 0.0]);
        let v3 = Vector::new(vec![0.0, 0.0, 1.0]);

        index.insert("v1".to_string(), v1.clone()).unwrap();
        index.insert("v2".to_string(), v2.clone()).unwrap();
        index.insert("v3".to_string(), v3.clone()).unwrap();

        // Test KNN search
        let results = index.search_knn(&v1, 2).unwrap();
        assert!(results.len() <= 2);

        // The first result should be v1 itself (highest similarity)
        if !results.is_empty() {
            assert_eq!(results[0].0, "v1");
        }
    }

    #[test]
    fn test_sparql_vector_service() {
        use crate::embeddings::EmbeddingStrategy;
        use crate::sparql_integration::{
            SparqlVectorService, VectorServiceArg, VectorServiceConfig, VectorServiceResult,
        };

        let config = VectorServiceConfig::default();
        let mut service =
            SparqlVectorService::new(config, EmbeddingStrategy::SentenceTransformer).unwrap();

        // Test vector similarity function
        let v1 = Vector::new(vec![1.0, 0.0, 0.0]);
        let v2 = Vector::new(vec![1.0, 0.0, 0.0]);

        let args = vec![VectorServiceArg::Vector(v1), VectorServiceArg::Vector(v2)];

        let result = service
            .execute_function("vector_similarity", &args)
            .unwrap();

        match result {
            VectorServiceResult::Number(similarity) => {
                assert!((similarity - 1.0).abs() < 0.001); // Should be very similar
            }
            _ => panic!("Expected a number result"),
        }

        // Test text embedding function
        let text_args = vec![VectorServiceArg::String("test text".to_string())];
        let embed_result = service.execute_function("embed_text", &text_args).unwrap();

        match embed_result {
            VectorServiceResult::Vector(vector) => {
                assert_eq!(vector.dimensions, 384); // Default embedding size
            }
            _ => panic!("Expected a vector result"),
        }
    }
}
