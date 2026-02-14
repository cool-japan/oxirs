//! # OxiRS Stream - Ultra-High Performance RDF Streaming Platform
//!
//! [![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/cool-japan/oxirs/releases)
//! [![docs.rs](https://docs.rs/oxirs-stream/badge.svg)](https://docs.rs/oxirs-stream)
//!
//! **Status**: Production Release (v0.1.0)
//! **Stability**: Public APIs are stable. Production-ready with comprehensive testing.
//!
//! Real-time streaming support with Kafka/NATS/Redis I/O, RDF Patch, SPARQL Update delta,
//! and advanced event processing capabilities.
//!
//! This crate provides enterprise-grade real-time data streaming capabilities for RDF datasets,
//! supporting multiple messaging backends with high-throughput, low-latency guarantees.
//!
//! ## Features
//! - **Multi-Backend Support**: Kafka, NATS JetStream, Redis Streams, AWS Kinesis, Memory
//! - **High Performance**: 100K+ events/second, <10ms latency, exactly-once delivery
//! - **Advanced Event Processing**: Real-time pattern detection, windowing, aggregations
//! - **Enterprise Features**: Circuit breakers, connection pooling, health monitoring
//! - **Standards Compliance**: RDF Patch protocol, SPARQL Update streaming
//!
//! ## Performance Targets
//! - **Throughput**: 100K+ events/second sustained
//! - **Latency**: P99 <10ms for real-time processing
//! - **Reliability**: 99.99% delivery success rate
//! - **Scalability**: Linear scaling to 1000+ partitions

#![allow(dead_code)]

/// Re-export commonly used types for convenience
pub use backend_optimizer::{
    BackendOptimizer, BackendPerformance, BackendRecommendation, ConsistencyLevel, CostModel,
    OptimizationDecision, OptimizationStats, OptimizerConfig, PatternType, WorkloadPattern,
};
pub use backpressure::{
    BackpressureConfig, BackpressureController, BackpressureStats, BackpressureStrategy,
    FlowControlSignal, RateLimiter as BackpressureRateLimiter,
};
pub use bridge::{
    BridgeInfo, BridgeStatistics, BridgeType, ExternalMessage, ExternalSystemConfig,
    ExternalSystemType, MessageBridgeManager, MessageTransformer, RoutingRule,
};
pub use circuit_breaker::{
    CircuitBreakerError, CircuitBreakerMetrics, FailureType, SharedCircuitBreakerExt,
};
pub use connection_pool::{
    ConnectionFactory, ConnectionPool, DetailedPoolMetrics, LoadBalancingStrategy, PoolConfig,
    PoolStats, PoolStatus,
};
pub use cqrs::{
    CQRSConfig, CQRSHealthStatus, CQRSSystem, Command, CommandBus, CommandBusMetrics,
    CommandHandler, CommandResult, Query, QueryBus, QueryBusMetrics, QueryCacheConfig,
    QueryHandler, QueryResult as CQRSQueryResult, ReadModelManager, ReadModelMetrics,
    ReadModelProjection, RetryConfig as CQRSRetryConfig,
};
pub use delta::{BatchDeltaProcessor, DeltaComputer, DeltaProcessor, ProcessorStats};
pub use dlq::{
    DeadLetterQueue, DlqConfig, DlqEventProcessor, DlqStats as DlqStatsExport, FailedEvent,
    FailureReason,
};
pub use event::{
    EventCategory, EventMetadata, EventPriority, IsolationLevel, QueryResult as EventQueryResult,
    SchemaChangeType, SchemaType, SparqlOperationType, StreamEvent,
};
pub use event_sourcing::{
    EventQuery, EventSnapshot, EventStore, EventStoreConfig, PersistenceBackend, QueryOrder,
    RetentionPolicy, SnapshotConfig, StoredEvent, TimeRange as EventSourcingTimeRange,
};
pub use failover::{ConnectionEndpoint, FailoverConfig, FailoverManager};
pub use graphql_bridge::{
    BridgeConfig, BridgeStats, GraphQLBridge, GraphQLSubscription, GraphQLUpdate,
    GraphQLUpdateType, SubscriptionFilter,
};
pub use multi_region_replication::{
    ConflictResolution, ConflictType, GeographicLocation, MultiRegionReplicationManager,
    RegionConfig, RegionHealth, ReplicatedEvent, ReplicationConfig, ReplicationStats,
    ReplicationStrategy, VectorClock,
};
pub use patch::{PatchParser, PatchSerializer};
pub use performance_optimizer::{
    AdaptiveBatcher, AggregationFunction, AutoTuner, BatchPerformancePoint, BatchSizePredictor,
    BatchingStats, EnhancedMLConfig, MemoryPool, MemoryPoolStats,
    PerformanceConfig as OptimizerPerformanceConfig, ProcessingResult, ProcessingStats,
    ProcessingStatus, TuningDecision, ZeroCopyEvent,
};
pub use schema_registry::{
    CompatibilityMode, ExternalRegistryConfig, RegistryAuth, SchemaDefinition, SchemaFormat,
    SchemaRegistry, SchemaRegistryConfig, ValidationResult, ValidationStats,
};
pub use sparql_streaming::{
    ContinuousQueryManager, QueryManagerConfig, QueryMetadata, QueryResultChannel,
    QueryResultUpdate, UpdateType,
};
pub use store_integration::{
    ChangeDetectionStrategy, ChangeNotification, RealtimeUpdateManager, StoreChangeDetector,
    UpdateChannel, UpdateFilter, UpdateNotification,
};

// Stream, StreamConsumer, and StreamProducer are defined below in this module
pub use biological_computing::{
    AminoAcid, BiologicalProcessingStats, BiologicalStreamProcessor, Cell, CellState,
    CellularAutomaton, ComputationalFunction, DNASequence, EvolutionaryOptimizer, FunctionalDomain,
    Individual, Nucleotide, ProteinStructure, SequenceMetadata,
};
pub use consciousness_streaming::{
    ConsciousnessLevel, ConsciousnessStats, ConsciousnessStreamProcessor, DreamSequence,
    EmotionalContext, IntuitiveEngine, MeditationState,
};
pub use disaster_recovery::{
    BackupCompression, BackupConfig, BackupEncryption, BackupFrequency, BackupJob,
    BackupRetentionPolicy, BackupSchedule, BackupStatus, BackupStorage, BackupType,
    BackupVerification, BackupVerificationResult, BackupWindow, BusinessContinuityConfig,
    ChecksumAlgorithm, CompressionAlgorithm, DRMetrics, DisasterRecoveryConfig,
    DisasterRecoveryManager, DisasterScenario, EncryptionAlgorithm as BackupEncryptionAlgorithm,
    FailoverConfig as DRFailoverConfig, ImpactLevel, KeyDerivationFunction, RecoveryConfig,
    RecoveryOperation, RecoveryPriority, RecoveryRunbook, RecoveryStatus, RecoveryType,
    ReplicationConfig as DRReplicationConfig, ReplicationMode as DRReplicationMode,
    ReplicationTarget as DRReplicationTarget, RunbookExecution, RunbookExecutionStatus,
    RunbookStep, StorageLocation,
};
pub use enterprise_audit::{
    ActionResult, AuditEncryptionConfig, AuditEventType, AuditFilterConfig, AuditMetrics,
    AuditRetentionConfig, AuditSeverity, AuditStorageBackend, AuditStorageConfig,
    AuditStreamingConfig, AuthType, ComplianceConfig, ComplianceFinding, ComplianceReport,
    ComplianceStandard, CompressionType as AuditCompressionType, DestinationAuth, DestinationType,
    EncryptionAlgorithm, EnterpriseAuditConfig, EnterpriseAuditEvent, EnterpriseAuditLogger,
    FindingType, KeyManagementConfig, KmsType, S3AuditConfig, StreamingDestination,
};
pub use enterprise_monitoring::{
    Alert, AlertCondition, AlertManager, AlertRule, AlertSeverity as MonitoringAlertSeverity,
    AlertingConfig, BreachNotificationConfig, ComparisonOperator, EnterpriseMonitoringConfig,
    EnterpriseMonitoringSystem, EscalationLevel, EscalationPolicy, HealthCheckConfig,
    HealthCheckEndpoint, HealthCheckType, MeasurementWindow, MetricDefinition, MetricType,
    MetricValue, MetricsCollector, MetricsConfig, MetricsEndpoint, MetricsEndpointType,
    MetricsExportConfig, MetricsFormat, NotificationChannel, ProfilingConfig, SlaBreach, SlaConfig,
    SlaMeasurement, SlaMetricType, SlaObjective, SlaSeverity, SlaStatus, SlaTracker,
};
pub use multi_tenancy::{
    IsolationMode, MultiTenancyConfig, MultiTenancyManager, MultiTenancyMetrics,
    NamespaceResources, ResourceAllocationStrategy, ResourceType, ResourceUsage, Tenant,
    TenantLifecycleConfig, TenantNamespace, TenantQuota, TenantStatus, TenantTier,
};
pub use observability::{
    AlertConfig, AlertEvent, AlertSeverity, AlertType, BusinessMetrics, SpanLog, SpanStatus,
    StreamObservability, StreamingMetrics, TelemetryConfig, TraceSpan,
};
pub use performance_utils::{
    AdaptiveRateLimiter, IntelligentMemoryPool, IntelligentPrefetcher, ParallelStreamProcessor,
    PerformanceUtilsConfig,
};
pub use quantum_communication::{
    BellState, EntanglementDistribution, QuantumCommConfig, QuantumCommSystem,
    QuantumOperation as QuantumCommOperation, QuantumSecurityProtocol,
    QuantumState as QuantumCommState, Qubit,
};
pub use quantum_streaming::{
    QuantumEvent, QuantumOperation, QuantumProcessingStats, QuantumState, QuantumStreamProcessor,
};
pub use reliability::{BulkReplayResult, DlqStats, ReplayStatus};
pub use rsp::{
    RspConfig, RspLanguage, RspProcessor, RspQuery, StreamClause, StreamDescriptor, Window,
    WindowConfig, WindowSize, WindowStats, WindowType,
};
pub use security::{
    AuditConfig, AuditLogEntry, AuditLogger, AuthConfig, AuthMethod, AuthenticationProvider,
    AuthorizationProvider, AuthzConfig, Credentials, EncryptionConfig, Permission, RateLimitConfig,
    RateLimiter, SecurityConfig as StreamSecurityConfig, SecurityContext, SecurityManager,
    SecurityMetrics, SessionConfig, ThreatAlert, ThreatDetectionConfig, ThreatDetector,
};
pub use temporal_join::{
    IntervalJoin, JoinResult, LateDataConfig, LateDataStrategy, TemporalJoin, TemporalJoinConfig,
    TemporalJoinMetrics, TemporalJoinType, TemporalWindow, TimeSemantics, WatermarkConfig,
    WatermarkStrategy,
};
pub use time_travel::{
    AggregationType, TemporalAggregations, TemporalFilter, TemporalOrdering, TemporalProjection,
    TemporalQuery, TemporalQueryResult, TemporalResultMetadata, TemporalStatistics, TimePoint,
    TimeRange as TimeTravelTimeRange, TimeTravelConfig, TimeTravelEngine, TimeTravelMetrics,
    TimelinePoint,
};
pub use tls_security::{
    CertRotationConfig, CertificateConfig, CertificateFormat, CertificateInfo, CipherSuite,
    ExpiryWarning, MutualTlsConfig, OcspConfig, RevocationCheckConfig, SessionResumptionConfig,
    TlsConfig, TlsManager, TlsMetrics, TlsSessionInfo, TlsVersion,
};
pub use wasm_edge_computing::{
    EdgeExecutionResult, EdgeLocation, OptimizationLevel, PerformanceProfile, PluginCapability,
    PluginSchema, ProcessingSpecialization, ResourceMetrics, SecurityLevel, WasmEdgeConfig,
    WasmEdgeProcessor, WasmPlugin, WasmProcessingResult, WasmProcessorStats, WasmResourceLimits,
};
pub use webhook::{
    EventFilter as WebhookEventFilter, HttpMethod, RateLimit, RetryConfig as WebhookRetryConfig,
    WebhookConfig, WebhookInfo, WebhookManager, WebhookMetadata, WebhookSecurity,
    WebhookStatistics,
};

// New v0.1.0 feature exports
pub use custom_serialization::{
    BenchmarkResults, BsonSerializer, CustomSerializer, FlexBuffersSerializer, IonSerializer,
    RonSerializer, SerializerBenchmark, SerializerBenchmarkSuite, SerializerRegistry,
    SerializerStats, ThriftSerializer,
};
pub use end_to_end_encryption::{
    E2EEConfig, E2EEEncryptionAlgorithm, E2EEManager, E2EEStats, EncryptedMessage,
    HomomorphicEncryption, KeyExchangeAlgorithm, KeyPair, KeyRotationConfig, MultiPartyConfig,
    ZeroKnowledgeProof,
};
pub use gpu_acceleration::{
    AggregationOp, GpuBackend, GpuBuffer, GpuConfig, GpuContext, GpuProcessorConfig, GpuStats,
    GpuStreamProcessor,
};
pub use ml_integration::{
    AnomalyDetectionAlgorithm, AnomalyDetectionConfig, AnomalyDetector, AnomalyResult,
    AnomalyStats, FeatureConfig, FeatureExtractor, FeatureVector, MLIntegrationManager,
    MLModelConfig, ModelMetrics, ModelType, OnlineLearningModel, PredictionResult,
};
pub use rate_limiting::{
    QuotaCheckResult, QuotaEnforcementMode, QuotaLimits, QuotaManager, QuotaOperation,
    RateLimitAlgorithm, RateLimitConfig as AdvancedRateLimitConfig, RateLimitMonitoringConfig,
    RateLimitStats as AdvancedRateLimitStats, RateLimiter as AdvancedRateLimiter,
    RejectionStrategy,
};
pub use scalability::{
    AdaptiveBuffer, AutoScaler, LoadBalancingStrategy as ScalingLoadBalancingStrategy,
    Node as ScalingNode, NodeHealth, Partition, PartitionManager, PartitionStrategy,
    ResourceLimits, ResourceUsage as ScalingResourceUsage, ScalingConfig, ScalingDirection,
    ScalingMode,
};
pub use schema_evolution::{
    CompatibilityCheckResult, CompatibilityIssue, CompatibilityIssueType,
    CompatibilityMode as SchemaCompatibilityMode, DeprecationInfo, EvolutionResult,
    FieldDefinition, FieldType, IssueSeverity, MigrationRule, MigrationStrategy, SchemaChange,
    SchemaDefinition as SchemaEvolutionDefinition, SchemaEvolutionManager,
    SchemaFormat as SchemaEvolutionFormat, SchemaVersion,
};
pub use stream_replay::{
    EventProcessor, ReplayCheckpoint, ReplayConfig, ReplayFilter, ReplayMode, ReplaySpeed,
    ReplayStats, ReplayStatus as StreamReplayStatus, ReplayTransformation, StateSnapshot,
    StreamReplayManager, TransformationType,
};
pub use transactional_processing::{
    IsolationLevel as TransactionalIsolationLevel, LogEntryType, TransactionCheckpoint,
    TransactionLogEntry, TransactionMetadata, TransactionState, TransactionalConfig,
    TransactionalProcessor, TransactionalStats,
};
pub use zero_copy::{
    MemoryMappedBuffer, SharedRefBuffer, SimdBatchProcessor, SimdOperation, SplicedBuffer,
    ZeroCopyBuffer, ZeroCopyConfig, ZeroCopyManager, ZeroCopyStats,
};

// New v0.1.0 exports for developer experience and performance
pub use numa_processing::{
    CpuAffinityMode, HugePageSize, MemoryBandwidthMonitor, MemoryInterleavePolicy, NodeBufferStats,
    NodeProcessorStats, NumaAllocationStrategy, NumaBuffer, NumaBufferPool, NumaBufferPoolConfig,
    NumaBufferPoolStats, NumaConfig, NumaNode, NumaProcessorStats, NumaStreamProcessor,
    NumaThreadPool, NumaThreadPoolStats, NumaTopology, NumaWorker, NumaWorkerStats,
    WorkerDistributionStrategy,
};
pub use out_of_order::{
    EmitStrategy, GapFillingStrategy, LateEventStrategy, OrderedEvent, OutOfOrderConfig,
    OutOfOrderHandler, OutOfOrderHandlerBuilder, OutOfOrderStats, SequenceTracker, Watermark,
};
pub use performance_profiler::{
    HistogramStats, LatencyHistogram, OperationTimer, PerformanceProfiler, PerformanceReport,
    PerformanceSample, PerformanceWarning, ProfilerBuilder, ProfilerConfig, ProfilerStats,
    Recommendation, RecommendationCategory, RecommendationEffort, RecommendationImpact, Span,
    WarningSeverity, WarningThresholds, WarningType,
};
pub use stream_sql::{
    AggregateFunction, BinaryOperator, ColumnDefinition, CreateStreamStatement, DataType,
    Expression, FromClause, JoinType, Lexer, OrderByItem, Parser,
    QueryResult as StreamSqlQueryResult, QueryType, ResultRow, SelectItem, SelectStatement,
    SqlValue, StreamMetadata, StreamSqlConfig, StreamSqlEngine, StreamSqlStats, Token,
    UnaryOperator, WindowSpec, WindowType as SqlWindowType,
};
pub use testing_framework::{
    Assertion, AssertionType, CapturedEvent, EventGenerator, EventMatcher, GeneratorConfig,
    GeneratorType, MockClock, PerformanceMetric, TestFixture, TestHarness, TestHarnessBuilder,
    TestHarnessConfig, TestMetrics, TestReport, TestStatus,
};

// New v0.1.0 exports for ML, versioning, and migration
pub use anomaly_detection::{
    Anomaly, AnomalyAlert, AnomalyConfig, AnomalyDetector as AdaptiveAnomalyDetector,
    AnomalySeverity, AnomalyStats as AdaptiveAnomalyStats, DetectorType, MultiDimensionalDetector,
};
pub use migration_tools::{
    APIMapping, ConceptMapping, GeneratedFile, GeneratedFileType, ManualReviewItem,
    MigrationConfig, MigrationError, MigrationReport, MigrationSuggestion, MigrationTool,
    MigrationWarning, QuickStart, ReviewPriority, SourcePlatform, SuggestionCategory,
};
pub use online_learning::{
    ABTestConfig, ABTestResult, DriftDetection, ModelCheckpoint,
    ModelMetrics as OnlineModelMetrics, ModelType as OnlineModelType, OnlineLearningConfig,
    OnlineLearningModel as StreamOnlineLearningModel, OnlineLearningStats, Prediction, Sample,
    StreamFeatureExtractor,
};
pub use stream_versioning::{
    Branch, BranchId, Change, ChangeType, Changeset, Snapshot, StreamVersioning, TimeTravelQuery,
    TimeTravelTarget, VersionDiff, VersionId, VersionMetadata, VersionedEvent, VersioningConfig,
    VersioningStats,
};

// New v0.1.0 advanced ML exports
pub use automl_stream::{
    Algorithm, AutoML, AutoMLConfig, AutoMLStats, HyperParameters, ModelPerformance, TaskType,
    TrainedModel,
};
pub use feature_engineering::{
    Feature, FeatureExtractionConfig, FeatureMetadata, FeaturePipeline, FeatureSet, FeatureStore,
    FeatureTransform, FeatureValue, ImputationStrategy, PipelineStats,
};
pub use neural_architecture_search::{
    ActivationType, Architecture, ArchitecturePerformance, LayerType, NASConfig, NASStats,
    ObjectiveWeights, SearchSpace, SearchStrategy, NAS,
};
pub use predictive_analytics::{
    AccuracyMetrics, ForecastAlgorithm, ForecastResult, ForecastingConfig, PredictiveAnalytics,
    PredictiveStats, SeasonalityType, TrendDirection,
};
pub use reinforcement_learning::{
    Action, Experience, RLAgent, RLAlgorithm, RLConfig, RLStats, State as RLState,
};

// Utility exports
pub use utils::{
    create_dev_stream, create_prod_stream, BatchProcessor, EventFilter, EventSampler,
    SimpleRateLimiter, StreamMultiplexer, StreamStats,
};

// Advanced SciRS2 optimization exports
pub use advanced_scirs2_optimization::{
    AdvancedOptimizerConfig, AdvancedStreamOptimizer, MovingStats, OptimizerMetrics,
};
pub use cdc_processor::{
    CdcConfig, CdcConnector, CdcEvent, CdcEventBuilder, CdcMetrics, CdcOperation, CdcProcessor,
    CdcSource,
};

// Adaptive load shedding exports
pub use adaptive_load_shedding::{
    DropStrategy, LoadMetrics, LoadSheddingConfig, LoadSheddingManager, LoadSheddingStats,
};

// Stream fusion optimizer exports
pub use stream_fusion::{
    FusableChain, FusedOperation, FusedType, FusionAnalysis, FusionConfig, FusionOptimizer,
    FusionStats, Operation,
};

// Complex Event Processing (CEP) engine exports
pub use cep_engine::{
    CepAggregationFunction, CepConfig, CepEngine, CepMetrics, CepStatistics, CompleteMatch,
    CorrelationFunction, CorrelationResult, CorrelationStats, DetectedPattern, DetectionAlgorithm,
    DetectionStats, EnrichmentData, EnrichmentService, EnrichmentSource, EnrichmentSourceType,
    EnrichmentStats, EventBuffer, EventCorrelator, EventPattern, FieldPredicate, PartialMatch,
    PatternDetector, ProcessingRule, RuleAction, RuleCondition, RuleEngine, RuleExecutionStats,
    State, StateMachine, TemporalOperator, TimestampedEvent,
};

// Data quality and validation framework exports
pub use data_quality::{
    AlertCondition as QualityAlertCondition, AlertManager as QualityAlertManager,
    AlertRule as QualityAlertRule, AlertSeverity as QualityAlertSeverity,
    AlertStats as QualityAlertStats, AlertType as QualityAlertType, AuditAction, AuditEntry,
    AuditStats, AuditTrail, CleansingRule, CleansingStats, CorrectionType, DataCleanser,
    DataCorrection, DataProfiler, DataQualityValidator, DuplicateDetector, DuplicateStats,
    FailureSeverity, FieldProfile, OutlierMethod, ProfileStats, ProfiledEvent, QualityAlert,
    QualityConfig, QualityDimension, QualityMetrics, QualityReport, QualityScorer, ScoringStats,
    ValidationFailure, ValidationResult as QualityValidationResult, ValidationRule,
};

// Advanced sampling techniques exports
pub use advanced_sampling::{
    AdvancedSamplingManager, BloomFilter, BloomFilterStats, CountMinSketch, CountMinSketchStats,
    HyperLogLog, HyperLogLogStats, ReservoirSampler, ReservoirStats, SamplingConfig,
    SamplingManagerStats, StratifiedSampler, StratifiedStats, TDigest, TDigestStats,
};

pub mod backend;
pub mod backend_optimizer;
pub mod backpressure;
pub mod biological_computing;
pub mod bridge;
pub mod circuit_breaker;
pub mod config;
pub mod connection_pool;
pub mod consciousness_streaming;
pub mod consumer;
pub mod cqels;
pub mod cqrs;
pub mod csparql;
pub mod delta;
pub mod diagnostics;
pub mod disaster_recovery;
pub mod dlq;
pub mod enterprise_audit;
pub mod enterprise_monitoring;
pub mod error;
pub mod event;
pub mod event_sourcing;
pub mod failover;
pub mod graphql_bridge;
pub mod graphql_subscriptions;
pub mod health_monitor;
pub mod join;
pub mod monitoring;
pub mod multi_region_replication;
pub mod multi_tenancy;
pub mod observability;
pub mod patch;
pub mod performance_optimizer;
pub mod performance_utils;
pub mod processing;
pub mod producer;
pub mod quantum_communication;
pub mod quantum_processing;
pub mod quantum_streaming;
pub mod reconnect;
pub mod reliability;
pub mod rsp;
pub mod schema_registry;
pub mod security;
pub mod serialization;
pub mod sparql_streaming;
pub mod state;
pub mod store_integration;
pub mod temporal_join;
pub mod time_travel;
pub mod tls_security;
pub mod types;
pub mod wasm_edge_computing;
pub mod webhook;

// New v0.1.0 modules for advanced features
pub mod custom_serialization;
pub mod end_to_end_encryption;
pub mod gpu_acceleration;
pub mod ml_integration;
pub mod rate_limiting;
pub mod scalability;
pub mod schema_evolution;
pub mod stream_replay;
pub mod transactional_processing;
pub mod zero_copy;

// New v0.1.0 modules for developer experience and performance
pub mod numa_processing;
pub mod out_of_order;
pub mod performance_profiler;
pub mod stream_sql;
pub mod testing_framework;

// New v0.1.0 modules for ML, versioning, and migration
pub mod anomaly_detection;
pub mod migration_tools;
pub mod online_learning;
pub mod stream_versioning;

// Advanced ML modules for v0.1.0 completion
pub mod automl_stream;
pub mod feature_engineering;
pub mod neural_architecture_search;
pub mod predictive_analytics;
pub mod reinforcement_learning;

// Utilities module
pub mod utils;

// Advanced SciRS2 optimization module
pub mod advanced_scirs2_optimization;
pub mod cdc_processor;

// Adaptive load shedding module
pub mod adaptive_load_shedding;

// Stream fusion optimizer module
pub mod stream_fusion;

// Complex Event Processing (CEP) engine module
pub mod cep_engine;

// Data quality and validation framework module
pub mod data_quality;

// Advanced sampling techniques module
pub mod advanced_sampling;

// Extracted type definitions to comply with 2000-line policy
mod lib_types;
pub use lib_types::*;
