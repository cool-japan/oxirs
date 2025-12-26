//! # OxiRS ARQ - SPARQL Query Engine
//!
//! [![Version](https://img.shields.io/badge/version-0.1.0--rc.1-blue)](https://github.com/cool-japan/oxirs/releases)
//! [![docs.rs](https://docs.rs/oxirs-arq/badge.svg)](https://docs.rs/oxirs-arq)
//!
//! **Status**: Beta Release (v0.1.0-rc.1)
//! **Stability**: Public APIs are stable. Production-ready with comprehensive testing.
//!
//! Advanced SPARQL 1.1/1.2 query engine with optimization, federation support, and custom functions.
//! Provides Jena ARQ-style SPARQL algebra with modern Rust performance.
//!
//! ## Features
//!
//! - **SPARQL 1.1 Query** - Complete SPARQL 1.1 query support
//! - **Query Optimization** - Cost-based optimization and join reordering
//! - **Parallel Execution** - Multi-threaded query processing
//! - **Custom Functions** - Extensible function framework
//! - **Federation Support** - Basic federated query capabilities
//! - **Result Streaming** - Memory-efficient result iteration
//!
//! ## Quick Start
//!
//! ```rust
//! use oxirs_core::query::QueryEngine;
//! use oxirs_core::RdfStore;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let engine = QueryEngine::new();
//! let store = RdfStore::new()?;
//!
//! let sparql = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10";
//! let results = engine.query(sparql, &store)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## See Also
//!
//! - [`oxirs-core`](https://docs.rs/oxirs-core) - RDF data model
//! - [`oxirs-fuseki`](https://docs.rs/oxirs-fuseki) - SPARQL HTTP server

// Core modules
pub mod adaptive_execution; // Adaptive query execution (v0.1.0)
pub mod aggregates_ext;
pub mod algebra;
pub mod algebra_generation;
pub mod bgp_optimizer;
pub mod bgp_optimizer_types;
pub mod buffer_management;
pub mod builtin;
pub mod builtin_fixed;
pub mod cache_integration;
pub mod cardinality_estimator; // Advanced cardinality estimation (v0.1.0)
pub mod cost_model;
pub mod debug_utilities; // Debugging utilities for SPARQL queries (v0.1.0)
pub mod distributed;
pub mod executor;
pub mod expression;
pub mod extensions;
pub mod federation; // Enhanced federated query execution (v0.1.0)
pub mod gpu_accelerated_ops; // GPU-accelerated SPARQL operations (Beta.2+)
pub mod graphql_translator; // GraphQL to SPARQL translation (v0.1.0)
pub mod integrated_query_planner;
pub mod interactive_query_builder; // Interactive SPARQL query builder (v0.1.0)
pub mod jit_compiler; // JIT compilation for SPARQL queries (v0.1.0)
pub mod join_algorithms;
pub mod materialization; // Query result materialization strategies (v0.1.0)
pub mod materialized_views;
pub mod optimizer;
pub mod parallel;
pub mod path;
pub mod path_extensions;
pub mod procedures;
pub mod production; // Production hardening features (Beta.1)
pub mod property_functions;
pub mod query;
pub mod query_analysis;
pub mod query_builder;
pub mod query_plan_cache;
pub mod query_profiler; // Query performance profiling (NEW)
pub mod query_rewriter; // Advanced query rewriting (v0.1.0)
pub mod query_validator; // Comprehensive query validation (v0.1.0)
pub mod result_formats;
pub mod results;
pub mod scirs_optimize_integration;
pub mod service_description;
pub mod simd_query_ops; // SIMD-accelerated query operations (v0.1.0)
pub mod statistics_collector;
pub mod streaming;
pub mod string_functions_ext;
pub mod term;
pub mod triple_functions;
pub mod update;
pub mod values_support;
pub mod vector_query_optimizer;
pub mod websocket_streaming; // WebSocket streaming for SPARQL results (v0.1.0)

// Beta.2+ Advanced modules
pub mod adaptive_index_advisor; // Adaptive index recommendations (Beta.2+)
pub mod cost_model_calibration; // Cost model calibration for adaptive learning (Beta.2+)
pub mod query_batch_executor;
pub mod query_execution_history; // Query execution history tracking (Beta.2+)
pub mod query_fingerprinting; // Advanced query fingerprinting for caching and analysis (Beta.2+)
pub mod query_hints; // Query hints system for optimizer guidance (Beta.2+)
pub mod query_optimization_advisor; // Automatic query optimization suggestions (Beta.2+)
pub mod query_pagination; // Query result pagination for large datasets (Beta.2+)
pub mod query_plan_diff; // Query plan comparison and diff utilities (Beta.2+++)
pub mod query_plan_export;
pub mod query_regression_testing;
pub mod query_result_cache; // Query result caching with fingerprint keys (Beta.2++++)
pub mod query_templates; // SPARQL query template system (Beta.2+++) // Query regression testing framework (Beta.2+) // Query plan export to various formats (Beta.2+) // Smart batch query executor with parallel execution (Beta.2+++++++) NEW!

// RDF-star / SPARQL-star integration
#[cfg(feature = "star")]
pub mod star_integration;

// Advanced modules
pub mod advanced_optimizer;
// Temporarily disabled - require scirs2-core API migration (248/316 errors fixed as of Dec 9, 2025)
// Progress: Counter/Timer API updated (.inc(), .get(), String parameters), 68 errors resolved
// Remaining: Clone/Debug trait issues, missing methods, async Send/Sync fixes needed
// TODO: Complete API migration in future release when scirs2-core stabilizes further
// pub mod advanced_statistics;
// pub mod ai_shape_learning;
// pub mod distributed_consensus;
// pub mod memory_management;
// pub mod quantum_optimization;
// pub mod realtime_streaming;
// pub mod unified_optimization_framework;

// Re-export commonly used types
pub use aggregates_ext::{
    Accumulator, AggregateFactory, AggregateMetadata, AggregateOptimization, AggregateRegistry,
    MemoryUsage,
};
pub use algebra::{
    Aggregate, Algebra, BinaryOperator, Binding, Expression, GroupCondition, Iri, Literal,
    OrderCondition, Solution, Term, TriplePattern, UnaryOperator, Variable,
};
pub use debug_utilities::{
    DebugBreakpoint, DebugConfig, DebugReport, ExecutionState as DebugExecutionState, JoinType,
    Operation, QueryDebugger, RewriteStep, TraceEntry, VariableBinding, VisualizationFormat,
};
pub use executor::{Dataset, ExecutionContext, InMemoryDataset, ParallelConfig, QueryExecutor};
pub use graphql_translator::{
    GraphQLDirective, GraphQLDocument, GraphQLField, GraphQLFragment, GraphQLOperation,
    GraphQLOperationType, GraphQLSelection, GraphQLTranslator, GraphQLValue,
    GraphQLVariableDefinition, SchemaMapping, TranslationError, TranslationResult,
    TranslationStats, TranslatorConfig,
};
pub use jit_compiler::{
    CompiledQuery, CompilerStats, ExecutionPlan, ExecutionStats, FilterType, JitCompilerConfig,
    JitJoinStrategy, PatternType, PlanOperation, QueryJitCompiler, QueryMetadata, Specialization,
    SpecializationType,
};
pub use path_extensions::{
    BidirectionalPathSearch, CacheStats, CachedPathEvaluator, CostBasedPathOptimizer, PathAnalyzer,
    PathCache, PathComplexity, PathEvaluationStrategy, PathOptimizationConfig,
    PathOptimizationHint, PathStatistics, ReachabilityIndex, StrategyCostEstimate,
};
pub use procedures::{
    Procedure, ProcedureArgs, ProcedureContext, ProcedureFactory, ProcedureRegistry,
    ProcedureResult,
};
pub use property_functions::{
    PropFuncArg, PropertyFunction, PropertyFunctionContext, PropertyFunctionRegistry,
    PropertyFunctionResult,
};
pub use query_builder::{AskBuilder, ConstructBuilder, DescribeBuilder, SelectBuilder};
pub use query_plan_cache::{
    CacheStats as QueryPlanCacheStats, CachedPlan, CachingConfig, QueryPlanCache, QuerySignature,
    StatisticsSnapshot,
};
pub use query_profiler::{AverageStats, QueryPhase, QueryProfiler, QueryStats};
pub use query_validator::{
    QueryValidator, ValidationConfig, ValidationResult, ValidationStatistics, ValidationWarning,
    ValidationWarningType,
};
pub use result_formats::{
    BinaryResultSerializer, CustomFormatSerializer, FormatConverter, FormatRegistry,
    StreamingResultIterator, XmlResultSerializer,
};
pub use results::{QueryResult, ResultFormat, ResultSerializer};
pub use scirs_optimize_integration::{
    OptimizationResult, PerformanceAnalysis, QueryInfo, QueryOptimizationConfig,
    SciRS2QueryOptimizer,
};
pub use service_description::{
    create_default_service_description, AggregateInfo, DatasetDescription, ExtensionFunction,
    Feature, LanguageExtension, NamedGraphDescription, ParameterInfo, ProcedureInfo,
    PropertyFunctionInfo, ServiceDescription, ServiceDescriptionBuilder,
    ServiceDescriptionRegistry, ServiceLimitations,
};
pub use string_functions_ext::{
    StrAfterFunction, StrBeforeFunction, StrDtFunction, StrLangDirFunction, StrLangFunction,
};
pub use triple_functions::{
    IsTripleFunction, ObjectFunction, PredicateFunction, SubjectFunction, TripleFunction,
};
pub use values_support::{
    IndexedValues, JoinStrategy, OptimizedValues, ValuesBuilder, ValuesClause,
    ValuesExecutionStrategy, ValuesExecutor, ValuesJoinOptimizer, ValuesOptimizer,
    ValuesStatistics,
};
// Temporarily disabled - require scirs2-core beta.4 APIs
/*
pub use memory_management::{
    MemoryConfig, MemoryManagedContext, MemoryPerformanceReport, MemoryStats, MemoryLeakReport,
    ProcessedSolution, MemoryPressureStrategy,
};
pub use quantum_optimization::{
    QuantumOptimizationConfig, QuantumJoinOptimizer, QuantumCardinalityEstimator,
    HybridQuantumOptimizer, QuantumOptimizationStats, HybridOptimizationStats,
    QuantumOptimizationStrategy, QuantumQueryState,
};
pub use realtime_streaming::{
    StreamingSparqlProcessor, StreamingConfig, StreamingTriple, WindowedResult,
    SignalPipelineConfig, StreamingStatistics, DetectedPattern, AnomalyIndicator,
    PatternType, AnomalyType, WatermarkStrategy,
};
pub use ai_shape_learning::{
    AIShapeLearner, ShapeLearningConfig, LearnedShape, PropertyConstraint,
    ValidationResult, ValidationViolation, ShapeLearningStatistics,
    RdfDataBatch, ValidationStrategy, PatternConstraint,
};
pub use distributed_consensus::{
    DistributedConsensusCoordinator, ConsensusConfig, ConsensusValue, ConsensusResult,
    ConsensusState, ConsensusStatistics, ConsensusAlgorithm, ByzantineConfig,
};
*/

pub use adaptive_index_advisor::{
    AccessPattern, AdvisorConfig, AdvisorStatistics, AnalysisSummary, IndexAdvisor,
    IndexAnalysisReport, IndexConfiguration, IndexRecommendation, IndexType, IndexUsageStats,
    PatternComponent, QueryPattern, RecommendationPriority,
};
pub use cost_model_calibration::{
    CalibrationConfig, CalibrationExport, CalibrationReport, CalibratorStatistics,
    CostModelCalibrator, CostModelParameters, ExecutionSample, OperationCalibrationExport,
    OperationCalibrationStats, OperationSummary, OperationType,
};
pub use federation::{
    EndpointCapabilities, EndpointCriteria, EndpointDiscovery, EndpointHealth, FederatedSubquery,
    FederationConfig, FederationExecutor, FederationStats, LoadBalancingStrategy,
};
pub use gpu_accelerated_ops::{DeviceSelection, GpuConfig, GpuOperationStats, GpuQueryEngine};
pub use interactive_query_builder::{
    helpers as query_helpers, InteractiveQueryBuilder, PatternBuilder, QueryType,
};
pub use materialization::{
    MaterializationAnalysis, MaterializationConfig, MaterializationSelector, MaterializationStats,
    MaterializationStrategy, MaterializedResults, ResultIterator, VariableStats,
};
pub use production::{
    // Core production types
    AuditEventType,
    // Beta.2 enhancements
    BaselineTrackerConfig,
    CostEstimatorConfig,
    CostEstimatorStatistics,
    CostRecommendation,
    ErrorSeverity,
    GlobalStatistics,
    HealthStatus,
    MemoryStats,
    PerformanceBaselineTracker,
    PerformanceTrend,
    PrioritizedQuery,
    PrioritySchedulerConfig,
    PrioritySchedulerStats,
    QueryAuditEvent,
    QueryAuditTrail,
    QueryCancellationToken,
    QueryCircuitBreaker,
    QueryCostEstimate,
    QueryCostEstimator,
    QueryEngineHealth,
    QueryErrorContext,
    QueryFeatures,
    QueryMemoryTracker,
    QueryPriority,
    QueryPriorityScheduler,
    QueryRateLimiter,
    QueryResourceQuota,
    QuerySession,
    QuerySessionManager,
    QueryStatistics,
    QueryTimeoutManager,
    QueryTimeoutState,
    RegressionReport,
    RegressionSeverity,
    SparqlPerformanceMonitor,
    SparqlProductionError,
    TimeoutAction,
    TimeoutCheckResult,
};
pub use query_execution_history::{
    ExecutionMetrics, ExecutionRecord, ExecutionStatus, FormDistribution, HistoryAnalysis,
    HistoryConfig, HistoryStatistics, PeriodStatistics, QueryExecutionHistory, QueryFormType,
    QueryGroupStats, SlowQueryEntry,
};
pub use query_fingerprinting::{
    FingerprintConfig, FingerprintingStatistics, HashAlgorithm, ParameterSlot, ParameterType,
    QueryFeatures as FingerprintQueryFeatures, QueryFingerprint, QueryFingerprinter,
    QueryForm as FingerprintQueryForm,
};
pub use query_hints::{
    CacheHint, CardinalityHint, FilterHint, FilterPushdownDirective, HintApplicationResult,
    HintParser, HintParserStats, HintValidationWarning, HintValidator, IndexDirective, IndexHint,
    JoinAlgorithmHint, JoinBuildSide, JoinHint, JoinOrderHint, JoinOrderStrategy,
    MaterializationHint, MaterializationStrategy as HintMaterializationStrategy, MemoryHint,
    ParallelismHint, QueryHints, QueryHintsBuilder, WarningSeverity,
};
pub use query_plan_export::{
    CostEstimate, ExecutionStats as PlanExecutionStats, ExportConfig, ExportError, ExportFormat,
    ExporterStats, OperatorType, PlanNode, QueryPlanExporter,
};
pub use query_regression_testing::{
    ExecutionResult as RegressionExecutionResult, ExecutionStatistics as RegressionExecutionStats,
    GoldenQuery, QueryRegressionAnalysis, RegressionConfig,
    RegressionReport as QueryRegressionReport, RegressionStatus, RegressionTestSuite,
    RegressionTestSuiteBuilder, ReportComparison, ReportSummary, SuiteExport, SuiteStatistics,
};
pub use query_result_cache::{
    CacheConfig as ResultCacheConfig, CacheStatistics as ResultCacheStatistics, QueryResultCache,
    QueryResultCacheBuilder,
};
pub use simd_query_ops::{
    ComparisonOp, JoinStats, SimdAggregations, SimdConfig, SimdFilterEvaluator, SimdHashJoin,
    SimdStringOps, SimdTripleMatcher, TripleCandidate,
};
pub use websocket_streaming::{
    ConnectionStats, ManagerStats, WebSocketConfig, WebSocketManager, WebSocketMessage,
    WebSocketSession,
};

// RDF-star / SPARQL-star exports (when feature is enabled)
#[cfg(feature = "star")]
pub use star_integration::{
    pattern_matching, sparql_star_functions, star_statistics::SparqlStarStatistics,
    SparqlStarExecutor,
};

// Common Result type for the crate
pub type Result<T> = anyhow::Result<T>;
