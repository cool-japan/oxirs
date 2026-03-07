//! # OxiRS ARQ - SPARQL Query Engine
//!
//! [![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/cool-japan/oxirs/releases)
//! [![docs.rs](https://docs.rs/oxirs-arq/badge.svg)](https://docs.rs/oxirs-arq)
//!
//! **Status**: Production Release (v0.1.0)
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
pub mod adaptive_execution;
pub mod aggregates_ext;
pub mod algebra;
pub mod algebra_generation;
pub mod analytics;
pub mod bgp_optimizer;
pub mod bgp_optimizer_types;
pub mod buffer_management;
pub mod builtin;
pub mod builtin_fixed;
pub mod cache;
pub mod cache_integration;
pub mod cardinality_estimator;
pub mod cost_model;
pub mod debug_utilities;
pub mod distributed;
pub mod executor;
pub mod executor_pipeline;
pub mod expression;
pub mod expression_compiler;
pub mod extensions;
pub mod federation;
pub mod gpu_accelerated_ops;
pub mod graphql_translator;
pub mod integrated_query_planner;
pub mod interactive_query_builder;
pub mod jit_compiler;
pub mod join_algorithms;
pub mod lateral_join;
pub mod materialization;
pub mod materialized_views;
pub mod optimizer;
pub mod parallel;
pub mod path;
pub mod path_extensions;
pub mod plan_visualizer;
pub mod procedures;
pub mod production;
pub mod property_functions;
pub mod query;
pub mod query_analysis;
pub mod query_builder;
pub mod query_cache_lru;
pub mod query_plan_cache;
pub mod query_profiler;
pub mod query_rewriter;
pub mod query_validator;
pub mod result_formats;
pub mod results;
pub mod scirs_optimize_integration;
pub mod service_description;
pub mod simd_query_ops;
pub mod statistics_collector;
pub mod stats;
pub mod streaming;
pub mod string_functions_ext;
pub mod system_load_monitor;
pub mod term;
pub mod triple_functions;
pub mod update;
pub mod update_graph_management;
pub mod update_protocol;
pub mod values_support;
pub mod vector_query_optimizer;
pub mod websocket_streaming;

// Advanced modules
pub mod adaptive_index_advisor;
pub mod cost_model_calibration;
pub mod query_batch_executor;
pub mod query_execution_history;
pub mod query_fingerprinting;
pub mod query_hints;
pub mod query_optimization_advisor;
pub mod query_pagination;
pub mod query_plan_diff;
pub mod query_plan_export;
pub mod query_regression_testing;
pub mod query_result_cache;
pub mod query_templates;

// RDF-star / SPARQL-star integration
#[cfg(test)]
pub mod conformance;
pub mod execution;
pub mod rdf_star;
#[cfg(feature = "star")]
pub mod star_integration;
pub mod statistics;

// Adaptive query optimization and parallel evaluation (v0.2.0 Phase 2)
pub use execution::{ParallelBgpEvaluator, PatternDependencyGraph, TripleStore};
pub use optimizer::adaptive::{
    AdaptiveJoinOrderOptimizer, AdaptiveStatsStore, JoinAlgorithm, JoinPlanNode, PatternTerm,
    PlanTimer, RuntimeStats, TriplePatternInfo,
};
pub use optimizer::materialized_view::{
    BindingRow, MaterializedView, MaterializedViewManager, RdfTerm, ViewManagerConfig,
    ViewManagerStats,
};
// Adaptive join ordering (v0.2.0 Phase 2 - join_order)
pub use optimizer::join_order::{
    AdaptiveQueryPlan, CardinalityEstimate, JoinGraphStats, JoinOrderOptimizer, PatternCost,
};
// View registry (v0.2.0 Phase 2 - view_registry)
pub use optimizer::view_registry::{algebra_hash, BindingSet, ViewDefinition, ViewRegistry};
// Parallel pipeline executor (v0.2.0 Phase 2)
pub use executor_pipeline::{BindingMap, ParallelPipelineStage, UnionParallelExecutor};
// LRU query cache (v0.2.0 Phase 2)
pub use query_cache_lru::{
    CacheEntry, CacheFingerprint, CacheManagerStats, CompiledQueryCache, LruQueryCache,
    QueryCacheManager,
};

// Advanced modules
pub mod advanced_optimizer;
pub mod explain;
pub mod timeout;
// Temporarily disabled - require scirs2-core API migration
// TODO: Complete API migration in future release when scirs2-core API stabilizes
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

pub use lateral_join::{
    LateralCostEstimate, LateralJoin, LateralJoinConfig, LateralJoinError, LateralJoinExecutor,
    LateralJoinStats, LateralOptimizer, LateralOptimizerConfig, LateralParser, LateralStrategy,
    LateralSubquery, LateralValidationResult, LateralValidator, LateralValue, OrderSpec,
    SolutionMapping,
};

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
// Cache invalidation system (v0.2.0 Phase 1.1)
pub use cache::{
    CacheCoordinator, CacheLevel, DependencyGraph, InvalidationConfig, InvalidationEngine,
    InvalidationStatistics, InvalidationStrategy, RdfUpdateListener,
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
pub use explain::{
    ExplainFormat, IndexType as ExplainIndexType, PlanNode as ExplainPlanNode, QueryExplainer,
    QueryExplainerBuilder, QueryPlan as QueryExplainPlan,
};
pub use rdf_star::{
    bind_pattern, instantiate_quoted_triple, sparql_star_builtins, AnnotatedTriple, Annotation,
    QuotedTriple, RdfStarStore, StarBinding, StarObject, StarOperator, StarPattern, StarPredicate,
    StarSubject,
};

pub type Result<T> = anyhow::Result<T>;

// Graph analytics exports (v0.3.0)
pub use analytics::{
    is_dag, is_reachable, BetweennessCentrality, ConnectedComponents, DegreeCentrality, EdgeWeight,
    GraphStats, GraphStatsSummary, LouvainCommunities, NodeId, PageRank, RdfGraphAdapter,
    ShortestPaths,
};

// SPARQL 1.1 UPDATE Graph Management Operations (v0.3.0)
pub use update_graph_management::{
    GraphManagementDataset, GraphManagementExecutor, GraphManagementOp, GraphManagementResult,
    GraphManagementTarget, Triple as GraphManagementTriple,
};

// Runtime statistics collector (v0.2.0 enhancement)
pub use stats::{PatternStats, QueryExecutionStats, RuntimeStatsCollector};

// SPARQL 1.1 Update Protocol — standalone parser and in-memory executor (v0.2.0 enhancement)
pub use update_protocol::{
    ArqError as UpdateArqError, ClearType, DropType, ParseError as UpdateParseError,
    PatternTerm as UpdatePatternTerm, SparqlUpdate, SparqlUpdateParser, Triple as UpdateTriple,
    TriplePattern as UpdateTriplePattern, UpdateExecutor as SparqlUpdateExecutor,
    UpdateResult as SparqlUpdateResult,
};

// Optimizer passes (v0.2.0 enhancement)
pub use optimizer::{
    ConstantFoldingPass, OptimizationPass, OptimizationPipeline, PipelineResult,
    RedundantJoinEliminationPass, UnusedVariableEliminationPass,
};

// Subgraph isomorphism / pattern matching (v1.1.0 round 5)
pub mod subgraph_matcher;

// SPARQL 1.1 aggregate function executor (v1.1.0 round 6)
pub mod aggregate_executor;

// SPARQL-style window functions: ROW_NUMBER, RANK, DENSE_RANK, NTILE, LAG, LEAD (v1.1.0 round 7)
pub mod window_function;

// SPARQL 1.1 VALUES clause (inline data) (v1.1.0 round 9)
pub mod values_clause;

// SPARQL 1.1 SERVICE clause for federated queries (v1.1.0 round 10)
pub mod service_clause;

// SPARQL subquery (SELECT within SELECT) support (v1.1.0 round 11)
pub mod subquery;

// Cost-based join reordering for SPARQL query optimization (v1.1.0 round 12)
pub mod join_optimizer;

// SPARQL 1.1 expression/filter evaluator (v1.1.0 round 13)
pub mod expression_evaluator;

// SPARQL 1.1 GROUP BY clause evaluator (v1.1.0 round 11)
pub mod group_by_evaluator;

// SPARQL 1.1 OPTIONAL clause evaluator (left outer join) (v1.1.0 round 12)
pub mod optional_evaluator;

// SPARQL CONSTRUCT query builder — template instantiation, blank nodes, deduplication (v1.1.0 round 13)
pub mod construct_builder;

// SPARQL MINUS pattern evaluation (set-difference algebra) (v1.1.0 round 14)
pub mod minus_evaluator;

// SPARQL EXISTS and NOT EXISTS graph pattern evaluation (v1.1.0 round 15)
pub mod exists_evaluator;

// SPARQL 1.1 property path expression parser and evaluator (v1.1.0 round 16)
pub mod path_expression;
