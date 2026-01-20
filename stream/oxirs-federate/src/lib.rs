//! # OxiRS Federation - Federated Query Engine
//!
//! [![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/cool-japan/oxirs/releases)
//! [![docs.rs](https://docs.rs/oxirs-federate/badge.svg)](https://docs.rs/oxirs-federate)
//!
//! **Status**: Production Release (v0.1.0)
//! **Stability**: Public APIs are stable. Production-ready with comprehensive testing.
//!
//! Federated query processing capabilities for SPARQL and GraphQL with service discovery,
//! query decomposition, result integration, and fault tolerance.

#![allow(ambiguous_glob_reexports)]
//!
//! # Features
//!
//! - SPARQL SERVICE planner and executor for federated SPARQL queries
//! - GraphQL schema stitching and federation
//! - Service discovery and capability detection
//! - Query decomposition and optimization across multiple sources
//! - Result integration with fault tolerance and partial result handling
//! - Load balancing and performance monitoring
//!
//! # Architecture
//!
//! The federation engine consists of several key components:
//!
//! - `ServiceRegistry`: Manages available federated services and their capabilities
//! - `QueryPlanner`: Decomposes queries across multiple services
//! - `Executor`: Executes federated queries with parallel processing
//! - `ResultIntegrator`: Combines results from multiple sources
//! - `FaultHandler`: Manages service failures and retries

use anyhow::{anyhow, Result};
use oxirs_core::Term;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::executor::types::{QuotedTripleValue, RdfTerm};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{info, warn};
// PlannerConfig will be imported via planner module re-exports

// Import types needed for the main struct
use crate::{
    auto_discovery::{AutoDiscovery, AutoDiscoveryConfig, DiscoveredEndpoint},
    cache::{CacheConfig, CacheStats, FederationCache, QueryResultCache},
    capability_assessment::{AssessmentResult, CapabilityAssessor},
    discovery::ServiceDiscovery,
    executor::{FederatedExecutor, FederatedExecutorConfig},
    graphql::{GraphQLFederation, GraphQLFederationConfig},
    integration::{ResultIntegrator, ResultIntegratorConfig},
    monitoring::{FederationMonitor, FederationMonitorConfig, MonitorStats},
    planner::{PlannerConfig, QueryPlanner},
    service_registry::{RegistryConfig, RegistryStats, ServiceRegistry},
    vector_similarity_federation::{
        VectorFederationConfig, VectorFederationStats, VectorServiceMetadata,
        VectorSimilarityFederation,
    },
};

pub mod adaptive_load_balancer;
pub mod advanced_anomaly_detection;
pub mod advanced_benchmarking;
pub mod advanced_consensus;
pub mod advanced_enterprise_features;
pub mod advanced_ml_optimizer;
pub mod advanced_query_optimizer;
pub mod advanced_security_hardening;
pub mod advanced_semantic_features;
pub mod advanced_visualization;
pub mod anomaly_detection;
pub mod auth;
pub mod auto_discovery;
pub mod automl_pipeline;
pub mod cache;
pub mod capability_assessment;
pub mod cdc;
pub mod cloud_cost_optimizer;
pub mod connection_pool_manager;
pub mod discovery;
pub mod distributed_consensus;
pub mod distributed_ml_trainer;
pub mod distributed_tracing;
pub mod distributed_transactions;
pub mod executor;
pub mod external_ml_integration;
pub mod gpu_accelerated_query;
pub mod graph_algorithms;
pub mod graphql;
pub mod integration;
pub mod jit_query_compiler;
pub mod join_optimizer;
pub mod k8s_discovery;
pub mod materialized_views;
pub mod memory_efficient_datasets;
pub mod metadata;
pub mod ml_model_serving;
pub mod ml_optimizer;
pub mod monitoring;
pub mod multi_level_federation;
pub mod nats_federation;
pub mod network_optimizer;
pub mod optimization_cache;
pub mod performance_analyzer;
pub mod performance_benchmarks;
pub mod planner;
pub mod privacy;
pub mod production_hardening;
pub mod profiling_metrics;
pub mod query_decomposition;
pub mod query_plan_explainer;
pub mod request_batcher;
pub mod result_streaming;
pub mod schema_alignment;
pub mod semantic_enhancer;
pub mod semantic_reasoner;
pub mod service;
pub mod service_client;
pub mod service_executor;
pub mod service_optimizer;
pub mod service_registry;
pub mod simd_optimized_joins;
pub mod source_selection;
pub mod streaming;
pub mod streaming_optimizer;
pub mod test_infrastructure;
pub mod vector_similarity_federation;

// Minimal imports to ensure compilation - only core types
// Re-enabled specific non-duplicate exports after fixing ServiceRegistry conflicts
// AutoDiscovery and CacheConfig already imported above, skipping duplicates
pub use adaptive_load_balancer::AdaptiveLoadBalancer;
pub use advanced_query_optimizer::{
    AdvancedOptimizerConfig, AdvancedQueryOptimizer, HardwareProfile, OptimizedPlan, QueryPlan,
    TrainingExample,
};
// Advanced ML Optimizer (v0.1.0 advanced features)
pub use advanced_ml_optimizer::{
    ActivationType, AdvancedMLConfig, AdvancedMLOptimizer, AutoML, DeepCardinalityEstimator,
    ExplainableAI, JoinType, NeuralArchitectureSearch, OnlineLearningManager, RLJoinOptimizer,
    TrainingEpoch, TransferLearningManager,
};
// Advanced Benchmarking (v0.1.0 advanced features)
pub use advanced_benchmarking::{
    AdvancedBenchmarkConfig, AdvancedBenchmarkSuite, BenchmarkResult, CustomBenchmarkConfig,
    CustomBenchmarkGenerator, LUBMSuite, RegressionDetectionResult, SP2BenchSuite,
    ScalabilityTestResult, StressTestResult, WatDivSuite, WorkloadCharacterization,
};
// Advanced Semantic Features (v0.1.0 advanced features)
pub use advanced_semantic_features::{
    AdvancedSemanticConfig, AdvancedSemanticFeatures, AutoMapping, AutoMappingGenerator,
    ChangeType, ConceptRelationship, DeepOntologyMatcher, Entity, EntityMatch, EntityResolver,
    MappingType, MatchType, MultiLingualConcept, MultiLingualSchemaManager, OntologyConcept,
    OntologyMatch, RelationType, SchemaChange, SchemaEvolutionTracker, SchemaVersion,
};
// Advanced Anomaly Detection (v0.1.0 advanced features)
pub use advanced_anomaly_detection::{
    AdvancedAnomalyConfig, AdvancedAnomalyDetection, HealingAction, HealingActionType,
    IsolationForest, IssueType, LSTMPredictor, MaintenancePriority, MaintenanceTask,
    PredictiveMaintenanceScheduler, RootCause, RootCauseAnalyzer, SelfHealingEngine,
    Severity as AnomalySeverity,
};
// Advanced Consensus (v0.1.0 advanced features)
pub use advanced_consensus::{
    AdvancedConsensusSystem, ByzantineFaultTolerance, DistributedLock, GCounter,
    NetworkPartitionDetector, PNCounter, VectorClock,
};
// Advanced Enterprise Features (v0.1.0 advanced features)
pub use advanced_enterprise_features::{
    AdvancedEnterpriseFeatures, AuditLogEntry, AuditLogger, AuditResult, DataLineageTracker,
    DataSubject, DeletionRequest, EdgeComputingManager, EdgeNode, EdgeNodeStatus,
    GDPRComplianceManager, GeoLocation, GeographicQueryRouter, LineageNode, MultiTenancyConfig,
    MultiTenancyManager, PrivacyPreservingFederation, QuantumResistantSecurity, ResourceQuota,
    ResourceUsage, Tenant,
};
// Advanced Security Hardening (v0.2.0 Phase 3)
pub use advanced_security_hardening::{
    AdvancedRateLimiter, AdvancedSecurityHardening, AuditConfig, AuditEvent, AuditEventType,
    AuthMethod, AuthSession, AuthenticationManager, ComplianceChecker, ComplianceFramework,
    ComplianceStatus, EncryptionConfig, EncryptionKey, EncryptionManager, IdsConfig, IdsResult,
    IntrusionDetectionSystem, MtlsCertificate, OAuth2Provider, OidcProvider, RateLimitConfig,
    SecurityAlert, SecurityCheckResult, SecurityConfig, SecurityContext, ThreatCategory,
    ThreatSeverity, ThreatSignature, TrustScore, UserInfo, Vulnerability, VulnerabilityScanResult,
    VulnerabilityScanner, VulnerabilitySeverity, ZeroTrustConfig, ZeroTrustController,
};
// Advanced Visualization & Dashboarding (v0.2.0 Phase 3)
pub use advanced_visualization::{
    AdvancedVisualization, AggregationType, Alert, AlertGrouping, AlertSeverity, AlertTimeline,
    AlertVisualizer, ChartData, ChartGenerator, ChartSeries, ChartTheme, ColorScale, CustomTheme,
    Dashboard, DashboardLayout, DataSource, EdgeType, ExportFormat, HeatmapCell, HeatmapData,
    LayoutAlgorithm, MetricAggregation, MetricsCollector, NodeStatus, NodeType, PieChartData,
    PieSlice, PositionedNode, TimeSeries, TopologyEdge, TopologyNode, TopologyVisualization,
    TopologyVisualizer, VisualizationConfig, VisualizationType, Widget, WidgetConfig,
    WidgetPosition, WidgetSize, WidgetType,
};
// Re-export DataPoint from advanced_visualization as VizDataPoint to avoid conflict
pub use advanced_visualization::DataPoint as VizDataPoint;
// v0.2.0 Performance & Scalability Features
pub use anomaly_detection::{
    AnomalyAlert, AnomalyDetector, AnomalyDetectorConfig, DataPoint, Severity, Trend, TrendAnalysis,
};
pub use auth::AuthManager;
pub use automl_pipeline::{
    ArchitectureCandidate, AutoMLConfig, AutoMLPipeline, HyperparameterConfig, LayerType,
    MetaLearningTask, OptimizationStatistics, SearchSpace, TrialMetrics, TrialResult,
};
pub use cloud_cost_optimizer::{
    CloudCostOptimizer, CloudInstance, CloudProvider, CostOptimizerConfig, CostRecommendation,
    CostTracking, DeploymentConfig, InstanceStatus, OptimizationStrategy, RecommendationPriority,
    RecommendationType, RoutingDecision, WorkloadMetrics,
};
pub use connection_pool_manager::ConnectionPoolManager;
pub use distributed_ml_trainer::{
    AggregationStrategy, DistributedMLConfig, DistributedMLTrainer, TrainingMetrics, TrainingMode,
    WorkerInfo, WorkerMetrics, WorkerStatus,
};
pub use distributed_tracing::TracingConfig;
pub use distributed_transactions::{
    DistributedTransactionCoordinator, IsolationLevel, Operation, OperationState, OperationType,
    Participant, ParticipantState, SagaLog, SagaStep, SagaStepState, Transaction,
    TransactionConfig, TransactionProtocol, TransactionResult, TransactionState,
};
pub use executor::ExecutionStatus; // ExecutionMetrics not exported from executor
                                   // External ML Frameworks Integration (v0.2.0 Phase 3)
pub use external_ml_integration::{
    ConversionResult, DataType, ExternalMLIntegration, FrameworkAdapter, FrameworkCapabilities,
    HuggingFaceAdapter, InferenceBackend, InferenceEngine, MLFramework, MLIntegrationConfig,
    MLModel, MLTaskType, MockModel, ModelConverter, ModelFormat, ModelMetadata, ModelRegistry,
    ModelVersion, ONNXAdapter, OptimizationConfig, OptimizationResult, PerformanceMetrics,
    PyTorchAdapter, RegisteredModel, TensorFlowAdapter, TensorSpec,
};
pub use gpu_accelerated_query::{
    FilterCondition, FilterOperator, GpuAccelerationConfig, GpuBackendType, GpuJoinProcessor,
    GpuProcessingResult, GpuProcessingStats, GpuQueryProcessor, QueryBatch,
};
pub use graph_algorithms::{
    AStar, BellmanFord, CentralityAnalyzer, ConnectivityAnalyzer, Dijkstra, Edge, FederationGraph,
    FloydWarshall, PrimMST, ShortestPathResult,
};
pub use jit_query_compiler::{
    CompiledQuery, ExecutionMode, JitCompilationConfig, JitCompilationStats, JitQueryCompiler,
    JitQueryOptimizer, OptimizationRule,
};
pub use ml_model_serving::{
    ABTestConfig, ABTestResults, MLModelServing, ModelMetrics, ModelServingConfig, ModelStatus,
    ModelType, QueryTransformerModel, TransformerConfig,
};
// Re-export ModelVersion from ml_model_serving as MLModelVersion to avoid conflict
pub use ml_model_serving::ModelVersion as MLModelVersion;
pub use simd_optimized_joins::{JoinAlgorithm, JoinStatistics, SimdJoinConfig, SimdJoinProcessor};
// Import from graphql module - minimal types to avoid conflicts
// pub use graphql::GraphQLFederation;

// More specific imports to avoid conflicts - conservative approach
// Re-enabled after fixing module exports - only exports that actually exist
// pub use integration::ResultIntegratorConfig; // Already imported above
pub use ml_optimizer::MLOptimizer;
pub use multi_level_federation::{
    FederationCapability, FederationMetrics, FederationNode, MultiLevelConfig,
    MultiLevelFederation, TopologyOptimizationResult, TopologyStats,
};
pub use network_optimizer::NetworkOptimizer;
pub use optimization_cache::OptimizationCache;
pub use production_hardening::{
    CircuitBreakerState, ComplexityResult, HardeningConfig, HardeningStatistics,
    ProductionHardening, QueryRequest as HardeningQueryRequest, ValidationResult,
};
pub use schema_alignment::{
    Alignment, AlignmentConfig, AlignmentResult, AlignmentType, ClassMetadata, MappingExample,
    PropertyMetadata, SchemaAligner, VocabularyMetadata,
};
pub use semantic_reasoner::{
    InconsistencyReport, InconsistencyType, ReasonerConfig, SemanticReasoner, Triple,
};
// Re-enabled: JoinOptimizer now properly implemented
pub use join_optimizer::JoinOptimizer;
// pub use k8s_discovery::KubernetesDiscovery;
// pub use materialized_views::MaterializedView;
// pub use metadata::MetadataConfig;
// pub use monitoring::MonitoringService;
// pub use nats_federation::NatsFederation;
// pub use performance_analyzer::PerformanceAnalyzer;
// Import from planner module excluding GraphQLFederationConfig to avoid conflict
pub use planner::{
    EntityResolutionPlan, EntityResolutionStep, ExecutionContext, ExecutionPlan, ExecutionStep,
    FederatedQueryPlanner, FederatedSchema, FilterExpression, GraphQLOperationType,
    HistoricalPerformance, ParsedQuery, QueryInfo, QueryType, ReoptimizationAnalysis, RetryConfig,
    ServiceQuery, StepType, TriplePattern, UnifiedSchema,
};
pub use privacy::*;
// Re-enabled query decomposition exports
pub use query_decomposition::advanced_pattern_analysis::{
    AdvancedPatternAnalyzer, ComplexityAssessment, OptimizationOpportunity, PatternAnalysisResult,
    ServiceRecommendation,
};
pub use query_decomposition::types::QueryDecomposer;
pub use query_plan_explainer::{
    ExplainFormat, OptimizationSuggestion, PlanExplanation, QueryPlanExplainer, StepExplanation,
    SuggestionCategory, SuggestionSeverity,
};
pub use request_batcher::RequestBatcher;
pub use result_streaming::ResultStreamingManager;
pub use semantic_enhancer::SemanticEnhancer;
// Export main service types (from service.rs)
pub use service::{
    AuthCredentials, FederatedService, ServiceAuthConfig, ServiceCapability, ServicePerformance,
    ServiceType,
};
// Re-enabled service implementation exports
pub use service_client::ServiceClient;
pub use service_executor::ServiceExecutor;
pub use service_optimizer::ServiceOptimizer;
// Export specific types from service_registry (non-conflicting types only)
pub use service_registry::{
    GraphQLService, HealthStatus as ServiceHealthStatus, ServiceCapabilities, SparqlEndpoint,
};
// pub use source_selection::SourceSelector;
// pub use streaming::StreamProcessor;
// pub use streaming_optimizer::StreamOptimizer;
// pub use test_infrastructure::TestRunner;
// pub use vector_similarity_federation::VectorFederation;

/// Main federation engine that coordinates all federated query processing
#[derive(Debug, Clone)]
pub struct FederationEngine {
    /// Registry of available services
    service_registry: Arc<RwLock<ServiceRegistry>>,
    /// Query planner for service selection and decomposition
    query_planner: Arc<QueryPlanner>,
    /// Execution engine for federated queries
    executor: Arc<FederatedExecutor>,
    /// Result integration engine
    integrator: Arc<ResultIntegrator>,
    /// GraphQL federation manager
    graphql_federation: Arc<GraphQLFederation>,
    /// Performance monitoring
    monitor: Arc<FederationMonitor>,
    /// Advanced caching system
    cache: Arc<FederationCache>,
    /// Automatic service discovery
    auto_discovery: Arc<RwLock<Option<AutoDiscovery>>>,
    /// Vector similarity federation
    vector_federation: Arc<RwLock<Option<VectorSimilarityFederation>>>,
}

impl FederationEngine {
    /// Create a new federation engine with default configuration
    pub fn new() -> Self {
        let service_registry = Arc::new(RwLock::new(ServiceRegistry::new()));
        let query_planner = Arc::new(QueryPlanner::new());
        let executor = Arc::new(FederatedExecutor::new());
        let integrator = Arc::new(ResultIntegrator::new());
        let graphql_federation = Arc::new(GraphQLFederation::new());
        let monitor = Arc::new(FederationMonitor::new());
        let cache = Arc::new(FederationCache::new());

        Self {
            service_registry,
            query_planner,
            executor,
            integrator,
            graphql_federation,
            monitor,
            cache,
            auto_discovery: Arc::new(RwLock::new(None)),
            vector_federation: Arc::new(RwLock::new(None)),
        }
    }

    /// Create a new federation engine with custom configuration
    pub fn with_config(config: FederationConfig) -> Self {
        let service_registry = Arc::new(RwLock::new(ServiceRegistry::with_config(
            config.registry_config,
        )));
        let query_planner = Arc::new(QueryPlanner::with_config(config.planner_config));
        let executor = Arc::new(FederatedExecutor::with_config(config.executor_config));
        let integrator = Arc::new(ResultIntegrator::with_config(config.integrator_config));
        let graphql_federation = Arc::new(GraphQLFederation::with_config(config.graphql_config));
        let monitor = Arc::new(FederationMonitor::with_config(config.monitor_config));
        let cache = Arc::new(FederationCache::with_config(config.cache_config));

        Self {
            service_registry,
            query_planner,
            executor,
            integrator,
            graphql_federation,
            monitor,
            cache,
            auto_discovery: Arc::new(RwLock::new(None)),
            vector_federation: Arc::new(RwLock::new(None)),
        }
    }

    /// Register a new federated service
    pub async fn register_service(&self, service: FederatedService) -> Result<()> {
        let registry = self.service_registry.write().await;
        registry.register(service).await
    }

    /// Unregister a federated service
    pub async fn unregister_service(&self, service_id: &str) -> Result<()> {
        let registry = self.service_registry.write().await;
        registry.unregister(service_id).await
    }

    /// Execute a federated SPARQL query
    pub async fn execute_sparql(&self, query: &str) -> Result<FederatedResult> {
        let start_time = Instant::now();

        // Parse and analyze the query
        let query_info = self.query_planner.analyze_sparql(query).await?;

        // Generate cache key
        let cache_key = self.cache.generate_query_key(&query_info);

        // Check cache first
        if let Some(cached_result) = self.cache.get_query_result(&cache_key).await {
            let execution_time = start_time.elapsed();

            // Record cache hit
            self.monitor.record_cache_hit("query_cache", true).await;
            self.monitor
                .record_query_execution("sparql", execution_time, true)
                .await;

            return match cached_result {
                QueryResultCache::Sparql(sparql_results) => {
                    // Convert back to FederatedResult
                    let result_bindings: Vec<HashMap<String, oxirs_core::Term>> = sparql_results
                        .results
                        .bindings
                        .into_iter()
                        .map(|binding| {
                            // Convert SparqlBinding to HashMap<String, Term>
                            binding
                                .into_iter()
                                .filter_map(|(var, sparql_value)| {
                                    match sparql_value.value_type.as_str() {
                                        "uri" => {
                                            if let Ok(iri) =
                                                oxirs_core::NamedNode::new(&sparql_value.value)
                                            {
                                                Some((var, oxirs_core::Term::NamedNode(iri)))
                                            } else {
                                                None
                                            }
                                        }
                                        "literal" => {
                                            if let Some(datatype_str) = sparql_value.datatype {
                                                if let Ok(datatype) =
                                                    oxirs_core::NamedNode::new(&datatype_str)
                                                {
                                                    Some((
                                                        var,
                                                        oxirs_core::Term::Literal(
                                                            oxirs_core::Literal::new_typed(
                                                                &sparql_value.value,
                                                                datatype,
                                                            ),
                                                        ),
                                                    ))
                                                } else {
                                                    Some((
                                                        var,
                                                        oxirs_core::Term::Literal(
                                                            oxirs_core::Literal::new(
                                                                &sparql_value.value,
                                                            ),
                                                        ),
                                                    ))
                                                }
                                            } else if let Some(lang) = sparql_value.lang {
                                                if let Ok(literal) = oxirs_core::Literal::new_lang(
                                                    &sparql_value.value,
                                                    &lang,
                                                ) {
                                                    Some((var, oxirs_core::Term::Literal(literal)))
                                                } else {
                                                    Some((
                                                        var,
                                                        oxirs_core::Term::Literal(
                                                            oxirs_core::Literal::new(
                                                                &sparql_value.value,
                                                            ),
                                                        ),
                                                    ))
                                                }
                                            } else {
                                                Some((
                                                    var,
                                                    oxirs_core::Term::Literal(
                                                        oxirs_core::Literal::new(
                                                            &sparql_value.value,
                                                        ),
                                                    ),
                                                ))
                                            }
                                        }
                                        "bnode" => {
                                            if let Ok(bnode) =
                                                oxirs_core::BlankNode::new(&sparql_value.value)
                                            {
                                                Some((var, oxirs_core::Term::BlankNode(bnode)))
                                            } else {
                                                None
                                            }
                                        }
                                        _ => None,
                                    }
                                })
                                .collect()
                        })
                        .collect();

                    Ok(FederatedResult {
                        data: QueryResult::Sparql(result_bindings),
                        metadata: ExecutionMetadata {
                            execution_time,
                            services_used: 0, // From cache
                            subqueries_executed: 0,
                            cache_hit: true,
                            plan_summary: "Cached result".to_string(),
                        },
                        errors: vec![],
                    })
                }
                _ => {
                    // Invalid cache entry type
                    self.cache.remove(&cache_key).await;
                    return Err(anyhow!("Invalid cached result type for SPARQL query"));
                }
            };
        }

        // Cache miss - execute normally
        self.monitor.record_cache_hit("query_cache", false).await;

        // Plan the federated execution
        let registry = self.service_registry.read().await;
        let execution_plan = self
            .query_planner
            .plan_sparql(&query_info, &registry)
            .await?;
        drop(registry);

        // Execute the plan
        let partial_results = self.executor.execute_plan(&execution_plan).await?;

        // Integrate results
        let final_result = self
            .integrator
            .integrate_sparql_results(partial_results)
            .await?;

        // Cache the result if successful
        if final_result.is_success() {
            if let QueryResult::Sparql(ref result_bindings) = final_result.data {
                // Convert to cacheable format
                let sparql_bindings: Vec<crate::executor::SparqlBinding> = result_bindings
                    .iter()
                    .map(|binding| {
                        binding
                            .iter()
                            .map(|(var, term)| {
                                let sparql_value = match term {
                                    oxirs_core::Term::NamedNode(node) => {
                                        crate::executor::SparqlValue {
                                            value_type: "uri".to_string(),
                                            value: node.to_string(),
                                            datatype: None,
                                            lang: None,
                                            quoted_triple: None,
                                        }
                                    }
                                    oxirs_core::Term::Literal(literal) => {
                                        if let Some(lang) = literal.language() {
                                            crate::executor::SparqlValue {
                                                value_type: "literal".to_string(),
                                                value: literal.value().to_string(),
                                                datatype: None,
                                                lang: Some(lang.to_string()),
                                                quoted_triple: None,
                                            }
                                        } else {
                                            crate::executor::SparqlValue {
                                                value_type: "literal".to_string(),
                                                value: literal.value().to_string(),
                                                datatype: Some(literal.datatype().to_string()),
                                                lang: None,
                                                quoted_triple: None,
                                            }
                                        }
                                    }
                                    oxirs_core::Term::BlankNode(bnode) => {
                                        crate::executor::SparqlValue {
                                            value_type: "bnode".to_string(),
                                            value: bnode.to_string(),
                                            datatype: None,
                                            lang: None,
                                            quoted_triple: None,
                                        }
                                    }
                                    oxirs_core::Term::Variable(var) => {
                                        crate::executor::SparqlValue {
                                            value_type: "variable".to_string(),
                                            value: var.to_string(),
                                            datatype: None,
                                            lang: None,
                                            quoted_triple: None,
                                        }
                                    }
                                    oxirs_core::Term::QuotedTriple(triple) => {
                                        // Convert oxirs_core terms to our RDF term representation
                                        let subject = convert_subject_to_rdf_term(triple.subject());
                                        let predicate =
                                            convert_predicate_to_rdf_term(triple.predicate());
                                        let object = convert_object_to_rdf_term(triple.object());

                                        crate::executor::SparqlValue::quoted_triple(
                                            subject, predicate, object,
                                        )
                                    }
                                };
                                (var.clone(), sparql_value)
                            })
                            .collect()
                    })
                    .collect();

                let cached_result = crate::executor::SparqlResults {
                    head: crate::executor::SparqlHead {
                        vars: result_bindings
                            .first()
                            .map(|binding| binding.keys().cloned().collect())
                            .unwrap_or_default(),
                    },
                    results: crate::executor::SparqlResultsData {
                        bindings: sparql_bindings,
                    },
                };

                // Cache with default TTL
                self.cache
                    .put_query_result(&cache_key, QueryResultCache::Sparql(cached_result), None)
                    .await;
            }
        }

        // Record metrics
        let execution_time = start_time.elapsed();
        self.monitor
            .record_query_execution("sparql", execution_time, final_result.is_success())
            .await;

        Ok(final_result)
    }

    /// Execute a federated GraphQL query
    pub async fn execute_graphql(
        &self,
        query: &str,
        variables: Option<serde_json::Value>,
    ) -> Result<FederatedResult> {
        let start_time = Instant::now();

        // Parse and analyze the GraphQL query
        let query_info = self
            .query_planner
            .analyze_graphql(query, variables.as_ref())
            .await?;

        // Plan the federated execution
        let registry = self.service_registry.read().await;
        let execution_plan = self
            .query_planner
            .plan_graphql(&query_info, &registry)
            .await?;
        drop(registry);

        // Execute the plan using GraphQL federation
        let partial_results = self
            .graphql_federation
            .execute_federated(&execution_plan)
            .await?;

        // Integrate results
        let final_result = self
            .integrator
            .integrate_graphql_results(partial_results)
            .await?;

        // Record metrics
        let execution_time = start_time.elapsed();
        self.monitor
            .record_query_execution("graphql", execution_time, final_result.is_success())
            .await;

        Ok(final_result)
    }

    /// Get federation statistics and health information
    pub async fn get_stats(&self) -> Result<FederationStats> {
        let registry_stats = {
            let registry = self.service_registry.read().await;
            registry.get_stats().await
        };

        let monitor_stats = self.monitor.get_stats().await;

        Ok(FederationStats {
            registry: registry_stats?,
            monitor: monitor_stats,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Perform health check on all registered services
    pub async fn health_check(&self) -> Result<HealthStatus> {
        let registry = self.service_registry.read().await;
        let health_statuses = registry.health_check().await?;

        // Convert Vec<HealthStatus> to a single HealthStatus
        let overall_status = if health_statuses
            .iter()
            .all(|s| matches!(s.status, crate::service_registry::HealthState::Healthy))
        {
            ServiceStatus::Healthy
        } else if health_statuses
            .iter()
            .any(|s| matches!(s.status, crate::service_registry::HealthState::Healthy))
        {
            ServiceStatus::Degraded
        } else {
            ServiceStatus::Unavailable
        };

        let total_services = health_statuses.len();
        let healthy_services = health_statuses
            .iter()
            .filter(|h| matches!(h.status, crate::service_registry::HealthState::Healthy))
            .count();

        let service_statuses = health_statuses
            .iter()
            .map(|h| {
                let status = match h.status {
                    crate::service_registry::HealthState::Healthy => ServiceStatus::Healthy,
                    crate::service_registry::HealthState::Degraded => ServiceStatus::Degraded,
                    crate::service_registry::HealthState::Unhealthy => ServiceStatus::Unavailable,
                    crate::service_registry::HealthState::Unknown => ServiceStatus::Unknown,
                };
                (h.service_id.clone(), status)
            })
            .collect();

        Ok(HealthStatus {
            overall_status,
            service_statuses,
            total_services,
            healthy_services,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Update service capabilities through discovery
    pub async fn discover_services(&self) -> Result<()> {
        // Fixed: lib.rs now uses FederatedServiceRegistry from service.rs
        // discovery.rs uses ServiceRegistry from service_registry.rs - different purposes
        warn!("Service discovery temporarily disabled due to type conflicts");
        Ok(())
    }

    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> CacheStats {
        self.cache.get_stats().await
    }

    /// Invalidate cache for a specific service
    pub async fn invalidate_service_cache(&self, service_id: &str) {
        self.cache.invalidate_service(service_id).await;
    }

    /// Invalidate all query caches
    pub async fn invalidate_query_cache(&self) {
        self.cache.invalidate_queries().await;
    }

    /// Warm up the cache with commonly used data
    pub async fn warmup_cache(&self) -> Result<()> {
        self.cache.warmup().await
    }

    /// Clean up expired cache entries
    pub async fn cleanup_cache(&self) {
        self.cache.cleanup_expired().await;
    }

    /// Start automatic service discovery
    pub async fn start_auto_discovery(&self, config: AutoDiscoveryConfig) -> Result<()> {
        let mut auto_discovery_guard = self.auto_discovery.write().await;

        if auto_discovery_guard.is_some() {
            return Err(anyhow!("Auto-discovery is already running"));
        }

        let mut discovery = AutoDiscovery::new(config);
        let mut receiver = discovery.start().await?;

        let registry = self.service_registry.clone();
        let service_discovery = ServiceDiscovery::new();

        // Spawn task to handle discovered services
        tokio::spawn(async move {
            while let Some(discovered) = receiver.recv().await {
                info!(
                    "Auto-discovered service: {} via {:?}",
                    discovered.url, discovered.discovery_method
                );

                // Register the discovered service
                if let Ok(Some(service)) = service_discovery
                    .discover_service_at_endpoint(&discovered.url)
                    .await
                {
                    let registry_guard = registry.write().await;
                    if let Err(e) = registry_guard.register(service).await {
                        warn!("Failed to register auto-discovered service: {}", e);
                    }
                }
            }
        });

        *auto_discovery_guard = Some(discovery);
        info!("Auto-discovery started");
        Ok(())
    }

    /// Stop automatic service discovery
    pub async fn stop_auto_discovery(&self) -> Result<()> {
        let mut auto_discovery_guard = self.auto_discovery.write().await;

        match auto_discovery_guard.take() {
            Some(mut discovery) => {
                discovery.stop().await;
                info!("Auto-discovery stopped");
                Ok(())
            }
            _ => Err(anyhow!("Auto-discovery is not running")),
        }
    }

    /// Get auto-discovered services
    pub async fn get_auto_discovered_services(&self) -> Result<Vec<DiscoveredEndpoint>> {
        let auto_discovery_guard = self.auto_discovery.read().await;

        if let Some(ref discovery) = *auto_discovery_guard {
            Ok(discovery.get_discovered_services().await)
        } else {
            Err(anyhow!("Auto-discovery is not running"))
        }
    }

    /// Assess capabilities of a registered service
    pub async fn assess_service_capabilities(&self, service_id: &str) -> Result<AssessmentResult> {
        let registry = self.service_registry.read().await;
        let service = registry
            .get_service(service_id)
            .ok_or_else(|| anyhow!("Service {} not found", service_id))?
            .clone();
        drop(registry);

        let assessor = CapabilityAssessor::new();
        let assessment = assessor.assess_service(&service).await?;

        // Update service with detected capabilities
        let registry = self.service_registry.read().await;
        registry.update_service_capabilities(service_id, &assessment.detected_capabilities)?;
        drop(registry);

        info!(
            "Capability assessment completed for service: {} - Updated with {} detected capabilities",
            service_id,
            assessment.detected_capabilities.len()
        );
        Ok(assessment)
    }

    /// Assess all registered services
    pub async fn assess_all_services(&self) -> Result<Vec<AssessmentResult>> {
        let service_ids: Vec<String> = {
            let registry = self.service_registry.read().await;
            registry
                .get_all_services()
                .into_iter()
                .map(|s| s.id.clone())
                .collect()
        };

        let mut results = Vec::new();
        for service_id in service_ids {
            match self.assess_service_capabilities(&service_id).await {
                Ok(assessment) => results.push(assessment),
                Err(e) => warn!("Failed to assess service {}: {}", service_id, e),
            }
        }

        Ok(results)
    }

    /// Enable vector similarity federation
    pub async fn enable_vector_federation(&self, config: VectorFederationConfig) -> Result<()> {
        let vector_federation =
            VectorSimilarityFederation::new(config, self.service_registry.clone()).await?;

        let mut vec_fed = self.vector_federation.write().await;
        *vec_fed = Some(vector_federation);

        info!("Vector similarity federation enabled");
        Ok(())
    }

    /// Register a vector-enabled service
    pub async fn register_vector_service(&self, metadata: VectorServiceMetadata) -> Result<()> {
        let vec_fed = self.vector_federation.read().await;
        if let Some(ref federation) = *vec_fed {
            federation.register_vector_service(metadata).await
        } else {
            Err(anyhow!("Vector federation is not enabled"))
        }
    }

    /// Execute semantic query routing
    pub async fn semantic_query_routing(&self, query: &str) -> Result<Vec<String>> {
        let vec_fed = self.vector_federation.read().await;
        if let Some(ref federation) = *vec_fed {
            let query_embedding = federation.generate_query_embedding(query).await?;
            federation
                .semantic_query_routing(&query_embedding, query)
                .await
        } else {
            Ok(Vec::new()) // Return empty if vector federation is disabled
        }
    }

    /// Get vector federation statistics
    pub async fn get_vector_statistics(&self) -> Result<Option<VectorFederationStats>> {
        let vec_fed = self.vector_federation.read().await;
        if let Some(ref federation) = *vec_fed {
            Ok(Some(federation.get_statistics().await?))
        } else {
            Ok(None)
        }
    }
}

impl Default for FederationEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for the federation engine
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FederationConfig {
    pub registry_config: RegistryConfig,
    pub planner_config: PlannerConfig,
    pub executor_config: FederatedExecutorConfig,
    pub integrator_config: ResultIntegratorConfig,
    pub graphql_config: GraphQLFederationConfig,
    pub monitor_config: FederationMonitorConfig,
    pub cache_config: CacheConfig,
}

/// Result of a federated query execution
#[derive(Debug, Clone)]
pub struct FederatedResult {
    /// The integrated query results
    pub data: QueryResult,
    /// Metadata about the execution
    pub metadata: ExecutionMetadata,
    /// Any errors or warnings from the execution
    pub errors: Vec<FederationError>,
}

impl FederatedResult {
    /// Check if the execution was successful (no critical errors)
    pub fn is_success(&self) -> bool {
        !self.errors.iter().any(|e| e.is_critical())
    }

    /// Get the number of results
    pub fn result_count(&self) -> usize {
        match &self.data {
            QueryResult::Sparql(results) => results.len(),
            QueryResult::GraphQL(result) => {
                if result.is_object() {
                    1
                } else if result.is_array() {
                    result.as_array().map(|a| a.len()).unwrap_or(0)
                } else {
                    0
                }
            }
        }
    }
}

/// Enumeration of different query result types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryResult {
    Sparql(Vec<HashMap<String, Term>>),
    GraphQL(serde_json::Value),
}

/// Metadata about query execution
#[derive(Debug, Clone)]
pub struct ExecutionMetadata {
    /// Total execution time
    pub execution_time: Duration,
    /// Number of services involved
    pub services_used: usize,
    /// Number of subqueries executed
    pub subqueries_executed: usize,
    /// Whether results were cached
    pub cache_hit: bool,
    /// Execution plan used
    pub plan_summary: String,
}

/// Federation-specific error types
#[derive(Debug, Clone, thiserror::Error)]
pub enum FederationError {
    #[error("Service unavailable: {service_id}")]
    ServiceUnavailable { service_id: String },

    #[error("Query planning failed: {reason}")]
    PlanningFailed { reason: String },

    #[error("Execution timeout after {timeout:?}")]
    ExecutionTimeout { timeout: Duration },

    #[error("Result integration failed: {reason}")]
    IntegrationFailed { reason: String },

    #[error("Partial results: {successful_services}/{total_services} services responded")]
    PartialResults {
        successful_services: usize,
        total_services: usize,
    },

    #[error("Schema conflict: {conflict}")]
    SchemaConflict { conflict: String },

    #[error("Authentication failed for service: {service_id}")]
    AuthenticationFailed { service_id: String },

    #[error("Rate limit exceeded for service: {service_id}")]
    RateLimitExceeded { service_id: String },
}

impl FederationError {
    /// Check if this error should cause the entire query to fail
    pub fn is_critical(&self) -> bool {
        match self {
            FederationError::ServiceUnavailable { .. } => false, // Can continue with other services
            FederationError::PlanningFailed { .. } => true,
            FederationError::ExecutionTimeout { .. } => true,
            FederationError::IntegrationFailed { .. } => true,
            FederationError::PartialResults { .. } => false, // Warning, not critical
            FederationError::SchemaConflict { .. } => true,
            FederationError::AuthenticationFailed { .. } => false,
            FederationError::RateLimitExceeded { .. } => false,
        }
    }
}

/// Statistics about federation performance
#[derive(Debug, Clone, Serialize)]
pub struct FederationStats {
    pub registry: RegistryStats,
    pub monitor: MonitorStats,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Health status of the federation system
#[derive(Debug, Clone, Serialize)]
pub struct HealthStatus {
    pub overall_status: ServiceStatus,
    pub service_statuses: HashMap<String, ServiceStatus>,
    pub total_services: usize,
    pub healthy_services: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Service status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceStatus {
    Healthy,
    Degraded,
    Unavailable,
    Unknown,
}

/// Spatial coverage for geospatial services
#[derive(Debug, Clone)]
pub struct SpatialCoverage {
    pub coverage_type: SpatialCoverageType,
    pub min_lat: f64,
    pub max_lat: f64,
    pub min_lon: f64,
    pub max_lon: f64,
}

/// Types of spatial coverage
#[derive(Debug, Clone)]
pub enum SpatialCoverageType {
    BoundingBox,
    Circle,
    Polygon,
}

/// Numeric range for numeric services
#[derive(Debug, Clone)]
pub struct NumericRange {
    pub min: f64,
    pub max: f64,
    pub data_type: String,
}

/// Extended service metadata for enhanced optimization
#[derive(Debug, Clone, Default)]
pub struct ExtendedServiceMetadata {
    pub estimated_triple_count: Option<u64>,
    pub domain_specializations: Option<Vec<String>>,
    pub known_vocabularies: Option<Vec<String>>,
    pub schema_mappings: Option<HashMap<String, String>>,
    pub performance_history: Option<HashMap<String, PerformanceRecord>>,
    pub successful_query_patterns: Option<Vec<PatternFeatures>>,
    pub temporal_coverage: Option<TemporalRange>,
    pub spatial_coverage: Option<SpatialCoverage>,
    pub numeric_ranges: Option<Vec<NumericRange>>,
}

/// Performance record for service history
#[derive(Debug, Clone)]
pub struct PerformanceRecord {
    pub avg_response_time_score: f64,
    pub success_rate: f64,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Pattern features for ML analysis (reexport from materialized_views)
pub use materialized_views::PatternFeatures;
pub use materialized_views::TemporalRange;

/// Helper function to convert oxirs_core::Term to our RdfTerm representation for RDF-star support
#[allow(dead_code)]
fn convert_core_term_to_rdf_term(term: &oxirs_core::Term) -> RdfTerm {
    match term {
        oxirs_core::Term::NamedNode(node) => RdfTerm::IRI(node.as_str().to_string()),
        oxirs_core::Term::BlankNode(node) => RdfTerm::BlankNode(node.as_str().to_string()),
        oxirs_core::Term::Literal(literal) => {
            let value = literal.value().to_string();
            let datatype = if literal.datatype() != *oxirs_core::vocab::xsd::STRING {
                Some(literal.datatype().as_str().to_string())
            } else {
                None
            };
            let lang = literal.language().map(|l| l.to_string());

            RdfTerm::Literal {
                value,
                datatype,
                lang,
            }
        }
        oxirs_core::Term::QuotedTriple(triple) => {
            let subject = Box::new(convert_subject_to_rdf_term(triple.subject()));
            let predicate = Box::new(convert_predicate_to_rdf_term(triple.predicate()));
            let object = Box::new(convert_object_to_rdf_term(triple.object()));

            RdfTerm::QuotedTriple(QuotedTripleValue {
                subject,
                predicate,
                object,
            })
        }
        oxirs_core::Term::Variable(var) => {
            // Variables are represented as literals in RDF terms
            RdfTerm::Literal {
                value: format!("?{}", var.as_str()),
                datatype: None,
                lang: None,
            }
        }
    }
}

/// Helper function to convert oxirs_core::Subject to our RdfTerm representation
fn convert_subject_to_rdf_term(subject: &oxirs_core::Subject) -> RdfTerm {
    match subject {
        oxirs_core::Subject::NamedNode(node) => RdfTerm::IRI(node.as_str().to_string()),
        oxirs_core::Subject::BlankNode(node) => RdfTerm::BlankNode(node.as_str().to_string()),
        oxirs_core::Subject::QuotedTriple(triple) => {
            let subject = Box::new(convert_subject_to_rdf_term(triple.subject()));
            let predicate = Box::new(convert_predicate_to_rdf_term(triple.predicate()));
            let object = Box::new(convert_object_to_rdf_term(triple.object()));
            RdfTerm::QuotedTriple(QuotedTripleValue {
                subject,
                predicate,
                object,
            })
        }
        oxirs_core::Subject::Variable(var) => RdfTerm::Literal {
            value: format!("?{}", var.as_str()),
            datatype: None,
            lang: None,
        },
    }
}

/// Helper function to convert oxirs_core::Predicate to our RdfTerm representation
fn convert_predicate_to_rdf_term(predicate: &oxirs_core::Predicate) -> RdfTerm {
    match predicate {
        oxirs_core::Predicate::NamedNode(node) => RdfTerm::IRI(node.as_str().to_string()),
        oxirs_core::Predicate::Variable(var) => RdfTerm::Literal {
            value: format!("?{}", var.as_str()),
            datatype: None,
            lang: None,
        },
    }
}

/// Helper function to convert oxirs_core::Object to our RdfTerm representation
fn convert_object_to_rdf_term(object: &oxirs_core::Object) -> RdfTerm {
    match object {
        oxirs_core::Object::NamedNode(node) => RdfTerm::IRI(node.as_str().to_string()),
        oxirs_core::Object::BlankNode(node) => RdfTerm::BlankNode(node.as_str().to_string()),
        oxirs_core::Object::Literal(literal) => {
            let value = literal.value().to_string();
            let datatype = if literal.datatype() != *oxirs_core::vocab::xsd::STRING {
                Some(literal.datatype().as_str().to_string())
            } else {
                None
            };
            let lang = literal.language().map(|l| l.to_string());
            RdfTerm::Literal {
                value,
                datatype,
                lang,
            }
        }
        oxirs_core::Object::QuotedTriple(triple) => {
            let subject = Box::new(convert_subject_to_rdf_term(triple.subject()));
            let predicate = Box::new(convert_predicate_to_rdf_term(triple.predicate()));
            let object = Box::new(convert_object_to_rdf_term(triple.object()));
            RdfTerm::QuotedTriple(QuotedTripleValue {
                subject,
                predicate,
                object,
            })
        }
        oxirs_core::Object::Variable(var) => RdfTerm::Literal {
            value: format!("?{}", var.as_str()),
            datatype: None,
            lang: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_federation_engine_creation() {
        let engine = FederationEngine::new();
        let stats = engine.get_stats().await.unwrap();

        assert_eq!(stats.registry.total_sparql_endpoints, 0);
    }

    #[tokio::test]
    async fn test_federation_engine_with_config() {
        let config = FederationConfig::default();
        let engine = FederationEngine::with_config(config);
        let health = engine.health_check().await.unwrap();

        assert_eq!(health.overall_status, ServiceStatus::Healthy);
        assert_eq!(health.total_services, 0);
    }

    #[tokio::test]
    async fn test_federation_error_criticality() {
        let critical_error = FederationError::PlanningFailed {
            reason: "Test error".to_string(),
        };
        assert!(critical_error.is_critical());

        let non_critical_error = FederationError::ServiceUnavailable {
            service_id: "test-service".to_string(),
        };
        assert!(!non_critical_error.is_critical());
    }

    #[tokio::test]
    async fn test_federated_result_success() {
        let result = FederatedResult {
            data: QueryResult::Sparql(vec![]),
            metadata: ExecutionMetadata {
                execution_time: Duration::from_millis(100),
                services_used: 1,
                subqueries_executed: 1,
                cache_hit: false,
                plan_summary: "Test plan".to_string(),
            },
            errors: vec![],
        };

        assert!(result.is_success());
        assert_eq!(result.result_count(), 0);
    }
}
