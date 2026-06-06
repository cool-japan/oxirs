//! Type definitions for the materialized view subsystem.
//!
//! This sibling module hosts the configuration, view, view-data, metadata,
//! maintenance, cost, dependency, and recommendation types used by the
//! [`materialized_views`](crate::materialized_views) facade.  The runtime
//! manager, storage, scheduler, and recommendation engine implementations
//! live in their own sibling modules.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

use crate::algebra::Solution;
use crate::algebra::{Algebra, Expression, TriplePattern, Variable};
use crate::cost_model::{CostEstimate, CostModel};
use crate::statistics_collector::StatisticsCollector;

/// Materialized view manager for query optimization
pub struct MaterializedViewManager {
    pub(crate) config: MaterializedViewConfig,
    pub(crate) views: Arc<RwLock<HashMap<String, MaterializedView>>>,
    pub(crate) view_storage: Arc<RwLock<ViewStorage>>,
    pub(crate) rewriter: QueryRewriter,
    pub(crate) maintenance_scheduler: MaintenanceScheduler,
    pub(crate) cost_model: Arc<Mutex<CostModel>>,
    #[allow(dead_code)]
    pub(crate) statistics_collector: Arc<StatisticsCollector>,
    pub(crate) usage_statistics: Arc<RwLock<ViewUsageStatistics>>,
    pub(crate) recommendation_engine: ViewRecommendationEngine,
}

/// Configuration for materialized view management
#[derive(Debug, Clone)]
pub struct MaterializedViewConfig {
    /// Maximum number of materialized views to maintain
    pub max_views: usize,
    /// Maximum memory usage for views (bytes)
    pub max_memory_usage: usize,
    /// Enable automatic view creation based on query patterns
    pub auto_view_creation: bool,
    /// Maintenance strategy for view updates
    pub maintenance_strategy: MaintenanceStrategy,
    /// Threshold for view utilization before considering removal
    pub utilization_threshold: f64,
    /// Maximum staleness allowed for views (seconds)
    pub max_staleness: Duration,
    /// Enable cost-based view selection
    pub cost_based_selection: bool,
    /// Enable incremental maintenance
    pub incremental_maintenance: bool,
}

impl Default for MaterializedViewConfig {
    fn default() -> Self {
        Self {
            max_views: 100,
            max_memory_usage: 2 * 1024 * 1024 * 1024, // 2GB
            auto_view_creation: true,
            maintenance_strategy: MaintenanceStrategy::Lazy,
            utilization_threshold: 0.1,               // 10% utilization
            max_staleness: Duration::from_secs(3600), // 1 hour
            cost_based_selection: true,
            incremental_maintenance: true,
        }
    }
}

/// Maintenance strategies for materialized views
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaintenanceStrategy {
    /// Update views immediately when base data changes
    Immediate,
    /// Update views periodically
    Periodic(Duration),
    /// Update views when accessed and stale
    Lazy,
    /// Update views based on cost analysis
    CostBased,
    /// Hybrid approach combining multiple strategies
    Hybrid,
}

/// Definition of a materialized view
#[derive(Debug, Clone)]
pub struct MaterializedView {
    /// Unique identifier for the view
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Algebra expression defining the view query
    pub definition: Algebra,
    /// Current materialized data
    pub data: ViewData,
    /// Metadata about the view
    pub metadata: ViewMetadata,
    /// Maintenance information
    pub maintenance_info: MaintenanceInfo,
    /// Cost estimates for using this view
    pub cost_estimates: ViewCostEstimates,
    /// Dependencies on base data
    pub dependencies: ViewDependencies,
}

/// Materialized data for a view
#[derive(Debug, Clone)]
pub struct ViewData {
    /// Result set from the view query
    pub results: Solution,
    /// Size of the materialized data in bytes
    pub size_bytes: usize,
    /// Number of rows in the view
    pub row_count: usize,
    /// Timestamp when data was last materialized
    pub materialized_at: SystemTime,
    /// Checksum for data integrity
    pub checksum: u64,
}

/// Metadata about a materialized view
#[derive(Debug, Clone)]
pub struct ViewMetadata {
    /// When the view was created
    pub created_at: SystemTime,
    /// Who or what created the view
    pub created_by: String,
    /// Description of the view's purpose
    pub description: String,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Priority for maintenance (higher = more important)
    pub priority: u8,
    /// Expected lifetime of the view
    pub expected_lifetime: Duration,
}

/// Maintenance information for a view
#[derive(Debug, Clone)]
pub struct MaintenanceInfo {
    /// Last time the view was updated
    pub last_updated: SystemTime,
    /// Next scheduled maintenance time
    pub next_maintenance: Option<SystemTime>,
    /// Maintenance strategy for this specific view
    pub strategy: MaintenanceStrategy,
    /// Number of times the view has been updated
    pub update_count: usize,
    /// Total time spent maintaining the view
    pub total_maintenance_time: Duration,
    /// Whether the view needs updating
    pub needs_update: bool,
    /// Incremental update state
    pub incremental_state: Option<IncrementalState>,
}

/// State for incremental view maintenance
#[derive(Debug, Clone)]
pub struct IncrementalState {
    /// Last processed transaction ID
    pub last_transaction_id: u64,
    /// Change log for incremental updates
    pub change_log: Vec<ChangeLogEntry>,
    /// Delta computation state
    pub delta_state: DeltaState,
}

/// Entry in the change log for incremental maintenance
#[derive(Debug, Clone)]
pub struct ChangeLogEntry {
    /// Type of change (insert, delete, update)
    pub change_type: ChangeType,
    /// Affected triple or quad
    pub affected_data: TriplePattern,
    /// Timestamp of the change
    pub timestamp: SystemTime,
    /// Transaction ID
    pub transaction_id: u64,
}

/// Types of changes for incremental maintenance
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeType {
    Insert,
    Delete,
    Update,
}

/// Delta computation state for incremental updates
#[derive(Debug, Clone)]
pub struct DeltaState {
    /// Positive delta (insertions)
    pub positive_delta: Solution,
    /// Negative delta (deletions)
    pub negative_delta: Solution,
    /// Dirty flags for affected partitions
    pub dirty_partitions: HashSet<u64>,
}

/// Cost estimates for using a materialized view
#[derive(Debug, Clone)]
pub struct ViewCostEstimates {
    /// Cost of accessing the view
    pub access_cost: CostEstimate,
    /// Cost of maintaining the view
    pub maintenance_cost: CostEstimate,
    /// Storage cost (memory/disk usage)
    pub storage_cost: f64,
    /// Cost benefit compared to computing from scratch
    pub benefit_ratio: f64,
    /// Last time costs were estimated
    pub last_estimated: SystemTime,
}

/// Dependencies of a view on base data
#[derive(Debug, Clone)]
pub struct ViewDependencies {
    /// Base tables/graphs referenced by the view
    pub base_tables: Vec<String>,
    /// Specific triple patterns the view depends on
    pub dependent_patterns: Vec<TriplePattern>,
    /// Variables that affect view results
    pub dependent_variables: HashSet<Variable>,
    /// Join dependencies
    pub join_dependencies: Vec<JoinDependency>,
}

/// Join dependency information
#[derive(Debug, Clone)]
pub struct JoinDependency {
    /// Left side of the join
    pub left_pattern: TriplePattern,
    /// Right side of the join
    pub right_pattern: TriplePattern,
    /// Join variables
    pub join_variables: Vec<Variable>,
    /// Estimated selectivity
    pub selectivity: f64,
}

/// Storage for materialized view data
#[derive(Debug)]
pub struct ViewStorage {
    /// In-memory storage for view data
    pub(crate) memory_storage: HashMap<String, ViewData>,
    /// Disk-based storage path
    pub(crate) disk_storage_path: Option<std::path::PathBuf>,
    /// Maximum memory usage allowed
    pub(crate) max_memory: usize,
    /// Current memory usage
    pub(crate) memory_usage: usize,
    /// Storage statistics
    pub(crate) storage_stats: StorageStatistics,
}

/// Statistics about view storage
#[derive(Debug, Clone, Default)]
pub struct StorageStatistics {
    /// Total memory usage
    pub total_memory_usage: usize,
    /// Total disk usage
    pub total_disk_usage: usize,
    /// Number of views stored in memory
    pub memory_view_count: usize,
    /// Number of views stored on disk
    pub disk_view_count: usize,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Average access time
    pub average_access_time: Duration,
}

/// Query rewriter for utilizing materialized views
pub struct QueryRewriter {
    pub(crate) view_index: ViewIndex,
    #[allow(dead_code)]
    pub(crate) rewrite_rules: Vec<RewriteRule>,
    #[allow(dead_code)]
    pub(crate) cost_threshold: f64,
}

/// Index for efficient view lookup during query rewriting
#[derive(Debug)]
pub struct ViewIndex {
    /// Index by pattern structure
    #[allow(dead_code)]
    pub(crate) pattern_index: HashMap<String, Vec<String>>,
    /// Index by variables
    #[allow(dead_code)]
    pub(crate) variable_index: HashMap<Variable, Vec<String>>,
    /// Index by predicates
    #[allow(dead_code)]
    pub(crate) predicate_index: HashMap<String, Vec<String>>,
    /// Index by query characteristics
    pub(crate) characteristic_index: HashMap<QueryCharacteristic, Vec<String>>,
}

/// Query characteristics for view indexing
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum QueryCharacteristic {
    HasJoin,
    HasFilter,
    HasAggregation,
    HasUnion,
    PatternCount(usize),
    VariableCount(usize),
}

/// Rule for query rewriting
#[derive(Debug, Clone)]
pub struct RewriteRule {
    /// Name of the rule
    pub name: String,
    /// Pattern to match
    pub pattern_matcher: PatternMatcher,
    /// Rewrite transformation
    pub transformation: RewriteTransformation,
    /// Cost threshold for applying the rule
    pub cost_threshold: f64,
    /// Priority of the rule
    pub priority: u8,
}

/// Pattern matcher for rewrite rules
#[derive(Debug, Clone)]
pub enum PatternMatcher {
    /// Exact algebra match
    ExactMatch(Algebra),
    /// Structural pattern match
    StructuralMatch(AlgebraPattern),
    /// Semantic equivalence match
    SemanticMatch(SemanticPattern),
    /// Custom matcher function
    Custom(String), // Function name for custom matching
}

/// Structural pattern for matching algebra expressions
#[derive(Debug, Clone)]
pub struct AlgebraPattern {
    /// Pattern type
    pub pattern_type: AlgebraPatternType,
    /// Sub-patterns
    pub sub_patterns: Vec<AlgebraPattern>,
    /// Variable bindings
    pub bindings: HashMap<String, Variable>,
}

/// Types of algebra patterns
#[derive(Debug, Clone)]
pub enum AlgebraPatternType {
    BGP,
    Join,
    Union,
    Filter,
    Any,
}

/// Semantic pattern for advanced matching
#[derive(Debug, Clone)]
pub struct SemanticPattern {
    /// Semantic equivalence rules
    pub equivalence_rules: Vec<String>,
    /// Containment relationships
    pub containment_rules: Vec<String>,
}

/// Transformation for query rewriting
#[derive(Debug, Clone)]
pub enum RewriteTransformation {
    /// Replace with view access
    ReplaceWithView(String),
    /// Partial replacement
    PartialReplace(Box<PartialReplacement>),
    /// Join with view
    JoinWithView(JoinTransformation),
    /// Union with view
    UnionWithView(UnionTransformation),
}

/// Partial replacement transformation
#[derive(Debug, Clone)]
pub struct PartialReplacement {
    /// View to use for partial replacement
    pub view_id: String,
    /// Remaining query parts
    pub remaining_query: Algebra,
    /// How to combine view results with remaining query
    pub combination_strategy: CombinationStrategy,
}

/// Strategy for combining view results with remaining query
#[derive(Debug, Clone)]
pub enum CombinationStrategy {
    Join(Vec<Variable>),
    Union,
    Filter(Expression),
}

/// Join transformation with a view
#[derive(Debug, Clone)]
pub struct JoinTransformation {
    /// View to join with
    pub view_id: String,
    /// Join variables
    pub join_variables: Vec<Variable>,
    /// Join type
    pub join_type: JoinType,
}

/// Types of joins for view transformations
#[derive(Debug, Clone)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
}

/// Union transformation with a view
#[derive(Debug, Clone)]
pub struct UnionTransformation {
    /// View to union with
    pub view_id: String,
    /// Whether to apply DISTINCT
    pub distinct: bool,
}

/// Maintenance scheduler for materialized views
pub struct MaintenanceScheduler {
    pub(crate) scheduled_tasks: Arc<RwLock<VecDeque<MaintenanceTask>>>,
    #[allow(dead_code)]
    pub(crate) active_tasks: Arc<RwLock<HashMap<String, ActiveTask>>>,
    #[allow(dead_code)]
    pub(crate) config: SchedulerConfig,
}

/// Configuration for the maintenance scheduler
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum concurrent maintenance tasks
    pub max_concurrent_tasks: usize,
    /// Default maintenance interval
    pub default_interval: Duration,
    /// Priority threshold for immediate scheduling
    pub priority_threshold: u8,
    /// Resource limits for maintenance
    pub resource_limits: ResourceLimits,
}

/// Resource limits for maintenance operations
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Maximum CPU usage percentage
    pub max_cpu_usage: f64,
    /// Maximum memory usage for maintenance
    pub max_memory_usage: usize,
    /// Maximum I/O bandwidth
    pub max_io_bandwidth: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_cpu_usage: 50.0,
            max_memory_usage: 1024 * 1024 * 512, // 512MB
            max_io_bandwidth: 1024 * 1024 * 100, // 100MB/s
        }
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: 4,
            default_interval: Duration::from_secs(3600), // 1 hour
            priority_threshold: 8,
            resource_limits: ResourceLimits::default(),
        }
    }
}

/// Maintenance task for a view
#[derive(Debug, Clone)]
pub struct MaintenanceTask {
    /// View to maintain
    pub view_id: String,
    /// Type of maintenance
    pub task_type: MaintenanceTaskType,
    /// Priority (higher = more urgent)
    pub priority: u8,
    /// Scheduled execution time
    pub scheduled_time: SystemTime,
    /// Estimated execution time
    pub estimated_duration: Duration,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
}

/// Types of maintenance tasks
#[derive(Debug, Clone)]
pub enum MaintenanceTaskType {
    /// Full refresh of the view
    FullRefresh,
    /// Incremental update
    IncrementalUpdate,
    /// Recompute statistics
    StatisticsUpdate,
    /// Optimize view storage
    StorageOptimization,
    /// Validate view integrity
    IntegrityCheck,
}

/// Resource requirements for a maintenance task
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    /// Estimated CPU usage
    pub cpu_usage: f64,
    /// Estimated memory usage
    pub memory_usage: usize,
    /// Estimated I/O operations
    pub io_operations: usize,
    /// Network bandwidth requirements
    pub network_bandwidth: usize,
}

/// Active maintenance task
#[derive(Debug)]
pub struct ActiveTask {
    /// Task information
    pub task: MaintenanceTask,
    /// Start time
    pub start_time: Instant,
    /// Current progress (0.0 to 1.0)
    pub progress: f64,
    /// Cancellation flag
    pub cancelled: bool,
}

/// Usage statistics for views
#[derive(Debug, Default)]
pub struct ViewUsageStatistics {
    /// Access count per view
    pub(crate) access_counts: HashMap<String, usize>,
    /// Total query time saved per view
    pub(crate) time_saved: HashMap<String, Duration>,
    /// Hit rate per view
    pub(crate) hit_rates: HashMap<String, f64>,
    /// Cost benefit per view
    pub(crate) cost_benefits: HashMap<String, f64>,
    /// Usage patterns over time
    pub(crate) usage_history: HashMap<String, VecDeque<UsageRecord>>,
}

/// Record of view usage
#[derive(Debug, Clone)]
pub struct UsageRecord {
    /// Timestamp of usage
    pub timestamp: SystemTime,
    /// Query that used the view
    pub query_hash: u64,
    /// Time saved by using the view
    pub time_saved: Duration,
    /// Cost benefit achieved
    pub cost_benefit: f64,
}

/// Engine for recommending new materialized views
pub struct ViewRecommendationEngine {
    #[allow(dead_code)]
    pub(crate) query_patterns: Arc<RwLock<QueryPatternAnalyzer>>,
    #[allow(dead_code)]
    pub(crate) cost_analyzer: CostAnalyzer,
    #[allow(dead_code)]
    pub(crate) benefit_estimator: BenefitEstimator,
    #[allow(dead_code)]
    pub(crate) recommendation_cache: Arc<RwLock<HashMap<String, ViewRecommendation>>>,
}

/// Analyzer for query patterns
#[derive(Debug)]
pub struct QueryPatternAnalyzer {
    /// Observed query patterns
    #[allow(dead_code)]
    pub(crate) patterns: HashMap<String, QueryPattern>,
    /// Pattern frequency
    #[allow(dead_code)]
    pub(crate) pattern_frequency: HashMap<String, usize>,
    /// Pattern cost statistics
    #[allow(dead_code)]
    pub(crate) pattern_costs: HashMap<String, CostStatistics>,
}

/// Observed query pattern
#[derive(Debug, Clone)]
pub struct QueryPattern {
    /// Pattern signature
    pub signature: String,
    /// Algebra structure
    pub algebra_structure: Algebra,
    /// Common sub-patterns
    pub sub_patterns: Vec<SubPattern>,
    /// Variable usage patterns
    pub variable_patterns: VariablePattern,
    /// Join patterns
    pub join_patterns: Vec<JoinPattern>,
}

/// Sub-pattern within a query
#[derive(Debug, Clone)]
pub struct SubPattern {
    /// Pattern identifier
    pub id: String,
    /// Algebra expression
    pub algebra: Algebra,
    /// Frequency of occurrence
    pub frequency: usize,
    /// Estimated cost
    pub estimated_cost: f64,
}

/// Variable usage pattern
#[derive(Debug, Clone)]
pub struct VariablePattern {
    /// Variables used in the pattern
    pub variables: HashSet<Variable>,
    /// Variable binding patterns
    pub binding_patterns: HashMap<Variable, BindingPattern>,
    /// Variable selectivity
    pub variable_selectivity: HashMap<Variable, f64>,
}

/// Binding pattern for a variable
#[derive(Debug, Clone)]
pub enum BindingPattern {
    /// Always bound to constants
    Constant(Vec<String>),
    /// Bound through joins
    Join(Vec<Variable>),
    /// Bound through filters
    Filter(Vec<Expression>),
    /// Mixed binding pattern
    Mixed,
}

/// Join pattern in queries
#[derive(Debug, Clone)]
pub struct JoinPattern {
    /// Left side pattern
    pub left_pattern: TriplePattern,
    /// Right side pattern
    pub right_pattern: TriplePattern,
    /// Join variables
    pub join_variables: Vec<Variable>,
    /// Join selectivity
    pub selectivity: f64,
    /// Join cost
    pub cost: f64,
}

/// Cost statistics for query patterns
#[derive(Debug, Clone, Default)]
pub struct CostStatistics {
    /// Average execution cost
    pub average_cost: f64,
    /// Minimum execution cost
    pub min_cost: f64,
    /// Maximum execution cost
    pub max_cost: f64,
    /// Standard deviation
    pub std_deviation: f64,
    /// Number of samples
    pub sample_count: usize,
}

/// Cost analyzer for view recommendations
pub struct CostAnalyzer {
    #[allow(dead_code)]
    pub(crate) historical_costs: HashMap<String, Vec<f64>>,
    #[allow(dead_code)]
    pub(crate) cost_models: HashMap<String, CostModel>,
}

/// Benefit estimator for materialized views
pub struct BenefitEstimator {
    /// Historical benefit data
    #[allow(dead_code)]
    pub(crate) benefit_history: HashMap<String, Vec<f64>>,
    /// Benefit prediction models
    #[allow(dead_code)]
    pub(crate) prediction_models: HashMap<String, BenefitModel>,
}

/// Model for predicting view benefits
#[derive(Debug, Clone)]
pub struct BenefitModel {
    /// Model type
    pub model_type: BenefitModelType,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    /// Accuracy metrics
    pub accuracy: f64,
}

/// Types of benefit prediction models
#[derive(Debug, Clone)]
pub enum BenefitModelType {
    Linear,
    Polynomial,
    ExponentialDecay,
    MachineLearning(String), // ML model type
}

/// Recommendation for a new materialized view
#[derive(Debug, Clone)]
pub struct ViewRecommendation {
    /// Proposed view definition
    pub view_definition: Algebra,
    /// Estimated benefit
    pub estimated_benefit: f64,
    /// Confidence in the recommendation
    pub confidence: f64,
    /// Estimated creation cost
    pub creation_cost: f64,
    /// Estimated maintenance cost
    pub maintenance_cost: f64,
    /// Recommended maintenance strategy
    pub maintenance_strategy: MaintenanceStrategy,
    /// Supporting query patterns
    pub supporting_patterns: Vec<String>,
    /// Justification for the recommendation
    pub justification: String,
}

/// Statistics for view usage
#[derive(Debug, Clone)]
pub struct ViewUsageStats {
    pub access_count: usize,
    pub total_time_saved: Duration,
    pub hit_rate: f64,
    pub cost_benefit: f64,
}
