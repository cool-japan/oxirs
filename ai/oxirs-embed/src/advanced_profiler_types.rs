//! Advanced Profiler Types
//!
//! Profiling event types, metric types, trace types, configuration structs,
//! and all supporting enums for the advanced profiler subsystem.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for advanced profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Maximum number of concurrent profiling sessions
    pub max_sessions: usize,
    /// Sampling rate (0.0 to 1.0)
    pub sampling_rate: f64,
    /// Buffer size for performance data
    pub buffer_size: usize,
    /// Analysis window size in seconds
    pub analysis_window_seconds: u64,
    /// Enable memory profiling
    pub enable_memory_profiling: bool,
    /// Enable CPU profiling
    pub enable_cpu_profiling: bool,
    /// Enable GPU profiling
    pub enable_gpu_profiling: bool,
    /// Enable network profiling
    pub enable_network_profiling: bool,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            max_sessions: 10,
            sampling_rate: 0.01,
            buffer_size: 100000,
            analysis_window_seconds: 300,
            enable_memory_profiling: true,
            enable_cpu_profiling: true,
            enable_gpu_profiling: true,
            enable_network_profiling: true,
        }
    }
}

// ─── Session ──────────────────────────────────────────────────────────────────

/// Individual profiling session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingSession {
    pub session_id: String,
    pub name: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub status: SessionStatus,
    pub metrics: Vec<MetricDataPoint>,
    pub tags: HashMap<String, String>,
}

/// Status of a profiling session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionStatus {
    Active,
    Completed,
    Failed(String),
    Cancelled,
}

// ─── Metrics ──────────────────────────────────────────────────────────────────

/// Individual metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDataPoint {
    pub timestamp: DateTime<Utc>,
    pub metric_name: String,
    pub value: f64,
    pub unit: String,
    pub metadata: HashMap<String, String>,
    pub thread_id: Option<String>,
    pub component: String,
}

/// Collection statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CollectionStats {
    pub total_points: u64,
    pub collection_rate: f64,
    pub memory_usage_bytes: u64,
    pub drop_rate: f64,
}

/// Performance tracker for specific components
#[derive(Debug, Clone)]
pub struct PerformanceTracker {
    pub name: String,
    pub start_time: Instant,
    pub measurements: Vec<TimedMeasurement>,
    pub state: TrackerState,
}

/// Timed measurement
#[derive(Debug, Clone)]
pub struct TimedMeasurement {
    pub timestamp: Duration,
    pub measurement_type: MeasurementType,
    pub value: f64,
    pub context: String,
}

/// Types of measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeasurementType {
    Latency,
    Throughput,
    MemoryUsage,
    CpuUsage,
    GpuUsage,
    NetworkLatency,
    DiskIo,
    CacheHitRate,
    ErrorRate,
    QueueLength,
}

/// Tracker state
#[derive(Debug, Clone)]
pub enum TrackerState {
    Active,
    Paused,
    Stopped,
}

// ─── Analysis ─────────────────────────────────────────────────────────────────

/// Analysis algorithm interface
#[derive(Debug, Clone)]
pub struct AnalysisAlgorithm {
    pub name: String,
    pub algorithm_type: AlgorithmType,
    pub parameters: HashMap<String, f64>,
}

/// Types of analysis algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlgorithmType {
    TrendAnalysis,
    BottleneckDetection,
    PerformanceRegression,
    ResourceUtilization,
    CapacityPlanning,
    LoadBalancing,
}

// ─── Pattern Detection ────────────────────────────────────────────────────────

/// Detected performance pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePattern {
    pub id: String,
    pub pattern_type: PatternType,
    pub confidence: f64,
    pub time_window: (DateTime<Utc>, DateTime<Utc>),
    pub affected_components: Vec<String>,
    pub description: String,
}

/// Types of performance patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    PeriodicSpike,
    GradualDegradation,
    SuddenDrop,
    MemoryLeak,
    ThresholdBreach,
    LoadPattern,
    SeasonalVariation,
}

/// Pattern template for recognition
#[derive(Debug, Clone)]
pub struct PatternTemplate {
    pub name: String,
    pub signature: PatternSignature,
    pub criteria: MatchingCriteria,
}

/// Pattern signature
#[derive(Debug, Clone)]
pub struct PatternSignature {
    pub characteristics: Vec<StatisticalCharacteristic>,
    pub temporal_features: Vec<TemporalFeature>,
}

/// Statistical characteristic
#[derive(Debug, Clone)]
pub struct StatisticalCharacteristic {
    pub metric: String,
    pub property: StatisticalProperty,
    pub value_range: (f64, f64),
}

/// Statistical properties
#[derive(Debug, Clone)]
pub enum StatisticalProperty {
    Mean,
    Median,
    StandardDeviation,
    Variance,
    Skewness,
    Kurtosis,
    Percentile(u8),
}

/// Temporal feature
#[derive(Debug, Clone)]
pub struct TemporalFeature {
    pub feature_type: TemporalFeatureType,
    pub time_scale: Duration,
    pub threshold: f64,
}

/// Types of temporal features
#[derive(Debug, Clone)]
pub enum TemporalFeatureType {
    Periodicity,
    Trend,
    Seasonality,
    Autocorrelation,
    ChangePoint,
}

/// Matching criteria for pattern recognition
#[derive(Debug, Clone)]
pub struct MatchingCriteria {
    pub min_confidence: f64,
    pub min_data_points: usize,
    pub time_window_requirements: TimeWindowRequirements,
}

/// Time window requirements
#[derive(Debug, Clone)]
pub struct TimeWindowRequirements {
    pub min_duration: Duration,
    pub max_duration: Duration,
    pub coverage_ratio: f64,
}

// ─── Anomaly Detection ────────────────────────────────────────────────────────

/// Anomaly detection algorithm
#[derive(Debug, Clone)]
pub struct AnomalyAlgorithm {
    pub name: String,
    pub algorithm_type: AnomalyAlgorithmType,
    pub sensitivity: f64,
    pub config: HashMap<String, f64>,
}

/// Types of anomaly detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyAlgorithmType {
    StatisticalOutlier,
    IsolationForest,
    LocalOutlierFactor,
    OneClassSvm,
    AutoEncoder,
    TimeSeriesAnomaly,
}

/// Detected performance anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnomaly {
    pub id: String,
    pub anomaly_type: AnomalyType,
    pub severity: AnomalySeverity,
    pub detected_at: DateTime<Utc>,
    pub affected_metrics: Vec<String>,
    pub anomaly_score: f64,
    pub context: AnomalyContext,
}

/// Types of anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    LatencySpike,
    ThroughputDrop,
    MemoryLeak,
    CpuSaturation,
    ErrorRateIncrease,
    ResourceStarvation,
    UnexpectedPattern,
}

/// Anomaly severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Anomaly context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyContext {
    pub component: String,
    pub related_events: Vec<String>,
    pub environmental_factors: HashMap<String, String>,
    pub potential_causes: Vec<String>,
}

/// Baseline model for normal behavior
#[derive(Debug, Clone)]
pub struct BaselineModel {
    pub name: String,
    pub distribution: StatisticalDistribution,
    pub temporal_characteristics: TemporalCharacteristics,
    pub confidence: f64,
    pub last_updated: DateTime<Utc>,
}

/// Statistical distribution model
#[derive(Debug, Clone)]
pub struct StatisticalDistribution {
    pub distribution_type: DistributionType,
    pub parameters: Vec<f64>,
    pub goodness_of_fit: f64,
}

/// Types of statistical distributions
#[derive(Debug, Clone)]
pub enum DistributionType {
    Normal,
    LogNormal,
    Exponential,
    Gamma,
    Beta,
    Weibull,
    Custom,
}

/// Temporal characteristics of metrics
#[derive(Debug, Clone)]
pub struct TemporalCharacteristics {
    pub seasonality: Vec<SeasonalComponent>,
    pub trend: TrendInformation,
    pub autocorrelation: AutocorrelationStructure,
}

/// Seasonal component
#[derive(Debug, Clone)]
pub struct SeasonalComponent {
    pub period: Duration,
    pub amplitude: f64,
    pub phase: f64,
    pub strength: f64,
}

/// Trend information
#[derive(Debug, Clone)]
pub struct TrendInformation {
    pub direction: TrendDirection,
    pub strength: f64,
    pub linear_coefficient: f64,
    pub polynomial_coefficients: Vec<f64>,
}

/// Trend direction
#[derive(Debug, Clone)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Oscillating,
}

/// Autocorrelation structure
#[derive(Debug, Clone)]
pub struct AutocorrelationStructure {
    pub lag_correlations: Vec<(Duration, f64)>,
    pub partial_autocorrelations: Vec<(Duration, f64)>,
    pub significant_lags: Vec<Duration>,
}

// ─── Optimization Recommendations ────────────────────────────────────────────

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub id: String,
    pub recommendation_type: RecommendationType,
    pub priority: RecommendationPriority,
    pub component: String,
    pub current_state: String,
    pub recommended_state: String,
    pub expected_improvement: ExpectedImprovement,
    pub implementation_effort: ImplementationEffort,
    pub risk_assessment: RiskAssessment,
    pub description: String,
    pub implementation_steps: Vec<String>,
}

/// Types of optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    ResourceScaling,
    ConfigurationTuning,
    CacheOptimization,
    LoadBalancing,
    HardwareUpgrade,
    SoftwareUpdate,
    ArchitecturalChange,
    ProcessOptimization,
}

/// Recommendation priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Expected improvement metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedImprovement {
    pub latency_improvement_percent: f64,
    pub throughput_improvement_percent: f64,
    pub resource_savings_percent: f64,
    pub cost_reduction_percent: f64,
    pub confidence: f64,
}

/// Implementation effort assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationEffort {
    pub estimated_hours: f64,
    pub required_skills: Vec<String>,
    pub complexity: ComplexityLevel,
    pub dependencies: Vec<String>,
}

/// Complexity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Risk assessment for recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub risk_level: RiskLevel,
    pub potential_impacts: Vec<PotentialImpact>,
    pub mitigation_strategies: Vec<String>,
    pub rollback_plan: String,
}

/// Risk levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Potential impact of changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PotentialImpact {
    pub impact_type: ImpactType,
    pub severity: ImpactSeverity,
    pub probability: f64,
    pub description: String,
}

/// Types of potential impacts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactType {
    PerformanceDegradation,
    ServiceDisruption,
    DataLoss,
    SecurityVulnerability,
    IncreasedCosts,
    UserExperience,
}

/// Impact severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactSeverity {
    Negligible,
    Minor,
    Moderate,
    Major,
    Severe,
}

/// Recommendation rule
#[derive(Debug, Clone)]
pub struct RecommendationRule {
    pub name: String,
    pub conditions: Vec<TriggerCondition>,
    pub recommendation_template: RecommendationTemplate,
    pub priority: i32,
}

/// Trigger condition for recommendations
#[derive(Debug, Clone)]
pub struct TriggerCondition {
    pub metric: String,
    pub operator: ComparisonOperator,
    pub threshold: f64,
    pub time_window: Duration,
}

/// Comparison operators
#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Equal,
    NotEqual,
    Between(f64, f64),
}

/// Template for generating recommendations
#[derive(Debug, Clone)]
pub struct RecommendationTemplate {
    pub recommendation_type: RecommendationType,
    pub description_template: String,
    pub default_priority: RecommendationPriority,
    pub default_effort: ImplementationEffort,
}

/// Historical recommendation data
#[derive(Debug, Clone)]
pub struct RecommendationHistory {
    pub recommendation_id: String,
    pub implemented_at: Option<DateTime<Utc>>,
    pub actual_improvement: Option<ExpectedImprovement>,
    pub feedback: Option<String>,
    pub success_rating: Option<f64>,
}

// ─── Analysis Report types ────────────────────────────────────────────────────

/// Performance analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalysisReport {
    pub id: String,
    pub session_id: String,
    pub generated_at: DateTime<Utc>,
    pub analysis_results: Vec<AnalysisResult>,
    pub detected_patterns: Vec<PerformancePattern>,
    pub detected_anomalies: Vec<PerformanceAnomaly>,
    pub health_score: f64,
    pub summary: String,
}

impl PerformanceAnalysisReport {
    pub fn new(session_id: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            session_id,
            generated_at: Utc::now(),
            analysis_results: Vec::new(),
            detected_patterns: Vec::new(),
            detected_anomalies: Vec::new(),
            health_score: 100.0,
            summary: "Analysis in progress".to_string(),
        }
    }

    pub fn add_analysis_result(&mut self, result: AnalysisResult) {
        self.analysis_results.push(result);
    }

    pub fn set_detected_patterns(&mut self, patterns: Vec<PerformancePattern>) {
        self.detected_patterns = patterns;
    }

    pub fn set_detected_anomalies(&mut self, anomalies: Vec<PerformanceAnomaly>) {
        self.detected_anomalies = anomalies;
    }
}

/// Result from an analysis algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub algorithm_name: String,
    pub result_type: AlgorithmType,
    pub findings: Vec<Finding>,
    pub execution_time: Duration,
}

/// Individual finding from analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    pub title: String,
    pub description: String,
    pub severity: FindingSeverity,
    pub confidence: f64,
    pub affected_metrics: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Severity levels for findings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FindingSeverity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

// ─── Collector internals ──────────────────────────────────────────────────────

/// Performance data collector
#[derive(Debug)]
pub struct PerformanceCollector {
    pub buffer: VecDeque<MetricDataPoint>,
    pub stats: CollectionStats,
    pub trackers: HashMap<String, PerformanceTracker>,
}

// ─── Analyzer structs ─────────────────────────────────────────────────────────

/// Main performance analyzer — holds analysis algorithms and sub-detectors
#[derive(Debug)]
pub struct PerformanceAnalyzer {
    pub algorithms: Vec<AnalysisAlgorithm>,
    pub pattern_detector: PatternDetector,
    pub anomaly_detector: AnomalyDetector,
}

/// Pattern detector — holds pattern templates and detected patterns
#[derive(Debug)]
pub struct PatternDetector {
    pub patterns: Vec<PerformancePattern>,
    pub templates: Vec<PatternTemplate>,
}

/// Anomaly detector — holds anomaly detection algorithms, detected anomalies, and baselines
#[derive(Debug)]
pub struct AnomalyDetector {
    pub algorithms: Vec<AnomalyAlgorithm>,
    pub anomalies: Vec<PerformanceAnomaly>,
    pub baselines: HashMap<String, BaselineModel>,
}

/// Optimization recommender — holds rules, pending recommendations, and history
#[derive(Debug)]
pub struct OptimizationRecommender {
    pub rules: Vec<RecommendationRule>,
    pub recommendations: Vec<OptimizationRecommendation>,
    pub history: VecDeque<RecommendationHistory>,
}

// ─── Top-level profiler ───────────────────────────────────────────────────────

/// Advanced performance profiler for embedding systems
#[derive(Debug)]
pub struct AdvancedProfiler {
    /// Configuration for profiling
    pub(super) config: ProfilerConfig,
    /// Active profiling sessions
    pub(super) sessions: Arc<RwLock<HashMap<String, ProfilingSession>>>,
    /// Performance data collector
    pub(super) collector: Arc<Mutex<PerformanceCollector>>,
    /// Analysis engine
    pub(super) analyzer: PerformanceAnalyzer,
    /// Optimization recommender
    pub(super) recommender: OptimizationRecommender,
}
