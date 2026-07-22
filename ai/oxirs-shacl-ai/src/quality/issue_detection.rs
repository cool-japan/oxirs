//! Quality Issue Detection for SHACL-AI
//!
//! This module implements comprehensive quality issue detection including
//! anomaly detection, quality degradation monitoring, and proactive issue identification.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use oxirs_core::Store;
use oxirs_shacl::{Severity, Shape, ValidationReport};

use super::QualityReport;
use crate::{Result, ShaclAiError};

/// Quality issue detection engine
#[derive(Debug)]
pub struct QualityIssueDetector {
    config: IssueDetectionConfig,
    detection_models: DetectionModels,
    historical_data: Vec<QualitySnapshot>,
    statistics: IssueDetectionStatistics,
}

/// Configuration for quality issue detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IssueDetectionConfig {
    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,

    /// Enable quality degradation monitoring
    pub enable_degradation_monitoring: bool,

    /// Enable proactive issue detection
    pub enable_proactive_detection: bool,

    /// Anomaly detection sensitivity (0.0 - 1.0)
    pub anomaly_sensitivity: f64,

    /// Quality degradation threshold
    pub degradation_threshold: f64,

    /// Minimum historical data points for trend analysis
    pub min_historical_points: usize,

    /// Detection interval in seconds
    pub detection_interval_seconds: u64,

    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,

    /// Enable real-time monitoring
    pub enable_realtime_monitoring: bool,
}

impl Default for IssueDetectionConfig {
    fn default() -> Self {
        Self {
            enable_anomaly_detection: true,
            enable_degradation_monitoring: true,
            enable_proactive_detection: true,
            anomaly_sensitivity: 0.8,
            degradation_threshold: 0.1, // 10% degradation threshold
            min_historical_points: 10,
            detection_interval_seconds: 300, // 5 minutes
            alert_thresholds: AlertThresholds::default(),
            enable_realtime_monitoring: true,
        }
    }
}

/// Alert thresholds for different issue types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub critical_anomaly_score: f64,
    pub high_anomaly_score: f64,
    pub medium_anomaly_score: f64,
    pub degradation_rate_threshold: f64,
    pub performance_degradation_threshold: f64,
    pub data_quality_threshold: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            critical_anomaly_score: 0.9,
            high_anomaly_score: 0.7,
            medium_anomaly_score: 0.5,
            degradation_rate_threshold: 0.15,
            performance_degradation_threshold: 0.2,
            data_quality_threshold: 0.75,
        }
    }
}

/// Detection models for different types of issues
#[derive(Debug)]
struct DetectionModels {
    statistical_anomaly_detector: Option<StatisticalAnomalyDetector>,
    semantic_anomaly_detector: Option<SemanticAnomalyDetector>,
    structural_anomaly_detector: Option<StructuralAnomalyDetector>,
    behavioral_anomaly_detector: Option<BehavioralAnomalyDetector>,
    temporal_anomaly_detector: Option<TemporalAnomalyDetector>,
    degradation_monitor: Option<DegradationMonitor>,
}

/// Quality snapshot for historical tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySnapshot {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub overall_quality_score: f64,
    pub quality_dimensions: HashMap<String, f64>,
    pub performance_metrics: PerformanceSnapshot,
    pub data_characteristics: DataCharacteristics,
    pub validation_results: ValidationSnapshot,
}

/// Performance snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    pub validation_time_ms: f64,
    pub memory_usage_mb: f64,
    pub throughput_ops_per_sec: f64,
    pub error_rate: f64,
    pub resource_utilization: f64,
}

/// Data characteristics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCharacteristics {
    pub total_triples: usize,
    pub unique_subjects: usize,
    pub unique_predicates: usize,
    pub unique_objects: usize,
    pub schema_complexity: f64,
    pub data_density: f64,
}

/// Validation snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSnapshot {
    pub total_validations: usize,
    pub successful_validations: usize,
    pub violation_count: usize,
    pub average_violation_severity: f64,
    pub validation_coverage: f64,
}

/// Issue detection statistics
#[derive(Debug, Clone, Default)]
pub struct IssueDetectionStatistics {
    pub total_detections: usize,
    pub anomalies_detected: usize,
    pub degradations_detected: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub detection_accuracy: f64,
    pub average_detection_time: std::time::Duration,
}

/// Comprehensive issue detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IssueDetectionResult {
    pub anomaly_detection: AnomalyDetectionResult,
    pub degradation_detection: DegradationDetectionResult,
    pub proactive_alerts: Vec<ProactiveAlert>,
    pub issue_summary: IssueSummary,
    pub recommendations: Vec<IssueRecommendation>,
    pub detection_metadata: DetectionMetadata,
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionResult {
    pub statistical_anomalies: Vec<StatisticalAnomaly>,
    pub semantic_anomalies: Vec<SemanticAnomaly>,
    pub structural_anomalies: Vec<StructuralAnomaly>,
    pub behavioral_anomalies: Vec<BehavioralAnomaly>,
    pub temporal_anomalies: Vec<TemporalAnomaly>,
    pub cross_reference_anomalies: Vec<CrossReferenceAnomaly>,
    pub overall_anomaly_score: f64,
    pub anomaly_trends: AnomalyTrends,
}

/// Quality degradation detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationDetectionResult {
    pub quality_trend_analysis: QualityTrendAnalysis,
    pub performance_degradation: PerformanceDegradation,
    pub content_deterioration: ContentDeterioration,
    pub usage_pattern_changes: UsagePatternChanges,
    pub system_health_status: SystemHealthStatus,
    pub degradation_forecast: DegradationForecast,
}

/// Proactive alert for potential issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProactiveAlert {
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub confidence: f64,
    pub description: String,
    pub affected_components: Vec<String>,
    pub predicted_impact: f64,
    pub time_to_criticality: Option<chrono::Duration>,
    pub recommended_actions: Vec<String>,
}

/// Types of alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    QualityDegradation,
    PerformanceIssue,
    DataAnomaly,
    ValidationFailure,
    SystemOverload,
    SchemaEvolution,
    SecurityConcern,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

/// Issue summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IssueSummary {
    pub total_issues: usize,
    pub critical_issues: usize,
    pub high_priority_issues: usize,
    pub issue_categories: HashMap<String, usize>,
    pub trend_direction: TrendDirection,
    pub overall_health_score: f64,
}

/// Trend direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Declining,
    Volatile,
}

/// Issue recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IssueRecommendation {
    pub recommendation_type: RecommendationType,
    pub priority: RecommendationPriority,
    pub description: String,
    pub expected_impact: f64,
    pub implementation_effort: ImplementationEffort,
    pub urgency: Urgency,
    pub success_probability: f64,
}

/// Types of recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    ImmediateAction,
    PreventiveMeasure,
    SystemOptimization,
    DataCleaning,
    SchemaAdjustment,
    MonitoringEnhancement,
    ProcessImprovement,
}

/// Recommendation priority
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Urgent,
    Critical,
}

/// Implementation effort levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Minimal,
    Low,
    Medium,
    High,
    Extensive,
}

/// Urgency levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Urgency {
    Immediate,
    Today,
    ThisWeek,
    ThisMonth,
    Planned,
}

/// Detection metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionMetadata {
    pub detection_timestamp: chrono::DateTime<chrono::Utc>,
    pub detection_duration: std::time::Duration,
    pub models_used: Vec<String>,
    pub data_sources: Vec<String>,
    pub confidence_score: f64,
    pub coverage_percentage: f64,
}

// Specific anomaly types

/// Statistical anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnomaly {
    pub anomaly_type: StatisticalAnomalyType,
    pub anomaly_score: f64,
    pub confidence: f64,
    pub affected_properties: Vec<String>,
    pub statistical_measures: StatisticalMeasures,
    pub context: String,
}

/// Types of statistical anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatisticalAnomalyType {
    OutlierValues,
    DistributionShift,
    VarianceChange,
    CorrelationBreakdown,
    FrequencyAnomaly,
    RangeViolation,
}

/// Statistical measures for anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalMeasures {
    pub z_score: f64,
    pub p_value: f64,
    pub deviation_magnitude: f64,
    pub percentile_rank: f64,
}

/// Semantic anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAnomaly {
    pub anomaly_type: SemanticAnomalyType,
    pub anomaly_score: f64,
    pub confidence: f64,
    pub affected_concepts: Vec<String>,
    pub semantic_context: SemanticContext,
    pub explanation: String,
}

/// Types of semantic anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SemanticAnomalyType {
    ConceptualInconsistency,
    RelationshipAnomaly,
    OntologyViolation,
    SemanticDrift,
    ContextualMismatch,
    MeaninglessRelation,
}

/// Semantic context for anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticContext {
    pub domain: String,
    pub expected_relationships: Vec<String>,
    pub violated_constraints: Vec<String>,
    pub similarity_scores: HashMap<String, f64>,
}

/// Structural anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralAnomaly {
    pub anomaly_type: StructuralAnomalyType,
    pub anomaly_score: f64,
    pub confidence: f64,
    pub affected_structure: StructuralComponent,
    pub structural_impact: StructuralImpact,
}

/// Types of structural anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StructuralAnomalyType {
    GraphTopologyChange,
    ConnectivityAnomaly,
    HierarchyViolation,
    CyclicDependency,
    IsolatedComponents,
    ExcessiveComplexity,
}

/// Structural component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralComponent {
    pub component_type: String,
    pub component_id: String,
    pub normal_characteristics: HashMap<String, f64>,
    pub anomalous_characteristics: HashMap<String, f64>,
}

/// Structural impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralImpact {
    pub connectivity_impact: f64,
    pub performance_impact: f64,
    pub maintainability_impact: f64,
    pub scalability_impact: f64,
}

/// Behavioral anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralAnomaly {
    pub anomaly_type: BehavioralAnomalyType,
    pub anomaly_score: f64,
    pub confidence: f64,
    pub behavior_pattern: BehaviorPattern,
    pub deviation_analysis: DeviationAnalysis,
}

/// Types of behavioral anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BehavioralAnomalyType {
    UsagePatternChange,
    AccessPatternAnomaly,
    PerformanceAnomaly,
    ErrorRateSpike,
    ResourceConsumptionAnomaly,
    ValidationPatternChange,
}

/// Behavior pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorPattern {
    pub pattern_name: String,
    pub normal_behavior: HashMap<String, f64>,
    pub observed_behavior: HashMap<String, f64>,
    pub deviation_metrics: HashMap<String, f64>,
}

/// Deviation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviationAnalysis {
    pub deviation_magnitude: f64,
    pub deviation_duration: chrono::Duration,
    pub recovery_probability: f64,
    pub impact_assessment: f64,
}

/// Temporal anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAnomaly {
    pub anomaly_type: TemporalAnomalyType,
    pub anomaly_score: f64,
    pub confidence: f64,
    pub time_series_analysis: TimeSeriesAnalysis,
    pub temporal_context: TemporalContext,
}

/// Types of temporal anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalAnomalyType {
    SeasonalityBreakdown,
    TrendReversal,
    CyclicPatternAnomaly,
    TemporalInconsistency,
    FreshnessAnomaly,
    UpdateFrequencyAnomaly,
}

/// Time series analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesAnalysis {
    pub trend_component: f64,
    pub seasonal_component: f64,
    pub irregular_component: f64,
    pub autocorrelation: f64,
    pub stationarity_score: f64,
}

/// Temporal context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalContext {
    pub time_window: chrono::Duration,
    pub expected_patterns: Vec<String>,
    pub observed_patterns: Vec<String>,
    pub temporal_relationships: HashMap<String, f64>,
}

/// Cross-reference anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossReferenceAnomaly {
    pub anomaly_type: CrossReferenceAnomalyType,
    pub anomaly_score: f64,
    pub confidence: f64,
    pub reference_consistency: ReferenceConsistency,
    pub integrity_analysis: IntegrityAnalysis,
}

/// Types of cross-reference anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrossReferenceAnomalyType {
    BrokenReference,
    CircularReference,
    MissingReference,
    DanglingReference,
    InconsistentReference,
    InvalidReference,
}

/// Reference consistency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceConsistency {
    pub total_references: usize,
    pub valid_references: usize,
    pub broken_references: usize,
    pub consistency_score: f64,
}

/// Integrity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityAnalysis {
    pub referential_integrity: f64,
    pub domain_integrity: f64,
    pub entity_integrity: f64,
    pub user_defined_integrity: f64,
}

/// Anomaly trends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyTrends {
    pub trend_direction: TrendDirection,
    pub anomaly_frequency: f64,
    pub severity_trend: f64,
    pub patterns_identified: Vec<String>,
    pub forecasted_anomalies: Vec<ForecastedAnomaly>,
}

/// Forecasted anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastedAnomaly {
    pub predicted_time: chrono::DateTime<chrono::Utc>,
    pub anomaly_type: String,
    pub probability: f64,
    pub expected_severity: f64,
}

// Quality degradation structures

/// Quality trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityTrendAnalysis {
    pub overall_trend: TrendDirection,
    pub trend_strength: f64,
    pub degradation_rate: f64,
    pub affected_dimensions: Vec<String>,
    pub trend_forecast: TrendForecast,
}

/// Trend forecast
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendForecast {
    pub forecast_horizon: chrono::Duration,
    pub predicted_quality_scores: Vec<QualityForecastPoint>,
    pub confidence_intervals: Vec<ConfidenceInterval>,
    pub risk_assessment: f64,
}

/// Quality forecast point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityForecastPoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub predicted_score: f64,
    pub confidence: f64,
}

/// Confidence interval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub confidence_level: f64,
}

/// Performance degradation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDegradation {
    pub degradation_detected: bool,
    pub performance_metrics: PerformanceMetrics,
    pub bottlenecks_identified: Vec<PerformanceBottleneck>,
    pub degradation_causes: Vec<DegradationCause>,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub response_time_trend: f64,
    pub throughput_trend: f64,
    pub error_rate_trend: f64,
    pub resource_utilization_trend: f64,
    pub availability_trend: f64,
}

/// Performance bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    pub bottleneck_type: String,
    pub severity: f64,
    pub affected_components: Vec<String>,
    pub impact_on_performance: f64,
    pub resolution_suggestions: Vec<String>,
}

/// Degradation cause
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationCause {
    pub cause_type: String,
    pub likelihood: f64,
    pub impact: f64,
    pub evidence: Vec<String>,
    pub mitigation_strategies: Vec<String>,
}

/// Content deterioration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentDeterioration {
    pub deterioration_detected: bool,
    pub data_quality_decline: DataQualityDecline,
    pub schema_evolution_issues: Vec<SchemaEvolutionIssue>,
    pub consistency_degradation: ConsistencyDegradation,
}

/// Data quality decline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityDecline {
    pub completeness_decline: f64,
    pub accuracy_decline: f64,
    pub consistency_decline: f64,
    pub timeliness_decline: f64,
    pub validity_decline: f64,
}

/// Schema evolution issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaEvolutionIssue {
    pub issue_type: String,
    pub severity: f64,
    pub description: String,
    pub affected_entities: Vec<String>,
    pub backward_compatibility: bool,
}

/// Consistency degradation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyDegradation {
    pub logical_consistency_decline: f64,
    pub referential_consistency_decline: f64,
    pub semantic_consistency_decline: f64,
    pub temporal_consistency_decline: f64,
}

/// Usage pattern changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsagePatternChanges {
    pub pattern_changes_detected: bool,
    pub access_pattern_changes: Vec<AccessPatternChange>,
    pub query_pattern_changes: Vec<QueryPatternChange>,
    pub validation_pattern_changes: Vec<ValidationPatternChange>,
}

/// Access pattern change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPatternChange {
    pub pattern_type: String,
    pub change_magnitude: f64,
    pub change_direction: String,
    pub affected_resources: Vec<String>,
    pub impact_assessment: f64,
}

/// Query pattern change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPatternChange {
    pub query_type: String,
    pub frequency_change: f64,
    pub complexity_change: f64,
    pub performance_impact: f64,
    pub resource_impact: f64,
}

/// Validation pattern change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationPatternChange {
    pub validation_type: String,
    pub success_rate_change: f64,
    pub error_pattern_change: String,
    pub performance_change: f64,
}

/// System health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthStatus {
    pub overall_health_score: f64,
    pub health_components: HashMap<String, f64>,
    pub critical_issues: Vec<CriticalIssue>,
    pub health_trend: TrendDirection,
    pub recovery_recommendations: Vec<String>,
}

/// Critical issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalIssue {
    pub issue_type: String,
    pub severity: f64,
    pub description: String,
    pub immediate_actions: Vec<String>,
    pub time_to_failure: Option<chrono::Duration>,
}

/// Degradation forecast
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationForecast {
    pub forecast_horizon: chrono::Duration,
    pub predicted_degradations: Vec<PredictedDegradation>,
    pub risk_timeline: Vec<RiskTimelinePoint>,
    pub mitigation_opportunities: Vec<MitigationOpportunity>,
}

/// Predicted degradation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedDegradation {
    pub degradation_type: String,
    pub predicted_time: chrono::DateTime<chrono::Utc>,
    pub probability: f64,
    pub expected_impact: f64,
    pub early_warning_signs: Vec<String>,
}

/// Risk timeline point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskTimelinePoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub risk_level: f64,
    pub risk_factors: Vec<String>,
    pub mitigation_actions: Vec<String>,
}

/// Mitigation opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MitigationOpportunity {
    pub opportunity_type: String,
    pub effectiveness: f64,
    pub implementation_cost: f64,
    pub time_window: chrono::Duration,
    pub expected_benefits: Vec<String>,
}

// Detection model placeholders

#[derive(Debug)]
struct StatisticalAnomalyDetector {
    sensitivity: f64,
    algorithms: Vec<String>,
}

#[derive(Debug)]
struct SemanticAnomalyDetector {
    embedding_model: String,
    similarity_threshold: f64,
}

#[derive(Debug)]
struct StructuralAnomalyDetector {
    graph_metrics: Vec<String>,
    change_threshold: f64,
}

#[derive(Debug)]
struct BehavioralAnomalyDetector {
    behavior_models: Vec<String>,
    deviation_threshold: f64,
}

#[derive(Debug)]
struct TemporalAnomalyDetector {
    time_series_models: Vec<String>,
    seasonality_detection: bool,
}

#[derive(Debug)]
struct DegradationMonitor {
    monitoring_interval: chrono::Duration,
    trend_analysis_window: chrono::Duration,
}

impl QualityIssueDetector {
    /// Create a new quality issue detector
    pub fn new() -> Self {
        Self::with_config(IssueDetectionConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: IssueDetectionConfig) -> Self {
        Self {
            config,
            detection_models: DetectionModels::new(),
            historical_data: Vec::new(),
            statistics: IssueDetectionStatistics::default(),
        }
    }

    /// Detect quality issues comprehensively
    pub fn detect_quality_issues(
        &mut self,
        store: &dyn Store,
        shapes: &[Shape],
        quality_report: &QualityReport,
        validation_report: Option<&ValidationReport>,
    ) -> Result<IssueDetectionResult> {
        tracing::info!("Starting comprehensive quality issue detection");
        let start_time = Instant::now();

        // Create current quality snapshot
        let current_snapshot =
            self.create_quality_snapshot(store, shapes, quality_report, validation_report)?;

        // Add to historical data
        self.historical_data.push(current_snapshot.clone());

        // Keep only recent history
        if self.historical_data.len() > 100 {
            self.historical_data.remove(0);
        }

        // Detect anomalies
        let anomaly_detection = if self.config.enable_anomaly_detection {
            self.detect_anomalies(&current_snapshot)?
        } else {
            AnomalyDetectionResult::default()
        };

        // Detect quality degradation
        let degradation_detection = if self.config.enable_degradation_monitoring {
            self.detect_quality_degradation()?
        } else {
            DegradationDetectionResult::default()
        };

        // Generate proactive alerts
        let proactive_alerts = if self.config.enable_proactive_detection {
            self.generate_proactive_alerts(&anomaly_detection, &degradation_detection)?
        } else {
            Vec::new()
        };

        // Generate issue summary
        let issue_summary = self.generate_issue_summary(
            &anomaly_detection,
            &degradation_detection,
            &proactive_alerts,
        );

        // Generate recommendations
        let recommendations = self.generate_issue_recommendations(
            &anomaly_detection,
            &degradation_detection,
            &proactive_alerts,
        )?;

        let detection_duration = start_time.elapsed();
        let detection_metadata = DetectionMetadata {
            detection_timestamp: chrono::Utc::now(),
            detection_duration,
            models_used: self.get_models_used(),
            data_sources: vec!["RDF Store".to_string(), "SHACL Shapes".to_string()],
            confidence_score: self
                .calculate_overall_confidence(&anomaly_detection, &degradation_detection),
            coverage_percentage: self.calculate_coverage_percentage(store),
        };

        // Update statistics
        self.update_statistics(
            &anomaly_detection,
            &degradation_detection,
            detection_duration,
        );

        let result = IssueDetectionResult {
            anomaly_detection,
            degradation_detection,
            proactive_alerts,
            issue_summary,
            recommendations,
            detection_metadata,
        };

        tracing::info!(
            "Quality issue detection completed in {:?}. Found {} issues",
            detection_duration,
            result.issue_summary.total_issues
        );

        Ok(result)
    }

    /// Initialize detection models
    pub fn initialize_models(&mut self) -> Result<()> {
        tracing::info!("Initializing quality issue detection models");

        self.detection_models = DetectionModels {
            statistical_anomaly_detector: Some(StatisticalAnomalyDetector {
                sensitivity: self.config.anomaly_sensitivity,
                algorithms: vec![
                    "Z-Score".to_string(),
                    "IQR".to_string(),
                    "Isolation Forest".to_string(),
                ],
            }),
            semantic_anomaly_detector: Some(SemanticAnomalyDetector {
                embedding_model: "transformer-based".to_string(),
                similarity_threshold: 0.7,
            }),
            structural_anomaly_detector: Some(StructuralAnomalyDetector {
                graph_metrics: vec![
                    "Degree Distribution".to_string(),
                    "Clustering Coefficient".to_string(),
                    "Path Length".to_string(),
                ],
                change_threshold: 0.2,
            }),
            behavioral_anomaly_detector: Some(BehavioralAnomalyDetector {
                behavior_models: vec![
                    "Usage Pattern Model".to_string(),
                    "Performance Model".to_string(),
                ],
                deviation_threshold: 0.3,
            }),
            temporal_anomaly_detector: Some(TemporalAnomalyDetector {
                time_series_models: vec!["ARIMA".to_string(), "Seasonal Decomposition".to_string()],
                seasonality_detection: true,
            }),
            degradation_monitor: Some(DegradationMonitor {
                monitoring_interval: chrono::Duration::seconds(
                    self.config.detection_interval_seconds as i64,
                ),
                trend_analysis_window: chrono::Duration::days(7),
            }),
        };

        Ok(())
    }

    /// Add historical quality data
    pub fn add_historical_data(&mut self, snapshot: QualitySnapshot) {
        self.historical_data.push(snapshot);

        // Keep only recent history
        if self.historical_data.len() > 100 {
            self.historical_data.remove(0);
        }
    }

    /// Get detection statistics
    pub fn get_statistics(&self) -> &IssueDetectionStatistics {
        &self.statistics
    }

    /// Clear historical data
    pub fn clear_history(&mut self) {
        self.historical_data.clear();
    }

    // Private implementation methods

    fn create_quality_snapshot(
        &self,
        store: &dyn Store,
        shapes: &[Shape],
        quality_report: &QualityReport,
        validation_report: Option<&ValidationReport>,
    ) -> Result<QualitySnapshot> {
        // Create snapshot from current data
        let mut quality_dimensions = HashMap::new();
        quality_dimensions.insert(
            "completeness".to_string(),
            quality_report.completeness_score,
        );
        quality_dimensions.insert("consistency".to_string(), quality_report.consistency_score);
        quality_dimensions.insert("accuracy".to_string(), quality_report.accuracy_score);
        quality_dimensions.insert("conformance".to_string(), quality_report.conformance_score);
        quality_dimensions.insert(
            "schema_adherence".to_string(),
            quality_report.schema_adherence_score,
        );

        let data_characteristics = Self::compute_data_characteristics(store)?;
        let validation_results =
            Self::compute_validation_snapshot(shapes, quality_report, validation_report);

        // The error rate is a real signal derived from validation results; the
        // remaining performance fields are not measured at this call site and
        // are honestly reported as unknown (0.0) rather than fabricated.
        let error_rate = if validation_results.total_validations > 0 {
            validation_results.violation_count as f64 / validation_results.total_validations as f64
        } else {
            0.0
        };

        Ok(QualitySnapshot {
            timestamp: chrono::Utc::now(),
            overall_quality_score: quality_report.overall_score,
            quality_dimensions,
            performance_metrics: PerformanceSnapshot {
                validation_time_ms: 0.0,
                memory_usage_mb: 0.0,
                throughput_ops_per_sec: 0.0,
                error_rate,
                resource_utilization: 0.0,
            },
            data_characteristics,
            validation_results,
        })
    }

    /// Derive real data characteristics by scanning the store's quads.
    fn compute_data_characteristics(store: &dyn Store) -> Result<DataCharacteristics> {
        let quads = store.find_quads(None, None, None, None).map_err(|e| {
            ShaclAiError::QualityAssessment(format!("failed to scan store for snapshot: {e}"))
        })?;

        let total_triples = quads.len();
        let mut subjects: HashSet<String> = HashSet::new();
        let mut predicates: HashSet<String> = HashSet::new();
        let mut objects: HashSet<String> = HashSet::new();
        for quad in &quads {
            subjects.insert(format!("{:?}", quad.subject()));
            predicates.insert(format!("{:?}", quad.predicate()));
            objects.insert(format!("{:?}", quad.object()));
        }

        let unique_subjects = subjects.len();
        let unique_predicates = predicates.len();
        let unique_objects = objects.len();

        // Schema complexity: saturating function of the number of distinct
        // predicates (more predicates -> richer schema), bounded to [0, 1].
        let schema_complexity = unique_predicates as f64 / (unique_predicates as f64 + 20.0);

        // Data density: how densely the subject x predicate space is populated.
        let cells = unique_subjects.saturating_mul(unique_predicates);
        let data_density = if cells > 0 {
            (total_triples as f64 / cells as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };

        Ok(DataCharacteristics {
            total_triples,
            unique_subjects,
            unique_predicates,
            unique_objects,
            schema_complexity,
            data_density,
        })
    }

    /// Derive a validation snapshot from the validation report when available,
    /// otherwise from the quality report's conformance score.
    fn compute_validation_snapshot(
        shapes: &[Shape],
        quality_report: &QualityReport,
        validation_report: Option<&ValidationReport>,
    ) -> ValidationSnapshot {
        let total_shapes = shapes.len();
        match validation_report {
            Some(report) => {
                let violations = report.violations();
                let violation_count = violations.len();

                // Severity as a number: Violation = 1.0, Warning = 0.5, Info = 0.25.
                let severity_sum: f64 = violations
                    .iter()
                    .map(|v| match v.result_severity {
                        Severity::Violation => 1.0,
                        Severity::Warning => 0.5,
                        Severity::Info => 0.25,
                    })
                    .sum();
                let average_violation_severity = if violation_count > 0 {
                    severity_sum / violation_count as f64
                } else {
                    0.0
                };

                // Shapes that produced at least one violation.
                let failing_shapes: HashSet<String> = violations
                    .iter()
                    .map(|v| format!("{:?}", v.source_shape))
                    .collect();
                let total_validations = total_shapes.max(1);
                let successful_validations = total_validations.saturating_sub(failing_shapes.len());
                let validation_coverage = if total_shapes > 0 { 1.0 } else { 0.0 };

                ValidationSnapshot {
                    total_validations,
                    successful_validations,
                    violation_count,
                    average_violation_severity,
                    validation_coverage,
                }
            }
            None => {
                // Fall back to the conformance score: estimate how many of the
                // evaluated shapes conform.
                let conformance = quality_report.conformance_score.clamp(0.0, 1.0);
                let total_validations = total_shapes.max(1);
                let successful_validations =
                    (conformance * total_validations as f64).round() as usize;
                let violation_count = total_validations.saturating_sub(successful_validations);
                ValidationSnapshot {
                    total_validations,
                    successful_validations,
                    violation_count,
                    average_violation_severity: 1.0 - conformance,
                    validation_coverage: if total_shapes > 0 { 1.0 } else { 0.0 },
                }
            }
        }
    }

    /// Detect statistical anomalies by comparing the current snapshot's quality
    /// signals against the distribution of the accumulated historical baseline.
    ///
    /// For the overall score and each tracked dimension, a z-score is computed
    /// against the historical mean/standard-deviation; a signal whose magnitude
    /// exceeds a sensitivity-derived threshold is reported as an
    /// [`StatisticalAnomaly`]. When there is not yet enough history, no anomaly
    /// is reported (and the overall anomaly score is 0.0), rather than emitting
    /// a fabricated constant.
    fn detect_anomalies(
        &self,
        current_snapshot: &QualitySnapshot,
    ) -> Result<AnomalyDetectionResult> {
        let mut statistical_anomalies = Vec::new();
        let mut max_abs_z = 0.0_f64;

        // Baseline excludes the current snapshot (which detect_quality_issues
        // has already appended to `historical_data`).
        let baseline_len = self.historical_data.len().saturating_sub(1);
        let min_points = self.config.min_historical_points.max(2);
        let z_threshold = anomaly_z_threshold(self.config.anomaly_sensitivity);

        if baseline_len >= min_points {
            let baseline = &self.historical_data[..baseline_len];

            // Overall quality score.
            let overall_hist: Vec<f64> = baseline.iter().map(|s| s.overall_quality_score).collect();
            if let Some(anomaly) = check_statistical_anomaly(
                "overall_quality_score",
                current_snapshot.overall_quality_score,
                &overall_hist,
                z_threshold,
            ) {
                max_abs_z = max_abs_z.max(anomaly.statistical_measures.z_score.abs());
                statistical_anomalies.push(anomaly);
            }

            // Each quality dimension present in the current snapshot.
            for (dimension, &current_value) in &current_snapshot.quality_dimensions {
                let hist: Vec<f64> = baseline
                    .iter()
                    .filter_map(|s| s.quality_dimensions.get(dimension).copied())
                    .collect();
                if hist.len() < min_points {
                    continue;
                }
                if let Some(anomaly) =
                    check_statistical_anomaly(dimension, current_value, &hist, z_threshold)
                {
                    max_abs_z = max_abs_z.max(anomaly.statistical_measures.z_score.abs());
                    statistical_anomalies.push(anomaly);
                }
            }

            // Error rate (higher is worse).
            let error_hist: Vec<f64> = baseline
                .iter()
                .map(|s| s.performance_metrics.error_rate)
                .collect();
            if let Some(anomaly) = check_statistical_anomaly(
                "error_rate",
                current_snapshot.performance_metrics.error_rate,
                &error_hist,
                z_threshold,
            ) {
                max_abs_z = max_abs_z.max(anomaly.statistical_measures.z_score.abs());
                statistical_anomalies.push(anomaly);
            }
        }

        // Normalize the strongest deviation to a [0, 1] anomaly score
        // (z of 6 saturates to 1.0).
        let overall_anomaly_score = (max_abs_z / 6.0).clamp(0.0, 1.0);

        // Trend of the overall score across the full history.
        let full_hist: Vec<f64> = self
            .historical_data
            .iter()
            .map(|s| s.overall_quality_score)
            .collect();
        let slope = linear_slope(&full_hist);
        let volatility = mean_std(&full_hist).map(|(_, s)| s).unwrap_or(0.0);
        let trend_direction = classify_trend(slope, volatility);

        let anomaly_count = statistical_anomalies.len();
        let anomaly_frequency = if !self.historical_data.is_empty() {
            anomaly_count as f64 / self.historical_data.len() as f64
        } else {
            0.0
        };

        let anomaly_trends = AnomalyTrends {
            trend_direction,
            anomaly_frequency,
            severity_trend: overall_anomaly_score,
            patterns_identified: statistical_anomalies
                .iter()
                .flat_map(|a| a.affected_properties.clone())
                .collect(),
            forecasted_anomalies: vec![],
        };

        Ok(AnomalyDetectionResult {
            statistical_anomalies,
            semantic_anomalies: vec![],
            structural_anomalies: vec![],
            behavioral_anomalies: vec![],
            temporal_anomalies: vec![],
            cross_reference_anomalies: vec![],
            overall_anomaly_score,
            anomaly_trends,
        })
    }

    /// Detect quality degradation by comparing a recent window of the history
    /// against an older window. All trends/declines are computed from the
    /// accumulated [`Self::historical_data`]; with insufficient history the
    /// result reports no degradation rather than fabricating decline figures.
    fn detect_quality_degradation(&self) -> Result<DegradationDetectionResult> {
        let history = &self.historical_data;

        // Split history into older vs recent halves for comparison.
        let (older, recent) = split_windows(history);
        let older_overall = mean(&window_scores(older, |s| s.overall_quality_score));
        let recent_overall = mean(&window_scores(recent, |s| s.overall_quality_score));

        // Positive degradation_rate == quality has dropped.
        let degradation_rate = match (older_overall, recent_overall) {
            (Some(o), Some(r)) => o - r,
            _ => 0.0,
        };

        let overall_scores: Vec<f64> = history.iter().map(|s| s.overall_quality_score).collect();
        let slope = linear_slope(&overall_scores);
        let volatility = mean_std(&overall_scores).map(|(_, s)| s).unwrap_or(0.0);
        let overall_trend = classify_trend(slope, volatility);
        let trend_strength = slope.abs().clamp(0.0, 1.0);

        // Per-dimension decline (older mean - recent mean, clamped to >= 0).
        let dim_decline = |dim: &str| -> f64 {
            let o = mean(&window_scores(older, |s| {
                s.quality_dimensions.get(dim).copied().unwrap_or(0.0)
            }));
            let r = mean(&window_scores(recent, |s| {
                s.quality_dimensions.get(dim).copied().unwrap_or(0.0)
            }));
            match (o, r) {
                (Some(o), Some(r)) => (o - r).max(0.0),
                _ => 0.0,
            }
        };
        let completeness_decline = dim_decline("completeness");
        let accuracy_decline = dim_decline("accuracy");
        let consistency_decline = dim_decline("consistency");
        let conformance_decline = dim_decline("conformance");

        let mut affected_dimensions = Vec::new();
        for (name, decline) in [
            ("completeness", completeness_decline),
            ("accuracy", accuracy_decline),
            ("consistency", consistency_decline),
            ("conformance", conformance_decline),
        ] {
            if decline > self.config.degradation_threshold {
                affected_dimensions.push(name.to_string());
            }
        }

        // Performance trends across snapshots.
        let error_rate_trend = linear_slope(
            &history
                .iter()
                .map(|s| s.performance_metrics.error_rate)
                .collect::<Vec<_>>(),
        );
        let throughput_trend = linear_slope(
            &history
                .iter()
                .map(|s| s.performance_metrics.throughput_ops_per_sec)
                .collect::<Vec<_>>(),
        );
        let response_time_trend = linear_slope(
            &history
                .iter()
                .map(|s| s.performance_metrics.validation_time_ms)
                .collect::<Vec<_>>(),
        );
        let resource_utilization_trend = linear_slope(
            &history
                .iter()
                .map(|s| s.performance_metrics.resource_utilization)
                .collect::<Vec<_>>(),
        );
        let performance_degradation_detected = error_rate_trend
            > self
                .config
                .alert_thresholds
                .performance_degradation_threshold
            || response_time_trend
                > self
                    .config
                    .alert_thresholds
                    .performance_degradation_threshold;

        let quality_trend_analysis = QualityTrendAnalysis {
            overall_trend: overall_trend.clone(),
            trend_strength,
            degradation_rate,
            affected_dimensions: affected_dimensions.clone(),
            trend_forecast: TrendForecast {
                forecast_horizon: chrono::Duration::days(30),
                predicted_quality_scores: vec![],
                confidence_intervals: vec![],
                risk_assessment: degradation_rate.clamp(0.0, 1.0),
            },
        };

        let performance_degradation = PerformanceDegradation {
            degradation_detected: performance_degradation_detected,
            performance_metrics: PerformanceMetrics {
                response_time_trend,
                throughput_trend,
                error_rate_trend,
                resource_utilization_trend,
                availability_trend: 0.0,
            },
            bottlenecks_identified: vec![],
            degradation_causes: vec![],
        };

        let content_deterioration_detected =
            !affected_dimensions.is_empty() || degradation_rate > self.config.degradation_threshold;
        let content_deterioration = ContentDeterioration {
            deterioration_detected: content_deterioration_detected,
            data_quality_decline: DataQualityDecline {
                completeness_decline,
                accuracy_decline,
                consistency_decline,
                timeliness_decline: 0.0,
                validity_decline: conformance_decline,
            },
            schema_evolution_issues: vec![],
            consistency_degradation: ConsistencyDegradation {
                logical_consistency_decline: consistency_decline,
                referential_consistency_decline: 0.0,
                semantic_consistency_decline: 0.0,
                temporal_consistency_decline: 0.0,
            },
        };

        let usage_pattern_changes = UsagePatternChanges {
            pattern_changes_detected: false,
            access_pattern_changes: vec![],
            query_pattern_changes: vec![],
            validation_pattern_changes: vec![],
        };

        // Health score anchored on the most recent overall quality score,
        // penalized by the observed degradation.
        let current_quality = overall_scores.last().copied().unwrap_or(1.0);
        let overall_health_score = (current_quality - degradation_rate.max(0.0)).clamp(0.0, 1.0);

        let mut critical_issues = Vec::new();
        if degradation_rate > self.config.alert_thresholds.degradation_rate_threshold {
            critical_issues.push(CriticalIssue {
                issue_type: "QualityDegradation".to_string(),
                severity: degradation_rate.clamp(0.0, 1.0),
                description: format!(
                    "Overall quality dropped by {:.3} between the older and recent windows",
                    degradation_rate
                ),
                immediate_actions: vec![
                    "Investigate recent data ingestion and validation changes".to_string()
                ],
                time_to_failure: None,
            });
        }

        let system_health_status = SystemHealthStatus {
            overall_health_score,
            health_components: HashMap::new(),
            critical_issues,
            health_trend: overall_trend,
            recovery_recommendations: vec![],
        };

        let degradation_forecast = DegradationForecast {
            forecast_horizon: chrono::Duration::days(30),
            predicted_degradations: vec![],
            risk_timeline: vec![],
            mitigation_opportunities: vec![],
        };

        Ok(DegradationDetectionResult {
            quality_trend_analysis,
            performance_degradation,
            content_deterioration,
            usage_pattern_changes,
            system_health_status,
            degradation_forecast,
        })
    }

    /// Emit proactive alerts when the anomaly score or measured degradation
    /// crosses the configured thresholds. Returns an empty vector only when the
    /// signals are genuinely below threshold.
    fn generate_proactive_alerts(
        &self,
        anomaly_detection: &AnomalyDetectionResult,
        degradation_detection: &DegradationDetectionResult,
    ) -> Result<Vec<ProactiveAlert>> {
        let mut alerts = Vec::new();
        let thresholds = &self.config.alert_thresholds;

        // Anomaly-based alert.
        let anomaly_score = anomaly_detection.overall_anomaly_score;
        if anomaly_score >= thresholds.medium_anomaly_score {
            let severity = if anomaly_score >= thresholds.critical_anomaly_score {
                AlertSeverity::Critical
            } else if anomaly_score >= thresholds.high_anomaly_score {
                AlertSeverity::High
            } else {
                AlertSeverity::Medium
            };
            let affected: Vec<String> = anomaly_detection
                .statistical_anomalies
                .iter()
                .flat_map(|a| a.affected_properties.clone())
                .collect();
            alerts.push(ProactiveAlert {
                alert_type: AlertType::DataAnomaly,
                severity,
                confidence: anomaly_score,
                description: format!(
                    "Statistical anomaly score {anomaly_score:.2} across {} detected signal(s)",
                    anomaly_detection.statistical_anomalies.len()
                ),
                affected_components: affected,
                predicted_impact: anomaly_score,
                time_to_criticality: None,
                recommended_actions: vec![
                    "Review the flagged quality signals against recent data changes".to_string(),
                ],
            });
        }

        // Degradation-based alert.
        let degradation_rate = degradation_detection
            .quality_trend_analysis
            .degradation_rate;
        if degradation_rate >= thresholds.degradation_rate_threshold {
            alerts.push(ProactiveAlert {
                alert_type: AlertType::QualityDegradation,
                severity: if degradation_rate >= thresholds.degradation_rate_threshold * 2.0 {
                    AlertSeverity::High
                } else {
                    AlertSeverity::Medium
                },
                confidence: degradation_rate.clamp(0.0, 1.0),
                description: format!(
                    "Quality degradation rate {degradation_rate:.3} exceeds threshold {:.3}",
                    thresholds.degradation_rate_threshold
                ),
                affected_components: degradation_detection
                    .quality_trend_analysis
                    .affected_dimensions
                    .clone(),
                predicted_impact: degradation_rate.clamp(0.0, 1.0),
                time_to_criticality: None,
                recommended_actions: vec![
                    "Investigate the declining quality dimensions".to_string()
                ],
            });
        }

        // Performance-based alert.
        if degradation_detection
            .performance_degradation
            .degradation_detected
        {
            alerts.push(ProactiveAlert {
                alert_type: AlertType::PerformanceIssue,
                severity: AlertSeverity::Medium,
                confidence: 0.7,
                description: "Performance degradation trend detected across snapshots".to_string(),
                affected_components: vec!["validation-pipeline".to_string()],
                predicted_impact: 0.5,
                time_to_criticality: None,
                recommended_actions: vec![
                    "Profile validation throughput and error rate trends".to_string()
                ],
            });
        }

        Ok(alerts)
    }

    fn generate_issue_summary(
        &self,
        anomaly_detection: &AnomalyDetectionResult,
        degradation_detection: &DegradationDetectionResult,
        proactive_alerts: &[ProactiveAlert],
    ) -> IssueSummary {
        let total_anomalies = anomaly_detection.statistical_anomalies.len()
            + anomaly_detection.semantic_anomalies.len()
            + anomaly_detection.structural_anomalies.len()
            + anomaly_detection.behavioral_anomalies.len()
            + anomaly_detection.temporal_anomalies.len()
            + anomaly_detection.cross_reference_anomalies.len();

        let total_issues = total_anomalies + proactive_alerts.len();

        let critical_issues = proactive_alerts
            .iter()
            .filter(|alert| alert.severity == AlertSeverity::Critical)
            .count();

        let high_priority_issues = proactive_alerts
            .iter()
            .filter(|alert| alert.severity >= AlertSeverity::High)
            .count();

        let mut issue_categories = HashMap::new();
        issue_categories.insert("Anomalies".to_string(), total_anomalies);
        issue_categories.insert(
            "Performance".to_string(),
            if degradation_detection
                .performance_degradation
                .degradation_detected
            {
                1
            } else {
                0
            },
        );
        issue_categories.insert(
            "Content".to_string(),
            if degradation_detection
                .content_deterioration
                .deterioration_detected
            {
                1
            } else {
                0
            },
        );

        IssueSummary {
            total_issues,
            critical_issues,
            high_priority_issues,
            issue_categories,
            trend_direction: degradation_detection
                .quality_trend_analysis
                .overall_trend
                .clone(),
            overall_health_score: degradation_detection
                .system_health_status
                .overall_health_score,
        }
    }

    fn generate_issue_recommendations(
        &self,
        _anomaly_detection: &AnomalyDetectionResult,
        _degradation_detection: &DegradationDetectionResult,
        _proactive_alerts: &[ProactiveAlert],
    ) -> Result<Vec<IssueRecommendation>> {
        // Placeholder implementation
        Ok(vec![IssueRecommendation {
            recommendation_type: RecommendationType::MonitoringEnhancement,
            priority: RecommendationPriority::Medium,
            description: "Enhance monitoring for early detection of quality issues".to_string(),
            expected_impact: 0.7,
            implementation_effort: ImplementationEffort::Medium,
            urgency: Urgency::ThisWeek,
            success_probability: 0.8,
        }])
    }

    fn get_models_used(&self) -> Vec<String> {
        let mut models = Vec::new();
        if self.detection_models.statistical_anomaly_detector.is_some() {
            models.push("StatisticalAnomalyDetector".to_string());
        }
        if self.detection_models.semantic_anomaly_detector.is_some() {
            models.push("SemanticAnomalyDetector".to_string());
        }
        if self.detection_models.degradation_monitor.is_some() {
            models.push("DegradationMonitor".to_string());
        }
        models
    }

    fn calculate_overall_confidence(
        &self,
        _anomaly_detection: &AnomalyDetectionResult,
        _degradation_detection: &DegradationDetectionResult,
    ) -> f64 {
        0.82 // Placeholder confidence score
    }

    fn calculate_coverage_percentage(&self, _store: &dyn Store) -> f64 {
        85.0 // Placeholder coverage percentage
    }

    fn update_statistics(
        &mut self,
        anomaly_detection: &AnomalyDetectionResult,
        _degradation_detection: &DegradationDetectionResult,
        detection_duration: std::time::Duration,
    ) {
        self.statistics.total_detections += 1;

        let total_anomalies = anomaly_detection.statistical_anomalies.len()
            + anomaly_detection.semantic_anomalies.len()
            + anomaly_detection.structural_anomalies.len()
            + anomaly_detection.behavioral_anomalies.len()
            + anomaly_detection.temporal_anomalies.len()
            + anomaly_detection.cross_reference_anomalies.len();

        self.statistics.anomalies_detected += total_anomalies;

        // Update average detection time
        let total_time = self.statistics.average_detection_time.as_millis() as f64
            * (self.statistics.total_detections - 1) as f64
            + detection_duration.as_millis() as f64;
        let new_average = total_time / self.statistics.total_detections as f64;
        self.statistics.average_detection_time =
            std::time::Duration::from_millis(new_average as u64);
    }
}

impl Default for QualityIssueDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for AnomalyDetectionResult {
    fn default() -> Self {
        Self {
            statistical_anomalies: vec![],
            semantic_anomalies: vec![],
            structural_anomalies: vec![],
            behavioral_anomalies: vec![],
            temporal_anomalies: vec![],
            cross_reference_anomalies: vec![],
            overall_anomaly_score: 0.0,
            anomaly_trends: AnomalyTrends {
                trend_direction: TrendDirection::Stable,
                anomaly_frequency: 0.0,
                severity_trend: 0.0,
                patterns_identified: vec![],
                forecasted_anomalies: vec![],
            },
        }
    }
}

impl Default for DegradationDetectionResult {
    fn default() -> Self {
        Self {
            quality_trend_analysis: QualityTrendAnalysis {
                overall_trend: TrendDirection::Stable,
                trend_strength: 0.0,
                degradation_rate: 0.0,
                affected_dimensions: vec![],
                trend_forecast: TrendForecast {
                    forecast_horizon: chrono::Duration::days(30),
                    predicted_quality_scores: vec![],
                    confidence_intervals: vec![],
                    risk_assessment: 0.0,
                },
            },
            performance_degradation: PerformanceDegradation {
                degradation_detected: false,
                performance_metrics: PerformanceMetrics {
                    response_time_trend: 0.0,
                    throughput_trend: 0.0,
                    error_rate_trend: 0.0,
                    resource_utilization_trend: 0.0,
                    availability_trend: 0.0,
                },
                bottlenecks_identified: vec![],
                degradation_causes: vec![],
            },
            content_deterioration: ContentDeterioration {
                deterioration_detected: false,
                data_quality_decline: DataQualityDecline {
                    completeness_decline: 0.0,
                    accuracy_decline: 0.0,
                    consistency_decline: 0.0,
                    timeliness_decline: 0.0,
                    validity_decline: 0.0,
                },
                schema_evolution_issues: vec![],
                consistency_degradation: ConsistencyDegradation {
                    logical_consistency_decline: 0.0,
                    referential_consistency_decline: 0.0,
                    semantic_consistency_decline: 0.0,
                    temporal_consistency_decline: 0.0,
                },
            },
            usage_pattern_changes: UsagePatternChanges {
                pattern_changes_detected: false,
                access_pattern_changes: vec![],
                query_pattern_changes: vec![],
                validation_pattern_changes: vec![],
            },
            system_health_status: SystemHealthStatus {
                overall_health_score: 1.0,
                health_components: HashMap::new(),
                critical_issues: vec![],
                health_trend: TrendDirection::Stable,
                recovery_recommendations: vec![],
            },
            degradation_forecast: DegradationForecast {
                forecast_horizon: chrono::Duration::days(30),
                predicted_degradations: vec![],
                risk_timeline: vec![],
                mitigation_opportunities: vec![],
            },
        }
    }
}

impl DetectionModels {
    fn new() -> Self {
        Self {
            statistical_anomaly_detector: None,
            semantic_anomaly_detector: None,
            structural_anomaly_detector: None,
            behavioral_anomaly_detector: None,
            temporal_anomaly_detector: None,
            degradation_monitor: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Statistical helpers for anomaly / degradation detection
// ---------------------------------------------------------------------------

/// Map an anomaly sensitivity in `[0, 1]` to a z-score threshold: higher
/// sensitivity means a lower threshold (flags smaller deviations).
fn anomaly_z_threshold(sensitivity: f64) -> f64 {
    // sensitivity 0.0 -> 3.0 sigma, 1.0 -> 1.0 sigma.
    (3.0 - 2.0 * sensitivity.clamp(0.0, 1.0)).max(0.5)
}

/// Mean of a slice, or `None` if empty.
fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

/// Mean and population standard deviation, or `None` if empty.
fn mean_std(values: &[f64]) -> Option<(f64, f64)> {
    let m = mean(values)?;
    let variance = values.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / values.len() as f64;
    Some((m, variance.sqrt()))
}

/// Least-squares slope of `values` against their indices (0, 1, 2, ...).
/// Returns 0.0 for fewer than two points.
fn linear_slope(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let n_f = n as f64;
    let mean_x = (n_f - 1.0) / 2.0;
    let mean_y = values.iter().sum::<f64>() / n_f;
    let mut num = 0.0;
    let mut den = 0.0;
    for (i, &y) in values.iter().enumerate() {
        let dx = i as f64 - mean_x;
        num += dx * (y - mean_y);
        den += dx * dx;
    }
    if den.abs() < 1e-12 {
        0.0
    } else {
        num / den
    }
}

/// Classify a trend from its slope and volatility.
fn classify_trend(slope: f64, volatility: f64) -> TrendDirection {
    // High volatility with a shallow slope reads as volatile rather than a
    // directional trend.
    if volatility > 0.15 && slope.abs() < 0.01 {
        return TrendDirection::Volatile;
    }
    if slope > 0.005 {
        TrendDirection::Improving
    } else if slope < -0.005 {
        TrendDirection::Declining
    } else {
        TrendDirection::Stable
    }
}

/// Split a history slice into (older, recent) halves for window comparison.
fn split_windows<T>(history: &[T]) -> (&[T], &[T]) {
    let mid = history.len() / 2;
    (&history[..mid], &history[mid..])
}

/// Project a window of snapshots onto a scalar via `f`.
fn window_scores<F>(window: &[QualitySnapshot], f: F) -> Vec<f64>
where
    F: Fn(&QualitySnapshot) -> f64,
{
    window.iter().map(f).collect()
}

/// Build a [`StatisticalAnomaly`] when `current` deviates from the historical
/// distribution by more than `z_threshold` standard deviations.
fn check_statistical_anomaly(
    property: &str,
    current: f64,
    history: &[f64],
    z_threshold: f64,
) -> Option<StatisticalAnomaly> {
    let (m, std) = mean_std(history)?;
    if std < 1e-9 {
        // No variance in the baseline: only a differing value is anomalous.
        if (current - m).abs() < 1e-9 {
            return None;
        }
        return Some(build_statistical_anomaly(
            property,
            current,
            m,
            6.0,
            z_threshold,
        ));
    }
    let z = (current - m) / std;
    if z.abs() < z_threshold {
        return None;
    }
    Some(build_statistical_anomaly(
        property,
        current,
        m,
        z,
        z_threshold,
    ))
}

fn build_statistical_anomaly(
    property: &str,
    current: f64,
    baseline_mean: f64,
    z: f64,
    z_threshold: f64,
) -> StatisticalAnomaly {
    let abs_z = z.abs();
    // Confidence grows with how far the z-score exceeds the threshold.
    let confidence = (abs_z / (abs_z + z_threshold)).clamp(0.0, 1.0);
    StatisticalAnomaly {
        anomaly_type: StatisticalAnomalyType::OutlierValues,
        anomaly_score: (abs_z / 6.0).clamp(0.0, 1.0),
        confidence,
        affected_properties: vec![property.to_string()],
        statistical_measures: StatisticalMeasures {
            z_score: z,
            // Two-sided normal tail approximation of the p-value.
            p_value: normal_two_sided_p_value(abs_z),
            deviation_magnitude: (current - baseline_mean).abs(),
            percentile_rank: normal_cdf(z),
        },
        context: format!(
            "'{property}' = {current:.4} deviates {z:.2}σ from baseline mean {baseline_mean:.4}"
        ),
    }
}

/// Standard-normal CDF via the error function approximation.
fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Two-sided p-value for a standard-normal z-score.
fn normal_two_sided_p_value(abs_z: f64) -> f64 {
    (2.0 * (1.0 - normal_cdf(abs_z))).clamp(0.0, 1.0)
}

/// Abramowitz & Stegun 7.1.26 approximation of the error function.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_issue_detector_creation() {
        let detector = QualityIssueDetector::new();
        assert!(detector.config.enable_anomaly_detection);
        assert!(detector.config.enable_degradation_monitoring);
        assert_eq!(detector.config.anomaly_sensitivity, 0.8);
    }

    #[test]
    fn test_issue_detection_config() {
        let config = IssueDetectionConfig::default();
        assert!(config.enable_proactive_detection);
        assert_eq!(config.degradation_threshold, 0.1);
        assert_eq!(config.min_historical_points, 10);
    }

    #[test]
    fn test_alert_thresholds() {
        let thresholds = AlertThresholds::default();
        assert_eq!(thresholds.critical_anomaly_score, 0.9);
        assert_eq!(thresholds.high_anomaly_score, 0.7);
        assert_eq!(thresholds.degradation_rate_threshold, 0.15);
    }

    #[test]
    fn test_quality_snapshot_creation() {
        let snapshot = QualitySnapshot {
            timestamp: chrono::Utc::now(),
            overall_quality_score: 0.85,
            quality_dimensions: HashMap::new(),
            performance_metrics: PerformanceSnapshot {
                validation_time_ms: 100.0,
                memory_usage_mb: 128.0,
                throughput_ops_per_sec: 50.0,
                error_rate: 0.02,
                resource_utilization: 0.6,
            },
            data_characteristics: DataCharacteristics {
                total_triples: 5000,
                unique_subjects: 1000,
                unique_predicates: 25,
                unique_objects: 2500,
                schema_complexity: 0.5,
                data_density: 0.7,
            },
            validation_results: ValidationSnapshot {
                total_validations: 50,
                successful_validations: 48,
                violation_count: 2,
                average_violation_severity: 0.2,
                validation_coverage: 0.95,
            },
        };

        assert_eq!(snapshot.overall_quality_score, 0.85);
        assert_eq!(snapshot.data_characteristics.total_triples, 5000);
    }
}

#[cfg(test)]
mod regression_tests;
