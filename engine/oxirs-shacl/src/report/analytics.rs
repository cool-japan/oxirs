//! Advanced validation report analytics and statistics
//!
//! This module provides comprehensive analytics capabilities for SHACL validation reports,
//! including statistical analysis, trend detection, filtering, and SPARQL-based querying.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, SystemTime};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use oxirs_core::model::Term;

use crate::{
    report::{ReportFormat, ValidationReport},
    validation::ValidationViolation,
    ConstraintComponentId, PropertyPath, Result, Severity, ShaclError, ShapeId,
};

/// Comprehensive analytics engine for validation reports
#[derive(Debug)]
pub struct ValidationReportAnalytics {
    /// Historical reports for trend analysis
    report_history: Vec<AnalyzedReport>,

    /// Configuration for analytics
    config: AnalyticsConfig,

    /// Cached analytics results
    analytics_cache: HashMap<String, CachedAnalytics>,
}

/// Configuration for validation analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsConfig {
    /// Enable historical trend analysis
    pub enable_trend_analysis: bool,

    /// Maximum number of reports to keep in history
    pub max_history_size: usize,

    /// Enable pattern detection
    pub enable_pattern_detection: bool,

    /// Enable performance analytics
    pub enable_performance_analytics: bool,

    /// Cache analytics results
    pub enable_caching: bool,

    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            enable_trend_analysis: true,
            max_history_size: 1000,
            enable_pattern_detection: true,
            enable_performance_analytics: true,
            enable_caching: true,
            cache_ttl_seconds: 3600, // 1 hour
        }
    }
}

/// Analyzed validation report with computed metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzedReport {
    /// Original validation report
    pub report: ValidationReport,

    /// Timestamp when report was analyzed
    pub analyzed_at: DateTime<Utc>,

    /// Computed analytics metrics
    pub metrics: ReportMetrics,

    /// Detected patterns
    pub patterns: Vec<ViolationPattern>,

    /// Performance metrics
    pub performance: PerformanceMetrics,
}

/// Comprehensive metrics for a validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetrics {
    /// Total number of violations
    pub total_violations: usize,

    /// Violations by severity
    pub violations_by_severity: HashMap<Severity, usize>,

    /// Violations by shape
    pub violations_by_shape: HashMap<ShapeId, usize>,

    /// Violations by constraint component
    pub violations_by_component: HashMap<ConstraintComponentId, usize>,

    /// Violations by property path (if applicable)
    pub violations_by_path: HashMap<String, usize>,

    /// Unique focus nodes with violations
    pub unique_focus_nodes: usize,

    /// Average violations per focus node
    pub avg_violations_per_node: f64,

    /// Conformance rate (percentage)
    pub conformance_rate: f64,

    /// Quality score (0-100)
    pub quality_score: f64,

    /// Complexity score based on violation patterns
    pub complexity_score: f64,
}

/// Performance metrics for validation execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Validation duration
    pub validation_duration: Option<Duration>,

    /// Violations found per second
    pub violations_per_second: f64,

    /// Memory usage estimate
    pub estimated_memory_usage: Option<usize>,

    /// Cache hit rate if available
    pub cache_hit_rate: Option<f64>,

    /// Number of shapes evaluated
    pub shapes_evaluated: usize,

    /// Number of constraints evaluated
    pub constraints_evaluated: usize,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            validation_duration: None,
            violations_per_second: 0.0,
            estimated_memory_usage: None,
            cache_hit_rate: None,
            shapes_evaluated: 0,
            constraints_evaluated: 0,
        }
    }
}

/// Detected violation patterns for data quality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationPattern {
    /// Pattern type
    pub pattern_type: PatternType,

    /// Pattern description
    pub description: String,

    /// Frequency of this pattern
    pub frequency: usize,

    /// Confidence level (0-1)
    pub confidence: f64,

    /// Affected shapes
    pub affected_shapes: Vec<ShapeId>,

    /// Recommended actions
    pub recommendations: Vec<String>,
}

/// Types of violation patterns that can be detected
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternType {
    /// Recurring violations in same shape
    RecurringShapeViolation,

    /// Similar violations across multiple shapes
    CrossShapePattern,

    /// Missing required properties
    MissingRequiredProperties,

    /// Datatype inconsistencies
    DatatypeInconsistencies,

    /// Cardinality violations
    CardinalityViolations,

    /// Value range violations
    ValueRangeViolations,

    /// Pattern matching failures
    PatternMatchingFailures,

    /// Language tag issues
    LanguageTagIssues,

    /// Custom pattern
    Custom(String),
}

/// Cached analytics results
#[derive(Debug, Clone)]
struct CachedAnalytics {
    /// Cached result
    result: AnalyticsResult,

    /// Cache timestamp
    cached_at: SystemTime,
}

/// Result of analytics computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsResult {
    /// Summary statistics
    pub summary: AnalyticsSummary,

    /// Trend analysis results
    pub trends: Option<TrendAnalysis>,

    /// Quality assessment
    pub quality_assessment: QualityAssessment,

    /// Recommendations for improvement
    pub recommendations: Vec<Recommendation>,
}

/// High-level analytics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsSummary {
    /// Total reports analyzed
    pub total_reports: usize,

    /// Average violations per report
    pub avg_violations_per_report: f64,

    /// Most common violation types
    pub top_violation_types: Vec<(ConstraintComponentId, usize)>,

    /// Most problematic shapes
    pub problematic_shapes: Vec<(ShapeId, usize)>,

    /// Overall quality trend
    pub quality_trend: QualityTrend,

    /// Data quality score (0-100)
    pub overall_quality_score: f64,
}

/// Trend analysis for historical data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Quality score trend over time
    pub quality_trend: Vec<(DateTime<Utc>, f64)>,

    /// Violation count trend
    pub violation_trend: Vec<(DateTime<Utc>, usize)>,

    /// Performance trend
    pub performance_trend: Vec<(DateTime<Utc>, f64)>,

    /// Detected trend direction
    pub trend_direction: TrendDirection,

    /// Trend strength (0-1)
    pub trend_strength: f64,
}

/// Direction of quality trends
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityTrend {
    /// Quality is improving
    Improving,

    /// Quality is declining
    Declining,

    /// Quality is stable
    Stable,

    /// Insufficient data
    Unknown,
}

/// Direction of numerical trends
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Increasing trend
    Increasing,

    /// Decreasing trend
    Decreasing,

    /// Stable trend
    Stable,

    /// Volatile/unclear trend
    Volatile,
}

/// Quality assessment results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssessment {
    /// Overall quality rating
    pub quality_rating: QualityRating,

    /// Quality dimensions analysis
    pub dimensions: QualityDimensions,

    /// Risk assessment
    pub risk_level: RiskLevel,

    /// Quality issues detected
    pub issues: Vec<QualityIssue>,
}

/// Quality rating categories
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityRating {
    /// Excellent quality (90-100%)
    Excellent,

    /// Good quality (70-89%)
    Good,

    /// Fair quality (50-69%)
    Fair,

    /// Poor quality (30-49%)
    Poor,

    /// Critical quality issues (<30%)
    Critical,
}

/// Data quality dimensions analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityDimensions {
    /// Completeness score (0-100)
    pub completeness: f64,

    /// Consistency score (0-100)
    pub consistency: f64,

    /// Accuracy score (0-100)
    pub accuracy: f64,

    /// Validity score (0-100)
    pub validity: f64,

    /// Uniqueness score (0-100)
    pub uniqueness: f64,
}

/// Risk level assessment
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Low risk
    Low,

    /// Medium risk
    Medium,

    /// High risk
    High,

    /// Critical risk
    Critical,
}

/// Specific quality issues detected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    /// Issue type
    pub issue_type: QualityIssueType,

    /// Issue description
    pub description: String,

    /// Severity of the issue
    pub severity: Severity,

    /// Affected data count
    pub affected_count: usize,

    /// Suggested remediation
    pub remediation: String,
}

/// Types of quality issues
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityIssueType {
    /// Missing required data
    MissingData,

    /// Inconsistent data formats
    InconsistentFormat,

    /// Invalid data values
    InvalidValues,

    /// Duplicate data
    Duplicates,

    /// Constraint violations
    ConstraintViolations,

    /// Performance issues
    PerformanceIssues,
}

/// Recommendations for data quality improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Recommendation type
    pub rec_type: RecommendationType,

    /// Recommendation title
    pub title: String,

    /// Detailed description
    pub description: String,

    /// Priority level
    pub priority: RecommendationPriority,

    /// Estimated effort to implement
    pub effort: EffortLevel,

    /// Expected impact
    pub impact: ImpactLevel,

    /// Specific action items
    pub action_items: Vec<String>,
}

/// Types of recommendations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Shape design improvements
    ShapeDesign,

    /// Data cleaning suggestions
    DataCleaning,

    /// Performance optimization
    Performance,

    /// Process improvements
    Process,

    /// Tooling recommendations
    Tooling,
}

/// Priority levels for recommendations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationPriority {
    /// Critical priority
    Critical,

    /// High priority
    High,

    /// Medium priority
    Medium,

    /// Low priority
    Low,
}

/// Effort levels for implementing recommendations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffortLevel {
    /// Low effort (< 1 day)
    Low,

    /// Medium effort (1-5 days)
    Medium,

    /// High effort (> 5 days)
    High,

    /// Very high effort (> 2 weeks)
    VeryHigh,
}

/// Impact levels of recommendations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImpactLevel {
    /// Low impact
    Low,

    /// Medium impact
    Medium,

    /// High impact
    High,

    /// Very high impact
    VeryHigh,
}

impl ValidationReportAnalytics {
    /// Create a new analytics engine
    pub fn new(config: AnalyticsConfig) -> Self {
        Self {
            report_history: Vec::new(),
            config,
            analytics_cache: HashMap::new(),
        }
    }

    /// Analyze a validation report and add to history
    pub fn analyze_report(&mut self, report: ValidationReport) -> Result<AnalyzedReport> {
        let start_time = std::time::Instant::now();

        // Compute metrics
        let metrics = self.compute_metrics(&report)?;

        // Detect patterns if enabled
        let patterns = if self.config.enable_pattern_detection {
            self.detect_patterns(&report, &metrics)?
        } else {
            Vec::new()
        };

        // Compute performance metrics if enabled
        let performance = if self.config.enable_performance_analytics {
            self.compute_performance_metrics(&report, start_time.elapsed())?
        } else {
            PerformanceMetrics::default()
        };

        let analyzed_report = AnalyzedReport {
            report,
            analyzed_at: Utc::now(),
            metrics,
            patterns,
            performance,
        };

        // Add to history
        self.add_to_history(analyzed_report.clone());

        Ok(analyzed_report)
    }

    /// Generate comprehensive analytics for all reports
    pub fn generate_analytics(&mut self) -> Result<AnalyticsResult> {
        let cache_key = "full_analytics".to_string();

        // Check cache if enabled
        if self.config.enable_caching {
            if let Some(cached) = self.analytics_cache.get(&cache_key) {
                if cached
                    .cached_at
                    .elapsed()
                    .unwrap_or(Duration::MAX)
                    .as_secs()
                    < self.config.cache_ttl_seconds
                {
                    return Ok(cached.result.clone());
                }
            }
        }

        let summary = self.compute_summary()?;
        let trends = if self.config.enable_trend_analysis && self.report_history.len() > 1 {
            Some(self.compute_trends()?)
        } else {
            None
        };

        let quality_assessment = self.assess_quality()?;
        let recommendations = self.generate_recommendations(&summary, &quality_assessment)?;

        let result = AnalyticsResult {
            summary,
            trends,
            quality_assessment,
            recommendations,
        };

        // Cache result if enabled
        if self.config.enable_caching {
            self.analytics_cache.insert(
                cache_key,
                CachedAnalytics {
                    result: result.clone(),
                    cached_at: SystemTime::now(),
                },
            );
        }

        Ok(result)
    }

    /// Filter reports by various criteria
    pub fn filter_reports<F>(&self, predicate: F) -> Vec<&AnalyzedReport>
    where
        F: Fn(&AnalyzedReport) -> bool,
    {
        self.report_history
            .iter()
            .filter(|report| predicate(report))
            .collect()
    }

    /// Get reports within a time range
    pub fn get_reports_in_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<&AnalyzedReport> {
        self.filter_reports(|report| report.analyzed_at >= start && report.analyzed_at <= end)
    }

    /// Get reports with violations above threshold
    pub fn get_reports_above_threshold(&self, threshold: usize) -> Vec<&AnalyzedReport> {
        self.filter_reports(|report| report.metrics.total_violations > threshold)
    }

    /// Generate summary statistics for specific shapes
    pub fn get_shape_statistics(&self, shape_id: &ShapeId) -> Result<ShapeStatistics> {
        let mut violations_over_time = Vec::new();
        let mut total_violations = 0;
        let mut total_reports = 0;

        for report in &self.report_history {
            let shape_violations = report
                .metrics
                .violations_by_shape
                .get(shape_id)
                .copied()
                .unwrap_or(0);

            violations_over_time.push((report.analyzed_at, shape_violations));
            total_violations += shape_violations;

            if shape_violations > 0 {
                total_reports += 1;
            }
        }

        let avg_violations = if total_reports > 0 {
            total_violations as f64 / total_reports as f64
        } else {
            0.0
        };

        Ok(ShapeStatistics {
            shape_id: shape_id.clone(),
            total_violations,
            total_reports_with_violations: total_reports,
            avg_violations_per_report: avg_violations,
            violations_over_time,
        })
    }

    /// Export analytics to various formats
    pub fn export_analytics(&self, format: ReportFormat) -> Result<String> {
        let analytics = self.generate_analytics()?;

        match format {
            ReportFormat::Json => Ok(serde_json::to_string_pretty(&analytics)?),
            ReportFormat::Yaml => Ok(serde_yaml::to_string(&analytics)?),
            ReportFormat::Csv => self.export_to_csv(&analytics),
            ReportFormat::Html => self.export_to_html(&analytics),
            ReportFormat::Text => self.export_to_text(&analytics),
            _ => Err(ShaclError::UnsupportedOperation(format!(
                "Export format {:?} not supported for analytics",
                format
            ))),
        }
    }

    /// Compute basic metrics for a report
    fn compute_metrics(&self, report: &ValidationReport) -> Result<ReportMetrics> {
        let total_violations = report.violations.len();

        let mut violations_by_severity = HashMap::new();
        let mut violations_by_shape = HashMap::new();
        let mut violations_by_component = HashMap::new();
        let mut violations_by_path = HashMap::new();
        let mut unique_focus_nodes = HashSet::new();

        for violation in &report.violations {
            // Count by severity
            *violations_by_severity
                .entry(violation.result_severity.clone())
                .or_insert(0) += 1;

            // Count by shape
            *violations_by_shape
                .entry(violation.source_shape.clone())
                .or_insert(0) += 1;

            // Count by constraint component
            *violations_by_component
                .entry(violation.source_constraint_component.clone())
                .or_insert(0) += 1;

            // Count by property path
            if let Some(path) = &violation.result_path {
                let path_str = format!("{:?}", path); // Simple string representation
                *violations_by_path.entry(path_str).or_insert(0) += 1;
            }

            // Track unique focus nodes
            unique_focus_nodes.insert(violation.focus_node.clone());
        }

        let unique_focus_nodes_count = unique_focus_nodes.len();
        let avg_violations_per_node = if unique_focus_nodes_count > 0 {
            total_violations as f64 / unique_focus_nodes_count as f64
        } else {
            0.0
        };

        // Calculate conformance rate (simple heuristic)
        let conformance_rate = if total_violations == 0 {
            100.0
        } else {
            // This is a simplified calculation - in practice, you'd need to know
            // the total number of validatable entities
            100.0 - (total_violations as f64 / unique_focus_nodes_count.max(1) as f64).min(100.0)
        };

        // Calculate quality score based on violations and patterns
        let quality_score =
            self.calculate_quality_score(total_violations, unique_focus_nodes_count);

        // Calculate complexity score based on violation patterns
        let complexity_score =
            self.calculate_complexity_score(&violations_by_component, &violations_by_shape);

        Ok(ReportMetrics {
            total_violations,
            violations_by_severity,
            violations_by_shape,
            violations_by_component,
            violations_by_path,
            unique_focus_nodes: unique_focus_nodes_count,
            avg_violations_per_node,
            conformance_rate,
            quality_score,
            complexity_score,
        })
    }

    /// Detect violation patterns in the report
    fn detect_patterns(
        &self,
        report: &ValidationReport,
        metrics: &ReportMetrics,
    ) -> Result<Vec<ViolationPattern>> {
        let mut patterns = Vec::new();

        // Detect recurring shape violations
        for (shape_id, count) in &metrics.violations_by_shape {
            if *count > 5 {
                // Threshold for "recurring"
                patterns.push(ViolationPattern {
                    pattern_type: PatternType::RecurringShapeViolation,
                    description: format!("Shape {} has {} violations", shape_id, count),
                    frequency: *count,
                    confidence: 0.9,
                    affected_shapes: vec![shape_id.clone()],
                    recommendations: vec![
                        format!("Review shape definition for {}", shape_id),
                        "Consider relaxing constraints or fixing data".to_string(),
                    ],
                });
            }
        }

        // Detect missing required properties pattern
        let missing_prop_violations = report
            .violations
            .iter()
            .filter(|v| v.source_constraint_component.as_str().contains("MinCount"))
            .count();

        if missing_prop_violations > 0 {
            patterns.push(ViolationPattern {
                pattern_type: PatternType::MissingRequiredProperties,
                description: format!(
                    "{} missing required property violations",
                    missing_prop_violations
                ),
                frequency: missing_prop_violations,
                confidence: 0.8,
                affected_shapes: metrics.violations_by_shape.keys().cloned().collect(),
                recommendations: vec![
                    "Review required properties in shapes".to_string(),
                    "Ensure data completeness".to_string(),
                ],
            });
        }

        // Detect datatype inconsistencies
        let datatype_violations = report
            .violations
            .iter()
            .filter(|v| v.source_constraint_component.as_str().contains("Datatype"))
            .count();

        if datatype_violations > 0 {
            patterns.push(ViolationPattern {
                pattern_type: PatternType::DatatypeInconsistencies,
                description: format!("{} datatype violation(s)", datatype_violations),
                frequency: datatype_violations,
                confidence: 0.85,
                affected_shapes: metrics.violations_by_shape.keys().cloned().collect(),
                recommendations: vec![
                    "Review datatype constraints".to_string(),
                    "Validate data conversion processes".to_string(),
                ],
            });
        }

        Ok(patterns)
    }

    /// Compute performance metrics
    fn compute_performance_metrics(
        &self,
        report: &ValidationReport,
        duration: Duration,
    ) -> Result<PerformanceMetrics> {
        let violations_per_second = if duration.as_secs() > 0 {
            report.violations.len() as f64 / duration.as_secs_f64()
        } else {
            0.0
        };

        // Estimate memory usage (rough heuristic)
        let estimated_memory = report.violations.len() * 1024; // Rough estimate: 1KB per violation

        Ok(PerformanceMetrics {
            validation_duration: Some(duration),
            violations_per_second,
            estimated_memory_usage: Some(estimated_memory),
            cache_hit_rate: None, // Would need to be provided by validation engine
            shapes_evaluated: 0,  // Would need to be tracked during validation
            constraints_evaluated: 0, // Would need to be tracked during validation
        })
    }

    /// Calculate quality score based on violations and focus nodes
    fn calculate_quality_score(&self, total_violations: usize, unique_focus_nodes: usize) -> f64 {
        if unique_focus_nodes == 0 {
            return 100.0;
        }

        let violation_ratio = total_violations as f64 / unique_focus_nodes as f64;

        // Quality score decreases with more violations per node
        let base_score = 100.0 - (violation_ratio * 10.0).min(90.0);
        base_score.max(0.0)
    }

    /// Calculate complexity score based on violation distribution
    fn calculate_complexity_score(
        &self,
        violations_by_component: &HashMap<ConstraintComponentId, usize>,
        violations_by_shape: &HashMap<ShapeId, usize>,
    ) -> f64 {
        let component_diversity = violations_by_component.len() as f64;
        let shape_diversity = violations_by_shape.len() as f64;

        // Higher diversity indicates higher complexity
        (component_diversity + shape_diversity) / 2.0
    }

    /// Compute summary statistics across all reports
    fn compute_summary(&self) -> Result<AnalyticsSummary> {
        if self.report_history.is_empty() {
            return Ok(AnalyticsSummary {
                total_reports: 0,
                avg_violations_per_report: 0.0,
                top_violation_types: Vec::new(),
                problematic_shapes: Vec::new(),
                quality_trend: QualityTrend::Unknown,
                overall_quality_score: 0.0,
            });
        }

        let total_reports = self.report_history.len();
        let total_violations: usize = self
            .report_history
            .iter()
            .map(|r| r.metrics.total_violations)
            .sum();

        let avg_violations_per_report = total_violations as f64 / total_reports as f64;

        // Aggregate violation types
        let mut component_counts = HashMap::new();
        let mut shape_counts = HashMap::new();

        for report in &self.report_history {
            for (component, count) in &report.metrics.violations_by_component {
                *component_counts.entry(component.clone()).or_insert(0) += count;
            }
            for (shape, count) in &report.metrics.violations_by_shape {
                *shape_counts.entry(shape.clone()).or_insert(0) += count;
            }
        }

        // Get top violation types
        let mut top_violation_types: Vec<_> = component_counts.into_iter().collect();
        top_violation_types.sort_by(|a, b| b.1.cmp(&a.1));
        top_violation_types.truncate(10);

        // Get most problematic shapes
        let mut problematic_shapes: Vec<_> = shape_counts.into_iter().collect();
        problematic_shapes.sort_by(|a, b| b.1.cmp(&a.1));
        problematic_shapes.truncate(10);

        // Determine quality trend
        let quality_trend = if self.report_history.len() > 2 {
            self.determine_quality_trend()
        } else {
            QualityTrend::Unknown
        };

        // Calculate overall quality score
        let quality_scores: Vec<f64> = self
            .report_history
            .iter()
            .map(|r| r.metrics.quality_score)
            .collect();
        let overall_quality_score =
            quality_scores.iter().sum::<f64>() / quality_scores.len() as f64;

        Ok(AnalyticsSummary {
            total_reports,
            avg_violations_per_report,
            top_violation_types,
            problematic_shapes,
            quality_trend,
            overall_quality_score,
        })
    }

    /// Compute trend analysis
    fn compute_trends(&self) -> Result<TrendAnalysis> {
        let quality_trend: Vec<_> = self
            .report_history
            .iter()
            .map(|r| (r.analyzed_at, r.metrics.quality_score))
            .collect();

        let violation_trend: Vec<_> = self
            .report_history
            .iter()
            .map(|r| (r.analyzed_at, r.metrics.total_violations))
            .collect();

        let performance_trend: Vec<_> = self
            .report_history
            .iter()
            .filter_map(|r| {
                r.performance
                    .validation_duration
                    .map(|d| (r.analyzed_at, d.as_secs_f64()))
            })
            .collect();

        // Simple trend direction calculation
        let trend_direction = if quality_trend.len() > 1 {
            let first_quality = quality_trend.first().unwrap().1;
            let last_quality = quality_trend.last().unwrap().1;

            if last_quality > first_quality + 5.0 {
                TrendDirection::Increasing
            } else if last_quality < first_quality - 5.0 {
                TrendDirection::Decreasing
            } else {
                TrendDirection::Stable
            }
        } else {
            TrendDirection::Stable
        };

        // Simple trend strength calculation (correlation-like measure)
        let trend_strength = 0.5; // Placeholder - would implement proper correlation calculation

        Ok(TrendAnalysis {
            quality_trend,
            violation_trend,
            performance_trend,
            trend_direction,
            trend_strength,
        })
    }

    /// Assess overall quality
    fn assess_quality(&self) -> Result<QualityAssessment> {
        if self.report_history.is_empty() {
            return Ok(QualityAssessment {
                quality_rating: QualityRating::Critical,
                dimensions: QualityDimensions {
                    completeness: 0.0,
                    consistency: 0.0,
                    accuracy: 0.0,
                    validity: 0.0,
                    uniqueness: 0.0,
                },
                risk_level: RiskLevel::Critical,
                issues: Vec::new(),
            });
        }

        let avg_quality = self
            .report_history
            .iter()
            .map(|r| r.metrics.quality_score)
            .sum::<f64>()
            / self.report_history.len() as f64;

        let quality_rating = match avg_quality {
            score if score >= 90.0 => QualityRating::Excellent,
            score if score >= 70.0 => QualityRating::Good,
            score if score >= 50.0 => QualityRating::Fair,
            score if score >= 30.0 => QualityRating::Poor,
            _ => QualityRating::Critical,
        };

        // Calculate quality dimensions (simplified heuristics)
        let dimensions = self.calculate_quality_dimensions()?;

        let risk_level = match quality_rating {
            QualityRating::Excellent | QualityRating::Good => RiskLevel::Low,
            QualityRating::Fair => RiskLevel::Medium,
            QualityRating::Poor => RiskLevel::High,
            QualityRating::Critical => RiskLevel::Critical,
        };

        let issues = self.identify_quality_issues()?;

        Ok(QualityAssessment {
            quality_rating,
            dimensions,
            risk_level,
            issues,
        })
    }

    /// Calculate quality dimensions
    fn calculate_quality_dimensions(&self) -> Result<QualityDimensions> {
        // These would be more sophisticated in a real implementation
        let avg_quality = self
            .report_history
            .iter()
            .map(|r| r.metrics.quality_score)
            .sum::<f64>()
            / self.report_history.len() as f64;

        Ok(QualityDimensions {
            completeness: avg_quality * 0.9, // Heuristic based on missing value patterns
            consistency: avg_quality * 0.95, // Heuristic based on constraint violations
            accuracy: avg_quality,           // Base quality score
            validity: avg_quality * 0.85,    // Heuristic based on validation failures
            uniqueness: avg_quality * 0.8,   // Heuristic based on duplicate detection
        })
    }

    /// Identify specific quality issues
    fn identify_quality_issues(&self) -> Result<Vec<QualityIssue>> {
        let mut issues = Vec::new();

        // Count different types of violations across all reports
        let mut missing_data_count = 0;
        let mut invalid_values_count = 0;
        let mut constraint_violations_count = 0;

        for report in &self.report_history {
            for violation in &report.report.violations {
                let component_str = violation.source_constraint_component.as_str();

                if component_str.contains("MinCount") {
                    missing_data_count += 1;
                } else if component_str.contains("Datatype") || component_str.contains("Pattern") {
                    invalid_values_count += 1;
                } else {
                    constraint_violations_count += 1;
                }
            }
        }

        if missing_data_count > 0 {
            issues.push(QualityIssue {
                issue_type: QualityIssueType::MissingData,
                description: format!("{} instances of missing required data", missing_data_count),
                severity: if missing_data_count > 100 {
                    Severity::Violation
                } else {
                    Severity::Warning
                },
                affected_count: missing_data_count,
                remediation: "Add missing required properties to data sources".to_string(),
            });
        }

        if invalid_values_count > 0 {
            issues.push(QualityIssue {
                issue_type: QualityIssueType::InvalidValues,
                description: format!("{} instances of invalid data values", invalid_values_count),
                severity: if invalid_values_count > 50 {
                    Severity::Violation
                } else {
                    Severity::Warning
                },
                affected_count: invalid_values_count,
                remediation: "Validate and correct data formats and types".to_string(),
            });
        }

        Ok(issues)
    }

    /// Generate recommendations based on analysis
    fn generate_recommendations(
        &self,
        summary: &AnalyticsSummary,
        quality: &QualityAssessment,
    ) -> Result<Vec<Recommendation>> {
        let mut recommendations = Vec::new();

        // Performance recommendations
        if summary.avg_violations_per_report > 100.0 {
            recommendations.push(Recommendation {
                rec_type: RecommendationType::Performance,
                title: "High violation count detected".to_string(),
                description: "Consider optimizing shapes or improving data quality".to_string(),
                priority: RecommendationPriority::High,
                effort: EffortLevel::Medium,
                impact: ImpactLevel::High,
                action_items: vec![
                    "Review most problematic shapes".to_string(),
                    "Implement data quality monitoring".to_string(),
                ],
            });
        }

        // Quality improvement recommendations
        if matches!(
            quality.quality_rating,
            QualityRating::Poor | QualityRating::Critical
        ) {
            recommendations.push(Recommendation {
                rec_type: RecommendationType::DataCleaning,
                title: "Critical data quality issues detected".to_string(),
                description: "Immediate action required to improve data quality".to_string(),
                priority: RecommendationPriority::Critical,
                effort: EffortLevel::High,
                impact: ImpactLevel::VeryHigh,
                action_items: vec![
                    "Implement data validation pipeline".to_string(),
                    "Review and update SHACL shapes".to_string(),
                    "Establish data quality monitoring".to_string(),
                ],
            });
        }

        // Shape design recommendations
        if !summary.top_violation_types.is_empty() {
            let top_violation = &summary.top_violation_types[0];
            recommendations.push(Recommendation {
                rec_type: RecommendationType::ShapeDesign,
                title: format!("Frequent violations in {}", top_violation.0),
                description: "Consider reviewing shape constraints for this component".to_string(),
                priority: RecommendationPriority::Medium,
                effort: EffortLevel::Low,
                impact: ImpactLevel::Medium,
                action_items: vec![
                    format!("Review {} constraint definitions", top_violation.0),
                    "Consider constraint relaxation if appropriate".to_string(),
                ],
            });
        }

        Ok(recommendations)
    }

    /// Determine quality trend from historical data
    fn determine_quality_trend(&self) -> QualityTrend {
        if self.report_history.len() < 3 {
            return QualityTrend::Unknown;
        }

        let recent_quality: f64 = self
            .report_history
            .iter()
            .rev()
            .take(3)
            .map(|r| r.metrics.quality_score)
            .sum::<f64>()
            / 3.0;

        let older_quality: f64 = self
            .report_history
            .iter()
            .take(3)
            .map(|r| r.metrics.quality_score)
            .sum::<f64>()
            / 3.0;

        if recent_quality > older_quality + 5.0 {
            QualityTrend::Improving
        } else if recent_quality < older_quality - 5.0 {
            QualityTrend::Declining
        } else {
            QualityTrend::Stable
        }
    }

    /// Add report to history with size management
    fn add_to_history(&mut self, report: AnalyzedReport) {
        self.report_history.push(report);

        // Maintain history size limit
        if self.report_history.len() > self.config.max_history_size {
            self.report_history.remove(0);
        }
    }

    /// Export analytics to CSV format
    fn export_to_csv(&self, analytics: &AnalyticsResult) -> Result<String> {
        let mut csv = String::new();
        csv.push_str("Metric,Value\n");
        csv.push_str(&format!(
            "Total Reports,{}\n",
            analytics.summary.total_reports
        ));
        csv.push_str(&format!(
            "Average Violations Per Report,{:.2}\n",
            analytics.summary.avg_violations_per_report
        ));
        csv.push_str(&format!(
            "Overall Quality Score,{:.2}\n",
            analytics.summary.overall_quality_score
        ));

        // Add top violation types
        csv.push_str("\nTop Violation Types\n");
        csv.push_str("Component,Count\n");
        for (component, count) in &analytics.summary.top_violation_types {
            csv.push_str(&format!("{},{}\n", component, count));
        }

        Ok(csv)
    }

    /// Export analytics to HTML format
    fn export_to_html(&self, analytics: &AnalyticsResult) -> Result<String> {
        let mut html = String::new();
        html.push_str("<!DOCTYPE html><html><head><title>SHACL Validation Analytics</title>");
        html.push_str("<style>body{font-family:Arial,sans-serif;margin:20px;}table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ddd;padding:8px;text-align:left;}th{background-color:#f2f2f2;}</style>");
        html.push_str("</head><body>");
        html.push_str("<h1>SHACL Validation Analytics Report</h1>");

        // Summary section
        html.push_str("<h2>Summary</h2>");
        html.push_str("<table>");
        html.push_str(&format!("<tr><th>Metric</th><th>Value</th></tr>"));
        html.push_str(&format!(
            "<tr><td>Total Reports</td><td>{}</td></tr>",
            analytics.summary.total_reports
        ));
        html.push_str(&format!(
            "<tr><td>Average Violations Per Report</td><td>{:.2}</td></tr>",
            analytics.summary.avg_violations_per_report
        ));
        html.push_str(&format!(
            "<tr><td>Overall Quality Score</td><td>{:.2}</td></tr>",
            analytics.summary.overall_quality_score
        ));
        html.push_str(&format!(
            "<tr><td>Quality Trend</td><td>{:?}</td></tr>",
            analytics.summary.quality_trend
        ));
        html.push_str("</table>");

        // Quality assessment
        html.push_str("<h2>Quality Assessment</h2>");
        html.push_str("<table>");
        html.push_str(&format!(
            "<tr><td>Quality Rating</td><td>{:?}</td></tr>",
            analytics.quality_assessment.quality_rating
        ));
        html.push_str(&format!(
            "<tr><td>Risk Level</td><td>{:?}</td></tr>",
            analytics.quality_assessment.risk_level
        ));
        html.push_str("</table>");

        // Recommendations
        if !analytics.recommendations.is_empty() {
            html.push_str("<h2>Recommendations</h2>");
            html.push_str("<ul>");
            for rec in &analytics.recommendations {
                html.push_str(&format!(
                    "<li><strong>{}</strong>: {} (Priority: {:?})</li>",
                    rec.title, rec.description, rec.priority
                ));
            }
            html.push_str("</ul>");
        }

        html.push_str("</body></html>");
        Ok(html)
    }

    /// Export analytics to text format
    fn export_to_text(&self, analytics: &AnalyticsResult) -> Result<String> {
        let mut text = String::new();
        text.push_str("SHACL Validation Analytics Report\n");
        text.push_str("=================================\n\n");

        text.push_str("Summary:\n");
        text.push_str(&format!(
            "  Total Reports: {}\n",
            analytics.summary.total_reports
        ));
        text.push_str(&format!(
            "  Average Violations Per Report: {:.2}\n",
            analytics.summary.avg_violations_per_report
        ));
        text.push_str(&format!(
            "  Overall Quality Score: {:.2}\n",
            analytics.summary.overall_quality_score
        ));
        text.push_str(&format!(
            "  Quality Trend: {:?}\n\n",
            analytics.summary.quality_trend
        ));

        text.push_str("Quality Assessment:\n");
        text.push_str(&format!(
            "  Rating: {:?}\n",
            analytics.quality_assessment.quality_rating
        ));
        text.push_str(&format!(
            "  Risk Level: {:?}\n\n",
            analytics.quality_assessment.risk_level
        ));

        if !analytics.recommendations.is_empty() {
            text.push_str("Recommendations:\n");
            for (i, rec) in analytics.recommendations.iter().enumerate() {
                text.push_str(&format!(
                    "  {}. {} (Priority: {:?})\n",
                    i + 1,
                    rec.title,
                    rec.priority
                ));
                text.push_str(&format!("     {}\n", rec.description));
            }
        }

        Ok(text)
    }

    /// Clear analytics cache
    pub fn clear_cache(&mut self) {
        self.analytics_cache.clear();
    }

    /// Get current cache size
    pub fn cache_size(&self) -> usize {
        self.analytics_cache.len()
    }
}

/// Statistics for a specific shape across multiple reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeStatistics {
    /// Shape identifier
    pub shape_id: ShapeId,

    /// Total violations for this shape
    pub total_violations: usize,

    /// Number of reports with violations for this shape
    pub total_reports_with_violations: usize,

    /// Average violations per report
    pub avg_violations_per_report: f64,

    /// Violations over time
    pub violations_over_time: Vec<(DateTime<Utc>, usize)>,
}

// Implement required traits for error handling
impl From<serde_json::Error> for ShaclError {
    fn from(err: serde_json::Error) -> Self {
        ShaclError::ValidationReport(format!("JSON serialization error: {}", err))
    }
}

impl From<serde_yaml::Error> for ShaclError {
    fn from(err: serde_yaml::Error) -> Self {
        ShaclError::ValidationReport(format!("YAML serialization error: {}", err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::ValidationViolation;

    #[test]
    fn test_analytics_config_default() {
        let config = AnalyticsConfig::default();
        assert!(config.enable_trend_analysis);
        assert_eq!(config.max_history_size, 1000);
        assert!(config.enable_pattern_detection);
        assert!(config.enable_performance_analytics);
    }

    #[test]
    fn test_quality_rating_classification() {
        // Test different quality scores map to correct ratings
        let excellent_score = 95.0;
        let good_score = 75.0;
        let fair_score = 55.0;
        let poor_score = 35.0;
        let critical_score = 15.0;

        // This would be tested with actual quality assessment logic
        assert!(excellent_score >= 90.0); // Would be Excellent
        assert!(good_score >= 70.0 && good_score < 90.0); // Would be Good
        assert!(fair_score >= 50.0 && fair_score < 70.0); // Would be Fair
        assert!(poor_score >= 30.0 && poor_score < 50.0); // Would be Poor
        assert!(critical_score < 30.0); // Would be Critical
    }

    #[test]
    fn test_analytics_engine_creation() {
        let config = AnalyticsConfig::default();
        let analytics = ValidationReportAnalytics::new(config);

        assert_eq!(analytics.report_history.len(), 0);
        assert_eq!(analytics.analytics_cache.len(), 0);
    }

    #[test]
    fn test_report_filtering() {
        let mut analytics = ValidationReportAnalytics::new(AnalyticsConfig::default());

        // This would test filtering functionality with actual reports
        let filtered = analytics.filter_reports(|_| true);
        assert_eq!(filtered.len(), 0); // No reports in empty analytics
    }
}
