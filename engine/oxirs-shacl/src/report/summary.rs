//\! Validation summary statistics and analysis

use crate::{validation::ValidationViolation, ConstraintComponentId, Severity, ShapeId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Summary statistics for validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    /// Total number of violations by severity
    pub violations_by_severity: HashMap<Severity, usize>,

    /// Violations grouped by shape
    pub violations_by_shape: HashMap<ShapeId, usize>,

    /// Violations grouped by constraint component
    pub violations_by_component: HashMap<ConstraintComponentId, usize>,

    /// Total number of nodes validated
    pub nodes_validated: usize,

    /// Total number of shapes validated
    pub shapes_validated: usize,

    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,

    /// Most common violation types
    pub top_violation_types: Vec<(ConstraintComponentId, usize)>,

    /// Most problematic shapes
    pub top_problematic_shapes: Vec<(ShapeId, usize)>,
}

impl ValidationSummary {
    /// Create a new empty summary
    pub fn new() -> Self {
        Self {
            violations_by_severity: HashMap::new(),
            violations_by_shape: HashMap::new(),
            violations_by_component: HashMap::new(),
            nodes_validated: 0,
            shapes_validated: 0,
            success_rate: 1.0,
            top_violation_types: Vec::new(),
            top_problematic_shapes: Vec::new(),
        }
    }

    /// Calculate summary from violations
    pub fn from_violations(violations: &[ValidationViolation]) -> Self {
        let mut summary = Self::new();
        summary.calculate_from_violations(violations);
        summary
    }

    /// Update summary with violations data
    pub fn calculate_from_violations(&mut self, violations: &[ValidationViolation]) {
        // Clear existing data
        self.violations_by_severity.clear();
        self.violations_by_shape.clear();
        self.violations_by_component.clear();
        self.top_violation_types.clear();
        self.top_problematic_shapes.clear();

        // Count violations by different categories
        for violation in violations {
            // By severity
            *self
                .violations_by_severity
                .entry(violation.result_severity)
                .or_insert(0) += 1;

            // By shape
            *self
                .violations_by_shape
                .entry(violation.source_shape.clone())
                .or_insert(0) += 1;

            // By constraint component
            if let Some(component) = &violation.source_constraint_component {
                *self
                    .violations_by_component
                    .entry(component.clone())
                    .or_insert(0) += 1;
            }
        }

        // Calculate top violation types
        let mut component_counts: Vec<_> = self
            .violations_by_component
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        component_counts.sort_by(|a, b| b.1.cmp(&a.1));
        self.top_violation_types = component_counts.into_iter().take(5).collect();

        // Calculate top problematic shapes
        let mut shape_counts: Vec<_> = self
            .violations_by_shape
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        shape_counts.sort_by(|a, b| b.1.cmp(&a.1));
        self.top_problematic_shapes = shape_counts.into_iter().take(5).collect();

        // Calculate success rate (simplified - in practice would consider total nodes)
        self.success_rate = if self.nodes_validated > 0 {
            1.0 - (violations.len() as f64 / self.nodes_validated as f64)
        } else if violations.is_empty() {
            1.0
        } else {
            0.0
        };
    }

    /// Set the number of nodes validated
    pub fn with_nodes_validated(mut self, count: usize) -> Self {
        self.nodes_validated = count;
        self.recalculate_success_rate();
        self
    }

    /// Set the number of shapes validated
    pub fn with_shapes_validated(mut self, count: usize) -> Self {
        self.shapes_validated = count;
        self
    }

    /// Get total violation count
    pub fn total_violations(&self) -> usize {
        self.violations_by_severity.values().sum()
    }

    /// Get violation count for a specific severity
    pub fn violations_for_severity(&self, severity: Severity) -> usize {
        self.violations_by_severity
            .get(&severity)
            .copied()
            .unwrap_or(0)
    }

    /// Get error count
    pub fn error_count(&self) -> usize {
        self.violations_for_severity(Severity::Violation)
    }

    /// Get warning count
    pub fn warning_count(&self) -> usize {
        self.violations_for_severity(Severity::Warning)
    }

    /// Get info count
    pub fn info_count(&self) -> usize {
        self.violations_for_severity(Severity::Info)
    }

    /// Check if validation passed (no errors)
    pub fn passed(&self) -> bool {
        self.error_count() == 0
    }

    /// Check if there are any violations
    pub fn has_violations(&self) -> bool {
        self.total_violations() > 0
    }

    /// Get the most problematic shape
    pub fn most_problematic_shape(&self) -> Option<&ShapeId> {
        self.top_problematic_shapes.first().map(|(shape, _)| shape)
    }

    /// Get the most common violation type
    pub fn most_common_violation_type(&self) -> Option<&ConstraintComponentId> {
        self.top_violation_types
            .first()
            .map(|(component, _)| component)
    }

    /// Generate a text summary
    pub fn text_summary(&self) -> String {
        let mut summary = Vec::new();

        summary.push(format!("Total violations: {}", self.total_violations()));

        if self.has_violations() {
            summary.push(format!(
                "Errors: {}, Warnings: {}, Info: {}",
                self.error_count(),
                self.warning_count(),
                self.info_count()
            ));
        }

        summary.push(format!("Success rate: {:.1}%", self.success_rate * 100.0));

        if let Some(shape) = self.most_problematic_shape() {
            let count = self.violations_by_shape.get(shape).unwrap_or(&0);
            summary.push(format!(
                "Most problematic shape: {} ({} violations)",
                shape, count
            ));
        }

        if let Some(component) = self.most_common_violation_type() {
            let count = self.violations_by_component.get(component).unwrap_or(&0);
            summary.push(format!(
                "Most common violation: {} ({} occurrences)",
                component, count
            ));
        }

        summary.join("\n")
    }

    /// Get severity distribution as percentages
    pub fn severity_distribution(&self) -> HashMap<Severity, f64> {
        let total = self.total_violations() as f64;
        let mut distribution = HashMap::new();

        if total > 0.0 {
            for (&severity, &count) in &self.violations_by_severity {
                distribution.insert(severity, (count as f64 / total) * 100.0);
            }
        }

        distribution
    }

    /// Get quality score (0.0 to 1.0, higher is better)
    pub fn quality_score(&self) -> f64 {
        let total_violations = self.total_violations() as f64;
        let errors = self.error_count() as f64;
        let warnings = self.warning_count() as f64;

        if total_violations == 0.0 {
            return 1.0;
        }

        // Weight errors more heavily than warnings
        let weighted_score = 1.0
            - ((errors * 1.0 + warnings * 0.5)
                / (self.nodes_validated as f64).max(total_violations));

        weighted_score.max(0.0).min(1.0)
    }

    /// Check if data quality is acceptable (customizable threshold)
    pub fn is_acceptable_quality(&self, min_score: f64) -> bool {
        self.quality_score() >= min_score
    }

    fn recalculate_success_rate(&mut self) {
        let total_violations = self.total_violations();
        self.success_rate = if self.nodes_validated > 0 {
            1.0 - (total_violations as f64 / self.nodes_validated as f64)
        } else if total_violations == 0 {
            1.0
        } else {
            0.0
        };
    }
}

impl Default for ValidationSummary {
    fn default() -> Self {
        Self::new()
    }
}

/// Detailed analysis of validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationAnalysis {
    /// Basic summary
    pub summary: ValidationSummary,

    /// Data quality insights
    pub quality_insights: Vec<QualityInsight>,

    /// Recommendations for improvement
    pub recommendations: Vec<String>,

    /// Trends analysis (if historical data available)
    pub trends: Option<TrendAnalysis>,
}

impl ValidationAnalysis {
    /// Create analysis from violations
    pub fn analyze(violations: &[ValidationViolation]) -> Self {
        let summary = ValidationSummary::from_violations(violations);
        let quality_insights = Self::generate_quality_insights(&summary, violations);
        let recommendations = Self::generate_recommendations(&summary, violations);

        Self {
            summary,
            quality_insights,
            recommendations,
            trends: None,
        }
    }

    fn generate_quality_insights(
        summary: &ValidationSummary,
        violations: &[ValidationViolation],
    ) -> Vec<QualityInsight> {
        let mut insights = Vec::new();

        // High error rate insight
        if summary.error_count() > summary.total_violations() / 2 {
            insights.push(QualityInsight {
                insight_type: InsightType::HighErrorRate,
                description: "High proportion of errors detected".to_string(),
                severity: Severity::Violation,
                affected_count: summary.error_count(),
                recommendation: "Review data quality processes and shape definitions".to_string(),
            });
        }

        // Concentrated violations insight
        if let Some((shape, count)) = summary.top_problematic_shapes.first() {
            if *count > summary.total_violations() / 3 {
                insights.push(QualityInsight {
                    insight_type: InsightType::ConcentratedViolations,
                    description: format!("Shape {} has many violations", shape),
                    severity: Severity::Warning,
                    affected_count: *count,
                    recommendation: "Review and possibly refine this shape definition".to_string(),
                });
            }
        }

        // Common constraint pattern
        if let Some((component, count)) = summary.top_violation_types.first() {
            if *count > summary.total_violations() / 2 {
                insights.push(QualityInsight {
                    insight_type: InsightType::CommonPattern,
                    description: format!("Constraint {} frequently violated", component),
                    severity: Severity::Info,
                    affected_count: *count,
                    recommendation: "Consider if this constraint is too strict".to_string(),
                });
            }
        }

        insights
    }

    fn generate_recommendations(
        summary: &ValidationSummary,
        _violations: &[ValidationViolation],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if summary.error_count() > 0 {
            recommendations
                .push("Address all error-level violations before deployment".to_string());
        }

        if summary.quality_score() < 0.8 {
            recommendations.push("Consider improving data quality processes".to_string());
        }

        if summary.top_problematic_shapes.len() > 1 {
            recommendations
                .push("Review the most problematic shapes for possible refinement".to_string());
        }

        if recommendations.is_empty() {
            recommendations
                .push("Data quality looks good! Consider periodic re-validation".to_string());
        }

        recommendations
    }
}

/// Quality insight about validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityInsight {
    pub insight_type: InsightType,
    pub description: String,
    pub severity: Severity,
    pub affected_count: usize,
    pub recommendation: String,
}

/// Types of quality insights
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InsightType {
    HighErrorRate,
    ConcentratedViolations,
    CommonPattern,
    DataQualityTrend,
    PerformanceIssue,
}

/// Trend analysis for historical validation data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub improving: bool,
    pub trend_direction: TrendDirection,
    pub change_percentage: f64,
}

/// Direction of trends
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
}
