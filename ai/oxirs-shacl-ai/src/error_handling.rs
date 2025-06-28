//! Intelligent Error Handling System for SHACL-AI
//!
//! This module provides AI-powered error classification, impact assessment,
//! and automated repair suggestions for SHACL validation errors.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use oxirs_core::{model::Term, Store};
use oxirs_shacl::{Severity, Shape, ShapeId, ValidationReport, ValidationViolation as Violation};

use crate::{
    analytics::AnalyticsEngine,
    ml::{ModelParams, ShapeLearningModel},
    patterns::{Pattern, PatternAnalyzer},
    quality::QualityAssessor,
    Result, ShaclAiError,
};

/// Intelligent error handling system for SHACL validation
#[derive(Debug)]
pub struct IntelligentErrorHandler {
    /// Error classifier for categorizing validation errors
    error_classifier: ErrorClassifier,

    /// Repair suggestion engine
    repair_engine: RepairSuggestionEngine,

    /// Error impact assessor
    impact_assessor: ErrorImpactAssessor,

    /// Prevention strategy generator
    prevention_generator: PreventionStrategyGenerator,

    /// Configuration
    config: ErrorHandlingConfig,
}

/// Configuration for error handling system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorHandlingConfig {
    /// Enable machine learning-based error classification
    pub enable_ml_classification: bool,

    /// Minimum confidence threshold for repair suggestions
    pub min_repair_confidence: f64,

    /// Maximum number of repair suggestions per error
    pub max_repair_suggestions: usize,

    /// Enable automated impact assessment
    pub enable_impact_assessment: bool,

    /// Enable prevention strategy generation
    pub enable_prevention_strategies: bool,

    /// Severity threshold for critical errors
    pub critical_severity_threshold: f64,

    /// Business impact weight in priority calculation
    pub business_impact_weight: f64,
}

impl Default for ErrorHandlingConfig {
    fn default() -> Self {
        Self {
            enable_ml_classification: true,
            min_repair_confidence: 0.7,
            max_repair_suggestions: 5,
            enable_impact_assessment: true,
            enable_prevention_strategies: true,
            critical_severity_threshold: 0.8,
            business_impact_weight: 0.3,
        }
    }
}

/// Comprehensive error classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorClassificationResult {
    /// Primary error type from taxonomy
    pub error_type: ErrorType,

    /// Detailed error subtype
    pub error_subtype: String,

    /// Severity classification
    pub severity: ErrorSeverity,

    /// Impact assessment
    pub impact: ErrorImpact,

    /// Priority assignment
    pub priority: ErrorPriority,

    /// Resolution difficulty estimate
    pub resolution_difficulty: ResolutionDifficulty,

    /// Business criticality assessment
    pub business_criticality: BusinessCriticality,

    /// Classification confidence
    pub confidence: f64,

    /// Supporting evidence
    pub evidence: HashMap<String, f64>,
}

/// Error type taxonomy
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorType {
    /// Constraint violation errors
    ConstraintViolation,

    /// Schema inconsistency errors
    SchemaInconsistency,

    /// Data quality errors
    DataQuality,

    /// Performance-related errors
    Performance,

    /// Semantic errors
    Semantic,

    /// Structural errors
    Structural,

    /// Temporal errors
    Temporal,

    /// Cross-reference errors
    CrossReference,
}

impl std::fmt::Display for ErrorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorType::ConstraintViolation => write!(f, "ConstraintViolation"),
            ErrorType::SchemaInconsistency => write!(f, "SchemaInconsistency"),
            ErrorType::DataQuality => write!(f, "DataQuality"),
            ErrorType::Performance => write!(f, "Performance"),
            ErrorType::Semantic => write!(f, "Semantic"),
            ErrorType::Structural => write!(f, "Structural"),
            ErrorType::Temporal => write!(f, "Temporal"),
            ErrorType::CrossReference => write!(f, "CrossReference"),
        }
    }
}

/// Error severity levels with numeric values
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
    Blocker = 5,
}

/// Error impact assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorImpact {
    /// Data integrity impact (0.0 to 1.0)
    pub data_integrity: f64,

    /// System performance impact
    pub performance: f64,

    /// Business process impact
    pub business_process: f64,

    /// User experience impact
    pub user_experience: f64,

    /// Security impact
    pub security: f64,

    /// Compliance impact
    pub compliance: f64,
}

/// Error priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ErrorPriority {
    Low = 1,
    Medium = 2,
    High = 3,
    Urgent = 4,
    Critical = 5,
}

/// Resolution difficulty levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolutionDifficulty {
    Trivial,
    Easy,
    Moderate,
    Hard,
    Complex,
}

/// Business criticality levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BusinessCriticality {
    None,
    Low,
    Medium,
    High,
    Mission,
}

/// Automated repair suggestion with detailed information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairSuggestion {
    /// Type of repair suggestion
    pub suggestion_type: RepairType,

    /// Human-readable description
    pub description: String,

    /// Detailed repair instructions
    pub instructions: Vec<String>,

    /// Confidence in the suggestion
    pub confidence: f64,

    /// Expected effectiveness
    pub effectiveness: f64,

    /// Implementation effort estimate
    pub effort: ImplementationEffort,

    /// Risk assessment
    pub risk: RiskAssessment,

    /// Alternative approaches
    pub alternatives: Vec<AlternativeApproach>,

    /// Dependencies and prerequisites
    pub prerequisites: Vec<String>,
}

/// Types of repair suggestions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RepairType {
    /// Constraint relaxation
    ConstraintRelaxation,

    /// Data correction
    DataCorrection,

    /// Schema modification
    SchemaModification,

    /// Process improvement
    ProcessImprovement,

    /// Configuration change
    ConfigurationChange,

    /// Tool recommendation
    ToolRecommendation,
}

/// Implementation effort estimation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Minimal,
    Low,
    Medium,
    High,
    Extensive,
}

/// Risk assessment for repair suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// Overall risk level
    pub risk_level: RiskLevel,

    /// Specific risks identified
    pub risks: Vec<String>,

    /// Mitigation strategies
    pub mitigations: Vec<String>,

    /// Rollback feasibility
    pub rollback_feasibility: f64,
}

/// Risk levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Alternative repair approach
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativeApproach {
    /// Description of alternative
    pub description: String,

    /// Relative effectiveness compared to primary suggestion
    pub relative_effectiveness: f64,

    /// Relative effort compared to primary suggestion
    pub relative_effort: f64,

    /// Trade-offs and considerations
    pub tradeoffs: Vec<String>,
}

/// Prevention strategy for avoiding similar errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreventionStrategy {
    /// Strategy type
    pub strategy_type: PreventionType,

    /// Description
    pub description: String,

    /// Implementation steps
    pub implementation_steps: Vec<String>,

    /// Expected effectiveness
    pub effectiveness: f64,

    /// Cost-benefit analysis
    pub cost_benefit: CostBenefit,

    /// Monitoring recommendations
    pub monitoring: Vec<String>,
}

/// Types of prevention strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreventionType {
    ProcessImprovement,
    ToolIntegration,
    TrainingEnhancement,
    ValidationRules,
    MonitoringEnhancement,
    DocumentationUpdate,
}

/// Cost-benefit analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBenefit {
    /// Implementation cost estimate
    pub implementation_cost: f64,

    /// Ongoing maintenance cost
    pub maintenance_cost: f64,

    /// Expected savings per year
    pub annual_savings: f64,

    /// ROI estimate
    pub roi_estimate: f64,

    /// Payback period in months
    pub payback_period: u32,
}

impl IntelligentErrorHandler {
    /// Create a new intelligent error handler
    pub fn new() -> Self {
        Self::with_config(ErrorHandlingConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: ErrorHandlingConfig) -> Self {
        Self {
            error_classifier: ErrorClassifier::new(config.clone()),
            repair_engine: RepairSuggestionEngine::new(config.clone()),
            impact_assessor: ErrorImpactAssessor::new(config.clone()),
            prevention_generator: PreventionStrategyGenerator::new(config.clone()),
            config,
        }
    }

    /// Process validation errors with comprehensive analysis
    pub fn process_validation_errors(
        &mut self,
        validation_report: &ValidationReport,
        store: &Store,
        shapes: &[Shape],
    ) -> Result<SmartErrorAnalysis> {
        tracing::info!(
            "Processing {} validation errors with intelligent analysis",
            validation_report.violations.len()
        );

        let mut error_analyses = Vec::new();

        for violation in &validation_report.violations {
            let error_analysis = self.analyze_single_error(violation, store, shapes)?;
            error_analyses.push(error_analysis);
        }

        // Generate comprehensive analysis
        let comprehensive_analysis =
            self.generate_comprehensive_analysis(&error_analyses, store, shapes)?;

        let processing_metadata = ProcessingMetadata {
            processing_time: Duration::from_millis(100), // Placeholder
            errors_processed: validation_report.violations.len(),
            ml_models_used: vec!["ErrorClassifier".to_string(), "RepairEngine".to_string()],
            confidence_distribution: self.calculate_confidence_distribution(&error_analyses),
        };

        Ok(SmartErrorAnalysis {
            individual_analyses: error_analyses,
            comprehensive_analysis,
            processing_metadata,
        })
    }

    /// Analyze a single validation error
    fn analyze_single_error(
        &mut self,
        violation: &Violation,
        store: &Store,
        shapes: &[Shape],
    ) -> Result<SingleErrorAnalysis> {
        // Classify the error
        let classification = self
            .error_classifier
            .classify_error(violation, store, shapes)?;

        // Generate repair suggestions
        let repair_suggestions = if self.config.enable_ml_classification {
            self.repair_engine.generate_repair_suggestions(
                &classification,
                violation,
                store,
                shapes,
            )?
        } else {
            Vec::new()
        };

        // Assess impact
        let impact_assessment = if self.config.enable_impact_assessment {
            self.impact_assessor
                .assess_impact(&classification, violation, store)?
        } else {
            DetailedImpactAssessment::default()
        };

        // Generate prevention strategies
        let prevention_strategies = if self.config.enable_prevention_strategies {
            self.prevention_generator
                .generate_strategies(&classification, violation)?
        } else {
            Vec::new()
        };

        Ok(SingleErrorAnalysis {
            violation: violation.clone(),
            classification,
            repair_suggestions,
            impact_assessment,
            prevention_strategies,
            analysis_metadata: AnalysisMetadata {
                analysis_timestamp: chrono::Utc::now(),
                processing_duration: Duration::from_millis(50),
                models_used: vec!["ErrorClassifier".to_string()],
                confidence_score: 0.85,
            },
        })
    }

    /// Generate comprehensive analysis across all errors
    fn generate_comprehensive_analysis(
        &self,
        error_analyses: &[SingleErrorAnalysis],
        store: &Store,
        shapes: &[Shape],
    ) -> Result<ComprehensiveErrorAnalysis> {
        // Identify error patterns
        let error_patterns = self.identify_error_patterns(error_analyses)?;

        // Generate systemic recommendations
        let systemic_recommendations =
            self.generate_systemic_recommendations(error_analyses, &error_patterns)?;

        // Assess overall quality impact
        let quality_impact = self.assess_overall_quality_impact(error_analyses)?;

        // Generate priority ranking
        let priority_ranking = self.generate_priority_ranking(error_analyses)?;

        Ok(ComprehensiveErrorAnalysis {
            error_patterns,
            systemic_recommendations,
            quality_impact,
            priority_ranking,
            aggregated_metrics: self.calculate_aggregated_metrics(error_analyses)?,
            trend_analysis: self.analyze_error_trends(error_analyses)?,
            root_cause_analysis: self.perform_root_cause_analysis(error_analyses)?,
        })
    }

    /// Identify patterns across multiple errors
    fn identify_error_patterns(
        &self,
        analyses: &[SingleErrorAnalysis],
    ) -> Result<Vec<ErrorPattern>> {
        let mut patterns = Vec::new();

        // Group errors by type
        let mut type_groups: HashMap<ErrorType, Vec<&SingleErrorAnalysis>> = HashMap::new();
        for analysis in analyses {
            type_groups
                .entry(analysis.classification.error_type.clone())
                .or_insert_with(Vec::new)
                .push(analysis);
        }

        // Identify patterns within each type
        for (error_type, group) in type_groups {
            if group.len() >= 2 {
                let pattern = ErrorPattern {
                    pattern_type: error_type,
                    frequency: group.len(),
                    confidence: 0.8,
                    description: format!("Recurring {} errors detected", group.len()),
                    affected_shapes: group
                        .iter()
                        .map(|a| a.violation.source_shape.clone())
                        .collect(),
                    common_characteristics: self.extract_common_characteristics(&group)?,
                };
                patterns.push(pattern);
            }
        }

        Ok(patterns)
    }

    /// Extract common characteristics from error group
    fn extract_common_characteristics(
        &self,
        group: &[&SingleErrorAnalysis],
    ) -> Result<Vec<String>> {
        let mut characteristics = Vec::new();

        // Common severity
        if let Some(first_severity) = group.first().map(|a| &a.classification.severity) {
            if group
                .iter()
                .all(|a| a.classification.severity == *first_severity)
            {
                characteristics.push(format!("All errors have {:?} severity", first_severity));
            }
        }

        // Common shape patterns
        let shape_ids: Vec<_> = group.iter().map(|a| &a.violation.source_shape).collect();
        if shape_ids.iter().all(|id| *id == shape_ids[0]) {
            characteristics.push("All errors from same shape".to_string());
        }

        Ok(characteristics)
    }

    /// Generate systemic recommendations
    fn generate_systemic_recommendations(
        &self,
        analyses: &[SingleErrorAnalysis],
        patterns: &[ErrorPattern],
    ) -> Result<Vec<SystemicRecommendation>> {
        let mut recommendations = Vec::new();

        // Recommendations based on patterns
        for pattern in patterns {
            if pattern.frequency > 5 {
                recommendations.push(SystemicRecommendation {
                    recommendation_type: SystemicRecommendationType::ProcessImprovement,
                    title: format!("Address recurring {} pattern", pattern.pattern_type),
                    description: format!(
                        "Implement systematic solution for {} recurring errors",
                        pattern.frequency
                    ),
                    implementation_priority: SystemicPriority::High,
                    expected_impact: 0.8,
                    implementation_steps: vec![
                        "Analyze root cause of pattern".to_string(),
                        "Design preventive measures".to_string(),
                        "Implement monitoring".to_string(),
                    ],
                    success_metrics: vec!["Reduction in similar errors".to_string()],
                });
            }
        }

        // High-level process recommendations
        let critical_count = analyses
            .iter()
            .filter(|a| a.classification.severity >= ErrorSeverity::Critical)
            .count();

        if critical_count > 0 {
            recommendations.push(SystemicRecommendation {
                recommendation_type: SystemicRecommendationType::QualityGate,
                title: "Implement critical error prevention".to_string(),
                description: format!(
                    "Address {} critical errors with quality gates",
                    critical_count
                ),
                implementation_priority: SystemicPriority::Urgent,
                expected_impact: 0.9,
                implementation_steps: vec![
                    "Review validation pipeline".to_string(),
                    "Add pre-validation checks".to_string(),
                    "Implement alerts".to_string(),
                ],
                success_metrics: vec!["Zero critical errors".to_string()],
            });
        }

        Ok(recommendations)
    }

    /// Calculate confidence distribution
    fn calculate_confidence_distribution(
        &self,
        analyses: &[SingleErrorAnalysis],
    ) -> HashMap<String, f64> {
        let mut distribution = HashMap::new();

        if analyses.is_empty() {
            return distribution;
        }

        let confidences: Vec<f64> = analyses
            .iter()
            .map(|a| a.classification.confidence)
            .collect();

        distribution.insert(
            "mean".to_string(),
            confidences.iter().sum::<f64>() / confidences.len() as f64,
        );
        distribution.insert(
            "min".to_string(),
            confidences.iter().cloned().fold(f64::INFINITY, f64::min),
        );
        distribution.insert(
            "max".to_string(),
            confidences
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max),
        );

        // Calculate quartiles (simplified)
        let mut sorted_confidences = confidences.clone();
        sorted_confidences.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if sorted_confidences.len() >= 4 {
            distribution.insert(
                "q1".to_string(),
                sorted_confidences[sorted_confidences.len() / 4],
            );
            distribution.insert(
                "median".to_string(),
                sorted_confidences[sorted_confidences.len() / 2],
            );
            distribution.insert(
                "q3".to_string(),
                sorted_confidences[3 * sorted_confidences.len() / 4],
            );
        }

        distribution
    }

    /// Assess overall quality impact
    fn assess_overall_quality_impact(
        &self,
        analyses: &[SingleErrorAnalysis],
    ) -> Result<OverallQualityImpact> {
        let total_errors = analyses.len() as f64;
        if total_errors == 0.0 {
            return Ok(OverallQualityImpact::default());
        }

        // Calculate severity distribution
        let critical_count = analyses
            .iter()
            .filter(|a| a.classification.severity >= ErrorSeverity::Critical)
            .count() as f64;
        let high_count = analyses
            .iter()
            .filter(|a| a.classification.severity == ErrorSeverity::High)
            .count() as f64;

        // Calculate overall impact scores
        let data_integrity_impact = analyses
            .iter()
            .map(|a| a.impact_assessment.data_integrity_impact)
            .sum::<f64>()
            / total_errors;

        let performance_impact = analyses
            .iter()
            .map(|a| a.impact_assessment.performance_impact)
            .sum::<f64>()
            / total_errors;

        let business_impact = analyses
            .iter()
            .map(|a| a.impact_assessment.business_impact)
            .sum::<f64>()
            / total_errors;

        Ok(OverallQualityImpact {
            severity_distribution: SeverityDistribution {
                critical_percentage: (critical_count / total_errors) * 100.0,
                high_percentage: (high_count / total_errors) * 100.0,
                total_errors: total_errors as usize,
            },
            cumulative_impact: CumulativeImpact {
                data_integrity: data_integrity_impact,
                performance: performance_impact,
                business: business_impact,
                overall_score: (data_integrity_impact + performance_impact + business_impact) / 3.0,
            },
            quality_degradation: QualityDegradation {
                estimated_degradation: (critical_count * 0.3 + high_count * 0.1) / total_errors,
                confidence: 0.8,
                trend_indicator: if critical_count > 0.0 {
                    "Declining"
                } else {
                    "Stable"
                }
                .to_string(),
            },
        })
    }

    /// Generate priority ranking for errors
    fn generate_priority_ranking(
        &self,
        analyses: &[SingleErrorAnalysis],
    ) -> Result<Vec<PriorityRankedError>> {
        let mut ranked_errors = Vec::new();

        for (index, analysis) in analyses.iter().enumerate() {
            let priority_score = self.calculate_priority_score(analysis);

            ranked_errors.push(PriorityRankedError {
                error_index: index,
                priority_score,
                ranking_factors: PriorityFactors {
                    severity_weight: self.severity_to_weight(&analysis.classification.severity),
                    business_impact_weight: analysis.impact_assessment.business_impact,
                    resolution_difficulty_weight: self
                        .difficulty_to_weight(&analysis.classification.resolution_difficulty),
                    urgency_weight: self.calculate_urgency_weight(analysis),
                },
                recommended_timeline: self.calculate_recommended_timeline(analysis),
            });
        }

        // Sort by priority score (descending)
        ranked_errors.sort_by(|a, b| b.priority_score.partial_cmp(&a.priority_score).unwrap());

        Ok(ranked_errors)
    }

    /// Calculate priority score for an error
    fn calculate_priority_score(&self, analysis: &SingleErrorAnalysis) -> f64 {
        let severity_weight = self.severity_to_weight(&analysis.classification.severity);
        let business_impact = analysis.impact_assessment.business_impact;
        let difficulty_penalty =
            self.difficulty_to_weight(&analysis.classification.resolution_difficulty);

        // Weighted priority calculation
        (severity_weight * 0.4
            + business_impact * self.config.business_impact_weight
            + (1.0 - difficulty_penalty) * 0.3)
            .min(1.0)
    }

    /// Convert severity to numeric weight
    fn severity_to_weight(&self, severity: &ErrorSeverity) -> f64 {
        match severity {
            ErrorSeverity::Low => 0.2,
            ErrorSeverity::Medium => 0.4,
            ErrorSeverity::High => 0.6,
            ErrorSeverity::Critical => 0.8,
            ErrorSeverity::Blocker => 1.0,
        }
    }

    /// Convert difficulty to numeric weight (inverse for priority)
    fn difficulty_to_weight(&self, difficulty: &ResolutionDifficulty) -> f64 {
        match difficulty {
            ResolutionDifficulty::Trivial => 0.1,
            ResolutionDifficulty::Easy => 0.2,
            ResolutionDifficulty::Moderate => 0.4,
            ResolutionDifficulty::Hard => 0.6,
            ResolutionDifficulty::Complex => 0.8,
        }
    }

    /// Calculate urgency weight based on various factors
    fn calculate_urgency_weight(&self, analysis: &SingleErrorAnalysis) -> f64 {
        let mut urgency: f64 = 0.5; // Base urgency

        // Increase urgency for critical business impact
        if matches!(
            analysis.classification.business_criticality,
            BusinessCriticality::Mission | BusinessCriticality::High
        ) {
            urgency += 0.3;
        }

        // Increase urgency for security impact
        if analysis.impact_assessment.security_impact > 0.5 {
            urgency += 0.2;
        }

        urgency.min(1.0)
    }

    /// Calculate recommended timeline for error resolution
    fn calculate_recommended_timeline(&self, analysis: &SingleErrorAnalysis) -> Duration {
        match analysis.classification.severity {
            ErrorSeverity::Blocker => Duration::from_secs(4 * 3600), // 4 hours
            ErrorSeverity::Critical => Duration::from_secs(24 * 3600), // 24 hours
            ErrorSeverity::High => Duration::from_secs(3 * 24 * 3600), // 3 days
            ErrorSeverity::Medium => Duration::from_secs(7 * 24 * 3600), // 7 days
            ErrorSeverity::Low => Duration::from_secs(14 * 24 * 3600), // 14 days
        }
    }

    /// Calculate aggregated metrics across all errors
    fn calculate_aggregated_metrics(
        &self,
        analyses: &[SingleErrorAnalysis],
    ) -> Result<AggregatedMetrics> {
        if analyses.is_empty() {
            return Ok(AggregatedMetrics::default());
        }

        let total_count = analyses.len() as f64;

        // Calculate type distribution
        let mut type_distribution = HashMap::new();
        for analysis in analyses {
            *type_distribution
                .entry(analysis.classification.error_type.clone())
                .or_insert(0) += 1;
        }

        // Calculate average metrics
        let avg_confidence = analyses
            .iter()
            .map(|a| a.classification.confidence)
            .sum::<f64>()
            / total_count;
        let avg_impact = analyses
            .iter()
            .map(|a| a.impact_assessment.business_impact)
            .sum::<f64>()
            / total_count;

        Ok(AggregatedMetrics {
            total_errors: analyses.len(),
            type_distribution,
            average_confidence: avg_confidence,
            average_impact: avg_impact,
            resolution_complexity: self.calculate_overall_complexity(analyses),
        })
    }

    /// Calculate overall resolution complexity
    fn calculate_overall_complexity(&self, analyses: &[SingleErrorAnalysis]) -> f64 {
        if analyses.is_empty() {
            return 0.0;
        }

        analyses
            .iter()
            .map(|a| self.difficulty_to_weight(&a.classification.resolution_difficulty))
            .sum::<f64>()
            / analyses.len() as f64
    }

    /// Analyze error trends (placeholder implementation)
    fn analyze_error_trends(&self, analyses: &[SingleErrorAnalysis]) -> Result<TrendAnalysis> {
        Ok(TrendAnalysis {
            trend_direction: "stable".to_string(),
            trend_confidence: 0.7,
            seasonal_patterns: Vec::new(),
            forecast: Vec::new(),
        })
    }

    /// Perform root cause analysis (placeholder implementation)
    fn perform_root_cause_analysis(
        &self,
        analyses: &[SingleErrorAnalysis],
    ) -> Result<RootCauseAnalysis> {
        Ok(RootCauseAnalysis {
            primary_causes: vec!["Schema inconsistency".to_string()],
            contributing_factors: vec!["Data quality issues".to_string()],
            systemic_issues: vec!["Validation process gaps".to_string()],
            confidence: 0.75,
        })
    }
}

/// Error classifier for categorizing validation errors
#[derive(Debug)]
pub struct ErrorClassifier {
    config: ErrorHandlingConfig,
}

impl ErrorClassifier {
    pub fn new(config: ErrorHandlingConfig) -> Self {
        Self { config }
    }

    /// Classify a validation error
    pub fn classify_error(
        &self,
        violation: &Violation,
        store: &Store,
        shapes: &[Shape],
    ) -> Result<ErrorClassificationResult> {
        // Simplified classification logic - would use ML models in production
        let error_type = self.determine_error_type(violation)?;
        let severity = self.assess_severity(violation, &error_type)?;
        let impact = self.assess_impact(violation, &error_type)?;
        let priority = self.calculate_priority(&severity, &impact)?;
        let resolution_difficulty = self.assess_resolution_difficulty(&error_type, violation)?;
        let business_criticality = self.assess_business_criticality(&impact)?;

        Ok(ErrorClassificationResult {
            error_type,
            error_subtype: "general".to_string(),
            severity,
            impact,
            priority,
            resolution_difficulty,
            business_criticality,
            confidence: 0.8, // Simplified confidence
            evidence: HashMap::new(),
        })
    }

    fn determine_error_type(&self, violation: &Violation) -> Result<ErrorType> {
        // Analyze violation message and constraint to determine type
        match violation.source_constraint_component.0.as_str() {
            "sh:minCount" | "sh:maxCount" => Ok(ErrorType::ConstraintViolation),
            "sh:datatype" => Ok(ErrorType::DataQuality),
            "sh:class" => Ok(ErrorType::Structural),
            _ => Ok(ErrorType::ConstraintViolation), // Default
        }
    }

    fn assess_severity(
        &self,
        violation: &Violation,
        error_type: &ErrorType,
    ) -> Result<ErrorSeverity> {
        // Assess severity based on violation details and type
        match (violation.result_severity.clone(), error_type) {
            (Severity::Violation, ErrorType::DataQuality) => Ok(ErrorSeverity::High),
            (Severity::Warning, _) => Ok(ErrorSeverity::Medium),
            (Severity::Info, _) => Ok(ErrorSeverity::Low),
            _ => Ok(ErrorSeverity::Medium),
        }
    }

    fn assess_impact(&self, violation: &Violation, error_type: &ErrorType) -> Result<ErrorImpact> {
        // Simplified impact assessment
        let base_impact = match error_type {
            ErrorType::ConstraintViolation => 0.6,
            ErrorType::DataQuality => 0.7,
            ErrorType::SchemaInconsistency => 0.8,
            ErrorType::Performance => 0.5,
            ErrorType::Semantic => 0.6,
            ErrorType::Structural => 0.7,
            ErrorType::Temporal => 0.4,
            ErrorType::CrossReference => 0.5,
        };

        Ok(ErrorImpact {
            data_integrity: base_impact,
            performance: base_impact * 0.8,
            business_process: base_impact * 0.7,
            user_experience: base_impact * 0.6,
            security: base_impact * 0.3,
            compliance: base_impact * 0.5,
        })
    }

    fn calculate_priority(
        &self,
        severity: &ErrorSeverity,
        impact: &ErrorImpact,
    ) -> Result<ErrorPriority> {
        let severity_score = match severity {
            ErrorSeverity::Low => 1,
            ErrorSeverity::Medium => 2,
            ErrorSeverity::High => 3,
            ErrorSeverity::Critical => 4,
            ErrorSeverity::Blocker => 5,
        };

        let impact_score = (impact.business_process * 5.0) as u8;
        let combined_score = (severity_score + impact_score) / 2;

        match combined_score {
            1 => Ok(ErrorPriority::Low),
            2 => Ok(ErrorPriority::Medium),
            3 => Ok(ErrorPriority::High),
            4 => Ok(ErrorPriority::Urgent),
            _ => Ok(ErrorPriority::Critical),
        }
    }

    fn assess_resolution_difficulty(
        &self,
        error_type: &ErrorType,
        violation: &Violation,
    ) -> Result<ResolutionDifficulty> {
        match error_type {
            ErrorType::ConstraintViolation => Ok(ResolutionDifficulty::Easy),
            ErrorType::DataQuality => Ok(ResolutionDifficulty::Moderate),
            ErrorType::SchemaInconsistency => Ok(ResolutionDifficulty::Hard),
            ErrorType::Performance => Ok(ResolutionDifficulty::Complex),
            _ => Ok(ResolutionDifficulty::Moderate),
        }
    }

    fn assess_business_criticality(&self, impact: &ErrorImpact) -> Result<BusinessCriticality> {
        let business_score = impact.business_process;

        if business_score >= 0.9 {
            Ok(BusinessCriticality::Mission)
        } else if business_score >= 0.7 {
            Ok(BusinessCriticality::High)
        } else if business_score >= 0.5 {
            Ok(BusinessCriticality::Medium)
        } else if business_score >= 0.3 {
            Ok(BusinessCriticality::Low)
        } else {
            Ok(BusinessCriticality::None)
        }
    }
}

/// Repair suggestion engine for generating automated fix recommendations
#[derive(Debug)]
pub struct RepairSuggestionEngine {
    config: ErrorHandlingConfig,
}

impl RepairSuggestionEngine {
    pub fn new(config: ErrorHandlingConfig) -> Self {
        Self { config }
    }

    /// Generate repair suggestions for a classified error
    pub fn generate_repair_suggestions(
        &self,
        classification: &ErrorClassificationResult,
        violation: &Violation,
        store: &Store,
        shapes: &[Shape],
    ) -> Result<Vec<RepairSuggestion>> {
        let mut suggestions = Vec::new();

        // Generate suggestions based on error type
        match classification.error_type {
            ErrorType::ConstraintViolation => {
                suggestions.extend(self.generate_constraint_repair_suggestions(violation)?);
            }
            ErrorType::DataQuality => {
                suggestions.extend(self.generate_data_quality_repair_suggestions(violation)?);
            }
            ErrorType::SchemaInconsistency => {
                suggestions.extend(self.generate_schema_repair_suggestions(violation, shapes)?);
            }
            _ => {
                suggestions.extend(self.generate_generic_repair_suggestions(violation)?);
            }
        }

        // Filter by confidence threshold
        suggestions.retain(|s| s.confidence >= self.config.min_repair_confidence);

        // Limit number of suggestions
        suggestions.truncate(self.config.max_repair_suggestions);

        Ok(suggestions)
    }

    fn generate_constraint_repair_suggestions(
        &self,
        violation: &Violation,
    ) -> Result<Vec<RepairSuggestion>> {
        let mut suggestions = Vec::new();

        // Constraint relaxation suggestion
        suggestions.push(RepairSuggestion {
            suggestion_type: RepairType::ConstraintRelaxation,
            description: "Relax constraint to allow current data patterns".to_string(),
            instructions: vec![
                "Review constraint necessity".to_string(),
                "Adjust constraint parameters".to_string(),
                "Validate against business requirements".to_string(),
            ],
            confidence: 0.75,
            effectiveness: 0.8,
            effort: ImplementationEffort::Low,
            risk: RiskAssessment {
                risk_level: RiskLevel::Medium,
                risks: vec!["May allow invalid data".to_string()],
                mitigations: vec!["Add data validation rules".to_string()],
                rollback_feasibility: 0.9,
            },
            alternatives: vec![AlternativeApproach {
                description: "Add data transformation before validation".to_string(),
                relative_effectiveness: 0.9,
                relative_effort: 1.5,
                tradeoffs: vec!["More complex but preserves strict validation".to_string()],
            }],
            prerequisites: vec!["Business stakeholder approval".to_string()],
        });

        Ok(suggestions)
    }

    fn generate_data_quality_repair_suggestions(
        &self,
        violation: &Violation,
    ) -> Result<Vec<RepairSuggestion>> {
        let mut suggestions = Vec::new();

        // Data correction suggestion
        suggestions.push(RepairSuggestion {
            suggestion_type: RepairType::DataCorrection,
            description: "Correct data to meet quality requirements".to_string(),
            instructions: vec![
                "Identify data source issues".to_string(),
                "Apply data cleansing rules".to_string(),
                "Validate corrected data".to_string(),
            ],
            confidence: 0.8,
            effectiveness: 0.9,
            effort: ImplementationEffort::Medium,
            risk: RiskAssessment {
                risk_level: RiskLevel::Low,
                risks: vec!["Potential data loss during correction".to_string()],
                mitigations: vec!["Backup data before changes".to_string()],
                rollback_feasibility: 0.95,
            },
            alternatives: vec![],
            prerequisites: vec!["Data backup completed".to_string()],
        });

        Ok(suggestions)
    }

    fn generate_schema_repair_suggestions(
        &self,
        violation: &Violation,
        shapes: &[Shape],
    ) -> Result<Vec<RepairSuggestion>> {
        let mut suggestions = Vec::new();

        // Schema modification suggestion
        suggestions.push(RepairSuggestion {
            suggestion_type: RepairType::SchemaModification,
            description: "Modify schema to resolve inconsistency".to_string(),
            instructions: vec![
                "Analyze schema dependencies".to_string(),
                "Design schema changes".to_string(),
                "Test with sample data".to_string(),
                "Deploy incrementally".to_string(),
            ],
            confidence: 0.7,
            effectiveness: 0.85,
            effort: ImplementationEffort::High,
            risk: RiskAssessment {
                risk_level: RiskLevel::High,
                risks: vec!["Breaking changes to existing applications".to_string()],
                mitigations: vec!["Phased deployment", "Backward compatibility layer"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                rollback_feasibility: 0.6,
            },
            alternatives: vec![AlternativeApproach {
                description: "Create new schema version".to_string(),
                relative_effectiveness: 0.9,
                relative_effort: 1.3,
                tradeoffs: vec![
                    "Maintains backward compatibility but increases complexity".to_string()
                ],
            }],
            prerequisites: vec![
                "Schema governance approval".to_string(),
                "Impact assessment completed".to_string(),
            ],
        });

        Ok(suggestions)
    }

    fn generate_generic_repair_suggestions(
        &self,
        violation: &Violation,
    ) -> Result<Vec<RepairSuggestion>> {
        let mut suggestions = Vec::new();

        // Configuration change suggestion
        suggestions.push(RepairSuggestion {
            suggestion_type: RepairType::ConfigurationChange,
            description: "Adjust validation configuration".to_string(),
            instructions: vec![
                "Review validation settings".to_string(),
                "Adjust tolerance levels".to_string(),
                "Test configuration changes".to_string(),
            ],
            confidence: 0.6,
            effectiveness: 0.7,
            effort: ImplementationEffort::Low,
            risk: RiskAssessment {
                risk_level: RiskLevel::Low,
                risks: vec!["May mask underlying issues".to_string()],
                mitigations: vec!["Monitor error patterns after changes".to_string()],
                rollback_feasibility: 0.95,
            },
            alternatives: vec![],
            prerequisites: vec![],
        });

        Ok(suggestions)
    }
}

/// Error impact assessor for detailed impact analysis
#[derive(Debug)]
pub struct ErrorImpactAssessor {
    config: ErrorHandlingConfig,
}

impl ErrorImpactAssessor {
    pub fn new(config: ErrorHandlingConfig) -> Self {
        Self { config }
    }

    /// Assess detailed impact of an error
    pub fn assess_impact(
        &self,
        classification: &ErrorClassificationResult,
        violation: &Violation,
        store: &Store,
    ) -> Result<DetailedImpactAssessment> {
        let data_integrity_impact = self.assess_data_integrity_impact(classification, violation)?;
        let performance_impact = self.assess_performance_impact(classification)?;
        let business_impact = self.assess_business_impact(classification)?;
        let security_impact = self.assess_security_impact(classification)?;

        Ok(DetailedImpactAssessment {
            data_integrity_impact,
            performance_impact,
            business_impact,
            security_impact,
            user_impact: UserImpact {
                affected_user_count: 100, // Placeholder
                user_experience_degradation: 0.3,
                workflow_disruption: 0.4,
            },
            system_impact: SystemImpact {
                resource_consumption: 0.2,
                scalability_impact: 0.3,
                reliability_impact: 0.4,
            },
        })
    }

    fn assess_data_integrity_impact(
        &self,
        classification: &ErrorClassificationResult,
        violation: &Violation,
    ) -> Result<f64> {
        match classification.error_type {
            ErrorType::DataQuality => Ok(0.8),
            ErrorType::ConstraintViolation => Ok(0.6),
            ErrorType::SchemaInconsistency => Ok(0.9),
            _ => Ok(0.4),
        }
    }

    fn assess_performance_impact(&self, classification: &ErrorClassificationResult) -> Result<f64> {
        match classification.error_type {
            ErrorType::Performance => Ok(0.9),
            ErrorType::ConstraintViolation => Ok(0.3),
            _ => Ok(0.2),
        }
    }

    fn assess_business_impact(&self, classification: &ErrorClassificationResult) -> Result<f64> {
        match classification.business_criticality {
            BusinessCriticality::Mission => Ok(0.95),
            BusinessCriticality::High => Ok(0.8),
            BusinessCriticality::Medium => Ok(0.6),
            BusinessCriticality::Low => Ok(0.3),
            BusinessCriticality::None => Ok(0.1),
        }
    }

    fn assess_security_impact(&self, classification: &ErrorClassificationResult) -> Result<f64> {
        // Security impact based on error type and criticality
        let base_security_impact = match classification.error_type {
            ErrorType::Structural => 0.4,
            ErrorType::DataQuality => 0.3,
            _ => 0.1,
        };

        // Amplify based on business criticality
        let criticality_multiplier = match classification.business_criticality {
            BusinessCriticality::Mission => 1.5,
            BusinessCriticality::High => 1.2,
            _ => 1.0,
        };

        Ok((base_security_impact * criticality_multiplier).min(1.0))
    }
}

/// Prevention strategy generator
#[derive(Debug)]
pub struct PreventionStrategyGenerator {
    config: ErrorHandlingConfig,
}

impl PreventionStrategyGenerator {
    pub fn new(config: ErrorHandlingConfig) -> Self {
        Self { config }
    }

    /// Generate prevention strategies for an error type
    pub fn generate_strategies(
        &self,
        classification: &ErrorClassificationResult,
        violation: &Violation,
    ) -> Result<Vec<PreventionStrategy>> {
        let mut strategies = Vec::new();

        // Process improvement strategy
        strategies.push(PreventionStrategy {
            strategy_type: PreventionType::ProcessImprovement,
            description: "Enhance validation process to prevent similar errors".to_string(),
            implementation_steps: vec![
                "Analyze current validation workflow".to_string(),
                "Identify process gaps".to_string(),
                "Design improved process".to_string(),
                "Train team on new process".to_string(),
            ],
            effectiveness: 0.8,
            cost_benefit: CostBenefit {
                implementation_cost: 5000.0,
                maintenance_cost: 1000.0,
                annual_savings: 15000.0,
                roi_estimate: 200.0,
                payback_period: 4,
            },
            monitoring: vec![
                "Track error reduction rate".to_string(),
                "Monitor process compliance".to_string(),
            ],
        });

        // Tool integration strategy
        if matches!(classification.error_type, ErrorType::DataQuality) {
            strategies.push(PreventionStrategy {
                strategy_type: PreventionType::ToolIntegration,
                description: "Integrate data quality tools to prevent quality issues".to_string(),
                implementation_steps: vec![
                    "Evaluate data quality tools".to_string(),
                    "Select appropriate tool".to_string(),
                    "Configure validation rules".to_string(),
                    "Integrate into pipeline".to_string(),
                ],
                effectiveness: 0.9,
                cost_benefit: CostBenefit {
                    implementation_cost: 10000.0,
                    maintenance_cost: 2000.0,
                    annual_savings: 25000.0,
                    roi_estimate: 150.0,
                    payback_period: 5,
                },
                monitoring: vec![
                    "Track data quality metrics".to_string(),
                    "Monitor tool performance".to_string(),
                ],
            });
        }

        Ok(strategies)
    }
}

// Supporting data structures for comprehensive error analysis

/// Complete smart error analysis result
#[derive(Debug, Clone)]
pub struct SmartErrorAnalysis {
    pub individual_analyses: Vec<SingleErrorAnalysis>,
    pub comprehensive_analysis: ComprehensiveErrorAnalysis,
    pub processing_metadata: ProcessingMetadata,
}

/// Analysis of a single error
#[derive(Debug, Clone)]
pub struct SingleErrorAnalysis {
    pub violation: Violation,
    pub classification: ErrorClassificationResult,
    pub repair_suggestions: Vec<RepairSuggestion>,
    pub impact_assessment: DetailedImpactAssessment,
    pub prevention_strategies: Vec<PreventionStrategy>,
    pub analysis_metadata: AnalysisMetadata,
}

/// Comprehensive analysis across all errors
#[derive(Debug, Clone)]
pub struct ComprehensiveErrorAnalysis {
    pub error_patterns: Vec<ErrorPattern>,
    pub systemic_recommendations: Vec<SystemicRecommendation>,
    pub quality_impact: OverallQualityImpact,
    pub priority_ranking: Vec<PriorityRankedError>,
    pub aggregated_metrics: AggregatedMetrics,
    pub trend_analysis: TrendAnalysis,
    pub root_cause_analysis: RootCauseAnalysis,
}

/// Error pattern identification
#[derive(Debug, Clone)]
pub struct ErrorPattern {
    pub pattern_type: ErrorType,
    pub frequency: usize,
    pub confidence: f64,
    pub description: String,
    pub affected_shapes: Vec<ShapeId>,
    pub common_characteristics: Vec<String>,
}

/// Systemic recommendation for addressing multiple errors
#[derive(Debug, Clone)]
pub struct SystemicRecommendation {
    pub recommendation_type: SystemicRecommendationType,
    pub title: String,
    pub description: String,
    pub implementation_priority: SystemicPriority,
    pub expected_impact: f64,
    pub implementation_steps: Vec<String>,
    pub success_metrics: Vec<String>,
}

/// Types of systemic recommendations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SystemicRecommendationType {
    ProcessImprovement,
    QualityGate,
    ToolUpgrade,
    TrainingProgram,
    PolicyChange,
}

/// Systemic priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum SystemicPriority {
    Low = 1,
    Medium = 2,
    High = 3,
    Urgent = 4,
}

/// Overall quality impact assessment
#[derive(Debug, Clone)]
pub struct OverallQualityImpact {
    pub severity_distribution: SeverityDistribution,
    pub cumulative_impact: CumulativeImpact,
    pub quality_degradation: QualityDegradation,
}

impl Default for OverallQualityImpact {
    fn default() -> Self {
        Self {
            severity_distribution: SeverityDistribution {
                critical_percentage: 0.0,
                high_percentage: 0.0,
                total_errors: 0,
            },
            cumulative_impact: CumulativeImpact {
                data_integrity: 0.0,
                performance: 0.0,
                business: 0.0,
                overall_score: 0.0,
            },
            quality_degradation: QualityDegradation {
                estimated_degradation: 0.0,
                confidence: 0.0,
                trend_indicator: "stable".to_string(),
            },
        }
    }
}

/// Distribution of error severities
#[derive(Debug, Clone)]
pub struct SeverityDistribution {
    pub critical_percentage: f64,
    pub high_percentage: f64,
    pub total_errors: usize,
}

/// Cumulative impact across all errors
#[derive(Debug, Clone)]
pub struct CumulativeImpact {
    pub data_integrity: f64,
    pub performance: f64,
    pub business: f64,
    pub overall_score: f64,
}

/// Quality degradation assessment
#[derive(Debug, Clone)]
pub struct QualityDegradation {
    pub estimated_degradation: f64,
    pub confidence: f64,
    pub trend_indicator: String,
}

/// Priority-ranked error
#[derive(Debug, Clone)]
pub struct PriorityRankedError {
    pub error_index: usize,
    pub priority_score: f64,
    pub ranking_factors: PriorityFactors,
    pub recommended_timeline: Duration,
}

/// Factors used in priority ranking
#[derive(Debug, Clone)]
pub struct PriorityFactors {
    pub severity_weight: f64,
    pub business_impact_weight: f64,
    pub resolution_difficulty_weight: f64,
    pub urgency_weight: f64,
}

/// Detailed impact assessment
#[derive(Debug, Clone)]
pub struct DetailedImpactAssessment {
    pub data_integrity_impact: f64,
    pub performance_impact: f64,
    pub business_impact: f64,
    pub security_impact: f64,
    pub user_impact: UserImpact,
    pub system_impact: SystemImpact,
}

impl Default for DetailedImpactAssessment {
    fn default() -> Self {
        Self {
            data_integrity_impact: 0.0,
            performance_impact: 0.0,
            business_impact: 0.0,
            security_impact: 0.0,
            user_impact: UserImpact {
                affected_user_count: 0,
                user_experience_degradation: 0.0,
                workflow_disruption: 0.0,
            },
            system_impact: SystemImpact {
                resource_consumption: 0.0,
                scalability_impact: 0.0,
                reliability_impact: 0.0,
            },
        }
    }
}

/// User impact details
#[derive(Debug, Clone)]
pub struct UserImpact {
    pub affected_user_count: usize,
    pub user_experience_degradation: f64,
    pub workflow_disruption: f64,
}

/// System impact details
#[derive(Debug, Clone)]
pub struct SystemImpact {
    pub resource_consumption: f64,
    pub scalability_impact: f64,
    pub reliability_impact: f64,
}

/// Aggregated metrics across all errors
#[derive(Debug, Clone)]
pub struct AggregatedMetrics {
    pub total_errors: usize,
    pub type_distribution: HashMap<ErrorType, usize>,
    pub average_confidence: f64,
    pub average_impact: f64,
    pub resolution_complexity: f64,
}

impl Default for AggregatedMetrics {
    fn default() -> Self {
        Self {
            total_errors: 0,
            type_distribution: HashMap::new(),
            average_confidence: 0.0,
            average_impact: 0.0,
            resolution_complexity: 0.0,
        }
    }
}

/// Trend analysis results
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    pub trend_direction: String,
    pub trend_confidence: f64,
    pub seasonal_patterns: Vec<String>,
    pub forecast: Vec<f64>,
}

/// Root cause analysis results
#[derive(Debug, Clone)]
pub struct RootCauseAnalysis {
    pub primary_causes: Vec<String>,
    pub contributing_factors: Vec<String>,
    pub systemic_issues: Vec<String>,
    pub confidence: f64,
}

/// Processing metadata
#[derive(Debug, Clone)]
pub struct ProcessingMetadata {
    pub processing_time: Duration,
    pub errors_processed: usize,
    pub ml_models_used: Vec<String>,
    pub confidence_distribution: HashMap<String, f64>,
}

/// Analysis metadata for individual errors
#[derive(Debug, Clone)]
pub struct AnalysisMetadata {
    pub analysis_timestamp: chrono::DateTime<chrono::Utc>,
    pub processing_duration: Duration,
    pub models_used: Vec<String>,
    pub confidence_score: f64,
}

impl Default for IntelligentErrorHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_handler_creation() {
        let handler = IntelligentErrorHandler::new();
        assert_eq!(handler.config.min_repair_confidence, 0.7);
    }

    #[test]
    fn test_error_classification_confidence() {
        let config = ErrorHandlingConfig::default();
        assert!(config.min_repair_confidence > 0.0);
        assert!(config.min_repair_confidence <= 1.0);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(ErrorSeverity::Critical > ErrorSeverity::High);
        assert!(ErrorSeverity::High > ErrorSeverity::Medium);
        assert!(ErrorSeverity::Medium > ErrorSeverity::Low);
    }

    #[test]
    fn test_priority_calculation() {
        let handler = IntelligentErrorHandler::new();
        let high_severity = ErrorSeverity::High;
        let low_severity = ErrorSeverity::Low;

        let high_weight = handler.severity_to_weight(&high_severity);
        let low_weight = handler.severity_to_weight(&low_severity);

        assert!(high_weight > low_weight);
    }
}
