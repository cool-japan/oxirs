//! Core types for the Recommendation Systems module.
//!
//! Contains all structs, enums, and data types used by the recommendation engine.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Configuration for recommendation systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationConfig {
    /// Enable machine learning-based recommendations
    pub enable_ml_recommendations: bool,

    /// Minimum confidence threshold for recommendations
    pub min_confidence_threshold: f64,

    /// Maximum number of recommendations per category
    pub max_recommendations_per_category: usize,

    /// Enable personalized recommendations based on user behavior
    pub enable_personalization: bool,

    /// Recommendation refresh interval in hours
    pub refresh_interval_hours: u64,

    /// Include experimental recommendations
    pub include_experimental: bool,

    /// Prioritize high-impact recommendations
    pub prioritize_high_impact: bool,
}

impl Default for RecommendationConfig {
    fn default() -> Self {
        Self {
            enable_ml_recommendations: true,
            min_confidence_threshold: 0.7,
            max_recommendations_per_category: 10,
            enable_personalization: true,
            refresh_interval_hours: 24,
            include_experimental: false,
            prioritize_high_impact: true,
        }
    }
}

/// Types of recommendations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Shape structure improvements
    ShapeImprovement,
    /// Validation strategy optimization
    ValidationStrategy,
    /// Performance optimization
    PerformanceOptimization,
    /// Quality enhancement
    QualityEnhancement,
    /// Tool recommendations
    ToolRecommendation,
    /// Process improvements
    ProcessImprovement,
    /// Training suggestions
    TrainingRecommendation,
    /// Investment priorities
    InvestmentPriority,
    /// Security enhancements
    SecurityEnhancement,
    /// Maintenance automation
    MaintenanceAutomation,
}

/// Priority levels for recommendations
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Information,
    Low,
    Medium,
    High,
    Critical,
}

/// Impact categories
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ImpactCategory {
    PerformanceImprovement,
    QualityEnhancement,
    CostReduction,
    UserExperience,
    Maintainability,
    Security,
    Scalability,
}

/// Individual recommendation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Recommendation {
    pub id: Uuid,
    pub recommendation_type: RecommendationType,
    pub priority: RecommendationPriority,
    pub title: String,
    pub description: String,
    pub rationale: String,
    pub confidence: f64,
    pub estimated_impact: EstimatedImpact,
    pub implementation_effort: ImplementationEffort,
    pub prerequisites: Vec<String>,
    pub expected_outcomes: Vec<String>,
    pub implementation_steps: Vec<ImplementationStep>,
    pub success_metrics: Vec<SuccessMetric>,
    pub related_recommendations: Vec<Uuid>,
    pub tags: HashSet<String>,
    pub created_at: DateTime<Utc>,
    pub valid_until: Option<DateTime<Utc>>,
    pub applied: bool,
    pub effectiveness_score: Option<f64>,
}

/// Estimated impact of a recommendation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EstimatedImpact {
    pub categories: Vec<ImpactCategory>,
    pub quantitative_benefits: HashMap<String, f64>,
    pub qualitative_benefits: Vec<String>,
    pub potential_risks: Vec<String>,
    pub roi_estimate: Option<f64>,
    pub payback_period_months: Option<u32>,
}

/// Implementation effort assessment
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImplementationEffort {
    pub complexity: EffortComplexity,
    pub estimated_hours: f64,
    pub required_skills: Vec<String>,
    pub required_resources: Vec<String>,
    pub dependencies: Vec<String>,
    pub risk_level: RiskLevel,
}

/// Effort complexity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffortComplexity {
    Trivial,
    Simple,
    Moderate,
    Complex,
    Significant,
}

/// Risk levels for implementation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    Minimal,
    Low,
    Medium,
    High,
    Critical,
}

/// Implementation step
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImplementationStep {
    pub step_number: usize,
    pub title: String,
    pub description: String,
    pub estimated_duration: f64, // hours
    pub required_roles: Vec<String>,
    pub deliverables: Vec<String>,
    pub validation_criteria: Vec<String>,
}

/// Success metric for tracking recommendation effectiveness
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuccessMetric {
    pub metric_name: String,
    pub measurement_method: String,
    pub target_value: f64,
    pub baseline_value: Option<f64>,
    pub measurement_frequency: String,
    pub threshold_for_success: f64,
}

/// Comprehensive recommendation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationReport {
    pub report_id: Uuid,
    pub generated_at: DateTime<Utc>,
    pub recommendations_by_type: HashMap<RecommendationType, Vec<Recommendation>>,
    pub prioritized_recommendations: Vec<Recommendation>,
    pub quick_wins: Vec<Recommendation>,
    pub strategic_initiatives: Vec<Recommendation>,
    pub summary: RecommendationSummary,
    pub implementation_roadmap: ImplementationRoadmap,
}

/// Summary of recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationSummary {
    pub total_recommendations: usize,
    pub critical_count: usize,
    pub high_priority_count: usize,
    pub estimated_total_impact: f64,
    pub estimated_total_effort: f64,
    pub top_impact_categories: Vec<ImpactCategory>,
    pub key_themes: Vec<String>,
}

/// Implementation roadmap
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationRoadmap {
    pub phases: Vec<RoadmapPhase>,
    pub total_duration_months: u32,
    pub resource_requirements: Vec<ResourceRequirement>,
    pub milestones: Vec<Milestone>,
}

/// Roadmap phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoadmapPhase {
    pub phase_number: usize,
    pub name: String,
    pub description: String,
    pub duration_months: u32,
    pub recommendations: Vec<Uuid>,
    pub objectives: Vec<String>,
    pub success_criteria: Vec<String>,
}

/// Resource requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirement {
    pub resource_type: String,
    pub quantity: f64,
    pub duration_months: u32,
    pub cost_estimate: Option<f64>,
}

/// Milestone
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Milestone {
    pub name: String,
    pub target_date_months: u32,
    pub deliverables: Vec<String>,
    pub success_metrics: Vec<String>,
}

/// Record of recommendation history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationRecord {
    pub recommendation: Recommendation,
    pub applied_at: Option<DateTime<Utc>>,
    pub outcome: Option<RecommendationOutcome>,
    pub user_feedback: Option<UserFeedback>,
}

/// Outcome of applying a recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationOutcome {
    pub success: bool,
    pub actual_impact: HashMap<String, f64>,
    pub actual_effort: f64,
    pub lessons_learned: Vec<String>,
    pub side_effects: Vec<String>,
}

/// User feedback on recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserFeedback {
    pub usefulness_rating: u8,         // 1-5 scale
    pub implementation_difficulty: u8, // 1-5 scale
    pub comments: String,
    pub would_recommend_to_others: bool,
}

/// Effectiveness tracking for recommendations
#[derive(Debug)]
pub struct EffectivenessTracker {
    pub applied_recommendations: HashMap<Uuid, RecommendationOutcome>,
    pub success_rates_by_type: HashMap<RecommendationType, f64>,
    pub average_roi_by_category: HashMap<ImpactCategory, f64>,
}

/// Machine learning model for generating recommendations
#[derive(Debug)]
pub struct RecommendationModel {
    pub model_weights: HashMap<String, f64>,
    pub feature_importance: HashMap<String, f64>,
    pub user_preferences: HashMap<String, f64>,
    pub historical_patterns: Vec<RecommendationPattern>,
}

/// Pattern learned from recommendation history
#[derive(Debug, Clone)]
pub struct RecommendationPattern {
    pub context_features: HashMap<String, f64>,
    pub successful_recommendation_types: Vec<RecommendationType>,
    pub pattern_confidence: f64,
}
