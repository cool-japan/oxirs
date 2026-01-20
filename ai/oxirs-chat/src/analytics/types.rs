//! Core types and configuration for analytics
//!
//! This module contains the fundamental types, enums, and configuration
//! structures used throughout the analytics system.

use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    time::{Duration, SystemTime},
};

/// Message intent classification with confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageIntent {
    pub intent_type: IntentType,
    pub confidence: f64,
    pub reasoning: String,
}

/// Types of message intents
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntentType {
    Question,
    Request,
    Gratitude,
    Aggregation,
    ListQuery,
    Comparison,
    Relationship,
    Definition,
    Complaint,
    Clarification,
    Complex,
    Exploration,
}

/// Metrics for measuring message complexity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    pub linguistic_complexity: f64,
    pub semantic_complexity: f64,
    pub context_dependency: f64,
    pub reasoning_depth: f64,
    pub overall_complexity: f64,
}

/// Confidence analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceMetrics {
    pub overall_confidence: f64,
    pub uncertainty_factors: Vec<UncertaintyFactor>,
    pub confidence_breakdown: HashMap<String, f64>,
}

/// Factor that contributes to uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyFactor {
    pub factor_type: String,
    pub impact: f64,
    pub description: String,
}

/// User satisfaction metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SatisfactionMetrics {
    pub overall_satisfaction: f64,
    pub satisfaction_breakdown: HashMap<String, f64>,
    pub implicit_signals: ImplicitSatisfactionSignals,
    pub explicit_feedback: Option<f64>,
}

/// Implicit signals that indicate user satisfaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplicitSatisfactionSignals {
    pub follow_up_questions: usize,
    pub positive_feedback_indicators: usize,
    pub task_completion_rate: f64,
    pub session_continuation: bool,
}

/// Score for a specific emotion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionScore {
    pub emotion: String,
    pub intensity: f64,
    pub confidence: f64,
}

/// Configuration for analytics collection and processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsConfig {
    pub enable_real_time_analytics: bool,
    pub enable_pattern_detection: bool,
    pub enable_sentiment_analysis: bool,
    pub enable_intent_tracking: bool,
    pub analytics_retention_days: usize,
    pub pattern_detection_window: Duration,
    pub anomaly_detection_threshold: f32,
    pub min_pattern_frequency: usize,
    pub privacy_mode: PrivacyMode,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            enable_real_time_analytics: true,
            enable_pattern_detection: true,
            enable_sentiment_analysis: false, // Requires additional dependencies
            enable_intent_tracking: true,
            analytics_retention_days: 30,
            pattern_detection_window: Duration::from_secs(3600), // 1 hour
            anomaly_detection_threshold: 2.0,                    // Standard deviations
            min_pattern_frequency: 3,
            privacy_mode: PrivacyMode::Aggregated,
        }
    }
}

/// Privacy modes for analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrivacyMode {
    Full,       // Store all data
    Aggregated, // Only aggregated statistics
    Anonymous,  // No personally identifiable information
    Disabled,   // No analytics collection
}

/// Comprehensive conversation analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationAnalytics {
    pub session_id: String,
    pub start_time: SystemTime,
    pub end_time: Option<SystemTime>,
    pub message_count: usize,
    pub user_message_count: usize,
    pub assistant_message_count: usize,
    pub average_response_time: Duration,
    pub total_tokens: usize,
    pub user_satisfaction: SatisfactionMetrics,
    pub conversation_quality: ConversationQuality,
    pub topics_discussed: Vec<String>,
    pub sentiment_progression: Vec<EmotionScore>,
    pub complexity_progression: Vec<ComplexityMetrics>,
    pub confidence_progression: Vec<ConfidenceMetrics>,
    pub intent_distribution: HashMap<IntentType, usize>,
    pub patterns_detected: Vec<ConversationPattern>,
    pub anomalies: Vec<ConversationAnomaly>,
    pub metadata: HashMap<String, String>,
}

/// Quality metrics for conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationQuality {
    pub coherence_score: f64,
    pub relevance_score: f64,
    pub helpfulness_score: f64,
    pub accuracy_score: f64,
    pub clarity_score: f64,
    pub completeness_score: f64,
    pub engagement_score: f64,
    pub error_rate: f64,
    pub response_appropriateness: f64,
    pub overall_quality: f64,
}

/// Detected conversation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationPattern {
    pub pattern_type: PatternType,
    pub description: String,
    pub confidence: f64,
    pub frequency: usize,
    pub examples: Vec<String>,
    pub insights: Vec<String>,
}

/// Types of patterns that can be detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    RepeatedQuestion,
    TopicProgression,
    SentimentShift,
    ComplexityEscalation,
    ErrorCascade,
    SuccessPattern,
    EngagementPattern,
    LearningPattern,
    FrustrationPattern,
    ExplorationPattern,
}

/// Detected anomalies in conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationAnomaly {
    pub anomaly_type: AnomalyType,
    pub description: String,
    pub severity: AnomalySeverity,
    pub detected_at: SystemTime,
    pub message_context: Vec<String>,
    pub suggested_action: Option<String>,
}

/// Types of anomalies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyType {
    UnusualResponseTime,
    LowQualityResponses,
    HighErrorRate,
    UnexpectedSentiment,
    ComplexitySpike,
    EngagementDrop,
    RepeatedErrors,
    ContextLoss,
    TopicDivergence,
    ConfidenceCollapse,
}

/// Severity levels for anomalies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Aggregated analytics across multiple conversations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedAnalytics {
    pub total_conversations: usize,
    pub total_messages: usize,
    pub average_conversation_length: f64,
    pub average_response_time: Duration,
    pub overall_satisfaction: SatisfactionStats,
    pub quality_metrics: ConversationQuality,
    pub popular_topics: Vec<(String, usize)>,
    pub common_patterns: Vec<ConversationPattern>,
    pub frequent_anomalies: Vec<(AnomalyType, usize)>,
    pub temporal_trends: HashMap<String, Vec<f64>>,
    pub user_engagement_stats: EngagementStats,
}

impl Default for AggregatedAnalytics {
    fn default() -> Self {
        Self {
            total_conversations: 0,
            total_messages: 0,
            average_conversation_length: 0.0,
            average_response_time: Duration::from_secs(0),
            overall_satisfaction: SatisfactionStats::default(),
            quality_metrics: ConversationQuality {
                coherence_score: 0.0,
                relevance_score: 0.0,
                helpfulness_score: 0.0,
                accuracy_score: 0.0,
                clarity_score: 0.0,
                completeness_score: 0.0,
                engagement_score: 0.0,
                error_rate: 0.0,
                response_appropriateness: 0.0,
                overall_quality: 0.0,
            },
            popular_topics: Vec::new(),
            common_patterns: Vec::new(),
            frequent_anomalies: Vec::new(),
            temporal_trends: HashMap::new(),
            user_engagement_stats: EngagementStats::default(),
        }
    }
}

/// Statistics for user satisfaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SatisfactionStats {
    pub average_satisfaction: f64,
    pub satisfaction_distribution: HashMap<String, f64>,
    pub improvement_trends: Vec<f64>,
}

impl Default for SatisfactionStats {
    fn default() -> Self {
        Self {
            average_satisfaction: 0.0,
            satisfaction_distribution: HashMap::new(),
            improvement_trends: Vec::new(),
        }
    }
}

/// User engagement statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngagementStats {
    pub average_session_duration: Duration,
    pub messages_per_session: f64,
    pub return_rate: f64,
    pub feature_usage: HashMap<String, usize>,
    pub peak_usage_hours: Vec<usize>,
}

impl Default for EngagementStats {
    fn default() -> Self {
        Self {
            average_session_duration: Duration::from_secs(0),
            messages_per_session: 0.0,
            return_rate: 0.0,
            feature_usage: HashMap::new(),
            peak_usage_hours: Vec::new(),
        }
    }
}
