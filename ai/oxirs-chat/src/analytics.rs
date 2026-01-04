//! Analytics module for conversation analysis

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Conversation analytics with comprehensive metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConversationAnalytics {
    // Core progression metrics
    pub complexity_progression: Vec<ComplexityMetrics>,
    pub confidence_progression: Vec<ConfidenceMetrics>,
    pub quality: ConversationQuality,
    pub satisfaction: SatisfactionMetrics,
    pub implicit_signals: ImplicitSatisfactionSignals,

    // Additional fields for persistence.rs
    pub session_id: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub message_count: usize,
    pub user_message_count: usize,
    pub assistant_message_count: usize,
    pub average_response_time: f64,
    pub total_tokens: usize,
    pub user_satisfaction: f32,
    pub conversation_quality: f32,
    pub topics_discussed: Vec<String>,
    pub sentiment_progression: Vec<f32>,
    pub intent_distribution: HashMap<String, usize>,
    pub patterns_detected: Vec<String>,
    pub anomalies: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    pub turn_number: usize,
    pub message_complexity: f32,
    pub topic_depth: f32,
    pub reasoning_complexity: f32,
    pub linguistic_complexity: f32,
    pub semantic_complexity: f32,
    pub context_dependency: f32,
    pub reasoning_depth: f32,
    pub overall_complexity: f32,
}

/// Confidence metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceMetrics {
    pub turn_number: usize,
    pub confidence_score: f32,
    pub uncertainty_markers: usize,
    pub overall_confidence: f32,
    pub uncertainty_factors: Vec<String>,
    pub confidence_breakdown: HashMap<String, f32>,
}

/// Conversation quality with detailed scoring
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConversationQuality {
    pub overall_score: f32,
    pub overall_quality: f32,
    pub coherence: f32,
    pub coherence_score: f32,
    pub relevance: f32,
    pub relevance_score: f32,
    pub completeness: f32,
    pub completeness_score: f32,
    pub helpfulness_score: f32,
    pub accuracy_score: f32,
    pub clarity_score: f32,
    pub engagement_score: f32,
    pub error_rate: f32,
    pub response_appropriateness: f32,
}

/// Satisfaction metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SatisfactionMetrics {
    pub overall_satisfaction: f32,
    pub response_quality: f32,
    pub helpfulness: f32,
    pub satisfaction_breakdown: HashMap<String, f32>,
    pub explicit_feedback: Vec<String>,
    pub implicit_signals: ImplicitSatisfactionSignals,
}

/// Implicit satisfaction signals
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImplicitSatisfactionSignals {
    pub positive_acknowledgments: usize,
    pub clarification_requests: usize,
    pub topic_continuity: f32,
    pub follow_up_questions: usize,
    pub positive_feedback_indicators: usize,
    pub task_completion_rate: f32,
    pub session_continuation: bool,
}

/// Emotion score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionScore {
    pub emotion: String,
    pub emotion_type: String,
    pub intensity: f32,
    pub confidence: f32,
}

/// Intent type with Hash for HashMap usage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntentType {
    Question,
    Command,
    Statement,
    Feedback,
    Clarification,
    Request,
    Gratitude,
    Complex,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analytics_creation() {
        let analytics = ConversationAnalytics::default();
        assert_eq!(analytics.complexity_progression.len(), 0);
        assert_eq!(analytics.message_count, 0);
    }

    #[test]
    fn test_intent_type_hash() {
        let mut map = HashMap::new();
        map.insert(IntentType::Question, 5);
        assert_eq!(map.get(&IntentType::Question), Some(&5));
    }
}
