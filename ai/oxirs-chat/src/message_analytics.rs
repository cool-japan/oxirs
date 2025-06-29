//! Message analytics module for OxiRS Chat
//!
//! This module provides analytics capabilities for chat messages including:
//! - Intent classification
//! - Sentiment analysis  
//! - Complexity scoring
//! - Confidence tracking
//! - Success metrics
//! - User satisfaction measurement

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::types::Message;

/// Message analytics result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageAnalytics {
    pub intent_classification: IntentClassification,
    pub sentiment_analysis: SentimentAnalysis,
    pub complexity_score: ComplexityScore,
    pub confidence_tracking: ConfidenceTracking,
    pub success_metrics: SuccessMetrics,
    pub quality_assessment: QualityAssessment,
}

/// Intent classification results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentClassification {
    pub primary_intent: Intent,
    pub secondary_intents: Vec<Intent>,
    pub confidence: f32,
    pub intent_scores: HashMap<String, f32>,
}

/// Intent types for user messages
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Intent {
    Query,              // User asking a question
    Exploration,        // User exploring data
    Learning,           // User learning about concepts
    Verification,       // User verifying information
    Comparison,         // User comparing entities
    Aggregation,        // User requesting summaries/aggregation
    Navigation,         // User navigating through data
    Configuration,      // User configuring system
    Feedback,           // User providing feedback
    Clarification,      // User asking for clarification
    Unknown,            // Intent cannot be determined
}

/// Sentiment analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentAnalysis {
    pub overall_sentiment: Sentiment,
    pub sentiment_score: f32,  // -1.0 (negative) to 1.0 (positive)
    pub emotion_indicators: Vec<EmotionIndicator>,
    pub confidence: f32,
}

/// Sentiment types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Sentiment {
    Positive,
    Neutral,
    Negative,
    Mixed,
}

/// Emotion indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionIndicator {
    pub emotion: Emotion,
    pub intensity: f32,  // 0.0 to 1.0
    pub indicators: Vec<String>,
}

/// Emotion types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Emotion {
    Frustration,
    Satisfaction,
    Curiosity,
    Confusion,
    Excitement,
    Impatience,
    Appreciation,
}

/// Complexity scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityScore {
    pub overall_complexity: ComplexityLevel,
    pub complexity_score: f32,  // 0.0 (simple) to 1.0 (complex)
    pub complexity_factors: Vec<ComplexityFactor>,
    pub readability_score: f32,
}

/// Complexity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Simple,
    Moderate,
    Complex,
    VeryComplex,
}

/// Factors contributing to complexity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityFactor {
    pub factor_type: ComplexityFactorType,
    pub contribution: f32,
    pub description: String,
}

/// Types of complexity factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityFactorType {
    VocabularyComplexity,
    SyntacticComplexity,
    ConceptualComplexity,
    QueryComplexity,
    DomainSpecificity,
    AmbiguityLevel,
}

/// Confidence tracking for responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceTracking {
    pub overall_confidence: f32,
    pub confidence_components: Vec<ConfidenceComponent>,
    pub uncertainty_indicators: Vec<UncertaintyIndicator>,
    pub confidence_trend: ConfidenceTrend,
}

/// Components contributing to confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceComponent {
    pub component_type: ConfidenceComponentType,
    pub score: f32,
    pub weight: f32,
    pub explanation: String,
}

/// Types of confidence components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfidenceComponentType {
    DataQuality,
    SourceReliability,
    QueryCoverage,
    ContextCompleteness,
    MethodReliability,
    PreviousSuccess,
}

/// Uncertainty indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyIndicator {
    pub indicator_type: UncertaintyType,
    pub severity: f32,
    pub description: String,
    pub mitigation_suggestions: Vec<String>,
}

/// Types of uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UncertaintyType {
    DataIncomplete,
    AmbiguousQuery,
    ConflictingInformation,
    OutdatedInformation,
    LowSampleSize,
    MethodLimitations,
}

/// Confidence trend over conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfidenceTrend {
    Increasing,
    Stable,
    Decreasing,
    Fluctuating,
}

/// Success metrics for responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessMetrics {
    pub task_completion_rate: f32,
    pub user_satisfaction_predicted: f32,
    pub response_relevance: f32,
    pub response_completeness: f32,
    pub response_accuracy: f32,
    pub follow_up_indicators: Vec<FollowUpIndicator>,
}

/// Indicators of likely follow-up questions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FollowUpIndicator {
    pub indicator_type: FollowUpType,
    pub likelihood: f32,
    pub suggested_questions: Vec<String>,
}

/// Types of follow-up questions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FollowUpType {
    Clarification,
    Elaboration,
    RelatedQuery,
    Verification,
    Comparison,
    NextStep,
}

/// Quality assessment of messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssessment {
    pub clarity_score: f32,
    pub helpfulness_score: f32,
    pub accuracy_score: f32,
    pub completeness_score: f32,
    pub relevance_score: f32,
    pub quality_issues: Vec<QualityIssue>,
}

/// Quality issues identified
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    pub issue_type: QualityIssueType,
    pub severity: QualityIssueSeverity,
    pub description: String,
    pub suggestions: Vec<String>,
}

/// Types of quality issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityIssueType {
    Ambiguity,
    Incompleteness,
    Inconsistency,
    IrrelevantInformation,
    TechnicalJargon,
    MissingContext,
    PoorStructure,
}

/// Severity of quality issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityIssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Message analytics engine
pub struct MessageAnalyticsEngine {
    intent_classifier: IntentClassifier,
    sentiment_analyzer: SentimentAnalyzer,
    complexity_scorer: ComplexityScorer,
    confidence_tracker: ConfidenceTracker,
    success_evaluator: SuccessEvaluator,
    quality_assessor: QualityAssessor,
}

impl MessageAnalyticsEngine {
    /// Create a new message analytics engine
    pub fn new() -> Self {
        Self {
            intent_classifier: IntentClassifier::new(),
            sentiment_analyzer: SentimentAnalyzer::new(),
            complexity_scorer: ComplexityScorer::new(),
            confidence_tracker: ConfidenceTracker::new(),
            success_evaluator: SuccessEvaluator::new(),
            quality_assessor: QualityAssessor::new(),
        }
    }

    /// Analyze a message and return comprehensive analytics
    pub async fn analyze_message(
        &self,
        message: &Message,
        context: &[Message],
        response: Option<&Message>,
    ) -> Result<MessageAnalytics> {
        let intent_classification = self.intent_classifier.classify(message, context).await?;
        let sentiment_analysis = self.sentiment_analyzer.analyze(message).await?;
        let complexity_score = self.complexity_scorer.score(message).await?;
        let confidence_tracking = self.confidence_tracker.track(message, response).await?;
        let success_metrics = self.success_evaluator.evaluate(message, response, context).await?;
        let quality_assessment = self.quality_assessor.assess(message, response).await?;

        Ok(MessageAnalytics {
            intent_classification,
            sentiment_analysis,
            complexity_score,
            confidence_tracking,
            success_metrics,
            quality_assessment,
        })
    }

    /// Analyze conversation trends
    pub async fn analyze_conversation_trends(
        &self,
        messages: &[Message],
    ) -> Result<ConversationTrends> {
        let mut intent_distribution = HashMap::new();
        let mut sentiment_progression = Vec::new();
        let mut complexity_progression = Vec::new();
        let mut satisfaction_trend = Vec::new();

        for message in messages {
            let analytics = self.analyze_message(message, &[], None).await?;
            
            // Track intent distribution
            let intent_str = format!("{:?}", analytics.intent_classification.primary_intent);
            *intent_distribution.entry(intent_str).or_insert(0) += 1;
            
            // Track sentiment progression
            sentiment_progression.push(analytics.sentiment_analysis.sentiment_score);
            
            // Track complexity progression
            complexity_progression.push(analytics.complexity_score.complexity_score);
            
            // Track satisfaction progression
            satisfaction_trend.push(analytics.success_metrics.user_satisfaction_predicted);
        }

        Ok(ConversationTrends {
            intent_distribution,
            sentiment_progression,
            complexity_progression,
            satisfaction_trend,
            engagement_metrics: self.calculate_engagement_metrics(messages).await?,
        })
    }

    /// Calculate engagement metrics
    async fn calculate_engagement_metrics(&self, messages: &[Message]) -> Result<EngagementMetrics> {
        let total_messages = messages.len();
        let avg_message_length = messages.iter()
            .map(|m| m.content.to_text().len())
            .sum::<usize>() as f32 / total_messages as f32;

        let conversation_duration = if messages.len() > 1 {
            let start = messages.first().unwrap().timestamp;
            let end = messages.last().unwrap().timestamp;
            end.signed_duration_since(start).num_minutes() as f32
        } else {
            0.0
        };

        let response_rate = messages.iter()
            .filter(|m| matches!(m.role, crate::types::MessageRole::Assistant))
            .count() as f32 / total_messages as f32;

        Ok(EngagementMetrics {
            total_messages: total_messages as u32,
            avg_message_length,
            conversation_duration_minutes: conversation_duration,
            response_rate,
            interaction_depth: self.calculate_interaction_depth(messages),
            topic_coherence: self.calculate_topic_coherence(messages).await?,
        })
    }

    fn calculate_interaction_depth(&self, messages: &[Message]) -> f32 {
        // Simple heuristic: measure how deep the conversation goes
        // by looking at follow-up patterns and question complexity
        let follow_up_count = messages.windows(2)
            .filter(|pair| {
                matches!(pair[0].role, crate::types::MessageRole::Assistant) &&
                matches!(pair[1].role, crate::types::MessageRole::User)
            })
            .count();

        follow_up_count as f32 / messages.len().max(1) as f32
    }

    async fn calculate_topic_coherence(&self, messages: &[Message]) -> Result<f32> {
        // Simplified topic coherence calculation
        // In practice, this would use topic modeling
        if messages.len() < 2 {
            return Ok(1.0);
        }

        let mut coherence_scores = Vec::new();
        for window in messages.windows(2) {
            let similarity = self.calculate_semantic_similarity(
                &window[0].content.to_text(),
                &window[1].content.to_text(),
            ).await?;
            coherence_scores.push(similarity);
        }

        Ok(coherence_scores.iter().sum::<f32>() / coherence_scores.len() as f32)
    }

    async fn calculate_semantic_similarity(&self, text1: &str, text2: &str) -> Result<f32> {
        // Mock implementation - would use actual embeddings/similarity
        let common_words = self.get_common_words(text1, text2);
        let total_words = self.get_unique_words(text1).len() + self.get_unique_words(text2).len();
        
        if total_words == 0 {
            Ok(0.0)
        } else {
            Ok(2.0 * common_words as f32 / total_words as f32)
        }
    }

    fn get_common_words(&self, text1: &str, text2: &str) -> usize {
        let words1: std::collections::HashSet<&str> = text1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = text2.split_whitespace().collect();
        words1.intersection(&words2).count()
    }

    fn get_unique_words<'a>(&self, text: &'a str) -> std::collections::HashSet<&'a str> {
        text.split_whitespace().collect()
    }
}

/// Conversation trends analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTrends {
    pub intent_distribution: HashMap<String, u32>,
    pub sentiment_progression: Vec<f32>,
    pub complexity_progression: Vec<f32>,
    pub satisfaction_trend: Vec<f32>,
    pub engagement_metrics: EngagementMetrics,
}

/// Engagement metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngagementMetrics {
    pub total_messages: u32,
    pub avg_message_length: f32,
    pub conversation_duration_minutes: f32,
    pub response_rate: f32,
    pub interaction_depth: f32,
    pub topic_coherence: f32,
}

// Individual component implementations

/// Intent classifier
pub struct IntentClassifier {
    intent_patterns: HashMap<Intent, Vec<String>>,
}

impl IntentClassifier {
    pub fn new() -> Self {
        let mut intent_patterns = HashMap::new();
        
        intent_patterns.insert(Intent::Query, vec![
            "what".to_string(), "how".to_string(), "why".to_string(), 
            "when".to_string(), "where".to_string(), "which".to_string()
        ]);
        
        intent_patterns.insert(Intent::Exploration, vec![
            "show me".to_string(), "explore".to_string(), "browse".to_string(),
            "navigate".to_string(), "discover".to_string()
        ]);
        
        intent_patterns.insert(Intent::Learning, vec![
            "learn".to_string(), "understand".to_string(), "explain".to_string(),
            "teach me".to_string(), "how does".to_string()
        ]);

        Self { intent_patterns }
    }

    pub async fn classify(&self, message: &Message, _context: &[Message]) -> Result<IntentClassification> {
        let text = message.content.to_text().to_lowercase();
        let mut intent_scores = HashMap::new();
        
        // Simple pattern matching for intent classification
        for (intent, patterns) in &self.intent_patterns {
            let mut score = 0.0;
            for pattern in patterns {
                if text.contains(pattern) {
                    score += 1.0;
                }
            }
            score = score / patterns.len() as f32;
            intent_scores.insert(format!("{:?}", intent), score);
        }
        
        // Find primary intent
        let primary_intent = intent_scores.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(intent, _)| match intent.as_str() {
                "Query" => Intent::Query,
                "Exploration" => Intent::Exploration,
                "Learning" => Intent::Learning,
                _ => Intent::Unknown,
            })
            .unwrap_or(Intent::Unknown);
            
        let confidence = intent_scores.values().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);

        Ok(IntentClassification {
            primary_intent,
            secondary_intents: vec![], // Would be populated with secondary intents
            confidence: *confidence,
            intent_scores,
        })
    }
}

/// Sentiment analyzer
pub struct SentimentAnalyzer {
    positive_words: Vec<String>,
    negative_words: Vec<String>,
}

impl SentimentAnalyzer {
    pub fn new() -> Self {
        Self {
            positive_words: vec![
                "good".to_string(), "great".to_string(), "excellent".to_string(),
                "helpful".to_string(), "useful".to_string(), "thanks".to_string()
            ],
            negative_words: vec![
                "bad".to_string(), "terrible".to_string(), "useless".to_string(),
                "frustrated".to_string(), "confused".to_string(), "wrong".to_string()
            ],
        }
    }

    pub async fn analyze(&self, message: &Message) -> Result<SentimentAnalysis> {
        let text = message.content.to_text().to_lowercase();
        let words: Vec<&str> = text.split_whitespace().collect();
        
        let mut positive_count = 0;
        let mut negative_count = 0;
        
        for word in &words {
            if self.positive_words.contains(&word.to_string()) {
                positive_count += 1;
            }
            if self.negative_words.contains(&word.to_string()) {
                negative_count += 1;
            }
        }
        
        let sentiment_score = if words.is_empty() {
            0.0
        } else {
            (positive_count as f32 - negative_count as f32) / words.len() as f32
        };
        
        let overall_sentiment = match sentiment_score {
            s if s > 0.1 => Sentiment::Positive,
            s if s < -0.1 => Sentiment::Negative,
            _ => Sentiment::Neutral,
        };
        
        Ok(SentimentAnalysis {
            overall_sentiment,
            sentiment_score,
            emotion_indicators: vec![], // Would be populated with emotion analysis
            confidence: 0.7, // Mock confidence
        })
    }
}

/// Complexity scorer
pub struct ComplexityScorer;

impl ComplexityScorer {
    pub fn new() -> Self {
        Self
    }

    pub async fn score(&self, message: &Message) -> Result<ComplexityScore> {
        let text = message.content.to_text();
        let word_count = text.split_whitespace().count();
        let sentence_count = text.split(['.', '!', '?']).count();
        let avg_sentence_length = if sentence_count > 0 {
            word_count as f32 / sentence_count as f32
        } else {
            0.0
        };
        
        // Simple complexity scoring based on length and structure
        let complexity_score = (avg_sentence_length / 20.0).min(1.0);
        
        let overall_complexity = match complexity_score {
            s if s < 0.3 => ComplexityLevel::Simple,
            s if s < 0.6 => ComplexityLevel::Moderate,
            s if s < 0.8 => ComplexityLevel::Complex,
            _ => ComplexityLevel::VeryComplex,
        };
        
        Ok(ComplexityScore {
            overall_complexity,
            complexity_score,
            complexity_factors: vec![], // Would be populated with detailed factors
            readability_score: 1.0 - complexity_score, // Inverse relationship
        })
    }
}

/// Confidence tracker
pub struct ConfidenceTracker;

impl ConfidenceTracker {
    pub fn new() -> Self {
        Self
    }

    pub async fn track(&self, _message: &Message, response: Option<&Message>) -> Result<ConfidenceTracking> {
        let overall_confidence = if response.is_some() {
            0.8 // Higher confidence if there's a response
        } else {
            0.5 // Lower confidence for standalone messages
        };
        
        Ok(ConfidenceTracking {
            overall_confidence,
            confidence_components: vec![], // Would be populated with detailed components
            uncertainty_indicators: vec![], // Would be populated with uncertainty analysis
            confidence_trend: ConfidenceTrend::Stable,
        })
    }
}

/// Success evaluator
pub struct SuccessEvaluator;

impl SuccessEvaluator {
    pub fn new() -> Self {
        Self
    }

    pub async fn evaluate(
        &self,
        _message: &Message,
        response: Option<&Message>,
        _context: &[Message],
    ) -> Result<SuccessMetrics> {
        let task_completion_rate = if response.is_some() { 0.9 } else { 0.0 };
        
        Ok(SuccessMetrics {
            task_completion_rate,
            user_satisfaction_predicted: 0.75,
            response_relevance: 0.8,
            response_completeness: 0.7,
            response_accuracy: 0.85,
            follow_up_indicators: vec![], // Would be populated with follow-up analysis
        })
    }
}

/// Quality assessor
pub struct QualityAssessor;

impl QualityAssessor {
    pub fn new() -> Self {
        Self
    }

    pub async fn assess(&self, message: &Message, response: Option<&Message>) -> Result<QualityAssessment> {
        let text = message.content.to_text();
        let clarity_score = if text.len() > 10 { 0.8 } else { 0.4 };
        
        Ok(QualityAssessment {
            clarity_score,
            helpfulness_score: 0.7,
            accuracy_score: 0.8,
            completeness_score: 0.75,
            relevance_score: 0.85,
            quality_issues: vec![], // Would be populated with quality analysis
        })
    }
}

impl Default for MessageAnalyticsEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MessageContent, MessageRole};

    #[tokio::test]
    async fn test_message_analytics_engine() {
        let engine = MessageAnalyticsEngine::new();
        
        let message = Message {
            id: "test".to_string(),
            role: MessageRole::User,
            content: MessageContent::Text("What is the capital of France?".to_string()),
            timestamp: chrono::Utc::now(),
            metadata: None,
            thread_id: None,
            parent_message_id: None,
            token_count: None,
            reactions: Vec::new(),
            attachments: Vec::new(),
            rich_elements: Vec::new(),
        };
        
        let analytics = engine.analyze_message(&message, &[], None).await.unwrap();
        
        assert!(matches!(analytics.intent_classification.primary_intent, Intent::Query));
        assert!(analytics.sentiment_analysis.sentiment_score >= -1.0);
        assert!(analytics.sentiment_analysis.sentiment_score <= 1.0);
        assert!(analytics.complexity_score.complexity_score >= 0.0);
        assert!(analytics.complexity_score.complexity_score <= 1.0);
    }

    #[tokio::test]
    async fn test_intent_classification() {
        let classifier = IntentClassifier::new();
        
        let message = Message {
            id: "test".to_string(),
            role: MessageRole::User,
            content: MessageContent::Text("How does this work?".to_string()),
            timestamp: chrono::Utc::now(),
            metadata: None,
            thread_id: None,
            parent_message_id: None,
            token_count: None,
            reactions: Vec::new(),
            attachments: Vec::new(),
            rich_elements: Vec::new(),
        };
        
        let classification = classifier.classify(&message, &[]).await.unwrap();
        
        assert!(matches!(classification.primary_intent, Intent::Query | Intent::Learning));
        assert!(classification.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_sentiment_analysis() {
        let analyzer = SentimentAnalyzer::new();
        
        let positive_message = Message {
            id: "test".to_string(),
            role: MessageRole::User,
            content: MessageContent::Text("This is great and very helpful!".to_string()),
            timestamp: chrono::Utc::now(),
            metadata: None,
            thread_id: None,
            parent_message_id: None,
            token_count: None,
            reactions: Vec::new(),
            attachments: Vec::new(),
            rich_elements: Vec::new(),
        };
        
        let sentiment = analyzer.analyze(&positive_message).await.unwrap();
        
        assert!(matches!(sentiment.overall_sentiment, Sentiment::Positive));
        assert!(sentiment.sentiment_score > 0.0);
    }
}