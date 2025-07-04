//! Advanced Conversation Analytics for OxiRS Chat
//!
//! Provides comprehensive analytics, insights, pattern recognition, and conversation intelligence
//! to help understand user behavior, optimize responses, and improve the chat experience.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::{Duration, SystemTime},
};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::{
    message_analytics::{
        ComplexityFactor, ComplexityLevel, ComplexityScore, ConfidenceTracking, ConfidenceTrend,
        IntentClassification, MessageAnalytics, QualityAssessment, SentimentAnalysis,
        SuccessMetrics,
    },
    rag::{EntityType, ExtractedEntity, QueryIntent},
    session_manager::TopicTransition,
    types::*,
    ChatSession, Message, MessageContent, MessageMetadata, MessageRole, SessionMetrics,
};

/// Message intent classification with confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageIntent {
    pub intent_type: IntentType,
    pub confidence: f64,
    pub reasoning: String,
}

/// Types of message intents
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
    pub unique_intents: Vec<QueryIntent>,
    pub topic_transitions: Vec<TopicTransition>,
    pub conversation_quality: ConversationQuality,
    pub user_satisfaction: Option<UserSatisfaction>,
    pub pattern_matches: Vec<ConversationPattern>,
    pub anomalies: Vec<ConversationAnomaly>,
}

/// Conversation quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationQuality {
    pub coherence_score: f32,
    pub relevance_score: f32,
    pub completion_rate: f32,
    pub error_rate: f32,
    pub response_accuracy: f32,
    pub user_engagement: f32,
}

/// User satisfaction indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSatisfaction {
    pub explicit_rating: Option<f32>, // 1-5 scale
    pub implicit_satisfaction: f32,   // Inferred from behavior
    pub task_completion: bool,
    pub follow_up_questions: usize,
    pub session_duration: Duration,
    pub abandonment_point: Option<String>,
}

/// Detected conversation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationPattern {
    pub pattern_type: PatternType,
    pub description: String,
    pub frequency: usize,
    pub confidence: f32,
    pub first_occurrence: SystemTime,
    pub last_occurrence: SystemTime,
    pub example_messages: Vec<String>,
}

/// Types of conversation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    QuestionSequence,
    TopicCluster,
    ErrorRecovery,
    TaskCompletion,
    ExploratoryBrowsing,
    RepetitiveQueries,
    DeepDive,
    Comparison,
    TutorialSequence,
    ProblemSolving,
}

/// Conversation anomalies
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    UnusualResponseTime,
    LowQualityResponses,
    HighErrorRate,
    UserFrustration,
    SystemMalfunction,
    UnexpectedTopicShift,
    CircularConversation,
    ResourceExhaustion,
}

/// Severity levels for anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Real-time analytics tracker
pub struct ConversationTracker {
    config: AnalyticsConfig,
    current_analytics: HashMap<String, ConversationAnalytics>,
    pattern_detector: PatternDetector,
    anomaly_detector: AnomalyDetector,
    quality_analyzer: QualityAnalyzer,
    historical_data: Arc<RwLock<VecDeque<ConversationAnalytics>>>,
}

impl ConversationTracker {
    pub fn new(config: AnalyticsConfig) -> Self {
        Self {
            pattern_detector: PatternDetector::new(&config),
            anomaly_detector: AnomalyDetector::new(&config),
            quality_analyzer: QualityAnalyzer::new(&config),
            current_analytics: HashMap::new(),
            historical_data: Arc::new(RwLock::new(VecDeque::new())),
            config,
        }
    }

    /// Start tracking a new conversation session
    pub async fn start_session(&mut self, session_id: String) -> Result<()> {
        let analytics = ConversationAnalytics {
            session_id: session_id.clone(),
            start_time: SystemTime::now(),
            end_time: None,
            message_count: 0,
            user_message_count: 0,
            assistant_message_count: 0,
            average_response_time: Duration::ZERO,
            total_tokens: 0,
            unique_intents: Vec::new(),
            topic_transitions: Vec::new(),
            conversation_quality: ConversationQuality {
                coherence_score: 0.0,
                relevance_score: 0.0,
                completion_rate: 0.0,
                error_rate: 0.0,
                response_accuracy: 0.0,
                user_engagement: 0.0,
            },
            user_satisfaction: None,
            pattern_matches: Vec::new(),
            anomalies: Vec::new(),
        };

        self.current_analytics.insert(session_id.clone(), analytics);
        debug!("Started analytics tracking for session: {}", session_id);
        Ok(())
    }

    /// Track a message in the conversation
    pub async fn track_message(
        &mut self,
        session_id: &str,
        message: &Message,
        response_time: Option<Duration>,
    ) -> Result<()> {
        // Extract intent outside of mutable borrow scope
        let parsed_intent = if let Some(ref metadata) = message.metadata {
            if let Some(intent_value) = metadata.custom_fields.get("intent_classification") {
                if let Some(intent_str) = intent_value.as_str() {
                    self.parse_intent(intent_str).ok()
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        if let Some(analytics) = self.current_analytics.get_mut(session_id) {
            // Update basic metrics
            analytics.message_count += 1;
            match message.role {
                MessageRole::User => analytics.user_message_count += 1,
                MessageRole::Assistant => analytics.assistant_message_count += 1,
                MessageRole::System => {}   // Don't count system messages
                MessageRole::Function => {} // Don't count function messages
            }

            if let Some(token_count) = message.token_count {
                analytics.total_tokens += token_count;
            }

            // Update response time
            if let Some(rt) = response_time {
                let current_avg = analytics.average_response_time.as_millis() as f64;
                let new_avg = (current_avg * (analytics.assistant_message_count - 1) as f64
                    + rt.as_millis() as f64)
                    / analytics.assistant_message_count as f64;
                analytics.average_response_time = Duration::from_millis(new_avg as u64);
            }

            // Track intent if available
            if let Some(intent) = parsed_intent {
                if !analytics.unique_intents.contains(&intent) {
                    analytics.unique_intents.push(intent);
                }
            }

            // Real-time pattern detection
            if self.config.enable_pattern_detection {
                let new_patterns = self
                    .pattern_detector
                    .analyze_message(message, analytics)
                    .await?;
                analytics.pattern_matches.extend(new_patterns);
            }

            // Anomaly detection
            let anomalies = self
                .anomaly_detector
                .analyze_message(message, analytics, response_time)
                .await?;
            analytics.anomalies.extend(anomalies);

            // Quality analysis for assistant messages
            if message.role == MessageRole::Assistant {
                self.quality_analyzer
                    .update_quality_metrics(message, analytics)
                    .await?;
            }

            debug!(
                "Tracked message for session {}: {} total messages",
                session_id, analytics.message_count
            );
        } else {
            warn!(
                "Attempted to track message for unknown session: {}",
                session_id
            );
        }

        Ok(())
    }

    /// End a conversation session
    pub async fn end_session(&mut self, session_id: &str) -> Result<ConversationAnalytics> {
        if let Some(mut analytics) = self.current_analytics.remove(session_id) {
            analytics.end_time = Some(SystemTime::now());

            // Final quality calculation
            analytics.conversation_quality = self
                .quality_analyzer
                .calculate_final_quality(&analytics)
                .await?;

            // Infer user satisfaction
            analytics.user_satisfaction = Some(self.infer_user_satisfaction(&analytics).await?);

            // Store in historical data
            if matches!(
                self.config.privacy_mode,
                PrivacyMode::Full | PrivacyMode::Aggregated
            ) {
                let mut historical = self.historical_data.write().await;
                historical.push_back(analytics.clone());

                // Maintain retention policy
                let retention_limit = self.config.analytics_retention_days * 24; // Rough hourly limit
                while historical.len() > retention_limit {
                    historical.pop_front();
                }
            }

            info!(
                "Ended analytics tracking for session {}: {} messages, {:.2}s avg response time",
                session_id,
                analytics.message_count,
                analytics.average_response_time.as_secs_f64()
            );
            Ok(analytics)
        } else {
            Err(anyhow!("Session not found: {}", session_id))
        }
    }

    /// Get real-time analytics for an active session
    pub fn get_session_analytics(&self, session_id: &str) -> Option<&ConversationAnalytics> {
        self.current_analytics.get(session_id)
    }

    /// Get aggregated analytics across all sessions
    pub async fn get_aggregated_analytics(
        &self,
        time_window: Option<Duration>,
    ) -> Result<AggregatedAnalytics> {
        let historical = self.historical_data.read().await;
        let cutoff_time = time_window.map(|w| SystemTime::now() - w);

        let relevant_sessions: Vec<&ConversationAnalytics> = historical
            .iter()
            .filter(|analytics| {
                if let Some(cutoff) = cutoff_time {
                    analytics.start_time >= cutoff
                } else {
                    true
                }
            })
            .collect();

        if relevant_sessions.is_empty() {
            return Ok(AggregatedAnalytics::default());
        }

        let total_sessions = relevant_sessions.len();
        let total_messages: usize = relevant_sessions.iter().map(|a| a.message_count).sum();
        let total_tokens: usize = relevant_sessions.iter().map(|a| a.total_tokens).sum();

        let avg_response_time = {
            let total_ms: u128 = relevant_sessions
                .iter()
                .map(|a| a.average_response_time.as_millis())
                .sum();
            Duration::from_millis((total_ms / total_sessions as u128) as u64)
        };

        let avg_quality = {
            let quality_sum: f32 = relevant_sessions
                .iter()
                .map(|a| {
                    (a.conversation_quality.coherence_score
                        + a.conversation_quality.relevance_score
                        + a.conversation_quality.response_accuracy)
                        / 3.0
                })
                .sum();
            quality_sum / total_sessions as f32
        };

        // Most common patterns
        let mut pattern_frequency: HashMap<String, usize> = HashMap::new();
        for session in &relevant_sessions {
            for pattern in &session.pattern_matches {
                *pattern_frequency
                    .entry(pattern.description.clone())
                    .or_insert(0) += pattern.frequency;
            }
        }

        let mut top_patterns: Vec<_> = pattern_frequency.into_iter().collect();
        top_patterns.sort_by(|a, b| b.1.cmp(&a.1));
        top_patterns.truncate(10);

        // Intent distribution
        let mut intent_distribution: HashMap<String, usize> = HashMap::new();
        for session in &relevant_sessions {
            for intent in &session.unique_intents {
                *intent_distribution
                    .entry(format!("{:?}", intent))
                    .or_insert(0) += 1;
            }
        }

        // Anomaly summary
        let total_anomalies: usize = relevant_sessions.iter().map(|a| a.anomalies.len()).sum();

        let critical_anomalies: usize = relevant_sessions
            .iter()
            .flat_map(|a| &a.anomalies)
            .filter(|anomaly| matches!(anomaly.severity, AnomalySeverity::Critical))
            .count();

        Ok(AggregatedAnalytics {
            time_window: time_window.unwrap_or(Duration::from_secs(86400 * 30)), // Default 30 days
            total_sessions,
            total_messages,
            total_tokens,
            average_messages_per_session: total_messages as f64 / total_sessions as f64,
            average_response_time: avg_response_time,
            average_quality_score: avg_quality,
            top_patterns,
            intent_distribution,
            total_anomalies,
            critical_anomalies,
            user_satisfaction_stats: self.calculate_satisfaction_stats(&relevant_sessions),
        })
    }

    /// Generate insights and recommendations
    pub async fn generate_insights(&self) -> Result<Vec<ConversationInsight>> {
        let aggregated = self
            .get_aggregated_analytics(Some(Duration::from_secs(86400 * 7)))
            .await?; // Last 7 days
        let mut insights = Vec::new();

        // Response time insights
        if aggregated.average_response_time > Duration::from_secs(3) {
            insights.push(ConversationInsight {
                insight_type: InsightType::Performance,
                title: "Slow Response Times Detected".to_string(),
                description: format!(
                    "Average response time is {:.2}s, which may impact user experience. Consider optimizing queries or adding caching.",
                    aggregated.average_response_time.as_secs_f64()
                ),
                severity: InsightSeverity::Medium,
                suggested_actions: vec![
                    "Enable response caching".to_string(),
                    "Optimize SPARQL queries".to_string(),
                    "Consider faster LLM models for simple queries".to_string(),
                ],
                confidence: 0.8,
            });
        }

        // Quality insights
        if aggregated.average_quality_score < 0.7 {
            insights.push(ConversationInsight {
                insight_type: InsightType::Quality,
                title: "Below Average Response Quality".to_string(),
                description: format!(
                    "Average quality score is {:.2}, indicating room for improvement in response relevance and accuracy.",
                    aggregated.average_quality_score
                ),
                severity: InsightSeverity::High,
                suggested_actions: vec![
                    "Review and improve RAG retrieval strategies".to_string(),
                    "Enhance entity linking accuracy".to_string(),
                    "Fine-tune LLM prompts for better responses".to_string(),
                ],
                confidence: 0.9,
            });
        }

        // Pattern insights
        if let Some((pattern, frequency)) = aggregated.top_patterns.first() {
            if *frequency > aggregated.total_sessions / 4 {
                insights.push(ConversationInsight {
                    insight_type: InsightType::Usage,
                    title: "Dominant Usage Pattern Identified".to_string(),
                    description: format!(
                        "Pattern '{}' appears in {}% of conversations. Consider optimizing for this use case.",
                        pattern, (*frequency * 100) / aggregated.total_sessions
                    ),
                    severity: InsightSeverity::Low,
                    suggested_actions: vec![
                        "Create specialized templates for this pattern".to_string(),
                        "Add quick shortcuts for common queries".to_string(),
                    ],
                    confidence: 0.85,
                });
            }
        }

        // Anomaly insights
        if aggregated.critical_anomalies > 0 {
            insights.push(ConversationInsight {
                insight_type: InsightType::Reliability,
                title: "Critical Anomalies Detected".to_string(),
                description: format!(
                    "{} critical anomalies detected in the last week. Immediate attention required.",
                    aggregated.critical_anomalies
                ),
                severity: InsightSeverity::Critical,
                suggested_actions: vec![
                    "Review system logs for errors".to_string(),
                    "Check resource utilization".to_string(),
                    "Validate data quality".to_string(),
                ],
                confidence: 0.95,
            });
        }

        info!(
            "Generated {} insights from conversation analytics",
            insights.len()
        );
        Ok(insights)
    }

    fn parse_intent(&self, intent_str: &str) -> Result<QueryIntent> {
        match intent_str.to_lowercase().as_str() {
            "factual_lookup" | "factuallookup" => Ok(QueryIntent::General),
            "relationship" => Ok(QueryIntent::General),
            "comparison" => Ok(QueryIntent::Comparison),
            "aggregation" => Ok(QueryIntent::Counting),
            "exploration" => Ok(QueryIntent::General),
            "definition" => Ok(QueryIntent::Definition),
            "list_query" | "listquery" => Ok(QueryIntent::Listing),
            "complex" => Ok(QueryIntent::General),
            _ => Ok(QueryIntent::General), // Default to General instead of error
        }
    }

    async fn infer_user_satisfaction(
        &self,
        analytics: &ConversationAnalytics,
    ) -> Result<UserSatisfaction> {
        // Infer satisfaction from conversation patterns
        let session_duration = analytics
            .end_time
            .unwrap_or_else(SystemTime::now)
            .duration_since(analytics.start_time)
            .unwrap_or(Duration::ZERO);

        let task_completion = analytics.conversation_quality.completion_rate > 0.8;

        let implicit_satisfaction = {
            let mut score: f32 = 0.5; // Baseline

            // Positive indicators
            if analytics.conversation_quality.response_accuracy > 0.8 {
                score += 0.2;
            }
            if analytics.conversation_quality.coherence_score > 0.8 {
                score += 0.1;
            }
            if task_completion {
                score += 0.2;
            }
            if session_duration > Duration::from_secs(60)
                && session_duration < Duration::from_secs(1800)
            {
                score += 0.1;
            }

            // Negative indicators
            if analytics.conversation_quality.error_rate > 0.2 {
                score -= 0.3;
            }
            if analytics.anomalies.iter().any(|a| {
                matches!(
                    a.severity,
                    AnomalySeverity::High | AnomalySeverity::Critical
                )
            }) {
                score -= 0.2;
            }
            if analytics.message_count > 20 && !task_completion {
                score -= 0.1;
            } // Long unsuccessful conversation

            score.max(0.0_f32).min(1.0_f32)
        };

        Ok(UserSatisfaction {
            explicit_rating: None,
            implicit_satisfaction,
            task_completion,
            follow_up_questions: analytics.user_message_count.saturating_sub(1),
            session_duration,
            abandonment_point: if !task_completion && analytics.message_count > 5 {
                Some("Mid-conversation".to_string())
            } else {
                None
            },
        })
    }

    fn calculate_satisfaction_stats(
        &self,
        sessions: &[&ConversationAnalytics],
    ) -> SatisfactionStats {
        if sessions.is_empty() {
            return SatisfactionStats::default();
        }

        let satisfactions: Vec<&UserSatisfaction> = sessions
            .iter()
            .filter_map(|s| s.user_satisfaction.as_ref())
            .collect();

        let avg_implicit = satisfactions
            .iter()
            .map(|s| s.implicit_satisfaction)
            .sum::<f32>()
            / satisfactions.len() as f32;

        let completion_rate = satisfactions.iter().filter(|s| s.task_completion).count() as f32
            / satisfactions.len() as f32;

        let avg_session_duration = {
            let total_duration: Duration = satisfactions.iter().map(|s| s.session_duration).sum();
            total_duration / satisfactions.len() as u32
        };

        SatisfactionStats {
            average_implicit_satisfaction: avg_implicit,
            task_completion_rate: completion_rate,
            average_session_duration: avg_session_duration,
            abandonment_rate: satisfactions
                .iter()
                .filter(|s| s.abandonment_point.is_some())
                .count() as f32
                / satisfactions.len() as f32,
        }
    }
}

/// Aggregated analytics across multiple conversations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedAnalytics {
    pub time_window: Duration,
    pub total_sessions: usize,
    pub total_messages: usize,
    pub total_tokens: usize,
    pub average_messages_per_session: f64,
    pub average_response_time: Duration,
    pub average_quality_score: f32,
    pub top_patterns: Vec<(String, usize)>,
    pub intent_distribution: HashMap<String, usize>,
    pub total_anomalies: usize,
    pub critical_anomalies: usize,
    pub user_satisfaction_stats: SatisfactionStats,
}

impl Default for AggregatedAnalytics {
    fn default() -> Self {
        Self {
            time_window: Duration::ZERO,
            total_sessions: 0,
            total_messages: 0,
            total_tokens: 0,
            average_messages_per_session: 0.0,
            average_response_time: Duration::ZERO,
            average_quality_score: 0.0,
            top_patterns: Vec::new(),
            intent_distribution: HashMap::new(),
            total_anomalies: 0,
            critical_anomalies: 0,
            user_satisfaction_stats: SatisfactionStats::default(),
        }
    }
}

/// User satisfaction statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SatisfactionStats {
    pub average_implicit_satisfaction: f32,
    pub task_completion_rate: f32,
    pub average_session_duration: Duration,
    pub abandonment_rate: f32,
}

impl Default for SatisfactionStats {
    fn default() -> Self {
        Self {
            average_implicit_satisfaction: 0.0,
            task_completion_rate: 0.0,
            average_session_duration: Duration::ZERO,
            abandonment_rate: 0.0,
        }
    }
}

/// Conversation insights and recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationInsight {
    pub insight_type: InsightType,
    pub title: String,
    pub description: String,
    pub severity: InsightSeverity,
    pub suggested_actions: Vec<String>,
    pub confidence: f32,
}

/// Types of insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InsightType {
    Performance,
    Quality,
    Usage,
    Reliability,
    UserExperience,
    BusinessValue,
}

/// Severity levels for insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InsightSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Pattern detection component
struct PatternDetector {
    config: AnalyticsConfig,
    recent_messages: VecDeque<Message>,
}

impl PatternDetector {
    fn new(config: &AnalyticsConfig) -> Self {
        Self {
            config: config.clone(),
            recent_messages: VecDeque::new(),
        }
    }

    async fn analyze_message(
        &mut self,
        message: &Message,
        _analytics: &ConversationAnalytics,
    ) -> Result<Vec<ConversationPattern>> {
        self.recent_messages.push_back(message.clone());

        // Keep only recent messages within the detection window
        let cutoff_time = SystemTime::now() - self.config.pattern_detection_window;
        while let Some(front_msg) = self.recent_messages.front() {
            // Convert chrono::DateTime to SystemTime for comparison
            let msg_system_time = SystemTime::UNIX_EPOCH
                + Duration::from_secs(front_msg.timestamp.timestamp() as u64);
            if msg_system_time < cutoff_time {
                self.recent_messages.pop_front();
            } else {
                break;
            }
        }

        let mut patterns = Vec::new();

        // Detect question sequences
        if self.recent_messages.len() >= 3 {
            let user_messages: Vec<&Message> = self
                .recent_messages
                .iter()
                .filter(|m| m.role == MessageRole::User)
                .collect();

            if user_messages.len() >= 3 {
                let recent_questions = user_messages
                    .iter()
                    .rev()
                    .take(3)
                    .filter(|m| m.content.contains('?'))
                    .count();

                if recent_questions >= 2 {
                    patterns.push(ConversationPattern {
                        pattern_type: PatternType::QuestionSequence,
                        description: "Consecutive question pattern detected".to_string(),
                        frequency: recent_questions,
                        confidence: 0.8,
                        first_occurrence: SystemTime::now(),
                        last_occurrence: SystemTime::now(),
                        example_messages: user_messages
                            .iter()
                            .rev()
                            .take(3)
                            .map(|m| m.content.to_string())
                            .collect(),
                    });
                }
            }
        }

        // Detect repetitive queries
        if message.role == MessageRole::User {
            let similar_count = self
                .recent_messages
                .iter()
                .filter(|m| m.role == MessageRole::User && m.id != message.id)
                .filter(|m| {
                    self.calculate_similarity(m.content.to_text(), message.content.to_text()) > 0.7
                })
                .count();

            if similar_count >= self.config.min_pattern_frequency {
                patterns.push(ConversationPattern {
                    pattern_type: PatternType::RepetitiveQueries,
                    description: "Repetitive query pattern detected".to_string(),
                    frequency: similar_count + 1,
                    confidence: 0.9,
                    first_occurrence: SystemTime::now(),
                    last_occurrence: SystemTime::now(),
                    example_messages: vec![message.content.to_string()],
                });
            }
        }

        Ok(patterns)
    }

    fn calculate_similarity(&self, text1: &str, text2: &str) -> f32 {
        // Simple Jaccard similarity based on words
        let words1: std::collections::HashSet<&str> = text1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = text2.split_whitespace().collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }
}

/// Anomaly detection component
struct AnomalyDetector {
    config: AnalyticsConfig,
    response_time_history: VecDeque<Duration>,
    quality_history: VecDeque<f32>,
}

impl AnomalyDetector {
    fn new(config: &AnalyticsConfig) -> Self {
        Self {
            config: config.clone(),
            response_time_history: VecDeque::new(),
            quality_history: VecDeque::new(),
        }
    }

    async fn analyze_message(
        &mut self,
        message: &Message,
        analytics: &ConversationAnalytics,
        response_time: Option<Duration>,
    ) -> Result<Vec<ConversationAnomaly>> {
        let mut anomalies = Vec::new();

        // Analyze response time anomalies
        if let Some(rt) = response_time {
            self.response_time_history.push_back(rt);
            if self.response_time_history.len() > 20 {
                self.response_time_history.pop_front();
            }

            if self.response_time_history.len() >= 5 {
                let mean = self.calculate_mean_duration(&self.response_time_history);
                let std_dev = self.calculate_std_dev_duration(&self.response_time_history, mean);

                if rt
                    > mean
                        + Duration::from_millis(
                            (std_dev.as_millis() as f32 * self.config.anomaly_detection_threshold)
                                as u64,
                        )
                {
                    anomalies.push(ConversationAnomaly {
                        anomaly_type: AnomalyType::UnusualResponseTime,
                        description: format!(
                            "Response time {:.2}s is significantly higher than average {:.2}s",
                            rt.as_secs_f64(),
                            mean.as_secs_f64()
                        ),
                        severity: if rt > mean * 3 {
                            AnomalySeverity::High
                        } else {
                            AnomalySeverity::Medium
                        },
                        detected_at: SystemTime::now(),
                        message_context: vec![message.content.to_string()],
                        suggested_action: Some(
                            "Check system resources and query complexity".to_string(),
                        ),
                    });
                }
            }
        }

        // Analyze quality anomalies
        if let Some(ref metadata) = message.metadata {
            if let Some(confidence) = metadata.confidence {
                self.quality_history.push_back(confidence as f32);
                if self.quality_history.len() > 10 {
                    self.quality_history.pop_front();
                }

                if confidence < 0.3 {
                    anomalies.push(ConversationAnomaly {
                        anomaly_type: AnomalyType::LowQualityResponses,
                        description: format!(
                            "Response confidence {:.2} is below acceptable threshold",
                            confidence
                        ),
                        severity: AnomalySeverity::Medium,
                        detected_at: SystemTime::now(),
                        message_context: vec![message.content.to_string()],
                        suggested_action: Some(
                            "Review retrieval results and LLM outputs".to_string(),
                        ),
                    });
                }
            }
        }

        // Analyze error rate anomalies
        if analytics.conversation_quality.error_rate > 0.5 && analytics.message_count > 5 {
            anomalies.push(ConversationAnomaly {
                anomaly_type: AnomalyType::HighErrorRate,
                description: format!(
                    "Error rate {:.2}% is above acceptable threshold",
                    analytics.conversation_quality.error_rate * 100.0
                ),
                severity: AnomalySeverity::High,
                detected_at: SystemTime::now(),
                message_context: vec![],
                suggested_action: Some(
                    "Investigate system errors and data quality issues".to_string(),
                ),
            });
        }

        Ok(anomalies)
    }

    fn calculate_mean_duration(&self, durations: &VecDeque<Duration>) -> Duration {
        let total: u128 = durations.iter().map(|d| d.as_millis()).sum();
        Duration::from_millis((total / durations.len() as u128) as u64)
    }

    fn calculate_std_dev_duration(
        &self,
        durations: &VecDeque<Duration>,
        mean: Duration,
    ) -> Duration {
        let variance: f64 = durations
            .iter()
            .map(|d| {
                let diff = d.as_millis() as f64 - mean.as_millis() as f64;
                diff * diff
            })
            .sum::<f64>()
            / durations.len() as f64;

        Duration::from_millis(variance.sqrt() as u64)
    }
}

/// Quality analysis component
struct QualityAnalyzer {
    _config: AnalyticsConfig,
}

impl QualityAnalyzer {
    fn new(config: &AnalyticsConfig) -> Self {
        Self {
            _config: config.clone(),
        }
    }

    async fn update_quality_metrics(
        &self,
        message: &Message,
        analytics: &mut ConversationAnalytics,
    ) -> Result<()> {
        // Update quality metrics based on message metadata
        if let Some(ref metadata) = message.metadata {
            let mut quality = &mut analytics.conversation_quality;

            // Update response accuracy
            if let Some(confidence) = metadata.confidence {
                quality.response_accuracy = (quality.response_accuracy
                    * (analytics.assistant_message_count - 1) as f32
                    + confidence as f32)
                    / analytics.assistant_message_count as f32;
            }

            // Update relevance score (simplified)
            let context_used = metadata
                .custom_fields
                .get("context_used")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let relevance = if context_used { 0.8 } else { 0.5 };
            quality.relevance_score = (quality.relevance_score
                * (analytics.assistant_message_count - 1) as f32
                + relevance)
                / analytics.assistant_message_count as f32;

            // Update coherence score (based on message length and structure)
            let coherence = self.calculate_coherence_score(message.content.to_text());
            quality.coherence_score = (quality.coherence_score
                * (analytics.assistant_message_count - 1) as f32
                + coherence)
                / analytics.assistant_message_count as f32;
        }

        Ok(())
    }

    async fn calculate_final_quality(
        &self,
        analytics: &ConversationAnalytics,
    ) -> Result<ConversationQuality> {
        let mut quality = analytics.conversation_quality.clone();

        // Calculate completion rate based on conversation patterns
        quality.completion_rate = if analytics.message_count > 1 {
            // Simple heuristic: if conversation ended naturally (even number of messages)
            // and user didn't abandon mid-conversation
            if analytics.message_count % 2 == 0 && analytics.user_message_count > 0 {
                0.8
            } else {
                0.6
            }
        } else {
            0.0
        };

        // Calculate user engagement
        quality.user_engagement = if analytics.message_count > 0 {
            (analytics.user_message_count as f32 / analytics.message_count as f32)
                * (analytics.message_count as f32 / 10.0).min(1.0) // Normalize by expected conversation length
        } else {
            0.0
        };

        Ok(quality)
    }

    fn calculate_coherence_score(&self, content: &str) -> f32 {
        // Simple coherence calculation based on structure
        let word_count = content.split_whitespace().count();
        let sentence_count = content
            .split(&['.', '!', '?'][..])
            .filter(|s| !s.trim().is_empty())
            .count();

        if sentence_count == 0 {
            return 0.5;
        }

        let avg_sentence_length = word_count as f32 / sentence_count as f32;

        // Optimal sentence length is around 15-20 words
        let length_score = if avg_sentence_length >= 10.0 && avg_sentence_length <= 25.0 {
            1.0
        } else if avg_sentence_length < 5.0 || avg_sentence_length > 40.0 {
            0.3
        } else {
            0.7
        };

        // Has proper punctuation
        let punctuation_score = if content.ends_with(&['.', '!', '?'][..]) {
            1.0
        } else {
            0.8
        };

        (length_score + punctuation_score) / 2.0
    }
}

/// Message analytics processor for individual message analysis
pub struct MessageAnalyticsProcessor {
    config: AnalyticsConfig,
    intent_classifier: IntentClassifier,
    sentiment_analyzer: SentimentAnalyzer,
    complexity_analyzer: ComplexityAnalyzer,
    entity_extractor: EntityExtractor,
}

impl MessageAnalyticsProcessor {
    pub fn new(config: AnalyticsConfig) -> Self {
        Self {
            intent_classifier: IntentClassifier::new(&config),
            sentiment_analyzer: SentimentAnalyzer::new(&config),
            complexity_analyzer: ComplexityAnalyzer::new(&config),
            entity_extractor: EntityExtractor::new(&config),
            config,
        }
    }

    /// Process a message and generate comprehensive analytics
    pub async fn process_message(
        &self,
        message: &Message,
        context: Option<&[Message]>,
    ) -> Result<MessageAnalytics> {
        let content_text = message.content.to_text();

        // Parallel processing of different analytics components
        let (intent, sentiment, complexity, entities) = tokio::try_join!(
            self.intent_classifier
                .classify_intent(content_text, context),
            self.sentiment_analyzer.analyze_sentiment(content_text),
            self.complexity_analyzer
                .analyze_complexity(content_text, message),
            self.entity_extractor.extract_entities(content_text)
        )?;

        // Calculate confidence metrics
        let confidence = self
            .calculate_confidence_metrics(&intent, &sentiment, &complexity, message)
            .await?;

        // Calculate success metrics
        let success_metrics = self.calculate_success_metrics(message, &intent).await?;

        // Calculate satisfaction metrics (requires conversation context)
        let satisfaction = self
            .calculate_satisfaction_metrics(message, context)
            .await?;

        Ok(MessageAnalytics {
            intent_classification: Self::convert_to_intent_classification(intent),
            sentiment_analysis: sentiment,
            complexity_score: Self::convert_to_complexity_score(complexity),
            confidence_tracking: Self::convert_to_confidence_tracking(confidence),
            success_metrics,
            quality_assessment: Self::convert_to_quality_assessment(satisfaction),
        })
    }

    async fn calculate_confidence_metrics(
        &self,
        intent: &MessageIntent,
        sentiment: &SentimentAnalysis,
        complexity: &ComplexityMetrics,
        message: &Message,
    ) -> Result<ConfidenceMetrics> {
        let mut uncertainty_factors = Vec::new();

        // Intent confidence factor
        if intent.confidence < 0.7 {
            uncertainty_factors.push(UncertaintyFactor {
                factor_type: "Intent Classification".to_string(),
                impact: 1.0 - intent.confidence,
                description: "Low confidence in intent classification".to_string(),
            });
        }

        // Sentiment confidence factor
        if sentiment.confidence < 0.7 {
            uncertainty_factors.push(UncertaintyFactor {
                factor_type: "Sentiment Analysis".to_string(),
                impact: 1.0 - sentiment.confidence as f64,
                description: "Low confidence in sentiment analysis".to_string(),
            });
        }

        // Complexity factor
        if complexity.overall_complexity > 0.8 {
            uncertainty_factors.push(UncertaintyFactor {
                factor_type: "High Complexity".to_string(),
                impact: complexity.overall_complexity - 0.8,
                description: "Message complexity may affect processing accuracy".to_string(),
            });
        }

        // Metadata confidence factor
        let data_confidence = if let Some(ref metadata) = message.metadata {
            metadata.confidence.unwrap_or(0.5) as f32
        } else {
            0.3
        };

        if data_confidence < 0.5 {
            uncertainty_factors.push(UncertaintyFactor {
                factor_type: "Data Quality".to_string(),
                impact: 0.5 - data_confidence as f64,
                description: "Low confidence in underlying data".to_string(),
            });
        }

        let understanding_confidence = (intent.confidence + sentiment.confidence as f64) / 2.0;
        let response_confidence = data_confidence as f64;
        let reasoning_confidence = 1.0 - (complexity.overall_complexity * 0.3);

        let overall_confidence = (understanding_confidence
            + response_confidence
            + data_confidence as f64
            + reasoning_confidence)
            / 4.0;

        let mut confidence_breakdown = HashMap::new();
        confidence_breakdown.insert("understanding".to_string(), understanding_confidence);
        confidence_breakdown.insert("response".to_string(), response_confidence);
        confidence_breakdown.insert("data".to_string(), data_confidence as f64);
        confidence_breakdown.insert("reasoning".to_string(), reasoning_confidence);

        Ok(ConfidenceMetrics {
            overall_confidence,
            uncertainty_factors,
            confidence_breakdown,
        })
    }

    async fn calculate_success_metrics(
        &self,
        message: &Message,
        intent: &MessageIntent,
    ) -> Result<SuccessMetrics> {
        let processing_time = message
            .metadata
            .as_ref()
            .and_then(|m| m.processing_time_ms)
            .unwrap_or(0);

        let context_retrieved = message
            .metadata
            .as_ref()
            .and_then(|m| m.custom_fields.get("context_used"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let entities_found = false; // TODO: Check extracted entities from context
        let sparql_executed = message
            .metadata
            .as_ref()
            .and_then(|m| m.custom_fields.get("sparql_query"))
            .is_some();

        let query_successful = message
            .metadata
            .as_ref()
            .and_then(|m| m.confidence)
            .map(|score| score > 0.5)
            .unwrap_or(false);

        let response_generated = !message.content.to_text().is_empty();
        let results_returned = response_generated && context_retrieved;

        Ok(SuccessMetrics {
            task_completion_rate: if query_successful { 1.0 } else { 0.0 },
            user_satisfaction_predicted: if response_generated { 0.8 } else { 0.2 },
            response_relevance: if context_retrieved { 0.9 } else { 0.5 },
            response_completeness: if results_returned { 0.9 } else { 0.6 },
            response_accuracy: if sparql_executed { 0.85 } else { 0.7 },
            follow_up_indicators: Vec::new(), // TODO: Implement follow-up analysis
        })
    }

    async fn calculate_satisfaction_metrics(
        &self,
        message: &Message,
        context: Option<&[Message]>,
    ) -> Result<SatisfactionMetrics> {
        let mut follow_up_questions = 0;
        let mut clarification_requests = 0;

        if let Some(context) = context {
            for msg in context.iter().rev().take(5) {
                if msg.role == MessageRole::User {
                    let content = msg.content.to_text().to_lowercase();
                    if content.contains("?") {
                        follow_up_questions += 1;
                    }
                    if content.contains("clarify")
                        || content.contains("explain")
                        || content.contains("what do you mean")
                        || content.contains("i don't understand")
                    {
                        clarification_requests += 1;
                    }
                }
            }
        }

        // Calculate implicit signals
        let engagement_score = if message.role == MessageRole::User {
            let word_count = message.content.to_text().split_whitespace().count();
            // Higher word count indicates higher engagement (up to a point)
            (word_count as f32 / 50.0).min(1.0)
        } else {
            0.5 // Neutral for assistant messages
        };

        let topic_shift_occurred = false; // Would need conversation history analysis
        let repeat_query_pattern = false; // Would need pattern analysis

        let implicit_signals = ImplicitSatisfactionSignals {
            follow_up_questions: context.map(|c| c.len().saturating_sub(1)).unwrap_or(0),
            positive_feedback_indicators: 0, // TODO: Analyze for positive feedback indicators
            task_completion_rate: engagement_score as f64,
            session_continuation: context.map(|c| c.len() > 1).unwrap_or(false),
        };

        let mut satisfaction_breakdown = HashMap::new();
        satisfaction_breakdown.insert("engagement".to_string(), engagement_score as f64);
        satisfaction_breakdown.insert("response_quality".to_string(), 0.7); // Default score
        satisfaction_breakdown.insert("context_relevance".to_string(), 0.8); // Default score

        Ok(SatisfactionMetrics {
            overall_satisfaction: engagement_score as f64,
            satisfaction_breakdown,
            implicit_signals,
            explicit_feedback: None, // Set separately when user provides feedback
        })
    }

    // Conversion functions to bridge between analytics types and message_analytics types
    fn convert_to_intent_classification(intent: MessageIntent) -> IntentClassification {
        use crate::message_analytics::Intent;

        let primary_intent = match intent.intent_type {
            IntentType::Question => Intent::Query,
            IntentType::Request => Intent::Query,
            IntentType::Gratitude => Intent::Feedback,
            IntentType::Aggregation => Intent::Aggregation,
            IntentType::ListQuery => Intent::Query,
            IntentType::Comparison => Intent::Query,
            IntentType::Relationship => Intent::Query,
            IntentType::Definition => Intent::Query,
            IntentType::Complaint => Intent::Feedback,
            IntentType::Clarification => Intent::Query,
            IntentType::Complex => Intent::Query,
            IntentType::Exploration => Intent::Query,
        };

        IntentClassification {
            primary_intent,
            secondary_intents: vec![],
            confidence: intent.confidence as f32,
            intent_scores: HashMap::new(),
        }
    }

    fn convert_to_complexity_score(complexity: ComplexityMetrics) -> ComplexityScore {
        use crate::message_analytics::ComplexityLevel;

        ComplexityScore {
            overall_complexity: if complexity.overall_complexity > 0.8 {
                ComplexityLevel::VeryComplex
            } else if complexity.overall_complexity > 0.6 {
                ComplexityLevel::Complex
            } else if complexity.overall_complexity > 0.4 {
                ComplexityLevel::Moderate
            } else {
                ComplexityLevel::Simple
            },
            complexity_score: complexity.overall_complexity as f32,
            complexity_factors: vec![],
            readability_score: complexity.semantic_complexity as f32,
        }
    }

    fn convert_to_confidence_tracking(confidence: ConfidenceMetrics) -> ConfidenceTracking {
        use crate::message_analytics::ConfidenceTrend;

        ConfidenceTracking {
            overall_confidence: confidence.overall_confidence as f32,
            confidence_components: vec![],
            uncertainty_indicators: vec![],
            confidence_trend: ConfidenceTrend::Stable,
        }
    }

    fn convert_to_quality_assessment(satisfaction: SatisfactionMetrics) -> QualityAssessment {
        QualityAssessment {
            clarity_score: satisfaction.overall_satisfaction as f32,
            helpfulness_score: satisfaction.overall_satisfaction as f32,
            accuracy_score: satisfaction.overall_satisfaction as f32,
            completeness_score: satisfaction.overall_satisfaction as f32,
            relevance_score: satisfaction.overall_satisfaction as f32,
            quality_issues: vec![],
        }
    }
}

/// Intent classification component
struct IntentClassifier {
    _config: AnalyticsConfig,
}

impl IntentClassifier {
    fn new(config: &AnalyticsConfig) -> Self {
        Self {
            _config: config.clone(),
        }
    }

    async fn classify_intent(
        &self,
        content: &str,
        _context: Option<&[Message]>,
    ) -> Result<MessageIntent> {
        let content_lower = content.to_lowercase();

        // Keywords for different intent types
        let question_indicators = ["what", "who", "when", "where", "why", "how", "?"];
        let request_indicators = [
            "please",
            "can you",
            "could you",
            "would you",
            "help me",
            "show me",
        ];
        let gratitude_indicators = ["thank", "thanks", "appreciate", "grateful"];
        let complaint_indicators = [
            "wrong",
            "error",
            "problem",
            "issue",
            "broken",
            "not working",
        ];
        let clarification_indicators =
            ["clarify", "explain", "what do you mean", "don't understand"];
        let comparison_indicators = ["compare", "difference", "vs", "versus", "better", "worse"];
        let list_indicators = ["list", "show all", "give me all", "what are"];
        let aggregation_indicators = ["count", "how many", "total", "sum", "average"];
        let relationship_indicators = [
            "relationship",
            "connected",
            "related",
            "link",
            "association",
        ];

        // Classify primary intent
        let primary_intent = if question_indicators
            .iter()
            .any(|&ind| content_lower.contains(ind))
        {
            if aggregation_indicators
                .iter()
                .any(|&ind| content_lower.contains(ind))
            {
                IntentType::Aggregation
            } else if list_indicators
                .iter()
                .any(|&ind| content_lower.contains(ind))
            {
                IntentType::ListQuery
            } else if comparison_indicators
                .iter()
                .any(|&ind| content_lower.contains(ind))
            {
                IntentType::Comparison
            } else if relationship_indicators
                .iter()
                .any(|&ind| content_lower.contains(ind))
            {
                IntentType::Relationship
            } else if content_lower.contains("define") || content_lower.contains("definition") {
                IntentType::Definition
            } else {
                IntentType::Question
            }
        } else if request_indicators
            .iter()
            .any(|&ind| content_lower.contains(ind))
        {
            IntentType::Request
        } else if gratitude_indicators
            .iter()
            .any(|&ind| content_lower.contains(ind))
        {
            IntentType::Gratitude
        } else if complaint_indicators
            .iter()
            .any(|&ind| content_lower.contains(ind))
        {
            IntentType::Complaint
        } else if clarification_indicators
            .iter()
            .any(|&ind| content_lower.contains(ind))
        {
            IntentType::Clarification
        } else if content.split_whitespace().count() > 50 || content.matches(" and ").count() > 2 {
            IntentType::Complex
        } else {
            IntentType::Exploration
        };

        // Extract keywords (simple approach)
        let keywords: Vec<String> = content_lower
            .split_whitespace()
            .filter(|word| word.len() > 3)
            .filter(|word| {
                ![
                    "what", "that", "this", "with", "from", "they", "them", "were", "said", "each",
                    "which", "their", "time", "about",
                ]
                .contains(word)
            })
            .take(10)
            .map(|s| s.to_string())
            .collect();

        // Calculate confidence based on keyword matches and patterns
        let confidence = self.calculate_intent_confidence(&primary_intent, &content_lower);

        Ok(MessageIntent {
            intent_type: primary_intent,
            confidence: confidence as f64,
            reasoning: format!("Detected intent based on keywords: {:?}", keywords),
        })
    }

    fn calculate_intent_confidence(&self, intent: &IntentType, content: &str) -> f32 {
        // Base confidence
        let mut confidence: f32 = 0.6;

        // Boost confidence based on strong indicators
        match intent {
            IntentType::Question => {
                if content.contains("?") {
                    confidence += 0.3;
                }
                if content.starts_with("what") || content.starts_with("how") {
                    confidence += 0.2;
                }
            }
            IntentType::Request => {
                if content.contains("please") {
                    confidence += 0.2;
                }
                if content.contains("can you") {
                    confidence += 0.3;
                }
            }
            IntentType::Gratitude => {
                if content.contains("thank") {
                    confidence += 0.4;
                }
            }
            IntentType::Aggregation => {
                if content.contains("how many") || content.contains("count") {
                    confidence += 0.3;
                }
            }
            IntentType::ListQuery => {
                if content.contains("list") || content.contains("show all") {
                    confidence += 0.3;
                }
            }
            _ => {}
        }

        confidence.min(1.0)
    }
}

/// Sentiment analysis component
struct SentimentAnalyzer {
    _config: AnalyticsConfig,
}

impl SentimentAnalyzer {
    fn new(config: &AnalyticsConfig) -> Self {
        Self {
            _config: config.clone(),
        }
    }

    async fn analyze_sentiment(&self, content: &str) -> Result<SentimentAnalysis> {
        let content_lower = content.to_lowercase();

        // Simple lexicon-based sentiment analysis
        let positive_words = [
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "fantastic",
            "perfect",
            "love",
            "like",
            "enjoy",
            "happy",
            "pleased",
            "satisfied",
            "thank",
            "helpful",
        ];

        let negative_words = [
            "bad",
            "terrible",
            "awful",
            "horrible",
            "hate",
            "dislike",
            "angry",
            "frustrated",
            "confused",
            "problem",
            "issue",
            "error",
            "wrong",
            "broken",
            "useless",
            "disappointed",
        ];

        let neutral_words = ["okay", "fine", "alright", "maybe", "perhaps", "possibly"];

        let words: Vec<&str> = content_lower.split_whitespace().collect();
        let mut positive_score = 0;
        let mut negative_score = 0;
        let mut neutral_score = 0;

        for word in &words {
            if positive_words.contains(word) {
                positive_score += 1;
            } else if negative_words.contains(word) {
                negative_score += 1;
            } else if neutral_words.contains(word) {
                neutral_score += 1;
            }
        }

        let (polarity, confidence) =
            if positive_score > negative_score && positive_score > neutral_score {
                (
                    crate::message_analytics::Sentiment::Positive,
                    positive_score as f32 / words.len() as f32,
                )
            } else if negative_score > positive_score && negative_score > neutral_score {
                (
                    crate::message_analytics::Sentiment::Negative,
                    negative_score as f32 / words.len() as f32,
                )
            } else {
                (crate::message_analytics::Sentiment::Neutral, 0.5)
            };

        // Simple emotion detection based on keywords
        let mut emotions = Vec::new();
        if content_lower.contains("happy") || content_lower.contains("joy") {
            emotions.push(EmotionScore {
                emotion: "joy".to_string(),
                intensity: 0.8,
                confidence: 0.7,
            });
        }
        if content_lower.contains("angry") || content_lower.contains("mad") {
            emotions.push(EmotionScore {
                emotion: "anger".to_string(),
                intensity: 0.8,
                confidence: 0.7,
            });
        }
        if content_lower.contains("confused") || content_lower.contains("don't understand") {
            emotions.push(EmotionScore {
                emotion: "confusion".to_string(),
                intensity: 0.7,
                confidence: 0.6,
            });
        }
        if content_lower.contains("frustrated") || content_lower.contains("annoyed") {
            emotions.push(EmotionScore {
                emotion: "frustration".to_string(),
                intensity: 0.7,
                confidence: 0.6,
            });
        }
        if content_lower.contains("satisfied") || content_lower.contains("pleased") {
            emotions.push(EmotionScore {
                emotion: "satisfaction".to_string(),
                intensity: 0.8,
                confidence: 0.7,
            });
        }

        // Calculate subjectivity (0 = objective, 1 = subjective)
        let subjective_indicators = ["think", "feel", "believe", "opinion", "personally", "i"];
        let subjectivity = subjective_indicators
            .iter()
            .filter(|&&word| content_lower.contains(word))
            .count() as f32
            / words.len() as f32;

        // Convert emotions to EmotionIndicator format
        let emotion_indicators: Vec<crate::message_analytics::EmotionIndicator> = emotions
            .into_iter()
            .map(|emotion_score| crate::message_analytics::EmotionIndicator {
                emotion: match emotion_score.emotion.as_str() {
                    "joy" => crate::message_analytics::Emotion::Excitement,
                    "anger" => crate::message_analytics::Emotion::Frustration,
                    "confusion" => crate::message_analytics::Emotion::Confusion,
                    "frustration" => crate::message_analytics::Emotion::Frustration,
                    "satisfaction" => crate::message_analytics::Emotion::Satisfaction,
                    _ => crate::message_analytics::Emotion::Curiosity,
                },
                intensity: emotion_score.intensity as f32,
                indicators: vec![emotion_score.emotion.clone()], // Use emotion name as indicator
            })
            .collect();

        // Convert sentiment polarity to score
        let sentiment_score = match polarity {
            crate::message_analytics::Sentiment::Positive => 0.7,
            crate::message_analytics::Sentiment::Negative => -0.7,
            crate::message_analytics::Sentiment::Neutral => 0.0,
            crate::message_analytics::Sentiment::Mixed => 0.0,
        };

        Ok(SentimentAnalysis {
            overall_sentiment: polarity,
            sentiment_score,
            emotion_indicators,
            confidence: confidence.min(1.0).max(0.3),
        })
    }
}

/// Complexity analysis component
struct ComplexityAnalyzer {
    _config: AnalyticsConfig,
}

impl ComplexityAnalyzer {
    fn new(config: &AnalyticsConfig) -> Self {
        Self {
            _config: config.clone(),
        }
    }

    async fn analyze_complexity(
        &self,
        content: &str,
        message: &Message,
    ) -> Result<ComplexityMetrics> {
        let words: Vec<&str> = content.split_whitespace().collect();
        let sentences: Vec<&str> = content
            .split(&['.', '!', '?'][..])
            .filter(|s| !s.trim().is_empty())
            .collect();

        // Linguistic complexity
        let avg_word_length =
            words.iter().map(|w| w.len()).sum::<usize>() as f32 / words.len() as f32;
        let avg_sentence_length = if !sentences.is_empty() {
            words.len() as f32 / sentences.len() as f32
        } else {
            0.0
        };

        let linguistic_complexity = ((avg_word_length - 4.0) / 6.0).max(0.0).min(1.0)
            + ((avg_sentence_length - 15.0) / 20.0).max(0.0).min(1.0);

        // Conceptual complexity (based on domain-specific terms)
        let technical_terms = [
            "sparql",
            "rdf",
            "ontology",
            "semantic",
            "triple",
            "graph",
            "query",
            "database",
            "algorithm",
            "machine learning",
            "artificial intelligence",
            "neural network",
        ];
        let technical_count = words
            .iter()
            .filter(|word| technical_terms.contains(&word.to_lowercase().as_str()))
            .count();
        let conceptual_complexity = (technical_count as f32 / words.len() as f32 * 10.0).min(1.0);

        // Structural complexity (nesting, conjunctions, etc.)
        let conjunction_count = ["and", "or", "but", "however", "although"]
            .iter()
            .map(|word| content.matches(word).count())
            .sum::<usize>();
        let question_count = content.matches('?').count();
        let structural_complexity =
            ((conjunction_count + question_count) as f32 / sentences.len() as f32).min(1.0);

        // Domain complexity (based on metadata)
        let domain_complexity = if let Some(ref metadata) = message.metadata {
            if metadata.custom_fields.contains_key("sparql_query") {
                0.8 // SPARQL queries add complexity
            } else if metadata
                .custom_fields
                .get("entities_extracted")
                .and_then(|v| v.as_array())
                .map(|e| e.len())
                .unwrap_or(0)
                > 3
            {
                0.6 // Multiple entities add complexity
            } else {
                0.4
            }
        } else {
            0.3
        };

        let overall_score = (linguistic_complexity
            + conceptual_complexity
            + structural_complexity
            + domain_complexity)
            / 4.0;

        let mut factors = Vec::new();
        if linguistic_complexity > 0.7 {
            factors.push(ComplexityFactor {
                factor_type: crate::message_analytics::ComplexityFactorType::VocabularyComplexity,
                contribution: linguistic_complexity,
                description: "Complex sentence structure and vocabulary".to_string(),
            });
        }
        if conceptual_complexity > 0.5 {
            factors.push(ComplexityFactor {
                factor_type: crate::message_analytics::ComplexityFactorType::ConceptualComplexity,
                contribution: conceptual_complexity,
                description: "Technical or domain-specific concepts".to_string(),
            });
        }

        Ok(ComplexityMetrics {
            linguistic_complexity: linguistic_complexity as f64,
            semantic_complexity: conceptual_complexity as f64,
            context_dependency: domain_complexity as f64,
            reasoning_depth: structural_complexity as f64,
            overall_complexity: overall_score as f64,
        })
    }
}

/// Entity extraction component
struct EntityExtractor {
    _config: AnalyticsConfig,
}

impl EntityExtractor {
    fn new(config: &AnalyticsConfig) -> Self {
        Self {
            _config: config.clone(),
        }
    }

    async fn extract_entities(&self, content: &str) -> Result<Vec<ExtractedEntity>> {
        let mut entities = Vec::new();
        let words: Vec<&str> = content.split_whitespace().collect();

        // Simple pattern-based entity extraction
        for (i, word) in words.iter().enumerate() {
            let word_lower = word.to_lowercase();

            // Numbers (map to Other since Number doesn't exist)
            if word.chars().all(|c| c.is_ascii_digit()) {
                entities.push(ExtractedEntity {
                    text: word.to_string(),
                    entity_type: EntityType::Other,
                    confidence: 0.9,
                    iri: None,
                    aliases: vec![],
                });
            }

            // Dates (simple patterns) - map to Event since Date doesn't exist
            if word.contains("/") && word.len() >= 8 {
                entities.push(ExtractedEntity {
                    text: word.to_string(),
                    entity_type: EntityType::Event,
                    confidence: 0.7,
                    iri: None,
                    aliases: vec![],
                });
            }

            // Proper nouns (capitalized words not at sentence start)
            if word
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false)
                && i > 0
                && !words[i - 1].ends_with(&['.', '!', '?'][..])
            {
                let entity_type = if word_lower.ends_with("corp")
                    || word_lower.ends_with("inc")
                    || word_lower.ends_with("ltd")
                    || word_lower.contains("company")
                {
                    EntityType::Organization
                } else {
                    EntityType::Person
                };

                entities.push(ExtractedEntity {
                    text: word.to_string(),
                    entity_type,
                    confidence: 0.6,
                    iri: None,
                    aliases: vec![],
                });
            }
        }

        Ok(entities)
    }
}
