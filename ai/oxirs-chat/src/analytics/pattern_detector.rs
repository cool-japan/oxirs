//! Pattern detection for conversation analytics
//!
//! This module contains the logic for detecting patterns in conversations,
//! including repeated questions, topic progressions, sentiment shifts, and other
//! conversational patterns.

use anyhow::Result;
use std::collections::{HashMap, VecDeque};
use tracing::debug;

use crate::{
    analytics::types::*,
    session_manager::{TopicTransition, TransitionType},
    Message, MessageRole,
};

/// Pattern detection component for conversation analytics
pub struct PatternDetector {
    pub config: AnalyticsConfig,
    pub message_history: VecDeque<Message>,
    pub topic_history: Vec<TopicTransition>,
    pub pattern_cache: HashMap<String, ConversationPattern>,
}

impl PatternDetector {
    pub fn new(config: AnalyticsConfig) -> Self {
        Self {
            config,
            message_history: VecDeque::new(),
            topic_history: Vec::new(),
            pattern_cache: HashMap::new(),
        }
    }

    pub fn add_message(&mut self, message: Message) {
        self.message_history.push_back(message);
        if self.message_history.len() > 100 {
            self.message_history.pop_front();
        }
    }

    pub fn add_topic_transition(&mut self, transition: TopicTransition) {
        self.topic_history.push(transition);
        if self.topic_history.len() > 50 {
            self.topic_history.remove(0);
        }
    }

    pub async fn detect_patterns(
        &mut self,
        analytics: &ConversationAnalytics,
    ) -> Result<Vec<ConversationPattern>> {
        let mut patterns = Vec::new();

        // Detect repeated questions
        if let Some(pattern) = self.detect_repeated_questions().await? {
            patterns.push(pattern);
        }

        // Detect topic progression patterns
        if let Some(pattern) = self.detect_topic_progression().await? {
            patterns.push(pattern);
        }

        // Detect sentiment shift patterns
        if let Some(pattern) = self.detect_sentiment_shifts(analytics).await? {
            patterns.push(pattern);
        }

        // Detect complexity escalation patterns
        if let Some(pattern) = self.detect_complexity_escalation(analytics).await? {
            patterns.push(pattern);
        }

        // Detect error cascade patterns
        if let Some(pattern) = self.detect_error_cascades(analytics).await? {
            patterns.push(pattern);
        }

        // Detect success patterns
        if let Some(pattern) = self.detect_success_patterns(analytics).await? {
            patterns.push(pattern);
        }

        // Detect engagement patterns
        if let Some(pattern) = self.detect_engagement_patterns(analytics).await? {
            patterns.push(pattern);
        }

        // Detect learning patterns
        if let Some(pattern) = self.detect_learning_patterns().await? {
            patterns.push(pattern);
        }

        // Detect frustration patterns
        if let Some(pattern) = self.detect_frustration_patterns(analytics).await? {
            patterns.push(pattern);
        }

        // Detect exploration patterns
        if let Some(pattern) = self.detect_exploration_patterns().await? {
            patterns.push(pattern);
        }

        // Cache patterns
        for pattern in &patterns {
            self.pattern_cache.insert(
                format!("{:?}_{}", pattern.pattern_type, pattern.confidence),
                pattern.clone(),
            );
        }

        debug!("Detected {} patterns", patterns.len());
        Ok(patterns)
    }

    async fn detect_repeated_questions(&self) -> Result<Option<ConversationPattern>> {
        let mut question_counts = HashMap::new();
        let mut repeated_questions = Vec::new();

        for message in &self.message_history {
            if message.role == MessageRole::User {
                let content = message.content.to_string().to_lowercase();
                if content.contains('?') {
                    let normalized = content.trim_end_matches('?').trim();
                    let count = question_counts.entry(normalized.to_string()).or_insert(0);
                    *count += 1;

                    if *count >= self.config.min_pattern_frequency {
                        repeated_questions.push(content.clone());
                    }
                }
            }
        }

        if !repeated_questions.is_empty() {
            return Ok(Some(ConversationPattern {
                pattern_type: PatternType::RepeatedQuestion,
                description: format!("User repeated {} questions", repeated_questions.len()),
                confidence: 0.8,
                frequency: repeated_questions.len(),
                examples: repeated_questions.clone(),
                insights: vec![
                    "User may not be getting satisfactory answers".to_string(),
                    "Consider improving response quality or asking clarifying questions"
                        .to_string(),
                ],
            }));
        }

        Ok(None)
    }

    async fn detect_topic_progression(&self) -> Result<Option<ConversationPattern>> {
        if self.topic_history.len() < 3 {
            return Ok(None);
        }

        let mut progression_score = 0.0;
        let mut natural_transitions = 0;

        for window in self.topic_history.windows(2) {
            if let [_prev, curr] = window {
                // Check if transition is natural
                if curr.transition_type == TransitionType::TopicShift {
                    natural_transitions += 1;
                    progression_score += 0.3;
                }

                // Check if topics are related
                if curr.confidence > 0.7 {
                    progression_score += 0.2;
                }
            }
        }

        if natural_transitions >= 2 {
            return Ok(Some(ConversationPattern {
                pattern_type: PatternType::TopicProgression,
                description: format!(
                    "Natural topic progression with {natural_transitions} transitions"
                ),
                confidence: progression_score / self.topic_history.len() as f64,
                frequency: natural_transitions,
                examples: self
                    .topic_history
                    .iter()
                    .map(|t| t.to_topics.join(", "))
                    .collect(),
                insights: vec![
                    "User is naturally exploring related topics".to_string(),
                    "Conversation flow is coherent and logical".to_string(),
                ],
            }));
        }

        Ok(None)
    }

    async fn detect_sentiment_shifts(
        &self,
        analytics: &ConversationAnalytics,
    ) -> Result<Option<ConversationPattern>> {
        if analytics.sentiment_progression.len() < 3 {
            return Ok(None);
        }

        let mut shifts = Vec::new();
        for window in analytics.sentiment_progression.windows(2) {
            if let [prev, curr] = window {
                let intensity_change = (curr.intensity - prev.intensity).abs();
                if intensity_change > 0.3 {
                    shifts.push(format!("{} -> {}", prev.emotion, curr.emotion));
                }
            }
        }

        if shifts.len() >= 2 {
            return Ok(Some(ConversationPattern {
                pattern_type: PatternType::SentimentShift,
                description: format!("Significant sentiment shifts detected: {}", shifts.len()),
                confidence: 0.7,
                frequency: shifts.len(),
                examples: shifts.clone(),
                insights: vec![
                    "User emotional state is changing during conversation".to_string(),
                    "Monitor for signs of frustration or satisfaction".to_string(),
                ],
            }));
        }

        Ok(None)
    }

    async fn detect_complexity_escalation(
        &self,
        analytics: &ConversationAnalytics,
    ) -> Result<Option<ConversationPattern>> {
        if analytics.complexity_progression.len() < 3 {
            return Ok(None);
        }

        let mut escalation_count = 0;
        for window in analytics.complexity_progression.windows(2) {
            if let [prev, curr] = window {
                if curr.overall_complexity > prev.overall_complexity + 0.2 {
                    escalation_count += 1;
                }
            }
        }

        if escalation_count >= 2 {
            return Ok(Some(ConversationPattern {
                pattern_type: PatternType::ComplexityEscalation,
                description: format!("Complexity escalation detected in {escalation_count} steps"),
                confidence: 0.6,
                frequency: escalation_count,
                examples: vec![format!("Complexity increased {} times", escalation_count)],
                insights: vec![
                    "User questions are becoming more complex".to_string(),
                    "Consider providing more detailed explanations".to_string(),
                ],
            }));
        }

        Ok(None)
    }

    async fn detect_error_cascades(
        &self,
        analytics: &ConversationAnalytics,
    ) -> Result<Option<ConversationPattern>> {
        if analytics.conversation_quality.error_rate > 0.3 && analytics.message_count > 5 {
            return Ok(Some(ConversationPattern {
                pattern_type: PatternType::ErrorCascade,
                description: format!(
                    "High error rate: {:.2}%",
                    analytics.conversation_quality.error_rate * 100.0
                ),
                confidence: 0.9,
                frequency: (analytics.conversation_quality.error_rate
                    * analytics.message_count as f64) as usize,
                examples: vec!["Multiple errors in sequence".to_string()],
                insights: vec![
                    "System is experiencing cascading errors".to_string(),
                    "Review error handling and data quality".to_string(),
                ],
            }));
        }

        Ok(None)
    }

    async fn detect_success_patterns(
        &self,
        analytics: &ConversationAnalytics,
    ) -> Result<Option<ConversationPattern>> {
        if analytics.user_satisfaction.overall_satisfaction > 0.8 && analytics.message_count > 3 {
            return Ok(Some(ConversationPattern {
                pattern_type: PatternType::SuccessPattern,
                description: format!(
                    "High satisfaction: {:.2}",
                    analytics.user_satisfaction.overall_satisfaction
                ),
                confidence: 0.8,
                frequency: 1,
                examples: vec!["Positive user feedback indicators".to_string()],
                insights: vec![
                    "User is satisfied with the conversation".to_string(),
                    "Maintain current approach and quality".to_string(),
                ],
            }));
        }

        Ok(None)
    }

    async fn detect_engagement_patterns(
        &self,
        analytics: &ConversationAnalytics,
    ) -> Result<Option<ConversationPattern>> {
        if analytics.conversation_quality.engagement_score > 0.7 && analytics.message_count > 5 {
            return Ok(Some(ConversationPattern {
                pattern_type: PatternType::EngagementPattern,
                description: format!(
                    "High engagement: {:.2}",
                    analytics.conversation_quality.engagement_score
                ),
                confidence: 0.7,
                frequency: analytics.message_count,
                examples: vec!["Active participation and follow-up questions".to_string()],
                insights: vec![
                    "User is actively engaged in the conversation".to_string(),
                    "Continue encouraging exploration and questions".to_string(),
                ],
            }));
        }

        Ok(None)
    }

    async fn detect_learning_patterns(&self) -> Result<Option<ConversationPattern>> {
        let mut learning_indicators = 0;
        let mut examples = Vec::new();

        for message in &self.message_history {
            if message.role == MessageRole::User {
                let content = message.content.to_string().to_lowercase();
                if content.contains("i understand")
                    || content.contains("i see")
                    || content.contains("that makes sense")
                {
                    learning_indicators += 1;
                    examples.push(content.clone());
                }
            }
        }

        if learning_indicators >= 2 {
            return Ok(Some(ConversationPattern {
                pattern_type: PatternType::LearningPattern,
                description: format!("Learning indicators detected: {learning_indicators}"),
                confidence: 0.6,
                frequency: learning_indicators,
                examples,
                insights: vec![
                    "User is demonstrating learning and understanding".to_string(),
                    "Educational approach is effective".to_string(),
                ],
            }));
        }

        Ok(None)
    }

    async fn detect_frustration_patterns(
        &self,
        analytics: &ConversationAnalytics,
    ) -> Result<Option<ConversationPattern>> {
        let mut frustration_score = 0.0;
        let mut examples = Vec::new();

        // Check for negative sentiment
        for emotion in &analytics.sentiment_progression {
            if emotion.emotion == "frustration" || emotion.emotion == "anger" {
                frustration_score += emotion.intensity;
                examples.push(format!("{}: {:.2}", emotion.emotion, emotion.intensity));
            }
        }

        // Check for repeated questions
        if let Some(_pattern) = self.pattern_cache.get("RepeatedQuestion_0.8") {
            frustration_score += 0.3;
            examples.push("Repeated questions detected".to_string());
        }

        if frustration_score > 0.5 {
            return Ok(Some(ConversationPattern {
                pattern_type: PatternType::FrustrationPattern,
                description: format!("Frustration indicators: {frustration_score:.2}"),
                confidence: 0.7,
                frequency: examples.len(),
                examples,
                insights: vec![
                    "User may be experiencing frustration".to_string(),
                    "Consider providing clearer explanations or alternative approaches".to_string(),
                ],
            }));
        }

        Ok(None)
    }

    async fn detect_exploration_patterns(&self) -> Result<Option<ConversationPattern>> {
        let mut exploration_indicators = 0;
        let mut examples = Vec::new();

        for message in &self.message_history {
            if message.role == MessageRole::User {
                let content = message.content.to_string().to_lowercase();
                if content.contains("what about")
                    || content.contains("how about")
                    || content.contains("tell me more")
                {
                    exploration_indicators += 1;
                    examples.push(content.clone());
                }
            }
        }

        if exploration_indicators >= 2 {
            return Ok(Some(ConversationPattern {
                pattern_type: PatternType::ExplorationPattern,
                description: format!("Exploration indicators: {exploration_indicators}"),
                confidence: 0.6,
                frequency: exploration_indicators,
                examples,
                insights: vec![
                    "User is actively exploring topics".to_string(),
                    "Encourage deeper exploration and related questions".to_string(),
                ],
            }));
        }

        Ok(None)
    }
}
