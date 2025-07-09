//! Anomaly detection for conversation analytics
//!
//! This module contains the logic for detecting anomalies in conversations,
//! including unusual response times, low quality responses, high error rates,
//! and other conversational anomalies.

use anyhow::Result;
use std::{
    collections::VecDeque,
    time::{Duration, SystemTime},
};
use tracing::debug;

use crate::{analytics::types::*, Message};

/// Anomaly detection component for conversation analytics
pub struct AnomalyDetector {
    pub config: AnalyticsConfig,
    pub response_time_history: VecDeque<f32>,
    pub quality_history: VecDeque<f32>,
    pub error_history: VecDeque<bool>,
    pub sentiment_history: VecDeque<f32>,
    pub complexity_history: VecDeque<f32>,
}

impl AnomalyDetector {
    pub fn new(config: AnalyticsConfig) -> Self {
        Self {
            config,
            response_time_history: VecDeque::new(),
            quality_history: VecDeque::new(),
            error_history: VecDeque::new(),
            sentiment_history: VecDeque::new(),
            complexity_history: VecDeque::new(),
        }
    }

    pub fn add_response_time(&mut self, response_time: Duration) {
        self.response_time_history
            .push_back(response_time.as_secs_f32());
        if self.response_time_history.len() > 20 {
            self.response_time_history.pop_front();
        }
    }

    pub fn add_quality_score(&mut self, quality: f32) {
        self.quality_history.push_back(quality);
        if self.quality_history.len() > 20 {
            self.quality_history.pop_front();
        }
    }

    pub fn add_error(&mut self, is_error: bool) {
        self.error_history.push_back(is_error);
        if self.error_history.len() > 20 {
            self.error_history.pop_front();
        }
    }

    pub fn add_sentiment_score(&mut self, sentiment: f32) {
        self.sentiment_history.push_back(sentiment);
        if self.sentiment_history.len() > 20 {
            self.sentiment_history.pop_front();
        }
    }

    pub fn add_complexity_score(&mut self, complexity: f32) {
        self.complexity_history.push_back(complexity);
        if self.complexity_history.len() > 20 {
            self.complexity_history.pop_front();
        }
    }

    pub async fn detect_anomalies(
        &mut self,
        analytics: &ConversationAnalytics,
        message: &Message,
    ) -> Result<Vec<ConversationAnomaly>> {
        let mut anomalies = Vec::new();

        // Detect response time anomalies
        if let Some(anomaly) = self.detect_response_time_anomalies(message).await? {
            anomalies.push(anomaly);
        }

        // Detect quality anomalies
        if let Some(anomaly) = self.detect_quality_anomalies(message).await? {
            anomalies.push(anomaly);
        }

        // Detect error rate anomalies
        if let Some(anomaly) = self.detect_error_rate_anomalies(analytics).await? {
            anomalies.push(anomaly);
        }

        // Detect sentiment anomalies
        if let Some(anomaly) = self.detect_sentiment_anomalies(analytics).await? {
            anomalies.push(anomaly);
        }

        // Detect complexity anomalies
        if let Some(anomaly) = self.detect_complexity_anomalies(analytics).await? {
            anomalies.push(anomaly);
        }

        // Detect engagement anomalies
        if let Some(anomaly) = self.detect_engagement_anomalies(analytics).await? {
            anomalies.push(anomaly);
        }

        // Detect context loss anomalies
        if let Some(anomaly) = self.detect_context_loss_anomalies(analytics).await? {
            anomalies.push(anomaly);
        }

        // Detect topic divergence anomalies
        if let Some(anomaly) = self.detect_topic_divergence_anomalies(analytics).await? {
            anomalies.push(anomaly);
        }

        // Detect confidence collapse anomalies
        if let Some(anomaly) = self.detect_confidence_collapse_anomalies(analytics).await? {
            anomalies.push(anomaly);
        }

        // Detect repeated error anomalies
        if let Some(anomaly) = self.detect_repeated_error_anomalies().await? {
            anomalies.push(anomaly);
        }

        debug!("Detected {} anomalies", anomalies.len());
        Ok(anomalies)
    }

    async fn detect_response_time_anomalies(
        &mut self,
        message: &Message,
    ) -> Result<Option<ConversationAnomaly>> {
        if let Some(ref metadata) = message.metadata {
            if let Some(response_time_ms) = metadata.processing_time_ms {
                let response_time_secs = response_time_ms as f32 / 1000.0;
                let response_time = std::time::Duration::from_millis(response_time_ms);
                self.add_response_time(response_time);

                if self.response_time_history.len() >= 5 {
                    let mean = self.response_time_history.iter().sum::<f32>()
                        / self.response_time_history.len() as f32;
                    let variance = self
                        .response_time_history
                        .iter()
                        .map(|x| (x - mean).powi(2))
                        .sum::<f32>()
                        / self.response_time_history.len() as f32;
                    let std_dev = variance.sqrt();

                    if response_time_secs
                        > mean + (self.config.anomaly_detection_threshold * std_dev)
                    {
                        return Ok(Some(ConversationAnomaly {
                            anomaly_type: AnomalyType::UnusualResponseTime,
                            description: format!(
                                "Response time {:.2}s is {:.2} standard deviations above average",
                                response_time_secs,
                                (response_time_secs - mean) / std_dev
                            ),
                            severity: if response_time_secs > mean + (3.0 * std_dev) {
                                AnomalySeverity::High
                            } else {
                                AnomalySeverity::Medium
                            },
                            detected_at: SystemTime::now(),
                            message_context: vec![message.content.to_string()],
                            suggested_action: Some(
                                "Check system performance and optimize slow queries".to_string(),
                            ),
                        }));
                    }
                }
            }
        }

        Ok(None)
    }

    async fn detect_quality_anomalies(
        &mut self,
        message: &Message,
    ) -> Result<Option<ConversationAnomaly>> {
        if let Some(ref metadata) = message.metadata {
            if let Some(confidence) = metadata.confidence {
                self.add_quality_score(confidence as f32);

                if confidence < 0.3 {
                    return Ok(Some(ConversationAnomaly {
                        anomaly_type: AnomalyType::LowQualityResponses,
                        description: format!(
                            "Response confidence {confidence:.2} is below acceptable threshold"
                        ),
                        severity: AnomalySeverity::Medium,
                        detected_at: SystemTime::now(),
                        message_context: vec![message.content.to_string()],
                        suggested_action: Some(
                            "Review retrieval results and LLM outputs".to_string(),
                        ),
                    }));
                }
            }
        }

        Ok(None)
    }

    async fn detect_error_rate_anomalies(
        &mut self,
        analytics: &ConversationAnalytics,
    ) -> Result<Option<ConversationAnomaly>> {
        if analytics.conversation_quality.error_rate > 0.5 && analytics.message_count > 5 {
            return Ok(Some(ConversationAnomaly {
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
            }));
        }

        Ok(None)
    }

    async fn detect_sentiment_anomalies(
        &mut self,
        analytics: &ConversationAnalytics,
    ) -> Result<Option<ConversationAnomaly>> {
        if analytics.sentiment_progression.len() >= 2 {
            let latest_sentiment =
                &analytics.sentiment_progression[analytics.sentiment_progression.len() - 1];
            let previous_sentiment =
                &analytics.sentiment_progression[analytics.sentiment_progression.len() - 2];

            let sentiment_change =
                (latest_sentiment.intensity - previous_sentiment.intensity).abs();

            if sentiment_change > 0.5 && latest_sentiment.intensity < -0.5 {
                return Ok(Some(ConversationAnomaly {
                    anomaly_type: AnomalyType::UnexpectedSentiment,
                    description: format!(
                        "Sudden negative sentiment shift: {} -> {}",
                        previous_sentiment.emotion, latest_sentiment.emotion
                    ),
                    severity: AnomalySeverity::Medium,
                    detected_at: SystemTime::now(),
                    message_context: vec![],
                    suggested_action: Some(
                        "Review recent responses for potential issues".to_string(),
                    ),
                }));
            }
        }

        Ok(None)
    }

    async fn detect_complexity_anomalies(
        &mut self,
        analytics: &ConversationAnalytics,
    ) -> Result<Option<ConversationAnomaly>> {
        if analytics.complexity_progression.len() >= 2 {
            let latest =
                &analytics.complexity_progression[analytics.complexity_progression.len() - 1];
            let previous =
                &analytics.complexity_progression[analytics.complexity_progression.len() - 2];

            let complexity_spike = latest.overall_complexity - previous.overall_complexity;

            if complexity_spike > 0.5 {
                return Ok(Some(ConversationAnomaly {
                    anomaly_type: AnomalyType::ComplexitySpike,
                    description: format!(
                        "Complexity spike detected: {:.2} -> {:.2}",
                        previous.overall_complexity, latest.overall_complexity
                    ),
                    severity: AnomalySeverity::Low,
                    detected_at: SystemTime::now(),
                    message_context: vec![],
                    suggested_action: Some(
                        "Consider providing simpler explanations or breaking down complex topics"
                            .to_string(),
                    ),
                }));
            }
        }

        Ok(None)
    }

    async fn detect_engagement_anomalies(
        &mut self,
        analytics: &ConversationAnalytics,
    ) -> Result<Option<ConversationAnomaly>> {
        if analytics.conversation_quality.engagement_score < 0.3 && analytics.message_count > 5 {
            return Ok(Some(ConversationAnomaly {
                anomaly_type: AnomalyType::EngagementDrop,
                description: format!(
                    "Low engagement score: {:.2}",
                    analytics.conversation_quality.engagement_score
                ),
                severity: AnomalySeverity::Medium,
                detected_at: SystemTime::now(),
                message_context: vec![],
                suggested_action: Some(
                    "Try to re-engage user with questions or interesting topics".to_string(),
                ),
            }));
        }

        Ok(None)
    }

    async fn detect_context_loss_anomalies(
        &mut self,
        analytics: &ConversationAnalytics,
    ) -> Result<Option<ConversationAnomaly>> {
        if analytics.conversation_quality.coherence_score < 0.4 && analytics.message_count > 3 {
            return Ok(Some(ConversationAnomaly {
                anomaly_type: AnomalyType::ContextLoss,
                description: format!(
                    "Low coherence score indicates potential context loss: {:.2}",
                    analytics.conversation_quality.coherence_score
                ),
                severity: AnomalySeverity::Medium,
                detected_at: SystemTime::now(),
                message_context: vec![],
                suggested_action: Some(
                    "Review context management and conversation history".to_string(),
                ),
            }));
        }

        Ok(None)
    }

    async fn detect_topic_divergence_anomalies(
        &mut self,
        analytics: &ConversationAnalytics,
    ) -> Result<Option<ConversationAnomaly>> {
        if analytics.topics_discussed.len() > 5
            && analytics.conversation_quality.relevance_score < 0.5
        {
            return Ok(Some(ConversationAnomaly {
                anomaly_type: AnomalyType::TopicDivergence,
                description: format!(
                    "Topic divergence detected: {} topics, relevance score: {:.2}",
                    analytics.topics_discussed.len(),
                    analytics.conversation_quality.relevance_score
                ),
                severity: AnomalySeverity::Low,
                detected_at: SystemTime::now(),
                message_context: analytics.topics_discussed.clone(),
                suggested_action: Some(
                    "Focus on main topics and maintain conversation relevance".to_string(),
                ),
            }));
        }

        Ok(None)
    }

    async fn detect_confidence_collapse_anomalies(
        &mut self,
        analytics: &ConversationAnalytics,
    ) -> Result<Option<ConversationAnomaly>> {
        if analytics.confidence_progression.len() >= 3 {
            let recent_confidences: Vec<f64> = analytics
                .confidence_progression
                .iter()
                .rev()
                .take(3)
                .map(|c| c.overall_confidence)
                .collect();

            let avg_confidence =
                recent_confidences.iter().sum::<f64>() / recent_confidences.len() as f64;

            if avg_confidence < 0.4 {
                return Ok(Some(ConversationAnomaly {
                    anomaly_type: AnomalyType::ConfidenceCollapse,
                    description: format!(
                        "Confidence collapse detected: average confidence {avg_confidence:.2}"
                    ),
                    severity: AnomalySeverity::High,
                    detected_at: SystemTime::now(),
                    message_context: vec![],
                    suggested_action: Some(
                        "Review knowledge base and improve retrieval quality".to_string(),
                    ),
                }));
            }
        }

        Ok(None)
    }

    async fn detect_repeated_error_anomalies(&mut self) -> Result<Option<ConversationAnomaly>> {
        if self.error_history.len() >= 3 {
            let recent_errors: Vec<bool> =
                self.error_history.iter().rev().take(3).cloned().collect();
            let error_count = recent_errors.iter().filter(|&&e| e).count();

            if error_count >= 2 {
                return Ok(Some(ConversationAnomaly {
                    anomaly_type: AnomalyType::RepeatedErrors,
                    description: format!(
                        "Repeated errors detected: {error_count} out of last 3 responses"
                    ),
                    severity: AnomalySeverity::High,
                    detected_at: SystemTime::now(),
                    message_context: vec![],
                    suggested_action: Some("Investigate root cause of repeated errors".to_string()),
                }));
            }
        }

        Ok(None)
    }
}
