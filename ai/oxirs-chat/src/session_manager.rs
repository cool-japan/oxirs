//! Session management and chat functionality

use crate::messages::Message;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    time::SystemTime,
};

/// Session configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatConfig {
    pub max_context_tokens: usize,
    pub sliding_window_size: usize,
    pub enable_context_compression: bool,
    pub temperature: f32,
    pub max_tokens: usize,
    pub timeout_seconds: u64,
    pub enable_topic_tracking: bool,
    pub enable_sentiment_analysis: bool,
    pub enable_intent_detection: bool,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            max_context_tokens: 8000,
            sliding_window_size: 20,
            enable_context_compression: true,
            temperature: 0.7,
            max_tokens: 2000,
            timeout_seconds: 30,
            enable_topic_tracking: true,
            enable_sentiment_analysis: true,
            enable_intent_detection: true,
        }
    }
}

/// Session state enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionState {
    Active,
    Idle,
    Suspended,
    Archived,
    Expired,
}

/// Context window for managing conversation context
#[derive(Debug, Clone)]
pub struct ContextWindow {
    pub window_size: usize,
    pub active_messages: VecDeque<String>, // Message IDs
    pub pinned_messages: HashSet<String>,
    pub context_summary: Option<String>,
    pub importance_scores: HashMap<String, f32>,
    pub token_count: usize,
    pub last_compression: Option<SystemTime>,
}

impl ContextWindow {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            active_messages: VecDeque::new(),
            pinned_messages: HashSet::new(),
            context_summary: None,
            importance_scores: HashMap::new(),
            token_count: 0,
            last_compression: None,
        }
    }

    pub fn add_message(&mut self, message_id: String, importance: f32, tokens: usize) {
        // Remove oldest if window is full and message isn't pinned
        while self.active_messages.len() >= self.window_size {
            if let Some(oldest_id) = self.active_messages.front() {
                if !self.pinned_messages.contains(oldest_id) {
                    let removed_id = self.active_messages.pop_front().unwrap();
                    if let Some(removed_tokens) = self.importance_scores.remove(&removed_id) {
                        // Estimate tokens from importance score (simplified)
                        self.token_count = self
                            .token_count
                            .saturating_sub((removed_tokens * 100.0) as usize);
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        self.active_messages.push_back(message_id.clone());
        self.importance_scores.insert(message_id, importance);
        self.token_count += tokens;
    }

    pub fn pin_message(&mut self, message_id: String) {
        self.pinned_messages.insert(message_id);
    }

    pub fn unpin_message(&mut self, message_id: &str) {
        self.pinned_messages.remove(message_id);
    }

    pub fn get_context_messages(&self) -> Vec<String> {
        self.active_messages.iter().cloned().collect()
    }

    pub fn compress_context(&mut self, summary: String) {
        self.context_summary = Some(summary);
        self.last_compression = Some(SystemTime::now());

        // Remove oldest unpinned messages after compression
        let mut to_remove = Vec::new();
        for message_id in self.active_messages.iter() {
            if !self.pinned_messages.contains(message_id) {
                to_remove.push(message_id.clone());
            }
        }

        // Keep only half after compression
        let keep_count = to_remove.len() / 2;
        for (i, message_id) in to_remove.iter().enumerate() {
            if i < to_remove.len() - keep_count {
                self.active_messages.retain(|id| id != message_id);
                self.importance_scores.remove(message_id);
            }
        }

        // Recalculate token count
        self.token_count = self.importance_scores.len() * 50; // Simplified estimation
    }

    pub fn needs_compression(&self, max_tokens: usize) -> bool {
        self.token_count > max_tokens && self.context_summary.is_none()
    }
}

/// Topic tracking for conversation analysis
#[derive(Debug, Clone)]
pub struct TopicTracker {
    pub current_topics: Vec<Topic>,
    pub topic_history: Vec<TopicTransition>,
    pub confidence_threshold: f32,
    pub max_topics: usize,
}

impl Default for TopicTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl TopicTracker {
    pub fn new() -> Self {
        Self {
            current_topics: Vec::new(),
            topic_history: Vec::new(),
            confidence_threshold: 0.6,
            max_topics: 5,
        }
    }

    pub fn analyze_message(&mut self, message: &Message) -> Option<TopicTransition> {
        // Simplified topic analysis - in real implementation, this would use NLP
        let content = message.content.to_text().to_lowercase();

        // Basic keyword-based topic detection
        let detected_topics = self.extract_topics(&content);

        if !detected_topics.is_empty() {
            let transition = self.update_topics(detected_topics, &message.id);
            if let Some(ref t) = transition {
                self.topic_history.push(t.clone());
            }
            transition
        } else {
            None
        }
    }

    fn extract_topics(&self, content: &str) -> Vec<String> {
        let mut topics = Vec::new();

        // Simple keyword matching - in practice, use proper NLP
        if content.contains("sparql") || content.contains("query") {
            topics.push("SPARQL Queries".to_string());
        }
        if content.contains("graph") || content.contains("rdf") {
            topics.push("Knowledge Graphs".to_string());
        }
        if content.contains("data") || content.contains("dataset") {
            topics.push("Data Management".to_string());
        }

        topics
    }

    fn update_topics(
        &mut self,
        new_topics: Vec<String>,
        trigger_message_id: &str,
    ) -> Option<TopicTransition> {
        // Determine transition type
        let transition_type = if self.current_topics.is_empty() {
            TransitionType::NewTopic
        } else {
            // Check for topic overlap
            let current_topic_names: HashSet<String> =
                self.current_topics.iter().map(|t| t.name.clone()).collect();
            let new_topic_names: HashSet<String> = new_topics.iter().cloned().collect();

            if current_topic_names.intersection(&new_topic_names).count() > 0 {
                TransitionType::TopicReturn
            } else {
                TransitionType::TopicShift
            }
        };

        // Update current topics
        self.current_topics.clear();
        for topic_name in new_topics {
            self.current_topics.push(Topic {
                name: topic_name.clone(),
                confidence: 0.8, // Simplified confidence
                first_mentioned: chrono::Utc::now(),
                last_mentioned: chrono::Utc::now(),
                message_count: 1,
                keywords: vec![topic_name.to_lowercase()],
            });
        }

        Some(TopicTransition {
            from_topics: Vec::new(), // Simplified
            to_topics: self.current_topics.iter().map(|t| t.name.clone()).collect(),
            timestamp: chrono::Utc::now(),
            trigger_message_id: trigger_message_id.to_string(),
            confidence: 0.8,
            transition_type,
        })
    }

    pub fn get_current_topic_summary(&self) -> String {
        if self.current_topics.is_empty() {
            "No specific topic detected".to_string()
        } else {
            let topic_names: Vec<String> =
                self.current_topics.iter().map(|t| t.name.clone()).collect();
            format!("Current topics: {}", topic_names.join(", "))
        }
    }
}

/// Topic information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topic {
    pub name: String,
    pub confidence: f32,
    pub first_mentioned: chrono::DateTime<chrono::Utc>,
    pub last_mentioned: chrono::DateTime<chrono::Utc>,
    pub message_count: usize,
    pub keywords: Vec<String>,
}

impl Topic {
    pub fn update_mention(&mut self) {
        self.last_mentioned = chrono::Utc::now();
        self.message_count += 1;
    }

    pub fn add_keyword(&mut self, keyword: String) {
        if !self.keywords.contains(&keyword) {
            self.keywords.push(keyword);
        }
    }

    pub fn get_relevance_score(&self) -> f32 {
        let time_decay = {
            let now = chrono::Utc::now();
            let hours_since = now.signed_duration_since(self.last_mentioned).num_hours() as f32;
            (-hours_since / 24.0).exp() // Exponential decay over days
        };

        let frequency_boost = (self.message_count as f32).ln().max(1.0);

        self.confidence * time_decay * frequency_boost
    }
}

/// Topic transition information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicTransition {
    pub from_topics: Vec<String>,
    pub to_topics: Vec<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub trigger_message_id: String,
    pub confidence: f32,
    pub transition_type: TransitionType,
}

/// Type of topic transition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransitionType {
    NewTopic,
    TopicShift,
    TopicReturn,
    TopicMerge,
    TopicSplit,
}

impl PartialEq<&str> for TransitionType {
    fn eq(&self, other: &&str) -> bool {
        matches!(
            (self, *other),
            (TransitionType::NewTopic, "new")
                | (TransitionType::TopicShift, "shift")
                | (TransitionType::TopicReturn, "return")
                | (TransitionType::TopicMerge, "merge")
                | (TransitionType::TopicSplit, "split")
        )
    }
}

/// Session performance metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SessionMetrics {
    pub total_messages: usize,
    pub user_messages: usize,
    pub assistant_messages: usize,
    pub average_response_time: f64,
    pub total_tokens_used: usize,
    pub successful_queries: usize,
    pub failed_queries: usize,
    pub context_compressions: usize,
    pub topic_transitions: usize,
    pub user_satisfaction_scores: Vec<f32>,
    pub error_count: usize,
    pub warning_count: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl SessionMetrics {
    pub fn update_response_time(&mut self, response_time_ms: u64) {
        let response_time_s = response_time_ms as f64 / 1000.0;
        if self.assistant_messages == 0 {
            self.average_response_time = response_time_s;
        } else {
            self.average_response_time =
                (self.average_response_time * self.assistant_messages as f64 + response_time_s)
                    / (self.assistant_messages as f64 + 1.0);
        }
        self.assistant_messages += 1;
        self.total_messages += 1;
        self.last_updated = chrono::Utc::now();
    }

    pub fn add_user_message(&mut self) {
        self.user_messages += 1;
        self.total_messages += 1;
        self.last_updated = chrono::Utc::now();
    }

    pub fn add_successful_query(&mut self, tokens_used: usize) {
        self.successful_queries += 1;
        self.total_tokens_used += tokens_used;
        self.last_updated = chrono::Utc::now();
    }

    pub fn add_failed_query(&mut self) {
        self.failed_queries += 1;
        self.error_count += 1;
        self.last_updated = chrono::Utc::now();
    }

    pub fn add_context_compression(&mut self) {
        self.context_compressions += 1;
        self.last_updated = chrono::Utc::now();
    }

    pub fn add_topic_transition(&mut self) {
        self.topic_transitions += 1;
        self.last_updated = chrono::Utc::now();
    }

    pub fn add_satisfaction_score(&mut self, score: f32) {
        self.user_satisfaction_scores.push(score.clamp(0.0, 5.0));
        self.last_updated = chrono::Utc::now();
    }

    pub fn get_average_satisfaction(&self) -> f32 {
        if self.user_satisfaction_scores.is_empty() {
            0.0
        } else {
            self.user_satisfaction_scores.iter().sum::<f32>()
                / self.user_satisfaction_scores.len() as f32
        }
    }

    pub fn get_query_success_rate(&self) -> f32 {
        let total_queries = self.successful_queries + self.failed_queries;
        if total_queries == 0 {
            0.0
        } else {
            self.successful_queries as f32 / total_queries as f32
        }
    }

    pub fn get_cache_hit_rate(&self) -> f32 {
        let total_cache_requests = self.cache_hits + self.cache_misses;
        if total_cache_requests == 0 {
            0.0
        } else {
            self.cache_hits as f32 / total_cache_requests as f32
        }
    }
}

/// Chat session data for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionData {
    pub id: String,
    pub config: ChatConfig,
    pub messages: Vec<Message>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_activity: chrono::DateTime<chrono::Utc>,
    pub user_preferences: HashMap<String, String>,
    pub session_state: SessionState,
    pub context_summary: Option<String>,
    pub pinned_messages: HashSet<String>,
    pub current_topics: Vec<Topic>,
    pub topic_history: Vec<TopicTransition>,
    pub performance_metrics: SessionMetrics,
}
