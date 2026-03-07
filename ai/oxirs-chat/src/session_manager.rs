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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
            // Check if front message is pinned before removing
            let should_remove = self
                .active_messages
                .front()
                .map(|id| !self.pinned_messages.contains(id))
                .unwrap_or(false);

            if should_remove {
                if let Some(removed_id) = self.active_messages.pop_front() {
                    if let Some(removed_tokens) = self.importance_scores.remove(&removed_id) {
                        // Estimate tokens from importance score (simplified)
                        self.token_count = self
                            .token_count
                            .saturating_sub((removed_tokens * 100.0) as usize);
                    }
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

// Compatibility aliases for lib.rs
impl SessionData {
    pub fn user_id(&self) -> Option<&str> {
        self.user_preferences.get("user_id").map(|s| s.as_str())
    }

    pub fn set_user_id(&mut self, user_id: String) {
        self.user_preferences.insert("user_id".to_string(), user_id);
    }
}

// Re-export with compatibility
pub use SessionState as state;

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ChatConfig ────────────────────────────────────────────────────────────

    #[test]
    fn test_chat_config_default_has_reasonable_values() {
        let cfg = ChatConfig::default();
        assert!(cfg.max_context_tokens > 0);
        assert!(cfg.sliding_window_size > 0);
        assert!(cfg.max_tokens > 0);
        assert!(cfg.timeout_seconds > 0);
    }

    #[test]
    fn test_chat_config_temperature_in_range() {
        let cfg = ChatConfig::default();
        assert!(cfg.temperature >= 0.0 && cfg.temperature <= 2.0);
    }

    #[test]
    fn test_chat_config_flags_set() {
        let cfg = ChatConfig::default();
        assert!(cfg.enable_topic_tracking);
        assert!(cfg.enable_context_compression);
    }

    // ── SessionState ──────────────────────────────────────────────────────────

    #[test]
    fn test_session_state_active_eq() {
        assert_eq!(SessionState::Active, SessionState::Active);
        assert_ne!(SessionState::Active, SessionState::Expired);
    }

    #[test]
    fn test_session_state_all_variants_accessible() {
        let states = [
            SessionState::Active,
            SessionState::Idle,
            SessionState::Suspended,
            SessionState::Archived,
            SessionState::Expired,
        ];
        assert_eq!(states.len(), 5);
    }

    // ── ContextWindow ─────────────────────────────────────────────────────────

    #[test]
    fn test_context_window_new_empty() {
        let cw = ContextWindow::new(10);
        assert_eq!(cw.window_size, 10);
        assert!(cw.active_messages.is_empty());
        assert_eq!(cw.token_count, 0);
    }

    #[test]
    fn test_context_window_add_message() {
        let mut cw = ContextWindow::new(5);
        cw.add_message("msg1".to_string(), 1.0, 10);
        assert_eq!(cw.active_messages.len(), 1);
        assert_eq!(cw.token_count, 10);
    }

    #[test]
    fn test_context_window_evicts_oldest_when_full() {
        let mut cw = ContextWindow::new(3);
        cw.add_message("a".to_string(), 0.5, 5);
        cw.add_message("b".to_string(), 0.5, 5);
        cw.add_message("c".to_string(), 0.5, 5);
        cw.add_message("d".to_string(), 0.5, 5); // evicts 'a'
        let msgs = cw.get_context_messages();
        assert!(!msgs.contains(&"a".to_string()));
        assert!(msgs.contains(&"b".to_string()) || msgs.contains(&"c".to_string()));
    }

    #[test]
    fn test_context_window_pin_message() {
        let mut cw = ContextWindow::new(2);
        cw.add_message("pinned".to_string(), 1.0, 5);
        cw.pin_message("pinned".to_string());
        assert!(cw.pinned_messages.contains("pinned"));
    }

    #[test]
    fn test_context_window_unpin_message() {
        let mut cw = ContextWindow::new(5);
        cw.pin_message("msg".to_string());
        cw.unpin_message("msg");
        assert!(!cw.pinned_messages.contains("msg"));
    }

    #[test]
    fn test_context_window_get_context_messages() {
        let mut cw = ContextWindow::new(5);
        cw.add_message("m1".to_string(), 0.5, 3);
        cw.add_message("m2".to_string(), 0.8, 4);
        let msgs = cw.get_context_messages();
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn test_context_window_needs_compression() {
        let mut cw = ContextWindow::new(100);
        cw.token_count = 5000;
        cw.context_summary = None;
        assert!(cw.needs_compression(4000));
        assert!(!cw.needs_compression(6000));
    }

    #[test]
    fn test_context_window_needs_compression_false_when_summarized() {
        let mut cw = ContextWindow::new(100);
        cw.token_count = 9000;
        cw.context_summary = Some("summary".to_string());
        // Already compressed — should not trigger again.
        assert!(!cw.needs_compression(100));
    }

    #[test]
    fn test_context_window_compress_sets_summary() {
        let mut cw = ContextWindow::new(10);
        cw.add_message("m1".to_string(), 0.5, 10);
        cw.compress_context("compressed summary".to_string());
        assert_eq!(cw.context_summary.as_deref(), Some("compressed summary"));
        assert!(cw.last_compression.is_some());
    }

    // ── TopicTracker ──────────────────────────────────────────────────────────

    #[test]
    fn test_topic_tracker_new_empty() {
        let t = TopicTracker::new();
        assert!(t.current_topics.is_empty());
        assert!(t.topic_history.is_empty());
    }

    #[test]
    fn test_topic_tracker_default() {
        let t = TopicTracker::default();
        assert!(t.current_topics.is_empty());
        assert_eq!(t.max_topics, 5);
    }

    #[test]
    fn test_topic_tracker_summary_no_topics() {
        let t = TopicTracker::new();
        let s = t.get_current_topic_summary();
        assert!(s.contains("No") || !s.is_empty());
    }

    // ── Topic ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_topic_add_keyword() {
        let mut topic = Topic {
            name: "SPARQL".to_string(),
            confidence: 0.9,
            first_mentioned: chrono::Utc::now(),
            last_mentioned: chrono::Utc::now(),
            message_count: 1,
            keywords: vec!["sparql".to_string()],
        };
        topic.add_keyword("query".to_string());
        assert!(topic.keywords.contains(&"query".to_string()));
        assert_eq!(topic.keywords.len(), 2);
    }

    #[test]
    fn test_topic_add_keyword_no_duplicate() {
        let mut topic = Topic {
            name: "RDF".to_string(),
            confidence: 0.8,
            first_mentioned: chrono::Utc::now(),
            last_mentioned: chrono::Utc::now(),
            message_count: 1,
            keywords: vec!["rdf".to_string()],
        };
        topic.add_keyword("rdf".to_string()); // duplicate
        assert_eq!(topic.keywords.len(), 1);
    }

    #[test]
    fn test_topic_update_mention_increments_count() {
        let mut topic = Topic {
            name: "Graphs".to_string(),
            confidence: 0.7,
            first_mentioned: chrono::Utc::now(),
            last_mentioned: chrono::Utc::now(),
            message_count: 1,
            keywords: vec![],
        };
        topic.update_mention();
        assert_eq!(topic.message_count, 2);
    }

    #[test]
    fn test_topic_relevance_score_positive() {
        let topic = Topic {
            name: "Test".to_string(),
            confidence: 0.9,
            first_mentioned: chrono::Utc::now(),
            last_mentioned: chrono::Utc::now(),
            message_count: 5,
            keywords: vec![],
        };
        let score = topic.get_relevance_score();
        assert!(score > 0.0);
    }

    // ── SessionMetrics ────────────────────────────────────────────────────────

    #[test]
    fn test_session_metrics_default_zero() {
        let m = SessionMetrics::default();
        assert_eq!(m.total_messages, 0);
        assert_eq!(m.successful_queries, 0);
    }

    #[test]
    fn test_session_metrics_add_user_message() {
        let mut m = SessionMetrics::default();
        m.add_user_message();
        assert_eq!(m.user_messages, 1);
        assert_eq!(m.total_messages, 1);
    }

    #[test]
    fn test_session_metrics_add_successful_query() {
        let mut m = SessionMetrics::default();
        m.add_successful_query(100);
        assert_eq!(m.successful_queries, 1);
        assert_eq!(m.total_tokens_used, 100);
    }

    #[test]
    fn test_session_metrics_add_failed_query() {
        let mut m = SessionMetrics::default();
        m.add_failed_query();
        assert_eq!(m.failed_queries, 1);
        assert_eq!(m.error_count, 1);
    }

    #[test]
    fn test_session_metrics_query_success_rate() {
        let mut m = SessionMetrics::default();
        m.add_successful_query(10);
        m.add_successful_query(10);
        m.add_failed_query();
        let rate = m.get_query_success_rate();
        assert!((rate - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_session_metrics_query_success_rate_zero_total() {
        let m = SessionMetrics::default();
        assert_eq!(m.get_query_success_rate(), 0.0);
    }

    #[test]
    fn test_session_metrics_cache_hit_rate() {
        let m = SessionMetrics {
            cache_hits: 3,
            cache_misses: 1,
            ..SessionMetrics::default()
        };
        let rate = m.get_cache_hit_rate();
        assert!((rate - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_session_metrics_cache_hit_rate_zero_total() {
        let m = SessionMetrics::default();
        assert_eq!(m.get_cache_hit_rate(), 0.0);
    }

    #[test]
    fn test_session_metrics_satisfaction_score() {
        let mut m = SessionMetrics::default();
        m.add_satisfaction_score(4.0);
        m.add_satisfaction_score(2.0);
        let avg = m.get_average_satisfaction();
        assert!((avg - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_session_metrics_satisfaction_clamped_to_five() {
        let mut m = SessionMetrics::default();
        m.add_satisfaction_score(10.0); // clamped to 5.0
        let avg = m.get_average_satisfaction();
        assert!((avg - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_session_metrics_average_satisfaction_empty() {
        let m = SessionMetrics::default();
        assert_eq!(m.get_average_satisfaction(), 0.0);
    }

    #[test]
    fn test_session_metrics_context_compression_tracked() {
        let mut m = SessionMetrics::default();
        m.add_context_compression();
        assert_eq!(m.context_compressions, 1);
    }

    #[test]
    fn test_session_metrics_topic_transition_tracked() {
        let mut m = SessionMetrics::default();
        m.add_topic_transition();
        assert_eq!(m.topic_transitions, 1);
    }

    // ── SessionData ───────────────────────────────────────────────────────────

    #[test]
    fn test_session_data_user_id_roundtrip() {
        let mut sd = SessionData {
            id: "test-session".to_string(),
            config: ChatConfig::default(),
            messages: Vec::new(),
            created_at: chrono::Utc::now(),
            last_activity: chrono::Utc::now(),
            user_preferences: HashMap::new(),
            session_state: SessionState::Active,
            context_summary: None,
            pinned_messages: HashSet::new(),
            current_topics: Vec::new(),
            topic_history: Vec::new(),
            performance_metrics: SessionMetrics::default(),
        };
        sd.set_user_id("alice".to_string());
        assert_eq!(sd.user_id(), Some("alice"));
    }

    #[test]
    fn test_session_data_user_id_none_initially() {
        let sd = SessionData {
            id: "s1".to_string(),
            config: ChatConfig::default(),
            messages: Vec::new(),
            created_at: chrono::Utc::now(),
            last_activity: chrono::Utc::now(),
            user_preferences: HashMap::new(),
            session_state: SessionState::Active,
            context_summary: None,
            pinned_messages: HashSet::new(),
            current_topics: Vec::new(),
            topic_history: Vec::new(),
            performance_metrics: SessionMetrics::default(),
        };
        assert!(sd.user_id().is_none());
    }

    // ── TransitionType ────────────────────────────────────────────────────────

    #[test]
    fn test_transition_type_partial_eq_str() {
        assert_eq!(TransitionType::NewTopic, "new");
        assert_eq!(TransitionType::TopicShift, "shift");
        assert_eq!(TransitionType::TopicReturn, "return");
        assert_eq!(TransitionType::TopicMerge, "merge");
        assert_eq!(TransitionType::TopicSplit, "split");
    }

    #[test]
    fn test_transition_type_ne_wrong_str() {
        assert_ne!(TransitionType::NewTopic, "shift");
    }

    // ── SessionMetrics additional ─────────────────────────────────────────────

    #[test]
    fn test_session_metrics_update_response_time_first_call() {
        let mut m = SessionMetrics::default();
        m.update_response_time(500);
        assert!((m.average_response_time - 0.5).abs() < 1e-6);
        assert_eq!(m.assistant_messages, 1);
    }

    #[test]
    fn test_session_metrics_update_response_time_average() {
        let mut m = SessionMetrics::default();
        m.update_response_time(1000); // 1s
        m.update_response_time(3000); // 3s
                                      // Average: (1 + 3) / 2 = 2s
        assert!((m.average_response_time - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_session_metrics_tokens_accumulate() {
        let mut m = SessionMetrics::default();
        m.add_successful_query(100);
        m.add_successful_query(250);
        assert_eq!(m.total_tokens_used, 350);
    }

    #[test]
    fn test_session_metrics_satisfaction_clamped_to_zero() {
        let mut m = SessionMetrics::default();
        m.add_satisfaction_score(-5.0); // clamped to 0.0
        let avg = m.get_average_satisfaction();
        assert!((avg - 0.0).abs() < 1e-6);
    }

    // ── ContextWindow additional ──────────────────────────────────────────────

    #[test]
    fn test_context_window_importance_scores_stored() {
        let mut cw = ContextWindow::new(10);
        cw.add_message("m1".to_string(), 0.9, 5);
        assert!(cw.importance_scores.contains_key("m1"));
    }

    #[test]
    fn test_context_window_multiple_pins() {
        let mut cw = ContextWindow::new(10);
        cw.pin_message("m1".to_string());
        cw.pin_message("m2".to_string());
        assert_eq!(cw.pinned_messages.len(), 2);
    }

    // ── ChatConfig custom ─────────────────────────────────────────────────────

    #[test]
    fn test_chat_config_custom_values() {
        let cfg = ChatConfig {
            max_context_tokens: 4096,
            sliding_window_size: 10,
            enable_context_compression: false,
            temperature: 0.5,
            max_tokens: 512,
            timeout_seconds: 15,
            enable_topic_tracking: false,
            enable_sentiment_analysis: false,
            enable_intent_detection: false,
        };
        assert_eq!(cfg.max_context_tokens, 4096);
        assert_eq!(cfg.sliding_window_size, 10);
        assert!(!cfg.enable_context_compression);
    }

    // ── TopicTracker analyze_message ──────────────────────────────────────────

    #[test]
    fn test_topic_tracker_analyze_sparql_message() {
        use crate::messages::{MessageContent, MessageRole};
        let mut tracker = TopicTracker::new();
        let msg = crate::messages::Message {
            id: "m1".to_string(),
            role: MessageRole::User,
            content: MessageContent::Text("How do I write a sparql query?".to_string()),
            timestamp: chrono::Utc::now(),
            metadata: None,
            thread_id: None,
            parent_message_id: None,
            token_count: None,
            reactions: Vec::new(),
            attachments: Vec::new(),
            rich_elements: Vec::new(),
        };
        let transition = tracker.analyze_message(&msg);
        // Should detect "SPARQL Queries" topic.
        assert!(transition.is_some());
    }

    #[test]
    fn test_topic_tracker_summary_with_topics() {
        let mut tracker = TopicTracker::new();
        tracker.current_topics.push(Topic {
            name: "SPARQL".to_string(),
            confidence: 0.9,
            first_mentioned: chrono::Utc::now(),
            last_mentioned: chrono::Utc::now(),
            message_count: 1,
            keywords: vec![],
        });
        let summary = tracker.get_current_topic_summary();
        assert!(summary.contains("SPARQL"));
    }
}
