//! Chat session implementation and management

use crate::messages::{Message, MessageRole};
use crate::session_manager::{
    ChatConfig, ContextWindow, SessionData, SessionMetrics, SessionState, TopicTracker,
};
use crate::types::*;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};

/// Chat session
pub struct ChatSession {
    pub id: String,
    pub config: ChatConfig,
    pub messages: Vec<Message>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_activity: chrono::DateTime<chrono::Utc>,
    pub user_preferences: HashMap<String, String>,
    pub session_state: SessionState,
    pub context_window: ContextWindow,
    pub topic_tracker: TopicTracker,
    pub performance_metrics: SessionMetrics,
    store: Arc<dyn oxirs_core::Store>,
}

impl ChatSession {
    pub fn new(id: String, store: Arc<dyn oxirs_core::Store>) -> Self {
        let now = chrono::Utc::now();
        let config = ChatConfig::default();
        Self {
            id,
            config: config.clone(),
            messages: Vec::new(),
            created_at: now,
            last_activity: now,
            user_preferences: HashMap::new(),
            session_state: SessionState::Active,
            context_window: ContextWindow::new(config.sliding_window_size),
            topic_tracker: TopicTracker::new(),
            performance_metrics: SessionMetrics::default(),
            store,
        }
    }

    pub fn from_data(data: SessionData, store: Arc<dyn oxirs_core::Store>) -> Self {
        let mut context_window = ContextWindow::new(data.config.sliding_window_size);
        context_window.pinned_messages = data.pinned_messages;
        context_window.context_summary = data.context_summary;

        let mut topic_tracker = TopicTracker::new();
        topic_tracker.current_topics = data.current_topics;
        topic_tracker.topic_history = data.topic_history;

        Self {
            id: data.id,
            config: data.config,
            messages: data.messages,
            created_at: data.created_at,
            last_activity: data.last_activity,
            user_preferences: data.user_preferences,
            session_state: data.session_state,
            context_window,
            topic_tracker,
            performance_metrics: data.performance_metrics,
            store,
        }
    }

    pub fn to_data(&self) -> SessionData {
        SessionData {
            id: self.id.clone(),
            config: self.config.clone(),
            messages: self.messages.clone(),
            created_at: self.created_at,
            last_activity: self.last_activity,
            user_preferences: self.user_preferences.clone(),
            session_state: self.session_state.clone(),
            context_summary: self.context_window.context_summary.clone(),
            pinned_messages: self.context_window.pinned_messages.clone(),
            current_topics: self.topic_tracker.current_topics.clone(),
            topic_history: self.topic_tracker.topic_history.clone(),
            performance_metrics: self.performance_metrics.clone(),
        }
    }

    pub fn update_activity(&mut self) {
        self.last_activity = chrono::Utc::now();
    }

    pub fn add_message(&mut self, mut message: Message) -> Result<()> {
        // Update activity timestamp
        self.update_activity();

        // Analyze topic changes
        if let Some(transition) = self.topic_tracker.analyze_message(&message) {
            info!(
                "Topic transition detected: {:?}",
                transition.transition_type
            );
            self.performance_metrics.add_topic_transition();
        }

        // Add to context window
        let importance = self.calculate_message_importance(&message);
        let estimated_tokens = message.content.len() / 4; // Rough estimate
        self.context_window
            .add_message(message.id.clone(), importance, estimated_tokens);

        // Update metrics
        match message.role {
            MessageRole::User => self.performance_metrics.add_user_message(),
            MessageRole::Assistant => {
                // Response time would be calculated externally and passed in
                self.performance_metrics.update_response_time(1000); // Default 1s
            }
            _ => {}
        }

        // Store message
        self.messages.push(message);

        // Check if context compression is needed
        if self
            .context_window
            .needs_compression(self.config.max_context_tokens)
        {
            self.compress_context()?;
        }

        Ok(())
    }

    fn calculate_message_importance(&self, message: &Message) -> f32 {
        let mut importance: f32 = 0.5; // Base importance

        // Boost importance for questions
        if message.content.contains('?') {
            importance += 0.2;
        }

        // Boost importance for SPARQL queries
        if message.content.to_lowercase().contains("select")
            || message.content.to_lowercase().contains("construct")
        {
            importance += 0.3;
        }

        // Boost importance for error messages
        if message.content.to_lowercase().contains("error")
            || message.content.to_lowercase().contains("fail")
        {
            importance += 0.4;
        }

        // Boost importance for code blocks
        if message.content.to_text().contains("```") {
            importance += 0.2;
        }

        importance.min(1.0)
    }

    fn compress_context(&mut self) -> Result<()> {
        // This would typically use an LLM to summarize the context
        // For now, we'll create a simple summary
        let recent_topics = self.topic_tracker.get_current_topic_summary();
        let message_count = self.context_window.get_context_messages().len();

        let summary = format!(
            "Context summary: {} messages discussing {}. Recent topics include: {}",
            message_count, "various topics", recent_topics
        );

        self.context_window.compress_context(summary);
        self.performance_metrics.add_context_compression();

        Ok(())
    }

    pub fn get_context_for_query(&self) -> Vec<&Message> {
        let context_message_ids = self.context_window.get_context_messages();
        self.messages.iter()
            .filter(|msg| context_message_ids.contains(&msg.id))
            .collect()
    }

    pub fn pin_message(&mut self, message_id: String) {
        self.context_window.pin_message(message_id);
    }

    pub fn unpin_message(&mut self, message_id: &str) {
        self.context_window.unpin_message(message_id);
    }

    pub fn set_preference(&mut self, key: String, value: String) {
        self.user_preferences.insert(key, value);
        self.update_activity();
    }

    pub fn get_preference(&self, key: &str) -> Option<&String> {
        self.user_preferences.get(key)
    }

    pub fn suspend(&mut self) {
        self.session_state = SessionState::Suspended;
        self.update_activity();
    }

    pub fn resume(&mut self) {
        self.session_state = SessionState::Active;
        self.update_activity();
    }

    pub fn archive(&mut self) {
        self.session_state = SessionState::Archived;
        self.update_activity();
    }

    pub fn is_active(&self) -> bool {
        matches!(self.session_state, SessionState::Active)
    }

    pub fn get_session_age(&self) -> chrono::Duration {
        chrono::Utc::now() - self.created_at
    }

    pub fn get_idle_time(&self) -> chrono::Duration {
        chrono::Utc::now() - self.last_activity
    }

    pub fn should_expire(&self, max_idle_duration: chrono::Duration) -> bool {
        self.get_idle_time() > max_idle_duration
    }

    pub fn get_message_by_id(&self, message_id: &str) -> Option<&Message> {
        self.messages.iter().find(|m| m.id == message_id)
    }

    pub fn get_messages_since(&self, since: chrono::DateTime<chrono::Utc>) -> Vec<&Message> {
        self.messages
            .iter()
            .filter(|m| m.timestamp > since)
            .collect()
    }

    pub fn get_user_messages(&self) -> Vec<&Message> {
        self.messages
            .iter()
            .filter(|m| matches!(m.role, MessageRole::User))
            .collect()
    }

    pub fn get_assistant_messages(&self) -> Vec<&Message> {
        self.messages
            .iter()
            .filter(|m| matches!(m.role, MessageRole::Assistant))
            .collect()
    }

    pub fn get_latest_message(&self) -> Option<&Message> {
        self.messages.last()
    }

    pub fn get_conversation_summary(&self) -> String {
        if let Some(ref summary) = self.context_window.context_summary {
            summary.clone()
        } else if self.messages.is_empty() {
            "No messages in this conversation".to_string()
        } else {
            format!("Conversation with {} messages", self.messages.len())
        }
    }

    pub fn export_data(&self) -> SessionData {
        self.to_data()
    }

    pub fn get_statistics(&self) -> SessionStatistics {
        SessionStatistics {
            session_id: self.id.clone(),
            total_messages: self.messages.len(),
            user_messages: self.get_user_messages().len(),
            assistant_messages: self.get_assistant_messages().len(),
            session_duration: self.get_session_age(),
            last_activity: self.last_activity,
            topic_count: self.topic_tracker.current_topics.len(),
            pinned_messages: self.context_window.pinned_messages.len(),
            context_compressions: self.performance_metrics.context_compressions,
            average_response_time: self.performance_metrics.average_response_time,
            query_success_rate: self.performance_metrics.get_query_success_rate(),
            user_satisfaction: self.performance_metrics.get_average_satisfaction(),
        }
    }
}

/// Session statistics for monitoring and analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStatistics {
    pub session_id: String,
    pub total_messages: usize,
    pub user_messages: usize,
    pub assistant_messages: usize,
    pub session_duration: chrono::Duration,
    pub last_activity: chrono::DateTime<chrono::Utc>,
    pub topic_count: usize,
    pub pinned_messages: usize,
    pub context_compressions: usize,
    pub average_response_time: f64,
    pub query_success_rate: f32,
    pub user_satisfaction: f32,
}
