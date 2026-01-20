//! Chat session implementation with message history and statistics

use crate::error::Result;
use crate::messages::{Message, MessageRole};
use crate::session_manager::{SessionData, SessionMetrics, SessionState};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Chat session managing a single conversation
pub struct ChatSession {
    /// Session identifier
    pub id: String,
    /// Messages in this session
    pub messages: Vec<Message>,
    /// Session creation time
    pub created_at: DateTime<Utc>,
    /// Last activity time
    pub last_activity: DateTime<Utc>,
    /// Session metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Session state
    pub state: SessionState,
    /// Reference to the RDF store
    pub store: Arc<dyn oxirs_core::Store>,
    /// Session metrics
    pub metrics: SessionMetrics,
}

impl ChatSession {
    /// Create a new chat session
    pub fn new(id: String, store: Arc<dyn oxirs_core::Store>) -> Self {
        Self {
            id,
            messages: Vec::new(),
            created_at: Utc::now(),
            last_activity: Utc::now(),
            metadata: HashMap::new(),
            state: SessionState::Active,
            store,
            metrics: SessionMetrics::default(),
        }
    }

    /// Add a message to the session
    pub fn add_message(&mut self, message: Message) -> Result<()> {
        // Update metrics
        self.metrics.total_messages += 1;
        match message.role {
            MessageRole::User => self.metrics.user_messages += 1,
            MessageRole::Assistant => self.metrics.assistant_messages += 1,
            _ => {}
        }

        if let Some(tokens) = message.token_count {
            self.metrics.total_tokens_used += tokens;
        }

        self.messages.push(message);
        self.last_activity = Utc::now();
        self.metrics.last_updated = Utc::now();
        Ok(())
    }

    /// Get all messages in the session
    pub fn get_messages(&self) -> &[Message] {
        &self.messages
    }

    /// Get recent messages (last N)
    pub fn get_recent_messages(&self, count: usize) -> &[Message] {
        let start = self.messages.len().saturating_sub(count);
        &self.messages[start..]
    }

    /// Get statistics for this session
    pub fn get_statistics(&self) -> SessionStatistics {
        let duration = (Utc::now() - self.created_at).num_seconds() as u64;

        SessionStatistics {
            session_id: self.id.clone(),
            total_messages: self.messages.len(),
            user_messages: self.metrics.user_messages,
            assistant_messages: self.metrics.assistant_messages,
            total_tokens: self.metrics.total_tokens_used,
            avg_response_time_ms: self.metrics.average_response_time * 1000.0, // Convert to ms
            session_duration_seconds: duration,
            rag_retrievals: self.metrics.successful_queries, // Map to successful_queries
            sparql_queries: self.metrics.successful_queries,
            error_count: self.metrics.error_count,
            created_at: self.created_at,
            last_activity: self.last_activity,
        }
    }

    /// Check if session should expire
    pub fn should_expire(&self, timeout: Duration) -> bool {
        Utc::now() - self.last_activity > timeout
    }

    /// Export session data for persistence
    pub fn export_data(&self) -> SessionData {
        use crate::session_manager::ChatConfig;
        use std::collections::HashSet;

        SessionData {
            id: self.id.clone(),
            config: ChatConfig::default(),
            created_at: self.created_at,
            last_activity: self.last_activity,
            messages: self.messages.clone(),
            user_preferences: self
                .metadata
                .iter()
                .map(|(k, v)| (k.clone(), v.to_string()))
                .collect(),
            session_state: self.state.clone(),
            context_summary: None,
            pinned_messages: HashSet::new(),
            current_topics: Vec::new(),
            topic_history: Vec::new(),
            performance_metrics: self.metrics.clone(),
        }
    }

    /// Convert session to data format for persistence
    pub fn to_data(&self) -> SessionData {
        self.export_data()
    }

    /// Create session from data
    pub fn from_data(data: SessionData, store: Arc<dyn oxirs_core::Store>) -> Self {
        let metadata: HashMap<String, serde_json::Value> = data
            .user_preferences
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
            .collect();

        Self {
            id: data.id,
            messages: data.messages,
            created_at: data.created_at,
            last_activity: data.last_activity,
            metadata,
            state: data.session_state,
            store,
            metrics: data.performance_metrics,
        }
    }

    /// Update a metric value
    pub fn record_rag_retrieval(&mut self) {
        self.metrics.successful_queries += 1;
        self.metrics.last_updated = Utc::now();
    }

    pub fn record_sparql_query(&mut self) {
        self.metrics.successful_queries += 1;
        self.metrics.last_updated = Utc::now();
    }

    pub fn record_error(&mut self) {
        self.metrics.error_count += 1;
        self.metrics.failed_queries += 1;
        self.metrics.last_updated = Utc::now();
    }

    pub fn record_response_time(&mut self, response_time_ms: u64) {
        self.metrics.update_response_time(response_time_ms);
        self.metrics.last_updated = Utc::now();
    }
}

/// Session statistics for analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStatistics {
    pub session_id: String,
    pub total_messages: usize,
    pub user_messages: usize,
    pub assistant_messages: usize,
    pub total_tokens: usize,
    pub avg_response_time_ms: f64,
    pub session_duration_seconds: u64,
    pub rag_retrievals: usize,
    pub sparql_queries: usize,
    pub error_count: usize,
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_creation() {
        let store = Arc::new(oxirs_core::ConcreteStore::new().unwrap());
        let session = ChatSession::new("test-session".to_string(), store);
        assert_eq!(session.id, "test-session");
        assert_eq!(session.messages.len(), 0);
        assert_eq!(session.state, SessionState::Active);
    }
}
