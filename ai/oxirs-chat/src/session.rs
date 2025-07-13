//! Session management for OxiRS Chat
//!
//! This module handles chat sessions, context windows, and session persistence.

use crate::context::ContextSummary;
use crate::types::*;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc, time::SystemTime};
use tracing::{debug, info, warn};

/// Persistent session data
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
    pub pinned_messages: Vec<String>,
    pub current_topics: Vec<Topic>,
    pub topic_history: Vec<TopicTransition>,
    pub performance_metrics: SessionMetrics,
}

/// Session state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionState {
    Active,
    Idle,
    Expired,
    Suspended,
}

/// Session performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetrics {
    pub total_queries: usize,
    pub successful_queries: usize,
    pub failed_queries: usize,
    pub average_response_time_ms: f64,
    pub total_tokens_processed: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub last_query_time: Option<chrono::DateTime<chrono::Utc>>,
}

impl Default for SessionMetrics {
    fn default() -> Self {
        Self {
            total_queries: 0,
            successful_queries: 0,
            failed_queries: 0,
            average_response_time_ms: 0.0,
            total_tokens_processed: 0,
            cache_hits: 0,
            cache_misses: 0,
            last_query_time: None,
        }
    }
}

/// Context window for managing conversation memory
#[derive(Debug, Clone)]
pub struct ContextWindow {
    pub window_size: usize,
    pub pinned_messages: Vec<String>,
    pub context_summary: Option<String>,
}

impl ContextWindow {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            pinned_messages: Vec::new(),
            context_summary: None,
        }
    }

    pub fn should_summarize(&self, total_messages: usize) -> bool {
        total_messages > self.window_size * 2
    }

    pub fn update_summary(&mut self, summary: String) {
        self.context_summary = Some(summary);
    }

    pub fn pin_message(&mut self, message_id: String) {
        if !self.pinned_messages.contains(&message_id) {
            self.pinned_messages.push(message_id);
        }
    }

    pub fn unpin_message(&mut self, message_id: &str) {
        self.pinned_messages.retain(|id| id != message_id);
    }

    pub fn get_context_messages<'a>(&self, messages: &'a [Message]) -> Vec<&'a Message> {
        // Return the last window_size messages plus any pinned messages
        let mut context_messages = Vec::new();

        // Add pinned messages first
        for message in messages {
            if self.pinned_messages.contains(&message.id) {
                context_messages.push(message);
            }
        }

        // Add recent messages (avoiding duplicates)
        let recent_start = if messages.len() > self.window_size {
            messages.len() - self.window_size
        } else {
            0
        };

        for message in &messages[recent_start..] {
            if !self.pinned_messages.contains(&message.id) {
                context_messages.push(message);
            }
        }

        context_messages
    }
}

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

    pub fn is_expired(&self) -> bool {
        let elapsed = chrono::Utc::now() - self.last_activity;
        elapsed > chrono::Duration::from_std(self.config.session_timeout).unwrap_or_default()
    }

    pub fn add_reaction(
        &mut self,
        message_id: &str,
        reaction: ReactionType,
        user_id: Option<String>,
    ) -> Result<()> {
        if let Some(message) = self.messages.iter_mut().find(|m| m.id == message_id) {
            message.reactions.push(MessageReaction {
                reaction_type: reaction,
                user_id,
                timestamp: chrono::Utc::now(),
            });
            Ok(())
        } else {
            Err(anyhow::anyhow!("Message not found"))
        }
    }

    pub async fn add_message(&mut self, mut message: Message) -> Result<()> {
        // Set metadata
        if message.metadata.is_none() {
            message.metadata = Some(MessageMetadata {
                session_id: self.id.clone(),
                turn_number: self.messages.len() + 1,
                processing_time_ms: None,
                retrieval_results: None,
                sparql_query: None,
                confidence_score: None,
                intent: None,
                entities: vec![],
                topics: vec![],
                quality_score: None,
                user_feedback: None,
                error_details: None,
            });
        }

        // Update topic tracking
        if let Some(transition) = self.topic_tracker.analyze_message(&message) {
            debug!("Topic transition detected: {:?}", transition);
        }

        // Update performance metrics
        self.performance_metrics.total_queries += 1;

        // Add message to session
        self.messages.push(message);

        // Update context window if needed
        if self.context_window.should_summarize(self.messages.len()) {
            debug!("Context summarization needed for session {}", self.id);

            // Implement context summarization
            match self.perform_context_summarization().await {
                Ok(summary) => {
                    info!(
                        "Context summarization completed for session {}: {} key points identified",
                        self.id,
                        summary.key_points.len()
                    );
                    self.context_window.update_summary(summary.text);
                }
                Err(e) => {
                    warn!(
                        "Context summarization failed for session {}: {}",
                        self.id, e
                    );
                }
            }
        }

        self.update_activity();
        Ok(())
    }

    pub fn get_context_messages(&self) -> Vec<&Message> {
        self.context_window.get_context_messages(&self.messages)
    }

    pub fn pin_message(&mut self, message_id: String) {
        self.context_window.pin_message(message_id);
    }

    pub fn unpin_message(&mut self, message_id: &str) {
        self.context_window.unpin_message(message_id);
    }

    pub fn update_preference(&mut self, key: String, value: String) {
        self.user_preferences.insert(key, value);
        self.update_activity();
    }

    pub fn get_preference(&self, key: &str) -> Option<&String> {
        self.user_preferences.get(key)
    }

    pub fn suspend(&mut self) {
        self.session_state = SessionState::Suspended;
    }

    pub fn resume(&mut self) {
        self.session_state = SessionState::Active;
        self.update_activity();
    }

    pub fn expire(&mut self) {
        self.session_state = SessionState::Expired;
    }

    pub fn get_message_by_id(&self, message_id: &str) -> Option<&Message> {
        self.messages.iter().find(|m| m.id == message_id)
    }

    pub fn get_message_by_id_mut(&mut self, message_id: &str) -> Option<&mut Message> {
        self.messages.iter_mut().find(|m| m.id == message_id)
    }

    pub fn delete_message(&mut self, message_id: &str) -> Result<()> {
        let initial_len = self.messages.len();
        self.messages.retain(|m| m.id != message_id);

        if self.messages.len() < initial_len {
            // Also remove from pinned messages if present
            self.context_window.unpin_message(message_id);
            self.update_activity();
            Ok(())
        } else {
            Err(anyhow::anyhow!("Message not found"))
        }
    }

    pub fn get_thread_messages(&self, thread_id: &str) -> Vec<&Message> {
        self.messages
            .iter()
            .filter(|m| m.thread_id.as_ref() == Some(&thread_id.to_string()))
            .collect()
    }

    pub fn create_thread(&mut self, parent_message_id: String) -> String {
        let thread_id = uuid::Uuid::new_v4().to_string();

        // Update the parent message to start the thread
        if let Some(parent_message) = self.get_message_by_id_mut(&parent_message_id) {
            parent_message.thread_id = Some(thread_id.clone());
        }

        thread_id
    }

    pub fn get_statistics(&self) -> SessionStatistics {
        let user_messages = self
            .messages
            .iter()
            .filter(|m| m.role == MessageRole::User)
            .count();
        let assistant_messages = self
            .messages
            .iter()
            .filter(|m| m.role == MessageRole::Assistant)
            .count();
        let system_messages = self
            .messages
            .iter()
            .filter(|m| m.role == MessageRole::System)
            .count();

        let total_tokens = self.messages.iter().filter_map(|m| m.token_count).sum();

        let pinned_count = self.context_window.pinned_messages.len();
        let thread_count = self
            .messages
            .iter()
            .filter_map(|m| m.thread_id.as_ref())
            .collect::<std::collections::HashSet<_>>()
            .len();

        SessionStatistics {
            total_messages: self.messages.len(),
            user_messages,
            assistant_messages,
            system_messages,
            total_tokens,
            pinned_messages: pinned_count,
            thread_count,
            session_age: chrono::Utc::now() - self.created_at,
            last_activity_ago: chrono::Utc::now() - self.last_activity,
            current_topics_count: self.topic_tracker.current_topics.len(),
            topic_transitions: self.topic_tracker.topic_history.len(),
        }
    }

    /// Perform intelligent context summarization
    async fn perform_context_summarization(&self) -> Result<ContextSummary> {
        debug!("Starting context summarization for session {}", self.id);

        // Get messages that need to be summarized
        let messages_to_summarize = self.get_messages_for_summarization();

        if messages_to_summarize.is_empty() {
            return Err(anyhow::anyhow!("No messages available for summarization"));
        }

        // Extract key information from messages
        let key_points = self.extract_key_points(&messages_to_summarize);
        let entities_mentioned = self.extract_entities(&messages_to_summarize);
        let topics_covered = self.extract_topics(&messages_to_summarize);

        // Generate summary text
        let summary_text = self.generate_summary_text(&key_points, &topics_covered);

        let summary = ContextSummary {
            text: summary_text,
            key_points,
            entities_mentioned,
            topics_covered,
            created_at: SystemTime::now(),
        };

        debug!(
            "Context summarization completed: {} key points, {} entities, {} topics",
            summary.key_points.len(),
            summary.entities_mentioned.len(),
            summary.topics_covered.len()
        );

        Ok(summary)
    }

    /// Get messages that should be summarized (older messages not in current window)
    fn get_messages_for_summarization(&self) -> Vec<&Message> {
        let window_size = self.config.sliding_window_size;
        let total_messages = self.messages.len();

        if total_messages <= window_size {
            return Vec::new();
        }

        // Get older messages that aren't pinned
        let cutoff_index = total_messages.saturating_sub(window_size);
        self.messages[..cutoff_index]
            .iter()
            .filter(|msg| !self.context_window.pinned_messages.contains(&msg.id))
            .collect()
    }

    /// Extract key points from messages using simple heuristics
    fn extract_key_points(&self, messages: &[&Message]) -> Vec<String> {
        let mut key_points = Vec::new();

        for message in messages {
            let content = message.content.to_text();

            // Look for questions (potential important topics)
            if content.contains('?') {
                key_points.push(format!("Question: {}", Self::truncate_text(content, 100)));
            }

            // Look for statements with high importance indicators
            if content.contains("important")
                || content.contains("crucial")
                || content.contains("key")
            {
                key_points.push(format!("Important: {}", Self::truncate_text(content, 100)));
            }

            // Look for conclusions or decisions
            if content.contains("conclusion")
                || content.contains("decided")
                || content.contains("result")
            {
                key_points.push(format!("Decision: {}", Self::truncate_text(content, 100)));
            }
        }

        // Limit to most recent key points
        key_points.truncate(10);
        key_points
    }

    /// Extract entities mentioned in messages
    fn extract_entities(&self, messages: &[&Message]) -> Vec<String> {
        let mut entities = std::collections::HashSet::new();

        for message in messages {
            let content = message.content.to_text();

            // Simple entity extraction based on capitalization and common patterns
            let words: Vec<&str> = content.split_whitespace().collect();
            for window in words.windows(2) {
                if let [first, second] = window {
                    // Look for capitalized words that might be entities
                    if first.chars().next().is_some_and(|c| c.is_uppercase())
                        && first.len() > 2
                        && !first.ends_with('.')
                    {
                        entities.insert(first.to_string());
                    }

                    // Look for two-word entities
                    if first.chars().next().is_some_and(|c| c.is_uppercase())
                        && second.chars().next().is_some_and(|c| c.is_uppercase())
                    {
                        entities.insert(format!("{first} {second}"));
                    }
                }
            }
        }

        entities.into_iter().take(20).collect()
    }

    /// Extract topics from messages
    fn extract_topics(&self, messages: &[&Message]) -> Vec<String> {
        let mut topic_words = std::collections::HashMap::new();

        for message in messages {
            let content = message.content.to_text().to_lowercase();

            // Count significant words (exclude common words)
            let words: Vec<&str> = content
                .split_whitespace()
                .filter(|w| w.len() > 4 && !Self::is_common_word(w))
                .collect();

            for word in words {
                *topic_words.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        // Get most frequent topics
        let mut topics: Vec<(String, usize)> = topic_words.into_iter().collect();
        topics.sort_by(|a, b| b.1.cmp(&a.1));

        topics.into_iter().take(8).map(|(word, _)| word).collect()
    }

    /// Generate summary text from key points and topics
    fn generate_summary_text(&self, key_points: &[String], topics: &[String]) -> String {
        let mut summary = String::new();

        if !topics.is_empty() {
            summary.push_str("Main topics discussed: ");
            summary.push_str(&topics.join(", "));
            summary.push_str(". ");
        }

        if !key_points.is_empty() {
            summary.push_str("Key points: ");
            for (i, point) in key_points.iter().enumerate() {
                if i > 0 {
                    summary.push_str("; ");
                }
                summary.push_str(point);
            }
            summary.push('.');
        }

        if summary.is_empty() {
            summary = "General conversation with various topics discussed.".to_string();
        }

        summary
    }

    /// Helper function to truncate text
    fn truncate_text(text: &str, max_length: usize) -> String {
        if text.len() <= max_length {
            text.to_string()
        } else {
            format!("{}...", &text[..max_length.saturating_sub(3)])
        }
    }

    /// Check if a word is a common word that shouldn't be considered a topic
    fn is_common_word(word: &str) -> bool {
        matches!(
            word,
            "this"
                | "that"
                | "with"
                | "from"
                | "they"
                | "were"
                | "been"
                | "have"
                | "their"
                | "said"
                | "each"
                | "which"
                | "about"
                | "would"
                | "there"
                | "could"
                | "other"
                | "after"
                | "first"
                | "well"
                | "water"
                | "very"
                | "what"
                | "know"
                | "while"
                | "here"
                | "think"
                | "also"
                | "its"
                | "now"
                | "find"
                | "any"
                | "may"
                | "say"
                | "these"
                | "some"
                | "time"
                | "people"
                | "take"
                | "year"
                | "your"
                | "good"
                | "make"
                | "way"
                | "work"
                | "life"
                | "day"
                | "get"
                | "use"
                | "man"
                | "new"
                | "write"
                | "our"
                | "out"
                | "go"
                | "come"
                | "see"
                | "than"
                | "call"
                | "who"
                | "oil"
                | "sit"
                | "set"
                | "run"
                | "eat"
                | "far"
                | "sea"
                | "eye"
        )
    }
}

/// Session statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStatistics {
    pub total_messages: usize,
    pub user_messages: usize,
    pub assistant_messages: usize,
    pub system_messages: usize,
    pub total_tokens: usize,
    pub pinned_messages: usize,
    pub thread_count: usize,
    pub session_age: chrono::Duration,
    pub last_activity_ago: chrono::Duration,
    pub current_topics_count: usize,
    pub topic_transitions: usize,
}

/// Topic tracking and analysis
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

    /// Analyze a message for topic changes with enhanced drift detection
    pub fn analyze_message(&mut self, message: &Message) -> Option<TopicTransition> {
        let detected_topics = self.extract_topics(message.content.to_text());

        if detected_topics.is_empty() {
            return None;
        }

        // Calculate similarity with current topics
        let mut max_similarity = 0.0;
        let mut most_similar_topic = None;

        for current_topic in &self.current_topics {
            for detected_topic in &detected_topics {
                let similarity = self.calculate_topic_similarity(current_topic, detected_topic);
                if similarity > max_similarity {
                    max_similarity = similarity;
                    most_similar_topic = Some(current_topic.clone());
                }
            }
        }

        // Determine if this is a topic shift
        let transition_type = if max_similarity < self.confidence_threshold {
            if self.current_topics.is_empty() {
                TransitionType::Introduction
            } else {
                TransitionType::Shift
            }
        } else if max_similarity > 0.8 {
            TransitionType::Continuation
        } else {
            TransitionType::Evolution
        };

        // Update current topics
        match transition_type {
            TransitionType::Introduction | TransitionType::Shift => {
                self.current_topics = detected_topics.clone();
            }
            TransitionType::Evolution => {
                // Merge topics
                for detected_topic in &detected_topics {
                    if !self
                        .current_topics
                        .iter()
                        .any(|t| t.name == detected_topic.name)
                    {
                        self.current_topics.push(detected_topic.clone());
                    }
                }
                // Limit topic count
                if self.current_topics.len() > self.max_topics {
                    self.current_topics.truncate(self.max_topics);
                }
            }
            TransitionType::Continuation => {
                // Update confidence scores
                for current_topic in &mut self.current_topics {
                    for detected_topic in &detected_topics {
                        if current_topic.name == detected_topic.name {
                            current_topic.confidence =
                                (current_topic.confidence + detected_topic.confidence) / 2.0;
                        }
                    }
                }
            }
        }

        // Create transition record
        let transition = TopicTransition {
            id: uuid::Uuid::new_v4().to_string(),
            from_topics: if let Some(topic) = most_similar_topic {
                vec![topic]
            } else {
                vec![]
            },
            to_topics: detected_topics.clone(),
            transition_type: transition_type.clone(),
            confidence: max_similarity,
            timestamp: chrono::Utc::now(),
            message_id: message.id.clone(),
            context: message
                .content
                .to_text()
                .chars()
                .take(100)
                .collect::<String>(),
            drift_magnitude: self.calculate_drift_magnitude(&detected_topics),
        };

        self.topic_history.push(transition.clone());

        // Limit history size
        if self.topic_history.len() > 100 {
            self.topic_history.remove(0);
        }

        Some(transition)
    }

    fn extract_topics(&self, content: &str) -> Vec<Topic> {
        // Simple keyword-based topic extraction
        // In production, this would use more sophisticated NLP
        let mut topics = Vec::new();

        let keywords = [
            ("sparql", TopicCategory::Query),
            ("query", TopicCategory::Query),
            ("data", TopicCategory::Data),
            ("graph", TopicCategory::Graph),
            ("ontology", TopicCategory::Ontology),
            ("semantic", TopicCategory::Ontology),
            ("rdf", TopicCategory::Data),
            ("knowledge", TopicCategory::General),
            ("search", TopicCategory::General),
            ("find", TopicCategory::General),
        ];

        let content_lower = content.to_lowercase();

        for (keyword, category) in &keywords {
            if content_lower.contains(keyword) {
                topics.push(Topic {
                    id: uuid::Uuid::new_v4().to_string(),
                    name: keyword.to_string(),
                    category: category.clone(),
                    confidence: 0.7, // Simple confidence based on presence
                    keywords: vec![keyword.to_string()],
                    description: None,
                    first_mentioned: chrono::Utc::now(),
                    last_mentioned: chrono::Utc::now(),
                    mention_count: 1,
                });
            }
        }

        topics
    }

    fn calculate_topic_similarity(&self, topic1: &Topic, topic2: &Topic) -> f32 {
        // Simple keyword overlap similarity
        let keywords1: std::collections::HashSet<_> = topic1.keywords.iter().collect();
        let keywords2: std::collections::HashSet<_> = topic2.keywords.iter().collect();

        let intersection = keywords1.intersection(&keywords2).count();
        let union = keywords1.union(&keywords2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    fn calculate_drift_magnitude(&self, new_topics: &[Topic]) -> f32 {
        if self.current_topics.is_empty() {
            return 1.0;
        }

        let mut total_similarity = 0.0;
        let mut count = 0;

        for current_topic in &self.current_topics {
            for new_topic in new_topics {
                total_similarity += self.calculate_topic_similarity(current_topic, new_topic);
                count += 1;
            }
        }

        if count == 0 {
            1.0
        } else {
            1.0 - (total_similarity / count as f32)
        }
    }
}

/// Topic representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topic {
    pub id: String,
    pub name: String,
    pub category: TopicCategory,
    pub confidence: f32,
    pub keywords: Vec<String>,
    pub description: Option<String>,
    pub first_mentioned: chrono::DateTime<chrono::Utc>,
    pub last_mentioned: chrono::DateTime<chrono::Utc>,
    pub mention_count: usize,
}

impl Topic {
    pub fn new(name: String, category: TopicCategory) -> Self {
        let now = chrono::Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.clone(),
            category,
            confidence: 0.5,
            keywords: vec![name],
            description: None,
            first_mentioned: now,
            last_mentioned: now,
            mention_count: 1,
        }
    }

    pub fn update_mention(&mut self) {
        self.last_mentioned = chrono::Utc::now();
        self.mention_count += 1;
    }

    pub fn add_keyword(&mut self, keyword: String) {
        if !self.keywords.contains(&keyword) {
            self.keywords.push(keyword);
        }
    }

    pub fn boost_confidence(&mut self, amount: f32) {
        self.confidence = (self.confidence + amount).min(1.0);
    }

    pub fn decay_confidence(&mut self, amount: f32) {
        self.confidence = (self.confidence - amount).max(0.0);
    }
}

/// Topic categories for classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TopicCategory {
    Query,
    Data,
    Graph,
    Ontology,
    Technical,
    General,
    Meta, // Questions about the system itself
}

/// Topic transition tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicTransition {
    pub id: String,
    pub from_topics: Vec<Topic>,
    pub to_topics: Vec<Topic>,
    pub transition_type: TransitionType,
    pub confidence: f32,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub message_id: String,
    pub context: String,
    pub drift_magnitude: f32,
}

/// Types of topic transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransitionType {
    Introduction, // First topic in conversation
    Continuation, // Same topic continues
    Evolution,    // Topic evolves/expands
    Shift,        // Complete topic change
}

/// Session-related statistics and metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SessionStats {
    pub total_sessions: usize,
    pub active_sessions: usize,
    pub idle_sessions: usize,
    pub expired_sessions: usize,
    pub suspended_sessions: usize,
    pub total_messages: usize,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct DetailedSessionMetrics {
    pub total_sessions: usize,
    pub active_sessions: usize,
    pub idle_sessions: usize,
    pub expired_sessions: usize,
    pub suspended_sessions: usize,
    pub locked_sessions: usize,
    pub high_error_sessions: usize,
    pub sessions_with_pinned_messages: usize,
    pub total_messages: usize,
    pub max_messages_per_session: usize,
    pub total_tokens: usize,
    pub average_response_time: f64,
    pub response_time_samples: usize,
    pub total_session_age: chrono::Duration,
    pub average_session_age: chrono::Duration,
}

/// Thread information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadInfo {
    pub id: String,
    pub message_count: usize,
    pub first_message_time: chrono::DateTime<chrono::Utc>,
    pub last_message_time: chrono::DateTime<chrono::Utc>,
    pub participants: Vec<String>,
}
