//! Chat session implementation and management

use crate::messages::{Message, MessageRole};
use crate::session_manager::{
    ChatConfig, ContextWindow, SessionData, SessionMetrics, SessionState, TopicTracker,
};
use crate::types::*;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use tracing::info;

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

    pub fn add_message(&mut self, message: Message) -> Result<()> {
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
        let content_lower = message.content.to_lowercase();
        let content_text = message.content.to_text();

        // Enhanced importance scoring with multiple factors

        // 1. Question analysis (weighted by complexity)
        if content_text.contains('?') {
            let question_count = content_text.matches('?').count();
            importance += (0.2 * question_count as f32).min(0.4);

            // Complex questions get higher importance
            if content_lower.contains("how") || content_lower.contains("why") {
                importance += 0.1;
            }
        }

        // 2. Technical content analysis
        if content_lower.contains("select")
            || content_lower.contains("construct")
            || content_lower.contains("ask")
            || content_lower.contains("describe")
        {
            importance += 0.3;

            // Complex SPARQL queries get extra boost
            if content_lower.contains("optional")
                || content_lower.contains("union")
                || content_lower.contains("subquery")
            {
                importance += 0.2;
            }
        }

        // 3. Error and problem indicators
        if content_lower.contains("error")
            || content_lower.contains("fail")
            || content_lower.contains("exception")
            || content_lower.contains("bug")
        {
            importance += 0.4;

            // Stack traces and detailed errors are very important
            if content_text.lines().count() > 5 {
                importance += 0.2;
            }
        }

        // 4. Code and structured content
        if content_text.contains("```") {
            let code_blocks = content_text.matches("```").count() / 2;
            importance += (0.2 * code_blocks as f32).min(0.5);
        }

        // 5. Context references (mentions of previous topics)
        let context_references = self.count_context_references(message);
        importance += (context_references * 0.1).min(0.3);

        // 6. Semantic density (information content)
        let semantic_score = self.calculate_semantic_density(content_text);
        importance += semantic_score * 0.2;

        // 7. User engagement indicators
        if message.role == MessageRole::User {
            // Long, detailed user messages are important
            if content_text.len() > 500 {
                importance += 0.15;
            }

            // Messages with specific technical terms
            if self.contains_technical_terms(&content_lower) {
                importance += 0.25;
            }
        } else if message.role == MessageRole::Assistant {
            // Assistant messages with examples or solutions
            if content_lower.contains("example") || content_lower.contains("solution") {
                importance += 0.2;
            }
        }

        // 8. Temporal relevance (recent messages in active conversation)
        let time_since_last = chrono::Utc::now()
            .signed_duration_since(self.last_activity)
            .num_minutes();
        if time_since_last < 5 {
            importance += 0.1; // Recent activity boost
        }

        // 9. Topic transition points get higher importance (check without modifying tracker)
        // Note: We check for topic keywords rather than analyzing to avoid mutability issues
        let current_topics = &self.topic_tracker.current_topics;
        for topic in current_topics {
            for keyword in &topic.keywords {
                if content_lower.contains(keyword) {
                    importance += 0.1;
                    break;
                }
            }
        }

        importance.min(1.0)
    }

    /// Count references to previous context in the message
    fn count_context_references(&self, message: &Message) -> f32 {
        let content_lower = message.content.to_lowercase();
        let mut references = 0.0;

        // Check for explicit references
        if content_lower.contains("above")
            || content_lower.contains("previous")
            || content_lower.contains("earlier")
            || content_lower.contains("before")
        {
            references += 1.0;
        }

        // Check for topic continuity
        for topic in &self.topic_tracker.current_topics {
            for keyword in &topic.keywords {
                if content_lower.contains(keyword) {
                    references += 0.5;
                }
            }
        }

        references
    }

    /// Calculate semantic density of content
    fn calculate_semantic_density(&self, content: &str) -> f32 {
        let words: Vec<&str> = content.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }

        // Count unique technical terms and concepts
        let technical_words = words
            .iter()
            .filter(|word| self.is_technical_word(word))
            .count();

        // Calculate density ratio
        let density = technical_words as f32 / words.len() as f32;
        density.min(1.0)
    }

    /// Check if content contains technical terms
    fn contains_technical_terms(&self, content: &str) -> bool {
        let technical_terms = [
            "sparql",
            "rdf",
            "ontology",
            "triple",
            "graph",
            "query",
            "construct",
            "select",
            "where",
            "filter",
            "optional",
            "union",
            "prefix",
            "bind",
            "group",
            "order",
            "limit",
            "offset",
            "distinct",
            "reduced",
            "ask",
            "describe",
            "insert",
            "delete",
            "update",
            "dataset",
            "endpoint",
            "namespace",
            "uri",
            "iri",
            "literal",
            "blank",
            "node",
        ];

        technical_terms.iter().any(|term| content.contains(term))
    }

    /// Check if a word is technical/domain-specific
    fn is_technical_word(&self, word: &str) -> bool {
        let word_lower = word.to_lowercase();

        // RDF/SPARQL technical terms
        let technical_terms = [
            "sparql",
            "rdf",
            "owl",
            "rdfs",
            "xml",
            "json",
            "turtle",
            "ntriples",
            "triple",
            "quad",
            "graph",
            "ontology",
            "namespace",
            "prefix",
            "select",
            "construct",
            "ask",
            "describe",
            "where",
            "filter",
            "optional",
            "union",
            "bind",
            "group",
            "order",
            "limit",
            "offset",
        ];

        technical_terms.contains(&word_lower.as_str())
            || word_lower.starts_with("http")
            || word_lower.contains(':')
            || word_lower.ends_with("ql")
    }

    fn compress_context(&mut self) -> Result<()> {
        // Enhanced context compression with semantic analysis
        let context_message_ids = self.context_window.get_context_messages();
        let context_messages: Vec<Message> = self
            .messages
            .iter()
            .filter(|msg| context_message_ids.contains(&msg.id))
            .cloned()
            .collect();

        if context_messages.is_empty() {
            return Ok(());
        }

        // Analyze conversation patterns and extract key information
        let context_refs: Vec<&Message> = context_messages.iter().collect();
        let conversation_analysis = self.analyze_conversation_patterns(&context_refs);

        // Extract key technical terms and concepts
        let key_concepts = self.extract_key_concepts(&context_refs);

        // Identify important decisions and solutions
        let key_outcomes = self.extract_key_outcomes(&context_refs);

        // Analyze user preferences and interaction patterns
        let interaction_patterns = self.analyze_interaction_patterns(&context_refs);

        // Create intelligent summary
        let summary = self.create_intelligent_summary(
            conversation_analysis,
            key_concepts,
            key_outcomes,
            interaction_patterns,
        );

        // Preserve high-importance messages even after compression
        self.preserve_critical_messages(&context_refs[..]);

        self.context_window.compress_context(summary);
        self.performance_metrics.add_context_compression();

        info!(
            "Context compressed with enhanced semantic analysis. Preserved {} critical messages",
            self.context_window.pinned_messages.len()
        );

        Ok(())
    }

    /// Analyze conversation patterns for better context understanding
    fn analyze_conversation_patterns(&self, messages: &[&Message]) -> ConversationAnalysis {
        let mut analysis = ConversationAnalysis::default();

        // Analyze question-answer patterns
        let mut pending_questions = Vec::new();

        for message in messages {
            let content_text = message.content.to_text();
            let content_lower = content_text.to_lowercase();

            // Track questions
            if content_text.contains('?') && message.role == MessageRole::User {
                pending_questions.push(message.id.clone());
                analysis.question_count += 1;
            }

            // Track answers and solutions
            if message.role == MessageRole::Assistant {
                analysis.assistant_response_count += 1;

                // Check if this answers pending questions
                if content_lower.contains("answer")
                    || content_lower.contains("solution")
                    || content_lower.contains("result")
                {
                    analysis.resolved_questions += pending_questions.len();
                    pending_questions.clear();
                }
            }

            // Track error resolution patterns
            if content_lower.contains("error") || content_lower.contains("problem") {
                analysis.error_mentions += 1;
            }

            // Track code examples and technical solutions
            if content_text.contains("```") {
                analysis.code_examples += 1;
            }
        }

        analysis.unresolved_questions = pending_questions.len();
        analysis
    }

    /// Extract key technical concepts from conversation
    fn extract_key_concepts(&self, messages: &[&Message]) -> Vec<KeyConcept> {
        let mut concept_frequency: HashMap<String, usize> = HashMap::new();
        let mut concepts = Vec::new();

        for message in messages {
            let content_lower = message.content.to_lowercase();
            let words: Vec<&str> = content_lower.split_whitespace().collect();

            // Extract technical terms
            for word in words {
                if self.is_technical_word(word) {
                    *concept_frequency.entry(word.to_string()).or_insert(0) += 1;
                }
            }

            // Extract SPARQL queries as concepts
            if content_lower.contains("select") || content_lower.contains("construct") {
                let query_type = if content_lower.contains("select") {
                    "SELECT"
                } else {
                    "CONSTRUCT"
                };
                *concept_frequency
                    .entry(format!("{query_type} Query"))
                    .or_insert(0) += 1;
            }
        }

        // Convert to KeyConcept objects, keeping only frequently mentioned concepts
        for (concept, frequency) in concept_frequency {
            if frequency > 1 {
                concepts.push(KeyConcept {
                    name: concept,
                    frequency,
                    importance: (frequency as f32).ln().max(1.0),
                    context: "technical".to_string(),
                });
            }
        }

        // Sort by importance
        concepts.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());
        concepts.truncate(10); // Keep top 10 concepts
        concepts
    }

    /// Extract key outcomes and decisions from conversation
    fn extract_key_outcomes(&self, messages: &[&Message]) -> Vec<KeyOutcome> {
        let mut outcomes = Vec::new();

        for message in messages {
            let content_lower = message.content.to_lowercase();
            let content_text = message.content.to_text();

            // Identify solution messages
            if content_lower.contains("solution")
                || content_lower.contains("fix")
                || content_lower.contains("resolve")
            {
                outcomes.push(KeyOutcome {
                    description: self.extract_outcome_summary(content_text),
                    message_id: message.id.clone(),
                    outcome_type: OutcomeType::Solution,
                    confidence: 0.8,
                });
            }

            // Identify decisions
            if content_lower.contains("decide")
                || content_lower.contains("choose")
                || content_lower.contains("recommend")
            {
                outcomes.push(KeyOutcome {
                    description: self.extract_outcome_summary(content_text),
                    message_id: message.id.clone(),
                    outcome_type: OutcomeType::Decision,
                    confidence: 0.7,
                });
            }

            // Identify working examples/code
            if content_text.contains("```")
                && (content_lower.contains("work") || content_lower.contains("correct"))
            {
                outcomes.push(KeyOutcome {
                    description: "Working code example provided".to_string(),
                    message_id: message.id.clone(),
                    outcome_type: OutcomeType::Example,
                    confidence: 0.9,
                });
            }
        }

        outcomes
    }

    /// Analyze user interaction patterns for personalization
    fn analyze_interaction_patterns(&self, messages: &[&Message]) -> InteractionPatterns {
        let mut patterns = InteractionPatterns::default();

        for message in messages {
            let content_text = message.content.to_text();
            let content_lower = content_text.to_lowercase();

            if message.role == MessageRole::User {
                patterns.user_message_count += 1;
                patterns.average_user_message_length += content_text.len();

                // Analyze question complexity
                if content_text.contains('?') {
                    if content_lower.contains("how") || content_lower.contains("why") {
                        patterns.complex_questions += 1;
                    } else {
                        patterns.simple_questions += 1;
                    }
                }

                // Analyze technical depth
                if self.contains_technical_terms(&content_lower) {
                    patterns.technical_messages += 1;
                }
            }
        }

        if patterns.user_message_count > 0 {
            patterns.average_user_message_length /= patterns.user_message_count;
        }

        patterns
    }

    /// Create intelligent summary from analysis
    fn create_intelligent_summary(
        &self,
        conversation: ConversationAnalysis,
        concepts: Vec<KeyConcept>,
        outcomes: Vec<KeyOutcome>,
        patterns: InteractionPatterns,
    ) -> String {
        let mut summary_parts = Vec::new();

        // Conversation overview
        summary_parts.push(format!(
            "Conversation overview: {} questions asked, {} resolved, {} unresolved. {} code examples shared.",
            conversation.question_count,
            conversation.resolved_questions,
            conversation.unresolved_questions,
            conversation.code_examples
        ));

        // Key concepts
        if !concepts.is_empty() {
            let concept_names: Vec<String> = concepts.into_iter().take(5).map(|c| c.name).collect();
            summary_parts.push(format!("Key topics: {}", concept_names.join(", ")));
        }

        // Important outcomes
        if !outcomes.is_empty() {
            let solution_count = outcomes
                .iter()
                .filter(|o| matches!(o.outcome_type, OutcomeType::Solution))
                .count();
            if solution_count > 0 {
                summary_parts.push(format!("{solution_count} solutions provided"));
            }
        }

        // User patterns
        if patterns.technical_messages > 0 {
            summary_parts.push(format!(
                "User shows {} technical expertise ({}% technical messages)",
                if patterns.technical_messages as f32 / patterns.user_message_count as f32 > 0.7 {
                    "high"
                } else {
                    "moderate"
                },
                (patterns.technical_messages * 100 / patterns.user_message_count)
            ));
        }

        summary_parts.join(". ")
    }

    /// Preserve critical messages that shouldn't be compressed away
    fn preserve_critical_messages(&mut self, messages: &[&Message]) {
        for message in messages {
            let importance = self.calculate_message_importance(message);

            // Preserve high-importance messages
            if importance > 0.8 {
                self.context_window.pin_message(message.id.clone());
            }

            // Preserve messages with solutions
            let content_lower = message.content.to_lowercase();
            if content_lower.contains("solution")
                || content_lower.contains("fix")
                || (content_lower.contains("work") && message.content.to_text().contains("```"))
            {
                self.context_window.pin_message(message.id.clone());
            }
        }
    }

    /// Extract a brief summary from outcome text
    fn extract_outcome_summary(&self, text: &str) -> String {
        // Take first sentence or first 100 chars, whichever is shorter
        let first_sentence = text.lines().next().unwrap_or(text);
        if first_sentence.len() > 100 {
            format!("{}...", &first_sentence[..97])
        } else {
            first_sentence.to_string()
        }
    }

    pub fn get_context_for_query(&self) -> Vec<&Message> {
        let context_message_ids = self.context_window.get_context_messages();
        self.messages
            .iter()
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
