//! # OxiRS Chat
//!
//! RAG chat API with LLM integration and natural language to SPARQL translation.
//!
//! This crate provides a conversational interface for knowledge graphs,
//! combining retrieval-augmented generation (RAG) with SPARQL querying.

use crate::rag::QueryIntent;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime},
};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

pub mod analytics;
pub mod cache;
pub mod chat;
pub mod context;
pub mod llm;
pub mod nl2sparql;
pub mod performance;
pub mod persistence;
pub mod rag;
pub mod server;
pub mod sparql_optimizer;

/// Chat session configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatConfig {
    pub max_context_length: usize,
    pub temperature: f32,
    pub max_retrieval_results: usize,
    pub enable_sparql_generation: bool,
    pub session_timeout: Duration,
    pub max_conversation_turns: usize,
    pub enable_context_summarization: bool,
    pub sliding_window_size: usize,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            max_context_length: 4096,
            temperature: 0.7,
            max_retrieval_results: 10,
            enable_sparql_generation: true,
            session_timeout: Duration::from_secs(3600),
            max_conversation_turns: 100,
            enable_context_summarization: true,
            sliding_window_size: 20,
        }
    }
}

/// Chat message
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Message {
    pub id: String,
    pub role: MessageRole,
    pub content: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metadata: Option<MessageMetadata>,
    pub thread_id: Option<String>,
    pub parent_message_id: Option<String>,
    pub token_count: Option<usize>,
    pub reactions: Vec<MessageReaction>,
}

/// Message reaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageReaction {
    pub emoji: String,
    pub user_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Message role
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

/// Message metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MessageMetadata {
    pub sparql_query: Option<String>,
    pub retrieved_triples: Option<Vec<String>>,
    pub confidence_score: Option<f32>,
    pub processing_time_ms: Option<u64>,
    pub model_used: Option<String>,
    pub intent_classification: Option<String>,
    pub entities_extracted: Option<Vec<String>>,
    pub context_used: Option<bool>,
}

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

        // Add recent messages up to window size
        let recent_start = messages.len().saturating_sub(self.window_size);
        for message in &messages[recent_start..] {
            if !context_messages.iter().any(|m| m.id == message.id) {
                context_messages.push(message);
            }
        }

        context_messages
    }
}

/// Topic tracking for conversation analysis
#[derive(Debug, Clone)]
pub struct TopicTracker {
    pub current_topics: Vec<Topic>,
    pub topic_history: Vec<TopicTransition>,
    pub topic_threshold: f32,
}

impl TopicTracker {
    pub fn new() -> Self {
        Self {
            current_topics: Vec::new(),
            topic_history: Vec::new(),
            topic_threshold: 0.7,
        }
    }

    /// Analyze a message for topic changes with enhanced drift detection
    pub fn analyze_message(&mut self, message: &Message) -> Option<TopicTransition> {
        let detected_topics = self.extract_topics(&message.content);

        if detected_topics.is_empty() {
            return None;
        }

        // Enhanced topic drift detection
        let drift_result = self.detect_topic_drift(&detected_topics, message);

        match drift_result {
            Some(drift) => {
                let transition = TopicTransition {
                    id: Uuid::new_v4().to_string(),
                    from_topics: self.current_topics.clone(),
                    to_topics: detected_topics.clone(),
                    timestamp: chrono::Utc::now(),
                    trigger_message_id: message.id.clone(),
                    confidence: drift.confidence,
                    transition_type: drift.transition_type,
                };

                self.topic_history.push(transition.clone());
                self.current_topics = detected_topics;

                return Some(transition);
            }
            None => {
                // Update existing topics with new mentions
                self.update_topic_mentions(&detected_topics);
            }
        }

        None
    }

    /// Enhanced topic drift detection with multiple strategies
    fn detect_topic_drift(&self, new_topics: &[Topic], message: &Message) -> Option<TopicDrift> {
        if self.current_topics.is_empty() {
            return Some(TopicDrift {
                confidence: 0.9,
                transition_type: TransitionType::NewTopic,
                drift_magnitude: 1.0,
            });
        }

        // Calculate semantic similarity between topic sets
        let semantic_similarity =
            self.calculate_topic_set_similarity(&self.current_topics, new_topics);

        // Detect different types of drift
        if semantic_similarity < 0.3 {
            // Major topic shift
            Some(TopicDrift {
                confidence: 1.0 - semantic_similarity,
                transition_type: TransitionType::TopicShift,
                drift_magnitude: 1.0 - semantic_similarity,
            })
        } else if semantic_similarity < 0.7 {
            // Gradual topic drift
            let drift_strength = self.calculate_drift_strength(new_topics, message);
            if drift_strength > 0.6 {
                Some(TopicDrift {
                    confidence: drift_strength,
                    transition_type: TransitionType::TopicShift,
                    drift_magnitude: drift_strength,
                })
            } else {
                None
            }
        } else {
            // Check for topic return patterns
            self.detect_topic_return(new_topics)
                .map(|confidence| TopicDrift {
                    confidence,
                    transition_type: TransitionType::TopicReturn,
                    drift_magnitude: confidence,
                })
        }
    }

    /// Calculate semantic similarity between two topic sets
    fn calculate_topic_set_similarity(&self, topics1: &[Topic], topics2: &[Topic]) -> f32 {
        if topics1.is_empty() || topics2.is_empty() {
            return 0.0;
        }

        let mut total_similarity = 0.0;
        let mut pair_count = 0;

        for topic1 in topics1 {
            for topic2 in topics2 {
                total_similarity += topic1.similarity(topic2);
                pair_count += 1;
            }
        }

        if pair_count > 0 {
            total_similarity / pair_count as f32
        } else {
            0.0
        }
    }

    /// Calculate drift strength based on multiple factors
    fn calculate_drift_strength(&self, new_topics: &[Topic], message: &Message) -> f32 {
        let mut drift_strength = 0.0;

        // Factor 1: Presence of transition words/phrases
        let transition_indicators = [
            "but",
            "however",
            "on the other hand",
            "meanwhile",
            "speaking of",
            "by the way",
            "let me ask about",
            "what about",
            "moving on",
            "actually",
            "wait",
            "hold on",
            "before we continue",
        ];

        let content_lower = message.content.to_lowercase();
        for indicator in &transition_indicators {
            if content_lower.contains(indicator) {
                drift_strength += 0.2;
                break;
            }
        }

        // Factor 2: Question patterns that suggest topic change
        if content_lower.contains("?")
            && (content_lower.starts_with("what about")
                || content_lower.starts_with("how about")
                || content_lower.starts_with("can we talk about")
                || content_lower.starts_with("let's discuss"))
        {
            drift_strength += 0.3;
        }

        // Factor 3: Length of current topic session
        let current_topic_duration = self
            .current_topics
            .iter()
            .map(|t| chrono::Utc::now() - t.first_mentioned)
            .min()
            .unwrap_or_default();

        if current_topic_duration > chrono::Duration::minutes(10) {
            drift_strength += 0.2;
        }

        // Factor 4: Keyword novelty
        let existing_keywords: HashSet<String> = self
            .current_topics
            .iter()
            .flat_map(|t| t.keywords.iter())
            .cloned()
            .collect();

        let new_keywords: HashSet<String> = new_topics
            .iter()
            .flat_map(|t| t.keywords.iter())
            .cloned()
            .collect();

        let keyword_overlap = existing_keywords.intersection(&new_keywords).count();
        let total_keywords = existing_keywords.union(&new_keywords).count();

        if total_keywords > 0 {
            let novelty = 1.0 - (keyword_overlap as f32 / total_keywords as f32);
            drift_strength += novelty * 0.3;
        }

        drift_strength.min(1.0)
    }

    /// Detect if user is returning to a previous topic
    fn detect_topic_return(&self, new_topics: &[Topic]) -> Option<f32> {
        // Look through recent topic history
        let recent_window = chrono::Duration::minutes(30);
        let cutoff_time = chrono::Utc::now() - recent_window;

        for historical_topic in self.topic_history.iter().rev().take(10) {
            if historical_topic.timestamp < cutoff_time {
                break;
            }

            for old_topic in &historical_topic.from_topics {
                for new_topic in new_topics {
                    let similarity = old_topic.similarity(new_topic);
                    if similarity > 0.8 {
                        return Some(similarity);
                    }
                }
            }
        }

        None
    }

    /// Update mention counts and timestamps for existing topics
    fn update_topic_mentions(&mut self, detected_topics: &[Topic]) {
        for detected in detected_topics {
            for current in &mut self.current_topics {
                if current.similarity(detected) > self.topic_threshold {
                    current.mention_count += 1;
                    current.last_mentioned = chrono::Utc::now();
                    // Merge keywords if new ones are found
                    for keyword in &detected.keywords {
                        if !current.keywords.contains(keyword) {
                            current.keywords.push(keyword.clone());
                        }
                    }
                }
            }
        }
    }

    /// Extract topics from message content (simplified implementation)
    fn extract_topics(&self, content: &str) -> Vec<Topic> {
        let keywords = content
            .to_lowercase()
            .split_whitespace()
            .filter(|word| word.len() > 3)
            .filter(|word| {
                ![
                    "what", "this", "that", "have", "been", "will", "with", "from", "they", "them",
                    "were", "said", "each", "which", "their", "time", "about",
                ]
                .contains(word)
            })
            .take(3)
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        if keywords.is_empty() {
            return Vec::new();
        }

        vec![Topic {
            id: Uuid::new_v4().to_string(),
            name: keywords.join(" "),
            keywords,
            confidence: 0.6,
            category: TopicCategory::General,
            entities: Vec::new(),
            first_mentioned: chrono::Utc::now(),
            last_mentioned: chrono::Utc::now(),
            mention_count: 1,
        }]
    }
}

/// Topic representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topic {
    pub id: String,
    pub name: String,
    pub keywords: Vec<String>,
    pub confidence: f32,
    pub category: TopicCategory,
    pub entities: Vec<String>,
    pub first_mentioned: chrono::DateTime<chrono::Utc>,
    pub last_mentioned: chrono::DateTime<chrono::Utc>,
    pub mention_count: usize,
}

impl Topic {
    /// Calculate similarity with another topic
    pub fn similarity(&self, other: &Topic) -> f32 {
        let self_keywords: std::collections::HashSet<_> = self.keywords.iter().collect();
        let other_keywords: std::collections::HashSet<_> = other.keywords.iter().collect();

        let intersection = self_keywords.intersection(&other_keywords).count();
        let union = self_keywords.union(&other_keywords).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }
}

/// Topic category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TopicCategory {
    General,
    Technical,
    Business,
    Personal,
    Research,
    Education,
}

/// Topic transition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicTransition {
    pub id: String,
    pub from_topics: Vec<Topic>,
    pub to_topics: Vec<Topic>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub trigger_message_id: String,
    pub confidence: f32,
    pub transition_type: TransitionType,
}

/// Type of topic transition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransitionType {
    NewTopic,
    TopicShift,
    TopicReturn,
    TopicMerge,
    TopicSplit,
}

/// Topic drift analysis result
#[derive(Debug, Clone)]
struct TopicDrift {
    confidence: f32,
    transition_type: TransitionType,
    drift_magnitude: f32,
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
    store: Arc<oxirs_core::Store>,
}

impl ChatSession {
    pub fn new(id: String, store: Arc<oxirs_core::Store>) -> Self {
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

    pub fn from_data(data: SessionData, store: Arc<oxirs_core::Store>) -> Self {
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

    pub fn add_reaction(&mut self, message_id: &str, emoji: String, user_id: String) -> Result<()> {
        if let Some(message) = self.messages.iter_mut().find(|m| m.id == message_id) {
            message.reactions.push(MessageReaction {
                emoji,
                user_id,
                timestamp: chrono::Utc::now(),
            });
            Ok(())
        } else {
            Err(anyhow::anyhow!("Message not found"))
        }
    }

    /// Process a user message and generate a response
    pub async fn process_message(&mut self, user_input: String) -> Result<Message> {
        self.process_message_with_options(user_input, None, None)
            .await
    }

    pub async fn process_message_with_options(
        &mut self,
        user_input: String,
        thread_id: Option<String>,
        parent_message_id: Option<String>,
    ) -> Result<Message> {
        use crate::{
            llm::{ChatMessage, ChatRole, LLMConfig, LLMManager, LLMRequest, Priority, UseCase},
            nl2sparql::{NL2SPARQLConfig, NL2SPARQLSystem},
            rag::{AssembledContext, QueryContext, QueryIntent, RAGConfig, RAGSystem},
        };

        // Add user message to history
        let user_message_id = Uuid::new_v4().to_string();
        let user_message = Message {
            id: user_message_id.clone(),
            role: MessageRole::User,
            content: user_input.clone(),
            timestamp: chrono::Utc::now(),
            metadata: None,
            thread_id: thread_id.clone(),
            parent_message_id,
            token_count: Some(user_input.split_whitespace().count()),
            reactions: Vec::new(),
        };
        self.messages.push(user_message);

        // Enhanced RAG pipeline implementation
        let mut retrieved_triples: Option<Vec<String>> = None;
        let mut sparql_query: Option<String> = None;
        let mut confidence_score = 0.5f32;

        // Create initial query context (entities will be extracted by RAG system)
        let query_context = QueryContext {
            query: user_input.clone(),
            intent: self.classify_intent(&user_input),
            entities: Vec::new(), // Will be filled by RAG system's extract_query_components
            relationships: Vec::new(), // Will be filled by RAG system's extract_query_components
            constraints: Vec::new(), // Will be filled by RAG system's extract_query_components
            conversation_history: self
                .messages
                .iter()
                .take(5) // Last 5 messages for context
                .map(|m| m.content.clone())
                .collect(),
        };

        // Initialize enhanced RAG system with proper embedding configuration
        let rag_config = RAGConfig::default();
        let embedding_config = crate::rag::EmbeddingConfig {
            provider_type: crate::rag::EmbeddingProviderType::Local, // Use local by default for better reliability
            model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            api_key: std::env::var("OPENAI_API_KEY").ok(), // Fallback to OpenAI if available
            base_url: None,
            cache_size: 5000,
            batch_size: 50,
            timeout_seconds: 30,
        };

        let rag_system = match RAGSystem::with_enhanced_embeddings(
            rag_config.clone(),
            self.store.clone(),
            embedding_config,
        )
        .await
        {
            Ok(system) => system,
            Err(e) => {
                warn!(
                    "Failed to create enhanced RAG system, falling back to basic: {}",
                    e
                );
                // Fallback to basic RAG system
                RAGSystem::new(
                    rag_config,
                    self.store.clone(),
                    None, // Vector index not available yet
                    None, // Embedding model not available yet
                )
            }
        };

        // Step 1: Retrieve relevant knowledge
        let retrieved_knowledge = match rag_system.retrieve_knowledge(&query_context).await {
            Ok(knowledge) => {
                confidence_score = knowledge.metadata.quality_score;
                retrieved_triples = Some(
                    knowledge
                        .triples
                        .iter()
                        .map(|t| format!("{} {} {}", t.subject(), t.predicate(), t.object()))
                        .collect(),
                );
                Some(knowledge)
            }
            Err(e) => {
                tracing::warn!("RAG retrieval failed: {}", e);
                None
            }
        };

        // Step 2: Generate SPARQL query if appropriate
        if self.should_generate_sparql(&query_context.intent) {
            let nl2sparql_config = NL2SPARQLConfig::default();
            if let Ok(mut nl2sparql_system) =
                NL2SPARQLSystem::with_store(nl2sparql_config, None, self.store.clone())
            {
                if let Ok(sparql_result) = nl2sparql_system.generate_sparql(&query_context).await {
                    sparql_query = Some(sparql_result.query.clone());
                    confidence_score = confidence_score.max(sparql_result.confidence);

                    // Execute the SPARQL query if it was generated successfully
                    if sparql_result.validation_result.is_valid {
                        if let Ok(execution_result) = nl2sparql_system
                            .execute_sparql_query(&sparql_result.query)
                            .await
                        {
                            if execution_result.result_count > 0 {
                                info!(
                                    "SPARQL query returned {} results",
                                    execution_result.result_count
                                );
                                // TODO: Format and include SPARQL results in response
                            }
                        }
                    }
                }
            }
        }

        // Step 3: Assemble context for LLM
        let context_text = if let Some(knowledge) = retrieved_knowledge {
            match rag_system
                .assemble_context(&knowledge, &query_context)
                .await
            {
                Ok(context) => {
                    confidence_score = confidence_score.max(context.quality_score);
                    Some(context.context_text)
                }
                Err(_) => None,
            }
        } else {
            None
        };

        // Step 4: Generate response using LLM or fallback
        let response_content = if let Some(ref context) = context_text {
            self.generate_llm_response(&user_input, &context, sparql_query.as_ref())
                .await
                .unwrap_or_else(|_| self.generate_fallback_response(&user_input, &context))
        } else {
            self.generate_simple_response(&user_input)
        };

        let response = Message {
            id: Uuid::new_v4().to_string(),
            role: MessageRole::Assistant,
            content: response_content.clone(),
            timestamp: chrono::Utc::now(),
            metadata: Some(MessageMetadata {
                sparql_query,
                retrieved_triples,
                confidence_score: Some(confidence_score),
                processing_time_ms: None,
                model_used: None,
                intent_classification: Some(format!("{:?}", self.classify_intent(&user_input))),
                entities_extracted: None,
                context_used: Some(context_text.is_some()),
            }),
            thread_id,
            parent_message_id: Some(user_message_id),
            token_count: Some(response_content.split_whitespace().count()),
            reactions: Vec::new(),
        };

        self.messages.push(response.clone());

        // Update performance metrics
        self.performance_metrics.total_queries += 1;
        self.performance_metrics.successful_queries += 1;
        self.performance_metrics.total_tokens_processed += response.token_count.unwrap_or(0);
        self.performance_metrics.last_query_time = Some(chrono::Utc::now());

        // Analyze topic transitions
        if let Some(transition) = self.topic_tracker.analyze_message(&response) {
            tracing::info!("Topic transition detected: {:?}", transition);
        }

        // Check if we should summarize the conversation
        if self.config.enable_context_summarization
            && self.context_window.should_summarize(self.messages.len())
        {
            self.summarize_context().await;
        }

        Ok(response)
    }

    /// Classify the intent of a user query
    fn classify_intent(&self, query: &str) -> QueryIntent {
        let query_lower = query.to_lowercase();

        if query_lower.contains("what is") || query_lower.contains("who is") {
            QueryIntent::FactualLookup
        } else if query_lower.contains("how are") || query_lower.contains("relationship") {
            QueryIntent::Relationship
        } else if query_lower.contains("list") || query_lower.contains("show me") {
            QueryIntent::ListQuery
        } else if query_lower.contains("compare") || query_lower.contains("difference") {
            QueryIntent::Comparison
        } else if query_lower.contains("count") || query_lower.contains("how many") {
            QueryIntent::Aggregation
        } else if query_lower.contains("define") || query_lower.contains("meaning") {
            QueryIntent::Definition
        } else if query_lower.len() > 100 || query_lower.matches("and").count() > 2 {
            QueryIntent::Complex
        } else {
            QueryIntent::Exploration
        }
    }

    /// Determine if SPARQL generation is appropriate for this intent
    fn should_generate_sparql(&self, intent: &QueryIntent) -> bool {
        matches!(
            intent,
            QueryIntent::FactualLookup
                | QueryIntent::ListQuery
                | QueryIntent::Aggregation
                | QueryIntent::Relationship
        )
    }

    /// Generate response using LLM with context
    async fn generate_llm_response(
        &self,
        query: &str,
        context: &str,
        sparql_query: Option<&String>,
    ) -> Result<String> {
        // This would require LLM integration - for now return enhanced fallback
        let mut response = format!(
            "Based on the knowledge graph, here's what I found for your query: {}\n\n",
            query
        );

        if let Some(sparql) = sparql_query {
            response.push_str(&format!(
                "Generated SPARQL query:\n```sparql\n{}\n```\n\n",
                sparql
            ));
        }

        response.push_str("Relevant information:\n");
        response.push_str(context);

        Ok(response)
    }

    /// Generate fallback response with context
    fn generate_fallback_response(&self, query: &str, context: &str) -> String {
        format!(
            "I found some information related to your query '{}' in the knowledge graph:\n\n{}",
            query, context
        )
    }

    /// Generate simple response without context
    fn generate_simple_response(&self, query: &str) -> String {
        format!("I understand you're asking about: '{}'. Let me search the knowledge graph for relevant information.", query)
    }

    /// Get chat history
    pub fn get_history(&self) -> &[Message] {
        &self.messages
    }

    /// Clear chat history
    pub fn clear_history(&mut self) {
        self.messages.clear();
    }

    /// Get messages in a thread
    pub fn get_thread_messages(&self, thread_id: &str) -> Vec<&Message> {
        self.messages
            .iter()
            .filter(|m| m.thread_id.as_ref() == Some(&thread_id.to_string()))
            .collect()
    }

    /// Create a new thread from a message
    pub fn create_thread_from_message(&self, message_id: &str) -> Result<String> {
        if self.messages.iter().any(|m| m.id == message_id) {
            Ok(Uuid::new_v4().to_string())
        } else {
            Err(anyhow::anyhow!("Message not found"))
        }
    }

    /// Get reply chain for a message
    pub fn get_reply_chain(&self, message_id: &str) -> Vec<&Message> {
        let mut chain = Vec::new();
        let mut current_id = Some(message_id.to_string());

        while let Some(id) = current_id {
            if let Some(message) = self.messages.iter().find(|m| m.id == id) {
                chain.push(message);
                current_id = message.parent_message_id.clone();
            } else {
                break;
            }
        }

        chain.reverse();
        chain
    }

    /// Get all replies to a message
    pub fn get_replies(&self, message_id: &str) -> Vec<&Message> {
        self.messages
            .iter()
            .filter(|m| m.parent_message_id.as_ref() == Some(&message_id.to_string()))
            .collect()
    }

    /// Summarize the conversation context using LLM
    async fn summarize_context(&mut self) {
        // Get messages that need to be summarized
        let messages_to_summarize: Vec<&Message> = self
            .messages
            .iter()
            .take(
                self.messages
                    .len()
                    .saturating_sub(self.config.sliding_window_size),
            )
            .filter(|m| !self.context_window.pinned_messages.contains(&m.id))
            .collect();

        if messages_to_summarize.is_empty() {
            return;
        }

        // Try LLM-powered summarization first, fallback to simple if needed
        let summary = match self.create_llm_summary(&messages_to_summarize).await {
            Ok(llm_summary) => llm_summary,
            Err(e) => {
                warn!("LLM summarization failed ({}), using fallback", e);
                self.create_fallback_summary(&messages_to_summarize)
            }
        };

        info!(
            "Created conversation summary: {} messages -> {} chars",
            messages_to_summarize.len(),
            summary.len()
        );

        self.context_window.update_summary(summary);

        // Remove old messages from memory (keep them in persistence)
        let keep_from = self
            .messages
            .len()
            .saturating_sub(self.config.sliding_window_size);
        if keep_from > 0 {
            // Keep pinned messages and recent messages
            let mut new_messages = Vec::new();

            // Add pinned messages
            for message in &self.messages {
                if self.context_window.pinned_messages.contains(&message.id) {
                    new_messages.push(message.clone());
                }
            }

            // Add recent messages
            new_messages.extend_from_slice(&self.messages[keep_from..]);

            self.messages = new_messages;
        }
    }

    /// Create an LLM-powered conversation summary
    async fn create_llm_summary(&self, messages: &[&Message]) -> Result<String> {
        use crate::llm::{
            ChatMessage, ChatRole, LLMConfig, LLMManager, LLMRequest, Priority, UseCase,
        };

        // Format messages for summarization
        let conversation_text = messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    MessageRole::User => "User",
                    MessageRole::Assistant => "Assistant",
                    MessageRole::System => "System",
                };
                format!("{}: {}", role, m.content)
            })
            .collect::<Vec<_>>()
            .join("\n");

        // Create summarization prompt
        let prompt = format!(
            r#"Please create a concise summary of the following conversation. 
Focus on:
- Main topics discussed
- Key information shared
- Important questions asked and answered
- Context that would be valuable for continuing the conversation

Conversation to summarize:
{}

Summary:"#,
            conversation_text
        );

        // Initialize LLM manager
        let llm_config = LLMConfig::default();
        let llm_manager = LLMManager::new(llm_config)?;

        // Create summarization request (use fast, cost-effective model)
        let chat_messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: "You are a helpful assistant that creates concise, informative conversation summaries. Keep summaries under 200 words while preserving essential context.".to_string(),
                metadata: None,
            },
            ChatMessage {
                role: ChatRole::User,
                content: prompt,
                metadata: None,
            },
        ];

        let request = LLMRequest {
            messages: chat_messages,
            system_prompt: Some("Create concise, informative conversation summaries.".to_string()),
            use_case: UseCase::SimpleQuery, // Use efficient model for summarization
            priority: Priority::Low,        // Not urgent
            max_tokens: Some(300),          // Limit response length
            temperature: 0.3f32,            // Lower creativity for consistent summaries
            timeout: Some(Duration::from_secs(30)),
        };

        // Send request and get summary
        let response = llm_manager.generate_response(request).await?;

        // Add metadata to summary
        let enhanced_summary = format!(
            "Conversation Summary ({} messages from {} to {}):\n\n{}",
            messages.len(),
            messages
                .first()
                .map(|m| m.timestamp.format("%Y-%m-%d %H:%M UTC").to_string())
                .unwrap_or_default(),
            messages
                .last()
                .map(|m| m.timestamp.format("%Y-%m-%d %H:%M UTC").to_string())
                .unwrap_or_default(),
            response.content.trim()
        );

        Ok(enhanced_summary)
    }

    /// Create a fallback summary when LLM is unavailable
    fn create_fallback_summary(&self, messages: &[&Message]) -> String {
        // Analyze message patterns for better fallback summary
        let mut _topics: std::collections::HashMap<String, i32> = std::collections::HashMap::new();
        let mut user_questions = Vec::new();
        let mut key_terms = std::collections::HashMap::new();

        for message in messages {
            let content_lower = message.content.to_lowercase();

            // Extract questions
            if message.role == MessageRole::User
                && (content_lower.contains("?")
                    || content_lower.starts_with("what")
                    || content_lower.starts_with("how")
                    || content_lower.starts_with("when")
                    || content_lower.starts_with("where")
                    || content_lower.starts_with("why")
                    || content_lower.starts_with("who"))
            {
                user_questions.push(message.content.clone());
            }

            // Extract key terms (simple keyword extraction)
            let words: Vec<&str> = content_lower
                .split_whitespace()
                .filter(|w| w.len() > 4 && !self.is_stop_word(w))
                .collect();

            for word in words {
                *key_terms.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        // Get most frequent terms
        let mut frequent_terms: Vec<_> = key_terms.into_iter().collect();
        frequent_terms.sort_by(|a, b| b.1.cmp(&a.1));
        frequent_terms.truncate(5);

        // Build structured summary
        let mut summary_parts = Vec::new();

        summary_parts.push(format!(
            "Conversation Summary ({} messages from {} to {})",
            messages.len(),
            messages
                .first()
                .map(|m| m.timestamp.format("%Y-%m-%d %H:%M UTC").to_string())
                .unwrap_or_default(),
            messages
                .last()
                .map(|m| m.timestamp.format("%Y-%m-%d %H:%M UTC").to_string())
                .unwrap_or_default()
        ));

        if !user_questions.is_empty() {
            summary_parts.push("Key Questions Asked:".to_string());
            for (i, question) in user_questions.iter().take(3).enumerate() {
                summary_parts.push(format!(
                    "{}. {}",
                    i + 1,
                    question.chars().take(100).collect::<String>()
                ));
            }
        }

        if !frequent_terms.is_empty() {
            let terms: Vec<String> = frequent_terms
                .iter()
                .map(|(term, _)| term.clone())
                .collect();
            summary_parts.push(format!("Main Topics: {}", terms.join(", ")));
        }

        // Add basic conversation flow
        let user_messages = messages
            .iter()
            .filter(|m| m.role == MessageRole::User)
            .count();
        let assistant_messages = messages
            .iter()
            .filter(|m| m.role == MessageRole::Assistant)
            .count();
        summary_parts.push(format!(
            "Conversation Flow: {} user messages, {} assistant responses",
            user_messages, assistant_messages
        ));

        summary_parts.join("\n")
    }

    /// Simple stop word filter for keyword extraction
    fn is_stop_word(&self, word: &str) -> bool {
        matches!(
            word,
            "the"
                | "and"
                | "or"
                | "but"
                | "in"
                | "on"
                | "at"
                | "to"
                | "for"
                | "of"
                | "with"
                | "by"
                | "from"
                | "up"
                | "about"
                | "into"
                | "through"
                | "during"
                | "before"
                | "after"
                | "above"
                | "below"
                | "between"
                | "among"
                | "this"
                | "that"
                | "these"
                | "those"
                | "i"
                | "you"
                | "he"
                | "she"
                | "it"
                | "we"
                | "they"
                | "me"
                | "him"
                | "her"
                | "us"
                | "them"
                | "my"
                | "your"
                | "his"
                | "its"
                | "our"
                | "their"
                | "am"
                | "is"
                | "are"
                | "was"
                | "were"
                | "be"
                | "been"
                | "being"
                | "have"
                | "has"
                | "had"
                | "do"
                | "does"
                | "did"
                | "will"
                | "would"
                | "could"
                | "should"
                | "may"
                | "might"
                | "must"
                | "can"
        )
    }

    /// Count messages by role
    pub fn count_messages_by_role(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();

        for message in &self.messages {
            let role_str = match &message.role {
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
                MessageRole::System => "system",
            };
            *counts.entry(role_str.to_string()).or_insert(0) += 1;
        }

        counts
    }

    /// Pin a message to keep it in context
    pub fn pin_message(&mut self, message_id: &str) -> Result<()> {
        if self.messages.iter().any(|m| m.id == message_id) {
            self.context_window.pin_message(message_id.to_string());
            Ok(())
        } else {
            Err(anyhow::anyhow!("Message not found"))
        }
    }

    /// Unpin a message
    pub fn unpin_message(&mut self, message_id: &str) {
        self.context_window.unpin_message(message_id);
    }

    /// Get current context messages
    pub fn get_context_messages(&self) -> Vec<&Message> {
        self.context_window.get_context_messages(&self.messages)
    }

    /// Get current topics
    pub fn get_current_topics(&self) -> &[Topic] {
        &self.topic_tracker.current_topics
    }

    /// Get topic history
    pub fn get_topic_history(&self) -> &[TopicTransition] {
        &self.topic_tracker.topic_history
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> &SessionMetrics {
        &self.performance_metrics
    }

    /// Update user preference
    pub fn set_preference(&mut self, key: String, value: String) {
        self.user_preferences.insert(key, value);
    }

    /// Get user preference
    pub fn get_preference(&self, key: &str) -> Option<&String> {
        self.user_preferences.get(key)
    }

    /// Suspend the session
    pub fn suspend(&mut self) {
        self.session_state = SessionState::Suspended;
        self.update_activity();
    }

    /// Resume the session
    pub fn resume(&mut self) {
        if matches!(self.session_state, SessionState::Suspended) {
            self.session_state = SessionState::Active;
            self.update_activity();
        }
    }

    /// Mark session as idle
    pub fn mark_idle(&mut self) {
        if matches!(self.session_state, SessionState::Active) {
            self.session_state = SessionState::Idle;
        }
    }

    /// Export session to JSON
    pub fn export_to_json(&self) -> Result<String> {
        let data = self.to_data();
        Ok(serde_json::to_string_pretty(&data)?)
    }

    /// Get conversation threads
    pub fn get_threads(&self) -> Vec<ThreadInfo> {
        let mut threads: HashMap<String, ThreadInfo> = HashMap::new();

        for message in &self.messages {
            if let Some(thread_id) = &message.thread_id {
                let thread = threads
                    .entry(thread_id.clone())
                    .or_insert_with(|| ThreadInfo {
                        id: thread_id.clone(),
                        message_count: 0,
                        first_message_time: message.timestamp,
                        last_message_time: message.timestamp,
                        participants: Vec::new(),
                    });

                thread.message_count += 1;
                thread.last_message_time = thread.last_message_time.max(message.timestamp);
                thread.first_message_time = thread.first_message_time.min(message.timestamp);

                let role_str = match &message.role {
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                    MessageRole::System => "system",
                };

                if !thread.participants.contains(&role_str.to_string()) {
                    thread.participants.push(role_str.to_string());
                }
            }
        }

        threads.into_values().collect()
    }
}

/// Chat manager for multiple sessions with persistence
pub struct ChatManager {
    sessions: Arc<RwLock<HashMap<String, Arc<Mutex<ChatSession>>>>>,
    store: Arc<oxirs_core::Store>,
    persistence: Option<Arc<sled::Db>>,
    persistence_path: Option<PathBuf>,
}

impl ChatManager {
    pub async fn new(store: Arc<oxirs_core::Store>) -> Result<Self> {
        Ok(Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            store,
            persistence: None,
            persistence_path: None,
        })
    }

    pub async fn with_persistence(
        store: Arc<oxirs_core::Store>,
        persistence_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let db = sled::open(&persistence_path)?;
        let mut manager = Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            store,
            persistence: Some(Arc::new(db)),
            persistence_path: Some(persistence_path.as_ref().to_path_buf()),
        };

        // Load existing sessions
        manager.load_sessions().await?;

        // Start expiration checker
        manager.start_expiration_checker();

        Ok(manager)
    }

    async fn load_sessions(&mut self) -> Result<()> {
        if let Some(db) = &self.persistence {
            let mut sessions = self.sessions.write().await;
            let mut loaded_count = 0;
            let mut skipped_count = 0;
            let mut error_count = 0;

            info!("Starting session recovery from persistence store");

            for item in db.iter() {
                match item {
                    Ok((key, value)) => {
                        let session_id = String::from_utf8_lossy(&key).to_string();

                        match bincode::deserialize::<SessionData>(&value) {
                            Ok(session_data) => {
                                // Validate session data integrity
                                if self.validate_session_data(&session_data) {
                                    let session =
                                        ChatSession::from_data(session_data, self.store.clone());
                                    if !session.is_expired() {
                                        sessions.insert(
                                            session_id.clone(),
                                            Arc::new(Mutex::new(session)),
                                        );
                                        loaded_count += 1;
                                        debug!("Recovered session: {}", session_id);
                                    } else {
                                        skipped_count += 1;
                                        debug!("Skipped expired session: {}", session_id);
                                    }
                                } else {
                                    error_count += 1;
                                    warn!("Invalid session data for session: {}", session_id);
                                }
                            }
                            Err(e) => {
                                error_count += 1;
                                error!("Failed to deserialize session {}: {}", session_id, e);
                            }
                        }
                    }
                    Err(e) => {
                        error_count += 1;
                        error!("Failed to read session from persistence: {}", e);
                    }
                }
            }

            info!(
                "Session recovery complete: {} loaded, {} skipped (expired), {} errors",
                loaded_count, skipped_count, error_count
            );
        }
        Ok(())
    }

    fn validate_session_data(&self, session_data: &SessionData) -> bool {
        // Basic validation checks
        if session_data.id.is_empty() {
            return false;
        }

        // Check if timestamps are reasonable
        let now = chrono::Utc::now();
        if session_data.created_at > now || session_data.last_activity > now {
            return false;
        }

        // Check if session is too old (older than 30 days)
        let max_age = chrono::Duration::days(30);
        if now - session_data.created_at > max_age {
            return false;
        }

        true
    }

    async fn save_session(&self, session_id: &str) -> Result<()> {
        if let Some(db) = &self.persistence {
            let sessions = self.sessions.read().await;
            if let Some(session_arc) = sessions.get(session_id) {
                let session = session_arc.lock().await;
                let data = session.to_data();

                // Validate data before serialization
                if !self.validate_session_data(&data) {
                    return Err(anyhow::anyhow!(
                        "Invalid session data for session: {}",
                        session_id
                    ));
                }

                match bincode::serialize(&data) {
                    Ok(serialized) => {
                        // Use a transaction-like approach with a temporary key
                        let temp_key = format!("{}_temp", session_id);

                        // Write to temporary key first
                        db.insert(temp_key.as_bytes(), serialized.clone())?;
                        db.flush()?;

                        // Then move to actual key (atomic on most filesystems)
                        db.insert(session_id.as_bytes(), serialized)?;
                        db.remove(temp_key.as_bytes())?;
                        db.flush()?;

                        debug!("Successfully saved session: {}", session_id);
                    }
                    Err(e) => {
                        error!("Failed to serialize session {}: {}", session_id, e);
                        return Err(anyhow::anyhow!("Serialization failed: {}", e));
                    }
                }
            } else {
                warn!("Attempted to save non-existent session: {}", session_id);
            }
        }
        Ok(())
    }

    pub async fn create_session(&self, session_id: String) -> Result<Arc<Mutex<ChatSession>>> {
        let session = ChatSession::new(session_id.clone(), self.store.clone());
        let session_arc = Arc::new(Mutex::new(session));

        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(session_id.clone(), session_arc.clone());
        }

        self.save_session(&session_id).await?;
        Ok(session_arc)
    }

    pub async fn get_session(&self, session_id: &str) -> Option<Arc<Mutex<ChatSession>>> {
        let sessions = self.sessions.read().await;
        sessions.get(session_id).cloned()
    }

    pub async fn get_or_create_session(
        &self,
        session_id: String,
    ) -> Result<Arc<Mutex<ChatSession>>> {
        // Try to get existing session first
        if let Some(session) = self.get_session(&session_id).await {
            // Update activity and save
            {
                let mut session_guard = session.lock().await;

                // Check if session is still valid
                if session_guard.is_expired() {
                    warn!("Session {} has expired, creating new session", session_id);
                    drop(session_guard);

                    // Remove expired session and create new one
                    self.remove_session(&session_id).await?;
                    return self.create_session(session_id).await;
                }

                session_guard.update_activity();
            }

            // Save updated session asynchronously (don't block on save errors)
            if let Err(e) = self.save_session(&session_id).await {
                warn!("Failed to save session activity update: {}", e);
            }

            Ok(session)
        } else {
            self.create_session(session_id).await
        }
    }

    pub async fn remove_session(&self, session_id: &str) -> Result<()> {
        {
            let mut sessions = self.sessions.write().await;
            sessions.remove(session_id);
        }

        if let Some(db) = &self.persistence {
            db.remove(session_id.as_bytes())?;
            db.flush()?;
        }

        Ok(())
    }

    pub async fn save_all_sessions(&self) -> Result<()> {
        let sessions = self.sessions.read().await;
        for session_id in sessions.keys() {
            self.save_session(session_id).await?;
        }
        Ok(())
    }

    pub async fn cleanup_expired_sessions(&self) -> Result<usize> {
        let mut expired_count = 0;
        let mut error_count = 0;

        // Get list of expired sessions
        let expired_ids = {
            let sessions = self.sessions.read().await;
            let mut expired_ids = Vec::new();

            for (id, session_arc) in sessions.iter() {
                match session_arc.try_lock() {
                    Ok(session) => {
                        if session.is_expired() {
                            expired_ids.push(id.clone());
                        }
                    }
                    Err(_) => {
                        // Session is locked, skip for now
                        debug!("Skipping cleanup for locked session: {}", id);
                    }
                }
            }
            expired_ids
        };

        info!("Found {} expired sessions to clean up", expired_ids.len());

        // Remove expired sessions
        for id in expired_ids {
            match self.remove_session(&id).await {
                Ok(_) => {
                    expired_count += 1;
                    debug!("Cleaned up expired session: {}", id);
                }
                Err(e) => {
                    error_count += 1;
                    error!("Failed to clean up session {}: {}", id, e);
                }
            }
        }

        if error_count > 0 {
            warn!(
                "Cleanup completed with {} errors out of {} expired sessions",
                error_count,
                expired_count + error_count
            );
        } else if expired_count > 0 {
            info!("Successfully cleaned up {} expired sessions", expired_count);
        }

        Ok(expired_count)
    }

    fn start_expiration_checker(&self) {
        let manager = self.sessions.clone();
        let persistence = self.persistence.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300)); // Check every 5 minutes
            let mut cleanup_interval = tokio::time::interval(Duration::from_secs(3600)); // Cleanup every hour

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        Self::check_session_health(&manager).await;
                    }
                    _ = cleanup_interval.tick() => {
                        Self::cleanup_expired_sessions_background(&manager, &persistence).await;
                    }
                }
            }
        });
    }

    async fn check_session_health(manager: &Arc<RwLock<HashMap<String, Arc<Mutex<ChatSession>>>>>) {
        let sessions = manager.read().await;
        let mut healthy_count = 0;
        let mut idle_count = 0;
        let mut expired_count = 0;
        let mut error_count = 0;

        for (id, session_arc) in sessions.iter() {
            match session_arc.try_lock() {
                Ok(session) => {
                    match session.session_state {
                        SessionState::Active => healthy_count += 1,
                        SessionState::Idle => idle_count += 1,
                        SessionState::Expired => expired_count += 1,
                        SessionState::Suspended => {}
                    }

                    // Check for sessions with high error rates
                    let error_rate = if session.performance_metrics.total_queries > 0 {
                        session.performance_metrics.failed_queries as f64
                            / session.performance_metrics.total_queries as f64
                    } else {
                        0.0
                    };

                    if error_rate > 0.5 && session.performance_metrics.total_queries > 10 {
                        warn!(
                            "Session {} has high error rate: {:.2}%",
                            id,
                            error_rate * 100.0
                        );
                        error_count += 1;
                    }
                }
                Err(_) => {
                    // Session is locked, that's normal
                }
            }
        }

        let total_sessions = sessions.len();
        drop(sessions);

        debug!(
            "Session health check: {} total, {} healthy, {} idle, {} expired, {} high-error",
            total_sessions, healthy_count, idle_count, expired_count, error_count
        );

        // Log warnings for concerning patterns
        if expired_count > total_sessions / 4 {
            warn!(
                "High number of expired sessions: {}/{}",
                expired_count, total_sessions
            );
        }

        if error_count > 0 {
            warn!("Found {} sessions with high error rates", error_count);
        }
    }

    async fn cleanup_expired_sessions_background(
        manager: &Arc<RwLock<HashMap<String, Arc<Mutex<ChatSession>>>>>,
        persistence: &Option<Arc<sled::Db>>,
    ) {
        let mut expired_ids = Vec::new();

        // Collect expired session IDs
        {
            let sessions = manager.read().await;
            for (id, session_arc) in sessions.iter() {
                if let Ok(session) = session_arc.try_lock() {
                    if session.is_expired() {
                        expired_ids.push(id.clone());
                    }
                }
            }
        }

        if expired_ids.is_empty() {
            return;
        }

        info!(
            "Background cleanup removing {} expired sessions",
            expired_ids.len()
        );

        // Remove expired sessions
        let mut removed_count = 0;
        for id in expired_ids {
            // Remove from memory
            {
                let mut sessions = manager.write().await;
                if sessions.remove(&id).is_some() {
                    removed_count += 1;
                }
            }

            // Remove from persistence
            if let Some(db) = persistence {
                if let Err(e) = db.remove(id.as_bytes()) {
                    error!("Failed to remove session {} from persistence: {}", id, e);
                }
            }
        }

        if removed_count > 0 {
            info!(
                "Background cleanup removed {} expired sessions",
                removed_count
            );

            // Flush persistence changes
            if let Some(db) = persistence {
                if let Err(e) = db.flush() {
                    error!("Failed to flush persistence after cleanup: {}", e);
                }
            }
        }
    }

    pub async fn get_active_session_count(&self) -> usize {
        let sessions = self.sessions.read().await;
        sessions.len()
    }

    pub async fn get_session_stats(&self) -> SessionStats {
        let sessions = self.sessions.read().await;
        let mut stats = SessionStats::default();
        stats.total_sessions = sessions.len();

        for session_arc in sessions.values() {
            let session = session_arc.lock().await;
            match session.session_state {
                SessionState::Active => stats.active_sessions += 1,
                SessionState::Idle => stats.idle_sessions += 1,
                SessionState::Expired => stats.expired_sessions += 1,
                SessionState::Suspended => stats.suspended_sessions += 1,
            }
            stats.total_messages += session.messages.len();
        }

        stats
    }

    /// Recover a session from error state
    pub async fn recover_session(&self, session_id: &str) -> Result<()> {
        if let Some(session_arc) = self.get_session(session_id).await {
            let mut session = session_arc.lock().await;

            // Reset session state if it's in an error condition
            match session.session_state {
                SessionState::Expired => {
                    session.session_state = SessionState::Active;
                    session.update_activity();
                }
                SessionState::Suspended => {
                    session.session_state = SessionState::Active;
                    session.update_activity();
                }
                _ => {}
            }

            // Clear any error flags in performance metrics
            if session.performance_metrics.failed_queries
                > session.performance_metrics.successful_queries
            {
                tracing::warn!(
                    "Session {} has high failure rate, resetting metrics",
                    session_id
                );
                session.performance_metrics.failed_queries = 0;
            }

            drop(session);
            self.save_session(session_id).await?;
        }

        Ok(())
    }

    /// Validate and repair persistence database
    pub async fn validate_persistence(&self) -> Result<ValidationReport> {
        let mut report = ValidationReport::default();

        if let Some(db) = &self.persistence {
            // Check database health
            report.total_entries = db.len();

            let mut corrupted_entries = Vec::new();
            let mut recoverable_entries = Vec::new();

            for item in db.iter() {
                match item {
                    Ok((key, value)) => {
                        let session_id = String::from_utf8_lossy(&key).to_string();

                        match bincode::deserialize::<SessionData>(&value) {
                            Ok(session_data) => {
                                // Validate session data
                                if session_data.messages.is_empty()
                                    && session_data.performance_metrics.total_queries > 0
                                {
                                    report.inconsistent_entries += 1;
                                    recoverable_entries.push(session_id.clone());
                                }
                                report.valid_entries += 1;
                            }
                            Err(e) => {
                                report.corrupted_entries += 1;
                                corrupted_entries.push(session_id.clone());
                                tracing::error!("Corrupted session data for {}: {}", session_id, e);
                            }
                        }
                    }
                    Err(e) => {
                        report.read_errors += 1;
                        tracing::error!("Database read error: {}", e);
                    }
                }
            }

            // Remove corrupted entries
            for id in corrupted_entries {
                db.remove(id.as_bytes())?;
                report.removed_entries += 1;
            }

            // Attempt to recover inconsistent entries
            for id in recoverable_entries {
                if let Ok(Some(value)) = db.get(id.as_bytes()) {
                    if let Ok(mut session_data) = bincode::deserialize::<SessionData>(&value) {
                        // Reset inconsistent state
                        session_data.performance_metrics = SessionMetrics::default();

                        if let Ok(serialized) = bincode::serialize(&session_data) {
                            db.insert(id.as_bytes(), serialized)?;
                            report.recovered_entries += 1;
                        }
                    }
                }
            }

            db.flush()?;
        }

        Ok(report)
    }

    /// Create a backup of all sessions with integrity checks
    pub async fn backup_sessions(&self, backup_path: impl AsRef<Path>) -> Result<BackupReport> {
        let mut report = BackupReport::default();
        let backup_dir = backup_path.as_ref();
        std::fs::create_dir_all(backup_dir)?;

        // Create a manifest file for backup integrity
        let manifest_path = backup_dir.join("backup_manifest.json");
        let mut manifest = serde_json::Map::new();
        manifest.insert(
            "created_at".to_string(),
            serde_json::Value::String(chrono::Utc::now().to_rfc3339()),
        );
        manifest.insert(
            "version".to_string(),
            serde_json::Value::String("1.0".to_string()),
        );

        let sessions = self.sessions.read().await;
        report.total_sessions = sessions.len();

        let mut session_hashes = serde_json::Map::new();

        for (session_id, session_arc) in sessions.iter() {
            match session_arc.try_lock() {
                Ok(session) => {
                    let session_data = session.to_data();

                    // Validate session data before backup
                    if !self.validate_session_data(&session_data) {
                        report.failed_backups += 1;
                        error!("Invalid session data for {}, skipping backup", session_id);
                        continue;
                    }

                    let backup_file = backup_dir.join(format!("{}.json", session_id));
                    match serde_json::to_string_pretty(&session_data) {
                        Ok(json) => {
                            // Calculate hash for integrity checking
                            use std::collections::hash_map::DefaultHasher;
                            use std::hash::{Hash, Hasher};

                            let mut hasher = DefaultHasher::new();
                            json.hash(&mut hasher);
                            let hash = hasher.finish();

                            match std::fs::write(&backup_file, &json) {
                                Ok(_) => {
                                    report.successful_backups += 1;
                                    session_hashes.insert(
                                        session_id.clone(),
                                        serde_json::Value::Number(serde_json::Number::from(hash)),
                                    );
                                    debug!("Backed up session: {}", session_id);
                                }
                                Err(e) => {
                                    report.failed_backups += 1;
                                    error!(
                                        "Failed to write backup for session {}: {}",
                                        session_id, e
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            report.failed_backups += 1;
                            error!("Failed to serialize session {}: {}", session_id, e);
                        }
                    }
                }
                Err(_) => {
                    // Session is locked, skip for now but don't count as failure
                    warn!("Session {} is locked, skipping backup", session_id);
                    report.total_sessions -= 1; // Adjust count
                }
            }
        }

        // Write manifest with session hashes
        manifest.insert(
            "sessions".to_string(),
            serde_json::Value::Object(session_hashes),
        );
        manifest.insert(
            "total_sessions".to_string(),
            serde_json::Value::Number(serde_json::Number::from(report.successful_backups)),
        );

        if let Err(e) = std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?) {
            warn!("Failed to write backup manifest: {}", e);
        }

        report.backup_path = backup_dir.to_path_buf();
        report.timestamp = chrono::Utc::now();

        info!(
            "Backup completed: {}/{} sessions backed up successfully",
            report.successful_backups, report.total_sessions
        );

        Ok(report)
    }

    /// Restore sessions from backup with integrity verification
    pub async fn restore_sessions(
        &mut self,
        backup_path: impl AsRef<Path>,
    ) -> Result<RestoreReport> {
        let mut report = RestoreReport::default();
        let backup_dir = backup_path.as_ref();

        if !backup_dir.exists() {
            return Err(anyhow::anyhow!("Backup directory does not exist"));
        }

        // Try to load and verify backup manifest
        let manifest_path = backup_dir.join("backup_manifest.json");
        let manifest = if manifest_path.exists() {
            match std::fs::read_to_string(&manifest_path) {
                Ok(content) => match serde_json::from_str::<serde_json::Value>(&content) {
                    Ok(manifest) => {
                        info!("Found backup manifest, verifying backup integrity");
                        Some(manifest)
                    }
                    Err(e) => {
                        warn!("Invalid backup manifest: {}", e);
                        None
                    }
                },
                Err(e) => {
                    warn!("Failed to read backup manifest: {}", e);
                    None
                }
            }
        } else {
            warn!("No backup manifest found, proceeding without integrity checks");
            None
        };

        // Get session hashes from manifest for verification
        let session_hashes = manifest
            .as_ref()
            .and_then(|m| m.get("sessions"))
            .and_then(|s| s.as_object())
            .cloned();

        for entry in std::fs::read_dir(backup_dir)? {
            let entry = entry?;
            let path = entry.path();

            // Skip non-JSON files and the manifest
            if path.extension().and_then(|s| s.to_str()) != Some("json")
                || path.file_name().and_then(|n| n.to_str()) == Some("backup_manifest.json")
            {
                continue;
            }

            report.total_files += 1;

            match std::fs::read_to_string(&path) {
                Ok(json) => {
                    // Verify hash if manifest is available
                    if let Some(ref hashes) = session_hashes {
                        let session_id = path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("unknown");

                        if let Some(expected_hash) = hashes.get(session_id).and_then(|h| h.as_u64())
                        {
                            use std::collections::hash_map::DefaultHasher;
                            use std::hash::{Hash, Hasher};

                            let mut hasher = DefaultHasher::new();
                            json.hash(&mut hasher);
                            let actual_hash = hasher.finish();

                            if actual_hash != expected_hash {
                                error!(
                                    "Hash mismatch for session {}, skipping restore",
                                    session_id
                                );
                                report.failed_restorations += 1;
                                continue;
                            }
                        }
                    }

                    match serde_json::from_str::<SessionData>(&json) {
                        Ok(session_data) => {
                            // Validate session data before restoring
                            if !self.validate_session_data(&session_data) {
                                error!("Invalid session data in backup file {:?}", path);
                                report.failed_restorations += 1;
                                continue;
                            }

                            let session = ChatSession::from_data(session_data, self.store.clone());
                            let session_id = session.id.clone();

                            // Check if session already exists
                            if self.get_session(&session_id).await.is_some() {
                                warn!("Session {} already exists, skipping restore", session_id);
                                report.failed_restorations += 1;
                                continue;
                            }

                            // Add to sessions
                            {
                                let mut sessions = self.sessions.write().await;
                                sessions.insert(session_id.clone(), Arc::new(Mutex::new(session)));
                            }

                            // Save to persistence
                            match self.save_session(&session_id).await {
                                Ok(_) => {
                                    report.restored_sessions += 1;
                                    debug!("Restored session: {}", session_id);
                                }
                                Err(e) => {
                                    error!("Failed to save restored session {}: {}", session_id, e);
                                    report.failed_restorations += 1;

                                    // Remove from memory since save failed
                                    let mut sessions = self.sessions.write().await;
                                    sessions.remove(&session_id);
                                }
                            }
                        }
                        Err(e) => {
                            report.failed_restorations += 1;
                            error!("Failed to deserialize session from {:?}: {}", path, e);
                        }
                    }
                }
                Err(e) => {
                    report.failed_restorations += 1;
                    error!("Failed to read backup file {:?}: {}", path, e);
                }
            }
        }

        info!(
            "Session restore completed: {}/{} sessions restored successfully",
            report.restored_sessions, report.total_files
        );

        Ok(report)
    }

    /// Concurrent session creation with proper locking
    pub async fn create_session_concurrent(
        &self,
        session_id: String,
    ) -> Result<Arc<Mutex<ChatSession>>> {
        // Use a more sophisticated approach to prevent race conditions
        // First, try to insert a placeholder
        {
            let mut sessions = self.sessions.write().await;
            if sessions.contains_key(&session_id) {
                drop(sessions);
                return self
                    .get_session(&session_id)
                    .await
                    .ok_or_else(|| anyhow::anyhow!("Session disappeared during creation"));
            }

            // Insert a temporary placeholder to reserve the slot
            let temp_session = ChatSession::new(session_id.clone(), self.store.clone());
            let session_arc = Arc::new(Mutex::new(temp_session));
            sessions.insert(session_id.clone(), session_arc.clone());
        }

        // Now we have exclusive access to this session ID
        // Save the session to persistence
        match self.save_session(&session_id).await {
            Ok(_) => {
                info!("Created new session: {}", session_id);
                Ok(self.get_session(&session_id).await.unwrap())
            }
            Err(e) => {
                // Remove the failed session from memory
                self.sessions.write().await.remove(&session_id);
                Err(e)
            }
        }
    }

    /// Batch session operations for efficiency
    pub async fn save_multiple_sessions(&self, session_ids: &[String]) -> Result<Vec<String>> {
        let mut successful_saves = Vec::new();
        let mut errors = Vec::new();

        for session_id in session_ids {
            match self.save_session(session_id).await {
                Ok(_) => successful_saves.push(session_id.clone()),
                Err(e) => errors.push(format!("{}: {}", session_id, e)),
            }
        }

        if !errors.is_empty() {
            warn!("Some session saves failed: {:?}", errors);
        }

        debug!(
            "Batch save completed: {}/{} sessions saved",
            successful_saves.len(),
            session_ids.len()
        );
        Ok(successful_saves)
    }

    /// Get detailed session metrics
    pub async fn get_detailed_metrics(&self) -> DetailedSessionMetrics {
        let sessions = self.sessions.read().await;
        let mut metrics = DetailedSessionMetrics::default();

        for (_session_id, session_arc) in sessions.iter() {
            if let Ok(session) = session_arc.try_lock() {
                metrics.total_sessions += 1;

                match session.session_state {
                    SessionState::Active => metrics.active_sessions += 1,
                    SessionState::Idle => metrics.idle_sessions += 1,
                    SessionState::Expired => metrics.expired_sessions += 1,
                    SessionState::Suspended => metrics.suspended_sessions += 1,
                }

                metrics.total_messages += session.messages.len();
                metrics.total_tokens += session.performance_metrics.total_tokens_processed;

                if session.performance_metrics.total_queries > 0 {
                    let error_rate = session.performance_metrics.failed_queries as f64
                        / session.performance_metrics.total_queries as f64;
                    if error_rate > 0.1 {
                        metrics.high_error_sessions += 1;
                    }

                    metrics.average_response_time +=
                        session.performance_metrics.average_response_time_ms;
                    metrics.response_time_samples += 1;
                }

                let session_age = chrono::Utc::now() - session.created_at;
                metrics.total_session_age += session_age;

                if session.messages.len() > metrics.max_messages_per_session {
                    metrics.max_messages_per_session = session.messages.len();
                }

                if session.context_window.pinned_messages.len() > 0 {
                    metrics.sessions_with_pinned_messages += 1;
                }
            } else {
                metrics.locked_sessions += 1;
            }
        }

        if metrics.response_time_samples > 0 {
            metrics.average_response_time /= metrics.response_time_samples as f64;
        }

        if metrics.total_sessions > 0 {
            metrics.average_session_age = metrics.total_session_age / metrics.total_sessions as i32;
        }

        metrics
    }
}

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
pub struct ValidationReport {
    pub total_entries: usize,
    pub valid_entries: usize,
    pub corrupted_entries: usize,
    pub inconsistent_entries: usize,
    pub read_errors: usize,
    pub removed_entries: usize,
    pub recovered_entries: usize,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct BackupReport {
    pub total_sessions: usize,
    pub successful_backups: usize,
    pub failed_backups: usize,
    pub backup_path: PathBuf,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct RestoreReport {
    pub total_files: usize,
    pub restored_sessions: usize,
    pub failed_restorations: usize,
}

/// Detailed session metrics for monitoring
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
