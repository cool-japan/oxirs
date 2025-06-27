//! # OxiRS Chat
//!
//! RAG chat API with LLM integration and natural language to SPARQL translation.
//!
//! This crate provides a conversational interface for knowledge graphs,
//! combining retrieval-augmented generation (RAG) with SPARQL querying.

use anyhow::Result;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime},
};
use tokio::sync::{Mutex, RwLock};
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use crate::rag::QueryIntent;

pub mod chat;
pub mod llm;
pub mod nl2sparql;
pub mod rag;
pub mod server;

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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
        self.process_message_with_options(user_input, None, None).await
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

        // Create query context
        let query_context = QueryContext {
            query: user_input.clone(),
            intent: self.classify_intent(&user_input),
            entities: Vec::new(),      // TODO: Extract entities
            relationships: Vec::new(), // TODO: Extract relationships
            constraints: Vec::new(),   // TODO: Extract constraints
            conversation_history: self
                .messages
                .iter()
                .take(5) // Last 5 messages for context
                .map(|m| m.content.clone())
                .collect(),
        };

        // Initialize systems (in production, these would be initialized once and reused)
        let rag_config = RAGConfig::default();
        let rag_system = RAGSystem::new(
            rag_config,
            self.store.clone(),
            None, // Vector index not available yet
            None, // Embedding model not available yet
        );

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
            if let Ok(mut nl2sparql_system) = NL2SPARQLSystem::new(nl2sparql_config, None) {
                if let Ok(sparql_result) = nl2sparql_system.generate_sparql(&query_context).await {
                    sparql_query = Some(sparql_result.query);
                    confidence_score = confidence_score.max(sparql_result.confidence);
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
        if self.config.enable_context_summarization && 
           self.context_window.should_summarize(self.messages.len()) {
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
    
    /// Summarize the conversation context
    async fn summarize_context(&mut self) {
        // Get messages that need to be summarized
        let messages_to_summarize: Vec<String> = self.messages
            .iter()
            .take(self.messages.len().saturating_sub(self.config.sliding_window_size))
            .filter(|m| !self.context_window.pinned_messages.contains(&m.id))
            .map(|m| format!("{:?}: {}", m.role, m.content))
            .collect();
        
        if messages_to_summarize.is_empty() {
            return;
        }
        
        // Create a simple summary (in production, this would use an LLM)
        let summary = format!(
            "Previous conversation summary ({} messages):\n{}",
            messages_to_summarize.len(),
            messages_to_summarize.join("\n").chars().take(500).collect::<String>()
        );
        
        self.context_window.update_summary(summary);
        
        // Remove old messages from memory (keep them in persistence)
        let keep_from = self.messages.len().saturating_sub(self.config.sliding_window_size);
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
                let thread = threads.entry(thread_id.clone()).or_insert_with(|| ThreadInfo {
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
            
            for item in db.iter() {
                let (key, value) = item?;
                let session_id = String::from_utf8_lossy(&key).to_string();
                
                if let Ok(session_data) = bincode::deserialize::<SessionData>(&value) {
                    let session = ChatSession::from_data(session_data, self.store.clone());
                    if !session.is_expired() {
                        sessions.insert(session_id, Arc::new(Mutex::new(session)));
                    }
                }
            }
        }
        Ok(())
    }

    async fn save_session(&self, session_id: &str) -> Result<()> {
        if let Some(db) = &self.persistence {
            let sessions = self.sessions.read().await;
            if let Some(session_arc) = sessions.get(session_id) {
                let session = session_arc.lock().await;
                let data = session.to_data();
                let serialized = bincode::serialize(&data)?;
                db.insert(session_id.as_bytes(), serialized)?;
                db.flush()?;
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

    pub async fn get_or_create_session(&self, session_id: String) -> Result<Arc<Mutex<ChatSession>>> {
        if let Some(session) = self.get_session(&session_id).await {
            // Update activity
            {
                let mut session_guard = session.lock().await;
                session_guard.update_activity();
            }
            self.save_session(&session_id).await?;
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
        let sessions = self.sessions.read().await;
        let mut expired_ids = Vec::new();
        
        for (id, session_arc) in sessions.iter() {
            let session = session_arc.lock().await;
            if session.is_expired() {
                expired_ids.push(id.clone());
            }
        }
        drop(sessions);
        
        for id in expired_ids {
            self.remove_session(&id).await?;
            expired_count += 1;
        }
        
        Ok(expired_count)
    }

    fn start_expiration_checker(&self) {
        let manager = self.sessions.clone();
        let persistence = self.persistence.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300)); // Check every 5 minutes
            
            loop {
                interval.tick().await;
                
                // Check for expired sessions
                let sessions = manager.read().await;
                let mut expired_ids = Vec::new();
                
                for (id, session_arc) in sessions.iter() {
                    let session = session_arc.lock().await;
                    if session.is_expired() {
                        expired_ids.push(id.clone());
                    }
                }
                drop(sessions);
                
                // Remove expired sessions
                for id in expired_ids {
                    let mut sessions = manager.write().await;
                    sessions.remove(&id);
                    
                    if let Some(db) = &persistence {
                        let _ = db.remove(id.as_bytes());
                    }
                }
            }
        });
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
            if session.performance_metrics.failed_queries > session.performance_metrics.successful_queries {
                tracing::warn!("Session {} has high failure rate, resetting metrics", session_id);
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
                                if session_data.messages.is_empty() && 
                                   session_data.performance_metrics.total_queries > 0 {
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
    
    /// Create a backup of all sessions
    pub async fn backup_sessions(&self, backup_path: impl AsRef<Path>) -> Result<BackupReport> {
        let mut report = BackupReport::default();
        let backup_dir = backup_path.as_ref();
        std::fs::create_dir_all(backup_dir)?;
        
        let sessions = self.sessions.read().await;
        report.total_sessions = sessions.len();
        
        for (session_id, session_arc) in sessions.iter() {
            let session = session_arc.lock().await;
            let session_data = session.to_data();
            
            let backup_file = backup_dir.join(format!("{}.json", session_id));
            match serde_json::to_string_pretty(&session_data) {
                Ok(json) => {
                    if let Err(e) = std::fs::write(&backup_file, json) {
                        report.failed_backups += 1;
                        tracing::error!("Failed to backup session {}: {}", session_id, e);
                    } else {
                        report.successful_backups += 1;
                    }
                }
                Err(e) => {
                    report.failed_backups += 1;
                    tracing::error!("Failed to serialize session {}: {}", session_id, e);
                }
            }
        }
        
        report.backup_path = backup_dir.to_path_buf();
        report.timestamp = chrono::Utc::now();
        
        Ok(report)
    }
    
    /// Restore sessions from backup
    pub async fn restore_sessions(&mut self, backup_path: impl AsRef<Path>) -> Result<RestoreReport> {
        let mut report = RestoreReport::default();
        let backup_dir = backup_path.as_ref();
        
        if !backup_dir.exists() {
            return Err(anyhow::anyhow!("Backup directory does not exist"));
        }
        
        for entry in std::fs::read_dir(backup_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                report.total_files += 1;
                
                match std::fs::read_to_string(&path) {
                    Ok(json) => {
                        match serde_json::from_str::<SessionData>(&json) {
                            Ok(session_data) => {
                                let session = ChatSession::from_data(session_data, self.store.clone());
                                let session_id = session.id.clone();
                                
                                let mut sessions = self.sessions.write().await;
                                sessions.insert(session_id.clone(), Arc::new(Mutex::new(session)));
                                drop(sessions);
                                
                                self.save_session(&session_id).await?;
                                report.restored_sessions += 1;
                            }
                            Err(e) => {
                                report.failed_restorations += 1;
                                tracing::error!("Failed to deserialize session from {:?}: {}", path, e);
                            }
                        }
                    }
                    Err(e) => {
                        report.failed_restorations += 1;
                        tracing::error!("Failed to read backup file {:?}: {}", path, e);
                    }
                }
            }
        }
        
        Ok(report)
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

/// Thread information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadInfo {
    pub id: String,
    pub message_count: usize,
    pub first_message_time: chrono::DateTime<chrono::Utc>,
    pub last_message_time: chrono::DateTime<chrono::Utc>,
    pub participants: Vec<String>,
}

/// Context window for managing conversation context
#[derive(Debug, Clone)]
pub struct ContextWindow {
    pub window_size: usize,
    pub pinned_messages: Vec<String>,
    pub context_summary: Option<String>,
    pub last_summarization: Option<chrono::DateTime<chrono::Utc>>,
}

impl ContextWindow {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            pinned_messages: Vec::new(),
            context_summary: None,
            last_summarization: None,
        }
    }

    pub fn get_context_messages<'a>(&self, messages: &'a [Message]) -> Vec<&'a Message> {
        let mut context_messages = Vec::new();
        
        // Add pinned messages first
        for pinned_id in &self.pinned_messages {
            if let Some(msg) = messages.iter().find(|m| &m.id == pinned_id) {
                context_messages.push(msg);
            }
        }
        
        // Add recent messages up to window size
        let recent_count = self.window_size.saturating_sub(context_messages.len());
        let recent_messages: Vec<&Message> = messages
            .iter()
            .rev()
            .filter(|m| !self.pinned_messages.contains(&m.id))
            .take(recent_count)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        
        context_messages.extend(recent_messages);
        context_messages
    }

    pub fn pin_message(&mut self, message_id: String) {
        if !self.pinned_messages.contains(&message_id) {
            self.pinned_messages.push(message_id);
        }
    }

    pub fn unpin_message(&mut self, message_id: &str) {
        self.pinned_messages.retain(|id| id != message_id);
    }

    pub fn should_summarize(&self, message_count: usize) -> bool {
        // Summarize when we have 2x the window size of messages
        // and haven't summarized in the last hour
        message_count > self.window_size * 2 &&
            self.last_summarization
                .map(|t| chrono::Utc::now() - t > chrono::Duration::hours(1))
                .unwrap_or(true)
    }

    pub fn update_summary(&mut self, summary: String) {
        self.context_summary = Some(summary);
        self.last_summarization = Some(chrono::Utc::now());
    }
}

/// Topic tracker for detecting topic drift
#[derive(Debug, Clone)]
pub struct TopicTracker {
    pub current_topics: Vec<Topic>,
    pub topic_history: Vec<TopicTransition>,
    pub drift_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topic {
    pub id: String,
    pub name: String,
    pub keywords: Vec<String>,
    pub confidence: f32,
    pub first_mentioned: chrono::DateTime<chrono::Utc>,
    pub last_mentioned: chrono::DateTime<chrono::Utc>,
    pub message_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicTransition {
    pub from_topic: String,
    pub to_topic: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub transition_type: TransitionType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransitionType {
    Natural,
    Abrupt,
    Return,
    Clarification,
}

impl TopicTracker {
    pub fn new() -> Self {
        Self {
            current_topics: Vec::new(),
            topic_history: Vec::new(),
            drift_threshold: 0.3,
        }
    }

    pub fn analyze_message(&mut self, message: &Message) -> Option<TopicTransition> {
        // Simple keyword-based topic detection
        let detected_topics = self.detect_topics(&message.content);
        
        if detected_topics.is_empty() {
            return None;
        }
        
        let primary_topic_id = detected_topics[0].id.clone();
        let primary_topic = &detected_topics[0];
        
        // Check if this is a new topic
        if let Some(current) = self.current_topics.first() {
            if current.id != primary_topic.id {
                let transition = TopicTransition {
                    from_topic: current.id.clone(),
                    to_topic: primary_topic.id.clone(),
                    timestamp: chrono::Utc::now(),
                    transition_type: self.classify_transition(current, primary_topic),
                };
                
                self.topic_history.push(transition.clone());
                self.current_topics = detected_topics;
                
                return Some(transition);
            }
        } else {
            self.current_topics = detected_topics;
        }
        
        // Update topic stats
        if let Some(topic) = self.current_topics.iter_mut().find(|t| t.id == primary_topic_id) {
            topic.last_mentioned = chrono::Utc::now();
            topic.message_count += 1;
        }
        
        None
    }

    fn detect_topics(&self, content: &str) -> Vec<Topic> {
        // Simplified topic detection based on keywords
        let mut topics = Vec::new();
        let content_lower = content.to_lowercase();
        
        // Knowledge graph related topics
        if content_lower.contains("sparql") || content_lower.contains("query") {
            topics.push(Topic {
                id: "sparql_queries".to_string(),
                name: "SPARQL Queries".to_string(),
                keywords: vec!["sparql".to_string(), "query".to_string(), "select".to_string()],
                confidence: 0.8,
                first_mentioned: chrono::Utc::now(),
                last_mentioned: chrono::Utc::now(),
                message_count: 1,
            });
        }
        
        if content_lower.contains("graph") || content_lower.contains("triple") || content_lower.contains("rdf") {
            topics.push(Topic {
                id: "knowledge_graph".to_string(),
                name: "Knowledge Graph".to_string(),
                keywords: vec!["graph".to_string(), "triple".to_string(), "rdf".to_string()],
                confidence: 0.7,
                first_mentioned: chrono::Utc::now(),
                last_mentioned: chrono::Utc::now(),
                message_count: 1,
            });
        }
        
        if content_lower.contains("entity") || content_lower.contains("relationship") {
            topics.push(Topic {
                id: "entities_relations".to_string(),
                name: "Entities and Relationships".to_string(),
                keywords: vec!["entity".to_string(), "relationship".to_string(), "property".to_string()],
                confidence: 0.6,
                first_mentioned: chrono::Utc::now(),
                last_mentioned: chrono::Utc::now(),
                message_count: 1,
            });
        }
        
        topics
    }

    fn classify_transition(&self, from: &Topic, to: &Topic) -> TransitionType {
        // Check if topics share keywords
        let shared_keywords = from.keywords.iter()
            .any(|k| to.keywords.contains(k));
        
        if shared_keywords {
            TransitionType::Natural
        } else if self.topic_history.iter().any(|t| t.to_topic == to.id) {
            TransitionType::Return
        } else {
            TransitionType::Abrupt
        }
    }

    pub fn get_topic_summary(&self) -> String {
        if self.current_topics.is_empty() {
            "No specific topic identified".to_string()
        } else {
            let topics: Vec<String> = self.current_topics
                .iter()
                .map(|t| t.name.clone())
                .collect();
            format!("Current topics: {}", topics.join(", "))
        }
    }
}
