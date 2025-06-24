//! # OxiRS Chat
//!
//! RAG chat API with LLM integration and natural language to SPARQL translation.
//!
//! This crate provides a conversational interface for knowledge graphs,
//! combining retrieval-augmented generation (RAG) with SPARQL querying.

use anyhow::Result;

pub mod rag;
pub mod llm;
pub mod nl2sparql;
pub mod chat;
pub mod server;

/// Chat session configuration
#[derive(Debug, Clone)]
pub struct ChatConfig {
    pub max_context_length: usize,
    pub temperature: f32,
    pub max_retrieval_results: usize,
    pub enable_sparql_generation: bool,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            max_context_length: 4096,
            temperature: 0.7,
            max_retrieval_results: 10,
            enable_sparql_generation: true,
        }
    }
}

/// Chat message
#[derive(Debug, Clone)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metadata: Option<MessageMetadata>,
}

/// Message role
#[derive(Debug, Clone)]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

/// Message metadata
#[derive(Debug, Clone)]
pub struct MessageMetadata {
    pub sparql_query: Option<String>,
    pub retrieved_triples: Option<Vec<String>>,
    pub confidence_score: Option<f32>,
}

/// Chat session
pub struct ChatSession {
    pub id: String,
    pub config: ChatConfig,
    pub messages: Vec<Message>,
    store: std::sync::Arc<oxirs_core::store::Store>,
}

impl ChatSession {
    pub fn new(id: String, store: std::sync::Arc<oxirs_core::store::Store>) -> Self {
        Self {
            id,
            config: ChatConfig::default(),
            messages: Vec::new(),
            store,
        }
    }
    
    /// Process a user message and generate a response
    pub async fn process_message(&mut self, user_input: String) -> Result<Message> {
        // Add user message to history
        let user_message = Message {
            role: MessageRole::User,
            content: user_input.clone(),
            timestamp: chrono::Utc::now(),
            metadata: None,
        };
        self.messages.push(user_message);
        
        // TODO: Implement RAG pipeline
        // 1. Extract entities from user input
        // 2. Retrieve relevant triples from knowledge graph
        // 3. Generate SPARQL query if needed
        // 4. Generate response using LLM
        
        let response = Message {
            role: MessageRole::Assistant,
            content: format!("I understand you're asking about: '{}'. Let me search the knowledge graph for relevant information.", user_input),
            timestamp: chrono::Utc::now(),
            metadata: Some(MessageMetadata {
                sparql_query: None,
                retrieved_triples: None,
                confidence_score: Some(0.8),
            }),
        };
        
        self.messages.push(response.clone());
        Ok(response)
    }
    
    /// Get chat history
    pub fn get_history(&self) -> &[Message] {
        &self.messages
    }
    
    /// Clear chat history
    pub fn clear_history(&mut self) {
        self.messages.clear();
    }
}

/// Chat manager for multiple sessions
pub struct ChatManager {
    sessions: std::collections::HashMap<String, ChatSession>,
    store: std::sync::Arc<oxirs_core::store::Store>,
}

impl ChatManager {
    pub fn new(store: std::sync::Arc<oxirs_core::store::Store>) -> Self {
        Self {
            sessions: std::collections::HashMap::new(),
            store,
        }
    }
    
    /// Create a new chat session
    pub fn create_session(&mut self, session_id: String) -> &mut ChatSession {
        let session = ChatSession::new(session_id.clone(), self.store.clone());
        self.sessions.insert(session_id.clone(), session);
        self.sessions.get_mut(&session_id).unwrap()
    }
    
    /// Get an existing session
    pub fn get_session(&mut self, session_id: &str) -> Option<&mut ChatSession> {
        self.sessions.get_mut(session_id)
    }
    
    /// Remove a session
    pub fn remove_session(&mut self, session_id: &str) -> Option<ChatSession> {
        self.sessions.remove(session_id)
    }
}