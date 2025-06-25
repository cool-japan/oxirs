//! # OxiRS Chat
//!
//! RAG chat API with LLM integration and natural language to SPARQL translation.
//!
//! This crate provides a conversational interface for knowledge graphs,
//! combining retrieval-augmented generation (RAG) with SPARQL querying.

use anyhow::Result;
use std::time::{Duration, SystemTime};
use tokio::sync::{Mutex, RwLock};

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
        use crate::{
            llm::{ChatMessage, ChatRole, LLMConfig, LLMManager, LLMRequest, Priority, UseCase},
            nl2sparql::{NL2SPARQLConfig, NL2SPARQLSystem},
            rag::{AssembledContext, QueryContext, QueryIntent, RAGConfig, RAGSystem},
        };

        // Add user message to history
        let user_message = Message {
            role: MessageRole::User,
            content: user_input.clone(),
            timestamp: chrono::Utc::now(),
            metadata: None,
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
                        .map(|t| format!("{} {} {}", t.subject, t.predicate, t.object))
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
        let response_content = if let Some(context) = context_text {
            self.generate_llm_response(&user_input, &context, sparql_query.as_ref())
                .await
                .unwrap_or_else(|_| self.generate_fallback_response(&user_input, &context))
        } else {
            self.generate_simple_response(&user_input)
        };

        let response = Message {
            role: MessageRole::Assistant,
            content: response_content,
            timestamp: chrono::Utc::now(),
            metadata: Some(MessageMetadata {
                sparql_query,
                retrieved_triples,
                confidence_score: Some(confidence_score),
            }),
        };

        self.messages.push(response.clone());
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
}

/// Chat manager for multiple sessions
pub struct ChatManager {
    pub sessions: std::collections::HashMap<String, ChatSession>,
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
