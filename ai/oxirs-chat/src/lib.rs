//! # OxiRS Chat
//!
//! RAG chat API with LLM integration and natural language to SPARQL translation.
//!
//! This crate provides a conversational interface for knowledge graphs,
//! combining retrieval-augmented generation (RAG) with SPARQL querying.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
    time::Duration,
};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};

// Core modules
pub mod analytics;
pub mod cache;
pub mod chat;
pub mod chat_session;
pub mod context;
pub mod enterprise_integration;
pub mod explanation;
pub mod external_services;
pub mod graph_exploration;
pub mod health_monitoring;
pub mod llm;
pub mod message_analytics;
pub mod messages;
pub mod nl2sparql;
pub mod performance;
pub mod persistence;
pub mod rag;
pub mod rich_content;
pub mod server;
pub mod session;
pub mod session_manager;
pub mod sparql_optimizer;
pub mod types;
pub mod workflow;

// Re-export commonly used types
pub use chat_session::{ChatSession, SessionStatistics};
pub use messages::{Message, MessageContent, MessageRole, RichContentElement, MessageAttachment};
pub use session_manager::{ChatConfig, SessionData, SessionState, ContextWindow, TopicTracker, SessionMetrics};
pub use session::*;
pub use types::*;

// Re-export key RAG types
pub use rag::{RAGSystem, RAGConfig, QueryContext, AssembledContext};

/// Main chat interface for OxiRS with advanced AI capabilities
pub struct OxiRSChat {
    /// Configuration for the chat system
    pub config: ChatConfig,
    /// RDF store for knowledge graph access
    pub store: Arc<oxirs_core::Store>,
    /// Session storage
    sessions: Arc<RwLock<HashMap<String, Arc<Mutex<ChatSession>>>>>,
    /// Session timeout duration
    session_timeout: Duration,
    /// Advanced RAG engine with quantum, consciousness, and reasoning capabilities
    rag_engine: Arc<Mutex<rag::RagEngine>>,
    /// LLM integration for natural language processing
    llm_manager: Arc<Mutex<llm::LLMManager>>,
    /// NL2SPARQL translation engine
    nl2sparql_engine: Arc<Mutex<nl2sparql::NL2SPARQLEngine>>,
}

impl OxiRSChat {
    /// Create a new OxiRS Chat instance with advanced AI capabilities
    pub async fn new(config: ChatConfig, store: Arc<oxirs_core::Store>) -> Result<Self> {
        // Initialize RAG engine with advanced features
        let rag_config = rag::RagConfig {
            retrieval: rag::RetrievalConfig {
                enable_quantum_enhancement: true,
                enable_consciousness_integration: true,
                ..Default::default()
            },
            quantum: rag::QuantumConfig {
                enabled: true,
                ..Default::default()
            },
            consciousness: rag::consciousness::ConsciousnessConfig {
                enabled: true,
                ..Default::default()
            },
            ..Default::default()
        };
        
        let mut rag_engine = rag::RagEngine::new(rag_config, store.clone() as Arc<dyn oxirs_core::Store>);
        rag_engine.initialize().await
            .context("Failed to initialize RAG engine")?;
        
        // Initialize LLM manager
        let llm_config = llm::LLMConfig::default();
        let llm_manager = llm::LLMManager::new(llm_config)?;
        
        // Initialize NL2SPARQL engine
        let nl2sparql_config = nl2sparql::NL2SPARQLConfig::default();
        let nl2sparql_engine = nl2sparql::NL2SPARQLEngine::new(nl2sparql_config, store.clone())?;
        
        Ok(Self {
            config,
            store,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            session_timeout: Duration::from_secs(3600), // 1 hour default
            rag_engine: Arc::new(Mutex::new(rag_engine)),
            llm_manager: Arc::new(Mutex::new(llm_manager)),
            nl2sparql_engine: Arc::new(Mutex::new(nl2sparql_engine)),
        })
    }

    /// Create a new chat session
    pub async fn create_session(&self, session_id: String) -> Result<Arc<Mutex<ChatSession>>> {
        let session = Arc::new(Mutex::new(ChatSession::new(session_id.clone(), self.store.clone())));
        
        let mut sessions = self.sessions.write().await;
        sessions.insert(session_id, session.clone());
        
        Ok(session)
    }

    /// Get an existing session
    pub async fn get_session(&self, session_id: &str) -> Option<Arc<Mutex<ChatSession>>> {
        let sessions = self.sessions.read().await;
        sessions.get(session_id).cloned()
    }

    /// Remove a session
    pub async fn remove_session(&self, session_id: &str) -> bool {
        let mut sessions = self.sessions.write().await;
        sessions.remove(session_id).is_some()
    }

    /// List all active sessions
    pub async fn list_sessions(&self) -> Vec<String> {
        let sessions = self.sessions.read().await;
        sessions.keys().cloned().collect()
    }

    /// Clean up expired sessions
    pub async fn cleanup_expired_sessions(&self) -> usize {
        let mut sessions = self.sessions.write().await;
        let mut expired_sessions = Vec::new();

        for (session_id, session) in sessions.iter() {
            if let Ok(session_guard) = session.try_lock() {
                if session_guard.should_expire(self.session_timeout) {
                    expired_sessions.push(session_id.clone());
                }
            }
        }

        for session_id in &expired_sessions {
            sessions.remove(session_id);
        }

        expired_sessions.len()
    }

    /// Get session count
    pub async fn session_count(&self) -> usize {
        let sessions = self.sessions.read().await;
        sessions.len()
    }

    /// Process a chat message with advanced AI capabilities (Quantum RAG, Consciousness, Reasoning)
    pub async fn process_message(
        &self,
        session_id: &str,
        user_message: String,
    ) -> Result<Message> {
        let processing_start = std::time::Instant::now();
        info!("Processing message for session {}: {}", session_id, 
             user_message.chars().take(100).collect::<String>());

        let session = self.get_session(session_id).await
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;

        let mut session = session.lock().await;
        
        // Create user message
        let user_msg = Message {
            id: uuid::Uuid::new_v4().to_string(),
            role: MessageRole::User,
            content: MessageContent::from_text(user_message.clone()),
            timestamp: chrono::Utc::now(),
            metadata: None,
            thread_id: None,
            parent_message_id: None,
            token_count: Some(user_message.len() / 4), // Rough estimate
            reactions: Vec::new(),
            attachments: Vec::new(),
            rich_elements: Vec::new(),
        };

        // Add user message to session
        session.add_message(user_msg)?;

        // **ADVANCED AI PROCESSING PIPELINE**
        
        // 1. Advanced RAG retrieval with quantum optimization and consciousness
        debug!("Starting advanced RAG retrieval with quantum and consciousness capabilities");
        let assembled_context = {
            let mut rag_engine = self.rag_engine.lock().await;
            rag_engine.retrieve(&user_message).await
                .context("Failed to perform advanced RAG retrieval")?
        };
        
        // 2. Determine if this is a SPARQL-related query
        let (sparql_query, sparql_results) = if self.is_sparql_query(&user_message) {
            debug!("Detected SPARQL query, performing NL2SPARQL translation");
            let mut nl2sparql = self.nl2sparql_engine.lock().await;
            match nl2sparql.translate_to_sparql(&user_message, &assembled_context).await {
                Ok(sparql) => {
                    debug!("Generated SPARQL: {}", sparql);
                    // Execute SPARQL query
                    match self.execute_sparql(&sparql).await {
                        Ok(results) => (Some(sparql), Some(results)),
                        Err(e) => {
                            warn!("SPARQL execution failed: {}", e);
                            (Some(sparql), None)
                        }
                    }
                }
                Err(e) => {
                    warn!("NL2SPARQL translation failed: {}", e);
                    (None, None)
                }
            }
        } else {
            (None, None)
        };

        // 3. Generate response using LLM with enhanced context
        debug!("Generating response using LLM with assembled context");
        let response_text = {
            let mut llm_manager = self.llm_manager.lock().await;
            self.generate_enhanced_response(
                &mut *llm_manager,
                &user_message,
                &assembled_context,
                sparql_query.as_ref(),
                sparql_results.as_ref()
            ).await
                .context("Failed to generate enhanced response")?
        };

        // 4. Create rich content elements based on context
        let mut rich_elements = Vec::new();
        
        // Add quantum results visualization if available
        if let Some(ref quantum_results) = assembled_context.quantum_results {
            if !quantum_results.is_empty() {
                rich_elements.push(RichContentElement::QuantumVisualization {
                    results: quantum_results.clone(),
                    entanglement_map: HashMap::new(),
                });
            }
        }
        
        // Add consciousness insights if available
        if let Some(ref consciousness_insights) = assembled_context.consciousness_insights {
            if !consciousness_insights.is_empty() {
                rich_elements.push(RichContentElement::ConsciousnessInsights {
                    insights: consciousness_insights.clone(),
                    awareness_level: 0.8, // From consciousness processing
                });
            }
        }
        
        // Add reasoning chains if available
        if let Some(ref reasoning_results) = assembled_context.reasoning_results {
            rich_elements.push(RichContentElement::ReasoningChain {
                reasoning_steps: reasoning_results.reasoning_chain.clone(),
                confidence_score: reasoning_results.reasoning_quality.overall_quality,
            });
        }
        
        // Add SPARQL results if available
        if let Some(ref results) = sparql_results {
            rich_elements.push(RichContentElement::SPARQLResults {
                query: sparql_query.unwrap_or_default(),
                results: results.clone(),
                execution_time: processing_start.elapsed(),
            });
        }

        // 5. Create comprehensive response message
        let response = Message {
            id: uuid::Uuid::new_v4().to_string(),
            role: MessageRole::Assistant,
            content: MessageContent::from_text(response_text),
            timestamp: chrono::Utc::now(),
            metadata: Some(self.create_response_metadata(&assembled_context, processing_start.elapsed())),
            thread_id: None,
            parent_message_id: Some(user_msg.id.clone()),
            token_count: Some(response_text.len() / 4), // Rough estimate
            reactions: Vec::new(),
            attachments: Vec::new(),
            rich_elements,
        };

        // Add response to session
        session.add_message(response.clone())?;

        info!("Advanced AI processing completed in {:?} with context score: {:.3}", 
              processing_start.elapsed(), assembled_context.context_score);

        Ok(response)
    }

    /// Helper: Detect if user message contains SPARQL-related intent
    fn is_sparql_query(&self, message: &str) -> bool {
        let sparql_keywords = [
            "select", "construct", "ask", "describe", "insert", "delete", "where",
            "prefix", "base", "distinct", "reduced", "from", "named", "graph",
            "optional", "union", "minus", "bind", "values", "filter", "order by",
            "group by", "having", "limit", "offset"
        ];
        
        let lowercase_message = message.to_lowercase();
        sparql_keywords.iter().any(|&keyword| lowercase_message.contains(keyword)) ||
        lowercase_message.contains("sparql") ||
        lowercase_message.contains("query") ||
        lowercase_message.contains("find all") ||
        lowercase_message.contains("show me") ||
        lowercase_message.contains("list")
    }

    /// Helper: Execute SPARQL query against the store
    async fn execute_sparql(&self, sparql: &str) -> Result<Vec<HashMap<String, String>>> {
        debug!("Executing SPARQL query: {}", sparql);
        
        // Prepare query against the store
        let query = self.store.prepare_query(sparql)
            .context("Failed to prepare SPARQL query")?;
        
        // Execute query and collect results
        let results = query.exec()
            .context("Failed to execute SPARQL query")?;
        
        let mut result_maps = Vec::new();
        
        // Convert results to string maps for easier handling
        // Note: This is a simplified conversion - real implementation would handle all RDF term types
        for solution in results {
            let mut result_map = HashMap::new();
            for (var, term) in solution.iter() {
                result_map.insert(var.to_string(), term.to_string());
            }
            result_maps.push(result_map);
        }
        
        debug!("SPARQL query returned {} results", result_maps.len());
        Ok(result_maps)
    }

    /// Helper: Generate enhanced response using LLM with all available context
    async fn generate_enhanced_response(
        &self,
        llm_manager: &mut llm::LLMManager,
        user_message: &str,
        assembled_context: &rag::AssembledContext,
        sparql_query: Option<&String>,
        sparql_results: Option<&Vec<HashMap<String, String>>>,
    ) -> Result<String> {
        // Build comprehensive prompt with all context
        let mut prompt = String::new();
        
        // System prompt
        prompt.push_str("You are an advanced AI assistant with access to a knowledge graph. ");
        prompt.push_str("You have quantum-enhanced retrieval, consciousness-aware processing, ");
        prompt.push_str("and advanced reasoning capabilities. ");
        prompt.push_str("Provide helpful, accurate, and insightful responses based on the available context.\n\n");
        
        // User query
        prompt.push_str(&format!("User Query: {}\n\n", user_message));
        
        // Add semantic search results
        if !assembled_context.semantic_results.is_empty() {
            prompt.push_str("Relevant Knowledge Graph Facts:\n");
            for (i, result) in assembled_context.semantic_results.iter().take(5).enumerate() {
                prompt.push_str(&format!("{}. {} (relevance: {:.2})\n", 
                                       i + 1, result.triple, result.score));
            }
            prompt.push('\n');
        }
        
        // Add entity information
        if !assembled_context.extracted_entities.is_empty() {
            prompt.push_str("Extracted Entities:\n");
            for entity in assembled_context.extracted_entities.iter().take(10) {
                prompt.push_str(&format!("- {} (type: {:?}, confidence: {:.2})\n", 
                                       entity.text, entity.entity_type, entity.confidence));
            }
            prompt.push('\n');
        }
        
        // Add reasoning results if available
        if let Some(ref reasoning_results) = assembled_context.reasoning_results {
            prompt.push_str("Advanced Reasoning Analysis:\n");
            for step in reasoning_results.reasoning_chain.iter().take(3) {
                prompt.push_str(&format!("- {}: {} (confidence: {:.2})\n", 
                                       step.reasoning_type, step.conclusion, step.confidence));
            }
            prompt.push('\n');
        }
        
        // Add consciousness insights if available
        if let Some(ref consciousness_insights) = assembled_context.consciousness_insights {
            if !consciousness_insights.is_empty() {
                prompt.push_str("Consciousness-Aware Insights:\n");
                for insight in consciousness_insights.iter().take(3) {
                    prompt.push_str(&format!("- {} (confidence: {:.2})\n", 
                                           insight.description, insight.confidence));
                }
                prompt.push('\n');
            }
        }
        
        // Add SPARQL information if available
        if let Some(sparql) = sparql_query {
            prompt.push_str(&format!("Generated SPARQL Query:\n{}\n\n", sparql));
            
            if let Some(results) = sparql_results {
                prompt.push_str("SPARQL Query Results:\n");
                for (i, result) in results.iter().take(10).enumerate() {
                    prompt.push_str(&format!("{}. ", i + 1));
                    for (key, value) in result {
                        prompt.push_str(&format!("{}: {} ", key, value));
                    }
                    prompt.push('\n');
                }
                prompt.push('\n');
            }
        }
        
        // Add quantum enhancement info if available
        if let Some(ref quantum_results) = assembled_context.quantum_results {
            if !quantum_results.is_empty() {
                prompt.push_str("Quantum-Enhanced Retrieval Information:\n");
                prompt.push_str(&format!("Found {} quantum-optimized results with enhanced relevance scoring.\n\n", 
                                       quantum_results.len()));
            }
        }
        
        prompt.push_str("Please provide a comprehensive, helpful response based on this information. ");
        prompt.push_str("If SPARQL results are available, incorporate them naturally into your answer. ");
        prompt.push_str("Highlight any interesting patterns or insights you discover.");
        
        // Generate response using LLM
        debug!("Generating LLM response with context length: {} chars", prompt.len());
        llm_manager.generate_response(&prompt).await
            .context("Failed to generate LLM response")
    }

    /// Helper: Create metadata for response message
    fn create_response_metadata(
        &self, 
        assembled_context: &rag::AssembledContext,
        processing_time: Duration
    ) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        
        metadata.insert("context_score".to_string(), assembled_context.context_score.to_string());
        metadata.insert("processing_time_ms".to_string(), processing_time.as_millis().to_string());
        metadata.insert("semantic_results_count".to_string(), assembled_context.semantic_results.len().to_string());
        metadata.insert("graph_results_count".to_string(), assembled_context.graph_results.len().to_string());
        metadata.insert("extracted_entities_count".to_string(), assembled_context.extracted_entities.len().to_string());
        metadata.insert("assembly_time_ms".to_string(), assembled_context.assembly_time.as_millis().to_string());
        
        // Add quantum metadata if available
        if let Some(ref quantum_results) = assembled_context.quantum_results {
            metadata.insert("quantum_results_count".to_string(), quantum_results.len().to_string());
            metadata.insert("quantum_enhanced".to_string(), "true".to_string());
        }
        
        // Add consciousness metadata if available
        if let Some(ref consciousness_insights) = assembled_context.consciousness_insights {
            metadata.insert("consciousness_insights_count".to_string(), consciousness_insights.len().to_string());
            metadata.insert("consciousness_enhanced".to_string(), "true".to_string());
        }
        
        // Add reasoning metadata if available
        if let Some(ref reasoning_results) = assembled_context.reasoning_results {
            metadata.insert("reasoning_quality".to_string(), reasoning_results.reasoning_quality.overall_quality.to_string());
            metadata.insert("reasoning_enhanced".to_string(), "true".to_string());
        }
        
        // Add knowledge extraction metadata if available
        if let Some(ref extracted_knowledge) = assembled_context.extracted_knowledge {
            metadata.insert("extracted_knowledge_score".to_string(), extracted_knowledge.confidence_score.to_string());
            metadata.insert("knowledge_extraction_enhanced".to_string(), "true".to_string());
        }
        
        metadata.insert("oxirs_chat_version".to_string(), VERSION.to_string());
        metadata.insert("advanced_ai_enabled".to_string(), "true".to_string());
        
        metadata
    }

    /// Get session statistics
    pub async fn get_session_statistics(&self, session_id: &str) -> Result<SessionStatistics> {
        let session = self.get_session(session_id).await
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;

        let session = session.lock().await;
        Ok(session.get_statistics())
    }

    /// Export session data
    pub async fn export_session(&self, session_id: &str) -> Result<SessionData> {
        let session = self.get_session(session_id).await
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;

        let session = session.lock().await;
        Ok(session.export_data())
    }

    /// Import session data
    pub async fn import_session(&self, session_data: SessionData) -> Result<()> {
        let session = Arc::new(Mutex::new(ChatSession::from_data(session_data.clone(), self.store.clone())));
        
        let mut sessions = self.sessions.write().await;
        sessions.insert(session_data.id, session);
        
        Ok(())
    }
}

/// Create a default OxiRS Chat instance (synchronous helper)
impl OxiRSChat {
    /// Create a default instance synchronously for testing
    pub fn create_default() -> Result<Self> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let store = Arc::new(oxirs_core::Store::new()?);
            Self::new(ChatConfig::default(), store).await
        })
    }
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Feature flags for optional functionality
pub mod features {
    pub const RAG_ENABLED: bool = true;
    pub const NL2SPARQL_ENABLED: bool = true;
    pub const ANALYTICS_ENABLED: bool = true;
    pub const CACHING_ENABLED: bool = true;
    pub const RICH_CONTENT_ENABLED: bool = true;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_chat_creation() {
        let store = Arc::new(oxirs_core::Store::new().expect("Failed to create store"));
        let chat = OxiRSChat::new(ChatConfig::default(), store).await.expect("Failed to create chat");
        
        assert_eq!(chat.session_count().await, 0);
    }

    #[tokio::test]
    async fn test_session_management() {
        let store = Arc::new(oxirs_core::Store::new().expect("Failed to create store"));
        let chat = OxiRSChat::new(ChatConfig::default(), store).await.expect("Failed to create chat");
        
        let session_id = "test-session".to_string();
        let session = chat.create_session(session_id.clone()).await.unwrap();
        
        assert_eq!(chat.session_count().await, 1);
        assert!(chat.get_session(&session_id).await.is_some());
        
        let removed = chat.remove_session(&session_id).await;
        assert!(removed);
        assert_eq!(chat.session_count().await, 0);
    }
}