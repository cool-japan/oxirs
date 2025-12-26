//! # OxiRS Chat
//!
//! [![Version](https://img.shields.io/badge/version-0.1.0--rc.1-blue)](https://github.com/cool-japan/oxirs/releases)
//! [![docs.rs](https://docs.rs/oxirs-chat/badge.svg)](https://docs.rs/oxirs-chat)
//!
//! **Status**: Beta Release (v0.1.0-rc.1)
//! **Stability**: Public APIs are stable. Production-ready with comprehensive testing.
//!
#![allow(dead_code)]
//!
//! Advanced RAG chat API with LLM integration, natural language to SPARQL translation,
//! streaming responses, self-healing capabilities, and consciousness-inspired computing.
//!
//! This crate provides a production-ready conversational interface for knowledge graphs,
//! combining retrieval-augmented generation (RAG) with SPARQL querying, vector search,
//! and advanced AI features including temporal reasoning and consciousness-guided processing.
//!
//! ## Key Features
//!
//! ### 🧠 Consciousness-Inspired Computing
//! - Temporal memory bank with event tracking
//! - Pattern recognition for conversation understanding
//! - Future projection and implication analysis
//! - Emotional context awareness and sentiment analysis
//! - Multi-level consciousness integration
//!
//! ### ⚡ Real-Time Streaming
//! - Progressive response streaming with status updates
//! - Context delivery during processing
//! - Word-by-word response generation
//! - Asynchronous processing with tokio integration
//! - Configurable chunk sizes and delays
//!
//! ### 🔧 Self-Healing System
//! - Automated health monitoring and issue detection
//! - 8 different healing action types for comprehensive recovery
//! - Recovery statistics tracking with success rate monitoring
//! - Cooldown management and attempt limiting
//! - Component-specific healing actions
//!
//! ### 🔍 Advanced Query Processing
//! - Natural language to SPARQL translation
//! - Vector similarity search integration
//! - Context-aware query understanding
//! - Multi-modal reasoning capabilities
//! - Enterprise security and authentication
//!
//! ## Quick Start Example
//!
//! ```rust,no_run
//! use oxirs_chat::{ChatSession, Message, MessageRole, OxiRSChat, ChatConfig};
//! use oxirs_core::ConcreteStore;
//! use std::sync::Arc;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Initialize the store and chat system
//! let store = Arc::new(ConcreteStore::new()?);
//! let config = ChatConfig::default();
//! let chat_system = OxiRSChat::new(config, store as Arc<dyn oxirs_core::Store>).await?;
//!
//! // Create a chat session
//! let session = chat_system.create_session("user123".to_string()).await?;
//!
//! // Process with integrated RAG  
//! let response = chat_system.process_message(
//!     "user123",
//!     "What genes are associated with breast cancer?".to_string()
//! ).await?;
//!
//! println!("Response: {:?}", response);
//! # Ok(())
//! # }
//! ```
//!
//! ## Streaming Response Example
//!
//! ```rust,no_run
//! use oxirs_chat::{OxiRSChat, ChatConfig};
//! use oxirs_core::ConcreteStore;
//! use std::sync::Arc;
//!
//! # async fn streaming_example() -> anyhow::Result<()> {
//! # let store = Arc::new(ConcreteStore::new()?);
//! # let config = ChatConfig::default();
//! # let chat_system = OxiRSChat::new(config, store as Arc<dyn oxirs_core::Store>).await?;
//! # let _session = chat_system.create_session("user123".to_string()).await?;
//! // Process message with streaming (feature under development)
//! let response = chat_system.process_message(
//!     "user123",
//!     "Explain the relationship between BRCA1 and cancer".to_string()
//! ).await?;
//!
//! println!("Response: {:?}", response);
//! // Note: Streaming API is available through internal components
//! // Future versions will expose streaming API directly
//! # Ok(())
//! # }
//! // Original streaming code for reference:
//! /*
//! while let Some(chunk) = stream.next().await {
//!     match chunk? {
//!         StreamResponseChunk::Status { stage, progress } => {
//!             println!("Stage: {:?}, Progress: {:.1}%", stage, progress * 100.0);
//!         }
//!         StreamResponseChunk::Context { facts, sparql_results } => {
//!             println!("Found {} facts", facts.len());
//!         }
//!         StreamResponseChunk::Content { text } => {
//!             print!("{}", text); // Stream text word by word
//!         }
//!         StreamResponseChunk::Complete { total_time } => {
//!             println!("\nCompleted in {:.2}s", total_time.as_secs_f64());
//!             break;
//!         }
//!         _ => {}
//!     }
//! }
//! */
//! ```
//!
//! ## Self-Healing System Example
//!
//! ```rust,no_run
//! use oxirs_chat::health_monitoring::{HealthMonitor, HealthMonitoringConfig, HealthStatus};
//!
//! # async fn healing_example() -> anyhow::Result<()> {
//! let config = HealthMonitoringConfig::default();
//! let health_monitor = HealthMonitor::new(config);
//!
//! // Generate health report
//! let health_report = health_monitor.generate_health_report().await?;
//!
//! match health_report.overall_status {
//!     HealthStatus::Healthy => println!("System is healthy"),
//!     HealthStatus::Degraded => println!("System performance is degraded"),
//!     HealthStatus::Unhealthy => println!("System has health issues"),
//!     HealthStatus::Critical => println!("System is in critical state"),
//! }
//!
//! println!("System uptime: {:?}", health_report.uptime);
//! # Ok(())
//! # }
//! ```
//!
//! ## Advanced Configuration
//!
//! ```rust,no_run
//! use oxirs_chat::{ChatConfig};
//! use std::time::Duration;
//!
//! # async fn config_example() -> anyhow::Result<()> {
//! let chat_config = ChatConfig {
//!     max_context_tokens: 16000,
//!     sliding_window_size: 50,
//!     enable_context_compression: true,
//!     temperature: 0.8,
//!     max_tokens: 4000,
//!     timeout_seconds: 60,
//!     enable_topic_tracking: true,
//!     enable_sentiment_analysis: true,
//!     enable_intent_detection: true,
//! };
//!
//! // Use the configuration to create a chat system
//! // let store = Arc::new(ConcreteStore::new());
//! // let chat_system = OxiRSChat::new(chat_config, store).await?;
//!
//! println!("Chat system configured with advanced features");
//! # Ok(())
//! # }
//! ```

use anyhow::{Context, Result};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};

// Core modules
// pub mod advanced_observability; // FUTURE: Advanced observability with audit trails (requires scirs2-core beta.4+)
pub mod analytics;
pub mod cache;
pub mod chat;
pub mod chat_session;
pub mod collaboration; // NEW: Real-time collaboration with shared sessions
pub mod collaboration_server; // NEW: Server endpoints for collaboration
pub mod context;
pub mod custom_prompts; // NEW: Custom prompts system for users
pub mod custom_tools; // NEW: Custom tools framework for extensibility
pub mod dashboard; // NEW: Analytics dashboard backend
pub mod dashboard_server; // NEW: Dashboard API endpoints
pub mod enterprise_integration;
pub mod error;
pub mod explanation;
pub mod exploration_guidance; // NEW: Data exploration guidance
pub mod external_services;
// pub mod gpu_embedding_cache; // FUTURE: GPU-accelerated embedding cache (requires scirs2-core beta.4+)
pub mod graph_exploration;
pub mod health_monitoring;
pub mod i18n; // NEW: Internationalization and multi-language support
pub mod knowledge_bases; // NEW: Wikipedia, PubMed, and external knowledge base connectors
pub mod llm;
pub mod message_analytics;
pub mod messages;
pub mod nl2sparql;
pub mod nlp; // Natural Language Processing (NEW: intent, sentiment, entities, coreference)
pub mod performance;
// pub mod performance_profiler; // FUTURE: Advanced performance profiling (requires scirs2-core beta.4+)
pub mod persistence;
pub mod query_refinement; // NEW: Query refinement system
pub mod rag;
pub mod schema_introspection; // NEW: Automatic schema discovery for better NL2SPARQL
                              // pub mod revolutionary_chat_optimization; // Temporarily disabled - requires scirs2-core beta.4 APIs
pub mod export; // Multi-format export (NEW)
pub mod plugins; // Plugin system (NEW)
pub mod rich_content;
pub mod server;
pub mod session;
pub mod session_manager;
pub mod sparql_optimizer;
pub mod suggestions; // Query suggestions (NEW)
pub mod types;
pub mod utils; // Utility modules for stats, NLP, and ranking
pub mod visualization; // NEW: Result visualization helpers
pub mod voice; // NEW: Voice interface with STT/TTS
pub mod webhooks; // Webhook support (NEW)
pub mod workflow;

// Re-export commonly used types
pub use chat_session::{ChatSession, SessionStatistics};
pub use messages::{Message, MessageAttachment, MessageContent, MessageRole, RichContentElement};
pub use session::*;
pub use session_manager::{
    ChatConfig, ContextWindow, SessionData, SessionMetrics, SessionState, TopicTracker,
};
pub use types::*;
pub use types::{SessionStats, ThreadInfo};

// Re-export key RAG types
pub use rag::{AssembledContext, QueryContext, RAGConfig, RAGSystem};

// Re-export schema introspection types
pub use schema_introspection::{
    DiscoveredSchema, IntrospectionConfig, RdfClass, RdfProperty, SchemaIntrospector,
};

// LLM manager type alias for chat functionality
pub type ChatManager = llm::manager::EnhancedLLMManager;

// Re-export LLM types including circuit breaker
pub use llm::{
    CircuitBreakerConfig, CircuitBreakerState, CircuitBreakerStats, LLMConfig, LLMResponse,
};

// Re-export collaboration types
pub use collaboration::{
    AccessControl, CollaborationConfig, CollaborationManager, CollaborationStats,
    CollaborationUpdate, CursorPosition, Participant, ParticipantRole, ParticipantStatus,
    SharedSession, TextRange,
};

// Re-export voice interface types
pub use voice::{
    AudioFormat, SpeechToTextProvider, SttProviderType, SttResult, SttStreamResult,
    TextToSpeechProvider, TtsProviderType, TtsResult, VoiceConfig, VoiceInterface, WordTimestamp,
};

// Re-export dashboard types
pub use dashboard::{
    ActivityDataPoint, DashboardAnalytics, DashboardConfig, DashboardOverview, ExportFormat,
    HealthAnalytics, HealthDataPoint, QueryAnalytics, QueryRecord, QueryType, SystemHealthMetrics,
    TimeRange, UserActivity, UserActivityTracker, UserAnalytics,
};

// Re-export revolutionary chat optimization types
// Temporarily disabled - requires scirs2-core beta.4 APIs
/*
pub use revolutionary_chat_optimization::{
    RevolutionaryChatOptimizer, RevolutionaryChatConfig, UnifiedOptimizationConfig,
    AdvancedStatisticsConfig, ConversationAnalysisConfig, ChatPerformanceTargets,
    ChatOptimizationResult, ConversationStatistics, ConversationInsights,
    ChatOptimizationStrategy, StreamingOptimizationResult, RevolutionaryChatOptimizerFactory,
    CoordinationStrategy, ChatOptimizationPriority, ChatProcessingContext,
};
*/

/// Main chat interface for OxiRS with advanced AI capabilities
pub struct OxiRSChat {
    /// Configuration for the chat system
    pub config: ChatConfig,
    /// RDF store for knowledge graph access
    pub store: Arc<dyn oxirs_core::Store>,
    /// Session storage
    sessions: Arc<RwLock<HashMap<String, Arc<Mutex<ChatSession>>>>>,
    /// Session timeout duration
    session_timeout: Duration,
    /// Advanced RAG engine with quantum, consciousness, and reasoning capabilities
    rag_engine: Arc<Mutex<rag::RagEngine>>,
    /// LLM integration for natural language processing
    llm_manager: Arc<Mutex<llm::LLMManager>>,
    /// NL2SPARQL translation engine
    nl2sparql_engine: Arc<Mutex<nl2sparql::NL2SPARQLSystem>>,
}

impl OxiRSChat {
    /// Create a new OxiRS Chat instance with advanced AI capabilities
    pub async fn new(config: ChatConfig, store: Arc<dyn oxirs_core::Store>) -> Result<Self> {
        Self::new_with_llm_config(config, store, None).await
    }

    /// Create a new OxiRS Chat instance with custom LLM configuration
    pub async fn new_with_llm_config(
        config: ChatConfig,
        store: Arc<dyn oxirs_core::Store>,
        llm_config: Option<llm::LLMConfig>,
    ) -> Result<Self> {
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

        let mut rag_engine =
            rag::RagEngine::new(rag_config, store.clone() as Arc<dyn oxirs_core::Store>);
        rag_engine
            .initialize()
            .await
            .context("Failed to initialize RAG engine")?;

        // Initialize LLM manager with provided config or default
        let llm_config = llm_config.unwrap_or_default();
        let llm_manager = llm::LLMManager::new(llm_config)?;

        // Initialize NL2SPARQL engine with store for schema discovery
        let nl2sparql_config = nl2sparql::NL2SPARQLConfig::default();
        let nl2sparql_engine =
            nl2sparql::NL2SPARQLSystem::with_store(nl2sparql_config, None, store.clone())?;

        // Optionally discover schema for schema-aware query generation
        // This can take some time for large datasets, so it's done in background
        let nl2sparql_for_schema = Arc::new(Mutex::new(nl2sparql_engine));
        let nl2sparql_clone = nl2sparql_for_schema.clone();

        // Spawn background task for schema discovery
        tokio::spawn(async move {
            let mut engine = nl2sparql_clone.lock().await;
            if let Err(e) = engine.discover_schema().await {
                warn!("Failed to discover schema for NL2SPARQL: {}", e);
            } else {
                info!("Schema discovery completed for NL2SPARQL enhancement");
            }
        });

        Ok(Self {
            config,
            store,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            session_timeout: Duration::from_secs(3600), // 1 hour default
            rag_engine: Arc::new(Mutex::new(rag_engine)),
            llm_manager: Arc::new(Mutex::new(llm_manager)),
            nl2sparql_engine: nl2sparql_for_schema,
        })
    }

    /// Manually trigger schema discovery for NL2SPARQL (if not done automatically)
    pub async fn discover_schema(&self) -> Result<()> {
        let mut nl2sparql = self.nl2sparql_engine.lock().await;
        nl2sparql.discover_schema().await
    }

    /// Get the discovered schema (if available)
    pub async fn get_discovered_schema(&self) -> Option<DiscoveredSchema> {
        let nl2sparql = self.nl2sparql_engine.lock().await;
        nl2sparql.get_schema().cloned()
    }

    /// Create a new chat session
    pub async fn create_session(&self, session_id: String) -> Result<Arc<Mutex<ChatSession>>> {
        let session = Arc::new(Mutex::new(ChatSession::new(
            session_id.clone(),
            self.store.clone(),
        )));

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
                if session_guard.should_expire(
                    chrono::Duration::from_std(self.session_timeout)
                        .unwrap_or(chrono::Duration::seconds(3600)),
                ) {
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

    /// Save all active sessions to disk
    pub async fn save_sessions<P: AsRef<std::path::Path>>(
        &self,
        persistence_path: P,
    ) -> Result<usize> {
        use std::fs;

        let sessions = self.sessions.read().await;
        let mut saved_count = 0;

        // Create persistence directory if it doesn't exist
        let persistence_dir = persistence_path.as_ref();
        if !persistence_dir.exists() {
            fs::create_dir_all(persistence_dir)
                .context("Failed to create persistence directory")?;
        }

        info!(
            "Saving {} active sessions to {:?}",
            sessions.len(),
            persistence_dir
        );

        for (session_id, session_arc) in sessions.iter() {
            match session_arc.try_lock() {
                Ok(session) => {
                    let session_data = session.to_data();
                    let session_file = persistence_dir.join(format!("{session_id}.json"));

                    match serde_json::to_string_pretty(&session_data) {
                        Ok(json_data) => {
                            if let Err(e) = fs::write(&session_file, json_data) {
                                error!("Failed to save session {}: {}", session_id, e);
                            } else {
                                debug!("Saved session {} to {:?}", session_id, session_file);
                                saved_count += 1;
                            }
                        }
                        Err(e) => {
                            error!("Failed to serialize session {}: {}", session_id, e);
                        }
                    }
                }
                Err(_) => {
                    warn!("Session {} is locked, skipping save", session_id);
                }
            }
        }

        info!(
            "Successfully saved {} out of {} sessions",
            saved_count,
            sessions.len()
        );
        Ok(saved_count)
    }

    /// Load sessions from disk
    pub async fn load_sessions<P: AsRef<std::path::Path>>(
        &self,
        persistence_path: P,
    ) -> Result<usize> {
        use crate::chat_session::ChatSession;
        use crate::session_manager::SessionData;
        use std::fs;

        let persistence_dir = persistence_path.as_ref();
        if !persistence_dir.exists() {
            info!(
                "Persistence directory {:?} does not exist, no sessions to load",
                persistence_dir
            );
            return Ok(0);
        }

        let mut loaded_count = 0;
        let mut sessions = self.sessions.write().await;

        info!("Loading sessions from {:?}", persistence_dir);

        for entry in fs::read_dir(persistence_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                let session_id = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown");

                match fs::read_to_string(&path) {
                    Ok(json_data) => match serde_json::from_str::<SessionData>(&json_data) {
                        Ok(session_data) => {
                            let session = ChatSession::from_data(session_data, self.store.clone());
                            sessions.insert(session_id.to_string(), Arc::new(Mutex::new(session)));
                            loaded_count += 1;
                            debug!("Loaded session {} from {:?}", session_id, path);
                        }
                        Err(e) => {
                            error!("Failed to deserialize session from {:?}: {}", path, e);
                        }
                    },
                    Err(e) => {
                        error!("Failed to read session file {:?}: {}", path, e);
                    }
                }
            }
        }

        info!("Successfully loaded {} sessions", loaded_count);
        Ok(loaded_count)
    }

    /// Process a chat message with advanced AI capabilities (Quantum RAG, Consciousness, Reasoning)
    pub async fn process_message(&self, session_id: &str, user_message: String) -> Result<Message> {
        let processing_start = std::time::Instant::now();
        info!(
            "Processing message for session {}: {}",
            session_id,
            user_message.chars().take(100).collect::<String>()
        );

        let session = self
            .get_session(session_id)
            .await
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

        // Store user message ID before moving
        let user_msg_id = user_msg.id.clone();

        // Add user message to session
        session.add_message(user_msg)?;

        // **ADVANCED AI PROCESSING PIPELINE**

        // 1. Advanced RAG retrieval with quantum optimization and consciousness
        debug!("Starting advanced RAG retrieval with quantum and consciousness capabilities");
        let assembled_context = {
            let mut rag_engine = self.rag_engine.lock().await;
            rag_engine
                .retrieve(&user_message)
                .await
                .context("Failed to perform advanced RAG retrieval")?
        };

        // 2. Determine if this is a SPARQL-related query
        let (sparql_query, sparql_results) = if self.is_sparql_query(&user_message) {
            debug!("Detected SPARQL query, performing NL2SPARQL translation");
            let mut nl2sparql = self.nl2sparql_engine.lock().await;
            let query_context = rag::QueryContext::new(session_id.to_string()).add_message(
                rag::ConversationMessage {
                    role: rag::MessageRole::User,
                    content: user_message.clone(),
                    timestamp: chrono::Utc::now(),
                },
            );
            match nl2sparql.generate_sparql(&query_context).await {
                Ok(sparql) => {
                    debug!("Generated SPARQL: {}", sparql.query);
                    // Execute SPARQL query
                    match self.execute_sparql(&sparql.query).await {
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
                &mut llm_manager,
                &user_message,
                &assembled_context,
                sparql_query.as_ref(),
                sparql_results.as_ref(),
            )
            .await
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
                reasoning_steps: reasoning_results.primary_chain.steps.clone(),
                confidence_score: reasoning_results.reasoning_quality.overall_quality,
            });
        }

        // Add SPARQL results if available
        if let Some(ref results) = sparql_results {
            rich_elements.push(RichContentElement::SPARQLResults {
                query: sparql_query.map(|s| s.query).unwrap_or_default(),
                results: results.clone(),
                execution_time: processing_start.elapsed(),
            });
        }

        // 5. Create comprehensive response message
        let response_text_len = response_text.len();
        let response = Message {
            id: uuid::Uuid::new_v4().to_string(),
            role: MessageRole::Assistant,
            content: MessageContent::from_text(response_text),
            timestamp: chrono::Utc::now(),
            metadata: Some(messages::MessageMetadata {
                source: Some("oxirs-chat".to_string()),
                confidence: Some(assembled_context.context_score as f64),
                processing_time_ms: Some(processing_start.elapsed().as_millis() as u64),
                model_used: Some("oxirs-chat-ai".to_string()),
                temperature: None,
                max_tokens: None,
                custom_fields: self
                    .create_response_metadata(&assembled_context, processing_start.elapsed())
                    .into_iter()
                    .map(|(k, v)| (k, serde_json::Value::String(v)))
                    .collect(),
            }),
            thread_id: None,
            parent_message_id: Some(user_msg_id),
            token_count: Some(response_text_len / 4), // Rough estimate
            reactions: Vec::new(),
            attachments: Vec::new(),
            rich_elements,
        };

        // Add response to session
        session.add_message(response.clone())?;

        info!(
            "Advanced AI processing completed in {:?} with context score: {:.3}",
            processing_start.elapsed(),
            assembled_context.context_score
        );

        Ok(response)
    }

    /// Helper: Detect if user message contains SPARQL-related intent
    fn is_sparql_query(&self, message: &str) -> bool {
        let sparql_keywords = [
            "select",
            "construct",
            "ask",
            "describe",
            "insert",
            "delete",
            "where",
            "prefix",
            "base",
            "distinct",
            "reduced",
            "from",
            "named",
            "graph",
            "optional",
            "union",
            "minus",
            "bind",
            "values",
            "filter",
            "order by",
            "group by",
            "having",
            "limit",
            "offset",
        ];

        let lowercase_message = message.to_lowercase();
        sparql_keywords
            .iter()
            .any(|&keyword| lowercase_message.contains(keyword))
            || lowercase_message.contains("sparql")
            || lowercase_message.contains("query")
            || lowercase_message.contains("find all")
            || lowercase_message.contains("show me")
            || lowercase_message.contains("list")
    }

    /// Helper: Execute SPARQL query against the store
    async fn execute_sparql(&self, sparql: &str) -> Result<Vec<HashMap<String, String>>> {
        debug!("Executing SPARQL query: {}", sparql);

        // Prepare query against the store
        let query = self
            .store
            .prepare_query(sparql)
            .context("Failed to prepare SPARQL query")?;

        // Execute query and collect results
        let results = query.exec().context("Failed to execute SPARQL query")?;

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
        sparql_query: Option<&nl2sparql::SPARQLGenerationResult>,
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
        prompt.push_str(&format!("User Query: {user_message}\n\n"));

        // Add semantic search results
        if !assembled_context.semantic_results.is_empty() {
            prompt.push_str("Relevant Knowledge Graph Facts:\n");
            for (i, result) in assembled_context
                .semantic_results
                .iter()
                .take(5)
                .enumerate()
            {
                prompt.push_str(&format!(
                    "{}. {} (relevance: {:.2})\n",
                    i + 1,
                    result.triple,
                    result.score
                ));
            }
            prompt.push('\n');
        }

        // Add entity information
        if !assembled_context.extracted_entities.is_empty() {
            prompt.push_str("Extracted Entities:\n");
            for entity in assembled_context.extracted_entities.iter().take(10) {
                prompt.push_str(&format!(
                    "- {} (type: {:?}, confidence: {:.2})\n",
                    entity.text, entity.entity_type, entity.confidence
                ));
            }
            prompt.push('\n');
        }

        // Add reasoning results if available
        if let Some(ref reasoning_results) = assembled_context.reasoning_results {
            prompt.push_str("Advanced Reasoning Analysis:\n");
            for step in reasoning_results.primary_chain.steps.iter().take(3) {
                prompt.push_str(&format!(
                    "- {:?}: {:?} (confidence: {:.2})\n",
                    step.reasoning_type, step.conclusion_triple, step.confidence
                ));
            }
            prompt.push('\n');
        }

        // Add consciousness insights if available
        if let Some(ref consciousness_insights) = assembled_context.consciousness_insights {
            if !consciousness_insights.is_empty() {
                prompt.push_str("Consciousness-Aware Insights:\n");
                for insight in consciousness_insights.iter().take(3) {
                    prompt.push_str(&format!(
                        "- {} (confidence: {:.2})\n",
                        insight.content, insight.confidence
                    ));
                }
                prompt.push('\n');
            }
        }

        // Add SPARQL information if available
        if let Some(sparql) = sparql_query {
            prompt.push_str(&format!("Generated SPARQL Query:\n{}\n\n", sparql.query));

            if let Some(results) = sparql_results {
                prompt.push_str("SPARQL Query Results:\n");
                for (i, result) in results.iter().take(10).enumerate() {
                    prompt.push_str(&format!("{}. ", i + 1));
                    for (key, value) in result {
                        prompt.push_str(&format!("{key}: {value} "));
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
                prompt.push_str(&format!(
                    "Found {} quantum-optimized results with enhanced relevance scoring.\n\n",
                    quantum_results.len()
                ));
            }
        }

        prompt.push_str(
            "Please provide a comprehensive, helpful response based on this information. ",
        );
        prompt.push_str(
            "If SPARQL results are available, incorporate them naturally into your answer. ",
        );
        prompt.push_str("Highlight any interesting patterns or insights you discover.");

        // Generate response using LLM
        debug!(
            "Generating LLM response with context length: {} chars",
            prompt.len()
        );
        let llm_request = llm::LLMRequest {
            messages: vec![llm::ChatMessage {
                role: llm::ChatRole::User,
                content: prompt.clone(),
                metadata: None,
            }],
            system_prompt: Some(
                "You are an advanced AI assistant with access to a knowledge graph.".to_string(),
            ),
            temperature: 0.7,
            max_tokens: Some(1000),
            use_case: llm::UseCase::Conversation,
            priority: llm::Priority::Normal,
            timeout: None,
        };

        let response = llm_manager
            .generate_response(llm_request)
            .await
            .context("Failed to generate LLM response")?;

        Ok(response.content)
    }

    /// Helper: Create metadata for response message
    fn create_response_metadata(
        &self,
        assembled_context: &rag::AssembledContext,
        processing_time: Duration,
    ) -> HashMap<String, String> {
        let mut metadata = HashMap::new();

        metadata.insert(
            "context_score".to_string(),
            assembled_context.context_score.to_string(),
        );
        metadata.insert(
            "processing_time_ms".to_string(),
            processing_time.as_millis().to_string(),
        );
        metadata.insert(
            "semantic_results_count".to_string(),
            assembled_context.semantic_results.len().to_string(),
        );
        metadata.insert(
            "graph_results_count".to_string(),
            assembled_context.graph_results.len().to_string(),
        );
        metadata.insert(
            "extracted_entities_count".to_string(),
            assembled_context.extracted_entities.len().to_string(),
        );
        metadata.insert(
            "assembly_time_ms".to_string(),
            assembled_context.assembly_time.as_millis().to_string(),
        );

        // Add quantum metadata if available
        if let Some(ref quantum_results) = assembled_context.quantum_results {
            metadata.insert(
                "quantum_results_count".to_string(),
                quantum_results.len().to_string(),
            );
            metadata.insert("quantum_enhanced".to_string(), "true".to_string());
        }

        // Add consciousness metadata if available
        if let Some(ref consciousness_insights) = assembled_context.consciousness_insights {
            metadata.insert(
                "consciousness_insights_count".to_string(),
                consciousness_insights.len().to_string(),
            );
            metadata.insert("consciousness_enhanced".to_string(), "true".to_string());
        }

        // Add reasoning metadata if available
        if let Some(ref reasoning_results) = assembled_context.reasoning_results {
            metadata.insert(
                "reasoning_quality".to_string(),
                reasoning_results
                    .reasoning_quality
                    .overall_quality
                    .to_string(),
            );
            metadata.insert("reasoning_enhanced".to_string(), "true".to_string());
        }

        // Add knowledge extraction metadata if available
        if let Some(ref extracted_knowledge) = assembled_context.extracted_knowledge {
            metadata.insert(
                "extracted_knowledge_score".to_string(),
                extracted_knowledge.confidence_score.to_string(),
            );
            metadata.insert(
                "knowledge_extraction_enhanced".to_string(),
                "true".to_string(),
            );
        }

        metadata.insert("oxirs_chat_version".to_string(), VERSION.to_string());
        metadata.insert("advanced_ai_enabled".to_string(), "true".to_string());

        metadata
    }

    /// Get session statistics
    pub async fn get_session_statistics(&self, session_id: &str) -> Result<SessionStatistics> {
        let session = self
            .get_session(session_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;

        let session = session.lock().await;
        Ok(session.get_statistics())
    }

    /// Export session data
    pub async fn export_session(&self, session_id: &str) -> Result<SessionData> {
        let session = self
            .get_session(session_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;

        let session = session.lock().await;
        Ok(session.export_data())
    }

    /// Import session data
    pub async fn import_session(&self, session_data: SessionData) -> Result<()> {
        let session = Arc::new(Mutex::new(ChatSession::from_data(
            session_data.clone(),
            self.store.clone(),
        )));

        let mut sessions = self.sessions.write().await;
        sessions.insert(session_data.id, session);

        Ok(())
    }

    /// Get circuit breaker statistics for all LLM providers
    pub async fn get_circuit_breaker_stats(
        &self,
    ) -> Result<HashMap<String, llm::CircuitBreakerStats>> {
        let llm_manager = self.llm_manager.lock().await;
        Ok(llm_manager.get_circuit_breaker_stats().await)
    }

    /// Reset circuit breaker for a specific LLM provider
    pub async fn reset_circuit_breaker(&self, provider_name: &str) -> Result<()> {
        let llm_manager = self.llm_manager.lock().await;
        llm_manager.reset_circuit_breaker(provider_name).await
    }

    /// Process a chat message with streaming response capability for better user experience
    pub async fn process_message_stream(
        &self,
        session_id: &str,
        user_message: String,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamResponseChunk>> {
        let processing_start = std::time::Instant::now();
        info!(
            "Processing streaming message for session {}: {}",
            session_id,
            user_message.chars().take(100).collect::<String>()
        );

        let (tx, rx) = tokio::sync::mpsc::channel(100);

        let session = self
            .get_session(session_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;

        // Clone necessary data for background processing
        let rag_engine = self.rag_engine.clone();
        let llm_manager = self.llm_manager.clone();
        let nl2sparql_engine = self.nl2sparql_engine.clone();
        let store = self.store.clone();
        let session_id = session_id.to_string();

        // Spawn background task for streaming processing
        tokio::spawn(async move {
            // Send initial status
            let _ = tx
                .send(StreamResponseChunk::Status {
                    stage: ProcessingStage::Initializing,
                    progress: 0.0,
                    message: Some("Starting message processing".to_string()),
                })
                .await;

            // Create and store user message
            let user_msg = Message {
                id: uuid::Uuid::new_v4().to_string(),
                role: MessageRole::User,
                content: MessageContent::from_text(user_message.clone()),
                timestamp: chrono::Utc::now(),
                metadata: None,
                thread_id: None,
                parent_message_id: None,
                token_count: Some(user_message.len() / 4),
                reactions: Vec::new(),
                attachments: Vec::new(),
                rich_elements: Vec::new(),
            };

            let user_msg_id = user_msg.id.clone();

            // Store user message
            {
                let mut session_guard = session.lock().await;
                if let Err(e) = session_guard.add_message(user_msg) {
                    let _ = tx
                        .send(StreamResponseChunk::Error {
                            error: StructuredError {
                                error_type: ErrorType::InternalError,
                                message: format!("Failed to store user message: {e}"),
                                error_code: Some("MSG_STORE_FAILED".to_string()),
                                component: "ChatSession".to_string(),
                                timestamp: chrono::Utc::now(),
                                context: std::collections::HashMap::new(),
                            },
                            recoverable: false,
                        })
                        .await;
                    return;
                }
            }

            // Stage 1: RAG Retrieval
            let _ = tx
                .send(StreamResponseChunk::Status {
                    stage: ProcessingStage::RetrievingContext,
                    progress: 0.1,
                    message: Some("Retrieving relevant context from knowledge graph".to_string()),
                })
                .await;

            let assembled_context = {
                let mut rag_engine = rag_engine.lock().await;
                match rag_engine.retrieve(&user_message).await {
                    Ok(context) => context,
                    Err(e) => {
                        let _ = tx
                            .send(StreamResponseChunk::Error {
                                error: StructuredError {
                                    error_type: ErrorType::RagRetrievalError,
                                    message: format!("RAG retrieval failed: {e}"),
                                    error_code: Some("RAG_RETRIEVAL_FAILED".to_string()),
                                    component: "RagEngine".to_string(),
                                    timestamp: chrono::Utc::now(),
                                    context: std::collections::HashMap::new(),
                                },
                                recoverable: true,
                            })
                            .await;
                        return;
                    }
                }
            };

            let _ = tx
                .send(StreamResponseChunk::Status {
                    stage: ProcessingStage::QuantumProcessing,
                    progress: 0.3,
                    message: Some("Context retrieval complete".to_string()),
                })
                .await;

            // Send context information as early chunks
            if !assembled_context.semantic_results.is_empty() {
                let facts: Vec<String> = assembled_context
                    .semantic_results
                    .iter()
                    .take(5)
                    .map(|result| result.triple.to_string())
                    .collect();

                let entities: Vec<String> = assembled_context
                    .extracted_entities
                    .iter()
                    .take(10)
                    .map(|entity| entity.text.clone())
                    .collect();

                let _ = tx
                    .send(StreamResponseChunk::Context {
                        facts,
                        sparql_results: None,
                        entities,
                    })
                    .await;
            }

            // Stage 2: SPARQL Processing (if applicable)
            let (_sparql_query, _sparql_results) = if user_message.to_lowercase().contains("sparql")
                || user_message.to_lowercase().contains("query")
            {
                let _ = tx
                    .send(StreamResponseChunk::Status {
                        stage: ProcessingStage::GeneratingSparql,
                        progress: 0.5,
                        message: Some("Generating SPARQL query".to_string()),
                    })
                    .await;

                let mut nl2sparql = nl2sparql_engine.lock().await;
                let query_context = rag::QueryContext::new(session_id.clone()).add_message(
                    rag::ConversationMessage {
                        role: rag::MessageRole::User,
                        content: user_message.clone(),
                        timestamp: chrono::Utc::now(),
                    },
                );

                match nl2sparql.generate_sparql(&query_context).await {
                    Ok(sparql) => {
                        let _ = tx
                            .send(StreamResponseChunk::Context {
                                facts: vec!["Generated SPARQL query".to_string()],
                                sparql_results: None,
                                entities: vec![],
                            })
                            .await;

                        // Execute SPARQL
                        let query_result = store.prepare_query(&sparql.query);
                        match query_result {
                            Ok(query) => match query.exec() {
                                Ok(results) => {
                                    let result_count = results.count();
                                    let _ = tx
                                        .send(StreamResponseChunk::Context {
                                            facts: vec![format!(
                                                "SPARQL query returned {} results",
                                                result_count
                                            )],
                                            sparql_results: None,
                                            entities: vec![],
                                        })
                                        .await;
                                    (Some(sparql), Some(Vec::<String>::new())) // Simplified for streaming
                                }
                                Err(_) => (Some(sparql), None),
                            },
                            Err(_) => (None, None),
                        }
                    }
                    Err(_) => (None, None),
                }
            } else {
                (None, None)
            };

            // Stage 3: Response Generation
            let _ = tx
                .send(StreamResponseChunk::Status {
                    stage: ProcessingStage::GeneratingResponse,
                    progress: 0.7,
                    message: Some("Generating AI response".to_string()),
                })
                .await;

            // Build prompt for LLM
            let mut prompt = String::new();
            prompt.push_str("You are an advanced AI assistant with access to a knowledge graph. ");
            prompt.push_str(&format!("User Query: {user_message}\n\n"));

            if !assembled_context.semantic_results.is_empty() {
                prompt.push_str("Relevant Knowledge Graph Facts:\n");
                for (i, result) in assembled_context
                    .semantic_results
                    .iter()
                    .take(3)
                    .enumerate()
                {
                    prompt.push_str(&format!(
                        "{}. {} (relevance: {:.2})\n",
                        i + 1,
                        result.triple,
                        result.score
                    ));
                }
            }

            // Generate response
            let response_text = {
                let mut llm_manager = llm_manager.lock().await;
                let llm_request = llm::LLMRequest {
                    messages: vec![llm::ChatMessage {
                        role: llm::ChatRole::User,
                        content: prompt,
                        metadata: None,
                    }],
                    system_prompt: Some("You are an advanced AI assistant.".to_string()),
                    temperature: 0.7,
                    max_tokens: Some(1000),
                    use_case: llm::UseCase::Conversation,
                    priority: llm::Priority::Normal,
                    timeout: None,
                };

                match llm_manager.generate_response(llm_request).await {
                    Ok(response) => response.content,
                    Err(e) => {
                        let _ = tx
                            .send(StreamResponseChunk::Error {
                                error: StructuredError {
                                    error_type: ErrorType::LlmGenerationError,
                                    message: format!("LLM generation failed: {e}"),
                                    error_code: Some("LLM_GENERATION_FAILED".to_string()),
                                    component: "LLMManager".to_string(),
                                    timestamp: chrono::Utc::now(),
                                    context: std::collections::HashMap::new(),
                                },
                                recoverable: true,
                            })
                            .await;
                        return;
                    }
                }
            };

            // Send response in chunks for streaming effect
            let words: Vec<&str> = response_text.split_whitespace().collect();
            let chunk_size = 3; // Words per chunk

            for (i, chunk) in words.chunks(chunk_size).enumerate() {
                let _progress = 0.8 + (0.2 * i as f32 / (words.len() / chunk_size) as f32);
                let _ = tx
                    .send(StreamResponseChunk::Content {
                        text: chunk.join(" ") + " ",
                        is_complete: false,
                    })
                    .await;

                // Small delay for streaming effect
                tokio::time::sleep(Duration::from_millis(50)).await;
            }

            // Create final response message
            let response = Message {
                id: uuid::Uuid::new_v4().to_string(),
                role: MessageRole::Assistant,
                content: MessageContent::from_text(response_text.clone()),
                timestamp: chrono::Utc::now(),
                metadata: Some(messages::MessageMetadata {
                    source: Some("oxirs-chat-streaming".to_string()),
                    confidence: Some(assembled_context.context_score as f64),
                    processing_time_ms: Some(processing_start.elapsed().as_millis() as u64),
                    model_used: Some("oxirs-chat-ai-streaming".to_string()),
                    temperature: None,
                    max_tokens: None,
                    custom_fields: HashMap::new(),
                }),
                thread_id: None,
                parent_message_id: Some(user_msg_id),
                token_count: Some(response_text.len() / 4),
                reactions: Vec::new(),
                attachments: Vec::new(),
                rich_elements: Vec::new(),
            };

            // Store final response
            {
                let mut session_guard = session.lock().await;
                let _ = session_guard.add_message(response.clone());
            }

            // Send completion
            let _ = tx
                .send(StreamResponseChunk::Complete {
                    total_time: processing_start.elapsed(),
                    token_count: response_text.len() / 4, // Rough estimate
                    final_message: Some("Response generation complete".to_string()),
                })
                .await;
        });

        Ok(rx)
    }
}

/// Create a default OxiRS Chat instance (synchronous helper)
impl OxiRSChat {
    /// Create a default instance synchronously for testing
    pub fn create_default() -> Result<Self> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let store = Arc::new(oxirs_core::ConcreteStore::new()?);
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
        let store = Arc::new(oxirs_core::ConcreteStore::new().expect("Failed to create store"));
        let chat = OxiRSChat::new(ChatConfig::default(), store)
            .await
            .expect("Failed to create chat");

        assert_eq!(chat.session_count().await, 0);
    }

    #[tokio::test]
    async fn test_session_management() {
        let store = Arc::new(oxirs_core::ConcreteStore::new().expect("Failed to create store"));
        let chat = OxiRSChat::new(ChatConfig::default(), store)
            .await
            .expect("Failed to create chat");

        let session_id = "test-session".to_string();
        let _session = chat.create_session(session_id.clone()).await.unwrap();

        assert_eq!(chat.session_count().await, 1);
        assert!(chat.get_session(&session_id).await.is_some());

        let removed = chat.remove_session(&session_id).await;
        assert!(removed);
        assert_eq!(chat.session_count().await, 0);
    }
}
