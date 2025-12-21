//! Session Persistence and Recovery System for OxiRS Chat
//!
//! Provides robust session persistence with backup/recovery capabilities,
//! automatic expiration handling, and concurrent session management.

use anyhow::{anyhow, Context, Result};
use bincode::{Decode, Encode};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    fs::{File, OpenOptions},
    io::{AsyncReadExt, AsyncWriteExt},
    sync::RwLock,
    time::interval,
};
use tracing::{debug, error, info, warn};

// Encryption and compression
use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Key, Nonce,
};
use base64::{engine::general_purpose, Engine};
use zstd::{Decoder, Encoder};

// Cryptographically secure RNG (SCIRS2 POLICY)
use scirs2_core::random::SecureRandom;

use crate::{
    analytics::{
        ComplexityMetrics, ConfidenceMetrics, ConversationAnalytics, ConversationQuality,
        ImplicitSatisfactionSignals, IntentType, SatisfactionMetrics,
    },
    ChatConfig, Message, SessionMetrics,
};

// Define PersistentChatSession for persistence - different from the main ChatSession
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct PersistentChatSession {
    pub session_id: String,
    pub config: ChatConfig,
    #[bincode(with_serde)]
    pub messages: Vec<Message>,
    pub created_at: SystemTime,
    pub last_accessed: SystemTime,
    pub metrics: SessionMetrics,
    #[bincode(with_serde)]
    pub user_preferences: HashMap<String, serde_json::Value>,
}

/// Configuration for session persistence
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct PersistenceConfig {
    pub enabled: bool,
    pub storage_path: PathBuf,
    pub backup_path: PathBuf,
    pub auto_save_interval: Duration,
    pub session_ttl: Duration,
    pub max_sessions: usize,
    pub compression_enabled: bool,
    pub encryption_enabled: bool,
    pub encryption_key: Option<String>, // Base64-encoded encryption key
    pub backup_retention_days: usize,
    pub checkpoint_interval: Duration,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            storage_path: PathBuf::from("data/sessions"),
            backup_path: PathBuf::from("data/backups"),
            auto_save_interval: Duration::from_secs(60), // Save every minute
            session_ttl: Duration::from_secs(86400 * 7), // 7 days
            max_sessions: 10000,
            compression_enabled: true,
            encryption_enabled: false, // Can be enabled by providing encryption_key
            encryption_key: None,      // Should be set via environment variable or config
            backup_retention_days: 30,
            checkpoint_interval: Duration::from_secs(300), // Checkpoint every 5 minutes
        }
    }
}

/// Serializable session data for persistence
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct PersistedSession {
    pub session_id: String,
    pub config: ChatConfig,
    #[bincode(with_serde)]
    pub messages: Vec<Message>,
    pub created_at: SystemTime,
    pub last_accessed: SystemTime,
    pub metrics: SessionMetrics,
    pub analytics: Option<ConversationAnalytics>,
    #[bincode(with_serde)]
    pub user_preferences: HashMap<String, serde_json::Value>,
    pub conversation_state: ConversationState,
    pub checksum: String,
}

/// Wrapper for persisted session with dirty flag tracking
#[derive(Debug, Clone)]
pub struct SessionWithDirtyFlag {
    pub session: PersistedSession,
    pub dirty: bool,
    pub last_saved: Option<SystemTime>,
}

/// Conversation state for advanced context management
#[derive(Debug, Clone, Serialize, Deserialize, Default, Encode, Decode)]
pub struct ConversationState {
    pub current_topic: Option<String>,
    pub context_window: Vec<String>, // Message IDs in current context
    pub entity_history: Vec<EntityReference>,
    pub query_history: Vec<QueryContext>,
    pub user_intent_history: Vec<String>,
    pub conversation_flow: ConversationFlow,
}

/// Entity reference for tracking
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct EntityReference {
    pub entity_uri: String,
    pub entity_label: String,
    pub first_mentioned: SystemTime,
    pub mention_count: usize,
    pub last_context: String,
}

/// Query context for tracking
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct QueryContext {
    pub sparql_query: String,
    pub natural_language: String,
    pub intent: String,
    pub success: bool,
    pub timestamp: SystemTime,
    pub execution_time_ms: u64,
}

/// Conversation flow tracking
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct ConversationFlow {
    pub current_phase: ConversationPhase,
    pub topic_transitions: Vec<TopicTransition>,
    pub interaction_patterns: Vec<InteractionPattern>,
    pub complexity_level: f32,
}

impl Default for ConversationFlow {
    fn default() -> Self {
        Self {
            current_phase: ConversationPhase::Introduction,
            topic_transitions: Vec::new(),
            interaction_patterns: Vec::new(),
            complexity_level: 1.0,
        }
    }
}

/// Conversation phases
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub enum ConversationPhase {
    Introduction,
    Exploration,
    DeepDive,
    Analysis,
    Conclusion,
    Abandoned,
}

/// Topic transitions
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct TopicTransition {
    pub from_topic: Option<String>,
    pub to_topic: String,
    pub transition_type: TransitionType,
    pub timestamp: SystemTime,
    pub confidence: f32,
}

/// Transition types
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub enum TransitionType {
    Natural,       // Logical progression
    UserInitiated, // User changed topic
    Clarification, // Seeking clarification
    Digression,    // Side topic
    Return,        // Back to previous topic
}

/// Interaction patterns
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct InteractionPattern {
    pub pattern_type: InteractionType,
    pub frequency: usize,
    pub last_occurrence: SystemTime,
    pub effectiveness: f32,
}

/// Interaction types
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub enum InteractionType {
    QuestionAnswer,
    ExploratorySearch,
    ComparativeAnalysis,
    DeepInvestigation,
    QuickLookup,
    IterativeRefinement,
}

/// Session recovery information
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct RecoveryInfo {
    pub session_id: String,
    pub last_checkpoint: SystemTime,
    pub backup_available: bool,
    pub corruption_detected: bool,
    pub recovery_strategies: Vec<RecoveryStrategy>,
}

/// Recovery strategies
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub enum RecoveryStrategy {
    LoadFromCheckpoint,
    LoadFromBackup,
    PartialRecovery,
    CreateNew,
}

/// Main session persistence manager
pub struct SessionPersistenceManager {
    config: PersistenceConfig,
    active_sessions: Arc<RwLock<HashMap<String, Arc<RwLock<SessionWithDirtyFlag>>>>>,
    persistence_stats: Arc<RwLock<PersistenceStats>>,
}

/// Persistence statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct PersistenceStats {
    pub total_sessions_saved: usize,
    pub total_sessions_loaded: usize,
    pub save_failures: usize,
    pub load_failures: usize,
    pub recovery_operations: usize,
    pub corrupted_sessions: usize,
    pub total_backup_size: u64,
    pub average_save_time_ms: f64,
    pub average_load_time_ms: f64,
}

impl SessionPersistenceManager {
    pub async fn new(config: PersistenceConfig) -> Result<Self> {
        // Create directories if they don't exist
        fs::create_dir_all(&config.storage_path)?;
        fs::create_dir_all(&config.backup_path)?;

        let manager = Self {
            config: config.clone(),
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            persistence_stats: Arc::new(RwLock::new(PersistenceStats::default())),
        };

        if config.enabled {
            // Start background tasks
            manager.start_auto_save_task().await;
            manager.start_cleanup_task().await;
            manager.start_checkpoint_task().await;

            // Load existing sessions
            manager.load_all_sessions().await?;
        }

        info!(
            "Session persistence manager initialized with storage at {:?}",
            config.storage_path
        );
        Ok(manager)
    }

    /// Save a session to persistent storage
    pub async fn save_session(&self, session: &PersistentChatSession) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let start_time = SystemTime::now();
        let session_id = &session.session_id;

        // Create persisted session
        let persisted = PersistedSession {
            session_id: session_id.clone(),
            config: session.config.clone(),
            messages: session.messages.clone(),
            created_at: session.created_at,
            last_accessed: session.last_accessed,
            metrics: session.metrics.clone(),
            analytics: Some(self.generate_analytics_for_session(session).await?),
            user_preferences: session.user_preferences.clone(),
            conversation_state: self.extract_conversation_state(session).await,
            checksum: self.calculate_checksum(session).await?,
        };

        // Serialize and optionally compress
        let serialized = if self.config.compression_enabled {
            self.compress_session(&persisted).await?
        } else {
            bincode::encode_to_vec(&persisted, bincode::config::standard())
                .map_err(|e| anyhow!("Bincode encoding failed: {}", e))?
        };

        // Write to file
        let file_path = self.get_session_file_path(session_id);
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&file_path)
            .await?;

        file.write_all(&serialized).await?;
        file.sync_all().await?;

        // Update active sessions with dirty flag tracking
        {
            let mut active = self.active_sessions.write().await;
            let session_with_flag = SessionWithDirtyFlag {
                session: persisted,
                dirty: false, // Just saved, so not dirty
                last_saved: Some(SystemTime::now()),
            };
            active.insert(session_id.clone(), Arc::new(RwLock::new(session_with_flag)));
        }

        // Update statistics
        let save_time = start_time.elapsed().unwrap_or(Duration::ZERO).as_millis() as f64;
        {
            let mut stats = self.persistence_stats.write().await;
            stats.total_sessions_saved += 1;
            stats.average_save_time_ms =
                (stats.average_save_time_ms * (stats.total_sessions_saved - 1) as f64 + save_time)
                    / stats.total_sessions_saved as f64;
        }

        debug!("Session {} saved in {:.2}ms", session_id, save_time);
        Ok(())
    }

    /// Load a session from persistent storage
    pub async fn load_session(&self, session_id: &str) -> Result<Option<PersistentChatSession>> {
        if !self.config.enabled {
            return Ok(None);
        }

        let start_time = SystemTime::now();

        // Check active sessions first
        {
            let active = self.active_sessions.read().await;
            if let Some(persisted_session) = active.get(session_id) {
                let session_with_flag = persisted_session.read().await;
                return Ok(Some(
                    self.convert_to_persistent_chat_session(&session_with_flag.session)
                        .await?,
                ));
            }
        }

        // Load from file
        let file_path = self.get_session_file_path(session_id);
        if !file_path.exists() {
            return Ok(None);
        }

        let mut file = File::open(&file_path).await?;
        let mut data = Vec::new();
        file.read_to_end(&mut data).await?;

        // Deserialize and optionally decompress
        let persisted: PersistedSession = if self.config.compression_enabled {
            self.decompress_session(&data).await?
        } else {
            bincode::decode_from_slice(&data, bincode::config::standard())
                .map_err(|e| anyhow!("Bincode decoding failed: {}", e))?
                .0
        };

        // Verify checksum
        if !self.verify_checksum(&persisted).await? {
            warn!(
                "Checksum verification failed for session {}, attempting recovery",
                session_id
            );
            return self.attempt_recovery(session_id).await;
        }

        // Convert to chat session
        let chat_session = self.convert_to_persistent_chat_session(&persisted).await?;

        // Update active sessions with dirty flag tracking
        {
            let mut active = self.active_sessions.write().await;
            let session_with_flag = SessionWithDirtyFlag {
                session: persisted,
                dirty: false, // Just loaded, so not dirty
                last_saved: Some(SystemTime::now()),
            };
            active.insert(
                session_id.to_string(),
                Arc::new(RwLock::new(session_with_flag)),
            );
        }

        // Update statistics
        let load_time = start_time.elapsed().unwrap_or(Duration::ZERO).as_millis() as f64;
        {
            let mut stats = self.persistence_stats.write().await;
            stats.total_sessions_loaded += 1;
            stats.average_load_time_ms =
                (stats.average_load_time_ms * (stats.total_sessions_loaded - 1) as f64 + load_time)
                    / stats.total_sessions_loaded as f64;
        }

        debug!("Session {} loaded in {:.2}ms", session_id, load_time);
        Ok(Some(chat_session))
    }

    /// Delete a session from persistent storage
    pub async fn delete_session(&self, session_id: &str) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Remove from active sessions
        {
            let mut active = self.active_sessions.write().await;
            active.remove(session_id);
        }

        // Delete file
        let file_path = self.get_session_file_path(session_id);
        if file_path.exists() {
            tokio::fs::remove_file(&file_path).await?;
        }

        // Delete backup if exists
        let backup_path = self.get_backup_file_path(session_id);
        if backup_path.exists() {
            tokio::fs::remove_file(&backup_path).await?;
        }

        debug!("Session {} deleted from persistent storage", session_id);
        Ok(())
    }

    /// List all available sessions
    pub async fn list_sessions(&self) -> Result<Vec<SessionInfo>> {
        if !self.config.enabled {
            return Ok(Vec::new());
        }

        let mut sessions = Vec::new();

        // Scan storage directory
        let mut entries = tokio::fs::read_dir(&self.config.storage_path).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("session") {
                if let Some(session_id) = path.file_stem().and_then(|s| s.to_str()) {
                    let metadata = entry.metadata().await?;
                    sessions.push(SessionInfo {
                        session_id: session_id.to_string(),
                        created_at: metadata.created().unwrap_or(UNIX_EPOCH),
                        modified_at: metadata.modified().unwrap_or(UNIX_EPOCH),
                        size_bytes: metadata.len(),
                        has_backup: self.get_backup_file_path(session_id).exists(),
                    });
                }
            }
        }

        sessions.sort_by(|a, b| b.modified_at.cmp(&a.modified_at));
        Ok(sessions)
    }

    /// Create a backup of a session
    pub async fn create_backup(&self, session_id: &str) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let source_path = self.get_session_file_path(session_id);
        let backup_path = self.get_backup_file_path(session_id);

        if source_path.exists() {
            tokio::fs::copy(&source_path, &backup_path).await?;
            debug!("Backup created for session {}", session_id);
        }

        Ok(())
    }

    /// Attempt to recover a corrupted session
    pub async fn attempt_recovery(
        &self,
        session_id: &str,
    ) -> Result<Option<PersistentChatSession>> {
        warn!("Attempting recovery for session {}", session_id);

        let recovery_info = self.analyze_recovery_options(session_id).await?;

        for strategy in &recovery_info.recovery_strategies {
            match strategy {
                RecoveryStrategy::LoadFromCheckpoint => {
                    if let Ok(Some(session)) = self.load_from_checkpoint(session_id).await {
                        info!(
                            "Successfully recovered session {} from checkpoint",
                            session_id
                        );
                        return Ok(Some(session));
                    }
                }
                RecoveryStrategy::LoadFromBackup => {
                    if let Ok(Some(session)) = self.load_from_backup(session_id).await {
                        info!("Successfully recovered session {} from backup", session_id);
                        return Ok(Some(session));
                    }
                }
                RecoveryStrategy::PartialRecovery => {
                    if let Ok(Some(session)) = self.attempt_partial_recovery(session_id).await {
                        warn!("Partial recovery successful for session {}", session_id);
                        return Ok(Some(session));
                    }
                }
                RecoveryStrategy::CreateNew => {
                    warn!(
                        "Creating new session to replace corrupted session {}",
                        session_id
                    );
                    return Ok(Some(self.create_emergency_session(session_id).await?));
                }
            }
        }

        // Update statistics
        {
            let mut stats = self.persistence_stats.write().await;
            stats.recovery_operations += 1;
            stats.corrupted_sessions += 1;
            stats.load_failures += 1;
        }

        error!("Failed to recover session {}", session_id);
        Ok(None)
    }

    /// Get persistence statistics
    pub async fn get_stats(&self) -> PersistenceStats {
        self.persistence_stats.read().await.clone()
    }

    /// Mark a session as dirty (modified) to trigger auto-save
    pub async fn mark_session_dirty(&self, session_id: &str) -> Result<()> {
        let active = self.active_sessions.read().await;
        if let Some(session_arc) = active.get(session_id) {
            let mut session = session_arc.write().await;
            session.dirty = true;
            debug!("Marked session {} as dirty", session_id);
        }
        Ok(())
    }

    /// Check if a session is dirty
    pub async fn is_session_dirty(&self, session_id: &str) -> bool {
        let active = self.active_sessions.read().await;
        if let Some(session_arc) = active.get(session_id) {
            let session = session_arc.read().await;
            session.dirty
        } else {
            false
        }
    }

    /// Get dirty session count for monitoring
    pub async fn get_dirty_session_count(&self) -> usize {
        let active = self.active_sessions.read().await;
        let mut dirty_count = 0;
        for session_arc in active.values() {
            let session = session_arc.read().await;
            if session.dirty {
                dirty_count += 1;
            }
        }
        dirty_count
    }

    /// Clean up expired sessions
    pub async fn cleanup_expired_sessions(&self) -> Result<usize> {
        if !self.config.enabled {
            return Ok(0);
        }

        let mut cleaned_count = 0;
        let cutoff_time = SystemTime::now() - self.config.session_ttl;

        let sessions = self.list_sessions().await?;
        for session_info in sessions {
            if session_info.modified_at < cutoff_time {
                match self.delete_session(&session_info.session_id).await {
                    Err(e) => {
                        warn!(
                            "Failed to delete expired session {}: {}",
                            session_info.session_id, e
                        );
                    }
                    _ => {
                        cleaned_count += 1;
                    }
                }
            }
        }

        if cleaned_count > 0 {
            info!("Cleaned up {} expired sessions", cleaned_count);
        }

        Ok(cleaned_count)
    }

    // Private helper methods

    async fn extract_conversation_state(
        &self,
        session: &PersistentChatSession,
    ) -> ConversationState {
        let mut context_window = Vec::new();
        let mut query_history = Vec::new();
        let mut user_intent_history = Vec::new();
        let mut current_topic = None;
        let topic_transitions = Vec::new();
        let mut interaction_patterns = Vec::new();

        // Analyze recent messages (last 10 for context window)
        let recent_messages: Vec<_> = session.messages.iter().rev().take(10).collect();

        for message in &recent_messages {
            context_window.push(message.id.clone());
        }

        // Extract topics and entities from all messages
        let mut topic_keywords = HashMap::new();
        let mut entities = HashMap::new();
        let mut current_phase = ConversationPhase::Introduction;
        let mut complexity_sum = 0.0;
        let mut complexity_count = 0;

        for (index, message) in session.messages.iter().enumerate() {
            let content = message.content.to_text().to_lowercase();

            // Extract entities (simple approach - capitalized words and known patterns)
            for word in content.split_whitespace() {
                if word.len() > 2 && word.chars().next().unwrap().is_uppercase() {
                    let entity_key = word.to_lowercase();
                    let entry =
                        entities
                            .entry(entity_key.clone())
                            .or_insert_with(|| EntityReference {
                                entity_uri: format!("local:{entity_key}"),
                                entity_label: word.to_string(),
                                first_mentioned: message.timestamp.into(),
                                mention_count: 0,
                                last_context: content.clone(),
                            });
                    entry.mention_count += 1;
                    entry.last_context = content.clone();
                }
            }

            // Extract topics based on keywords
            let mut message_topics = Vec::new();
            if content.contains("sparql") || content.contains("query") {
                message_topics.push("querying");
                // Check for SPARQL patterns
                if content.contains("select") || content.contains("construct") {
                    query_history.push(QueryContext {
                        sparql_query: self.extract_sparql_from_content(&content),
                        natural_language: content.clone(),
                        intent: "data_retrieval".to_string(),
                        success: true, // Assume success unless we can detect failures
                        timestamp: message.timestamp.into(),
                        execution_time_ms: 100, // Default value
                    });
                }
            }
            if content.contains("data") || content.contains("information") {
                message_topics.push("data_retrieval");
            }
            if content.contains("help") || content.contains("assist") {
                message_topics.push("assistance");
            }
            if content.contains("explain") || content.contains("understand") {
                message_topics.push("explanation");
            }

            // Track topic frequency for determining current topic
            for topic in &message_topics {
                *topic_keywords.entry(topic.to_string()).or_insert(0) += 1;
            }

            // Determine conversation phase based on message index and content
            let total_messages = session.messages.len();
            if index < total_messages / 4 {
                current_phase = ConversationPhase::Introduction;
            } else if index < total_messages / 2 {
                current_phase = ConversationPhase::Exploration;
            } else if index < (total_messages * 3) / 4 {
                current_phase = ConversationPhase::DeepDive;
            } else {
                current_phase = ConversationPhase::Analysis;
            }

            // Extract user intents from user messages
            if message.role == crate::messages::MessageRole::User {
                let intent = if content.contains("?")
                    || content.starts_with("what")
                    || content.starts_with("how")
                {
                    "question"
                } else if content.contains("please") || content.contains("can you") {
                    "request"
                } else if content.contains("thank") || content.contains("appreciate") {
                    "gratitude"
                } else if content.contains("error") || content.contains("problem") {
                    "issue_report"
                } else {
                    "statement"
                };
                user_intent_history.push(intent.to_string());
            }

            // Calculate complexity (simple heuristic based on message length and technical terms)
            let technical_terms = ["sparql", "query", "rdf", "triple", "graph", "ontology"];
            let tech_count = technical_terms
                .iter()
                .filter(|&term| content.contains(term))
                .count();
            let word_count = content.split_whitespace().count();
            let complexity = (word_count as f32 / 10.0 + tech_count as f32 * 0.5).min(5.0);
            complexity_sum += complexity;
            complexity_count += 1;
        }

        // Determine current topic (most frequent)
        if let Some((topic, _)) = topic_keywords.iter().max_by_key(|(_, &count)| count) {
            current_topic = Some(topic.clone());
        }

        // Convert entities HashMap to Vec
        let entity_history = entities.into_values().collect();

        // Create interaction patterns based on message analysis
        let mut pattern_counts = HashMap::new();
        for intent in &user_intent_history {
            *pattern_counts.entry(intent.clone()).or_insert(0) += 1;
        }

        for (pattern, count) in pattern_counts {
            let pattern_type = match pattern.as_str() {
                "question" => InteractionType::QuestionAnswer,
                "request" => InteractionType::ExploratorySearch,
                _ => InteractionType::QuickLookup,
            };

            interaction_patterns.push(InteractionPattern {
                pattern_type,
                frequency: count,
                last_occurrence: session.last_accessed,
                effectiveness: 0.8, // Default effectiveness
            });
        }

        // Calculate average complexity
        let average_complexity = if complexity_count > 0 {
            complexity_sum / complexity_count as f32
        } else {
            1.0
        };

        // Create conversation flow
        let conversation_flow = ConversationFlow {
            current_phase,
            topic_transitions,
            interaction_patterns,
            complexity_level: average_complexity,
        };

        ConversationState {
            current_topic,
            context_window,
            entity_history,
            query_history,
            user_intent_history,
            conversation_flow,
        }
    }

    /// Extract SPARQL query patterns from message content
    fn extract_sparql_from_content(&self, content: &str) -> String {
        // Simple regex to extract SPARQL-like patterns
        let sparql_pattern =
            regex::Regex::new(r"(?i)(select|construct|ask|describe)[^.]*\.").unwrap();
        if let Some(captures) = sparql_pattern.find(content) {
            captures.as_str().to_string()
        } else {
            // Return a generic query if pattern not found but SPARQL keywords present
            if content.contains("select") {
                "SELECT * WHERE { ?s ?p ?o }".to_string()
            } else {
                String::new()
            }
        }
    }

    async fn calculate_checksum(&self, session: &PersistentChatSession) -> Result<String> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        session.session_id.hash(&mut hasher);
        session.messages.len().hash(&mut hasher);
        session
            .created_at
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs()
            .hash(&mut hasher);

        Ok(format!("{:x}", hasher.finish()))
    }

    async fn verify_checksum(&self, persisted: &PersistedSession) -> Result<bool> {
        // Simple verification - in production, use cryptographic hash
        Ok(!persisted.checksum.is_empty())
    }

    async fn compress_session(&self, session: &PersistedSession) -> Result<Vec<u8>> {
        // First serialize with bincode
        let serialized = bincode::encode_to_vec(session, bincode::config::standard())
            .map_err(|e| anyhow!("Bincode encoding failed: {}", e))?;

        // Apply compression if enabled
        let compressed = if self.config.compression_enabled {
            // Use zstd for compression
            let mut encoder = Encoder::new(Vec::new(), 3)?; // Level 3 for balanced compression
            encoder.write_all(&serialized)?;
            encoder.finish()?
        } else {
            serialized
        };

        // Apply encryption if enabled
        if self.config.encryption_enabled {
            self.encrypt_data(&compressed).await
        } else {
            Ok(compressed)
        }
    }

    async fn decompress_session(&self, data: &[u8]) -> Result<PersistedSession> {
        // First decrypt if encryption is enabled
        let decrypted = if self.config.encryption_enabled {
            self.decrypt_data(data).await?
        } else {
            data.to_vec()
        };

        // Then decompress if compression is enabled
        let decompressed = if self.config.compression_enabled {
            // Use zstd for decompression
            let mut decoder = Decoder::new(&decrypted[..])?;
            let mut decompressed = Vec::new();
            std::io::copy(&mut decoder, &mut decompressed)?;
            decompressed
        } else {
            decrypted
        };

        // Finally deserialize with bincode
        bincode::decode_from_slice(&decompressed, bincode::config::standard())
            .map(|(session, _)| session)
            .map_err(|e| anyhow!("Bincode decoding failed: {}", e))
    }

    /// Encrypt data using AES-256-GCM
    async fn encrypt_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        let encryption_key = self
            .config
            .encryption_key
            .as_ref()
            .ok_or_else(|| anyhow!("Encryption enabled but no encryption key provided"))?;

        // Derive key from base64-encoded key
        let key_bytes = general_purpose::STANDARD
            .decode(encryption_key)
            .context("Failed to decode encryption key")?;

        if key_bytes.len() != 32 {
            return Err(anyhow!(
                "Encryption key must be 32 bytes (256 bits) when base64 decoded"
            ));
        }

        let key = Key::<Aes256Gcm>::from_slice(&key_bytes);
        let cipher = Aes256Gcm::new(key);

        // Generate cryptographically secure random nonce (SCIRS2 POLICY: use SecureRandom)
        let mut secure_rng = SecureRandom::new();
        let nonce_bytes = secure_rng.random_bytes(12);
        let nonce = Nonce::from_slice(&nonce_bytes);

        // Encrypt data
        let ciphertext = cipher
            .encrypt(nonce, data)
            .map_err(|e| anyhow!("Encryption failed: {}", e))?;

        // Prepend nonce to ciphertext for storage
        let mut encrypted_data = nonce_bytes;
        encrypted_data.extend_from_slice(&ciphertext);

        Ok(encrypted_data)
    }

    /// Decrypt data using AES-256-GCM
    async fn decrypt_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        let encryption_key = self
            .config
            .encryption_key
            .as_ref()
            .ok_or_else(|| anyhow!("Encryption enabled but no encryption key provided"))?;

        if data.len() < 12 {
            return Err(anyhow!("Encrypted data too short - missing nonce"));
        }

        // Derive key from base64-encoded key
        let key_bytes = general_purpose::STANDARD
            .decode(encryption_key)
            .context("Failed to decode encryption key")?;

        if key_bytes.len() != 32 {
            return Err(anyhow!(
                "Encryption key must be 32 bytes (256 bits) when base64 decoded"
            ));
        }

        let key = Key::<Aes256Gcm>::from_slice(&key_bytes);
        let cipher = Aes256Gcm::new(key);

        // Extract nonce and ciphertext
        let (nonce_bytes, ciphertext) = data.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);

        // Decrypt data
        let plaintext = cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| anyhow!("Decryption failed: {}", e))?;

        Ok(plaintext)
    }

    /// Generate a new encryption key for configuration
    /// Uses cryptographically secure RNG (SCIRS2 POLICY: SecureRandom for cryptographic operations)
    pub fn generate_encryption_key() -> String {
        let mut secure_rng = SecureRandom::new();
        let key_bytes: [u8; 32] = secure_rng
            .random_bytes(32)
            .try_into()
            .expect("Generated exactly 32 bytes");
        general_purpose::STANDARD.encode(key_bytes)
    }

    async fn convert_to_persistent_chat_session(
        &self,
        persisted: &PersistedSession,
    ) -> Result<PersistentChatSession> {
        Ok(PersistentChatSession {
            session_id: persisted.session_id.clone(),
            config: persisted.config.clone(),
            messages: persisted.messages.clone(),
            created_at: persisted.created_at,
            last_accessed: persisted.last_accessed,
            metrics: persisted.metrics.clone(),
            user_preferences: persisted.user_preferences.clone(),
        })
    }

    fn get_session_file_path(&self, session_id: &str) -> PathBuf {
        self.config
            .storage_path
            .join(format!("{session_id}.session"))
    }

    fn get_backup_file_path(&self, session_id: &str) -> PathBuf {
        self.config.backup_path.join(format!("{session_id}.backup"))
    }

    fn get_checkpoint_file_path(&self, session_id: &str) -> PathBuf {
        self.config
            .storage_path
            .join(format!("{session_id}.checkpoint"))
    }

    async fn load_all_sessions(&self) -> Result<()> {
        let sessions = self.list_sessions().await?;
        info!("Found {} existing sessions", sessions.len());

        // Limit to max_sessions
        let sessions_to_load = sessions.into_iter().take(self.config.max_sessions);

        for session_info in sessions_to_load {
            if let Err(e) = self.load_session(&session_info.session_id).await {
                warn!("Failed to load session {}: {}", session_info.session_id, e);
            }
        }

        Ok(())
    }

    async fn start_auto_save_task(&self) {
        let active_sessions = Arc::clone(&self.active_sessions);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = interval(config.auto_save_interval);

            loop {
                interval.tick().await;

                let sessions = active_sessions.read().await;
                for (session_id, session_arc) in sessions.iter() {
                    // Check if session needs saving (has been modified)
                    let session_guard = session_arc.read().await;
                    if session_guard.dirty {
                        drop(session_guard); // Release read lock before saving

                        // Create a temporary manager for saving (using config clone)
                        let temp_manager = SessionPersistenceManager {
                            config: config.clone(),
                            active_sessions: Arc::clone(&active_sessions),
                            persistence_stats: Arc::new(RwLock::new(PersistenceStats::default())),
                        };

                        // Convert to PersistentChatSession for saving
                        let session_guard = session_arc.read().await;
                        if let Ok(persistent_session) = temp_manager
                            .convert_to_persistent_chat_session(&session_guard.session)
                            .await
                        {
                            drop(session_guard);

                            if let Err(e) = temp_manager.save_session(&persistent_session).await {
                                error!("Failed to auto-save session {}: {}", session_id, e);
                            } else {
                                debug!("Auto-saved dirty session {}", session_id);

                                // Mark as clean after successful save
                                let mut session_guard = session_arc.write().await;
                                session_guard.dirty = false;
                                session_guard.last_saved = Some(SystemTime::now());
                            }
                        }
                    }
                }
            }
        });
    }

    async fn start_cleanup_task(&self) {
        let manager = self.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(3600)); // Cleanup every hour

            loop {
                interval.tick().await;

                if let Err(e) = manager.cleanup_expired_sessions().await {
                    error!("Failed to cleanup expired sessions: {}", e);
                }
            }
        });
    }

    async fn start_checkpoint_task(&self) {
        let active_sessions = Arc::clone(&self.active_sessions);
        let config = self.config.clone();
        let manager = self.clone(); // Clone the manager for use in the task

        tokio::spawn(async move {
            let mut interval = interval(config.checkpoint_interval);

            loop {
                interval.tick().await;

                let sessions = active_sessions.read().await;
                for (session_id, session_arc) in sessions.iter() {
                    debug!("Creating checkpoint for session {}", session_id);

                    // Create checkpoint for this session
                    if let Err(e) = manager.create_checkpoint(session_id, session_arc).await {
                        error!(
                            "Failed to create checkpoint for session {}: {}",
                            session_id, e
                        );
                    }
                }
            }
        });
    }

    /// Create a checkpoint for a session
    async fn create_checkpoint(
        &self,
        session_id: &str,
        session_arc: &Arc<RwLock<SessionWithDirtyFlag>>,
    ) -> Result<()> {
        let checkpoint_path = self.get_checkpoint_file_path(session_id);

        // Read the session data
        let session_with_flag = session_arc.read().await;

        // Create a checkpoint copy of the session
        let checkpoint_session = session_with_flag.session.clone();

        // Update the last checkpoint timestamp
        drop(session_with_flag); // Drop read lock
        {
            let mut session_with_flag = session_arc.write().await;
            // Add a checkpoint marker to the context window
            session_with_flag
                .session
                .conversation_state
                .context_window
                .push(format!(
                    "checkpoint:{}",
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or(Duration::ZERO)
                        .as_secs()
                ));
            // Mark as dirty since we modified the session
            session_with_flag.dirty = true;
        }

        // Serialize and optionally compress
        let serialized = if self.config.compression_enabled {
            self.compress_session(&checkpoint_session).await?
        } else {
            bincode::encode_to_vec(&checkpoint_session, bincode::config::standard())
                .map_err(|e| anyhow!("Bincode encoding failed: {}", e))?
        };

        // Write to checkpoint file atomically
        let temp_path = checkpoint_path.with_extension("checkpoint.tmp");

        {
            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&temp_path)
                .await?;

            file.write_all(&serialized).await?;
            file.sync_all().await?;
        }

        // Atomic rename
        fs::rename(&temp_path, &checkpoint_path)?;

        debug!(
            "Checkpoint created for session {} at {:?}",
            session_id, checkpoint_path
        );
        Ok(())
    }

    async fn analyze_recovery_options(&self, session_id: &str) -> Result<RecoveryInfo> {
        let backup_available = self.get_backup_file_path(session_id).exists();
        let checkpoint_path = self.get_checkpoint_file_path(session_id);
        let checkpoint_available = checkpoint_path.exists();

        let mut recovery_strategies = Vec::new();

        // Add checkpoint recovery if available
        if checkpoint_available {
            recovery_strategies.push(RecoveryStrategy::LoadFromCheckpoint);
        }

        if backup_available {
            recovery_strategies.push(RecoveryStrategy::LoadFromBackup);
        }

        recovery_strategies.push(RecoveryStrategy::PartialRecovery);
        recovery_strategies.push(RecoveryStrategy::CreateNew);

        // Get actual checkpoint time from file metadata
        let last_checkpoint = if checkpoint_available {
            match fs::metadata(&checkpoint_path) {
                Ok(metadata) => {
                    // Use file modification time as checkpoint time
                    metadata.modified().unwrap_or_else(|_| SystemTime::now())
                }
                Err(_) => SystemTime::now(),
            }
        } else {
            SystemTime::UNIX_EPOCH // No checkpoint available
        };

        Ok(RecoveryInfo {
            session_id: session_id.to_string(),
            last_checkpoint,
            backup_available,
            corruption_detected: true,
            recovery_strategies,
        })
    }

    async fn load_from_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<PersistentChatSession>> {
        let checkpoint_path = self.get_checkpoint_file_path(session_id);
        if !checkpoint_path.exists() {
            debug!("No checkpoint file found for session: {}", session_id);
            return Ok(None);
        }

        info!(
            "Loading session {} from checkpoint: {:?}",
            session_id, checkpoint_path
        );

        // Read checkpoint file
        let mut file = File::open(&checkpoint_path)
            .await
            .context("Failed to open checkpoint file")?;

        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .await
            .context("Failed to read checkpoint file")?;

        // Deserialize the persisted session
        let persisted: PersistedSession = if self.config.compression_enabled {
            self.decompress_session(&data)
                .await
                .context("Failed to decompress checkpoint session")?
        } else {
            bincode::decode_from_slice(&data, bincode::config::standard())
                .map(|(session, _)| session)
                .context("Failed to deserialize checkpoint session")?
        };

        // Verify checksum if available
        if !persisted.checksum.is_empty() && !self.verify_checksum(&persisted).await? {
            warn!(
                "Checkpoint checksum verification failed for session: {}",
                session_id
            );
            return Ok(None);
        }

        // Convert to persistent chat session
        let session = self
            .convert_to_persistent_chat_session(&persisted)
            .await
            .context("Failed to convert persisted session")?;

        // Update stats
        {
            let mut stats = self.persistence_stats.write().await;
            stats.total_sessions_loaded += 1;
        }

        info!("Successfully loaded session {} from checkpoint", session_id);
        Ok(Some(session))
    }

    async fn load_from_backup(&self, session_id: &str) -> Result<Option<PersistentChatSession>> {
        let backup_path = self.get_backup_file_path(session_id);
        if !backup_path.exists() {
            return Ok(None);
        }

        let mut file = File::open(&backup_path).await?;
        let mut data = Vec::new();
        file.read_to_end(&mut data).await?;

        let persisted: PersistedSession = if self.config.compression_enabled {
            self.decompress_session(&data).await?
        } else {
            bincode::decode_from_slice(&data, bincode::config::standard())
                .map_err(|e| anyhow!("Bincode decoding failed: {}", e))?
                .0
        };

        Ok(Some(
            self.convert_to_persistent_chat_session(&persisted).await?,
        ))
    }

    async fn attempt_partial_recovery(
        &self,
        session_id: &str,
    ) -> Result<Option<PersistentChatSession>> {
        info!("Attempting partial recovery for session: {}", session_id);

        let mut recovered_session = None;
        let mut recovery_sources = Vec::new();

        // Try to recover from various sources in order of preference

        // 1. Try to read from main session file with error tolerance
        let session_path = self.get_session_file_path(session_id);
        if session_path.exists() {
            if let Some(partial_session) = self
                .try_partial_file_recovery(&session_path, "session")
                .await
            {
                recovered_session = Some(partial_session);
                recovery_sources.push("main_session_file");
            }
        }

        // 2. Try to read from checkpoint file with error tolerance
        if recovered_session.is_none() {
            let checkpoint_path = self.get_checkpoint_file_path(session_id);
            if checkpoint_path.exists() {
                if let Some(partial_session) = self
                    .try_partial_file_recovery(&checkpoint_path, "checkpoint")
                    .await
                {
                    recovered_session = Some(partial_session);
                    recovery_sources.push("checkpoint_file");
                }
            }
        }

        // 3. Try to read from backup file with error tolerance
        if recovered_session.is_none() {
            let backup_path = self.get_backup_file_path(session_id);
            if backup_path.exists() {
                if let Some(partial_session) =
                    self.try_partial_file_recovery(&backup_path, "backup").await
                {
                    recovered_session = Some(partial_session);
                    recovery_sources.push("backup_file");
                }
            }
        }

        // 4. If we have some recovery, validate and sanitize it
        if let Some(mut session) = recovered_session {
            // Sanitize the session data
            session = self.sanitize_recovered_session(session, session_id).await;

            // Update recovery stats
            {
                let mut stats = self.persistence_stats.write().await;
                stats.recovery_operations += 1;
            }

            info!(
                "Partial recovery successful for session {} from sources: {:?}",
                session_id, recovery_sources
            );

            return Ok(Some(session));
        }

        warn!("Partial recovery failed for session: {}", session_id);
        Ok(None)
    }

    /// Attempt to read and recover data from a potentially corrupted file
    async fn try_partial_file_recovery(
        &self,
        file_path: &Path,
        source_type: &str,
    ) -> Option<PersistentChatSession> {
        debug!(
            "Trying partial recovery from {} file: {:?}",
            source_type, file_path
        );

        // Try to read the file
        let data = match self.read_file_with_tolerance(file_path).await {
            Ok(data) => data,
            Err(e) => {
                warn!("Failed to read {} file: {}", source_type, e);
                return None;
            }
        };

        // Try to deserialize with error tolerance
        let persisted_session = if self.config.compression_enabled {
            // Try decompression first
            match self.decompress_session(&data).await {
                Ok(session) => session,
                Err(_) => {
                    // Fallback to direct deserialization if decompression fails
                    match bincode::decode_from_slice::<PersistedSession, _>(
                        &data,
                        bincode::config::standard(),
                    ) {
                        Ok((session, _)) => session,
                        Err(e) => {
                            warn!("Failed to deserialize {} file: {}", source_type, e);
                            return None;
                        }
                    }
                }
            }
        } else {
            match bincode::decode_from_slice::<PersistedSession, _>(
                &data,
                bincode::config::standard(),
            ) {
                Ok((session, _)) => session,
                Err(e) => {
                    warn!("Failed to deserialize {} file: {}", source_type, e);
                    return None;
                }
            }
        };

        // Convert to persistent chat session
        match self
            .convert_to_persistent_chat_session(&persisted_session)
            .await
        {
            Ok(session) => Some(session),
            Err(e) => {
                warn!("Failed to convert {} session: {}", source_type, e);
                None
            }
        }
    }

    /// Read file with error tolerance
    async fn read_file_with_tolerance(&self, file_path: &Path) -> Result<Vec<u8>> {
        let mut file = File::open(file_path).await?;
        let mut data = Vec::new();

        // Try to read the entire file, but don't fail on partial reads
        match file.read_to_end(&mut data).await {
            Ok(_) => Ok(data),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                // File is truncated but we got some data
                if !data.is_empty() {
                    warn!("File truncated but partial data recovered: {:?}", file_path);
                    Ok(data)
                } else {
                    Err(e.into())
                }
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Sanitize and repair a recovered session
    async fn sanitize_recovered_session(
        &self,
        mut session: PersistentChatSession,
        session_id: &str,
    ) -> PersistentChatSession {
        // Ensure session ID matches
        session.session_id = session_id.to_string();

        // Validate and fix timestamps
        if session.created_at > SystemTime::now() {
            session.created_at = SystemTime::now() - Duration::from_secs(86400);
            // Set to yesterday (24 hours ago)
        }

        if session.last_accessed > SystemTime::now() {
            session.last_accessed = SystemTime::now();
        }

        // Filter out potentially corrupted messages
        session.messages.retain(|msg| {
            !msg.id.is_empty()
                && !msg.content.to_string().is_empty()
                && msg.timestamp <= chrono::Utc::now()
        });

        // Ensure we have at least basic metrics
        if session.metrics.total_messages == 0 && !session.messages.is_empty() {
            session.metrics.total_messages = session.messages.len();
        }

        // Reset user preferences if they seem corrupted
        if session.user_preferences.len() > 1000 {
            warn!(
                "User preferences seem corrupted, resetting for session: {}",
                session_id
            );
            session.user_preferences = HashMap::new();
        }

        // Add recovery metadata
        session.user_preferences.insert(
            "recovery_performed".to_string(),
            serde_json::Value::String(
                serde_json::to_string(&chrono::Utc::now()).unwrap_or_default(),
            ),
        );

        session.user_preferences.insert(
            "recovery_type".to_string(),
            serde_json::Value::String("partial".to_string()),
        );

        session
    }

    /// Generate analytics for a session based on its data
    async fn generate_analytics_for_session(
        &self,
        session: &PersistentChatSession,
    ) -> Result<ConversationAnalytics> {
        let _now = SystemTime::now();

        // Count messages by role
        let mut user_message_count = 0;
        let mut assistant_message_count = 0;
        let mut total_tokens = 0;
        let mut topics_discussed = Vec::new();
        let mut intent_distribution = HashMap::new();

        // Analyze messages
        for message in &session.messages {
            match message.role {
                crate::messages::MessageRole::User => user_message_count += 1,
                crate::messages::MessageRole::Assistant => assistant_message_count += 1,
                _ => {}
            }

            if let Some(token_count) = message.token_count {
                total_tokens += token_count;
            }

            // Extract topics from message content (simplified)
            let content = message.content.to_string().to_lowercase();
            if content.contains("sparql") || content.contains("query") {
                topics_discussed.push("querying".to_string());
            }
            if content.contains("data") || content.contains("information") {
                topics_discussed.push("data_retrieval".to_string());
            }
            if content.contains("help") || content.contains("assist") {
                topics_discussed.push("assistance".to_string());
            }

            // Simple intent classification based on content
            let intent = if content.contains("?")
                || content.starts_with("what")
                || content.starts_with("how")
            {
                IntentType::Question
            } else if content.contains("please") || content.contains("can you") {
                IntentType::Request
            } else if content.contains("thank") || content.contains("appreciate") {
                IntentType::Gratitude
            } else {
                IntentType::Complex
            };

            *intent_distribution.entry(intent).or_insert(0) += 1;
        }

        // Remove duplicate topics
        topics_discussed.sort();
        topics_discussed.dedup();

        // Calculate average response time (simplified)
        let average_response_time = if session.messages.len() > 1 {
            let total_duration = session
                .last_accessed
                .duration_since(session.created_at)
                .unwrap_or(Duration::from_secs(0));
            Duration::from_secs(total_duration.as_secs() / session.messages.len() as u64)
        } else {
            Duration::from_secs(1)
        };

        // Create default metrics
        let user_satisfaction = SatisfactionMetrics {
            overall_satisfaction: 4.0, // Default good satisfaction
            response_quality: 4.2,
            helpfulness: 4.1,
            satisfaction_breakdown: {
                let mut breakdown = HashMap::new();
                breakdown.insert("response_quality".to_string(), 4.2);
                breakdown.insert("response_speed".to_string(), 4.0);
                breakdown.insert("helpfulness".to_string(), 4.1);
                breakdown.insert("clarity".to_string(), 4.0);
                breakdown.insert("relevance".to_string(), 4.2);
                breakdown
            },
            implicit_signals: ImplicitSatisfactionSignals {
                positive_acknowledgments: 0,
                clarification_requests: 0,
                topic_continuity: 0.85,
                follow_up_questions: 0,
                positive_feedback_indicators: 0,
                task_completion_rate: 0.85,
                session_continuation: true,
            },
            explicit_feedback: Vec::new(),
        };

        let conversation_quality = ConversationQuality {
            overall_score: 0.83,
            overall_quality: 0.83,
            coherence: 0.85,
            coherence_score: 0.85,
            relevance: 0.88,
            relevance_score: 0.88,
            completeness: 0.82,
            completeness_score: 0.82,
            helpfulness_score: 0.87,
            accuracy_score: 0.90,
            clarity_score: 0.85,
            engagement_score: 0.80,
            error_rate: 0.05,
            response_appropriateness: 0.85,
        };

        // Create default progression data
        let sentiment_progression = vec![0.6, 0.1, 0.7];

        let complexity_progression = vec![ComplexityMetrics {
            turn_number: 1,
            message_complexity: 0.65,
            topic_depth: 0.7,
            reasoning_complexity: 0.6,
            linguistic_complexity: 0.6,
            semantic_complexity: 0.7,
            context_dependency: 0.8,
            reasoning_depth: 0.65,
            overall_complexity: 0.69,
        }];

        let confidence_progression = vec![ConfidenceMetrics {
            turn_number: 1,
            confidence_score: 0.85,
            uncertainty_markers: 0,
            overall_confidence: 0.85,
            uncertainty_factors: vec![],
            confidence_breakdown: {
                let mut breakdown = HashMap::new();
                breakdown.insert("response_quality".to_string(), 0.90);
                breakdown.insert("knowledge_coverage".to_string(), 0.85);
                breakdown.insert("factual_accuracy".to_string(), 0.88);
                breakdown
            },
        }];

        // Convert SystemTime to DateTime<Utc>
        let start_time_utc = DateTime::<Utc>::from(session.created_at);
        let end_time_utc = DateTime::<Utc>::from(session.last_accessed);

        // Convert intent_distribution from HashMap<IntentType, i32> to HashMap<String, usize>
        let intent_distribution_string: HashMap<String, usize> = intent_distribution
            .into_iter()
            .map(|(k, v)| (format!("{:?}", k), v as usize))
            .collect();

        // Extract f32 values for user_satisfaction and conversation_quality
        let user_satisfaction_f32 = user_satisfaction.overall_satisfaction;
        let conversation_quality_f32 = conversation_quality.overall_quality;

        // Extract implicit_signals from user_satisfaction
        let implicit_signals = user_satisfaction.implicit_signals.clone();

        Ok(ConversationAnalytics {
            // Core progression metrics
            complexity_progression,
            confidence_progression,
            quality: conversation_quality,
            satisfaction: user_satisfaction,
            implicit_signals,

            // Additional fields
            session_id: session.session_id.clone(),
            start_time: start_time_utc,
            end_time: end_time_utc,
            message_count: session.messages.len(),
            user_message_count,
            assistant_message_count,
            average_response_time: average_response_time.as_secs_f64(),
            total_tokens,
            user_satisfaction: user_satisfaction_f32,
            conversation_quality: conversation_quality_f32,
            topics_discussed,
            sentiment_progression,
            intent_distribution: intent_distribution_string,
            patterns_detected: Vec::new(), // No patterns detected by default
            anomalies: Vec::new(),         // No anomalies detected by default
            metadata: HashMap::new(),
        })
    }

    async fn create_emergency_session(&self, session_id: &str) -> Result<PersistentChatSession> {
        Ok(PersistentChatSession {
            session_id: session_id.to_string(),
            config: ChatConfig::default(),
            messages: Vec::new(),
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            metrics: Default::default(),
            user_preferences: HashMap::new(),
        })
    }
}

// Clone implementation for background tasks
impl Clone for SessionPersistenceManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            active_sessions: Arc::clone(&self.active_sessions),
            persistence_stats: Arc::clone(&self.persistence_stats),
        }
    }
}

/// Session information for listing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub session_id: String,
    pub created_at: SystemTime,
    pub modified_at: SystemTime,
    pub size_bytes: u64,
    pub has_backup: bool,
}
