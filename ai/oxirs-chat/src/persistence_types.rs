//! Session persistence types, configuration, and data models.
//!
//! Sibling module of [`crate::persistence`]. Contains the value-object
//! definitions used by the storage backend: configuration, persisted
//! session payloads, conversation state, entity tracking, recovery
//! descriptors, and persistence statistics.

use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::PathBuf,
    time::{Duration, SystemTime},
};

use crate::{analytics::ConversationAnalytics, ChatConfig, Message, SessionMetrics};

/// In-memory chat session used by the persistence layer. This is a
/// reduced view of the full `ChatSession`, suitable for serialising
/// to disk and re-hydrating later.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentChatSession {
    pub session_id: String,
    pub config: ChatConfig,
    pub messages: Vec<Message>,
    pub created_at: SystemTime,
    pub last_accessed: SystemTime,
    pub metrics: SessionMetrics,
    pub user_preferences: HashMap<String, serde_json::Value>,
}

/// Configuration for session persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    pub enabled: bool,
    pub storage_path: PathBuf,
    pub backup_path: PathBuf,
    pub auto_save_interval: Duration,
    pub session_ttl: Duration,
    pub max_sessions: usize,
    pub compression_enabled: bool,
    pub encryption_enabled: bool,
    /// Base64-encoded encryption key. Required when
    /// [`Self::encryption_enabled`] is `true`.
    pub encryption_key: Option<String>,
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

/// Serializable session data for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedSession {
    pub session_id: String,
    pub config: ChatConfig,
    pub messages: Vec<Message>,
    pub created_at: SystemTime,
    pub last_accessed: SystemTime,
    pub metrics: SessionMetrics,
    pub analytics: Option<ConversationAnalytics>,
    pub user_preferences: HashMap<String, serde_json::Value>,
    pub conversation_state: ConversationState,
    pub checksum: String,
}

/// Wrapper for persisted session with dirty flag tracking.
#[derive(Debug, Clone)]
pub struct SessionWithDirtyFlag {
    pub session: PersistedSession,
    pub dirty: bool,
    pub last_saved: Option<SystemTime>,
}

/// Conversation state for advanced context management.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConversationState {
    pub current_topic: Option<String>,
    /// Message IDs in current context.
    pub context_window: Vec<String>,
    pub entity_history: Vec<EntityReference>,
    pub query_history: Vec<QueryContext>,
    pub user_intent_history: Vec<String>,
    pub conversation_flow: ConversationFlow,
}

/// Entity reference for tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityReference {
    pub entity_uri: String,
    pub entity_label: String,
    pub first_mentioned: SystemTime,
    pub mention_count: usize,
    pub last_context: String,
}

/// Query context for tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryContext {
    pub sparql_query: String,
    pub natural_language: String,
    pub intent: String,
    pub success: bool,
    pub timestamp: SystemTime,
    pub execution_time_ms: u64,
}

/// Conversation flow tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Conversation phases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConversationPhase {
    Introduction,
    Exploration,
    DeepDive,
    Analysis,
    Conclusion,
    Abandoned,
}

/// Topic transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicTransition {
    pub from_topic: Option<String>,
    pub to_topic: String,
    pub transition_type: TransitionType,
    pub timestamp: SystemTime,
    pub confidence: f32,
}

/// Transition types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransitionType {
    /// Logical progression.
    Natural,
    /// User changed topic.
    UserInitiated,
    /// Seeking clarification.
    Clarification,
    /// Side topic.
    Digression,
    /// Back to previous topic.
    Return,
}

/// Interaction patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionPattern {
    pub pattern_type: InteractionType,
    pub frequency: usize,
    pub last_occurrence: SystemTime,
    pub effectiveness: f32,
}

/// Interaction types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    QuestionAnswer,
    ExploratorySearch,
    ComparativeAnalysis,
    DeepInvestigation,
    QuickLookup,
    IterativeRefinement,
}

/// Session recovery information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryInfo {
    pub session_id: String,
    pub last_checkpoint: SystemTime,
    pub backup_available: bool,
    pub corruption_detected: bool,
    pub recovery_strategies: Vec<RecoveryStrategy>,
}

/// Recovery strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    LoadFromCheckpoint,
    LoadFromBackup,
    PartialRecovery,
    CreateNew,
}

/// Persistence statistics.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
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

/// Session information for listing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub session_id: String,
    pub created_at: SystemTime,
    pub modified_at: SystemTime,
    pub size_bytes: u64,
    pub has_backup: bool,
}
