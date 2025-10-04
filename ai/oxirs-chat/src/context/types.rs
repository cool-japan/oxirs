//! Context Management Types and Structures

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};
use crate::Message;
use super::config::ContextConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PinReason {
    HighImportance,
    UserRequest,
    KeyInformation,
    ContextAnchor,
    Reference,
}

pub struct ContextWindow {
    pub config: ContextConfig,
    pub messages: VecDeque<ContextMessage>,
    pub pinned_messages: HashMap<String, PinnedMessage>,
    pub summary: Option<ContextSummary>,
    pub total_token_count: usize,
}

#[derive(Debug, Clone)]
pub struct ContextMessage {
    pub message: Message,
    pub importance_score: f32,
    pub added_at: SystemTime,
    pub last_accessed: SystemTime,
    pub access_count: usize,
}

#[derive(Debug, Clone)]
pub struct PinnedMessage {
    pub message_id: String,
    pub reason: PinReason,
    pub pinned_at: SystemTime,
    pub importance_score: f32,
}

pub struct AssembledContext {
    pub context_text: String,
    pub effective_messages: Vec<Message>,
    pub current_topics: Vec<Topic>,
    pub context_summary: Option<String>,
    pub quality_score: f32,
    pub coverage_score: f32,
    pub token_count: usize,
    pub structured_context: StructuredContext,
}

#[derive(Debug, Clone)]
pub struct StructuredContext {
    pub entities: Vec<String>,
    pub facts: Vec<String>,
    pub queries: Vec<String>,
    pub relationships: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Topic {
    pub name: String,
    pub confidence: f32,
    pub first_mentioned: SystemTime,
    pub last_mentioned: SystemTime,
    pub mention_count: usize,
}

#[derive(Debug, Clone)]
pub struct ContextUpdate {
    pub message_processed: String,
    pub importance_score: f32,
    pub window_update: WindowUpdate,
    pub topic_update: Option<TopicUpdate>,
    pub summarization_update: Option<SummarizationUpdate>,
    pub optimization_update: Option<OptimizationUpdate>,
    pub processing_time: Duration,
}

#[derive(Debug, Clone)]
pub struct WindowUpdate {
    pub message_added: bool,
    pub evicted_messages: Vec<Message>,
    pub current_size: usize,
    pub token_count: usize,
}

#[derive(Debug, Clone)]
pub struct TopicUpdate {
    pub new_topics: Vec<Topic>,
    pub topic_changes: Vec<TopicChange>,
    pub drift_detected: bool,
}

#[derive(Debug, Clone)]
pub struct TopicChange {
    pub topic_name: String,
    pub change_type: TopicChangeType,
    pub confidence_delta: f32,
}

#[derive(Debug, Clone)]
pub enum TopicChangeType {
    Introduced,
    Strengthened,
    Weakened,
    Abandoned,
}

#[derive(Debug, Clone)]
pub struct SummarizationUpdate {
    pub summary: ContextSummary,
    pub messages_summarized: usize,
    pub compression_ratio: f32,
}

#[derive(Debug, Clone)]
pub struct ContextSummary {
    pub text: String,
    pub key_points: Vec<String>,
    pub entities_mentioned: Vec<String>,
    pub topics_covered: Vec<String>,
    pub created_at: SystemTime,
}

#[derive(Debug, Clone)]
pub struct OptimizationUpdate {
    pub memory_saved: usize,
    pub operations_performed: Vec<OptimizationOperation>,
    pub efficiency_improvement: f32,
}

#[derive(Debug, Clone)]
pub enum OptimizationOperation {
    MessageDeduplication,
    ImportanceRescoring,
    TokenCompression,
    RedundancyRemoval,
}

#[derive(Debug, Clone)]
pub struct ContextSwitch {
    pub previous_state: ContextState,
    pub new_topic: String,
    pub topic_transition: TopicTransition,
    pub window_adjustment: WindowAdjustment,
    pub context_preserved: bool,
}

#[derive(Debug, Clone)]
pub struct ContextState {
    pub message_count: usize,
    pub pinned_count: usize,
    pub token_count: usize,
    pub has_summary: bool,
    pub current_topic: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TopicTransition {
    pub from_topic: Option<String>,
    pub to_topic: String,
    pub transition_reason: String,
    pub confidence: f32,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone)]
pub struct WindowAdjustment {
    pub messages_reordered: bool,
    pub importance_rescored: bool,
    pub window_size_adjusted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextStats {
    pub total_messages: usize,
    pub active_messages: usize,
    pub pinned_messages: usize,
    pub current_topics: usize,
    pub summarization_count: usize,
    pub memory_optimizations: usize,
    pub average_importance_score: f32,
    pub context_efficiency: f32,
}
