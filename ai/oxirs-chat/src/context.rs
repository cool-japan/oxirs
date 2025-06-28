//! Advanced Context Management for OxiRS Chat
//!
//! Implements intelligent context management with sliding windows, topic tracking,
//! context summarization, and adaptive memory optimization.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::{Duration, SystemTime},
};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::{
    Message, MessageRole, MessageMetadata,
    rag::QueryIntent,
    analytics::ConversationAnalytics,
};

/// Configuration for context management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    pub sliding_window_size: usize,
    pub max_context_length: usize,
    pub enable_summarization: bool,
    pub summarization_threshold: usize,
    pub enable_topic_tracking: bool,
    pub topic_drift_threshold: f32,
    pub enable_importance_scoring: bool,
    pub memory_optimization_enabled: bool,
    pub adaptive_window_size: bool,
    pub context_compression_ratio: f32,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            sliding_window_size: 20,
            max_context_length: 4096,
            enable_summarization: true,
            summarization_threshold: 40,
            enable_topic_tracking: true,
            topic_drift_threshold: 0.7,
            enable_importance_scoring: true,
            memory_optimization_enabled: true,
            adaptive_window_size: true,
            context_compression_ratio: 0.6,
        }
    }
}

/// Advanced context manager
pub struct AdvancedContextManager {
    config: ContextConfig,
    context_window: ContextWindow,
    topic_tracker: TopicTracker,
    importance_scorer: ImportanceScorer,
    summarization_engine: SummarizationEngine,
    memory_optimizer: MemoryOptimizer,
}

impl AdvancedContextManager {
    pub fn new(config: ContextConfig) -> Self {
        Self {
            context_window: ContextWindow::new(&config),
            topic_tracker: TopicTracker::new(&config),
            importance_scorer: ImportanceScorer::new(&config),
            summarization_engine: SummarizationEngine::new(&config),
            memory_optimizer: MemoryOptimizer::new(&config),
            config,
        }
    }

    /// Process a new message and update context
    pub async fn process_message(
        &mut self,
        message: &Message,
        conversation_analytics: Option<&ConversationAnalytics>,
    ) -> Result<ContextUpdate> {
        let start_time = SystemTime::now();

        // Calculate importance score
        let importance_score = self.importance_scorer.score_message(message, conversation_analytics).await?;

        // Update context window
        let window_update = self.context_window.add_message(message.clone(), importance_score).await?;

        // Track topic changes
        let topic_update = if self.config.enable_topic_tracking {
            Some(self.topic_tracker.process_message(message).await?)
        } else {
            None
        };

        // Check if summarization is needed
        let summarization_update = if self.config.enable_summarization 
            && self.context_window.should_summarize().await {
            Some(self.perform_summarization().await?)
        } else {
            None
        };

        // Optimize memory if needed
        let optimization_update = if self.config.memory_optimization_enabled {
            Some(self.memory_optimizer.optimize_context(&mut self.context_window).await?)
        } else {
            None
        };

        let processing_time = start_time.elapsed().unwrap_or(Duration::ZERO);

        Ok(ContextUpdate {
            message_processed: message.id.clone(),
            importance_score,
            window_update,
            topic_update,
            summarization_update,
            optimization_update,
            processing_time,
        })
    }

    /// Get current context for LLM
    pub async fn get_current_context(&self) -> Result<AssembledContext> {
        let effective_messages = self.context_window.get_effective_messages().await?;
        let current_topics = self.topic_tracker.get_current_topics().await;
        let context_summary = self.context_window.get_summary().await;

        // Assemble context with proper ordering and formatting
        let mut context_text = String::new();

        // Add summary if available
        if let Some(summary) = &context_summary {
            context_text.push_str("## Conversation Summary\n");
            context_text.push_str(summary);
            context_text.push_str("\n\n");
        }

        // Add current topics
        if !current_topics.is_empty() {
            context_text.push_str("## Current Topics\n");
            for topic in &current_topics {
                context_text.push_str(&format!("- {} (confidence: {:.2})\n", topic.name, topic.confidence));
            }
            context_text.push_str("\n");
        }

        // Add recent messages
        context_text.push_str("## Recent Messages\n");
        for message in &effective_messages {
            let role_indicator = match message.role {
                MessageRole::User => "User",
                MessageRole::Assistant => "Assistant",
                MessageRole::System => "System",
            };
            context_text.push_str(&format!("{}: {}\n", role_indicator, message.content));
        }

        // Calculate quality metrics
        let quality_score = self.calculate_context_quality(&effective_messages, &current_topics).await;
        let coverage_score = self.calculate_coverage_score(&effective_messages).await;

        // Calculate values before moving into the struct
        let token_count = self.estimate_token_count(&context_text).await;
        let structured_context = self.extract_structured_context(&effective_messages).await?;
        
        Ok(AssembledContext {
            context_text,
            effective_messages,
            current_topics,
            context_summary,
            quality_score,
            coverage_score,
            token_count,
            structured_context,
        })
    }

    /// Handle context switching
    pub async fn switch_context(&mut self, new_topic: &str, context_hint: Option<&str>) -> Result<ContextSwitch> {
        info!("Switching context to topic: {}", new_topic);

        // Save current context state
        let previous_state = self.context_window.get_state_snapshot().await;

        // Perform topic transition
        let topic_transition = self.topic_tracker.transition_to_topic(new_topic, context_hint).await?;

        // Adjust context window for new topic
        let window_adjustment = self.context_window.adjust_for_topic(&topic_transition).await?;

        // Update importance scoring for new context
        self.importance_scorer.update_for_context_switch(&topic_transition).await?;

        Ok(ContextSwitch {
            previous_state,
            new_topic: new_topic.to_string(),
            topic_transition,
            window_adjustment,
            context_preserved: true, // TODO: Implement actual preservation logic
        })
    }

    /// Pin an important message
    pub async fn pin_message(&mut self, message_id: &str, reason: PinReason) -> Result<()> {
        self.context_window.pin_message(message_id, reason).await
    }

    /// Unpin a message
    pub async fn unpin_message(&mut self, message_id: &str) -> Result<()> {
        self.context_window.unpin_message(message_id).await
    }

    /// Get context statistics
    pub async fn get_context_stats(&self) -> ContextStats {
        ContextStats {
            total_messages: self.context_window.total_messages().await,
            active_messages: self.context_window.active_messages().await,
            pinned_messages: self.context_window.pinned_count().await,
            current_topics: self.topic_tracker.topic_count().await,
            summarization_count: self.summarization_engine.summarization_count().await,
            memory_optimizations: self.memory_optimizer.optimization_count().await,
            average_importance_score: self.importance_scorer.average_score().await,
            context_efficiency: self.calculate_context_efficiency().await,
        }
    }

    // Private helper methods

    async fn perform_summarization(&mut self) -> Result<SummarizationUpdate> {
        let messages_to_summarize = self.context_window.get_messages_for_summarization().await?;
        let summary = self.summarization_engine.summarize_messages(&messages_to_summarize).await?;
        
        let summary_text = summary.text.clone();
        let summary_clone = summary.clone();
        self.context_window.apply_summarization(summary).await?;
        
        Ok(SummarizationUpdate {
            summary: summary_clone,
            messages_summarized: messages_to_summarize.len(),
            compression_ratio: self.calculate_compression_ratio(&messages_to_summarize, &summary_text).await,
        })
    }

    async fn calculate_context_quality(&self, messages: &[Message], topics: &[Topic]) -> f32 {
        let mut quality = 0.0;
        
        // Message relevance
        if !messages.is_empty() {
            let relevance_sum: f32 = messages.iter()
                .filter_map(|m| m.metadata.as_ref().and_then(|meta| meta.confidence_score))
                .sum();
            quality += relevance_sum / messages.len() as f32 * 0.4;
        }
        
        // Topic coherence
        if !topics.is_empty() {
            let topic_confidence: f32 = topics.iter().map(|t| t.confidence).sum();
            quality += (topic_confidence / topics.len() as f32) * 0.3;
        }
        
        // Context completeness
        let completeness = if messages.len() >= self.config.sliding_window_size / 2 { 1.0 } else { 0.5 };
        quality += completeness * 0.3;
        
        quality.min(1.0)
    }

    async fn calculate_coverage_score(&self, messages: &[Message]) -> f32 {
        // Simple coverage calculation based on message diversity
        let unique_intents: std::collections::HashSet<String> = messages.iter()
            .filter_map(|m| m.metadata.as_ref().and_then(|meta| meta.intent_classification.clone()))
            .collect();
        
        if messages.is_empty() {
            0.0
        } else {
            (unique_intents.len() as f32 / messages.len() as f32).min(1.0)
        }
    }

    async fn estimate_token_count(&self, text: &str) -> usize {
        // Rough token estimation: ~4 characters per token
        text.len() / 4
    }

    async fn extract_structured_context(&self, messages: &[Message]) -> Result<StructuredContext> {
        let mut entities = Vec::new();
        let mut facts = Vec::new();
        let mut queries = Vec::new();

        for message in messages {
            if let Some(metadata) = &message.metadata {
                // Extract entities
                if let Some(extracted_entities) = &metadata.entities_extracted {
                    entities.extend(extracted_entities.iter().cloned());
                }

                // Extract SPARQL queries
                if let Some(sparql) = &metadata.sparql_query {
                    queries.push(sparql.clone());
                }

                // Extract facts from retrieved triples
                if let Some(triples) = &metadata.retrieved_triples {
                    facts.extend(triples.iter().cloned());
                }
            }
        }

        Ok(StructuredContext {
            entities,
            facts,
            queries,
            relationships: Vec::new(), // TODO: Extract relationships
        })
    }

    async fn calculate_compression_ratio(&self, original_messages: &[Message], summary: &str) -> f32 {
        let original_length: usize = original_messages.iter().map(|m| m.content.len()).sum();
        if original_length == 0 {
            0.0
        } else {
            summary.len() as f32 / original_length as f32
        }
    }

    async fn calculate_context_efficiency(&self) -> f32 {
        // Calculate how efficiently the context is being used
        let active_ratio = self.context_window.active_messages().await as f32 / 
                          self.context_window.total_messages().await as f32;
        let importance_efficiency = self.importance_scorer.average_score().await;
        
        (active_ratio + importance_efficiency) / 2.0
    }
}

/// Context window with sliding window management
struct ContextWindow {
    config: ContextConfig,
    messages: VecDeque<ContextMessage>,
    pinned_messages: HashMap<String, PinnedMessage>,
    summary: Option<ContextSummary>,
    total_token_count: usize,
}

#[derive(Debug, Clone)]
struct ContextMessage {
    message: Message,
    importance_score: f32,
    added_at: SystemTime,
    last_accessed: SystemTime,
    access_count: usize,
}

#[derive(Debug, Clone)]
struct PinnedMessage {
    message_id: String,
    reason: PinReason,
    pinned_at: SystemTime,
    importance_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PinReason {
    HighImportance,
    UserRequest,
    KeyInformation,
    ContextAnchor,
    Reference,
}

impl ContextWindow {
    fn new(config: &ContextConfig) -> Self {
        Self {
            config: config.clone(),
            messages: VecDeque::new(),
            pinned_messages: HashMap::new(),
            summary: None,
            total_token_count: 0,
        }
    }

    async fn add_message(&mut self, message: Message, importance_score: f32) -> Result<WindowUpdate> {
        let context_message = ContextMessage {
            message,
            importance_score,
            added_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            access_count: 1,
        };

        // Estimate token count for this message
        let message_tokens = context_message.message.content.len() / 4; // Rough estimate
        self.total_token_count += message_tokens;

        self.messages.push_back(context_message);

        // Check if we need to trim the window
        let mut evicted_messages = Vec::new();
        while self.should_trim_window() {
            if let Some(evicted) = self.evict_least_important().await? {
                evicted_messages.push(evicted);
            } else {
                break;
            }
        }

        Ok(WindowUpdate {
            message_added: true,
            evicted_messages,
            current_size: self.messages.len(),
            token_count: self.total_token_count,
        })
    }

    async fn get_effective_messages(&self) -> Result<Vec<Message>> {
        let mut effective_messages = Vec::new();

        // Add pinned messages first
        for pinned in self.pinned_messages.values() {
            if let Some(context_msg) = self.messages.iter().find(|m| m.message.id == pinned.message_id) {
                effective_messages.push(context_msg.message.clone());
            }
        }

        // Add recent messages up to window size
        let recent_count = self.config.sliding_window_size.saturating_sub(effective_messages.len());
        for context_msg in self.messages.iter().rev().take(recent_count) {
            if !effective_messages.iter().any(|m| m.id == context_msg.message.id) {
                effective_messages.push(context_msg.message.clone());
            }
        }

        // Sort by timestamp to maintain conversation order
        effective_messages.sort_by_key(|m| m.timestamp);

        Ok(effective_messages)
    }

    async fn pin_message(&mut self, message_id: &str, reason: PinReason) -> Result<()> {
        if let Some(context_msg) = self.messages.iter().find(|m| m.message.id == message_id) {
            let pinned = PinnedMessage {
                message_id: message_id.to_string(),
                reason,
                pinned_at: SystemTime::now(),
                importance_score: context_msg.importance_score,
            };
            self.pinned_messages.insert(message_id.to_string(), pinned);
            debug!("Pinned message: {}", message_id);
        } else {
            return Err(anyhow!("Message not found in context window: {}", message_id));
        }
        Ok(())
    }

    async fn unpin_message(&mut self, message_id: &str) -> Result<()> {
        if self.pinned_messages.remove(message_id).is_some() {
            debug!("Unpinned message: {}", message_id);
        } else {
            warn!("Attempted to unpin non-pinned message: {}", message_id);
        }
        Ok(())
    }

    fn should_trim_window(&self) -> bool {
        self.messages.len() > self.config.sliding_window_size ||
        self.total_token_count > self.config.max_context_length
    }

    async fn evict_least_important(&mut self) -> Result<Option<Message>> {
        // Find least important non-pinned message
        let mut least_important_idx = None;
        let mut min_score = f32::MAX;

        for (idx, context_msg) in self.messages.iter().enumerate() {
            if !self.pinned_messages.contains_key(&context_msg.message.id) &&
               context_msg.importance_score < min_score {
                min_score = context_msg.importance_score;
                least_important_idx = Some(idx);
            }
        }

        if let Some(idx) = least_important_idx {
            if let Some(evicted) = self.messages.remove(idx) {
                let evicted_tokens = evicted.message.content.len() / 4;
                self.total_token_count = self.total_token_count.saturating_sub(evicted_tokens);
                return Ok(Some(evicted.message));
            }
        }

        Ok(None)
    }

    async fn should_summarize(&self) -> bool {
        self.config.enable_summarization &&
        self.messages.len() >= self.config.summarization_threshold
    }

    async fn get_messages_for_summarization(&self) -> Result<Vec<Message>> {
        // Get older messages that aren't pinned
        let cutoff_idx = self.messages.len().saturating_sub(self.config.sliding_window_size);
        
        Ok(self.messages.iter()
            .take(cutoff_idx)
            .filter(|m| !self.pinned_messages.contains_key(&m.message.id))
            .map(|m| m.message.clone())
            .collect::<Vec<_>>())
    }

    async fn apply_summarization(&mut self, summary: ContextSummary) -> Result<()> {
        // Remove summarized messages
        let cutoff_idx = self.messages.len().saturating_sub(self.config.sliding_window_size);
        for _ in 0..cutoff_idx {
            if let Some(removed) = self.messages.pop_front() {
                if !self.pinned_messages.contains_key(&removed.message.id) {
                    let removed_tokens = removed.message.content.len() / 4;
                    self.total_token_count = self.total_token_count.saturating_sub(removed_tokens);
                }
            }
        }

        self.summary = Some(summary);
        Ok(())
    }

    async fn get_summary(&self) -> Option<String> {
        self.summary.as_ref().map(|s| s.text.clone())
    }

    async fn total_messages(&self) -> usize {
        self.messages.len()
    }

    async fn active_messages(&self) -> usize {
        std::cmp::min(self.messages.len(), self.config.sliding_window_size)
    }

    async fn pinned_count(&self) -> usize {
        self.pinned_messages.len()
    }

    async fn get_state_snapshot(&self) -> ContextState {
        ContextState {
            message_count: self.messages.len(),
            pinned_count: self.pinned_messages.len(),
            token_count: self.total_token_count,
            has_summary: self.summary.is_some(),
            current_topic: None, // TODO: Get from topic tracker
        }
    }

    async fn adjust_for_topic(&mut self, _transition: &TopicTransition) -> Result<WindowAdjustment> {
        // TODO: Implement topic-specific adjustments
        Ok(WindowAdjustment {
            messages_reordered: false,
            importance_rescored: false,
            window_size_adjusted: false,
        })
    }
}

/// Supporting structures and types

#[derive(Debug, Clone)]
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

// Placeholder implementations for supporting components

struct TopicTracker {
    _config: ContextConfig,
}

impl TopicTracker {
    fn new(_config: &ContextConfig) -> Self {
        Self { _config: _config.clone() }
    }

    async fn process_message(&mut self, _message: &Message) -> Result<TopicUpdate> {
        Ok(TopicUpdate {
            new_topics: Vec::new(),
            topic_changes: Vec::new(),
            drift_detected: false,
        })
    }

    async fn get_current_topics(&self) -> Vec<Topic> {
        Vec::new()
    }

    async fn transition_to_topic(&mut self, topic: &str, _hint: Option<&str>) -> Result<TopicTransition> {
        Ok(TopicTransition {
            from_topic: None,
            to_topic: topic.to_string(),
            transition_reason: "User initiated".to_string(),
            confidence: 0.8,
            timestamp: SystemTime::now(),
        })
    }

    async fn topic_count(&self) -> usize {
        0
    }
}

struct ImportanceScorer {
    _config: ContextConfig,
}

impl ImportanceScorer {
    fn new(_config: &ContextConfig) -> Self {
        Self { _config: _config.clone() }
    }

    async fn score_message(&self, message: &Message, _analytics: Option<&ConversationAnalytics>) -> Result<f32> {
        // Simple importance scoring based on message characteristics
        let mut score: f32 = 0.5; // Base score

        // Boost for questions
        if message.content.contains('?') {
            score += 0.2;
        }

        // Boost for longer messages
        if message.content.len() > 100 {
            score += 0.1;
        }

        // Boost for messages with metadata
        if message.metadata.is_some() {
            score += 0.2;
        }

        Ok(score.min(1.0))
    }

    async fn update_for_context_switch(&mut self, _transition: &TopicTransition) -> Result<()> {
        Ok(())
    }

    async fn average_score(&self) -> f32 {
        0.7
    }
}

struct SummarizationEngine {
    _config: ContextConfig,
}

impl SummarizationEngine {
    fn new(_config: &ContextConfig) -> Self {
        Self { _config: _config.clone() }
    }

    async fn summarize_messages(&self, messages: &[Message]) -> Result<ContextSummary> {
        // Simple summarization - in production, use LLM
        let summary_text = if messages.is_empty() {
            "No messages to summarize".to_string()
        } else {
            format!("Summary of {} messages discussing various topics", messages.len())
        };

        Ok(ContextSummary {
            text: summary_text,
            key_points: Vec::new(),
            entities_mentioned: Vec::new(),
            topics_covered: Vec::new(),
            created_at: SystemTime::now(),
        })
    }

    async fn summarization_count(&self) -> usize {
        0
    }
}

struct MemoryOptimizer {
    _config: ContextConfig,
}

impl MemoryOptimizer {
    fn new(_config: &ContextConfig) -> Self {
        Self { _config: _config.clone() }
    }

    async fn optimize_context(&self, _window: &mut ContextWindow) -> Result<OptimizationUpdate> {
        Ok(OptimizationUpdate {
            memory_saved: 0,
            operations_performed: Vec::new(),
            efficiency_improvement: 0.0,
        })
    }

    async fn optimization_count(&self) -> usize {
        0
    }
}