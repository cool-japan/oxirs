//! Context Window Management
//!
//! Sliding window implementation with message pinning and adaptive sizing.

use super::config::ContextConfig;
use super::types::*;
use crate::Message;
use anyhow::{anyhow, Result};
use std::collections::{HashMap, VecDeque};
use std::time::SystemTime;
use tracing::{debug, info, warn};

impl ContextWindow {
    pub fn new(config: &ContextConfig) -> Self {
        Self {
            config: config.clone(),
            messages: VecDeque::new(),
            pinned_messages: HashMap::new(),
            summary: None,
            total_token_count: 0,
        }
    }

    pub async fn add_message(
        &mut self,
        message: Message,
        importance_score: f32,
    ) -> Result<WindowUpdate> {
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

    pub async fn get_effective_messages(&self) -> Result<Vec<Message>> {
        let mut effective_messages = Vec::new();

        // Add pinned messages first
        for pinned in self.pinned_messages.values() {
            if let Some(context_msg) = self
                .messages
                .iter()
                .find(|m| m.message.id == pinned.message_id)
            {
                effective_messages.push(context_msg.message.clone());
            }
        }

        // Add recent messages up to window size
        let recent_count = self
            .config
            .sliding_window_size
            .saturating_sub(effective_messages.len());
        for context_msg in self.messages.iter().rev().take(recent_count) {
            if !effective_messages
                .iter()
                .any(|m| m.id == context_msg.message.id)
            {
                effective_messages.push(context_msg.message.clone());
            }
        }

        // Sort by timestamp to maintain conversation order
        effective_messages.sort_by_key(|m| m.timestamp);

        Ok(effective_messages)
    }

    pub async fn pin_message(&mut self, message_id: &str, reason: PinReason) -> Result<()> {
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
            return Err(anyhow!(
                "Message not found in context window: {}",
                message_id
            ));
        }
        Ok(())
    }

    pub async fn unpin_message(&mut self, message_id: &str) -> Result<()> {
        if self.pinned_messages.remove(message_id).is_some() {
            debug!("Unpinned message: {}", message_id);
        } else {
            warn!("Attempted to unpin non-pinned message: {}", message_id);
        }
        Ok(())
    }

    fn should_trim_window(&self) -> bool {
        self.messages.len() > self.config.sliding_window_size
            || self.total_token_count > self.config.max_context_length
    }

    async fn evict_least_important(&mut self) -> Result<Option<Message>> {
        // Find least important non-pinned message
        let mut least_important_idx = None;
        let mut min_score = f32::MAX;

        for (idx, context_msg) in self.messages.iter().enumerate() {
            if !self.pinned_messages.contains_key(&context_msg.message.id)
                && context_msg.importance_score < min_score
            {
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

    pub async fn should_summarize(&self) -> bool {
        self.config.enable_summarization
            && self.messages.len() >= self.config.summarization_threshold
    }

    pub async fn get_messages_for_summarization(&self) -> Result<Vec<Message>> {
        // Get older messages that aren't pinned
        let cutoff_idx = self
            .messages
            .len()
            .saturating_sub(self.config.sliding_window_size);

        Ok(self
            .messages
            .iter()
            .take(cutoff_idx)
            .filter(|m| !self.pinned_messages.contains_key(&m.message.id))
            .map(|m| m.message.clone())
            .collect::<Vec<_>>())
    }

    pub async fn apply_summarization(&mut self, summary: ContextSummary) -> Result<()> {
        // Remove summarized messages
        let cutoff_idx = self
            .messages
            .len()
            .saturating_sub(self.config.sliding_window_size);
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

    pub async fn get_summary(&self) -> Option<ContextSummary> {
        self.summary.clone()
    }

    pub async fn total_messages(&self) -> usize {
        self.messages.len()
    }

    pub async fn active_messages(&self) -> usize {
        std::cmp::min(self.messages.len(), self.config.sliding_window_size)
    }

    pub async fn pinned_count(&self) -> usize {
        self.pinned_messages.len()
    }

    pub async fn get_state_snapshot(&self) -> ContextState {
        ContextState {
            message_count: self.messages.len(),
            pinned_count: self.pinned_messages.len(),
            token_count: self.total_token_count,
            has_summary: self.summary.is_some(),
            current_topic: None,
        }
    }

    pub async fn get_state_snapshot_with_topic(
        &self,
        current_topic: Option<String>,
    ) -> ContextState {
        ContextState {
            message_count: self.messages.len(),
            pinned_count: self.pinned_messages.len(),
            token_count: self.total_token_count,
            has_summary: self.summary.is_some(),
            current_topic,
        }
    }

    pub async fn adjust_for_topic(
        &mut self,
        transition: &TopicTransition,
    ) -> Result<WindowAdjustment> {
        let mut messages_reordered = false;
        let mut importance_rescored = false;
        let mut window_size_adjusted = false;

        // 1. Adjust window size based on topic complexity and confidence
        let optimal_window_size = self
            .calculate_optimal_window_size_for_topic(transition)
            .await;
        if optimal_window_size != self.config.sliding_window_size {
            self.config.sliding_window_size = optimal_window_size;
            window_size_adjusted = true;
            debug!(
                "Adjusted window size to {} for topic '{}'",
                optimal_window_size, transition.to_topic
            );
        }

        // 2. Rescore message importance based on topic relevance
        if transition.confidence > 0.6 {
            importance_rescored = self.rescore_messages_for_topic(transition).await?;
        }

        // 3. Reorder messages based on new importance scores
        if importance_rescored {
            messages_reordered = self.reorder_messages_by_importance().await;
        }

        // 4. Apply topic-specific filtering if needed
        if transition.confidence > 0.8 {
            self.apply_topic_specific_filtering(transition).await?;
        }

        // 5. Adjust message priorities for pinned messages
        self.adjust_pinned_message_priorities_for_topic(transition)
            .await?;

        info!(
            "Topic adjustments completed for '{}': reordered={}, rescored={}, window_adjusted={}",
            transition.to_topic, messages_reordered, importance_rescored, window_size_adjusted
        );

        Ok(WindowAdjustment {
            messages_reordered,
            importance_rescored,
            window_size_adjusted,
        })
    }

    /// Calculate optimal window size for the given topic
    async fn calculate_optimal_window_size_for_topic(&self, transition: &TopicTransition) -> usize {
        let base_size = self.config.sliding_window_size;

        // Adjust based on topic complexity (estimated from topic name length and confidence)
        let topic_complexity_factor = if transition.to_topic.len() > 20 {
            1.2 // Complex topics need more context
        } else if transition.to_topic.len() < 10 {
            0.8 // Simple topics need less context
        } else {
            1.0
        };

        // Adjust based on transition confidence
        let confidence_factor = if transition.confidence > 0.9 {
            1.1 // High confidence topics can use more context
        } else if transition.confidence < 0.5 {
            0.9 // Low confidence topics should use less context
        } else {
            1.0
        };

        let adjusted_size =
            (base_size as f32 * topic_complexity_factor * confidence_factor) as usize;

        // Clamp to reasonable bounds
        adjusted_size.clamp(10, 100)
    }

    /// Rescore messages based on their relevance to the new topic
    async fn rescore_messages_for_topic(&mut self, transition: &TopicTransition) -> Result<bool> {
        let mut rescored = false;
        let topic_keywords = self.extract_topic_keywords(&transition.to_topic);

        // Create a vector of adjustments to avoid borrow checker issues
        let mut adjustments = Vec::new();

        for (index, context_message) in self.messages.iter().enumerate() {
            let topic_relevance = Self::calculate_message_topic_relevance_static(
                &context_message.message,
                &topic_keywords,
            );

            let original_score = context_message.importance_score;
            let topic_adjustment = match topic_relevance {
                relevance if relevance > 0.8 => 1.3, // High relevance boost
                relevance if relevance > 0.5 => 1.1, // Moderate relevance boost
                relevance if relevance > 0.2 => 1.0, // No change
                _ => 0.8,                            // Low relevance penalty
            };

            let new_score = (original_score * topic_adjustment).min(1.0);
            adjustments.push((index, new_score, original_score));
        }

        // Apply adjustments
        for (index, new_score, original_score) in adjustments {
            if let Some(context_message) = self.messages.get_mut(index) {
                context_message.importance_score = new_score;
                if (new_score - original_score).abs() > 0.05 {
                    rescored = true;
                }
            }
        }

        if rescored {
            debug!(
                "Rescored {} messages for topic relevance",
                self.messages.len()
            );
        }

        Ok(rescored)
    }

    /// Extract keywords from topic name for relevance calculation
    fn extract_topic_keywords(&self, topic: &str) -> Vec<String> {
        topic
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|word| word.len() > 2) // Filter out short words
            .map(|word| word.to_string())
            .collect()
    }

    /// Calculate how relevant a message is to the current topic
    async fn calculate_message_topic_relevance(
        &self,
        message: &Message,
        topic_keywords: &[String],
    ) -> f32 {
        Self::calculate_message_topic_relevance_static(message, topic_keywords)
    }

    /// Static version to avoid borrow checker issues
    fn calculate_message_topic_relevance_static(
        message: &Message,
        topic_keywords: &[String],
    ) -> f32 {
        let message_text = message.content.to_lowercase();
        let mut relevance_score = 0.0;
        let mut keyword_matches = 0;

        for keyword in topic_keywords {
            if message_text.contains(keyword) {
                keyword_matches += 1;
                relevance_score += 0.2; // Base score per keyword match

                // Bonus for exact word matches (not just substring matches)
                if message_text.split_whitespace().any(|word| word == keyword) {
                    relevance_score += 0.1;
                }
            }
        }

        // Apply diminishing returns for multiple keyword matches
        if keyword_matches > 0 {
            relevance_score *= 1.0 - (keyword_matches as f32 * 0.05).min(0.3);
        }

        // Check message metadata for additional topic relevance indicators
        if let Some(_metadata) = &message.metadata {
            // Basic boost for messages with metadata (indicating they were processed)
            relevance_score += 0.05;
        }

        relevance_score.min(1.0)
    }

    /// Reorder messages by their importance scores
    async fn reorder_messages_by_importance(&mut self) -> bool {
        let original_order: Vec<_> = self.messages.iter().map(|m| m.message.id.clone()).collect();

        // Sort by importance score (descending) while maintaining relative chronological order for equal scores
        self.messages.make_contiguous().sort_by(|a, b| {
            b.importance_score
                .partial_cmp(&a.importance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.added_at.cmp(&b.added_at))
        });

        let new_order: Vec<_> = self.messages.iter().map(|m| m.message.id.clone()).collect();
        let reordered = original_order != new_order;

        if reordered {
            debug!("Reordered {} messages by importance", self.messages.len());
        }

        reordered
    }

    /// Apply topic-specific filtering to remove less relevant messages
    async fn apply_topic_specific_filtering(&mut self, transition: &TopicTransition) -> Result<()> {
        let original_count = self.messages.len();

        // Remove messages with very low importance scores (below threshold)
        let min_importance_threshold = if transition.confidence > 0.9 {
            0.3
        } else {
            0.2
        };

        self.messages
            .retain(|context_message| context_message.importance_score >= min_importance_threshold);

        let filtered_count = original_count - self.messages.len();
        if filtered_count > 0 {
            debug!(
                "Filtered out {} low-importance messages for topic '{}'",
                filtered_count, transition.to_topic
            );
        }

        Ok(())
    }

    /// Adjust priorities for pinned messages based on topic relevance
    async fn adjust_pinned_message_priorities_for_topic(
        &mut self,
        transition: &TopicTransition,
    ) -> Result<()> {
        let topic_keywords = self.extract_topic_keywords(&transition.to_topic);

        // Collect adjustments to avoid borrow checker issues
        let mut adjustments = Vec::new();

        for (message_id, pinned_message) in &self.pinned_messages {
            if let Some(context_message) =
                self.messages.iter().find(|m| m.message.id == *message_id)
            {
                let topic_relevance = Self::calculate_message_topic_relevance_static(
                    &context_message.message,
                    &topic_keywords,
                );

                let original_score = pinned_message.importance_score;
                let new_score = (original_score + topic_relevance * 0.3).min(1.0);

                adjustments.push((message_id.clone(), original_score, new_score));
            }
        }

        // Apply adjustments
        for (message_id, original_score, new_score) in adjustments {
            if let Some(pinned_message) = self.pinned_messages.get_mut(&message_id) {
                pinned_message.importance_score = new_score;

                debug!(
                    "Adjusted pinned message '{}' importance from {:.2} to {:.2} for topic relevance",
                    message_id, original_score, new_score
                );
            }
        }

        Ok(())
    }
}
