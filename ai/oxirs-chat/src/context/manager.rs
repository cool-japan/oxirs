//! Advanced Context Manager
//!
//! Main context management implementation with sliding windows and adaptive features.

use anyhow::Result;
use std::time::{Duration, SystemTime};
use tracing::{debug, info, warn};
use crate::{analytics::ConversationAnalytics, Message, MessageRole};
use super::config::ContextConfig;
use super::components::*;
use super::types::*;
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
        _conversation_analytics: Option<&ConversationAnalytics>,
    ) -> Result<ContextUpdate> {
        let start_time = SystemTime::now();

        // Calculate importance score
        let importance_score = self.importance_scorer.score_message(message).await;

        // Update context window
        let window_update = self
            .context_window
            .add_message(message.clone(), importance_score)
            .await?;

        // Track topic changes
        let topic_update = if self.config.enable_topic_tracking {
            Some(self.topic_tracker.process_message(message).await?)
        } else {
            None
        };

        // Check if summarization is needed
        let summarization_update =
            if self.config.enable_summarization && self.context_window.should_summarize().await {
                Some(self.perform_summarization().await?)
            } else {
                None
            };

        // Optimize memory if needed
        let optimization_update = if self.config.memory_optimization_enabled {
            Some(
                self.memory_optimizer
                    .optimize_context(&mut self.context_window)
                    .await?,
            )
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
            context_text.push_str(&summary.text);
            context_text.push_str("\n\n");
        }

        // Add current topics
        if !current_topics.is_empty() {
            context_text.push_str("## Current Topics\n");
            for topic in &current_topics {
                context_text.push_str(&format!(
                    "- {} (confidence: {:.2})\n",
                    topic.name, topic.confidence
                ));
            }
            context_text.push('\n');
        }

        // Add recent messages
        context_text.push_str("## Recent Messages\n");
        for message in &effective_messages {
            let role_indicator = match message.role {
                MessageRole::User => "User",
                MessageRole::Assistant => "Assistant",
                MessageRole::System => "System",
                MessageRole::Function => "Function",
            };
            context_text.push_str(&format!("{}: {}\n", role_indicator, message.content));
        }

        // Calculate quality metrics
        let quality_score = self
            .calculate_context_quality(&effective_messages, &current_topics)
            .await;
        let coverage_score = self.calculate_coverage_score(&effective_messages).await;

        // Calculate values before moving into the struct
        let token_count = self.estimate_token_count(&context_text).await;
        let structured_context = self.extract_structured_context(&effective_messages).await?;

        Ok(AssembledContext {
            context_text,
            effective_messages,
            current_topics,
            context_summary: context_summary.map(|s| s.text),
            quality_score,
            coverage_score,
            token_count,
            structured_context,
        })
    }

    /// Handle context switching
    pub async fn switch_context(
        &mut self,
        new_topic: &str,
        context_hint: Option<&str>,
    ) -> Result<ContextSwitch> {
        info!("Switching context to topic: {}", new_topic);

        // Save current context state
        let current_topic = self.topic_tracker.get_current_topic().await;
        let previous_state = self
            .context_window
            .get_state_snapshot_with_topic(current_topic)
            .await;

        // Perform topic transition
        let topic_transition = self
            .topic_tracker
            .transition_to_topic(new_topic, context_hint)
            .await?;

        // Adjust context window for new topic
        let window_adjustment = self
            .context_window
            .adjust_for_topic(&topic_transition)
            .await?;

        // Update importance scoring for new context
        self.importance_scorer
            .update_for_context_switch(&topic_transition)
            .await?;

        // Implement actual preservation logic
        let context_preserved = self
            .evaluate_context_preservation(&previous_state, &topic_transition, &window_adjustment)
            .await?;

        Ok(ContextSwitch {
            previous_state,
            new_topic: new_topic.to_string(),
            topic_transition,
            window_adjustment,
            context_preserved,
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

    /// Evaluate whether context was properly preserved during a topic switch
    async fn evaluate_context_preservation(
        &self,
        previous_state: &ContextState,
        topic_transition: &TopicTransition,
        window_adjustment: &WindowAdjustment,
    ) -> Result<bool> {
        let mut preservation_score = 0.0;
        let mut factors_checked = 0;

        // Factor 1: Topic transition confidence (weight: 0.3)
        if topic_transition.confidence >= 0.8 {
            preservation_score += 0.3;
        } else if topic_transition.confidence >= 0.6 {
            preservation_score += 0.15;
        }
        factors_checked += 1;

        // Factor 2: Message retention (weight: 0.25)
        let current_state = self.context_window.get_state_snapshot().await;
        let message_retention_ratio = if previous_state.message_count > 0 {
            current_state.message_count as f32 / previous_state.message_count as f32
        } else {
            1.0
        };

        if message_retention_ratio >= 0.8 {
            preservation_score += 0.25;
        } else if message_retention_ratio >= 0.6 {
            preservation_score += 0.15;
        } else if message_retention_ratio >= 0.4 {
            preservation_score += 0.1;
        }
        factors_checked += 1;

        // Factor 3: Pinned messages preservation (weight: 0.2)
        if previous_state.pinned_count > 0 {
            let pinned_retention_ratio =
                current_state.pinned_count as f32 / previous_state.pinned_count as f32;
            if pinned_retention_ratio >= 0.9 {
                preservation_score += 0.2;
            } else if pinned_retention_ratio >= 0.7 {
                preservation_score += 0.1;
            }
        } else {
            // If there were no pinned messages, this factor doesn't penalize
            preservation_score += 0.2;
        }
        factors_checked += 1;

        // Factor 4: Context continuity (weight: 0.15)
        if previous_state.has_summary && current_state.has_summary {
            preservation_score += 0.15;
        } else if !previous_state.has_summary && !current_state.has_summary {
            preservation_score += 0.1; // Consistency bonus
        } else if !previous_state.has_summary && current_state.has_summary {
            preservation_score += 0.05; // Slight bonus for improvement
        }
        factors_checked += 1;

        // Factor 5: Window adjustment success (weight: 0.1)
        let adjustment_success_score = [
            window_adjustment.messages_reordered,
            window_adjustment.importance_rescored,
            window_adjustment.window_size_adjusted,
        ]
        .iter()
        .filter(|&&success| success)
        .count() as f32
            / 3.0;

        preservation_score += 0.1 * adjustment_success_score;
        factors_checked += 1;

        // Calculate final score and determine if context was preserved
        let final_score = preservation_score;
        let context_preserved = final_score >= 0.7; // Require 70% preservation score

        debug!(
            "Context preservation evaluation: score={:.2}, factors_checked={}, preserved={}",
            final_score, factors_checked, context_preserved
        );

        if !context_preserved {
            warn!(
                "Context preservation failed: transition_confidence={:.2}, message_retention={:.2}, final_score={:.2}",
                topic_transition.confidence,
                message_retention_ratio,
                final_score
            );
        } else {
            info!(
                "Context successfully preserved: score={:.2}, transition to topic '{}'",
                final_score, topic_transition.to_topic
            );
        }

        Ok(context_preserved)
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
        let summary = self
            .summarization_engine
            .summarize_messages(&messages_to_summarize)
            .await?;

        let summary_text = summary.text.clone();
        let summary_clone = summary.clone();
        self.context_window.apply_summarization(summary).await?;

        Ok(SummarizationUpdate {
            summary: summary_clone,
            messages_summarized: messages_to_summarize.len(),
            compression_ratio: self
                .calculate_compression_ratio(&messages_to_summarize, &summary_text)
                .await,
        })
    }

    async fn calculate_context_quality(&self, messages: &[Message], topics: &[Topic]) -> f32 {
        let mut quality = 0.0;

        // Message relevance
        if !messages.is_empty() {
            let relevance_sum: f32 = messages
                .iter()
                .filter_map(|m| {
                    m.metadata
                        .as_ref()
                        .and_then(|meta| meta.confidence.map(|c| c as f32))
                })
                .sum();
            quality += relevance_sum / messages.len() as f32 * 0.4;
        }

        // Topic coherence
        if !topics.is_empty() {
            let topic_confidence: f32 = topics.iter().map(|t| t.confidence).sum();
            quality += (topic_confidence / topics.len() as f32) * 0.3;
        }

        // Context completeness
        let completeness = if messages.len() >= self.config.sliding_window_size / 2 {
            1.0
        } else {
            0.5
        };
        quality += completeness * 0.3;

        quality.min(1.0)
    }

    async fn calculate_coverage_score(&self, messages: &[Message]) -> f32 {
        // Simple coverage calculation based on message diversity
        let unique_intents: std::collections::HashSet<String> = messages
            .iter()
            .filter_map(|m| {
                m.metadata
                    .as_ref()
                    .and_then(|meta| meta.custom_fields.get("intent_classification"))
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
            })
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
        let mut relationships = Vec::new();

        for message in messages {
            if let Some(metadata) = &message.metadata {
                // Extract entities from custom fields
                if let Some(extracted_entities) = metadata.custom_fields.get("entities_extracted") {
                    if let Ok(entities_list) =
                        serde_json::from_value::<Vec<String>>(extracted_entities.clone())
                    {
                        entities.extend(entities_list);
                    }
                }

                // Extract SPARQL queries from custom fields
                if let Some(sparql) = metadata.custom_fields.get("sparql_query") {
                    if let Some(query_str) = sparql.as_str() {
                        queries.push(query_str.to_string());
                    }
                }

                // Extract facts from retrieved triples in custom fields
                if let Some(triples) = metadata.custom_fields.get("retrieved_triples") {
                    if let Ok(triples_list) = serde_json::from_value::<Vec<String>>(triples.clone())
                    {
                        facts.extend(triples_list);
                    }
                }

                // Extract relationships from custom fields
                if let Some(extracted_relationships) =
                    metadata.custom_fields.get("relationships_extracted")
                {
                    if let Ok(relationships_list) =
                        serde_json::from_value::<Vec<String>>(extracted_relationships.clone())
                    {
                        relationships.extend(relationships_list);
                    }
                }

                // Extract relationships from RAG extracted relationships
                if let Some(rag_relationships) =
                    metadata.custom_fields.get("extracted_relationships")
                {
                    if let Ok(rag_relationships_list) =
                        serde_json::from_value::<Vec<String>>(rag_relationships.clone())
                    {
                        relationships.extend(rag_relationships_list);
                    }
                }

                // Extract relationships from conversation analysis
                if let Some(conversation_relationships) =
                    metadata.custom_fields.get("conversation_relationships")
                {
                    if let Ok(conversation_relationships_list) =
                        serde_json::from_value::<Vec<String>>(conversation_relationships.clone())
                    {
                        relationships.extend(conversation_relationships_list);
                    }
                }
            }
        }

        // Deduplicate relationships
        relationships.sort();
        relationships.dedup();

        // Also extract implicit relationships from facts and entities
        let implicit_relationships = self.extract_implicit_relationships(&entities, &facts).await;
        relationships.extend(implicit_relationships);

        // Final deduplication
        relationships.sort();
        relationships.dedup();

        debug!(
            "Extracted structured context: {} entities, {} facts, {} queries, {} relationships",
            entities.len(),
            facts.len(),
            queries.len(),
            relationships.len()
        );

        Ok(StructuredContext {
            entities,
            facts,
            queries,
            relationships,
        })
    }

    /// Extract implicit relationships from entities and facts
    async fn extract_implicit_relationships(
        &self,
        entities: &[String],
        facts: &[String],
    ) -> Vec<String> {
        let mut implicit_relationships = Vec::new();

        // Extract relationships from RDF facts/triples
        for fact in facts {
            if let Some(relationship) = self.parse_relationship_from_triple(fact) {
                implicit_relationships.push(relationship);
            }
        }

        // Extract relationships from entity co-occurrence patterns
        if entities.len() >= 2 {
            for i in 0..entities.len() {
                for j in (i + 1)..entities.len() {
                    let entity1 = &entities[i];
                    let entity2 = &entities[j];

                    // Create a general relationship notation
                    let relationship = format!("{entity1} <-> {entity2}");
                    implicit_relationships.push(relationship);
                }
            }
        }

        // Limit the number of implicit relationships to avoid explosion
        if implicit_relationships.len() > 50 {
            implicit_relationships.truncate(50);
        }

        implicit_relationships
    }

    /// Parse relationship from RDF triple format
    fn parse_relationship_from_triple(&self, triple: &str) -> Option<String> {
        // Simple regex-based parsing of RDF triples
        let patterns = [
            // Standard RDF triple: <subject> <predicate> <object>
            r"<([^>]+)>\s+<([^>]+)>\s+<([^>]+)>",
            // With prefixes: prefix:subject prefix:predicate prefix:object
            r"(\w+:\w+)\s+(\w+:\w+)\s+(\w+:\w+)",
            // Mixed format
            r"([^\s]+)\s+([^\s]+)\s+([^\s]+)",
        ];

        for pattern in &patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                if let Some(captures) = regex.captures(triple) {
                    if captures.len() >= 4 {
                        let subject = captures.get(1)?.as_str();
                        let predicate = captures.get(2)?.as_str();
                        let object = captures.get(3)?.as_str();

                        // Clean up the predicate to make it more readable
                        let clean_predicate = predicate
                            .replace("http://", "")
                            .replace("https://", "")
                            .split('/')
                            .next_back()
                            .unwrap_or(predicate)
                            .replace('#', ":")
                            .to_string();

                        return Some(format!("{subject} --[{clean_predicate}]--> {object}"));
                    }
                }
            }
        }

        None
    }

    async fn calculate_compression_ratio(
        &self,
        original_messages: &[Message],
        summary: &str,
    ) -> f32 {
        let original_length: usize = original_messages.iter().map(|m| m.content.len()).sum();
        if original_length == 0 {
            0.0
        } else {
            summary.len() as f32 / original_length as f32
        }
    }

    async fn calculate_context_efficiency(&self) -> f32 {
        // Calculate how efficiently the context is being used
        let active_ratio = self.context_window.active_messages().await as f32
            / self.context_window.total_messages().await as f32;
        let importance_efficiency = self.importance_scorer.average_score().await;

        (active_ratio + importance_efficiency) / 2.0
    }
}
