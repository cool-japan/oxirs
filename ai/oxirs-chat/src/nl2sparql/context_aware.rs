//! Context-Aware Query Generation
//!
//! Generates SPARQL queries with awareness of conversation context and history.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

use super::types::SPARQLGenerationResult;

/// Context-aware generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextAwareConfig {
    /// Maximum context history to consider
    pub max_history: usize,
    /// Enable entity tracking across messages
    pub track_entities: bool,
    /// Enable variable reuse from previous queries
    pub reuse_variables: bool,
    /// Enable schema learning
    pub learn_schema: bool,
    /// Context window decay factor
    pub decay_factor: f32,
}

impl Default for ContextAwareConfig {
    fn default() -> Self {
        Self {
            max_history: 10,
            track_entities: true,
            reuse_variables: true,
            learn_schema: true,
            decay_factor: 0.9,
        }
    }
}

/// Conversation context for query generation
#[derive(Debug, Clone, Default)]
pub struct ConversationContext {
    /// Session ID
    pub session_id: String,
    /// Message history
    pub history: Vec<ContextMessage>,
    /// Tracked entities across conversation
    pub tracked_entities: HashMap<String, TrackedEntity>,
    /// Variable bindings from previous queries
    pub variable_bindings: HashMap<String, String>,
    /// Schema elements discovered
    pub discovered_schema: DiscoveredSchema,
    /// Current topic/focus
    pub current_topic: Option<String>,
}

/// Message in context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextMessage {
    /// Message ID
    pub id: String,
    /// Message content
    pub content: String,
    /// Generated SPARQL (if any)
    pub sparql: Option<String>,
    /// Entities mentioned
    pub entities: Vec<String>,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Relevance score (decays over time)
    pub relevance: f32,
}

/// Tracked entity across conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackedEntity {
    /// Entity text/URI
    pub entity: String,
    /// Entity type
    pub entity_type: String,
    /// First mentioned in message ID
    pub first_mention: String,
    /// Last mentioned in message ID
    pub last_mention: String,
    /// Mention count
    pub mention_count: usize,
    /// Resolved URI (if available)
    pub resolved_uri: Option<String>,
}

/// Discovered schema information
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiscoveredSchema {
    /// Discovered classes
    pub classes: Vec<String>,
    /// Discovered properties
    pub properties: Vec<String>,
    /// Discovered prefixes
    pub prefixes: HashMap<String, String>,
    /// Common patterns
    pub patterns: Vec<String>,
}

/// Context-aware query generator
pub struct ContextAwareGenerator {
    config: ContextAwareConfig,
}

impl ContextAwareGenerator {
    /// Create a new context-aware generator
    pub fn new(config: ContextAwareConfig) -> Self {
        info!("Initialized context-aware query generator");
        Self { config }
    }

    /// Generate query with conversation context
    pub fn generate_with_context(
        &self,
        query: &str,
        context: &mut ConversationContext,
    ) -> Result<SPARQLGenerationResult> {
        debug!("Generating context-aware query for: {}", query);

        // Update relevance scores based on time decay
        self.update_relevance_scores(context);

        // Extract entities from current query
        let current_entities = self.extract_entities(query)?;

        // Update tracked entities
        if self.config.track_entities {
            self.update_tracked_entities(context, &current_entities, query);
        }

        // Resolve anaphora (pronouns) using context
        let resolved_query = self.resolve_anaphora(query, context)?;

        // Reuse variables from previous queries
        let variable_hints = if self.config.reuse_variables {
            self.get_variable_hints(context)
        } else {
            HashMap::new()
        };

        // Generate base SPARQL
        let mut sparql = self.generate_base_sparql(&resolved_query, context)?;

        // Enhance with context-aware features
        sparql = self.enhance_with_context(sparql, context, &variable_hints)?;

        // Learn from generated query
        if self.config.learn_schema {
            self.learn_from_query(&sparql, context);
        }

        // Add to history
        self.add_to_history(context, query, &sparql, current_entities);

        Ok(SPARQLGenerationResult {
            query: sparql.clone(),
            confidence: 0.85,
            generation_method: crate::nl2sparql::types::GenerationMethod::RuleBased,
            parameters: HashMap::new(),
            explanation: Some(crate::nl2sparql::types::QueryExplanation {
                natural_language: "Generated based on conversation context".to_string(),
                reasoning_steps: vec![],
                parameter_mapping: HashMap::new(),
                alternatives: Vec::new(),
            }),
            validation_result: crate::nl2sparql::types::ValidationResult {
                is_valid: true,
                syntax_errors: Vec::new(),
                semantic_warnings: Vec::new(),
                schema_issues: Vec::new(),
                suggestions: Vec::new(),
            },
            optimization_hints: Vec::new(),
            metadata: crate::nl2sparql::types::GenerationMetadata {
                generation_time_ms: 0,
                template_used: None,
                llm_model_used: None,
                iterations: 1,
                fallback_used: false,
            },
        })
    }

    /// Update relevance scores with time decay
    fn update_relevance_scores(&self, context: &mut ConversationContext) {
        let now = chrono::Utc::now();

        for message in &mut context.history {
            let age_seconds = (now - message.timestamp).num_seconds() as f32;
            let decay = self.config.decay_factor.powf(age_seconds / 60.0); // Decay per minute
            message.relevance *= decay;
        }

        // Remove very old or irrelevant messages
        context.history.retain(|m| m.relevance > 0.1);
    }

    /// Extract entities from query
    fn extract_entities(&self, query: &str) -> Result<Vec<String>> {
        // Simplified entity extraction - capitalize words excluding question words
        let question_words = [
            "How", "What", "Where", "When", "Who", "Which", "Why", "Is", "Are", "Do", "Does",
            "Did", "Can", "Could", "Would", "Should", "Will", "The", "A", "An",
        ];

        let entities: Vec<String> = query
            .split_whitespace()
            .filter(|w| {
                let is_capitalized = w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
                is_capitalized
                    && !question_words.contains(&w.trim_end_matches(|c: char| !c.is_alphanumeric()))
            })
            .map(String::from)
            .collect();

        Ok(entities)
    }

    /// Update tracked entities
    fn update_tracked_entities(
        &self,
        context: &mut ConversationContext,
        entities: &[String],
        message_id: &str,
    ) {
        for entity in entities {
            context
                .tracked_entities
                .entry(entity.clone())
                .and_modify(|e| {
                    e.mention_count += 1;
                    e.last_mention = message_id.to_string();
                })
                .or_insert(TrackedEntity {
                    entity: entity.clone(),
                    entity_type: "Unknown".to_string(),
                    first_mention: message_id.to_string(),
                    last_mention: message_id.to_string(),
                    mention_count: 1,
                    resolved_uri: None,
                });
        }
    }

    /// Resolve pronouns and references using context
    fn resolve_anaphora(&self, query: &str, context: &ConversationContext) -> Result<String> {
        let mut resolved = query.to_string();

        // Replace "it" with most recent entity
        if resolved.to_lowercase().contains(" it ") || resolved.to_lowercase().ends_with(" it") {
            if let Some(last_entity) = self.get_most_recent_entity(context) {
                resolved = resolved.replace(" it ", &format!(" {} ", last_entity));
                resolved = resolved.replace(" it", &format!(" {}", last_entity));
            }
        }

        // Replace "them" with recent entities
        if resolved.to_lowercase().contains(" them ") {
            if let Some(recent_entities) = self.get_recent_entities(context, 3) {
                let entities_str = recent_entities.join(" and ");
                resolved = resolved.replace(" them ", &format!(" {} ", entities_str));
            }
        }

        // Replace "that" with previous topic
        if resolved.to_lowercase().contains(" that ") {
            if let Some(ref topic) = context.current_topic {
                resolved = resolved.replace(" that ", &format!(" {} ", topic));
            }
        }

        debug!("Resolved query: {} -> {}", query, resolved);

        Ok(resolved)
    }

    /// Get variable hints from previous queries
    fn get_variable_hints(&self, context: &ConversationContext) -> HashMap<String, String> {
        context.variable_bindings.clone()
    }

    /// Generate base SPARQL query
    fn generate_base_sparql(&self, query: &str, context: &ConversationContext) -> Result<String> {
        let lowercase = query.to_lowercase();

        // Determine query type
        let sparql = if lowercase.contains("count") || lowercase.contains("how many") {
            self.generate_count_query(query, context)?
        } else if lowercase.contains("list")
            || lowercase.contains("show")
            || lowercase.contains("find")
        {
            self.generate_select_query(query, context)?
        } else if lowercase.contains("describe") {
            self.generate_describe_query(query, context)?
        } else {
            // Default SELECT query
            self.generate_select_query(query, context)?
        };

        Ok(sparql)
    }

    /// Generate COUNT query
    fn generate_count_query(&self, query: &str, _context: &ConversationContext) -> Result<String> {
        let entities = self.extract_entities(query)?;
        let primary_entity = entities
            .first()
            .cloned()
            .unwrap_or_else(|| "thing".to_string());

        let mut sparql =
            String::from("PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n");
        sparql.push_str("PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n\n");
        sparql.push_str("SELECT (COUNT(?s) AS ?count) WHERE {\n");
        sparql.push_str("  ?s rdf:type ?type .\n");
        sparql.push_str(&format!(
            "  FILTER (contains(str(?type), \"{}\"))\n",
            primary_entity
        ));
        sparql.push_str("}\n");

        Ok(sparql)
    }

    /// Generate SELECT query
    fn generate_select_query(&self, query: &str, _context: &ConversationContext) -> Result<String> {
        let entities = self.extract_entities(query)?;

        let mut sparql =
            String::from("PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n");
        sparql.push_str("PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n\n");
        sparql.push_str("SELECT DISTINCT ?s ?p ?o WHERE {\n");

        if let Some(entity) = entities.first() {
            sparql.push_str("  ?s ?p ?o .\n");
            sparql.push_str(&format!(
                "  FILTER (contains(str(?s), \"{}\") || contains(str(?o), \"{}\"))\n",
                entity, entity
            ));
        } else {
            sparql.push_str("  ?s ?p ?o .\n");
        }

        sparql.push_str("}\n");
        sparql.push_str("LIMIT 100\n");

        Ok(sparql)
    }

    /// Generate DESCRIBE query
    fn generate_describe_query(
        &self,
        query: &str,
        _context: &ConversationContext,
    ) -> Result<String> {
        let entities = self.extract_entities(query)?;

        let mut sparql =
            String::from("PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n");
        sparql.push_str("PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n\n");

        if let Some(entity) = entities.first() {
            sparql.push_str(&format!("DESCRIBE <http://example.org/{}>\n", entity));
        } else {
            sparql.push_str("DESCRIBE ?s WHERE { ?s ?p ?o } LIMIT 1\n");
        }

        Ok(sparql)
    }

    /// Enhance query with context
    fn enhance_with_context(
        &self,
        mut sparql: String,
        context: &ConversationContext,
        _variable_hints: &HashMap<String, String>,
    ) -> Result<String> {
        // Add commonly used prefixes from discovered schema
        for (prefix, uri) in &context.discovered_schema.prefixes {
            if !sparql.contains(&format!("PREFIX {}", prefix)) {
                sparql = format!("PREFIX {}: <{}>\n{}", prefix, uri, sparql);
            }
        }

        // Add ORDER BY if there's a clear ordering criterion
        if context
            .current_topic
            .as_ref()
            .map(|t| t.contains("sorted") || t.contains("ordered"))
            .unwrap_or(false)
            && !sparql.contains("ORDER BY")
            && sparql.contains("SELECT")
        {
            // Add before LIMIT if present
            if let Some(limit_pos) = sparql.find("LIMIT") {
                sparql.insert_str(limit_pos, "ORDER BY ?s\n");
            } else {
                sparql.push_str("ORDER BY ?s\n");
            }
        }

        Ok(sparql)
    }

    /// Learn schema from generated query
    fn learn_from_query(&self, sparql: &str, context: &mut ConversationContext) {
        // Extract classes (rdf:type patterns)
        if let Some(_class_match) = sparql.find("rdf:type") {
            // Simplified - would use regex for better extraction
            context
                .discovered_schema
                .classes
                .push("NewClass".to_string());
        }

        // Extract properties
        let property_pattern = "?s ?p ?o";
        if sparql.contains(property_pattern) {
            // Would extract actual properties here
        }

        // Deduplicate
        context.discovered_schema.classes.sort();
        context.discovered_schema.classes.dedup();
        context.discovered_schema.properties.sort();
        context.discovered_schema.properties.dedup();
    }

    /// Add message to history
    fn add_to_history(
        &self,
        context: &mut ConversationContext,
        query: &str,
        sparql: &str,
        entities: Vec<String>,
    ) {
        let message = ContextMessage {
            id: uuid::Uuid::new_v4().to_string(),
            content: query.to_string(),
            sparql: Some(sparql.to_string()),
            entities,
            timestamp: chrono::Utc::now(),
            relevance: 1.0,
        };

        context.history.push(message);

        // Keep only recent history
        if context.history.len() > self.config.max_history {
            context.history.remove(0);
        }

        // Update current topic
        context.current_topic = Some(query.to_string());
    }

    /// Get most recent entity from context
    fn get_most_recent_entity(&self, context: &ConversationContext) -> Option<String> {
        context
            .tracked_entities
            .values()
            .max_by_key(|e| &e.last_mention)
            .map(|e| e.entity.clone())
    }

    /// Get recent entities
    fn get_recent_entities(
        &self,
        context: &ConversationContext,
        count: usize,
    ) -> Option<Vec<String>> {
        let mut entities: Vec<_> = context.tracked_entities.values().collect();
        entities.sort_by_key(|e| &e.last_mention);
        entities.reverse();

        let recent: Vec<String> = entities
            .iter()
            .take(count)
            .map(|e| e.entity.clone())
            .collect();

        if recent.is_empty() {
            None
        } else {
            Some(recent)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anaphora_resolution() {
        let generator = ContextAwareGenerator::new(ContextAwareConfig::default());
        let mut context = ConversationContext::default();

        context.tracked_entities.insert(
            "Inception".to_string(),
            TrackedEntity {
                entity: "Inception".to_string(),
                entity_type: "Movie".to_string(),
                first_mention: "msg1".to_string(),
                last_mention: "msg1".to_string(),
                mention_count: 1,
                resolved_uri: None,
            },
        );

        let resolved = generator
            .resolve_anaphora("Tell me more about it", &context)
            .unwrap();
        assert!(resolved.contains("Inception"));
    }

    #[test]
    fn test_entity_tracking() {
        let generator = ContextAwareGenerator::new(ContextAwareConfig::default());
        let mut context = ConversationContext::default();

        let entities = vec!["Movie".to_string(), "Director".to_string()];
        generator.update_tracked_entities(&mut context, &entities, "msg1");

        assert_eq!(context.tracked_entities.len(), 2);
        assert_eq!(
            context.tracked_entities.get("Movie").unwrap().mention_count,
            1
        );
    }

    #[test]
    fn test_count_query_generation() {
        let generator = ContextAwareGenerator::new(ContextAwareConfig::default());
        let context = ConversationContext::default();

        let sparql = generator
            .generate_count_query("How many Movies are there?", &context)
            .unwrap();

        println!("Generated SPARQL: {}", sparql);
        assert!(sparql.contains("COUNT"));
        // Entity extraction may produce lowercase "movie" or "movies"
        assert!(sparql.to_lowercase().contains("movie"));
    }
}
