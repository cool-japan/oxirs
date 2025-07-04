//! LLM-powered entity and relationship extraction
//!
//! Provides intelligent entity and relationship extraction from natural language queries
//! using both LLM-based and rule-based approaches.

use super::*;

/// Entity extractor for identifying entities and relationships in queries
pub struct EntityExtractor;

impl EntityExtractor {
    pub fn new() -> Self {
        Self
    }

    /// Extract entities and relationships from query
    pub async fn extract_entities_and_relationships(
        &self,
        query: &str,
    ) -> Result<(Vec<ExtractedEntity>, Vec<ExtractedRelationship>)> {
        // Try LLM extraction first, fall back to rule-based if needed
        if let Ok(result) = self.llm_extract_entities(query).await {
            Ok(result)
        } else {
            warn!("LLM extraction failed, falling back to rule-based extraction");
            self.rule_based_extraction(query).await
        }
    }

    /// LLM-powered entity and relationship extraction
    async fn llm_extract_entities(
        &self,
        query: &str,
    ) -> Result<(Vec<ExtractedEntity>, Vec<ExtractedRelationship>)> {
        use crate::llm::{
            ChatMessage, ChatRole, LLMConfig, LLMManager, LLMRequest, Priority, UseCase,
        };

        // Create extraction prompt
        let prompt = format!(
            r#"Extract entities and relationships from the following query. Return a JSON response with the following structure:

{{
  "entities": [
    {{
      "text": "entity name",
      "type": "Person|Organization|Location|Concept|Other",
      "confidence": 0.95
    }}
  ],
  "relationships": [
    {{
      "subject": "entity1",
      "predicate": "relationship type",
      "object": "entity2",
      "confidence": 0.85
    }}
  ]
}}

Query: "{}"

Focus on:
- Named entities (people, places, organizations, concepts)
- Implicit relationships between entities
- Technical terms and domain-specific concepts
- Only extract explicit entities mentioned in the query

JSON Response:"#,
            query
        );

        // Initialize LLM manager
        let llm_config = LLMConfig::default();
        let mut llm_manager = LLMManager::new(llm_config)?;

        let chat_messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: "You are an expert at extracting entities and relationships from text. Always respond with valid JSON only.".to_string(),
                metadata: None,
            },
            ChatMessage {
                role: ChatRole::User,
                content: prompt,
                metadata: None,
            },
        ];

        let request = LLMRequest {
            messages: chat_messages,
            system_prompt: Some("Extract entities and relationships as JSON.".to_string()),
            use_case: UseCase::SimpleQuery,
            priority: Priority::Normal,
            max_tokens: Some(500),
            temperature: 0.1f32, // Low temperature for consistent extraction
            timeout: Some(std::time::Duration::from_secs(15)),
        };

        let response = llm_manager.generate_response(request).await?;

        // Parse JSON response
        self.parse_extraction_response(&response.content)
    }

    /// Parse LLM extraction response
    fn parse_extraction_response(
        &self,
        response: &str,
    ) -> Result<(Vec<ExtractedEntity>, Vec<ExtractedRelationship>)> {
        // Clean response (remove markdown formatting if present)
        let cleaned_response = response
            .trim()
            .strip_prefix("```json")
            .unwrap_or(response)
            .strip_suffix("```")
            .unwrap_or(response)
            .trim();

        let parsed: serde_json::Value = serde_json::from_str(cleaned_response)?;

        let mut entities = Vec::new();
        let mut relationships = Vec::new();

        // Parse entities
        if let Some(entity_array) = parsed.get("entities").and_then(|e| e.as_array()) {
            for entity_obj in entity_array {
                if let (Some(text), Some(entity_type), Some(confidence)) = (
                    entity_obj.get("text").and_then(|v| v.as_str()),
                    entity_obj.get("type").and_then(|v| v.as_str()),
                    entity_obj.get("confidence").and_then(|v| v.as_f64()),
                ) {
                    let entity_type = match entity_type {
                        "Person" => EntityType::Person,
                        "Organization" => EntityType::Organization,
                        "Location" => EntityType::Location,
                        "Concept" => EntityType::Concept,
                        "Event" => EntityType::Event,
                        _ => EntityType::Other,
                    };

                    entities.push(ExtractedEntity {
                        text: text.to_string(),
                        entity_type,
                        iri: None, // Would be resolved separately
                        confidence: confidence as f32,
                        aliases: Vec::new(),
                    });
                }
            }
        }

        // Parse relationships
        if let Some(relationship_array) = parsed.get("relationships").and_then(|r| r.as_array()) {
            for rel_obj in relationship_array {
                if let (Some(subject), Some(predicate), Some(object), Some(confidence)) = (
                    rel_obj.get("subject").and_then(|v| v.as_str()),
                    rel_obj.get("predicate").and_then(|v| v.as_str()),
                    rel_obj.get("object").and_then(|v| v.as_str()),
                    rel_obj.get("confidence").and_then(|v| v.as_f64()),
                ) {
                    relationships.push(ExtractedRelationship {
                        subject: subject.to_string(),
                        predicate: predicate.to_string(),
                        object: object.to_string(),
                        confidence: confidence as f32,
                        relation_type: RelationType::Other, // Would be classified separately
                    });
                }
            }
        }

        debug!(
            "Extracted {} entities and {} relationships",
            entities.len(),
            relationships.len()
        );
        Ok((entities, relationships))
    }

    /// Fallback rule-based extraction
    async fn rule_based_extraction(
        &self,
        query: &str,
    ) -> Result<(Vec<ExtractedEntity>, Vec<ExtractedRelationship>)> {
        let mut entities = Vec::new();
        let mut relationships = Vec::new();

        // Simple pattern-based entity extraction
        let words: Vec<&str> = query.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            // Look for capitalized words (potential proper nouns)
            if word.chars().next().map_or(false, |c| c.is_uppercase()) && word.len() > 2 {
                // Skip common question words
                if !self.is_stop_word(&word.to_lowercase()) {
                    entities.push(ExtractedEntity {
                        text: word.to_string(),
                        entity_type: EntityType::Other,
                        iri: None,
                        confidence: 0.6, // Lower confidence for rule-based
                        aliases: Vec::new(),
                    });
                }
            }

            // Look for relationship patterns
            if i > 0 && i < words.len() - 1 {
                let prev_word = words[i - 1];
                let next_word = words[i + 1];

                if word.to_lowercase() == "is" || word.to_lowercase() == "has" {
                    relationships.push(ExtractedRelationship {
                        subject: prev_word.to_string(),
                        predicate: word.to_string(),
                        object: next_word.to_string(),
                        confidence: 0.5,
                        relation_type: RelationType::ConceptualRelation,
                    });
                }
            }
        }

        debug!(
            "Rule-based extraction found {} entities and {} relationships",
            entities.len(),
            relationships.len()
        );
        Ok((entities, relationships))
    }

    /// Check if a word is a stop word (for entity extraction)
    fn is_stop_word(&self, word: &str) -> bool {
        matches!(
            word,
            "the"
                | "and"
                | "or"
                | "but"
                | "in"
                | "on"
                | "at"
                | "to"
                | "for"
                | "of"
                | "with"
                | "by"
                | "from"
                | "up"
                | "about"
                | "into"
                | "through"
                | "during"
                | "before"
                | "after"
                | "above"
                | "below"
                | "between"
                | "among"
                | "this"
                | "that"
                | "these"
                | "those"
                | "i"
                | "you"
                | "he"
                | "she"
                | "it"
                | "we"
                | "they"
                | "me"
                | "him"
                | "her"
                | "us"
                | "them"
                | "my"
                | "your"
                | "his"
                | "its"
                | "our"
                | "their"
                | "am"
                | "is"
                | "are"
                | "was"
                | "were"
                | "be"
                | "been"
                | "being"
                | "have"
                | "has"
                | "had"
                | "do"
                | "does"
                | "did"
                | "will"
                | "would"
                | "could"
                | "should"
                | "may"
                | "might"
                | "must"
                | "can"
                | "what"
                | "when"
                | "where"
                | "who"
                | "why"
                | "how"
                | "which"
        )
    }
}

/// LLM entity extraction result wrapper
pub struct LLMEntityExtraction {
    pub entities: Vec<ExtractedEntity>,
    pub relationships: Vec<ExtractedRelationship>,
}

use super::graph_traversal::{EntityType, ExtractedEntity, ExtractedRelationship, RelationType};
