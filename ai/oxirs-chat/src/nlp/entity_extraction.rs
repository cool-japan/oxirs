//! Entity Extraction System
//!
//! Extracts named entities and concepts from user messages to improve query understanding.

use crate::utils::nlp::NamedEntityRecognizer;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

/// Entity types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    /// Person name
    Person,
    /// Organization
    Organization,
    /// Location
    Location,
    /// Date/Time
    DateTime,
    /// Number
    Number,
    /// URL/URI
    URL,
    /// Email
    Email,
    /// RDF Resource
    RDFResource,
    /// Property/Predicate
    Property,
    /// Class/Type
    Class,
    /// Other
    Other,
}

/// Extracted entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    /// Entity text
    pub text: String,
    /// Entity type
    pub entity_type: EntityType,
    /// Start position in original text
    pub start: usize,
    /// End position in original text
    pub end: usize,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Resolved URI (for RDF resources)
    pub resolved_uri: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Entity extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityExtractionConfig {
    /// Enable person name extraction
    pub extract_persons: bool,
    /// Enable organization extraction
    pub extract_organizations: bool,
    /// Enable location extraction
    pub extract_locations: bool,
    /// Enable date/time extraction
    pub extract_datetime: bool,
    /// Enable RDF resource detection
    pub extract_rdf_resources: bool,
    /// Minimum confidence threshold
    pub min_confidence: f32,
    /// Enable entity linking
    pub enable_linking: bool,
}

impl Default for EntityExtractionConfig {
    fn default() -> Self {
        Self {
            extract_persons: true,
            extract_organizations: true,
            extract_locations: true,
            extract_datetime: true,
            extract_rdf_resources: true,
            min_confidence: 0.6,
            enable_linking: true,
        }
    }
}

/// Entity extractor
pub struct EntityExtractor {
    config: EntityExtractionConfig,
    ner_model: Option<NamedEntityRecognizer>,
    patterns: HashMap<EntityType, Vec<regex::Regex>>,
}

impl EntityExtractor {
    /// Create a new entity extractor
    pub fn new(config: EntityExtractionConfig) -> Result<Self> {
        let patterns = Self::build_patterns()?;

        info!(
            "Initialized entity extractor with {} pattern types",
            patterns.len()
        );

        Ok(Self {
            config,
            ner_model: None,
            patterns,
        })
    }

    /// Build regex patterns for entity extraction
    fn build_patterns() -> Result<HashMap<EntityType, Vec<regex::Regex>>> {
        let mut patterns = HashMap::new();

        // URL pattern
        patterns.insert(
            EntityType::URL,
            vec![regex::Regex::new(r"https?://[^\s]+")?],
        );

        // Email pattern
        patterns.insert(
            EntityType::Email,
            vec![regex::Regex::new(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            )?],
        );

        // Number pattern
        patterns.insert(
            EntityType::Number,
            vec![
                regex::Regex::new(r"\b\d+\b")?,
                regex::Regex::new(r"\b\d+\.\d+\b")?,
            ],
        );

        // Date patterns (simplified)
        patterns.insert(
            EntityType::DateTime,
            vec![
                regex::Regex::new(r"\b\d{4}-\d{2}-\d{2}\b")?, // YYYY-MM-DD
                regex::Regex::new(r"\b\d{1,2}/\d{1,2}/\d{4}\b")?, // MM/DD/YYYY
            ],
        );

        // RDF URI pattern
        patterns.insert(
            EntityType::RDFResource,
            vec![
                regex::Regex::new(r"<[^>]+>")?, // <http://example.org/resource>
                regex::Regex::new(r"[a-z]+:[A-Za-z0-9_-]+")?, // prefix:localName
            ],
        );

        Ok(patterns)
    }

    /// Extract entities from text
    pub fn extract(&self, text: &str) -> Result<Vec<ExtractedEntity>> {
        debug!(
            "Extracting entities from: {}",
            text.chars().take(100).collect::<String>()
        );

        let mut entities = Vec::new();

        // Pattern-based extraction
        for (entity_type, regexes) in &self.patterns {
            for regex in regexes {
                for capture in regex.find_iter(text) {
                    let entity_text = capture.as_str().to_string();
                    let start = capture.start();
                    let end = capture.end();

                    entities.push(ExtractedEntity {
                        text: entity_text.clone(),
                        entity_type: *entity_type,
                        start,
                        end,
                        confidence: 0.9, // High confidence for regex matches
                        resolved_uri: self.resolve_uri(&entity_text, *entity_type),
                        metadata: HashMap::new(),
                    });
                }
            }
        }

        // Capitalize word detection (potential proper nouns)
        if self.config.extract_persons || self.config.extract_organizations {
            entities.extend(self.extract_capitalized_entities(text));
        }

        // Filter by confidence
        entities.retain(|e| e.confidence >= self.config.min_confidence);

        // Sort by position
        entities.sort_by_key(|e| e.start);

        debug!("Extracted {} entities", entities.len());

        Ok(entities)
    }

    /// Extract entities from capitalized words
    fn extract_capitalized_entities(&self, text: &str) -> Vec<ExtractedEntity> {
        let mut entities = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut pos = 0;

        for word in words {
            let start = text[pos..].find(word).map(|p| p + pos).unwrap_or(pos);
            let end = start + word.len();
            pos = end;

            // Check if word starts with capital letter and is not at sentence start
            if word
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false)
                && word.len() > 2
                && start > 0
            {
                // Determine if it's a person or organization (simplified heuristic)
                let entity_type =
                    if word.ends_with("Inc") || word.ends_with("Corp") || word.ends_with("Ltd") {
                        EntityType::Organization
                    } else {
                        EntityType::Person
                    };

                entities.push(ExtractedEntity {
                    text: word.to_string(),
                    entity_type,
                    start,
                    end,
                    confidence: 0.6, // Lower confidence for heuristic detection
                    resolved_uri: None,
                    metadata: HashMap::new(),
                });
            }
        }

        entities
    }

    /// Resolve URI for RDF resources
    fn resolve_uri(&self, text: &str, entity_type: EntityType) -> Option<String> {
        if !self.config.enable_linking {
            return None;
        }

        match entity_type {
            EntityType::RDFResource => {
                // If it's already a URI, return as-is
                if text.starts_with('<') && text.ends_with('>') {
                    Some(text[1..text.len() - 1].to_string())
                } else if text.contains(':') {
                    // Expand prefix (simplified - would use actual prefix mapping)
                    Some(format!("http://example.org/{}", text))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Extract and group related entities
    pub fn extract_with_relations(
        &self,
        text: &str,
    ) -> Result<HashMap<EntityType, Vec<ExtractedEntity>>> {
        let entities = self.extract(text)?;

        let mut grouped: HashMap<EntityType, Vec<ExtractedEntity>> = HashMap::new();
        for entity in entities {
            grouped.entry(entity.entity_type).or_default().push(entity);
        }

        Ok(grouped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_extraction() {
        let extractor = EntityExtractor::new(EntityExtractionConfig::default()).unwrap();
        let entities = extractor
            .extract("Check out https://example.org for more info")
            .unwrap();

        assert!(entities.iter().any(|e| e.entity_type == EntityType::URL));
    }

    #[test]
    fn test_email_extraction() {
        let extractor = EntityExtractor::new(EntityExtractionConfig::default()).unwrap();
        let entities = extractor
            .extract("Contact us at support@example.com")
            .unwrap();

        assert!(entities.iter().any(|e| e.entity_type == EntityType::Email));
    }

    #[test]
    fn test_number_extraction() {
        let extractor = EntityExtractor::new(EntityExtractionConfig::default()).unwrap();
        let entities = extractor
            .extract("There are 42 items in the database")
            .unwrap();

        assert!(entities.iter().any(|e| e.entity_type == EntityType::Number));
    }

    #[test]
    fn test_rdf_resource_extraction() {
        let extractor = EntityExtractor::new(EntityExtractionConfig::default()).unwrap();
        let entities = extractor
            .extract("Query for schema:Person resources")
            .unwrap();

        assert!(entities
            .iter()
            .any(|e| e.entity_type == EntityType::RDFResource));
    }
}
