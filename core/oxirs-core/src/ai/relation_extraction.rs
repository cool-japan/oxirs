//! Relation Extraction from Text using NLP
//!
//! This module provides automated relation extraction capabilities to build
//! knowledge graphs from unstructured text data.

use crate::ai::AiConfig;
use crate::model::{Literal, NamedNode, Triple};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Relation extraction module
pub struct RelationExtractor {
    /// Configuration
    config: ExtractionConfig,

    /// Named Entity Recognition model
    ner_model: Box<dyn NamedEntityRecognizer>,

    /// Relation classification model
    relation_model: Box<dyn RelationClassifier>,

    /// Entity linking module
    entity_linker: Box<dyn EntityLinker>,

    /// Confidence threshold
    confidence_threshold: f32,
}

/// Relation extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    /// Enable named entity recognition
    pub enable_ner: bool,

    /// Enable relation classification
    pub enable_relation_classification: bool,

    /// Enable entity linking
    pub enable_entity_linking: bool,

    /// Confidence threshold for extractions
    pub confidence_threshold: f32,

    /// Maximum sentence length
    pub max_sentence_length: usize,

    /// Language model to use
    pub language_model: String,

    /// Enable coreference resolution
    pub enable_coreference: bool,

    /// Supported languages
    pub supported_languages: Vec<String>,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            enable_ner: true,
            enable_relation_classification: true,
            enable_entity_linking: true,
            confidence_threshold: 0.7,
            max_sentence_length: 512,
            language_model: "bert-base-uncased".to_string(),
            enable_coreference: true,
            supported_languages: vec!["en".to_string()],
        }
    }
}

/// Extracted relation from text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedRelation {
    /// Subject entity
    pub subject: ExtractedEntity,

    /// Predicate/relation type
    pub predicate: String,

    /// Object entity
    pub object: ExtractedEntity,

    /// Confidence score
    pub confidence: f32,

    /// Source text span
    pub source_span: TextSpan,

    /// Context sentence
    pub context: String,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Extracted entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    /// Entity text
    pub text: String,

    /// Entity type
    pub entity_type: EntityType,

    /// Linked knowledge base ID (if available)
    pub kb_id: Option<String>,

    /// Confidence score
    pub confidence: f32,

    /// Text span in original document
    pub span: TextSpan,
}

/// Entity types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntityType {
    Person,
    Organization,
    Location,
    Date,
    Time,
    Money,
    Percent,
    Product,
    Event,
    Concept,
    Other(String),
}

/// Text span
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSpan {
    /// Start position
    pub start: usize,

    /// End position
    pub end: usize,

    /// Text content
    pub text: String,
}

/// Named Entity Recognition trait
pub trait NamedEntityRecognizer: Send + Sync {
    /// Extract named entities from text
    fn extract_entities(&self, text: &str) -> Result<Vec<ExtractedEntity>>;

    /// Get supported entity types
    fn supported_types(&self) -> Vec<EntityType>;
}

/// Relation classification trait
pub trait RelationClassifier: Send + Sync {
    /// Classify relation between two entities
    fn classify_relation(
        &self,
        text: &str,
        subject: &ExtractedEntity,
        object: &ExtractedEntity,
    ) -> Result<Option<(String, f32)>>;

    /// Get supported relation types
    fn supported_relations(&self) -> Vec<String>;
}

/// Entity linking trait
pub trait EntityLinker: Send + Sync {
    /// Link entity to knowledge base
    fn link_entity(&self, entity: &ExtractedEntity, context: &str) -> Result<Option<String>>;

    /// Get knowledge base info
    fn kb_info(&self) -> KnowledgeBaseInfo;
}

/// Knowledge base information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeBaseInfo {
    /// Knowledge base name
    pub name: String,

    /// Base URI
    pub base_uri: String,

    /// Version
    pub version: String,

    /// Entity count
    pub entity_count: usize,
}

impl RelationExtractor {
    /// Create a new relation extractor backed by the built-in **heuristic**
    /// (rule-based) NER, relation classifier, and entity linker.
    ///
    /// These heuristic backends are transparent, deterministic, and honest: they
    /// do NOT emulate a trained ML model, they do not fabricate confidence scores
    /// dressed up as model probabilities, and the entity linker does not invent
    /// unverified knowledge-base URIs. Every relation they produce is tagged with
    /// `metadata["extraction_method"] = "heuristic-keyword-match"` so downstream
    /// consumers can distinguish heuristic output from a real model's.
    ///
    /// To use a real ML backend (BERT-NER, a trained relation classifier, a live
    /// entity-linking service, etc.), construct real trait objects and pass them
    /// to [`with_backends`](Self::with_backends).
    ///
    /// `_config` is currently unused by the heuristic backends (they have no
    /// tunable model parameters); it is retained for API symmetry with the
    /// backend-injecting constructor.
    pub fn new(_config: &AiConfig) -> Result<Self> {
        Ok(Self {
            config: ExtractionConfig::default(),
            ner_model: Box::new(HeuristicNer::new()),
            relation_model: Box::new(HeuristicRelationClassifier::new()),
            entity_linker: Box::new(LocalEntityLinker::new()),
            confidence_threshold: 0.7,
        })
    }

    /// Create a relation extractor with caller-provided backends.
    ///
    /// This is the path for wiring in a real NER model, relation classifier, and
    /// entity linker. The `extraction_config`'s `confidence_threshold` is used to
    /// filter extracted relations.
    pub fn with_backends(
        extraction_config: ExtractionConfig,
        ner_model: Box<dyn NamedEntityRecognizer>,
        relation_model: Box<dyn RelationClassifier>,
        entity_linker: Box<dyn EntityLinker>,
    ) -> Self {
        let confidence_threshold = extraction_config.confidence_threshold;
        Self {
            config: extraction_config,
            ner_model,
            relation_model,
            entity_linker,
            confidence_threshold,
        }
    }

    /// Extract relations from text
    pub async fn extract_relations(&self, text: &str) -> Result<Vec<ExtractedRelation>> {
        // Step 1: Sentence segmentation
        let sentences = self.segment_sentences(text);

        let mut all_relations = Vec::new();

        for sentence in sentences {
            // Step 2: Named Entity Recognition
            let entities = if self.config.enable_ner {
                self.ner_model.extract_entities(&sentence)?
            } else {
                Vec::new()
            };

            // Step 3: Entity Linking
            let linked_entities = if self.config.enable_entity_linking {
                self.link_entities(&entities, &sentence).await?
            } else {
                entities
            };

            // Step 4: Relation Classification
            if self.config.enable_relation_classification {
                let relations =
                    self.extract_relations_from_entities(&sentence, &linked_entities)?;
                all_relations.extend(relations);
            }
        }

        // Step 5: Filter by confidence
        let filtered_relations = all_relations
            .into_iter()
            .filter(|r| r.confidence >= self.confidence_threshold)
            .collect();

        Ok(filtered_relations)
    }

    /// Convert extracted relations to RDF triples
    pub fn to_triples(&self, relations: &[ExtractedRelation]) -> Result<Vec<Triple>> {
        let mut triples = Vec::new();

        for relation in relations {
            // Create subject
            let subject = if let Some(kb_id) = &relation.subject.kb_id {
                NamedNode::new(kb_id)?
            } else {
                // Use text as identifier (simplified)
                NamedNode::new(format!(
                    "http://example.org/entity/{}",
                    relation.subject.text.replace(' ', "_")
                ))?
            };

            // Create predicate
            let predicate = NamedNode::new(format!(
                "http://example.org/relation/{}",
                relation.predicate.replace(' ', "_")
            ))?;

            // Create object
            let object = if let Some(kb_id) = &relation.object.kb_id {
                crate::model::Object::NamedNode(NamedNode::new(kb_id)?)
            } else {
                // Determine if it's a literal or named node
                match relation.object.entity_type {
                    EntityType::Date
                    | EntityType::Time
                    | EntityType::Money
                    | EntityType::Percent => {
                        crate::model::Object::Literal(Literal::new(&relation.object.text))
                    }
                    _ => crate::model::Object::NamedNode(NamedNode::new(format!(
                        "http://example.org/entity/{}",
                        relation.object.text.replace(' ', "_")
                    ))?),
                }
            };

            let triple = Triple::new(subject, predicate, object);
            triples.push(triple);
        }

        Ok(triples)
    }

    /// Segment text into sentences
    fn segment_sentences(&self, text: &str) -> Vec<String> {
        // Simplified sentence segmentation
        text.split(". ")
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Link entities to knowledge base
    async fn link_entities(
        &self,
        entities: &[ExtractedEntity],
        context: &str,
    ) -> Result<Vec<ExtractedEntity>> {
        let mut linked_entities = Vec::new();

        for entity in entities {
            let mut linked_entity = entity.clone();

            // Propagate linker errors (fail-loud) rather than silently dropping
            // them; a `None` result simply means "no confident KB link".
            if let Some(kb_id) = self.entity_linker.link_entity(entity, context)? {
                linked_entity.kb_id = Some(kb_id);
            }

            linked_entities.push(linked_entity);
        }

        Ok(linked_entities)
    }

    /// Extract relations from entities in a sentence
    fn extract_relations_from_entities(
        &self,
        sentence: &str,
        entities: &[ExtractedEntity],
    ) -> Result<Vec<ExtractedRelation>> {
        let mut relations = Vec::new();

        // Try all pairs of entities
        for (i, subject) in entities.iter().enumerate() {
            for (j, object) in entities.iter().enumerate() {
                if i != j {
                    // Propagate classifier errors (fail-loud) instead of silently
                    // swallowing them.
                    if let Some((relation_type, confidence)) = self
                        .relation_model
                        .classify_relation(sentence, subject, object)?
                    {
                        let mut metadata = HashMap::new();
                        metadata.insert(
                            "extraction_method".to_string(),
                            "heuristic-keyword-match".to_string(),
                        );
                        let relation = ExtractedRelation {
                            subject: subject.clone(),
                            predicate: relation_type,
                            object: object.clone(),
                            confidence,
                            source_span: TextSpan {
                                start: 0,
                                end: sentence.len(),
                                text: sentence.to_string(),
                            },
                            context: sentence.to_string(),
                            metadata,
                        };

                        relations.push(relation);
                    }
                }
            }
        }

        Ok(relations)
    }
}

/// Known organization name/suffix tokens for the heuristic NER gazetteer.
const ORG_GAZETTEER: &[&str] = &[
    "Inc",
    "Inc.",
    "Corp",
    "Corp.",
    "Corporation",
    "Ltd",
    "Ltd.",
    "LLC",
    "Company",
    "GmbH",
    "Microsoft",
    "Google",
    "Apple",
    "Amazon",
    "IBM",
    "Oracle",
    "Meta",
    "Intel",
    "Nvidia",
];

/// Known location tokens for the heuristic NER gazetteer.
const LOCATION_GAZETTEER: &[&str] = &[
    "Seattle",
    "London",
    "Paris",
    "Tokyo",
    "Berlin",
    "Washington",
    "California",
    "France",
    "Germany",
    "Japan",
    "China",
    "India",
    "Boston",
    "Chicago",
    "Amsterdam",
    "Madrid",
];

/// Heuristic (rule-based) named-entity recognizer.
///
/// This is deliberately NOT a trained model. It detects capitalized tokens as
/// candidate entities, computes **real** byte offsets into the source text, and
/// assigns a type only when a small gazetteer provides a real signal; otherwise
/// the type is honestly reported as [`EntityType::Other`]`("Unknown")` rather
/// than fabricating a specific class. Confidence reflects the heuristic's low
/// certainty, not a model probability.
struct HeuristicNer;

impl HeuristicNer {
    fn new() -> Self {
        Self
    }

    fn classify_token(token: &str) -> (EntityType, f32) {
        if LOCATION_GAZETTEER
            .iter()
            .any(|g| g.eq_ignore_ascii_case(token))
        {
            (EntityType::Location, 0.7)
        } else if ORG_GAZETTEER.iter().any(|g| g.eq_ignore_ascii_case(token)) {
            (EntityType::Organization, 0.7)
        } else {
            // No real signal about the class — do not fabricate "Person".
            (EntityType::Other("Unknown".to_string()), 0.5)
        }
    }
}

impl NamedEntityRecognizer for HeuristicNer {
    fn extract_entities(&self, text: &str) -> Result<Vec<ExtractedEntity>> {
        let mut entities = Vec::new();
        let mut search_start = 0usize;

        for word in text.split_whitespace() {
            // Locate this token's real byte offset in the source text.
            let start = match text[search_start..].find(word) {
                Some(rel) => search_start + rel,
                None => continue,
            };
            search_start = start + word.len();

            // Trim surrounding punctuation for the entity surface form.
            let trimmed = word.trim_matches(|c: char| !c.is_alphanumeric());
            if trimmed.is_empty() {
                continue;
            }
            let first = trimmed.chars().next().unwrap_or(' ');
            if !first.is_uppercase() {
                continue;
            }

            // Real offset of the trimmed token within the raw whitespace token.
            let inner_offset = word.find(trimmed).unwrap_or(0);
            let token_start = start + inner_offset;
            let token_end = token_start + trimmed.len();

            let (entity_type, confidence) = Self::classify_token(trimmed);
            entities.push(ExtractedEntity {
                text: trimmed.to_string(),
                entity_type,
                kb_id: None,
                confidence,
                span: TextSpan {
                    start: token_start,
                    end: token_end,
                    text: trimmed.to_string(),
                },
            });
        }

        Ok(entities)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        vec![
            EntityType::Organization,
            EntityType::Location,
            EntityType::Other("Unknown".to_string()),
        ]
    }
}

/// Heuristic (rule-based) relation classifier.
///
/// Matches surface keywords in the sentence. The returned confidences are honest
/// heuristic scores (keyword-match strength), not model probabilities, and
/// callers should treat the produced relations as heuristic (they are tagged with
/// `metadata["extraction_method"] = "heuristic-keyword-match"`).
struct HeuristicRelationClassifier;

impl HeuristicRelationClassifier {
    fn new() -> Self {
        Self
    }
}

impl RelationClassifier for HeuristicRelationClassifier {
    fn classify_relation(
        &self,
        text: &str,
        _subject: &ExtractedEntity,
        _object: &ExtractedEntity,
    ) -> Result<Option<(String, f32)>> {
        if text.contains("work") || text.contains("employ") {
            Ok(Some(("worksFor".to_string(), 0.75)))
        } else if text.contains("live") || text.contains("reside") {
            Ok(Some(("livesIn".to_string(), 0.75)))
        } else if text.contains("born") || text.contains("birth") {
            Ok(Some(("bornIn".to_string(), 0.75)))
        } else {
            Ok(None)
        }
    }

    fn supported_relations(&self) -> Vec<String> {
        vec![
            "worksFor".to_string(),
            "livesIn".to_string(),
            "bornIn".to_string(),
        ]
    }
}

/// Entity linker that performs **no** knowledge-base lookup.
///
/// A genuine entity linker requires a knowledge base to resolve and verify
/// against (e.g. DBpedia Spotlight, Wikidata). Without one, fabricating a
/// `http://dbpedia.org/resource/<text>` URI would assert the existence of a KB
/// resource that was never verified. This linker therefore returns `None` (no
/// confident link) for every entity, which is the honest result. Inject a real
/// [`EntityLinker`] via [`RelationExtractor::with_backends`] for actual linking.
struct LocalEntityLinker;

impl LocalEntityLinker {
    fn new() -> Self {
        Self
    }
}

impl EntityLinker for LocalEntityLinker {
    fn link_entity(&self, _entity: &ExtractedEntity, _context: &str) -> Result<Option<String>> {
        // No knowledge base is available; do not fabricate an unverified URI.
        Ok(None)
    }

    fn kb_info(&self) -> KnowledgeBaseInfo {
        KnowledgeBaseInfo {
            name: "none (no knowledge base configured)".to_string(),
            base_uri: String::new(),
            version: "n/a".to_string(),
            entity_count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::AiConfig;

    #[tokio::test]
    async fn test_relation_extractor_creation() {
        let config = AiConfig::default();
        let extractor = RelationExtractor::new(&config);
        assert!(extractor.is_ok());
    }

    #[tokio::test]
    async fn test_relation_extraction() {
        let config = AiConfig::default();
        let extractor = RelationExtractor::new(&config).expect("construction should succeed");

        let text = "John works for Microsoft. He lives in Seattle.";
        let relations = extractor
            .extract_relations(text)
            .await
            .expect("async operation should succeed");

        // Should extract some relations (depends on dummy implementation)
        assert!(!relations.is_empty());
    }

    #[test]
    fn test_sentence_segmentation() {
        let config = AiConfig::default();
        let extractor = RelationExtractor::new(&config).expect("construction should succeed");

        let text = "First sentence. Second sentence. Third sentence.";
        let sentences = extractor.segment_sentences(text);

        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "First sentence");
    }

    #[test]
    fn test_to_triples() {
        let config = AiConfig::default();
        let extractor = RelationExtractor::new(&config).expect("construction should succeed");

        let relation = ExtractedRelation {
            subject: ExtractedEntity {
                text: "John".to_string(),
                entity_type: EntityType::Person,
                kb_id: None,
                confidence: 0.9,
                span: TextSpan {
                    start: 0,
                    end: 4,
                    text: "John".to_string(),
                },
            },
            predicate: "worksFor".to_string(),
            object: ExtractedEntity {
                text: "Microsoft".to_string(),
                entity_type: EntityType::Organization,
                kb_id: None,
                confidence: 0.85,
                span: TextSpan {
                    start: 15,
                    end: 24,
                    text: "Microsoft".to_string(),
                },
            },
            confidence: 0.8,
            source_span: TextSpan {
                start: 0,
                end: 25,
                text: "John works for Microsoft.".to_string(),
            },
            context: "John works for Microsoft.".to_string(),
            metadata: HashMap::new(),
        };

        let triples = extractor
            .to_triples(&[relation])
            .expect("operation should succeed");
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn regression_entity_linker_does_not_fabricate_dbpedia_uris() {
        let linker = LocalEntityLinker::new();
        let entity = ExtractedEntity {
            text: "John".to_string(),
            entity_type: EntityType::Person,
            kb_id: None,
            confidence: 0.5,
            span: TextSpan {
                start: 0,
                end: 4,
                text: "John".to_string(),
            },
        };
        // Must NOT invent an unverified http://dbpedia.org/resource/John URI.
        let linked = linker.link_entity(&entity, "context").expect("link");
        assert_eq!(linked, None);

        // kb_info must not claim to be a populated DBpedia.
        let info = linker.kb_info();
        assert_eq!(info.entity_count, 0);
        assert!(!info.base_uri.contains("dbpedia"));
    }

    #[test]
    fn regression_ner_reports_real_byte_offsets() {
        let ner = HeuristicNer::new();
        let text = "John works for Microsoft";
        let entities = ner.extract_entities(text).expect("ner");

        // Every reported span must correspond to the actual substring in `text`.
        assert!(!entities.is_empty());
        for entity in &entities {
            assert_eq!(&text[entity.span.start..entity.span.end], entity.span.text);
            assert_eq!(entity.span.text, entity.text);
        }

        // "Microsoft" is in the org gazetteer -> classified as Organization,
        // and its offset must be the real position (15), not a fabricated i*5.
        let microsoft = entities
            .iter()
            .find(|e| e.text == "Microsoft")
            .expect("Microsoft detected");
        assert_eq!(microsoft.span.start, 15);
        assert!(matches!(microsoft.entity_type, EntityType::Organization));

        // "John" has no gazetteer signal -> honestly typed Other, not Person.
        let john = entities
            .iter()
            .find(|e| e.text == "John")
            .expect("John detected");
        assert!(matches!(john.entity_type, EntityType::Other(_)));
    }

    #[tokio::test]
    async fn regression_extracted_relations_tagged_as_heuristic() {
        let config = AiConfig::default();
        let extractor = RelationExtractor::new(&config).expect("construction");
        let relations = extractor
            .extract_relations("John works for Microsoft")
            .await
            .expect("extract");
        assert!(!relations.is_empty());
        for relation in &relations {
            assert_eq!(
                relation
                    .metadata
                    .get("extraction_method")
                    .map(String::as_str),
                Some("heuristic-keyword-match")
            );
        }
    }
}
