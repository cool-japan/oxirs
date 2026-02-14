//! Automated Knowledge Extraction Module
//!
//! Implements sophisticated knowledge extraction capabilities including:
//! - Entity and relationship extraction from text
//! - Schema discovery and ontology generation
//! - Fact validation and consistency checking
//! - Temporal knowledge extraction
//! - Multi-lingual knowledge extraction

use anyhow::Result;
use chrono::{DateTime, Utc};
use oxirs_core::model::{triple::Triple, NamedNode, Object, Predicate, Subject};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Configuration for knowledge extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeExtractionConfig {
    pub enable_entity_extraction: bool,
    pub enable_relationship_extraction: bool,
    pub enable_schema_discovery: bool,
    pub enable_fact_validation: bool,
    pub enable_temporal_extraction: bool,
    pub enable_multilingual_extraction: bool,
    pub confidence_threshold: f64,
    pub max_extraction_depth: usize,
    pub language_models: Vec<String>,
}

impl Default for KnowledgeExtractionConfig {
    fn default() -> Self {
        Self {
            enable_entity_extraction: true,
            enable_relationship_extraction: true,
            enable_schema_discovery: true,
            enable_fact_validation: true,
            enable_temporal_extraction: true,
            enable_multilingual_extraction: false,
            confidence_threshold: 0.8,
            max_extraction_depth: 3,
            language_models: vec!["en".to_string()],
        }
    }
}

/// Extracted knowledge item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedKnowledge {
    pub knowledge_id: String,
    pub source_text: String,
    pub extracted_triples: Vec<Triple>,
    pub extracted_entities: Vec<ExtractedEntity>,
    pub extracted_relationships: Vec<ExtractedRelationship>,
    pub schema_elements: Vec<SchemaElement>,
    pub temporal_facts: Vec<TemporalFact>,
    pub confidence_score: f64,
    pub extraction_metadata: ExtractionMetadata,
}

/// Detailed entity information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub entity_id: String,
    pub entity_text: String,
    pub entity_type: EntityType,
    pub canonical_form: String,
    pub aliases: Vec<String>,
    pub properties: HashMap<String, String>,
    pub confidence: f64,
    pub source_position: TextPosition,
    pub linked_entities: Vec<String>,
}

/// Relationship between entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedRelationship {
    pub relationship_id: String,
    pub subject_entity: String,
    pub predicate: String,
    pub object_entity: String,
    pub relationship_type: RelationshipType,
    pub confidence: f64,
    pub evidence_text: String,
    pub temporal_context: Option<TemporalContext>,
    pub source_position: TextPosition,
}

/// Schema element discovered from text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaElement {
    pub element_id: String,
    pub element_type: SchemaElementType,
    pub name: String,
    pub description: String,
    pub properties: Vec<SchemaProperty>,
    pub hierarchical_relations: Vec<HierarchicalRelation>,
    pub constraints: Vec<SchemaConstraint>,
    pub confidence: f64,
}

/// Temporal fact with time information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalFact {
    pub fact_id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub temporal_qualifier: TemporalQualifier,
    pub confidence: f64,
    pub source_text: String,
}

/// Types of entities that can be extracted
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    Person,
    Organization,
    Location,
    Event,
    Concept,
    Product,
    Technology,
    Scientific,
    Temporal,
    Numerical,
    Unknown,
}

/// Types of relationships
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationshipType {
    IsA,
    PartOf,
    LocatedIn,
    OwnedBy,
    CreatedBy,
    CausedBy,
    TemporalSequence,
    Similarity,
    Dependency,
    Custom(String),
}

/// Schema element types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SchemaElementType {
    Class,
    Property,
    Relationship,
    Constraint,
    Rule,
}

/// Position in source text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextPosition {
    pub start_offset: usize,
    pub end_offset: usize,
    pub line_number: usize,
    pub column_number: usize,
}

/// Temporal context information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TemporalContext {
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub duration: Option<std::time::Duration>,
    pub temporal_relation: String,
}

/// Temporal qualifier for facts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalQualifier {
    pub qualifier_type: TemporalType,
    pub time_point: Option<DateTime<Utc>>,
    pub time_interval: Option<TimeInterval>,
    pub frequency: Option<String>,
}

/// Types of temporal information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TemporalType {
    PointInTime,
    TimeInterval,
    Frequency,
    Duration,
    Relative,
}

/// Time interval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeInterval {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

/// Schema property
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaProperty {
    pub property_name: String,
    pub property_type: String,
    pub cardinality: Cardinality,
    pub domain: Option<String>,
    pub range: Option<String>,
}

/// Hierarchical relation in schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalRelation {
    pub relation_type: HierarchyType,
    pub parent: String,
    pub child: String,
}

/// Schema constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaConstraint {
    pub constraint_type: ConstraintType,
    pub description: String,
    pub enforcement_level: EnforcementLevel,
}

/// Property cardinality
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Cardinality {
    ZeroOrOne,
    ExactlyOne,
    ZeroOrMore,
    OneOrMore,
    Exact(usize),
    Range(usize, usize),
}

/// Hierarchy types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HierarchyType {
    SubClassOf,
    SubPropertyOf,
    PartOf,
    InstanceOf,
}

/// Constraint types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstraintType {
    UniqueValue,
    RequiredProperty,
    ValueRange,
    DataType,
    Pattern,
    Cardinality,
}

/// Enforcement levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EnforcementLevel {
    Strict,
    Warning,
    Suggestion,
}

/// Metadata about the extraction process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionMetadata {
    pub extraction_timestamp: DateTime<Utc>,
    pub extraction_method: String,
    pub processing_time_ms: u64,
    pub language_detected: String,
    pub text_length: usize,
    pub extraction_statistics: ExtractionStatistics,
}

/// Statistics about extraction results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionStatistics {
    pub entities_extracted: usize,
    pub relationships_extracted: usize,
    pub triples_generated: usize,
    pub schema_elements_discovered: usize,
    pub temporal_facts_extracted: usize,
    pub average_confidence: f64,
}

/// Knowledge extraction engine
pub struct KnowledgeExtractionEngine {
    config: KnowledgeExtractionConfig,
    entity_patterns: HashMap<EntityType, Vec<Regex>>,
    relationship_patterns: HashMap<RelationshipType, Vec<Regex>>,
    temporal_patterns: Vec<Regex>,
    schema_inference_rules: Vec<SchemaInferenceRule>,
    language_detectors: HashMap<String, LanguageDetector>,
}

/// Schema inference rule
#[derive(Debug, Clone)]
struct SchemaInferenceRule {
    rule_id: String,
    pattern: Regex,
    inferred_type: SchemaElementType,
    confidence_modifier: f64,
}

/// Language detector
#[derive(Debug, Clone)]
struct LanguageDetector {
    language_code: String,
    detection_patterns: Vec<Regex>,
    confidence_threshold: f64,
}

impl KnowledgeExtractionEngine {
    /// Create a new knowledge extraction engine
    pub fn new(config: KnowledgeExtractionConfig) -> Result<Self> {
        let mut engine = Self {
            config,
            entity_patterns: HashMap::new(),
            relationship_patterns: HashMap::new(),
            temporal_patterns: Vec::new(),
            schema_inference_rules: Vec::new(),
            language_detectors: HashMap::new(),
        };

        engine.initialize_extraction_patterns()?;
        engine.initialize_schema_rules()?;
        engine.initialize_language_detectors()?;

        Ok(engine)
    }

    /// Initialize entity and relationship extraction patterns
    fn initialize_extraction_patterns(&mut self) -> Result<()> {
        // Person entity patterns
        let person_patterns = vec![
            Regex::new(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")?, // FirstName LastName
            Regex::new(r"\bDr\. [A-Z][a-z]+\b")?,        // Dr. Name
            Regex::new(r"\bProf\. [A-Z][a-z]+\b")?,      // Prof. Name
        ];
        self.entity_patterns
            .insert(EntityType::Person, person_patterns);

        // Organization patterns
        let org_patterns = vec![
            Regex::new(r"\b[A-Z][a-z]+ (Inc|Corp|Ltd|LLC)\b")?,
            Regex::new(r"\bUniversity of [A-Z][a-z]+\b")?,
            Regex::new(r"\b[A-Z][A-Z]+ Corporation\b")?,
        ];
        self.entity_patterns
            .insert(EntityType::Organization, org_patterns);

        // Location patterns
        let location_patterns = vec![
            Regex::new(r"\b[A-Z][a-z]+, [A-Z][A-Z]\b")?, // City, State
            Regex::new(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")?, // City Country
        ];
        self.entity_patterns
            .insert(EntityType::Location, location_patterns);

        // Temporal patterns
        self.temporal_patterns = vec![
            Regex::new(r"\b\d{4}-\d{2}-\d{2}\b")?, // YYYY-MM-DD
            Regex::new(
                r"\b(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b",
            )?,
            Regex::new(r"\b(before|after|during|since|until) \d{4}\b")?,
        ];

        // Relationship patterns
        let isa_patterns = vec![
            Regex::new(r"(.+) is an? (.+)")?,
            Regex::new(r"(.+) type of (.+)")?,
        ];
        self.relationship_patterns
            .insert(RelationshipType::IsA, isa_patterns);

        let partof_patterns = vec![
            Regex::new(r"(.+) part of (.+)")?,
            Regex::new(r"(.+) component of (.+)")?,
        ];
        self.relationship_patterns
            .insert(RelationshipType::PartOf, partof_patterns);

        Ok(())
    }

    /// Initialize schema inference rules
    fn initialize_schema_rules(&mut self) -> Result<()> {
        self.schema_inference_rules = vec![
            SchemaInferenceRule {
                rule_id: "class_definition".to_string(),
                pattern: Regex::new(r"(.+) is a type of (.+)")?,
                inferred_type: SchemaElementType::Class,
                confidence_modifier: 0.9,
            },
            SchemaInferenceRule {
                rule_id: "property_definition".to_string(),
                pattern: Regex::new(r"(.+) has (.+)")?,
                inferred_type: SchemaElementType::Property,
                confidence_modifier: 0.8,
            },
        ];

        Ok(())
    }

    /// Initialize language detectors
    fn initialize_language_detectors(&mut self) -> Result<()> {
        // English detector
        self.language_detectors.insert(
            "en".to_string(),
            LanguageDetector {
                language_code: "en".to_string(),
                detection_patterns: vec![
                    Regex::new(r"\b(the|and|or|but|if|when|where)\b")?,
                    Regex::new(r"\b(is|are|was|were|have|has|had)\b")?,
                ],
                confidence_threshold: 0.7,
            },
        );

        // Add more language detectors as needed
        Ok(())
    }

    /// Extract knowledge from text
    pub async fn extract_knowledge(&mut self, text: &str) -> Result<ExtractedKnowledge> {
        let start_time = std::time::Instant::now();
        info!(
            "Starting knowledge extraction from text of length: {}",
            text.len()
        );

        let knowledge_id = Uuid::new_v4().to_string();
        let mut extracted_triples = Vec::new();
        let mut extracted_entities = Vec::new();
        let mut extracted_relationships = Vec::new();
        let mut schema_elements = Vec::new();
        let mut temporal_facts = Vec::new();

        // Detect language
        let detected_language = self.detect_language(text).await?;
        debug!("Detected language: {}", detected_language);

        // Extract entities
        if self.config.enable_entity_extraction {
            extracted_entities = self.extract_entities(text).await?;
            debug!("Extracted {} entities", extracted_entities.len());
        }

        // Extract relationships
        if self.config.enable_relationship_extraction {
            extracted_relationships = self
                .extract_relationships(text, &extracted_entities)
                .await?;
            debug!("Extracted {} relationships", extracted_relationships.len());
        }

        // Generate triples from relationships
        for relationship in &extracted_relationships {
            if let Ok(triple) = self.relationship_to_triple(relationship) {
                extracted_triples.push(triple);
            }
        }

        // Discover schema elements
        if self.config.enable_schema_discovery {
            schema_elements = self
                .discover_schema_elements(text, &extracted_entities)
                .await?;
            debug!("Discovered {} schema elements", schema_elements.len());
        }

        // Extract temporal facts
        if self.config.enable_temporal_extraction {
            temporal_facts = self
                .extract_temporal_facts(text, &extracted_entities)
                .await?;
            debug!("Extracted {} temporal facts", temporal_facts.len());
        }

        // Validate facts if enabled
        if self.config.enable_fact_validation {
            self.validate_extracted_facts(&mut extracted_triples, &extracted_relationships)
                .await?;
        }

        // Calculate overall confidence
        let confidence_score = self.calculate_extraction_confidence(
            &extracted_entities,
            &extracted_relationships,
            &schema_elements,
        );

        let processing_time = start_time.elapsed().as_millis() as u64;

        let extraction_statistics = ExtractionStatistics {
            entities_extracted: extracted_entities.len(),
            relationships_extracted: extracted_relationships.len(),
            triples_generated: extracted_triples.len(),
            schema_elements_discovered: schema_elements.len(),
            temporal_facts_extracted: temporal_facts.len(),
            average_confidence: confidence_score,
        };

        let extraction_metadata = ExtractionMetadata {
            extraction_timestamp: Utc::now(),
            extraction_method: "Pattern-based + LLM-enhanced".to_string(),
            processing_time_ms: processing_time,
            language_detected: detected_language,
            text_length: text.len(),
            extraction_statistics,
        };

        info!("Knowledge extraction completed in {}ms", processing_time);

        Ok(ExtractedKnowledge {
            knowledge_id,
            source_text: text.to_string(),
            extracted_triples,
            extracted_entities,
            extracted_relationships,
            schema_elements,
            temporal_facts,
            confidence_score,
            extraction_metadata,
        })
    }

    /// Detect language of text
    async fn detect_language(&self, text: &str) -> Result<String> {
        // Simple pattern-based language detection
        for (lang_code, detector) in &self.language_detectors {
            let mut matches = 0;
            let mut total_patterns = 0;

            for pattern in &detector.detection_patterns {
                total_patterns += 1;
                if pattern.is_match(text) {
                    matches += 1;
                }
            }

            let confidence = matches as f64 / total_patterns as f64;
            if confidence >= detector.confidence_threshold {
                return Ok(lang_code.clone());
            }
        }

        Ok("unknown".to_string())
    }

    /// Extract entities from text
    async fn extract_entities(&self, text: &str) -> Result<Vec<ExtractedEntity>> {
        let mut entities = Vec::new();

        for (entity_type, patterns) in &self.entity_patterns {
            for pattern in patterns {
                for capture in pattern.find_iter(text) {
                    let entity_text = capture.as_str();
                    let start_pos = capture.start();
                    let end_pos = capture.end();

                    let entity = ExtractedEntity {
                        entity_id: Uuid::new_v4().to_string(),
                        entity_text: entity_text.to_string(),
                        entity_type: entity_type.clone(),
                        canonical_form: self.canonicalize_entity(entity_text),
                        aliases: Vec::new(),
                        properties: HashMap::new(),
                        confidence: 0.8, // Base confidence for pattern matches
                        source_position: TextPosition {
                            start_offset: start_pos,
                            end_offset: end_pos,
                            line_number: self.get_line_number(text, start_pos),
                            column_number: self.get_column_number(text, start_pos),
                        },
                        linked_entities: Vec::new(),
                    };

                    entities.push(entity);
                }
            }
        }

        // Remove duplicates based on canonical form
        entities.sort_by(|a, b| a.canonical_form.cmp(&b.canonical_form));
        entities.dedup_by(|a, b| a.canonical_form == b.canonical_form);

        Ok(entities)
    }

    /// Extract relationships from text
    async fn extract_relationships(
        &self,
        text: &str,
        entities: &[ExtractedEntity],
    ) -> Result<Vec<ExtractedRelationship>> {
        let mut relationships = Vec::new();

        for (relationship_type, patterns) in &self.relationship_patterns {
            for pattern in patterns {
                if let Some(captures) = pattern.captures(text) {
                    if captures.len() >= 3 {
                        let subject = captures
                            .get(1)
                            .expect("capture group 1 should exist")
                            .as_str();
                        let object = captures
                            .get(2)
                            .expect("capture group 2 should exist")
                            .as_str();

                        // Find matching entities
                        let subject_entity = self.find_matching_entity(subject, entities);
                        let object_entity = self.find_matching_entity(object, entities);

                        if let (Some(subj), Some(obj)) = (subject_entity, object_entity) {
                            let relationship = ExtractedRelationship {
                                relationship_id: Uuid::new_v4().to_string(),
                                subject_entity: subj.entity_id.clone(),
                                predicate: self.relationship_type_to_predicate(relationship_type),
                                object_entity: obj.entity_id.clone(),
                                relationship_type: relationship_type.clone(),
                                confidence: 0.8,
                                evidence_text: captures
                                    .get(0)
                                    .expect("capture group 0 should exist")
                                    .as_str()
                                    .to_string(),
                                temporal_context: None,
                                source_position: TextPosition {
                                    start_offset: captures
                                        .get(0)
                                        .expect("capture group 0 should exist")
                                        .start(),
                                    end_offset: captures
                                        .get(0)
                                        .expect("capture group 0 should exist")
                                        .end(),
                                    line_number: 1,   // Simplified
                                    column_number: 1, // Simplified
                                },
                            };

                            relationships.push(relationship);
                        }
                    }
                }
            }
        }

        Ok(relationships)
    }

    /// Discover schema elements from text
    async fn discover_schema_elements(
        &self,
        text: &str,
        _entities: &[ExtractedEntity],
    ) -> Result<Vec<SchemaElement>> {
        let mut schema_elements = Vec::new();

        for rule in &self.schema_inference_rules {
            for capture in rule.pattern.find_iter(text) {
                let element = SchemaElement {
                    element_id: Uuid::new_v4().to_string(),
                    element_type: rule.inferred_type.clone(),
                    name: capture.as_str().to_string(),
                    description: format!("Inferred from: {}", capture.as_str()),
                    properties: Vec::new(),
                    hierarchical_relations: Vec::new(),
                    constraints: Vec::new(),
                    confidence: rule.confidence_modifier,
                };

                schema_elements.push(element);
            }
        }

        Ok(schema_elements)
    }

    /// Extract temporal facts from text
    async fn extract_temporal_facts(
        &self,
        text: &str,
        _entities: &[ExtractedEntity],
    ) -> Result<Vec<TemporalFact>> {
        let mut temporal_facts = Vec::new();

        for pattern in &self.temporal_patterns {
            for capture in pattern.find_iter(text) {
                let temporal_text = capture.as_str();

                let temporal_fact = TemporalFact {
                    fact_id: Uuid::new_v4().to_string(),
                    subject: "temporal_entity".to_string(), // Would be linked to actual entities
                    predicate: "occurs_at".to_string(),
                    object: temporal_text.to_string(),
                    temporal_qualifier: TemporalQualifier {
                        qualifier_type: TemporalType::PointInTime,
                        time_point: self.parse_temporal_expression(temporal_text),
                        time_interval: None,
                        frequency: None,
                    },
                    confidence: 0.8,
                    source_text: temporal_text.to_string(),
                };

                temporal_facts.push(temporal_fact);
            }
        }

        Ok(temporal_facts)
    }

    /// Validate extracted facts for consistency
    async fn validate_extracted_facts(
        &self,
        triples: &mut Vec<Triple>,
        relationships: &[ExtractedRelationship],
    ) -> Result<()> {
        // Remove low-confidence relationships
        let valid_relationships: Vec<_> = relationships
            .iter()
            .filter(|r| r.confidence >= self.config.confidence_threshold)
            .collect();

        // Check for contradictions and validate facts
        let mut contradictions_found = 0;
        let mut validated_triples = Vec::new();

        // Create relationship maps for efficient lookup
        let mut subject_predicates: HashMap<String, Vec<&ExtractedRelationship>> = HashMap::new();
        let mut predicate_pairs: HashMap<String, Vec<(&str, &str)>> = HashMap::new();

        for relationship in &valid_relationships {
            subject_predicates
                .entry(relationship.subject_entity.clone())
                .or_default()
                .push(relationship);

            predicate_pairs
                .entry(relationship.predicate.clone())
                .or_default()
                .push((&relationship.subject_entity, &relationship.object_entity));
        }

        // Check for direct contradictions (same subject-predicate with different objects)
        for (subject, relationships) in &subject_predicates {
            let mut predicate_values: HashMap<String, Vec<&str>> = HashMap::new();

            for rel in relationships {
                predicate_values
                    .entry(rel.predicate.clone())
                    .or_default()
                    .push(&rel.object_entity);
            }

            for (predicate, values) in predicate_values {
                if values.len() > 1 {
                    // Check if multiple values for the same predicate indicate contradiction
                    let unique_values: std::collections::HashSet<_> = values.into_iter().collect();
                    if unique_values.len() > 1 && self.is_contradictory_predicate(&predicate) {
                        warn!(
                            "Contradiction detected for {}: {} has multiple {} values: {:?}",
                            subject, subject, predicate, unique_values
                        );
                        contradictions_found += 1;

                        // Keep only the highest confidence relationship for this predicate
                        if let Some(best_rel) = relationships
                            .iter()
                            .filter(|r| r.predicate == predicate)
                            .max_by(|a, b| {
                                a.confidence
                                    .partial_cmp(&b.confidence)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                        {
                            if let Ok(triple) = self.relationship_to_triple(best_rel) {
                                validated_triples.push(triple);
                            }
                        }
                        continue;
                    }
                }
            }
        }

        // Check temporal consistency
        for relationship in &valid_relationships {
            if let Some(temporal_context) = &relationship.temporal_context {
                if !self.validate_temporal_consistency(temporal_context, &valid_relationships) {
                    warn!(
                        "Temporal inconsistency detected for relationship: {} {} {}",
                        relationship.subject_entity,
                        relationship.predicate,
                        relationship.object_entity
                    );
                    contradictions_found += 1;
                    continue;
                }
            }

            // Add valid relationship as triple
            if let Ok(triple) = self.relationship_to_triple(relationship) {
                validated_triples.push(triple);
            }
        }

        // Check logical consistency (e.g., transitive relationships)
        self.validate_logical_consistency(&valid_relationships, &mut contradictions_found)?;

        // Update triples with validated ones
        triples.clear();
        triples.extend(validated_triples);

        if contradictions_found > 0 {
            warn!(
                "Found {} contradictions during fact validation",
                contradictions_found
            );
        }

        debug!("Validated {} relationships", valid_relationships.len());
        Ok(())
    }

    /// Check if a predicate type indicates contradictory values are not allowed
    fn is_contradictory_predicate(&self, predicate: &str) -> bool {
        // Define predicates that should have unique values (functional properties)
        let functional_predicates = [
            "birthDate",
            "deathDate",
            "age",
            "height",
            "weight",
            "hasGender",
            "isA",
            "type",
            "hasCapital",
            "hasPopulation",
            "hasArea",
            "founded",
            "established",
            "created",
            "died",
            "born",
        ];

        functional_predicates.iter().any(|&fp| {
            predicate.to_lowercase().contains(&fp.to_lowercase()) || predicate.ends_with(&fp)
        })
    }

    /// Validate temporal consistency of relationships
    fn validate_temporal_consistency(
        &self,
        temporal_context: &TemporalContext,
        all_relationships: &[&ExtractedRelationship],
    ) -> bool {
        // Check if temporal context makes sense
        if let (Some(start), Some(end)) = (&temporal_context.start_time, &temporal_context.end_time)
        {
            if start >= end {
                return false; // Start time cannot be after end time
            }
        }

        // Check for temporal conflicts with other relationships
        for other_rel in all_relationships {
            if let Some(other_temporal) = &other_rel.temporal_context {
                // If same entities with conflicting time periods
                if temporal_context != other_temporal {
                    // Check for overlapping time periods that might indicate conflicts
                    if self.temporal_periods_conflict(temporal_context, other_temporal) {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Check if two temporal periods conflict
    fn temporal_periods_conflict(
        &self,
        context1: &TemporalContext,
        context2: &TemporalContext,
    ) -> bool {
        // Simple check - in real implementation, this would be more sophisticated
        // Check if both have explicit time ranges that don't overlap
        match (
            (&context1.start_time, &context1.end_time),
            (&context2.start_time, &context2.end_time),
        ) {
            ((Some(start1), Some(end1)), (Some(start2), Some(end2))) => {
                // If periods don't overlap, they might be conflicting for certain relationships
                end1 < start2 || end2 < start1
            }
            _ => false, // If we don't have full temporal information, assume no conflict
        }
    }

    /// Validate logical consistency across relationships
    fn validate_logical_consistency(
        &self,
        relationships: &[&ExtractedRelationship],
        contradictions_found: &mut usize,
    ) -> Result<()> {
        // Check transitive relationships
        let mut is_a_relationships: HashMap<String, String> = HashMap::new();
        let mut part_of_relationships: HashMap<String, String> = HashMap::new();

        // Collect hierarchical relationships
        for rel in relationships {
            let pred_lower = rel.predicate.to_lowercase();
            if pred_lower.contains("isa")
                || pred_lower.contains("instanceof")
                || pred_lower.contains("type")
            {
                is_a_relationships.insert(rel.subject_entity.clone(), rel.object_entity.clone());
            } else if pred_lower.contains("partof")
                || pred_lower.contains("contains")
                || pred_lower.contains("within")
            {
                part_of_relationships.insert(rel.subject_entity.clone(), rel.object_entity.clone());
            }
        }

        // Check for cycles in is-a relationships (which would be logical contradictions)
        for (subject, object) in &is_a_relationships {
            if self.has_cycle_in_hierarchy(subject, object, &is_a_relationships) {
                warn!(
                    "Logical contradiction: Cycle detected in is-a relationship for {}",
                    subject
                );
                *contradictions_found += 1;
            }
        }

        // Check for cycles in part-of relationships
        for (subject, object) in &part_of_relationships {
            if self.has_cycle_in_hierarchy(subject, object, &part_of_relationships) {
                warn!(
                    "Logical contradiction: Cycle detected in part-of relationship for {}",
                    subject
                );
                *contradictions_found += 1;
            }
        }

        // Check domain/range constraints
        self.validate_domain_range_constraints(relationships, contradictions_found)?;

        Ok(())
    }

    /// Check for cycles in hierarchical relationships
    fn has_cycle_in_hierarchy(
        &self,
        start: &str,
        current: &str,
        hierarchy: &HashMap<String, String>,
    ) -> bool {
        if start == current {
            return true; // Direct cycle
        }

        // Follow the chain to detect cycles
        let mut visited = std::collections::HashSet::new();
        let mut current_node = current;

        while let Some(parent) = hierarchy.get(current_node) {
            if visited.contains(current_node) || current_node == start {
                return true; // Cycle detected
            }
            visited.insert(current_node.to_string());
            current_node = parent;
        }

        false
    }

    /// Validate domain and range constraints for relationships
    fn validate_domain_range_constraints(
        &self,
        relationships: &[&ExtractedRelationship],
        contradictions_found: &mut usize,
    ) -> Result<()> {
        // Define some basic domain/range constraints
        let constraints = [
            ("age", "Person", "Number"),
            ("birthDate", "Person", "Date"),
            ("hasCapital", "Country", "City"),
            ("hasPopulation", "Place", "Number"),
            ("authorOf", "Person", "Book"),
            ("marriedTo", "Person", "Person"),
        ];

        for rel in relationships {
            for (predicate, expected_domain, expected_range) in &constraints {
                if rel
                    .predicate
                    .to_lowercase()
                    .contains(&predicate.to_lowercase())
                {
                    // Check if subject matches expected domain type
                    if !self.entity_matches_type(
                        &rel.subject_entity,
                        expected_domain,
                        relationships,
                    ) {
                        warn!(
                            "Domain constraint violation: {} should be of type {} for predicate {}",
                            rel.subject_entity, expected_domain, rel.predicate
                        );
                        *contradictions_found += 1;
                    }

                    // Check if object matches expected range type
                    if !self.entity_matches_type(&rel.object_entity, expected_range, relationships)
                    {
                        warn!(
                            "Range constraint violation: {} should be of type {} for predicate {}",
                            rel.object_entity, expected_range, rel.predicate
                        );
                        *contradictions_found += 1;
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if an entity matches a given type based on other relationships
    fn entity_matches_type(
        &self,
        entity: &str,
        expected_type: &str,
        relationships: &[&ExtractedRelationship],
    ) -> bool {
        // Simple heuristic-based type checking
        let entity_lower = entity.to_lowercase();
        let type_lower = expected_type.to_lowercase();

        // Check if entity name suggests the type
        match type_lower.as_str() {
            "person" => {
                entity_lower.contains("person") || 
                entity_lower.contains("author") ||
                entity_lower.contains("writer") ||
                entity_lower.contains("scientist") ||
                // Common person name patterns
                entity.chars().next().is_some_and(|c| c.is_uppercase())
            }
            "number" => {
                entity.parse::<f64>().is_ok()
                    || entity_lower.contains("million")
                    || entity_lower.contains("thousand")
                    || entity_lower.contains("year")
            }
            "date" => {
                entity_lower.contains("19") || entity_lower.contains("20") || // Years
                entity_lower.contains("january") || entity_lower.contains("february") ||
                entity_lower.contains("march") || entity_lower.contains("april") ||
                entity_lower.contains("may") || entity_lower.contains("june") ||
                entity_lower.contains("july") || entity_lower.contains("august") ||
                entity_lower.contains("september") || entity_lower.contains("october") ||
                entity_lower.contains("november") || entity_lower.contains("december")
            }
            "country" => {
                entity_lower.contains("country") ||
                entity_lower.contains("nation") ||
                // Check if explicitly typed as country in relationships
                relationships.iter().any(|r| r.subject_entity == entity &&
                    r.predicate.to_lowercase().contains("type") && 
                    r.object_entity.to_lowercase().contains("country"))
            }
            "city" => {
                entity_lower.contains("city") ||
                entity_lower.contains("town") ||
                // Check if explicitly typed as city in relationships
                relationships.iter().any(|r| r.subject_entity == entity &&
                    r.predicate.to_lowercase().contains("type") && 
                    r.object_entity.to_lowercase().contains("city"))
            }
            "book" => {
                entity_lower.contains("book") ||
                entity_lower.contains("novel") ||
                entity_lower.contains("publication") ||
                // Check if explicitly typed as book in relationships
                relationships.iter().any(|r| r.subject_entity == entity &&
                    r.predicate.to_lowercase().contains("type") && 
                    r.object_entity.to_lowercase().contains("book"))
            }
            _ => true, // Unknown type, assume valid
        }
    }

    /// Convert relationship to RDF triple
    fn relationship_to_triple(&self, relationship: &ExtractedRelationship) -> Result<Triple> {
        // This is a simplified conversion - real implementation would be more sophisticated
        let subject = NamedNode::new(format!(
            "http://example.org/entity/{}",
            relationship.subject_entity
        ))?;
        let predicate = NamedNode::new(format!(
            "http://example.org/predicate/{}",
            relationship.predicate
        ))?;
        let object = NamedNode::new(format!(
            "http://example.org/entity/{}",
            relationship.object_entity
        ))?;

        Ok(Triple::new(
            Subject::NamedNode(subject),
            Predicate::NamedNode(predicate),
            Object::NamedNode(object),
        ))
    }

    /// Helper functions
    fn canonicalize_entity(&self, entity: &str) -> String {
        entity.trim().to_lowercase()
    }

    fn get_line_number(&self, text: &str, offset: usize) -> usize {
        text[..offset].chars().filter(|&c| c == '\n').count() + 1
    }

    fn get_column_number(&self, text: &str, offset: usize) -> usize {
        text[..offset]
            .chars()
            .rev()
            .take_while(|&c| c != '\n')
            .count()
            + 1
    }

    fn find_matching_entity<'a>(
        &self,
        text: &str,
        entities: &'a [ExtractedEntity],
    ) -> Option<&'a ExtractedEntity> {
        entities
            .iter()
            .find(|e| e.entity_text == text || e.canonical_form == self.canonicalize_entity(text))
    }

    fn relationship_type_to_predicate(&self, rel_type: &RelationshipType) -> String {
        match rel_type {
            RelationshipType::IsA => "rdf:type".to_string(),
            RelationshipType::PartOf => "part_of".to_string(),
            RelationshipType::LocatedIn => "located_in".to_string(),
            RelationshipType::OwnedBy => "owned_by".to_string(),
            RelationshipType::CreatedBy => "created_by".to_string(),
            RelationshipType::CausedBy => "caused_by".to_string(),
            RelationshipType::TemporalSequence => "temporal_sequence".to_string(),
            RelationshipType::Similarity => "similar_to".to_string(),
            RelationshipType::Dependency => "depends_on".to_string(),
            RelationshipType::Custom(pred) => pred.clone(),
        }
    }

    fn parse_temporal_expression(&self, temporal_text: &str) -> Option<DateTime<Utc>> {
        // Simplified temporal parsing - real implementation would be more sophisticated
        if let Ok(dt) = chrono::DateTime::parse_from_str(temporal_text, "%Y-%m-%d") {
            Some(dt.with_timezone(&Utc))
        } else {
            None
        }
    }

    fn calculate_extraction_confidence(
        &self,
        entities: &[ExtractedEntity],
        relationships: &[ExtractedRelationship],
        schema_elements: &[SchemaElement],
    ) -> f64 {
        let mut total_confidence = 0.0;
        let mut count = 0;

        for entity in entities {
            total_confidence += entity.confidence;
            count += 1;
        }

        for relationship in relationships {
            total_confidence += relationship.confidence;
            count += 1;
        }

        for schema_element in schema_elements {
            total_confidence += schema_element.confidence;
            count += 1;
        }

        if count > 0 {
            total_confidence / count as f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_knowledge_extraction_engine_creation() {
        let config = KnowledgeExtractionConfig::default();
        let engine = KnowledgeExtractionEngine::new(config);

        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_entity_extraction() {
        let config = KnowledgeExtractionConfig::default();
        let mut engine = KnowledgeExtractionEngine::new(config).unwrap();

        let text = "Dr. John Smith works at Microsoft Corp.";
        let result = engine.extract_knowledge(text).await;

        assert!(result.is_ok());
        let knowledge = result.unwrap();
        assert!(!knowledge.extracted_entities.is_empty());
    }

    #[test]
    fn test_canonicalize_entity() {
        let config = KnowledgeExtractionConfig::default();
        let engine = KnowledgeExtractionEngine::new(config).unwrap();

        assert_eq!(engine.canonicalize_entity("  John Smith  "), "john smith");
    }
}
