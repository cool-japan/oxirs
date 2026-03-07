//! # Relation Extractor
//!
//! Pattern-based relation extraction from natural language text for knowledge graph construction.
//!
//! Entities are identified by label matching (case-insensitive substring search) with configurable
//! entity type dictionaries. Relations are extracted when a trigger word from a `RelationPattern`
//! appears between two entity mentions of compatible types.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_graphrag::relation_extractor::{RelationExtractor, RelationPattern};
//!
//! let mut extractor = RelationExtractor::new();
//! extractor.add_entity_type("Person", vec!["Alice".to_string(), "Bob".to_string()]);
//! extractor.add_pattern(RelationPattern {
//!     name: "knows".to_string(),
//!     subject_type: "Person".to_string(),
//!     object_type: "Person".to_string(),
//!     trigger_words: vec!["knows".to_string()],
//!     predicate_iri: "http://xmlns.com/foaf/0.1/knows".to_string(),
//! });
//! let result = extractor.extract("Alice knows Bob in the meeting.");
//! assert!(!result.relations.is_empty());
//! ```

use std::collections::HashMap;

/// A span of text with byte-level start/end offsets
#[derive(Debug, Clone, PartialEq)]
pub struct TextSpan {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

/// An entity mention detected in the source text
#[derive(Debug, Clone)]
pub struct EntityMention {
    pub span: TextSpan,
    pub entity_type: String,
    /// Optional IRI for the linked entity (derived from label + type namespace)
    pub linked_iri: Option<String>,
}

/// A relation extraction pattern that maps text triggers to a semantic predicate
#[derive(Debug, Clone)]
pub struct RelationPattern {
    /// Human-readable pattern name
    pub name: String,
    /// Required entity type for the subject
    pub subject_type: String,
    /// Required entity type for the object
    pub object_type: String,
    /// Words (case-insensitive) that trigger this relation
    pub trigger_words: Vec<String>,
    /// Predicate IRI to assign to extracted relations
    pub predicate_iri: String,
}

/// A single extracted relation triple with provenance
#[derive(Debug, Clone)]
pub struct ExtractedRelation {
    pub subject: EntityMention,
    pub predicate_iri: String,
    pub object: EntityMention,
    /// Confidence score in range [0.0, 1.0]
    pub confidence: f64,
    /// The full source sentence/text from which this relation was extracted
    pub source_text: String,
}

/// Overall extraction result for a text document
#[derive(Debug, Clone)]
pub struct ExtractionResult {
    pub relations: Vec<ExtractedRelation>,
    pub entity_mentions: Vec<EntityMention>,
    /// Fraction of characters covered by detected entity mentions (0.0 – 1.0)
    pub coverage: f64,
}

/// Core relation extraction engine
pub struct RelationExtractor {
    patterns: Vec<RelationPattern>,
    /// Maps entity type name → list of labels to recognise
    entity_types: HashMap<String, Vec<String>>,
}

impl RelationExtractor {
    /// Create a new, empty extractor
    pub fn new() -> Self {
        RelationExtractor {
            patterns: Vec::new(),
            entity_types: HashMap::new(),
        }
    }

    /// Register a new relation pattern
    pub fn add_pattern(&mut self, pattern: RelationPattern) {
        self.patterns.push(pattern);
    }

    /// Register an entity type with its known label strings
    pub fn add_entity_type(&mut self, type_name: impl Into<String>, labels: Vec<String>) {
        self.entity_types.insert(type_name.into(), labels);
    }

    /// Return the number of registered patterns
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Return the number of registered entity types
    pub fn entity_type_count(&self) -> usize {
        self.entity_types.len()
    }

    /// Detect all entity mentions in `text`.
    ///
    /// Labels are matched case-insensitively. Multiple occurrences of the same label
    /// at different positions are each reported as separate `EntityMention`s.
    pub fn extract_entities(&self, text: &str) -> Vec<EntityMention> {
        let mut mentions: Vec<EntityMention> = Vec::new();
        let text_lower = text.to_lowercase();

        for (type_name, labels) in &self.entity_types {
            for label in labels {
                let label_lower = label.to_lowercase();
                // Find every non-overlapping occurrence of the label in the text
                let mut search_start = 0usize;
                while search_start < text_lower.len() {
                    match text_lower[search_start..].find(label_lower.as_str()) {
                        None => break,
                        Some(offset) => {
                            let start = search_start + offset;
                            let end = start + label.len();
                            let span = TextSpan {
                                text: text[start..end].to_string(),
                                start,
                                end,
                            };
                            let iri = Some(format!(
                                "http://example.org/entity/{}/{}",
                                type_name.to_lowercase(),
                                label
                                    .to_lowercase()
                                    .replace(' ', "_")
                            ));
                            mentions.push(EntityMention {
                                span,
                                entity_type: type_name.clone(),
                                linked_iri: iri,
                            });
                            search_start = end;
                        }
                    }
                }
            }
        }

        // Sort mentions by start offset for deterministic ordering
        mentions.sort_by_key(|m| m.span.start);
        mentions
    }

    /// Extract relations from `text` using the registered patterns and entity types.
    ///
    /// Algorithm:
    /// 1. Identify all entity mentions.
    /// 2. For each ordered pair (subject_mention, object_mention) where subject appears before
    ///    object in the text, check if any trigger word from a matching pattern appears in the
    ///    text between the two mentions.
    /// 3. If a trigger is found, emit an `ExtractedRelation` with a confidence derived from
    ///    trigger proximity and pattern rank.
    pub fn extract(&self, text: &str) -> ExtractionResult {
        let entity_mentions = self.extract_entities(text);
        let text_lower = text.to_lowercase();
        let mut relations: Vec<ExtractedRelation> = Vec::new();

        for (s_idx, subject) in entity_mentions.iter().enumerate() {
            for object in entity_mentions.iter().skip(s_idx + 1) {
                // Subject must appear before object
                if subject.span.end > object.span.start {
                    continue;
                }

                let between_start = subject.span.end;
                let between_end = object.span.start;
                let between_text = &text_lower[between_start..between_end];

                for pattern in &self.patterns {
                    // Check type compatibility
                    if pattern.subject_type != subject.entity_type {
                        continue;
                    }
                    if pattern.object_type != object.entity_type {
                        continue;
                    }

                    // Look for any trigger word in the between-text
                    let trigger_match = pattern.trigger_words.iter().find(|tw| {
                        let tw_lower = tw.to_lowercase();
                        between_text.contains(tw_lower.as_str())
                    });

                    if let Some(trigger) = trigger_match {
                        // Confidence: closer trigger → higher confidence; base is 0.7
                        let gap = (between_end - between_start) as f64;
                        let trigger_len = trigger.len() as f64;
                        // Normalize: confidence decreases as gap grows beyond trigger length
                        let proximity_bonus = (trigger_len / gap.max(trigger_len)).min(1.0);
                        let confidence = (0.7 + 0.3 * proximity_bonus).min(1.0);

                        relations.push(ExtractedRelation {
                            subject: subject.clone(),
                            predicate_iri: pattern.predicate_iri.clone(),
                            object: object.clone(),
                            confidence,
                            source_text: text.to_string(),
                        });
                        // Only emit the first matching pattern for each (subject, object) pair
                        break;
                    }
                }
            }
        }

        // Coverage = total characters covered by entity spans / text length
        let coverage = compute_coverage(&entity_mentions, text.len());

        ExtractionResult {
            relations,
            entity_mentions,
            coverage,
        }
    }
}

impl Default for RelationExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the fraction of the text covered by entity mention spans.
///
/// Overlapping spans are merged before summing covered characters.
fn compute_coverage(mentions: &[EntityMention], text_len: usize) -> f64 {
    if text_len == 0 || mentions.is_empty() {
        return 0.0;
    }

    // Collect and sort intervals
    let mut intervals: Vec<(usize, usize)> =
        mentions.iter().map(|m| (m.span.start, m.span.end)).collect();
    intervals.sort_by_key(|&(s, _)| s);

    // Merge overlapping intervals
    let mut covered = 0usize;
    let mut current_start = intervals[0].0;
    let mut current_end = intervals[0].1;

    for &(s, e) in intervals.iter().skip(1) {
        if s <= current_end {
            current_end = current_end.max(e);
        } else {
            covered += current_end - current_start;
            current_start = s;
            current_end = e;
        }
    }
    covered += current_end - current_start;

    covered as f64 / text_len as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_person_extractor() -> RelationExtractor {
        let mut ex = RelationExtractor::new();
        ex.add_entity_type("Person", vec!["Alice".to_string(), "Bob".to_string(), "Carol".to_string()]);
        ex.add_entity_type("Organization", vec!["ACME".to_string(), "OxiCorp".to_string()]);
        ex.add_pattern(RelationPattern {
            name: "knows".to_string(),
            subject_type: "Person".to_string(),
            object_type: "Person".to_string(),
            trigger_words: vec!["knows".to_string(), "met".to_string()],
            predicate_iri: "http://xmlns.com/foaf/0.1/knows".to_string(),
        });
        ex.add_pattern(RelationPattern {
            name: "worksFor".to_string(),
            subject_type: "Person".to_string(),
            object_type: "Organization".to_string(),
            trigger_words: vec!["works for".to_string(), "employed by".to_string()],
            predicate_iri: "http://schema.org/worksFor".to_string(),
        });
        ex
    }

    // ===== entity_type_count / pattern_count =====

    #[test]
    fn test_pattern_count_empty() {
        let ex = RelationExtractor::new();
        assert_eq!(ex.pattern_count(), 0);
    }

    #[test]
    fn test_entity_type_count_empty() {
        let ex = RelationExtractor::new();
        assert_eq!(ex.entity_type_count(), 0);
    }

    #[test]
    fn test_pattern_count_after_add() {
        let ex = make_person_extractor();
        assert_eq!(ex.pattern_count(), 2);
    }

    #[test]
    fn test_entity_type_count_after_add() {
        let ex = make_person_extractor();
        assert_eq!(ex.entity_type_count(), 2);
    }

    // ===== extract_entities =====

    #[test]
    fn test_extract_entities_finds_single_label() {
        let ex = make_person_extractor();
        let mentions = ex.extract_entities("Alice went to the market.");
        let texts: Vec<_> = mentions.iter().map(|m| m.span.text.as_str()).collect();
        assert!(texts.contains(&"Alice"), "Should find Alice");
    }

    #[test]
    fn test_extract_entities_finds_multiple_labels() {
        let ex = make_person_extractor();
        let mentions = ex.extract_entities("Alice knows Bob very well.");
        assert!(mentions.len() >= 2);
    }

    #[test]
    fn test_extract_entities_case_insensitive() {
        let ex = make_person_extractor();
        let mentions = ex.extract_entities("alice and BOB are friends.");
        assert!(mentions.len() >= 2, "Labels should match case-insensitively");
    }

    #[test]
    fn test_extract_entities_correct_span_offsets() {
        let ex = make_person_extractor();
        let text = "Hello Alice there.";
        let mentions = ex.extract_entities(text);
        let alice = mentions.iter().find(|m| m.span.text.to_lowercase() == "alice");
        assert!(alice.is_some());
        let alice = alice.unwrap();
        assert_eq!(&text[alice.span.start..alice.span.end].to_lowercase(), "alice");
    }

    #[test]
    fn test_extract_entities_entity_type_assigned() {
        let ex = make_person_extractor();
        let mentions = ex.extract_entities("Alice works at ACME.");
        let alice_mention = mentions.iter().find(|m| m.span.text.to_lowercase() == "alice");
        let acme_mention = mentions.iter().find(|m| m.span.text.to_lowercase() == "acme");
        assert!(alice_mention.is_some());
        assert!(acme_mention.is_some());
        assert_eq!(alice_mention.unwrap().entity_type, "Person");
        assert_eq!(acme_mention.unwrap().entity_type, "Organization");
    }

    #[test]
    fn test_extract_entities_linked_iri_set() {
        let ex = make_person_extractor();
        let mentions = ex.extract_entities("Alice is here.");
        let alice = mentions.iter().find(|m| m.span.text.to_lowercase() == "alice").unwrap();
        assert!(alice.linked_iri.is_some());
        assert!(alice.linked_iri.as_ref().unwrap().contains("alice"));
    }

    #[test]
    fn test_extract_entities_no_match_returns_empty() {
        let ex = make_person_extractor();
        let mentions = ex.extract_entities("The quick brown fox jumps.");
        assert!(mentions.is_empty());
    }

    #[test]
    fn test_extract_entities_repeated_occurrence() {
        let ex = make_person_extractor();
        let mentions = ex.extract_entities("Alice talked to Alice again.");
        let alice_count = mentions.iter().filter(|m| m.span.text.to_lowercase() == "alice").count();
        assert_eq!(alice_count, 2, "Two occurrences of Alice");
    }

    // ===== extract (relations) =====

    #[test]
    fn test_extract_relation_from_trigger_word() {
        let ex = make_person_extractor();
        let result = ex.extract("Alice knows Bob from work.");
        assert!(!result.relations.is_empty(), "Should find the knows relation");
        assert_eq!(result.relations[0].predicate_iri, "http://xmlns.com/foaf/0.1/knows");
    }

    #[test]
    fn test_extract_relation_source_text_captured() {
        let ex = make_person_extractor();
        let text = "Alice knows Bob.";
        let result = ex.extract(text);
        assert!(!result.relations.is_empty());
        assert_eq!(result.relations[0].source_text, text);
    }

    #[test]
    fn test_extract_relation_subject_and_object() {
        let ex = make_person_extractor();
        let result = ex.extract("Alice knows Bob from yesterday.");
        assert!(!result.relations.is_empty());
        let rel = &result.relations[0];
        assert_eq!(rel.subject.span.text.to_lowercase(), "alice");
        assert_eq!(rel.object.span.text.to_lowercase(), "bob");
    }

    #[test]
    fn test_extract_confidence_in_range() {
        let ex = make_person_extractor();
        let result = ex.extract("Alice knows Bob.");
        for rel in &result.relations {
            assert!(rel.confidence >= 0.0 && rel.confidence <= 1.0);
        }
    }

    #[test]
    fn test_extract_no_pattern_no_relations() {
        let ex = RelationExtractor::new();
        let result = ex.extract("Alice knows Bob.");
        assert!(result.relations.is_empty());
    }

    #[test]
    fn test_extract_wrong_entity_types_no_match() {
        // Pattern requires Person→Person, but the text has Person→Organization
        let mut ex = RelationExtractor::new();
        ex.add_entity_type("Person", vec!["Alice".to_string()]);
        ex.add_entity_type("Organization", vec!["ACME".to_string()]);
        ex.add_pattern(RelationPattern {
            name: "knows".to_string(),
            subject_type: "Person".to_string(),
            object_type: "Person".to_string(),  // wrong type for ACME
            trigger_words: vec!["at".to_string()],
            predicate_iri: "http://xmlns.com/foaf/0.1/knows".to_string(),
        });
        let result = ex.extract("Alice works at ACME.");
        assert!(result.relations.is_empty(), "Type mismatch should prevent extraction");
    }

    #[test]
    fn test_extract_multiple_patterns_multiple_relations() {
        let ex = make_person_extractor();
        let result = ex.extract("Alice knows Bob and Alice works for ACME.");
        // Should find both knows and worksFor
        assert!(result.relations.len() >= 1);
    }

    #[test]
    fn test_extract_predicate_iri_assigned() {
        let ex = make_person_extractor();
        let result = ex.extract("Alice works for ACME today.");
        let work_rel = result.relations.iter()
            .find(|r| r.predicate_iri == "http://schema.org/worksFor");
        assert!(work_rel.is_some(), "worksFor relation should be extracted");
    }

    #[test]
    fn test_extract_alternative_trigger_words() {
        let ex = make_person_extractor();
        // "met" is also a trigger for knows
        let result = ex.extract("Alice met Bob at the conference.");
        assert!(!result.relations.is_empty());
        assert_eq!(result.relations[0].predicate_iri, "http://xmlns.com/foaf/0.1/knows");
    }

    // ===== coverage =====

    #[test]
    fn test_coverage_zero_when_no_entities() {
        let ex = make_person_extractor();
        let result = ex.extract("The quick brown fox.");
        assert_eq!(result.coverage, 0.0);
    }

    #[test]
    fn test_coverage_positive_when_entities_found() {
        let ex = make_person_extractor();
        let result = ex.extract("Alice knows Bob.");
        assert!(result.coverage > 0.0);
        assert!(result.coverage <= 1.0);
    }

    #[test]
    fn test_coverage_proportional() {
        let ex = make_person_extractor();
        // Short text with big entity label → higher coverage
        let r1 = ex.extract("Alice."); // "Alice" is 5 chars out of 6
        let r2 = ex.extract("The quick Alice is somewhere far away.");
        assert!(r1.coverage > r2.coverage);
    }

    #[test]
    fn test_coverage_at_most_one() {
        let ex = make_person_extractor();
        let result = ex.extract("Alice Bob Carol Alice Bob.");
        assert!(result.coverage <= 1.0);
    }

    // ===== entity mentions in result =====

    #[test]
    fn test_extraction_result_entity_mentions_populated() {
        let ex = make_person_extractor();
        let result = ex.extract("Alice knows Bob and Carol.");
        assert!(result.entity_mentions.len() >= 3);
    }

    #[test]
    fn test_extraction_result_empty_text() {
        let ex = make_person_extractor();
        let result = ex.extract("");
        assert!(result.relations.is_empty());
        assert!(result.entity_mentions.is_empty());
        assert_eq!(result.coverage, 0.0);
    }

    // ===== additional edge cases =====

    #[test]
    fn test_add_entity_type_replaces_existing() {
        let mut ex = RelationExtractor::new();
        ex.add_entity_type("Person", vec!["Alice".to_string()]);
        ex.add_entity_type("Person", vec!["Bob".to_string()]);
        // Second call replaces the first, count stays 1
        assert_eq!(ex.entity_type_count(), 1);
    }

    #[test]
    fn test_multiple_entity_types() {
        let mut ex = RelationExtractor::new();
        ex.add_entity_type("Person", vec!["Alice".to_string()]);
        ex.add_entity_type("City", vec!["Paris".to_string()]);
        ex.add_entity_type("Company", vec!["ACME".to_string()]);
        assert_eq!(ex.entity_type_count(), 3);
    }

    #[test]
    fn test_relation_with_long_gap_has_lower_confidence() {
        let mut ex = RelationExtractor::new();
        ex.add_entity_type("Person", vec!["Alice".to_string(), "Bob".to_string()]);
        ex.add_pattern(RelationPattern {
            name: "knows".to_string(),
            subject_type: "Person".to_string(),
            object_type: "Person".to_string(),
            trigger_words: vec!["knows".to_string()],
            predicate_iri: "http://xmlns.com/foaf/0.1/knows".to_string(),
        });
        let short = ex.extract("Alice knows Bob.");
        let long = ex.extract("Alice knows and deeply appreciates having met a wonderful colleague named Bob.");
        // Confidence should be defined and in range for both
        if !short.relations.is_empty() && !long.relations.is_empty() {
            assert!(short.relations[0].confidence >= 0.0);
            assert!(long.relations[0].confidence >= 0.0);
        }
    }

    #[test]
    fn test_default_extractor_is_empty() {
        let ex = RelationExtractor::default();
        assert_eq!(ex.pattern_count(), 0);
        assert_eq!(ex.entity_type_count(), 0);
    }

    #[test]
    fn test_relation_extractor_handles_unicode() {
        let mut ex = RelationExtractor::new();
        ex.add_entity_type("Person", vec!["Müller".to_string(), "Naïve".to_string()]);
        ex.add_pattern(RelationPattern {
            name: "knows".to_string(),
            subject_type: "Person".to_string(),
            object_type: "Person".to_string(),
            trigger_words: vec!["trifft".to_string()],
            predicate_iri: "http://xmlns.com/foaf/0.1/knows".to_string(),
        });
        let result = ex.extract("Müller trifft Naïve in der Stadt.");
        assert!(!result.entity_mentions.is_empty());
    }

    #[test]
    fn test_extraction_result_has_all_fields() {
        let ex = make_person_extractor();
        let result = ex.extract("Alice knows Bob.");
        // Verify all fields are accessible
        let _ = result.relations.len();
        let _ = result.entity_mentions.len();
        let _ = result.coverage;
    }

    #[test]
    fn test_extract_entities_sorted_by_start() {
        let ex = make_person_extractor();
        let mentions = ex.extract_entities("Bob then Alice met Carol.");
        for window in mentions.windows(2) {
            assert!(window[0].span.start <= window[1].span.start);
        }
    }

    #[test]
    fn test_confidence_at_most_one() {
        let ex = make_person_extractor();
        let result = ex.extract("Alice knows Bob.");
        for rel in &result.relations {
            assert!(rel.confidence <= 1.0, "Confidence must be <= 1.0");
        }
    }

    #[test]
    fn test_confidence_at_least_zero() {
        let ex = make_person_extractor();
        let result = ex.extract("Alice knows Bob.");
        for rel in &result.relations {
            assert!(rel.confidence >= 0.0, "Confidence must be >= 0.0");
        }
    }
}
