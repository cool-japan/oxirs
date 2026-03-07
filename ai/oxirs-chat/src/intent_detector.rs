//! User intent detection for SPARQL chat.
//!
//! Classifies natural-language user input into SPARQL query intents (SELECT,
//! ASK, DESCRIBE, CONSTRUCT), detects entity and property mentions, scores
//! intent confidence, handles multi-intent inputs, and recognises negation,
//! aggregation, and temporal modifiers.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Intent types
// ---------------------------------------------------------------------------

/// SPARQL query type that the user likely intends.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QueryIntent {
    /// The user wants to retrieve bindings (SELECT).
    Select,
    /// The user wants a yes/no answer (ASK).
    Ask,
    /// The user wants a description of an entity (DESCRIBE).
    Describe,
    /// The user wants to build a sub-graph (CONSTRUCT).
    Construct,
}

impl QueryIntent {
    /// Lowercase label for serialisation / display.
    pub fn as_str(&self) -> &'static str {
        match self {
            QueryIntent::Select => "select",
            QueryIntent::Ask => "ask",
            QueryIntent::Describe => "describe",
            QueryIntent::Construct => "construct",
        }
    }
}

/// Aggregation function the user is asking for.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AggregationIntent {
    Count,
    Sum,
    Average,
    Max,
    Min,
}

/// Temporal modifier detected in the input.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TemporalModifier {
    Before,
    After,
    During,
    Since,
}

// ---------------------------------------------------------------------------
// Detection result
// ---------------------------------------------------------------------------

/// A scored intent classification.
#[derive(Debug, Clone)]
pub struct ScoredIntent {
    /// The detected query intent.
    pub intent: QueryIntent,
    /// Confidence in `[0.0, 1.0]`.
    pub confidence: f64,
}

/// Full result of intent detection on a single user message.
#[derive(Debug, Clone)]
pub struct IntentDetectionResult {
    /// Primary intent (highest confidence).
    pub primary_intent: ScoredIntent,
    /// All detected intents with their scores.
    pub all_intents: Vec<ScoredIntent>,
    /// Entity mentions found in the input.
    pub entity_mentions: Vec<String>,
    /// Property mentions found in the input.
    pub property_mentions: Vec<String>,
    /// Whether negation was detected.
    pub negation_detected: bool,
    /// Aggregation intents detected.
    pub aggregations: Vec<AggregationIntent>,
    /// Temporal modifiers detected.
    pub temporal_modifiers: Vec<TemporalModifier>,
}

// ---------------------------------------------------------------------------
// IntentDetector
// ---------------------------------------------------------------------------

/// Configuration for the intent detector.
#[derive(Debug, Clone)]
pub struct IntentDetectorConfig {
    /// Known entity names / IRIs to look for.
    pub known_entities: Vec<String>,
    /// Known property names to look for.
    pub known_properties: Vec<String>,
    /// Minimum confidence to report an intent.
    pub min_confidence: f64,
}

impl Default for IntentDetectorConfig {
    fn default() -> Self {
        Self {
            known_entities: Vec::new(),
            known_properties: Vec::new(),
            min_confidence: 0.1,
        }
    }
}

/// Keyword-based user intent detector for SPARQL chat interfaces.
pub struct IntentDetector {
    config: IntentDetectorConfig,
}

impl IntentDetector {
    /// Create a new detector with default configuration.
    pub fn new() -> Self {
        Self {
            config: IntentDetectorConfig::default(),
        }
    }

    /// Create with a custom configuration.
    pub fn with_config(config: IntentDetectorConfig) -> Self {
        Self { config }
    }

    /// Register an entity name for mention detection.
    pub fn add_entity(&mut self, entity: impl Into<String>) {
        self.config.known_entities.push(entity.into());
    }

    /// Register a property name for mention detection.
    pub fn add_property(&mut self, property: impl Into<String>) {
        self.config.known_properties.push(property.into());
    }

    // ── Detection ────────────────────────────────────────────────────────

    /// Detect intents from a user message.
    pub fn detect(&self, text: &str) -> IntentDetectionResult {
        let lower = text.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        // Score each intent type.
        let mut scores: HashMap<QueryIntent, f64> = HashMap::new();
        scores.insert(QueryIntent::Select, self.score_select(&lower, &words));
        scores.insert(QueryIntent::Ask, self.score_ask(&lower, &words));
        scores.insert(QueryIntent::Describe, self.score_describe(&lower, &words));
        scores.insert(QueryIntent::Construct, self.score_construct(&lower, &words));

        // Normalise so max = 1.0 if any signal exists.
        let max_score = scores.values().copied().fold(0.0f64, f64::max);
        if max_score > 0.0 {
            for v in scores.values_mut() {
                *v /= max_score;
            }
        } else {
            // No signal → default to SELECT with low confidence.
            scores.insert(QueryIntent::Select, 0.3);
        }

        // Build sorted intent list.
        let mut all_intents: Vec<ScoredIntent> = scores
            .into_iter()
            .filter(|(_, c)| *c >= self.config.min_confidence)
            .map(|(intent, confidence)| ScoredIntent { intent, confidence })
            .collect();
        all_intents.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let primary = all_intents.first().cloned().unwrap_or(ScoredIntent {
            intent: QueryIntent::Select,
            confidence: 0.0,
        });

        let entity_mentions = self.detect_entities(&lower);
        let property_mentions = self.detect_properties(&lower);
        let negation_detected = self.detect_negation(&lower);
        let aggregations = self.detect_aggregations(&lower);
        let temporal_modifiers = self.detect_temporal(&lower);

        IntentDetectionResult {
            primary_intent: primary,
            all_intents,
            entity_mentions,
            property_mentions,
            negation_detected,
            aggregations,
            temporal_modifiers,
        }
    }

    // ── Intent scorers ───────────────────────────────────────────────────

    /// Score SELECT intent based on wh-words and list-related keywords.
    fn score_select(&self, text: &str, words: &[&str]) -> f64 {
        let mut score = 0.0;
        let select_keywords = [
            "what", "which", "who", "where", "when", "list", "show", "find", "get", "give", "tell",
            "retrieve", "fetch", "return",
        ];
        for kw in &select_keywords {
            if words.contains(kw) {
                score += 1.0;
            }
        }
        // "how many" is also select-ish (with aggregation)
        if text.contains("how many") || text.contains("how much") {
            score += 1.0;
        }
        score
    }

    /// Score ASK intent based on yes/no question patterns.
    fn score_ask(&self, text: &str, words: &[&str]) -> f64 {
        let mut score = 0.0;
        let ask_keywords = [
            "is", "are", "does", "do", "has", "have", "can", "was", "were", "will", "would",
            "could", "should",
        ];
        // The keyword must be the first word to signal a yes/no question.
        if let Some(&first) = words.first() {
            if ask_keywords.contains(&first) {
                score += 2.0;
            }
        }
        if text.contains("is there") || text.contains("does it") {
            score += 1.0;
        }
        if text.ends_with('?') && score > 0.0 {
            score += 0.5;
        }
        score
    }

    /// Score DESCRIBE intent.
    fn score_describe(&self, text: &str, words: &[&str]) -> f64 {
        let mut score = 0.0;
        let describe_keywords = [
            "describe",
            "explain",
            "detail",
            "about",
            "information",
            "definition",
            "overview",
        ];
        for kw in &describe_keywords {
            if words.contains(kw) {
                score += 1.5;
            }
        }
        if text.contains("tell me about") || text.contains("what is") {
            score += 1.0;
        }
        score
    }

    /// Score CONSTRUCT intent.
    fn score_construct(&self, _text: &str, words: &[&str]) -> f64 {
        let mut score = 0.0;
        let construct_keywords = [
            "construct",
            "build",
            "create",
            "generate",
            "produce",
            "graph",
            "subgraph",
            "triples",
            "rdf",
        ];
        for kw in &construct_keywords {
            if words.contains(kw) {
                score += 1.5;
            }
        }
        score
    }

    // ── Entity / property detection ──────────────────────────────────────

    fn detect_entities(&self, text: &str) -> Vec<String> {
        let mut found = Vec::new();

        // Check known entities (case-insensitive substring match).
        for entity in &self.config.known_entities {
            if text.contains(&entity.to_lowercase()) {
                found.push(entity.clone());
            }
        }

        // Also detect IRI-like patterns.
        for word in text.split_whitespace() {
            if word.starts_with("http://") || word.starts_with("https://") {
                let cleaned = word.trim_matches(|c: char| {
                    !c.is_alphanumeric() && c != ':' && c != '/' && c != '.' && c != '#'
                });
                if !found.contains(&cleaned.to_string()) {
                    found.push(cleaned.to_string());
                }
            }
        }

        found
    }

    fn detect_properties(&self, text: &str) -> Vec<String> {
        let mut found = Vec::new();
        for prop in &self.config.known_properties {
            if text.contains(&prop.to_lowercase()) {
                found.push(prop.clone());
            }
        }
        found
    }

    // ── Negation detection ───────────────────────────────────────────────

    fn detect_negation(&self, text: &str) -> bool {
        let negation_markers = [
            "not",
            "n't",
            "no",
            "never",
            "without",
            "except",
            "exclude",
            "excluding",
            "neither",
            "nor",
        ];
        let words: Vec<&str> = text.split_whitespace().collect();
        for marker in &negation_markers {
            if words.contains(marker) {
                return true;
            }
            // Handle contractions like "doesn't"
            if text.contains(marker) {
                return true;
            }
        }
        false
    }

    // ── Aggregation detection ────────────────────────────────────────────

    fn detect_aggregations(&self, text: &str) -> Vec<AggregationIntent> {
        let mut agg = Vec::new();
        if text.contains("count") || text.contains("how many") || text.contains("number of") {
            agg.push(AggregationIntent::Count);
        }
        if text.contains("average") || text.contains("avg") || text.contains("mean") {
            agg.push(AggregationIntent::Average);
        }
        if text.contains(" sum ") || text.contains("total") {
            agg.push(AggregationIntent::Sum);
        }
        if text.contains("maximum")
            || text.contains(" max ")
            || text.contains("highest")
            || text.contains("largest")
        {
            agg.push(AggregationIntent::Max);
        }
        if text.contains("minimum")
            || text.contains(" min ")
            || text.contains("lowest")
            || text.contains("smallest")
        {
            agg.push(AggregationIntent::Min);
        }
        agg
    }

    // ── Temporal detection ───────────────────────────────────────────────

    fn detect_temporal(&self, text: &str) -> Vec<TemporalModifier> {
        let mut mods = Vec::new();
        if text.contains("before") || text.contains("prior to") || text.contains("earlier than") {
            mods.push(TemporalModifier::Before);
        }
        if text.contains("after") || text.contains("later than") || text.contains("following") {
            mods.push(TemporalModifier::After);
        }
        if text.contains("during") || text.contains("while") || text.contains("in the period") {
            mods.push(TemporalModifier::During);
        }
        if text.contains("since") || text.contains("from") || text.contains("starting") {
            mods.push(TemporalModifier::Since);
        }
        mods
    }
}

impl Default for IntentDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn detector() -> IntentDetector {
        IntentDetector::new()
    }

    fn detector_with_entities() -> IntentDetector {
        let mut d = IntentDetector::new();
        d.add_entity("Alice");
        d.add_entity("BRCA1");
        d.add_property("knows");
        d.add_property("name");
        d
    }

    // ── QueryIntent enum ─────────────────────────────────────────────────

    #[test]
    fn test_query_intent_as_str_select() {
        assert_eq!(QueryIntent::Select.as_str(), "select");
    }

    #[test]
    fn test_query_intent_as_str_ask() {
        assert_eq!(QueryIntent::Ask.as_str(), "ask");
    }

    #[test]
    fn test_query_intent_as_str_describe() {
        assert_eq!(QueryIntent::Describe.as_str(), "describe");
    }

    #[test]
    fn test_query_intent_as_str_construct() {
        assert_eq!(QueryIntent::Construct.as_str(), "construct");
    }

    // ── Default construction ─────────────────────────────────────────────

    #[test]
    fn test_default_detector() {
        let d = IntentDetector::default();
        let r = d.detect("hello");
        assert!(r.primary_intent.confidence >= 0.0);
    }

    #[test]
    fn test_config_default() {
        let cfg = IntentDetectorConfig::default();
        assert!(cfg.known_entities.is_empty());
        assert!(cfg.known_properties.is_empty());
        assert!((cfg.min_confidence - 0.1).abs() < 1e-10);
    }

    // ── SELECT intent ────────────────────────────────────────────────────

    #[test]
    fn test_select_what_query() {
        let d = detector();
        let r = d.detect("What are the genes associated with cancer?");
        assert_eq!(r.primary_intent.intent, QueryIntent::Select);
    }

    #[test]
    fn test_select_who_query() {
        let d = detector();
        let r = d.detect("Who knows Alice?");
        assert_eq!(r.primary_intent.intent, QueryIntent::Select);
    }

    #[test]
    fn test_select_list_query() {
        let d = detector();
        let r = d.detect("List all proteins in the dataset");
        assert_eq!(r.primary_intent.intent, QueryIntent::Select);
    }

    #[test]
    fn test_select_show_query() {
        let d = detector();
        let r = d.detect("Show me the results");
        assert_eq!(r.primary_intent.intent, QueryIntent::Select);
    }

    // ── ASK intent ───────────────────────────────────────────────────────

    #[test]
    fn test_ask_is_there() {
        let d = detector();
        let r = d.detect("Is there a connection between A and B?");
        assert_eq!(r.primary_intent.intent, QueryIntent::Ask);
    }

    #[test]
    fn test_ask_does() {
        let d = detector();
        let r = d.detect("Does Alice know Bob?");
        assert_eq!(r.primary_intent.intent, QueryIntent::Ask);
    }

    #[test]
    fn test_ask_has() {
        let d = detector();
        let r = d.detect("Has the experiment been completed?");
        assert_eq!(r.primary_intent.intent, QueryIntent::Ask);
    }

    // ── DESCRIBE intent ──────────────────────────────────────────────────

    #[test]
    fn test_describe_explicit() {
        let d = detector();
        let r = d.detect("Describe the protein BRCA1");
        assert_eq!(r.primary_intent.intent, QueryIntent::Describe);
    }

    #[test]
    fn test_describe_tell_me_about() {
        let d = detector();
        let r = d.detect("Tell me about breast cancer genes");
        // "tell" triggers select, "about" triggers describe
        // "about" with 1.5 weight should win
        let has_describe = r
            .all_intents
            .iter()
            .any(|i| i.intent == QueryIntent::Describe);
        assert!(has_describe);
    }

    #[test]
    fn test_describe_explain() {
        let d = detector();
        let r = d.detect("Explain the relationship between X and Y");
        assert_eq!(r.primary_intent.intent, QueryIntent::Describe);
    }

    // ── CONSTRUCT intent ─────────────────────────────────────────────────

    #[test]
    fn test_construct_explicit() {
        let d = detector();
        let r = d.detect("Construct a subgraph of related triples");
        assert_eq!(r.primary_intent.intent, QueryIntent::Construct);
    }

    #[test]
    fn test_construct_build() {
        let d = detector();
        let r = d.detect("Build an RDF graph of the results");
        assert_eq!(r.primary_intent.intent, QueryIntent::Construct);
    }

    // ── Entity mention detection ─────────────────────────────────────────

    #[test]
    fn test_entity_mention_known() {
        let d = detector_with_entities();
        let r = d.detect("What do we know about alice?");
        assert!(r.entity_mentions.contains(&"Alice".to_string()));
    }

    #[test]
    fn test_entity_mention_iri() {
        let d = detector();
        let r = d.detect("Describe http://example.org/entity1 please");
        assert!(r
            .entity_mentions
            .iter()
            .any(|e| e.contains("http://example.org/entity1")));
    }

    #[test]
    fn test_entity_mention_none() {
        let d = detector_with_entities();
        let r = d.detect("How many items are there?");
        assert!(r.entity_mentions.is_empty());
    }

    // ── Property mention detection ───────────────────────────────────────

    #[test]
    fn test_property_mention_known() {
        let d = detector_with_entities();
        let r = d.detect("What is the name of Alice?");
        assert!(r.property_mentions.contains(&"name".to_string()));
    }

    #[test]
    fn test_property_mention_none() {
        let d = detector_with_entities();
        let r = d.detect("Hello world");
        assert!(r.property_mentions.is_empty());
    }

    // ── Negation detection ───────────────────────────────────────────────

    #[test]
    fn test_negation_not() {
        let d = detector();
        let r = d.detect("Show items that are not active");
        assert!(r.negation_detected);
    }

    #[test]
    fn test_negation_without() {
        let d = detector();
        let r = d.detect("Find proteins without mutations");
        assert!(r.negation_detected);
    }

    #[test]
    fn test_negation_except() {
        let d = detector();
        let r = d.detect("All genes except BRCA1");
        assert!(r.negation_detected);
    }

    #[test]
    fn test_no_negation() {
        let d = detector();
        let r = d.detect("List all active items");
        assert!(!r.negation_detected);
    }

    // ── Aggregation detection ────────────────────────────────────────────

    #[test]
    fn test_aggregation_count() {
        let d = detector();
        let r = d.detect("How many genes are there?");
        assert!(r.aggregations.contains(&AggregationIntent::Count));
    }

    #[test]
    fn test_aggregation_average() {
        let d = detector();
        let r = d.detect("What is the average score?");
        assert!(r.aggregations.contains(&AggregationIntent::Average));
    }

    #[test]
    fn test_aggregation_max() {
        let d = detector();
        let r = d.detect("What is the maximum value?");
        assert!(r.aggregations.contains(&AggregationIntent::Max));
    }

    #[test]
    fn test_aggregation_min() {
        let d = detector();
        let r = d.detect("What is the minimum temperature?");
        assert!(r.aggregations.contains(&AggregationIntent::Min));
    }

    #[test]
    fn test_aggregation_sum() {
        let d = detector();
        let r = d.detect("What is the total sum of sales?");
        assert!(r.aggregations.contains(&AggregationIntent::Sum));
    }

    #[test]
    fn test_no_aggregation() {
        let d = detector();
        let r = d.detect("Show me the list");
        assert!(r.aggregations.is_empty());
    }

    // ── Temporal detection ───────────────────────────────────────────────

    #[test]
    fn test_temporal_before() {
        let d = detector();
        let r = d.detect("Events before 2020");
        assert!(r.temporal_modifiers.contains(&TemporalModifier::Before));
    }

    #[test]
    fn test_temporal_after() {
        let d = detector();
        let r = d.detect("Publications after January");
        assert!(r.temporal_modifiers.contains(&TemporalModifier::After));
    }

    #[test]
    fn test_temporal_during() {
        let d = detector();
        let r = d.detect("Changes during the experiment");
        assert!(r.temporal_modifiers.contains(&TemporalModifier::During));
    }

    #[test]
    fn test_temporal_since() {
        let d = detector();
        let r = d.detect("Active since last year");
        assert!(r.temporal_modifiers.contains(&TemporalModifier::Since));
    }

    #[test]
    fn test_no_temporal() {
        let d = detector();
        let r = d.detect("List all items");
        assert!(r.temporal_modifiers.is_empty());
    }

    // ── Multi-intent ─────────────────────────────────────────────────────

    #[test]
    fn test_multi_intent_select_and_count() {
        let d = detector();
        let r = d.detect("How many genes are associated with cancer?");
        // Should have both SELECT (from "how") and aggregation COUNT
        assert_eq!(r.primary_intent.intent, QueryIntent::Select);
        assert!(r.aggregations.contains(&AggregationIntent::Count));
    }

    #[test]
    fn test_multi_intent_multiple_scored() {
        let d = detector();
        let r = d.detect("What is the description of this entity?");
        // Both SELECT ("what") and DESCRIBE ("description") should score
        assert!(r.all_intents.len() >= 2);
    }

    // ── Confidence scoring ───────────────────────────────────────────────

    #[test]
    fn test_confidence_in_range() {
        let d = detector();
        let r = d.detect("What genes are there?");
        for intent in &r.all_intents {
            assert!(intent.confidence >= 0.0 && intent.confidence <= 1.0);
        }
    }

    #[test]
    fn test_primary_intent_highest_confidence() {
        let d = detector();
        let r = d.detect("Describe the main protein");
        if r.all_intents.len() > 1 {
            assert!(r.primary_intent.confidence >= r.all_intents[1].confidence);
        }
    }

    // ── Edge cases ───────────────────────────────────────────────────────

    #[test]
    fn test_empty_input() {
        let d = detector();
        let r = d.detect("");
        // Should still produce a result (low-confidence default)
        assert!(r.primary_intent.confidence >= 0.0);
    }

    #[test]
    fn test_gibberish_input() {
        let d = detector();
        let r = d.detect("xyzzy plugh foo bar");
        // No keywords → default SELECT with low confidence
        assert!(r.primary_intent.confidence <= 0.5);
    }
}
