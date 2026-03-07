//! # Intent Classifier for SPARQL/RDF Chatbots
//!
//! Rule-based intent classification using keyword matching and simple
//! string pattern matching. Supports the full set of SPARQL/RDF intents
//! required for a semantic-web chatbot.
//!
//! ## Intents
//!
//! - **SparqlQuery**: SPARQL SELECT / ASK / CONSTRUCT queries
//! - **RdfInsert**: Insert triples
//! - **RdfDelete**: Delete triples
//! - **SchemaQuestion**: Ontology / schema questions
//! - **FactLookup**: Entity lookups and listing
//! - **Navigation**: Browsing the knowledge graph
//! - **Help**: How-to questions
//! - **Greeting**: Salutations
//! - **Farewell**: Goodbyes
//! - **Unknown**: No rule matched with sufficient confidence

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────
// Intent
// ─────────────────────────────────────────────

/// Recognised chatbot intent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Intent {
    SparqlQuery,
    RdfInsert,
    RdfDelete,
    SchemaQuestion,
    FactLookup,
    Navigation,
    Help,
    Greeting,
    Farewell,
    Unknown,
}

impl std::fmt::Display for Intent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

// ─────────────────────────────────────────────
// Rule
// ─────────────────────────────────────────────

/// A single classification rule for one intent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentRule {
    pub intent: Intent,
    /// Exact lowercase keywords to look for.
    pub keywords: Vec<String>,
    /// Literal substrings (case-insensitive) to search for.
    pub patterns: Vec<String>,
    /// Score multiplier for this rule.
    pub weight: f64,
}

impl IntentRule {
    fn new(intent: Intent, keywords: &[&str], patterns: &[&str], weight: f64) -> Self {
        Self {
            intent,
            keywords: keywords.iter().map(|s| s.to_string()).collect(),
            patterns: patterns.iter().map(|s| s.to_string()).collect(),
            weight,
        }
    }
}

// ─────────────────────────────────────────────
// Result
// ─────────────────────────────────────────────

/// Result of classifying a single utterance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub intent: Intent,
    /// Confidence in [0.0, 1.0].
    pub confidence: f64,
    /// Keywords that contributed to the match.
    pub matched_keywords: Vec<String>,
    /// Patterns that contributed to the match.
    pub matched_patterns: Vec<String>,
}

// ─────────────────────────────────────────────
// Classifier
// ─────────────────────────────────────────────

/// Rule-based intent classifier.
#[derive(Debug, Clone)]
pub struct IntentClassifier {
    rules: Vec<IntentRule>,
}

impl Default for IntentClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl IntentClassifier {
    /// Create a new classifier pre-loaded with default rules.
    pub fn new() -> Self {
        let rules = vec![
            IntentRule::new(
                Intent::SparqlQuery,
                &[
                    "select",
                    "where",
                    "sparql",
                    "query",
                    "ask",
                    "construct",
                    "describe",
                    "prefix",
                ],
                &["SELECT * WHERE", "PREFIX ", "ASK {", "CONSTRUCT {"],
                1.5,
            ),
            IntentRule::new(
                Intent::RdfInsert,
                &["insert", "add", "create", "triple", "assert"],
                &["insert data", "add triple", "INSERT DATA"],
                1.2,
            ),
            IntentRule::new(
                Intent::RdfDelete,
                &["delete", "remove", "drop", "retract"],
                &["delete data", "remove triple", "DROP GRAPH"],
                1.2,
            ),
            IntentRule::new(
                Intent::SchemaQuestion,
                &[
                    "schema",
                    "ontology",
                    "class",
                    "property",
                    "define",
                    "definition",
                    "concept",
                    "subclass",
                    "domain",
                    "range",
                ],
                &["what is", "what are", "define "],
                1.0,
            ),
            IntentRule::new(
                Intent::FactLookup,
                &[
                    "who", "when", "which", "find", "list", "show", "get", "fetch", "retrieve",
                ],
                &["tell me about", "find all", "list all"],
                1.0,
            ),
            IntentRule::new(
                Intent::Navigation,
                &[
                    "navigate", "browse", "explore", "graph", "node", "edge", "traverse", "path",
                ],
                &["show graph", "browse to", "navigate to"],
                0.9,
            ),
            IntentRule::new(
                Intent::Help,
                &[
                    "help", "assist", "support", "tutorial", "guide", "manual", "howto",
                ],
                &["how to", "how do i", "what can you", "show me how"],
                2.0,
            ),
            IntentRule::new(
                Intent::Greeting,
                &[
                    "hello",
                    "hi",
                    "hey",
                    "greetings",
                    "howdy",
                    "morning",
                    "afternoon",
                    "evening",
                ],
                &["good morning", "good afternoon", "good evening"],
                0.8,
            ),
            IntentRule::new(
                Intent::Farewell,
                &[
                    "bye", "goodbye", "farewell", "ciao", "later", "thanks", "thank",
                ],
                &["see you", "take care", "good bye"],
                1.5,
            ),
        ];
        Self { rules }
    }

    /// Classify a single utterance.
    pub fn classify(&self, text: &str) -> ClassificationResult {
        let normalised = Self::normalize_text(text);
        let mut best_score = 0.0_f64;
        let mut best_intent = Intent::Unknown;
        let mut best_kws: Vec<String> = vec![];
        let mut best_pats: Vec<String> = vec![];

        for rule in &self.rules {
            let (score, kws, pats) = self.score_rule(rule, &normalised);
            if score > best_score {
                best_score = score;
                best_intent = rule.intent.clone();
                best_kws = kws;
                best_pats = pats;
            }
        }

        let confidence = (best_score / 10.0_f64).min(1.0);

        ClassificationResult {
            intent: best_intent,
            confidence,
            matched_keywords: best_kws,
            matched_patterns: best_pats,
        }
    }

    /// Add a custom rule to the classifier.
    pub fn add_rule(&mut self, rule: IntentRule) {
        self.rules.push(rule);
    }

    /// Score a single rule against the normalised text.
    ///
    /// Returns `(score, matched_keywords, matched_patterns)`.
    pub fn score_rule(&self, rule: &IntentRule, text: &str) -> (f64, Vec<String>, Vec<String>) {
        let mut score = 0.0_f64;
        let mut matched_kws: Vec<String> = Vec::new();
        let mut matched_pats: Vec<String> = Vec::new();

        // Keyword matching: each matched keyword contributes 1 point × weight
        for kw in &rule.keywords {
            // Use word-boundary-like check: check if kw appears as a token
            if contains_word(text, kw) {
                score += rule.weight;
                matched_kws.push(kw.clone());
            }
        }

        // Pattern matching: each matched pattern contributes 2 points × weight
        for pat in &rule.patterns {
            if text.contains(pat.to_lowercase().as_str()) {
                score += rule.weight * 2.0;
                matched_pats.push(pat.clone());
            }
        }

        (score, matched_kws, matched_pats)
    }

    /// Lowercase and strip punctuation from text.
    pub fn normalize_text(text: &str) -> String {
        text.chars()
            .map(|c| {
                if c.is_alphanumeric() || c == ' ' || c == '_' {
                    c.to_ascii_lowercase()
                } else {
                    ' '
                }
            })
            .collect::<String>()
            // Collapse multiple spaces
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Classify a batch of texts.
    pub fn batch_classify(&self, texts: &[&str]) -> Vec<ClassificationResult> {
        texts.iter().map(|t| self.classify(t)).collect()
    }

    /// Return the top-`n` intents by descending score for the given text.
    pub fn top_n(&self, text: &str, n: usize) -> Vec<(Intent, f64)> {
        let normalised = Self::normalize_text(text);
        let mut scores: Vec<(Intent, f64)> = self
            .rules
            .iter()
            .map(|rule| {
                let (score, _, _) = self.score_rule(rule, &normalised);
                let confidence = (score / 10.0_f64).min(1.0);
                (rule.intent.clone(), confidence)
            })
            .collect();

        // Sort descending by score
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(n);
        scores
    }
}

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

/// Check if `word` appears as a whole token in `text` (space-delimited).
fn contains_word(text: &str, word: &str) -> bool {
    text.split_whitespace().any(|t| t == word)
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn clf() -> IntentClassifier {
        IntentClassifier::new()
    }

    // ── normalize_text ─────────────────────────────────────────────────

    #[test]
    fn test_normalize_lowercase() {
        assert_eq!(
            IntentClassifier::normalize_text("HELLO World"),
            "hello world"
        );
    }

    #[test]
    fn test_normalize_strips_punctuation() {
        let n = IntentClassifier::normalize_text("Hello, world!");
        assert!(!n.contains(','));
        assert!(!n.contains('!'));
    }

    #[test]
    fn test_normalize_collapses_spaces() {
        let n = IntentClassifier::normalize_text("foo   bar");
        assert_eq!(n, "foo bar");
    }

    #[test]
    fn test_normalize_empty() {
        assert_eq!(IntentClassifier::normalize_text(""), "");
    }

    // ── classify — SparqlQuery ──────────────────────────────────────────

    #[test]
    fn test_classify_sparql_select() {
        let result = clf().classify("SELECT * WHERE { ?s ?p ?o }");
        assert_eq!(result.intent, Intent::SparqlQuery);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_classify_sparql_ask() {
        let result = clf().classify("ask where the subject is defined");
        assert_eq!(result.intent, Intent::SparqlQuery);
    }

    #[test]
    fn test_classify_sparql_prefix() {
        let result = clf().classify("Can you run a sparql query for me?");
        assert_eq!(result.intent, Intent::SparqlQuery);
    }

    // ── classify — RdfInsert ────────────────────────────────────────────

    #[test]
    fn test_classify_rdf_insert() {
        let result = clf().classify("INSERT DATA { <s> <p> <o> }");
        assert_eq!(result.intent, Intent::RdfInsert);
    }

    #[test]
    fn test_classify_add_triple() {
        let result = clf().classify("add triple <a> <b> <c>");
        // "add" and "triple" both match RdfInsert keywords
        assert_eq!(result.intent, Intent::RdfInsert);
    }

    // ── classify — RdfDelete ────────────────────────────────────────────

    #[test]
    fn test_classify_rdf_delete() {
        let result = clf().classify("DELETE DATA { <s> <p> <o> }");
        assert_eq!(result.intent, Intent::RdfDelete);
    }

    #[test]
    fn test_classify_remove_triple() {
        let result = clf().classify("remove triple <a> <b>");
        assert_eq!(result.intent, Intent::RdfDelete);
    }

    // ── classify — SchemaQuestion ───────────────────────────────────────

    #[test]
    fn test_classify_schema_question_what_is() {
        let result = clf().classify("what is the ontology for Person class?");
        assert_eq!(result.intent, Intent::SchemaQuestion);
    }

    #[test]
    fn test_classify_schema_class() {
        let result = clf().classify("define the class hierarchy");
        assert_eq!(result.intent, Intent::SchemaQuestion);
    }

    // ── classify — FactLookup ───────────────────────────────────────────

    #[test]
    fn test_classify_fact_lookup_who() {
        let result = clf().classify("who created this resource");
        assert_eq!(result.intent, Intent::FactLookup);
    }

    #[test]
    fn test_classify_fact_lookup_find() {
        let result = clf().classify("find all authors in the graph");
        assert_eq!(result.intent, Intent::FactLookup);
    }

    #[test]
    fn test_classify_fact_lookup_list() {
        let result = clf().classify("list all available datasets");
        assert_eq!(result.intent, Intent::FactLookup);
    }

    // ── classify — Help ──────────────────────────────────────────────────

    #[test]
    fn test_classify_help() {
        let result = clf().classify("help me with SPARQL");
        assert_eq!(result.intent, Intent::Help);
    }

    #[test]
    fn test_classify_how_to() {
        let result = clf().classify("how to write a SPARQL query");
        assert_eq!(result.intent, Intent::Help);
    }

    // ── classify — Greeting ─────────────────────────────────────────────

    #[test]
    fn test_classify_hello() {
        let result = clf().classify("Hello!");
        assert_eq!(result.intent, Intent::Greeting);
    }

    #[test]
    fn test_classify_good_morning() {
        let result = clf().classify("Good morning everyone");
        assert_eq!(result.intent, Intent::Greeting);
    }

    #[test]
    fn test_classify_hi() {
        let result = clf().classify("Hi there");
        assert_eq!(result.intent, Intent::Greeting);
    }

    // ── classify — Farewell ─────────────────────────────────────────────

    #[test]
    fn test_classify_bye() {
        let result = clf().classify("Bye for now");
        assert_eq!(result.intent, Intent::Farewell);
    }

    #[test]
    fn test_classify_goodbye() {
        let result = clf().classify("Goodbye see you later");
        assert_eq!(result.intent, Intent::Farewell);
    }

    #[test]
    fn test_classify_thanks() {
        let result = clf().classify("thanks goodbye");
        assert_eq!(result.intent, Intent::Farewell);
    }

    // ── classify — Unknown ──────────────────────────────────────────────

    #[test]
    fn test_classify_unknown() {
        let result = clf().classify("xyzzy plugh");
        assert_eq!(result.intent, Intent::Unknown);
        assert!((result.confidence).abs() < 1e-9);
    }

    // ── classify confidence ─────────────────────────────────────────────

    #[test]
    fn test_classify_confidence_in_range() {
        let result = clf().classify("SELECT * WHERE { ?s ?p ?o }");
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    // ── add_rule ────────────────────────────────────────────────────────

    #[test]
    fn test_add_custom_rule() {
        let mut c = clf();
        c.add_rule(IntentRule::new(Intent::Navigation, &["teleport"], &[], 5.0));
        let result = c.classify("teleport me to the knowledge graph");
        assert_eq!(result.intent, Intent::Navigation);
    }

    // ── matched_keywords ────────────────────────────────────────────────

    #[test]
    fn test_matched_keywords_non_empty() {
        let result = clf().classify("select all triples where subject is known");
        assert!(!result.matched_keywords.is_empty());
    }

    // ── batch_classify ──────────────────────────────────────────────────

    #[test]
    fn test_batch_classify_correct_length() {
        let texts = vec!["hello", "select * where {?s ?p ?o}", "bye"];
        let results = clf().batch_classify(&texts);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_batch_classify_intents() {
        let texts = vec!["hello", "SELECT * WHERE { ?s ?p ?o }"];
        let results = clf().batch_classify(&texts);
        assert_eq!(results[0].intent, Intent::Greeting);
        assert_eq!(results[1].intent, Intent::SparqlQuery);
    }

    // ── top_n ────────────────────────────────────────────────────────────

    #[test]
    fn test_top_n_correct_length() {
        let result = clf().top_n("select where sparql", 3);
        assert!(result.len() <= 3);
    }

    #[test]
    fn test_top_n_sorted_descending() {
        let result = clf().top_n("select all from where construct ask", 5);
        for w in result.windows(2) {
            assert!(w[0].1 >= w[1].1, "not sorted: {:?} vs {:?}", w[0], w[1]);
        }
    }

    #[test]
    fn test_top_n_zero() {
        let result = clf().top_n("hello world", 0);
        assert!(result.is_empty());
    }

    // ── score_rule ────────────────────────────────────────────────────────

    #[test]
    fn test_score_rule_keyword_match() {
        let c = clf();
        let rule = IntentRule::new(Intent::Help, &["help"], &[], 1.0);
        let (score, kws, pats) = c.score_rule(&rule, "help me please");
        assert!(score > 0.0);
        assert!(kws.contains(&"help".to_string()));
        assert!(pats.is_empty());
    }

    #[test]
    fn test_score_rule_pattern_match() {
        let c = clf();
        let rule = IntentRule::new(Intent::Help, &[], &["how to"], 1.0);
        let (score, kws, pats) = c.score_rule(&rule, "how to write a query");
        assert!(score > 0.0);
        assert!(kws.is_empty());
        assert!(!pats.is_empty());
    }

    #[test]
    fn test_score_rule_no_match() {
        let c = clf();
        let rule = IntentRule::new(Intent::Greeting, &["hello"], &[], 1.0);
        let (score, _, _) = c.score_rule(&rule, "delete the graph");
        assert!((score).abs() < 1e-9);
    }

    // ── intent display ────────────────────────────────────────────────────

    #[test]
    fn test_intent_display() {
        assert_eq!(format!("{}", Intent::SparqlQuery), "SparqlQuery");
        assert_eq!(format!("{}", Intent::Unknown), "Unknown");
    }
}
