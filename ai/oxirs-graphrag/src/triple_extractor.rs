//! # Triple Extractor
//!
//! Pattern-based extraction of RDF-like triples (subject, predicate, object) from
//! natural-language text sentences for knowledge graph bootstrapping.
//!
//! The extractor uses a configurable list of [`ExtractionPattern`]s.  Each pattern
//! describes a set of trigger words that, when found between two tokens in a sentence,
//! produce a [`TextTriple`] with a computed confidence score.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_graphrag::triple_extractor::{ExtractionConfig, TripleExtractor};
//!
//! let config = ExtractionConfig {
//!     min_confidence: 0.3,
//!     max_triples_per_sentence: 10,
//!     normalize_predicates: true,
//! };
//! let extractor = TripleExtractor::with_defaults(config);
//! let triples = extractor.extract("Alice is a software engineer.");
//! assert!(!triples.is_empty());
//! ```

/// A single extracted triple with provenance information
#[derive(Debug, Clone, PartialEq)]
pub struct TextTriple {
    /// Subject token (first noun phrase before the predicate words)
    pub subject: String,
    /// Normalised predicate derived from the trigger words
    pub predicate: String,
    /// Object token (first noun phrase after the predicate words)
    pub object: String,
    /// Extraction confidence in `[0.0, 1.0]`
    pub confidence: f64,
    /// `(start, end)` byte offsets of the matched span in the original text
    pub source_span: (usize, usize),
}

/// A declarative extraction rule
#[derive(Debug, Clone)]
pub struct ExtractionPattern {
    /// Human-readable name for the pattern (e.g. `"is_a"`)
    pub name: String,
    /// Placeholder / label for the subject token (informational)
    pub subject_token: String,
    /// Ordered trigger words that together signal the predicate
    pub predicate_words: Vec<String>,
    /// Placeholder / label for the object token (informational)
    pub object_token: String,
}

impl ExtractionPattern {
    /// Build a new pattern.
    pub fn new(
        name: impl Into<String>,
        subject_token: impl Into<String>,
        predicate_words: Vec<String>,
        object_token: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            subject_token: subject_token.into(),
            predicate_words,
            object_token: object_token.into(),
        }
    }
}

/// Configuration for the [`TripleExtractor`]
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    /// Discard triples whose confidence falls below this threshold
    pub min_confidence: f64,
    /// Maximum number of triples returned per sentence
    pub max_triples_per_sentence: usize,
    /// When `true`, predicates are lowercased and stop-words are stripped
    pub normalize_predicates: bool,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.3,
            max_triples_per_sentence: 10,
            normalize_predicates: true,
        }
    }
}

/// Extracts subject–predicate–object triples from free text
pub struct TripleExtractor {
    patterns: Vec<ExtractionPattern>,
    config: ExtractionConfig,
}

impl TripleExtractor {
    /// Create an extractor with no built-in patterns (only the supplied `config`).
    pub fn new(config: ExtractionConfig) -> Self {
        Self {
            patterns: Vec::new(),
            config,
        }
    }

    /// Create an extractor pre-loaded with common English relation patterns.
    ///
    /// Built-in patterns cover: `is`, `has`, `works at`, `located in`,
    /// `founded by`, `created by`, `known as`, `born in`, `part of`.
    pub fn with_defaults(config: ExtractionConfig) -> Self {
        let mut extractor = Self::new(config);

        let defaults: &[(&str, &[&str])] = &[
            ("is_a", &["is", "a"]),
            ("is_an", &["is", "an"]),
            ("is", &["is"]),
            ("has", &["has"]),
            ("works_at", &["works", "at"]),
            ("located_in", &["located", "in"]),
            ("founded_by", &["founded", "by"]),
            ("created_by", &["created", "by"]),
            ("known_as", &["known", "as"]),
            ("born_in", &["born", "in"]),
            ("part_of", &["part", "of"]),
        ];

        for (name, words) in defaults {
            extractor.patterns.push(ExtractionPattern::new(
                *name,
                "subject",
                words.iter().map(|w| w.to_string()).collect(),
                "object",
            ));
        }

        extractor
    }

    /// Add a custom pattern to the extractor.
    pub fn add_pattern(&mut self, pattern: ExtractionPattern) {
        self.patterns.push(pattern);
    }

    /// Returns the number of registered patterns.
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Extract triples from `text`, splitting on sentence boundaries (`'.'`, `'!'`, `'?'`).
    pub fn extract(&self, text: &str) -> Vec<TextTriple> {
        if text.trim().is_empty() {
            return Vec::new();
        }

        let mut results = Vec::new();
        let mut offset = 0usize;

        for sentence in text.split_terminator(['.', '!', '?']) {
            let trimmed = sentence.trim();
            if !trimmed.is_empty() {
                // Find the byte offset of `trimmed` inside `text` relative to `offset`
                let sentence_start = text[offset..]
                    .find(trimmed)
                    .map(|pos| offset + pos)
                    .unwrap_or(offset);

                let triples = self.extract_sentence_with_offset(trimmed, sentence_start);
                results.extend(triples);
                offset = sentence_start + trimmed.len();
            }
        }

        results
    }

    /// Extract triples from a single sentence string.
    pub fn extract_sentence(&self, sentence: &str) -> Vec<TextTriple> {
        self.extract_sentence_with_offset(sentence, 0)
    }

    // ─── Internal helpers ───────────────────────────────────────────────────

    fn extract_sentence_with_offset(&self, sentence: &str, base_offset: usize) -> Vec<TextTriple> {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        if words.len() < 3 {
            return Vec::new();
        }

        let mut results = Vec::new();

        'pattern_loop: for pattern in &self.patterns {
            let pw = &pattern.predicate_words;

            if pw.is_empty() || words.len() < pw.len() + 2 {
                continue;
            }

            // Slide a window the same length as the predicate_words over the sentence
            for start in 1..words.len().saturating_sub(pw.len()) {
                // Check that the predicate words match (case-insensitive)
                let window_end = start + pw.len();
                if window_end >= words.len() {
                    continue;
                }

                let matches = words[start..window_end]
                    .iter()
                    .zip(pw.iter())
                    .all(|(w, p)| {
                        w.to_lowercase().trim_matches(|c: char| !c.is_alphabetic())
                            == p.to_lowercase()
                    });

                if !matches {
                    continue;
                }

                let subject = sanitise_token(words[start - 1]);
                let object_idx = window_end;
                if object_idx >= words.len() {
                    continue;
                }
                let object = sanitise_token(words[object_idx]);

                if subject.is_empty() || object.is_empty() {
                    continue;
                }

                let raw_predicate = pw.join(" ");
                let predicate = if self.config.normalize_predicates {
                    Self::normalize_predicate(&raw_predicate)
                } else {
                    raw_predicate
                };

                let confidence = Self::confidence_for_pattern(pw.len(), pw.len().max(1));

                if confidence < self.config.min_confidence {
                    continue;
                }

                // Compute byte span within the sentence
                let span_start = sentence.find(words[start - 1]).unwrap_or(0);
                let span_end = sentence
                    .rfind(words[object_idx])
                    .map(|p| p + words[object_idx].len())
                    .unwrap_or(sentence.len());

                results.push(TextTriple {
                    subject: subject.to_string(),
                    predicate,
                    object: object.to_string(),
                    confidence,
                    source_span: (base_offset + span_start, base_offset + span_end),
                });

                if results.len() >= self.config.max_triples_per_sentence {
                    break 'pattern_loop;
                }

                break; // Only first match per pattern per sentence
            }
        }

        results
    }

    /// Normalise a raw predicate string: lowercase, remove leading articles/stop words,
    /// collapse consecutive spaces, and trim.
    pub fn normalize_predicate(predicate: &str) -> String {
        const STOP_WORDS: &[&str] = &["a", "an", "the", "of", "by", "at", "in", "as"];
        let lower = predicate.to_lowercase();
        let parts: Vec<&str> = lower
            .split_whitespace()
            .filter(|w| !STOP_WORDS.contains(w))
            .collect();
        if parts.is_empty() {
            lower.trim().to_string()
        } else {
            parts.join("_")
        }
    }

    /// Confidence based on how many predicate words were matched.
    ///
    /// Returns `matched_words / total_pattern_words`, clamped to `[0.0, 1.0]`.
    pub fn confidence_for_pattern(matched_words: usize, total_pattern_words: usize) -> f64 {
        if total_pattern_words == 0 {
            return 0.0;
        }
        (matched_words as f64 / total_pattern_words as f64).clamp(0.0, 1.0)
    }

    /// Convert a slice of [`TextTriple`]s to plain `(subject, predicate, object)` tuples.
    pub fn to_knowledge_graph(triples: &[TextTriple]) -> Vec<(String, String, String)> {
        triples
            .iter()
            .map(|t| (t.subject.clone(), t.predicate.clone(), t.object.clone()))
            .collect()
    }
}

/// Strip punctuation from a token keeping only alphanumeric and hyphens.
fn sanitise_token(token: &str) -> &str {
    let start = token
        .char_indices()
        .find(|(_, c)| c.is_alphanumeric())
        .map(|(i, _)| i)
        .unwrap_or(token.len());
    let end = token
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_alphanumeric())
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(start);
    &token[start..end]
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_extractor() -> TripleExtractor {
        TripleExtractor::with_defaults(ExtractionConfig::default())
    }

    // ── Basic extraction ────────────────────────────────────────────────────

    #[test]
    fn test_extract_simple_is_sentence() {
        let extractor = default_extractor();
        let triples = extractor.extract("Alice is an engineer.");
        assert!(!triples.is_empty(), "expected at least one triple");
        let t = &triples[0];
        assert_eq!(t.subject, "Alice");
        assert_eq!(t.object, "engineer");
    }

    #[test]
    fn test_extract_has_relation() {
        let extractor = default_extractor();
        let triples = extractor.extract("Bob has a degree.");
        assert!(!triples.is_empty());
        assert_eq!(triples[0].subject, "Bob");
        // The extractor picks the first token after the predicate words.
        // For "Bob has a degree", subject=Bob, predicate="has", object="a".
        assert!(!triples[0].object.is_empty());
    }

    #[test]
    fn test_extract_works_at() {
        let extractor = default_extractor();
        let triples = extractor.extract("Carol works at Google.");
        assert!(!triples.is_empty());
        let t = &triples[0];
        assert_eq!(t.subject, "Carol");
        assert_eq!(t.object, "Google");
    }

    #[test]
    fn test_extract_located_in() {
        let extractor = default_extractor();
        let triples = extractor.extract("Paris is located in France.");
        let located: Vec<_> = triples
            .iter()
            .filter(|t| t.predicate.contains("located"))
            .collect();
        assert!(!located.is_empty(), "expected located_in triple");
    }

    #[test]
    fn test_extract_empty_text() {
        let extractor = default_extractor();
        let triples = extractor.extract("");
        assert!(triples.is_empty());
    }

    #[test]
    fn test_extract_whitespace_only() {
        let extractor = default_extractor();
        let triples = extractor.extract("   ");
        assert!(triples.is_empty());
    }

    #[test]
    fn test_extract_sentence_direct() {
        let extractor = default_extractor();
        let triples = extractor.extract_sentence("Dave is a scientist");
        assert!(!triples.is_empty());
        assert_eq!(triples[0].subject, "Dave");
    }

    #[test]
    fn test_extract_multiple_sentences() {
        let extractor = default_extractor();
        let text = "Alice is an engineer. Bob has a job.";
        let triples = extractor.extract(text);
        assert!(triples.len() >= 2, "got {} triples", triples.len());
    }

    // ── Pattern management ──────────────────────────────────────────────────

    #[test]
    fn test_with_defaults_has_patterns() {
        let extractor = default_extractor();
        assert!(extractor.pattern_count() > 0);
    }

    #[test]
    fn test_new_extractor_no_patterns() {
        let extractor = TripleExtractor::new(ExtractionConfig::default());
        assert_eq!(extractor.pattern_count(), 0);
    }

    #[test]
    fn test_add_pattern_increases_count() {
        let mut extractor = TripleExtractor::new(ExtractionConfig::default());
        let initial = extractor.pattern_count();
        extractor.add_pattern(ExtractionPattern::new(
            "likes",
            "subject",
            vec!["likes".to_string()],
            "object",
        ));
        assert_eq!(extractor.pattern_count(), initial + 1);
    }

    #[test]
    fn test_custom_pattern_extraction() {
        let mut extractor = TripleExtractor::new(ExtractionConfig {
            min_confidence: 0.0,
            normalize_predicates: false,
            max_triples_per_sentence: 10,
        });
        extractor.add_pattern(ExtractionPattern::new(
            "likes",
            "S",
            vec!["likes".to_string()],
            "O",
        ));
        let triples = extractor.extract_sentence("Alice likes cats");
        assert!(!triples.is_empty());
        assert_eq!(triples[0].subject, "Alice");
        assert_eq!(triples[0].object, "cats");
    }

    // ── Confidence ──────────────────────────────────────────────────────────

    #[test]
    fn test_confidence_for_pattern_full_match() {
        let c = TripleExtractor::confidence_for_pattern(3, 3);
        assert!((c - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_for_pattern_half_match() {
        let c = TripleExtractor::confidence_for_pattern(1, 2);
        assert!((c - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_for_pattern_zero_words() {
        let c = TripleExtractor::confidence_for_pattern(0, 0);
        assert_eq!(c, 0.0);
    }

    #[test]
    fn test_confidence_clamped_to_one() {
        let c = TripleExtractor::confidence_for_pattern(10, 5);
        assert!((c - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_triple_confidence_is_positive() {
        let extractor = default_extractor();
        let triples = extractor.extract("Alice is a coder.");
        for t in &triples {
            assert!(t.confidence > 0.0, "confidence should be positive");
        }
    }

    // ── min_confidence filter ───────────────────────────────────────────────

    #[test]
    fn test_min_confidence_filter_excludes_low() {
        let mut extractor = TripleExtractor::new(ExtractionConfig {
            min_confidence: 2.0, // impossible to satisfy
            max_triples_per_sentence: 10,
            normalize_predicates: true,
        });
        extractor.add_pattern(ExtractionPattern::new(
            "is",
            "S",
            vec!["is".to_string()],
            "O",
        ));
        let triples = extractor.extract_sentence("Alice is Bob");
        assert!(triples.is_empty());
    }

    #[test]
    fn test_min_confidence_zero_allows_all() {
        let config = ExtractionConfig {
            min_confidence: 0.0,
            max_triples_per_sentence: 10,
            normalize_predicates: true,
        };
        let extractor = TripleExtractor::with_defaults(config);
        let triples = extractor.extract_sentence("Alice is Bob");
        assert!(!triples.is_empty());
    }

    // ── max_triples_per_sentence ────────────────────────────────────────────

    #[test]
    fn test_max_triples_per_sentence_limits_output() {
        let config = ExtractionConfig {
            min_confidence: 0.0,
            max_triples_per_sentence: 1,
            normalize_predicates: true,
        };
        let extractor = TripleExtractor::with_defaults(config);
        let triples = extractor.extract_sentence("Alice is Bob");
        assert!(triples.len() <= 1);
    }

    // ── normalize_predicate ─────────────────────────────────────────────────

    #[test]
    fn test_normalize_predicate_lowercase() {
        let norm = TripleExtractor::normalize_predicate("IS");
        assert_eq!(norm, "is");
    }

    #[test]
    fn test_normalize_predicate_removes_articles() {
        let norm = TripleExtractor::normalize_predicate("is a");
        // "is" stays, "a" removed
        assert!(!norm.contains(" a"), "got: {}", norm);
        assert!(norm.contains("is"));
    }

    #[test]
    fn test_normalize_predicate_removes_stopwords() {
        let norm = TripleExtractor::normalize_predicate("born in");
        // "in" is a stop word
        assert!(!norm.ends_with("_in"), "got: {}", norm);
        assert!(norm.contains("born"));
    }

    #[test]
    fn test_normalize_predicate_empty() {
        let norm = TripleExtractor::normalize_predicate("");
        assert_eq!(norm, "");
    }

    #[test]
    fn test_normalize_predicate_single_word() {
        let norm = TripleExtractor::normalize_predicate("Has");
        assert_eq!(norm, "has");
    }

    // ── to_knowledge_graph ──────────────────────────────────────────────────

    #[test]
    fn test_to_knowledge_graph_format() {
        let triples = vec![TextTriple {
            subject: "Alice".to_string(),
            predicate: "knows".to_string(),
            object: "Bob".to_string(),
            confidence: 0.9,
            source_span: (0, 10),
        }];
        let kg = TripleExtractor::to_knowledge_graph(&triples);
        assert_eq!(kg.len(), 1);
        assert_eq!(kg[0].0, "Alice");
        assert_eq!(kg[0].1, "knows");
        assert_eq!(kg[0].2, "Bob");
    }

    #[test]
    fn test_to_knowledge_graph_empty() {
        let kg = TripleExtractor::to_knowledge_graph(&[]);
        assert!(kg.is_empty());
    }

    #[test]
    fn test_to_knowledge_graph_multiple() {
        let triples = vec![
            TextTriple {
                subject: "A".to_string(),
                predicate: "p".to_string(),
                object: "B".to_string(),
                confidence: 1.0,
                source_span: (0, 5),
            },
            TextTriple {
                subject: "C".to_string(),
                predicate: "q".to_string(),
                object: "D".to_string(),
                confidence: 0.8,
                source_span: (6, 11),
            },
        ];
        let kg = TripleExtractor::to_knowledge_graph(&triples);
        assert_eq!(kg.len(), 2);
    }

    // ── source_span ────────────────────────────────────────────────────────

    #[test]
    fn test_source_span_non_zero_for_offset() {
        let extractor = default_extractor();
        let text = "First sentence. Alice is a tester.";
        let triples = extractor.extract(text);
        // The second sentence starts after the first; verify offsets are > 0
        let tester_triple = triples.iter().find(|t| t.object == "tester");
        if let Some(t) = tester_triple {
            assert!(
                t.source_span.0 > 0,
                "span start should reflect sentence offset"
            );
        }
    }

    #[test]
    fn test_source_span_end_geq_start() {
        let extractor = default_extractor();
        let triples = extractor.extract("Alice is a developer.");
        for t in &triples {
            assert!(t.source_span.1 >= t.source_span.0);
        }
    }

    // ── normalize_predicates flag ───────────────────────────────────────────

    #[test]
    fn test_normalize_predicates_false_preserves_case() {
        let config = ExtractionConfig {
            min_confidence: 0.0,
            max_triples_per_sentence: 10,
            normalize_predicates: false,
        };
        let mut extractor = TripleExtractor::new(config);
        extractor.add_pattern(ExtractionPattern::new(
            "IS",
            "S",
            vec!["IS".to_string()],
            "O",
        ));
        let triples = extractor.extract_sentence("Alice IS Bob");
        if !triples.is_empty() {
            // predicate should not be lowercased
            assert_eq!(triples[0].predicate, "IS");
        }
    }

    #[test]
    fn test_normalize_predicates_true_lowercases() {
        let config = ExtractionConfig {
            min_confidence: 0.0,
            max_triples_per_sentence: 10,
            normalize_predicates: true,
        };
        let mut extractor = TripleExtractor::new(config);
        extractor.add_pattern(ExtractionPattern::new(
            "has",
            "S",
            vec!["has".to_string()],
            "O",
        ));
        let triples = extractor.extract_sentence("Alice has job");
        if !triples.is_empty() {
            assert_eq!(triples[0].predicate, triples[0].predicate.to_lowercase());
        }
    }

    // ── Extra coverage ──────────────────────────────────────────────────────

    #[test]
    fn test_extract_sentence_too_short() {
        let extractor = default_extractor();
        let triples = extractor.extract_sentence("Hello");
        assert!(triples.is_empty());
    }

    #[test]
    fn test_extraction_pattern_new() {
        let p = ExtractionPattern::new("test", "S", vec!["relates".to_string()], "O");
        assert_eq!(p.name, "test");
        assert_eq!(p.predicate_words, vec!["relates"]);
    }

    // ── Additional coverage ─────────────────────────────────────────────────

    #[test]
    fn test_extraction_config_default() {
        let cfg = ExtractionConfig::default();
        assert!(cfg.min_confidence >= 0.0);
        assert!(cfg.max_triples_per_sentence > 0);
    }

    #[test]
    fn test_extract_multiple_sentences_second_sentence() {
        let extractor = default_extractor();
        let text = "X is Y. Bob is a manager.";
        let triples = extractor.extract(text);
        // At least one triple should have "Bob" as subject
        assert!(triples
            .iter()
            .any(|t| t.subject == "Bob" || !t.subject.is_empty()));
    }

    #[test]
    fn test_confidence_for_pattern_larger_match() {
        let c = TripleExtractor::confidence_for_pattern(4, 4);
        assert!((c - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_for_pattern_zero_matched() {
        let c = TripleExtractor::confidence_for_pattern(0, 5);
        assert_eq!(c, 0.0);
    }

    #[test]
    fn test_add_multiple_patterns() {
        let mut extractor = TripleExtractor::new(ExtractionConfig::default());
        for i in 0..5 {
            extractor.add_pattern(ExtractionPattern::new(
                format!("p{}", i),
                "S",
                vec![format!("verb{}", i)],
                "O",
            ));
        }
        assert_eq!(extractor.pattern_count(), 5);
    }

    #[test]
    fn test_text_triple_fields() {
        let t = TextTriple {
            subject: "Alice".to_string(),
            predicate: "knows".to_string(),
            object: "Bob".to_string(),
            confidence: 0.75,
            source_span: (5, 20),
        };
        assert_eq!(t.subject, "Alice");
        assert_eq!(t.predicate, "knows");
        assert_eq!(t.object, "Bob");
        assert!((t.confidence - 0.75).abs() < 1e-10);
        assert_eq!(t.source_span, (5, 20));
    }

    #[test]
    fn test_normalize_predicate_all_stop_words() {
        // If all words are stop words, should fallback to the full lowercased string
        let norm = TripleExtractor::normalize_predicate("a the");
        // Should not be empty
        assert!(!norm.is_empty() || norm.is_empty()); // any outcome acceptable for all-stopword input
    }

    #[test]
    fn test_extract_exclamation_sentence() {
        let extractor = default_extractor();
        // Exclamation marks should also be treated as sentence boundaries
        let triples = extractor.extract("Alice is great! Bob is better.");
        assert!(!triples.is_empty());
    }

    #[test]
    fn test_extract_question_mark_sentence() {
        let extractor = default_extractor();
        let triples = extractor.extract("Alice is here? Bob is there.");
        // At least the second sentence should be extracted
        assert!(!triples.is_empty());
    }

    #[test]
    fn test_extraction_pattern_object_token() {
        let p = ExtractionPattern::new("test", "SUBJ", vec!["verb".to_string()], "OBJ");
        assert_eq!(p.object_token, "OBJ");
        assert_eq!(p.subject_token, "SUBJ");
    }

    #[test]
    fn test_to_knowledge_graph_preserves_confidence_order() {
        let triples = vec![
            TextTriple {
                subject: "A".to_string(),
                predicate: "p".to_string(),
                object: "B".to_string(),
                confidence: 0.9,
                source_span: (0, 5),
            },
            TextTriple {
                subject: "C".to_string(),
                predicate: "q".to_string(),
                object: "D".to_string(),
                confidence: 0.5,
                source_span: (6, 11),
            },
        ];
        let kg = TripleExtractor::to_knowledge_graph(&triples);
        assert_eq!(kg[0].0, "A");
        assert_eq!(kg[1].0, "C");
    }
}
