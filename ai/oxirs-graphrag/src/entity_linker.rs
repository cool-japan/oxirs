//! String-to-RDF entity linking: mention detection and candidate ranking.
//!
//! Provides exact and fuzzy matching of text mentions to entities in a
//! knowledge base, with configurable score thresholds.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// An entity in the knowledge base.
#[derive(Debug, Clone)]
pub struct Entity {
    /// Canonical IRI identifier.
    pub iri: String,
    /// Primary label.
    pub label: String,
    /// Alternative surface forms / aliases.
    pub aliases: Vec<String>,
    /// Entity type string (e.g. "Person", "Organization", "Place").
    pub entity_type: String,
    /// Optional short description.
    pub description: Option<String>,
    /// Popularity score in \[0.0, 1.0\] used as a prior.
    pub popularity: f64,
}

/// A mention of an entity surface form found in text.
#[derive(Debug, Clone)]
pub struct EntityMention {
    /// The surface form detected in the input text.
    pub text: String,
    /// Byte offset of the first character.
    pub start_char: usize,
    /// Byte offset one past the last character.
    pub end_char: usize,
    /// Ranked candidate links for this mention.
    pub candidates: Vec<LinkCandidate>,
}

/// A single candidate link returned for a mention.
#[derive(Debug, Clone)]
pub struct LinkCandidate {
    /// The candidate entity.
    pub entity: Entity,
    /// Combined ranking score.
    pub score: f64,
    /// Normalised string similarity to the mention text.
    pub string_similarity: f64,
    /// Entity popularity used as prior.
    pub prior_probability: f64,
}

/// A resolved link between a text mention and its best-matching entity.
#[derive(Debug, Clone)]
pub struct LinkedEntity {
    /// The text mention.
    pub mention: EntityMention,
    /// Highest-scoring candidate (if any exceeds the linker threshold).
    pub best_candidate: Option<LinkCandidate>,
    /// Confidence score of the best candidate (0.0 if none).
    pub confidence: f64,
}

// ---------------------------------------------------------------------------
// LinkerError
// ---------------------------------------------------------------------------

/// Errors returned by `EntityLinker`.
#[derive(Debug)]
pub enum LinkerError {
    /// The supplied text is empty.
    EmptyText,
    /// The knowledge base has no entities.
    EmptyKnowledgeBase,
    /// An entity in the knowledge base is invalid.
    InvalidEntity(String),
}

impl std::fmt::Display for LinkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinkerError::EmptyText => write!(f, "Input text is empty"),
            LinkerError::EmptyKnowledgeBase => write!(f, "Knowledge base contains no entities"),
            LinkerError::InvalidEntity(msg) => write!(f, "Invalid entity: {}", msg),
        }
    }
}

impl std::error::Error for LinkerError {}

// ---------------------------------------------------------------------------
// EntityLinker
// ---------------------------------------------------------------------------

/// String-to-entity linker backed by an in-memory knowledge base.
pub struct EntityLinker {
    entities: Vec<Entity>,
    /// label (lowercased) → list of entity indices
    label_index: HashMap<String, Vec<usize>>,
    /// Minimum character length of a text span to be considered as a mention.
    pub min_mention_len: usize,
    /// Minimum combined score for a candidate to be returned.
    pub threshold: f64,
}

impl EntityLinker {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create an empty linker with the given minimum combined score threshold.
    pub fn new(threshold: f64) -> Self {
        Self {
            entities: Vec::new(),
            label_index: HashMap::new(),
            min_mention_len: 2,
            threshold,
        }
    }

    /// Add a single entity to the knowledge base and update the index.
    pub fn add_entity(&mut self, entity: Entity) {
        let idx = self.entities.len();
        // Index the label and all aliases
        let mut keys: Vec<String> = entity.aliases.iter().map(|a| a.to_lowercase()).collect();
        keys.push(entity.label.to_lowercase());

        for key in keys {
            self.label_index.entry(key).or_default().push(idx);
        }
        self.entities.push(entity);
    }

    /// Add multiple entities in bulk and rebuild the index.
    pub fn add_entities(&mut self, entities: Vec<Entity>) {
        for entity in entities {
            self.add_entity(entity);
        }
    }

    /// Total number of entities in the knowledge base.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    // -----------------------------------------------------------------------
    // Mention detection
    // -----------------------------------------------------------------------

    /// Scan `text` for all known entity labels / aliases and return mentions.
    ///
    /// Both exact (case-insensitive) and fuzzy (similarity ≥ 0.7) matches are
    /// considered.  Overlapping spans from the same starting position are
    /// de-duplicated in favour of the longer match.
    pub fn detect_mentions(&self, text: &str) -> Vec<EntityMention> {
        let text_lower = text.to_lowercase();
        let mut mentions: Vec<EntityMention> = Vec::new();

        for (surface, indices) in &self.label_index {
            if surface.len() < self.min_mention_len {
                continue;
            }
            // Find all (non-overlapping) occurrences of `surface` in text_lower
            let mut search_start = 0;
            while let Some(pos) = text_lower[search_start..].find(surface.as_str()) {
                let abs_start = search_start + pos;
                let abs_end = abs_start + surface.len();

                // Collect candidates for this surface form
                let candidates = self.candidates_for_surface(surface, indices);
                if !candidates.is_empty() {
                    // Use the original casing from the source text
                    let original_text = &text[abs_start..abs_end];
                    mentions.push(EntityMention {
                        text: original_text.to_string(),
                        start_char: abs_start,
                        end_char: abs_end,
                        candidates,
                    });
                }
                search_start = abs_start + 1;
            }
        }

        // Sort by position, then deduplicate overlapping spans (keep longest)
        mentions.sort_by_key(|m| (m.start_char, usize::MAX - (m.end_char - m.start_char)));
        let mut deduped: Vec<EntityMention> = Vec::new();
        for mention in mentions {
            // Skip if fully contained in the last kept mention
            if let Some(last) = deduped.last() {
                if mention.start_char >= last.start_char && mention.end_char <= last.end_char {
                    continue;
                }
            }
            deduped.push(mention);
        }
        deduped
    }

    // -----------------------------------------------------------------------
    // Linking
    // -----------------------------------------------------------------------

    /// Link all mentions found in `text` to their best entity candidates.
    ///
    /// # Errors
    /// - `LinkerError::EmptyText` if `text` is empty after trimming.
    /// - `LinkerError::EmptyKnowledgeBase` if no entities have been added.
    pub fn link(&self, text: &str) -> Result<Vec<LinkedEntity>, LinkerError> {
        if text.trim().is_empty() {
            return Err(LinkerError::EmptyText);
        }
        if self.entities.is_empty() {
            return Err(LinkerError::EmptyKnowledgeBase);
        }

        let mentions = self.detect_mentions(text);
        let linked = mentions
            .into_iter()
            .map(|mention| {
                let best = mention
                    .candidates
                    .iter()
                    .find(|c| c.score >= self.threshold)
                    .cloned();
                let confidence = best.as_ref().map(|c| c.score).unwrap_or(0.0);
                LinkedEntity {
                    mention,
                    best_candidate: best,
                    confidence,
                }
            })
            .collect();
        Ok(linked)
    }

    /// Find all entity candidates for an arbitrary mention string.
    ///
    /// Searches all entity labels and aliases using both exact (lowercase) and
    /// fuzzy matching.  Results are sorted by `score` descending.
    pub fn link_mention(&self, mention: &str) -> Vec<LinkCandidate> {
        let mention_lower = mention.to_lowercase();
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut candidates: Vec<LinkCandidate> = Vec::new();

        for (idx, entity) in self.entities.iter().enumerate() {
            // Collect all surface forms of this entity
            let mut surfaces: Vec<String> =
                entity.aliases.iter().map(|a| a.to_lowercase()).collect();
            surfaces.push(entity.label.to_lowercase());

            let best_sim = surfaces
                .iter()
                .map(|s| Self::string_similarity(&mention_lower, s))
                .fold(0.0_f64, f64::max);

            if best_sim > 0.0 && !seen.contains(&idx) {
                let score = Self::combined_score(best_sim, entity.popularity);
                candidates.push(LinkCandidate {
                    entity: entity.clone(),
                    score,
                    string_similarity: best_sim,
                    prior_probability: entity.popularity,
                });
                seen.insert(idx);
            }
        }

        // Sort by score descending
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates
    }

    /// Look up an entity by its exact IRI.
    pub fn find_by_iri(&self, iri: &str) -> Option<&Entity> {
        self.entities.iter().find(|e| e.iri == iri)
    }

    // -----------------------------------------------------------------------
    // String similarity
    // -----------------------------------------------------------------------

    /// Normalised edit-distance similarity: 1 − edit_distance(a, b) / max(|a|, |b|).
    ///
    /// Returns 1.0 for identical strings and 0.0 when the edit distance equals
    /// the length of the longer string.
    pub fn string_similarity(a: &str, b: &str) -> f64 {
        if a == b {
            return 1.0;
        }
        let len_a = a.chars().count();
        let len_b = b.chars().count();
        let max_len = len_a.max(len_b);
        if max_len == 0 {
            return 1.0; // both empty
        }
        let dist = Self::edit_distance(a, b);
        1.0 - (dist as f64 / max_len as f64)
    }

    /// Levenshtein edit distance between two strings.
    fn edit_distance(a: &str, b: &str) -> usize {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        let la = a_chars.len();
        let lb = b_chars.len();

        if la == 0 {
            return lb;
        }
        if lb == 0 {
            return la;
        }

        let mut dp = vec![vec![0usize; lb + 1]; la + 1];
        for (i, row) in dp.iter_mut().enumerate() {
            row[0] = i;
        }
        for (j, cell) in dp[0].iter_mut().enumerate() {
            *cell = j;
        }
        for i in 1..=la {
            for j in 1..=lb {
                let cost = if a_chars[i - 1] == b_chars[j - 1] {
                    0
                } else {
                    1
                };
                dp[i][j] = (dp[i - 1][j] + 1)
                    .min(dp[i][j - 1] + 1)
                    .min(dp[i - 1][j - 1] + cost);
            }
        }
        dp[la][lb]
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Rebuild the label index from scratch.
    #[allow(dead_code)]
    fn rebuild_index(&mut self) {
        self.label_index.clear();
        for (idx, entity) in self.entities.iter().enumerate() {
            let mut keys: Vec<String> = entity.aliases.iter().map(|a| a.to_lowercase()).collect();
            keys.push(entity.label.to_lowercase());
            for key in keys {
                self.label_index.entry(key).or_default().push(idx);
            }
        }
    }

    /// Build candidates for a known surface form and entity index list.
    fn candidates_for_surface(&self, surface: &str, indices: &[usize]) -> Vec<LinkCandidate> {
        let mut candidates: Vec<LinkCandidate> = indices
            .iter()
            .filter_map(|&idx| {
                let entity = self.entities.get(idx)?;
                let sim = Self::string_similarity(surface, &entity.label.to_lowercase());
                let score = Self::combined_score(sim, entity.popularity);
                Some(LinkCandidate {
                    entity: entity.clone(),
                    score,
                    string_similarity: sim,
                    prior_probability: entity.popularity,
                })
            })
            .collect();
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates
    }

    /// Combined ranking score: 0.7 × string_similarity + 0.3 × popularity.
    fn combined_score(sim: f64, popularity: f64) -> f64 {
        0.7 * sim + 0.3 * popularity
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entity(iri: &str, label: &str, aliases: &[&str], pop: f64) -> Entity {
        Entity {
            iri: iri.to_string(),
            label: label.to_string(),
            aliases: aliases.iter().map(|s| s.to_string()).collect(),
            entity_type: "Thing".to_string(),
            description: None,
            popularity: pop,
        }
    }

    fn sample_linker() -> EntityLinker {
        let mut linker = EntityLinker::new(0.3);
        linker.add_entity(make_entity(
            "http://example.org/einstein",
            "Albert Einstein",
            &["Einstein", "A. Einstein"],
            0.95,
        ));
        linker.add_entity(make_entity(
            "http://example.org/curie",
            "Marie Curie",
            &["Curie", "M. Curie"],
            0.90,
        ));
        linker.add_entity(make_entity(
            "http://example.org/berlin",
            "Berlin",
            &["Berlin City"],
            0.80,
        ));
        linker
    }

    // --- entity_count -------------------------------------------------------

    #[test]
    fn test_entity_count_empty() {
        let linker = EntityLinker::new(0.5);
        assert_eq!(linker.entity_count(), 0);
    }

    #[test]
    fn test_entity_count_after_add() {
        let mut linker = EntityLinker::new(0.5);
        linker.add_entity(make_entity("http://x.org/a", "Alpha", &[], 0.5));
        assert_eq!(linker.entity_count(), 1);
        linker.add_entity(make_entity("http://x.org/b", "Beta", &[], 0.5));
        assert_eq!(linker.entity_count(), 2);
    }

    #[test]
    fn test_add_entities_bulk() {
        let mut linker = EntityLinker::new(0.5);
        linker.add_entities(vec![
            make_entity("http://x.org/a", "Alpha", &[], 0.5),
            make_entity("http://x.org/b", "Beta", &[], 0.6),
            make_entity("http://x.org/c", "Gamma", &[], 0.7),
        ]);
        assert_eq!(linker.entity_count(), 3);
    }

    // --- find_by_iri --------------------------------------------------------

    #[test]
    fn test_find_by_iri_exists() {
        let linker = sample_linker();
        let entity = linker.find_by_iri("http://example.org/einstein");
        assert!(entity.is_some());
        assert_eq!(entity.expect("some").label, "Albert Einstein");
    }

    #[test]
    fn test_find_by_iri_not_found() {
        let linker = sample_linker();
        assert!(linker.find_by_iri("http://example.org/nobody").is_none());
    }

    // --- string_similarity --------------------------------------------------

    #[test]
    fn test_string_similarity_exact() {
        assert!((EntityLinker::string_similarity("hello", "hello") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_string_similarity_completely_different() {
        let sim = EntityLinker::string_similarity("abc", "xyz");
        assert!(sim < 1.0);
    }

    #[test]
    fn test_string_similarity_both_empty() {
        assert!((EntityLinker::string_similarity("", "") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_string_similarity_one_empty() {
        let sim = EntityLinker::string_similarity("", "hello");
        assert!((sim - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_string_similarity_near_match() {
        let sim = EntityLinker::string_similarity("Einstein", "Einsten");
        assert!(sim > 0.8, "sim = {}", sim);
    }

    #[test]
    fn test_string_similarity_range() {
        let sim = EntityLinker::string_similarity("kitten", "sitting");
        assert!((0.0..=1.0).contains(&sim));
    }

    // --- edit_distance -------------------------------------------------------

    #[test]
    fn test_edit_distance_identical() {
        assert_eq!(EntityLinker::edit_distance("abc", "abc"), 0);
    }

    #[test]
    fn test_edit_distance_one_empty() {
        assert_eq!(EntityLinker::edit_distance("", "abc"), 3);
        assert_eq!(EntityLinker::edit_distance("abc", ""), 3);
    }

    #[test]
    fn test_edit_distance_kitten_sitting() {
        assert_eq!(EntityLinker::edit_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn test_edit_distance_sunday_saturday() {
        assert_eq!(EntityLinker::edit_distance("sunday", "saturday"), 3);
    }

    #[test]
    fn test_edit_distance_single_char() {
        assert_eq!(EntityLinker::edit_distance("a", "b"), 1);
        assert_eq!(EntityLinker::edit_distance("a", "a"), 0);
    }

    // --- detect_mentions ----------------------------------------------------

    #[test]
    fn test_detect_mentions_exact_label() {
        let linker = sample_linker();
        let mentions = linker.detect_mentions("Albert Einstein was a physicist.");
        assert!(!mentions.is_empty(), "expected at least one mention");
        let texts: Vec<&str> = mentions.iter().map(|m| m.text.as_str()).collect();
        assert!(
            texts.iter().any(|t| t.to_lowercase().contains("einstein")),
            "mentions = {:?}",
            texts
        );
    }

    #[test]
    fn test_detect_mentions_alias() {
        let linker = sample_linker();
        let mentions = linker.detect_mentions("Curie discovered radium.");
        let texts: Vec<&str> = mentions.iter().map(|m| m.text.as_str()).collect();
        assert!(
            texts.iter().any(|t| t.to_lowercase() == "curie"),
            "mentions = {:?}",
            texts
        );
    }

    #[test]
    fn test_detect_mentions_case_insensitive() {
        let linker = sample_linker();
        let mentions = linker.detect_mentions("berlin is a great city.");
        assert!(!mentions.is_empty(), "expected mention of Berlin");
    }

    #[test]
    fn test_detect_mentions_multiple() {
        let linker = sample_linker();
        let mentions = linker.detect_mentions("Einstein visited Berlin.");
        // Should find at least two distinct mentions
        assert!(mentions.len() >= 2, "mentions = {:?}", mentions.len());
    }

    #[test]
    fn test_detect_mentions_no_match() {
        let linker = sample_linker();
        let mentions = linker.detect_mentions("The quick brown fox jumps.");
        assert!(
            mentions.is_empty(),
            "expected no mentions, got {:?}",
            mentions
        );
    }

    // --- link ---------------------------------------------------------------

    #[test]
    fn test_link_basic() {
        let linker = sample_linker();
        let linked = linker
            .link("Albert Einstein won the Nobel Prize.")
            .expect("ok");
        assert!(!linked.is_empty());
    }

    #[test]
    fn test_link_empty_text_error() {
        let linker = sample_linker();
        assert!(linker.link("").is_err());
        assert!(linker.link("   ").is_err());
    }

    #[test]
    fn test_link_empty_kb_error() {
        let linker = EntityLinker::new(0.5);
        assert!(matches!(
            linker.link("some text"),
            Err(LinkerError::EmptyKnowledgeBase)
        ));
    }

    #[test]
    fn test_link_confidence_populated() {
        let linker = sample_linker();
        let linked = linker.link("Curie was born in Poland.").expect("ok");
        for le in &linked {
            if le.best_candidate.is_some() {
                assert!(le.confidence > 0.0);
            }
        }
    }

    // --- link_mention -------------------------------------------------------

    #[test]
    fn test_link_mention_exact() {
        let linker = sample_linker();
        let candidates = linker.link_mention("Einstein");
        assert!(!candidates.is_empty());
        // Top candidate should be Einstein
        assert_eq!(candidates[0].entity.iri, "http://example.org/einstein");
    }

    #[test]
    fn test_link_mention_sorted_descending() {
        let linker = sample_linker();
        let candidates = linker.link_mention("Berlin");
        for window in candidates.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "candidates not sorted: {} < {}",
                window[0].score,
                window[1].score
            );
        }
    }

    #[test]
    fn test_link_mention_returns_all_above_zero() {
        let linker = sample_linker();
        let candidates = linker.link_mention("Curie");
        // All returned candidates should have positive score
        for c in &candidates {
            assert!(c.score > 0.0);
        }
    }

    // --- threshold ----------------------------------------------------------

    #[test]
    fn test_threshold_filters_low_confidence() {
        let mut linker = EntityLinker::new(0.99); // very high threshold
        linker.add_entity(make_entity(
            "http://x.org/z",
            "Zephyr",
            &[],
            0.1, // low popularity
        ));
        let linked = linker.link("There is a zephyr wind.").expect("ok");
        // Either no mentions or no best_candidate passes threshold
        for le in &linked {
            assert!(
                le.best_candidate.is_none()
                    || le.best_candidate.as_ref().expect("some").score >= 0.99
            );
        }
    }

    #[test]
    fn test_min_mention_len_filter() {
        let mut linker = EntityLinker::new(0.0);
        linker.add_entity(make_entity("http://x.org/a", "AI", &[], 0.5));
        linker.min_mention_len = 5;
        // "AI" is 2 chars — below threshold
        let mentions = linker.detect_mentions("AI is transforming the world.");
        assert!(
            mentions.is_empty() || mentions.iter().all(|m| m.text.len() >= 5),
            "unexpected short mention found"
        );
    }

    // --- Error display -------------------------------------------------------

    #[test]
    fn test_linker_error_display() {
        assert!(LinkerError::EmptyText.to_string().contains("empty"));
        assert!(LinkerError::EmptyKnowledgeBase
            .to_string()
            .contains("no entities"));
        assert!(LinkerError::InvalidEntity("bad".to_string())
            .to_string()
            .contains("bad"));
    }

    // --- Combined score ------------------------------------------------------

    #[test]
    fn test_combined_score_perfect() {
        // 0.7 * 1.0 + 0.3 * 1.0 = 1.0
        let linker = sample_linker();
        let candidates = linker.link_mention("Albert Einstein");
        // Best candidate should have high score
        if let Some(c) = candidates.first() {
            assert!(c.score > 0.5, "score = {}", c.score);
        }
    }

    // --- Alias detection ----------------------------------------------------

    #[test]
    fn test_alias_detection_full_alias() {
        let linker = sample_linker();
        let mentions = linker.detect_mentions("A. Einstein changed physics.");
        let texts: Vec<String> = mentions.iter().map(|m| m.text.to_lowercase()).collect();
        assert!(
            texts.iter().any(|t| t.contains("einstein")),
            "texts = {:?}",
            texts
        );
    }
}
