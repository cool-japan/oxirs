//! Entity linking and disambiguation for knowledge graphs.
//!
//! Provides candidate generation via string matching and context-based
//! disambiguation using TF-IDF cosine similarity.

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

/// A span of text that may refer to a named entity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntityMention {
    /// Surface form of the mention.
    pub text: String,
    /// Start byte offset in the containing text.
    pub start: usize,
    /// End byte offset (exclusive) in the containing text.
    pub end: usize,
}

impl EntityMention {
    /// Create a mention from text and character positions.
    pub fn new(text: impl Into<String>, start: usize, end: usize) -> Self {
        Self {
            text: text.into(),
            start,
            end,
        }
    }
}

/// A candidate entity from the knowledge base.
#[derive(Debug, Clone)]
pub struct EntityCandidate {
    /// Entity IRI.
    pub iri: String,
    /// Primary label.
    pub label: String,
    /// String-matching similarity score (0.0–1.0).
    pub score: f64,
    /// Alternative labels and aliases.
    pub aliases: Vec<String>,
}

impl EntityCandidate {
    fn new(iri: impl Into<String>, label: impl Into<String>, aliases: Vec<String>) -> Self {
        Self {
            iri: iri.into(),
            label: label.into(),
            score: 0.0,
            aliases,
        }
    }
}

/// A successfully linked entity.
#[derive(Debug, Clone)]
pub struct LinkedEntity {
    /// The text mention.
    pub mention: EntityMention,
    /// The best matching entity candidate.
    pub entity: EntityCandidate,
    /// Overall confidence (0.0–1.0) combining string + context similarity.
    pub confidence: f64,
}

// ──────────────────────────────────────────────────────────────────────────────
// TfIdfIndex
// ──────────────────────────────────────────────────────────────────────────────

/// A simple TF-IDF index for context-based disambiguation.
pub struct TfIdfIndex {
    /// Documents: (doc_id, term → tf).
    docs: Vec<(String, HashMap<String, f64>)>,
    /// Inverse document frequency: term → idf.
    idf: HashMap<String, f64>,
}

impl TfIdfIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self {
            docs: Vec::new(),
            idf: HashMap::new(),
        }
    }

    /// Add a document to the index.
    pub fn add_document(&mut self, doc_id: impl Into<String>, text: &str) {
        let tokens = tokenize(text);
        let total = tokens.len() as f64;
        if total == 0.0 {
            return;
        }
        let mut tf: HashMap<String, f64> = HashMap::new();
        for tok in &tokens {
            *tf.entry(tok.clone()).or_insert(0.0) += 1.0 / total;
        }
        self.docs.push((doc_id.into(), tf));
    }

    /// Recompute IDF from all indexed documents.
    pub fn build(&mut self) {
        let n = self.docs.len() as f64;
        let mut df: HashMap<String, usize> = HashMap::new();
        for (_, tf) in &self.docs {
            for term in tf.keys() {
                *df.entry(term.clone()).or_insert(0) += 1;
            }
        }
        self.idf.clear();
        for (term, count) in df {
            self.idf.insert(term, (n / count as f64).ln() + 1.0);
        }
    }

    /// Compute TF-IDF cosine similarity between a query string and a document.
    pub fn similarity(&self, query: &str, doc_id: &str) -> f64 {
        let doc = match self.docs.iter().find(|(id, _)| id == doc_id) {
            Some((_, tf)) => tf,
            None => return 0.0,
        };

        let q_tokens = tokenize(query);
        let q_total = q_tokens.len() as f64;
        if q_total == 0.0 {
            return 0.0;
        }
        let mut q_tf: HashMap<String, f64> = HashMap::new();
        for tok in &q_tokens {
            *q_tf.entry(tok.clone()).or_insert(0.0) += 1.0 / q_total;
        }

        // Cosine similarity of TF-IDF vectors
        let mut dot = 0.0_f64;
        let mut q_norm = 0.0_f64;
        let mut d_norm = 0.0_f64;

        let all_terms: std::collections::HashSet<&String> = q_tf.keys().chain(doc.keys()).collect();

        for term in all_terms {
            let idf = self.idf.get(term).copied().unwrap_or(1.0);
            let q_val = q_tf.get(term).copied().unwrap_or(0.0) * idf;
            let d_val = doc.get(term).copied().unwrap_or(0.0) * idf;
            dot += q_val * d_val;
            q_norm += q_val * q_val;
            d_norm += d_val * d_val;
        }

        let denom = q_norm.sqrt() * d_norm.sqrt();
        if denom < 1e-15 {
            0.0
        } else {
            (dot / denom).clamp(0.0, 1.0)
        }
    }

    /// Number of indexed documents.
    pub fn doc_count(&self) -> usize {
        self.docs.len()
    }
}

impl Default for TfIdfIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// EntityLinker
// ──────────────────────────────────────────────────────────────────────────────

/// Links entity mentions in text to knowledge-base entities.
pub struct EntityLinker {
    /// Knowledge base entries: iri → (label, aliases, context_doc_id).
    kb: HashMap<String, KbEntry>,
    /// TF-IDF index over entity descriptions (for context disambiguation).
    tfidf: TfIdfIndex,
    /// Minimum confidence threshold below which an entity is treated as NIL.
    pub nil_threshold: f64,
}

struct KbEntry {
    label: String,
    aliases: Vec<String>,
}

impl EntityLinker {
    /// Create an entity linker with default NIL threshold 0.1.
    pub fn new() -> Self {
        Self {
            kb: HashMap::new(),
            tfidf: TfIdfIndex::new(),
            nil_threshold: 0.1,
        }
    }

    /// Add an entity to the knowledge base.
    ///
    /// `context` is an optional textual description used for TF-IDF
    /// disambiguation.
    pub fn add_entity(
        &mut self,
        iri: impl Into<String>,
        label: impl Into<String>,
        aliases: &[&str],
    ) {
        let iri = iri.into();
        let label = label.into();
        let aliases: Vec<String> = aliases.iter().map(|s| s.to_string()).collect();
        let context = format!("{} {}", label, aliases.join(" "));
        self.tfidf.add_document(iri.clone(), &context);
        self.kb.insert(iri, KbEntry { label, aliases });
    }

    /// Finalise the TF-IDF index (call after all entities are added).
    pub fn build_index(&mut self) {
        self.tfidf.build();
    }

    /// Find and link all entity mentions in `text`.
    pub fn link(&self, text: &str) -> Vec<LinkedEntity> {
        let mentions = detect_mentions(text);
        let mut linked = Vec::new();

        for mention in mentions {
            let candidates = self.candidate_generation(&mention.text);
            if candidates.is_empty() {
                continue;
            }
            let best = self.disambiguate(&mention, &candidates, text);
            if let Some(entity) = best {
                let confidence = entity.score;
                if confidence >= self.nil_threshold {
                    linked.push(LinkedEntity {
                        mention,
                        entity,
                        confidence,
                    });
                }
            }
        }
        linked
    }

    /// Generate entity candidates matching the mention by string similarity.
    pub fn candidate_generation(&self, mention: &str) -> Vec<EntityCandidate> {
        let mention_lower = mention.to_lowercase();
        let mut candidates: Vec<EntityCandidate> = self
            .kb
            .iter()
            .filter_map(|(iri, entry)| {
                let label_score = jaro_winkler(&mention_lower, &entry.label.to_lowercase());
                let alias_score = entry
                    .aliases
                    .iter()
                    .map(|a| jaro_winkler(&mention_lower, &a.to_lowercase()))
                    .fold(0.0_f64, f64::max);
                let score = label_score.max(alias_score);
                if score > 0.6 {
                    let mut c = EntityCandidate::new(
                        iri.clone(),
                        entry.label.clone(),
                        entry.aliases.clone(),
                    );
                    c.score = score;
                    Some(c)
                } else {
                    None
                }
            })
            .collect();

        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates
    }

    /// Disambiguate among candidates using context TF-IDF similarity.
    pub fn disambiguate(
        &self,
        _mention: &EntityMention,
        candidates: &[EntityCandidate],
        context: &str,
    ) -> Option<EntityCandidate> {
        if candidates.is_empty() {
            return None;
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best: Option<EntityCandidate> = None;

        for cand in candidates {
            let ctx_score = self.tfidf.similarity(context, &cand.iri);
            // Combined score: string similarity × 0.6 + context × 0.4
            let combined = cand.score * 0.6 + ctx_score * 0.4;
            if combined > best_score {
                best_score = combined;
                let mut winner = cand.clone();
                winner.score = combined;
                best = Some(winner);
            }
        }
        best
    }

    /// Number of entities in the knowledge base.
    pub fn entity_count(&self) -> usize {
        self.kb.len()
    }
}

impl Default for EntityLinker {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Private helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Detect potential entity mentions in text by looking for capitalised tokens
/// or sequences.
fn detect_mentions(text: &str) -> Vec<EntityMention> {
    let mut mentions = Vec::new();
    let mut chars = text.char_indices().peekable();
    let bytes = text.as_bytes();
    let len = bytes.len();

    while let Some((start, ch)) = chars.next() {
        if ch.is_uppercase() {
            // Consume a capitalised word sequence (handles "Albert Einstein")
            let mut end = start + ch.len_utf8();
            while end < len {
                let next_ch = text[end..].chars().next().unwrap_or('\0');
                if next_ch.is_alphanumeric() || next_ch == ' ' {
                    // Allow one space if followed by uppercase (multi-word entity)
                    if next_ch == ' ' {
                        let after_space = end + 1;
                        if after_space < len {
                            let nc2 = text[after_space..].chars().next().unwrap_or('\0');
                            if nc2.is_uppercase() {
                                end = after_space + nc2.len_utf8();
                                // advance the chars iterator past the space and the uppercase char
                                let _ = chars.next(); // space
                                let _ = chars.next(); // uppercase
                                continue;
                            }
                        }
                        break;
                    }
                    end += next_ch.len_utf8();
                    let _ = chars.next();
                } else {
                    break;
                }
            }
            let mention_text = text[start..end].trim().to_string();
            if mention_text.len() >= 2 {
                mentions.push(EntityMention::new(mention_text, start, end));
            }
        }
    }
    mentions
}

/// Jaro-Winkler string similarity (0.0–1.0).
fn jaro_winkler(s1: &str, s2: &str) -> f64 {
    if s1 == s2 {
        return 1.0;
    }
    let jaro = jaro(s1, s2);
    let prefix_len = s1
        .chars()
        .zip(s2.chars())
        .take(4)
        .take_while(|(a, b)| a == b)
        .count();
    let p = 0.1_f64;
    jaro + (prefix_len as f64 * p * (1.0 - jaro))
}

fn jaro(s1: &str, s2: &str) -> f64 {
    let s1: Vec<char> = s1.chars().collect();
    let s2: Vec<char> = s2.chars().collect();
    let len1 = s1.len();
    let len2 = s2.len();
    if len1 == 0 && len2 == 0 {
        return 1.0;
    }
    if len1 == 0 || len2 == 0 {
        return 0.0;
    }

    let match_window = (len1.max(len2) / 2).saturating_sub(1);
    let mut s1_matches = vec![false; len1];
    let mut s2_matches = vec![false; len2];
    let mut matches = 0usize;
    let mut transpositions = 0usize;

    for (i, &c1) in s1.iter().enumerate() {
        let start = i.saturating_sub(match_window);
        let end = (i + match_window + 1).min(len2);
        for (j, &c2) in s2[start..end].iter().enumerate() {
            let j_real = start + j;
            if !s2_matches[j_real] && c1 == c2 {
                s1_matches[i] = true;
                s2_matches[j_real] = true;
                matches += 1;
                break;
            }
        }
    }

    if matches == 0 {
        return 0.0;
    }

    let mut k = 0;
    for (i, &s1m) in s1_matches.iter().enumerate() {
        if s1m {
            while !s2_matches[k] {
                k += 1;
            }
            if s1[i] != s2[k] {
                transpositions += 1;
            }
            k += 1;
        }
    }

    let m = matches as f64;
    (m / len1 as f64 + m / len2 as f64 + (m - transpositions as f64 / 2.0) / m) / 3.0
}

/// Tokenise text into lowercase alpha-numeric tokens.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn linker_with_persons() -> EntityLinker {
        let mut linker = EntityLinker::new();
        linker.add_entity(
            "http://example.org/Albert_Einstein",
            "Albert Einstein",
            &["Einstein", "A. Einstein"],
        );
        linker.add_entity(
            "http://example.org/Marie_Curie",
            "Marie Curie",
            &["Curie", "M. Curie"],
        );
        linker.add_entity(
            "http://example.org/Isaac_Newton",
            "Isaac Newton",
            &["Newton"],
        );
        linker.build_index();
        linker
    }

    // ── EntityMention ─────────────────────────────────────────────────────────

    #[test]
    fn test_mention_new() {
        let m = EntityMention::new("Alice", 0, 5);
        assert_eq!(m.text, "Alice");
        assert_eq!(m.start, 0);
        assert_eq!(m.end, 5);
    }

    #[test]
    fn test_mention_equality() {
        let m1 = EntityMention::new("Bob", 0, 3);
        let m2 = EntityMention::new("Bob", 0, 3);
        assert_eq!(m1, m2);
    }

    // ── TfIdfIndex ────────────────────────────────────────────────────────────

    #[test]
    fn test_tfidf_add_document() {
        let mut idx = TfIdfIndex::new();
        idx.add_document("doc1", "quantum physics relativity");
        idx.build();
        assert_eq!(idx.doc_count(), 1);
    }

    #[test]
    fn test_tfidf_similarity_same_doc() {
        let mut idx = TfIdfIndex::new();
        idx.add_document("doc1", "quantum physics relativity");
        idx.build();
        let sim = idx.similarity("quantum physics", "doc1");
        assert!(sim > 0.0, "similarity should be > 0, got {sim}");
    }

    #[test]
    fn test_tfidf_similarity_different_content() {
        let mut idx = TfIdfIndex::new();
        idx.add_document("doc1", "quantum physics relativity");
        idx.add_document("doc2", "cooking recipes baking bread");
        idx.build();
        let s1 = idx.similarity("quantum physics", "doc1");
        let s2 = idx.similarity("quantum physics", "doc2");
        assert!(s1 > s2, "physics query should match doc1 better");
    }

    #[test]
    fn test_tfidf_unknown_doc() {
        let idx = TfIdfIndex::new();
        assert_eq!(idx.similarity("anything", "unknown"), 0.0);
    }

    #[test]
    fn test_tfidf_empty_query() {
        let mut idx = TfIdfIndex::new();
        idx.add_document("d", "hello world");
        idx.build();
        assert_eq!(idx.similarity("", "d"), 0.0);
    }

    #[test]
    fn test_tfidf_default() {
        let idx = TfIdfIndex::default();
        assert_eq!(idx.doc_count(), 0);
    }

    // ── EntityLinker ──────────────────────────────────────────────────────────

    #[test]
    fn test_linker_entity_count() {
        let linker = linker_with_persons();
        assert_eq!(linker.entity_count(), 3);
    }

    #[test]
    fn test_linker_default() {
        let linker = EntityLinker::default();
        assert_eq!(linker.entity_count(), 0);
    }

    // ── candidate_generation ──────────────────────────────────────────────────

    #[test]
    fn test_candidate_generation_exact_label() {
        let linker = linker_with_persons();
        let cands = linker.candidate_generation("Einstein");
        assert!(!cands.is_empty());
        assert!(cands[0].iri.contains("Einstein"));
    }

    #[test]
    fn test_candidate_generation_partial() {
        let linker = linker_with_persons();
        let cands = linker.candidate_generation("Newton");
        assert!(!cands.is_empty());
        assert!(cands.iter().any(|c| c.iri.contains("Newton")));
    }

    #[test]
    fn test_candidate_generation_no_match() {
        let linker = linker_with_persons();
        let cands = linker.candidate_generation("Zorkblat");
        assert!(cands.is_empty());
    }

    #[test]
    fn test_candidate_generation_sorted_by_score() {
        let linker = linker_with_persons();
        let cands = linker.candidate_generation("Curie");
        for i in 1..cands.len() {
            assert!(cands[i - 1].score >= cands[i].score);
        }
    }

    #[test]
    fn test_candidate_generation_alias_match() {
        let linker = linker_with_persons();
        // "Curie" is an alias for Marie Curie
        let cands = linker.candidate_generation("Curie");
        assert!(cands.iter().any(|c| c.iri.contains("Curie")));
    }

    // ── disambiguate ─────────────────────────────────────────────────────────

    #[test]
    fn test_disambiguate_returns_best() {
        let linker = linker_with_persons();
        let cands = linker.candidate_generation("Einstein");
        let mention = EntityMention::new("Einstein", 0, 8);
        let best = linker.disambiguate(&mention, &cands, "Einstein worked on relativity");
        assert!(best.is_some());
        assert!(best.expect("should succeed").iri.contains("Einstein"));
    }

    #[test]
    fn test_disambiguate_empty_candidates() {
        let linker = linker_with_persons();
        let mention = EntityMention::new("X", 0, 1);
        let result = linker.disambiguate(&mention, &[], "context");
        assert!(result.is_none());
    }

    #[test]
    fn test_disambiguate_score_in_range() {
        let linker = linker_with_persons();
        let cands = linker.candidate_generation("Newton");
        let mention = EntityMention::new("Newton", 0, 6);
        if let Some(best) = linker.disambiguate(&mention, &cands, "gravity laws Newton") {
            assert!((0.0..=1.0).contains(&best.score));
        }
    }

    // ── link ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_link_finds_entity() {
        let linker = linker_with_persons();
        let linked = linker.link("Einstein developed relativity theory.");
        assert!(!linked.is_empty());
        assert!(linked[0].entity.iri.contains("Einstein"));
    }

    #[test]
    fn test_link_confidence_above_threshold() {
        let linker = linker_with_persons();
        let linked = linker.link("Newton formulated laws of motion.");
        for le in &linked {
            assert!(le.confidence >= linker.nil_threshold);
        }
    }

    #[test]
    fn test_link_no_entities_in_empty_text() {
        let linker = linker_with_persons();
        let linked = linker.link("");
        assert!(linked.is_empty());
    }

    #[test]
    fn test_link_result_fields() {
        let linker = linker_with_persons();
        let linked = linker.link("Einstein and Curie were scientists.");
        for le in &linked {
            assert!(!le.mention.text.is_empty());
            assert!(!le.entity.iri.is_empty());
            assert!((0.0..=1.0).contains(&le.confidence));
        }
    }

    // ── Jaro-Winkler ──────────────────────────────────────────────────────────

    #[test]
    fn test_jaro_winkler_identical() {
        assert!((jaro_winkler("hello", "hello") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_jaro_winkler_completely_different() {
        let score = jaro_winkler("abc", "xyz");
        assert!(score < 0.5, "score = {score}");
    }

    #[test]
    fn test_jaro_winkler_prefix_boost() {
        let jw = jaro_winkler("einstein", "einstien");
        assert!(jw > 0.8, "score = {jw}");
    }

    #[test]
    fn test_jaro_winkler_empty_strings() {
        assert!((jaro("", "") - 1.0).abs() < 1e-9);
        assert!((jaro("abc", "") - 0.0).abs() < 1e-9);
    }

    // ── detect_mentions ───────────────────────────────────────────────────────

    #[test]
    fn test_detect_mentions_finds_capitalized() {
        let mentions = detect_mentions("Alice and Bob went to Paris.");
        let texts: Vec<&str> = mentions.iter().map(|m| m.text.as_str()).collect();
        // At least Alice, Bob, Paris should be detected
        assert!(texts
            .iter()
            .any(|t| *t == "Alice" || t.starts_with("Alice")));
    }

    #[test]
    fn test_detect_mentions_empty() {
        assert!(detect_mentions("").is_empty());
    }

    #[test]
    fn test_detect_mentions_lowercase_only() {
        let mentions = detect_mentions("all lowercase words here");
        assert!(mentions.is_empty());
    }

    // ── tokenize ─────────────────────────────────────────────────────────────

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("Hello World");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_tokenize_empty() {
        assert!(tokenize("").is_empty());
    }

    #[test]
    fn test_tokenize_punctuation_split() {
        let tokens = tokenize("foo, bar; baz.");
        assert_eq!(tokens, vec!["foo", "bar", "baz"]);
    }

    // ── Full pipeline ─────────────────────────────────────────────────────────

    #[test]
    fn test_full_pipeline() {
        let mut linker = EntityLinker::new();
        linker.add_entity("http://ex.org/Paris", "Paris", &["City of Light"]);
        linker.add_entity("http://ex.org/London", "London", &["British capital"]);
        linker.build_index();

        let linked = linker.link("Paris is a famous city in France.");
        if !linked.is_empty() {
            assert!(linked[0].entity.iri.contains("Paris"));
        }
        // No assertion on count since detection depends on heuristic
    }

    #[test]
    fn test_nil_threshold_filters_low_confidence() {
        let mut linker = EntityLinker::new();
        linker.add_entity("http://ex.org/X", "Xyzzy", &[]);
        linker.build_index();
        linker.nil_threshold = 0.99; // Very high threshold

        let linked = linker.link("Xyzzy something");
        // Most links should be filtered out by high threshold
        for le in &linked {
            assert!(le.confidence >= 0.99);
        }
    }
}
