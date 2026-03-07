//! TF-IDF full-text search index over RDF triple literals
//!
//! `TextSearchIndex` is a lightweight, in-memory inverted index that scores
//! search hits using classic TF-IDF.
//!
//! score(term, doc) = tf(term, doc) * ln((N + 1) / (df(term) + 1))
//!
//! where
//!   - tf  = term frequency in document
//!   - N   = total number of documents
//!   - df  = number of documents containing the term
//!
//! The aggregate score for a multi-term query is the sum of per-term scores.

use std::collections::{HashMap, HashSet};

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// A single text-search result.
#[derive(Debug, Clone)]
pub struct SearchHit {
    /// Subject IRI or blank node identifier
    pub subject: String,
    /// Predicate IRI
    pub predicate: String,
    /// The indexed literal text
    pub literal: String,
    /// TF-IDF relevance score (higher = more relevant)
    pub score: f64,
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal document representation
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct IndexedDoc {
    id: u32,
    subject: String,
    predicate: String,
    literal: String,
    /// Tokens with stop-words removed (used for TF counting)
    tokens: Vec<String>,
}

// ──────────────────────────────────────────────────────────────────────────────
// TextSearchIndex
// ──────────────────────────────────────────────────────────────────────────────

/// In-memory TF-IDF full-text search index over RDF literals.
///
/// Each indexed triple `(subject, predicate, literal)` is treated as a single
/// document.  When the same `(subject, predicate)` pair is re-indexed, the
/// prior document is replaced.
pub struct TextSearchIndex {
    /// term → set of doc_ids containing that term
    inverted: HashMap<String, HashSet<u32>>,
    /// doc_id → document
    docs: HashMap<u32, IndexedDoc>,
    /// (subject, predicate) → doc_id for deduplication
    sp_index: HashMap<(String, String), u32>,
    /// Monotonically-increasing ID counter
    next_id: u32,
    /// Stop words excluded from the index
    stop_words: HashSet<&'static str>,
}

impl TextSearchIndex {
    /// English stop words that are filtered out during indexing.
    const STOP_WORDS: &'static [&'static str] = &[
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", "has", "have",
        "he", "in", "is", "it", "its", "of", "on", "or", "she", "that", "the", "their", "there",
        "they", "this", "to", "was", "were", "will", "with",
    ];

    /// Create a new empty index.
    pub fn new() -> Self {
        TextSearchIndex {
            inverted: HashMap::new(),
            docs: HashMap::new(),
            sp_index: HashMap::new(),
            next_id: 0,
            stop_words: Self::STOP_WORDS.iter().cloned().collect(),
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Index maintenance
    // ──────────────────────────────────────────────────────────────────────────

    /// Index (or re-index) a triple's literal value.
    ///
    /// If the `(subject, predicate)` pair was already indexed, the previous
    /// document is removed first before inserting the new one.
    pub fn index_triple(&mut self, subject: &str, predicate: &str, literal: &str) {
        let sp_key = (subject.to_string(), predicate.to_string());

        // Remove the previous document for this (subject, predicate) if any
        if let Some(old_id) = self.sp_index.remove(&sp_key) {
            self.remove_doc_by_id(old_id);
        }

        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);

        let tokens = self.tokenize_and_filter(literal);

        // Build inverted index entries
        let term_set: HashSet<String> = tokens.iter().cloned().collect();
        for term in &term_set {
            self.inverted.entry(term.clone()).or_default().insert(id);
        }

        let doc = IndexedDoc {
            id,
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            literal: literal.to_string(),
            tokens,
        };

        self.docs.insert(id, doc);
        self.sp_index.insert(sp_key, id);
    }

    /// Remove all indexed documents for the given `(subject, predicate)` pair.
    ///
    /// Returns `true` if at least one document was removed.
    pub fn remove_triple(&mut self, subject: &str, predicate: &str) -> bool {
        let sp_key = (subject.to_string(), predicate.to_string());
        if let Some(id) = self.sp_index.remove(&sp_key) {
            self.remove_doc_by_id(id);
            true
        } else {
            false
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Search
    // ──────────────────────────────────────────────────────────────────────────

    /// Full-text search using TF-IDF scoring.
    ///
    /// Returns all matching documents sorted by score descending.
    /// A document matches if at least one query term appears in its tokens.
    pub fn search(&self, query: &str) -> Vec<SearchHit> {
        let query_terms = Self::tokenize_raw(query);
        if query_terms.is_empty() {
            return Vec::new();
        }

        let n = self.docs.len() as f64;
        if n == 0.0 {
            return Vec::new();
        }

        // Aggregate TF-IDF scores per doc_id
        let mut scores: HashMap<u32, f64> = HashMap::new();

        for term in &query_terms {
            // df = number of docs containing this term
            let posting = match self.inverted.get(term) {
                Some(set) => set,
                None => continue,
            };
            let df = posting.len() as f64;
            // Smoothed IDF: add 1 to ensure positive scores even for terms
            // appearing in every document (avoids ln(1) = 0).
            let idf = ((n + 1.0) / (df + 1.0)).ln() + 1.0;

            for &doc_id in posting {
                if let Some(doc) = self.docs.get(&doc_id) {
                    // tf = count of this term in document
                    let tf = doc.tokens.iter().filter(|t| *t == term).count() as f64;
                    *scores.entry(doc_id).or_insert(0.0) += tf * idf;
                }
            }
        }

        // Build sorted result list
        let mut hits: Vec<(u32, f64)> = scores.into_iter().collect();
        hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        hits.into_iter()
            .filter_map(|(doc_id, score)| {
                let doc = self.docs.get(&doc_id)?;
                Some(SearchHit {
                    subject: doc.subject.clone(),
                    predicate: doc.predicate.clone(),
                    literal: doc.literal.clone(),
                    score,
                })
            })
            .collect()
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Subject-level operations
    // ──────────────────────────────────────────────────────────────────────────

    /// Return all predicate IRIs indexed for the given subject.
    ///
    /// Used by `SparqlTextSearchExtension::remove_subject` to enumerate
    /// `(subject, predicate)` pairs without needing to search the literal text.
    pub fn predicates_for_subject(&self, subject: &str) -> Vec<String> {
        self.sp_index
            .keys()
            .filter(|(s, _)| s == subject)
            .map(|(_, p)| p.clone())
            .collect()
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Metrics
    // ──────────────────────────────────────────────────────────────────────────

    /// Number of unique indexed terms in the inverted index.
    pub fn size(&self) -> usize {
        self.inverted.len()
    }

    /// Number of indexed documents (triples).
    pub fn document_count(&self) -> usize {
        self.docs.len()
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ──────────────────────────────────────────────────────────────────────────

    /// Tokenize text, lowercase, split on non-alphanumeric characters.
    /// Filters tokens of length <= 1 but does NOT remove stop words.
    /// Used during search so that stop-word-free queries still work.
    fn tokenize_raw(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.len() > 1)
            .map(|t| t.to_string())
            .collect()
    }

    /// Tokenize and additionally filter stop words.
    /// Used during indexing.
    fn tokenize_and_filter(&self, text: &str) -> Vec<String> {
        Self::tokenize_raw(text)
            .into_iter()
            .filter(|t| !self.stop_words.contains(t.as_str()))
            .collect()
    }

    /// Remove a document from `docs` and the inverted index by its ID.
    fn remove_doc_by_id(&mut self, id: u32) {
        if let Some(doc) = self.docs.remove(&id) {
            let term_set: HashSet<String> = doc.tokens.into_iter().collect();
            for term in &term_set {
                if let Some(posting) = self.inverted.get_mut(term) {
                    posting.remove(&id);
                }
            }
            // Clean up empty posting lists
            self.inverted.retain(|_, posting| !posting.is_empty());
        }
    }
}

impl Default for TextSearchIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn subj(n: u32) -> String {
        format!("http://ex.org/s{}", n)
    }
    fn pred() -> &'static str {
        "http://ex.org/label"
    }

    // ── basic indexing & searching ────────────────────────────────────────────

    #[test]
    fn test_index_and_search_basic() {
        let mut idx = TextSearchIndex::new();
        idx.index_triple(&subj(1), pred(), "Rust programming language systems");
        idx.index_triple(&subj(2), pred(), "Python scripting language");

        let hits = idx.search("rust programming");
        assert!(!hits.is_empty(), "Should find 'rust programming'");
        assert_eq!(hits[0].subject, subj(1));
    }

    #[test]
    fn test_empty_search_returns_empty() {
        let idx = TextSearchIndex::new();
        let hits = idx.search("rust");
        assert!(hits.is_empty());
    }

    #[test]
    fn test_search_no_match_returns_empty() {
        let mut idx = TextSearchIndex::new();
        idx.index_triple(&subj(1), pred(), "hello world");
        let hits = idx.search("zzzyyyxxx");
        assert!(hits.is_empty());
    }

    // ── TF-IDF scoring ────────────────────────────────────────────────────────

    #[test]
    fn test_tfidf_higher_tf_ranks_higher() {
        let mut idx = TextSearchIndex::new();
        // doc1: "database" once
        idx.index_triple(&subj(1), pred(), "database management systems");
        // doc2: "database" three times
        idx.index_triple(&subj(2), pred(), "database database database storage");

        let hits = idx.search("database");
        assert_eq!(hits.len(), 2, "Both docs should match");
        assert_eq!(hits[0].subject, subj(2), "Higher TF should rank first");
    }

    #[test]
    fn test_tfidf_score_positive() {
        let mut idx = TextSearchIndex::new();
        idx.index_triple(&subj(1), pred(), "semantic web ontology");
        let hits = idx.search("semantic");
        assert_eq!(hits.len(), 1);
        assert!(hits[0].score > 0.0, "Score should be positive");
    }

    #[test]
    fn test_tfidf_multiterm_sum() {
        let mut idx = TextSearchIndex::new();
        idx.index_triple(&subj(1), pred(), "knowledge graph reasoning inference");
        // doc2 has "knowledge" and "graph" twice each
        idx.index_triple(
            &subj(2),
            pred(),
            "knowledge knowledge graph graph data modeling",
        );

        let hits = idx.search("knowledge graph");
        assert_eq!(hits.len(), 2);
        // doc2 has higher TF for both terms → should rank first
        assert_eq!(hits[0].subject, subj(2));
    }

    // ── remove_triple ─────────────────────────────────────────────────────────

    #[test]
    fn test_remove_triple_removes_document() {
        let mut idx = TextSearchIndex::new();
        idx.index_triple(&subj(1), pred(), "hello world");
        idx.index_triple(&subj(2), pred(), "hello rust");

        idx.remove_triple(&subj(1), pred());

        let hits = idx.search("hello");
        assert_eq!(hits.len(), 1, "Only s2 should remain");
        assert_eq!(hits[0].subject, subj(2));
    }

    #[test]
    fn test_remove_nonexistent_returns_false() {
        let mut idx = TextSearchIndex::new();
        let removed = idx.remove_triple(&subj(99), pred());
        assert!(!removed, "Removing nonexistent should return false");
    }

    #[test]
    fn test_remove_cleans_inverted_index() {
        let mut idx = TextSearchIndex::new();
        // Single token (no underscores/punctuation, not a stop word)
        idx.index_triple(&subj(1), pred(), "uniquetermzyx");
        assert_eq!(idx.size(), 1);
        idx.remove_triple(&subj(1), pred());
        assert_eq!(
            idx.size(),
            0,
            "Inverted index should be empty after removal"
        );
    }

    // ── deduplication (re-index same subject+predicate) ───────────────────────

    #[test]
    fn test_reindex_same_sp_replaces() {
        let mut idx = TextSearchIndex::new();
        idx.index_triple(&subj(1), pred(), "old value content");
        idx.index_triple(&subj(1), pred(), "new value content");

        // "old" should no longer be searchable for s1
        let hits = idx.search("old");
        assert!(
            hits.iter().all(|h| h.subject != subj(1)),
            "'old' should be gone after re-index"
        );

        // "new" should be found
        let hits = idx.search("new");
        assert!(hits.iter().any(|h| h.subject == subj(1)));
    }

    #[test]
    fn test_reindex_doc_count_unchanged() {
        let mut idx = TextSearchIndex::new();
        idx.index_triple(&subj(1), pred(), "version one");
        assert_eq!(idx.document_count(), 1);
        idx.index_triple(&subj(1), pred(), "version two");
        assert_eq!(idx.document_count(), 1, "Re-index should not add a new doc");
    }

    // ── stop word filtering ───────────────────────────────────────────────────

    #[test]
    fn test_stop_words_not_indexed() {
        let mut idx = TextSearchIndex::new();
        // Only stop words in the literal
        idx.index_triple(&subj(1), pred(), "the and or is");
        // None of the stop words should be in the inverted index
        assert_eq!(idx.size(), 0, "Stop words alone should produce empty index");
    }

    #[test]
    fn test_stop_words_mixed_with_content() {
        let mut idx = TextSearchIndex::new();
        idx.index_triple(&subj(1), pred(), "the quick brown fox");
        // "the" is a stop word; "quick", "brown", "fox" should be indexed
        let hits = idx.search("quick");
        assert!(!hits.is_empty(), "Non-stop-word should be found");
    }

    // ── size and document_count ───────────────────────────────────────────────

    #[test]
    fn test_size_reflects_term_count() {
        let mut idx = TextSearchIndex::new();
        assert_eq!(idx.size(), 0);
        idx.index_triple(&subj(1), pred(), "alpha beta gamma");
        assert!(idx.size() >= 3, "Should have at least 3 terms");
    }

    #[test]
    fn test_document_count() {
        let mut idx = TextSearchIndex::new();
        assert_eq!(idx.document_count(), 0);
        idx.index_triple(&subj(1), pred(), "doc one");
        assert_eq!(idx.document_count(), 1);
        idx.index_triple(&subj(2), pred(), "doc two");
        assert_eq!(idx.document_count(), 2);
    }

    // ── multiple predicates for same subject ──────────────────────────────────

    #[test]
    fn test_multiple_predicates_same_subject() {
        let mut idx = TextSearchIndex::new();
        idx.index_triple(&subj(1), "http://ex.org/title", "Semantic Web");
        idx.index_triple(&subj(1), "http://ex.org/desc", "linked data platform");

        assert_eq!(idx.document_count(), 2);

        // Both predicates searchable independently
        let title_hits = idx.search("semantic");
        assert!(!title_hits.is_empty());

        let desc_hits = idx.search("linked");
        assert!(!desc_hits.is_empty());
    }

    // ── search returns sorted results ─────────────────────────────────────────

    #[test]
    fn test_search_results_sorted_descending() {
        let mut idx = TextSearchIndex::new();
        idx.index_triple(&subj(1), pred(), "sparql query language protocol");
        idx.index_triple(
            &subj(2),
            pred(),
            "sparql sparql query query endpoint server",
        );
        idx.index_triple(&subj(3), pred(), "sparql endpoint");

        let hits = idx.search("sparql query");
        // Verify that scores are non-increasing
        for w in hits.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "Results should be sorted descending by score"
            );
        }
    }

    // ── SearchHit fields ──────────────────────────────────────────────────────

    #[test]
    fn test_hit_fields_populated() {
        let mut idx = TextSearchIndex::new();
        idx.index_triple(
            "http://ex.org/subject",
            "http://ex.org/predicate",
            "test literal value",
        );
        let hits = idx.search("literal");
        assert_eq!(hits.len(), 1);
        let hit = &hits[0];
        assert_eq!(hit.subject, "http://ex.org/subject");
        assert_eq!(hit.predicate, "http://ex.org/predicate");
        assert_eq!(hit.literal, "test literal value");
        assert!(hit.score > 0.0);
    }
}
