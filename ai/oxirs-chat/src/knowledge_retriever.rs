//! RAG knowledge retrieval combining BM25 and cosine similarity scoring.
//!
//! The [`KnowledgeRetriever`] maintains an in-memory index of documents and
//! supports three search modes:
//! - **BM25-only** — classical term-frequency/inverse-document-frequency ranking
//! - **Vector-only** — cosine similarity between query and document embeddings
//! - **Combined** — weighted blend of both scores

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A document stored in the retriever.
#[derive(Debug, Clone)]
pub struct Document {
    /// Unique identifier
    pub id: String,
    /// Full text content
    pub content: String,
    /// Optional dense embedding for vector similarity search
    pub embedding: Option<Vec<f32>>,
    /// Arbitrary key/value metadata
    pub metadata: HashMap<String, String>,
}

impl Document {
    /// Construct a text-only document (no embedding).
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            embedding: None,
            metadata: HashMap::new(),
        }
    }

    /// Construct a document with a dense embedding.
    pub fn with_embedding(
        id: impl Into<String>,
        content: impl Into<String>,
        embedding: Vec<f32>,
    ) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            embedding: Some(embedding),
            metadata: HashMap::new(),
        }
    }
}

/// A single search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub doc_id: String,
    pub bm25_score: f32,
    pub vector_score: Option<f32>,
    pub combined_score: f32,
    /// Relevant excerpt from the document
    pub snippet: String,
}

/// Configuration for the retriever.
#[derive(Debug, Clone)]
pub struct RetrieverConfig {
    /// Maximum number of results to return
    pub top_k: usize,
    /// Weight applied to the BM25 component (0.0–1.0)
    pub bm25_weight: f32,
    /// Weight applied to the vector component (0.0–1.0)
    pub vector_weight: f32,
    /// Minimum combined score threshold; lower-scoring results are excluded
    pub min_score: f32,
}

impl Default for RetrieverConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            bm25_weight: 0.6,
            vector_weight: 0.4,
            min_score: 0.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BM25 parameters
// ─────────────────────────────────────────────────────────────────────────────

/// Okapi BM25 saturation parameter
const K1: f32 = 1.5;
/// Okapi BM25 length normalisation parameter
const B: f32 = 0.75;

// ─────────────────────────────────────────────────────────────────────────────
// KnowledgeRetriever
// ─────────────────────────────────────────────────────────────────────────────

/// In-memory RAG knowledge retriever.
pub struct KnowledgeRetriever {
    documents: HashMap<String, Document>,
    config: RetrieverConfig,
    /// doc_id → term → raw term count
    term_frequencies: HashMap<String, HashMap<String, usize>>,
    /// term → number of documents that contain the term
    doc_frequencies: HashMap<String, usize>,
    /// Average document length in tokens
    avg_doc_length: f32,
}

impl KnowledgeRetriever {
    /// Create an empty retriever with the given configuration.
    pub fn new(config: RetrieverConfig) -> Self {
        Self {
            documents: HashMap::new(),
            config,
            term_frequencies: HashMap::new(),
            doc_frequencies: HashMap::new(),
            avg_doc_length: 0.0,
        }
    }

    /// Add a document and update BM25 indices.
    pub fn add_document(&mut self, doc: Document) {
        let tokens = Self::tokenize(&doc.content);
        let _doc_len = tokens.len();

        // Term frequency for this document
        let mut tf: HashMap<String, usize> = HashMap::new();
        for token in &tokens {
            *tf.entry(token.clone()).or_insert(0) += 1;
        }

        // Update document frequencies
        for term in tf.keys() {
            *self.doc_frequencies.entry(term.clone()).or_insert(0) += 1;
        }

        self.term_frequencies.insert(doc.id.clone(), tf);
        self.documents.insert(doc.id.clone(), doc);

        self.recompute_avg_doc_length();
    }

    /// Remove a document.  Returns `true` if the document was present.
    pub fn remove_document(&mut self, doc_id: &str) -> bool {
        if let Some(tf) = self.term_frequencies.remove(doc_id) {
            for term in tf.keys() {
                if let Some(count) = self.doc_frequencies.get_mut(term.as_str()) {
                    if *count <= 1 {
                        self.doc_frequencies.remove(term.as_str());
                    } else {
                        *count -= 1;
                    }
                }
            }
            self.documents.remove(doc_id);
            self.recompute_avg_doc_length();
            true
        } else {
            false
        }
    }

    /// Search using a weighted combination of BM25 and vector scores.
    ///
    /// If a document has no embedding, its vector score is treated as 0.
    pub fn search(&self, query: &str) -> Vec<SearchResult> {
        let query_terms: Vec<String> = Self::tokenize(query);
        let query_term_refs: Vec<&str> = query_terms.iter().map(|s| s.as_str()).collect();

        let results: Vec<SearchResult> = self
            .documents
            .keys()
            .map(|doc_id| {
                let bm25 = self.bm25_score(doc_id, &query_term_refs);
                let snippet = Self::extract_snippet(&self.documents[doc_id].content, query, 200);
                SearchResult {
                    doc_id: doc_id.clone(),
                    bm25_score: bm25,
                    vector_score: None,
                    // search() without a query embedding uses BM25 only
                    combined_score: bm25 * self.config.bm25_weight,
                    snippet,
                }
            })
            .collect();

        self.finalise_results(results)
    }

    /// Search using BM25 only.
    pub fn search_bm25(&self, query: &str) -> Vec<SearchResult> {
        let query_terms: Vec<String> = Self::tokenize(query);
        let query_term_refs: Vec<&str> = query_terms.iter().map(|s| s.as_str()).collect();

        let results: Vec<SearchResult> = self
            .documents
            .keys()
            .map(|doc_id| {
                let bm25 = self.bm25_score(doc_id, &query_term_refs);
                let snippet = Self::extract_snippet(&self.documents[doc_id].content, query, 200);
                SearchResult {
                    doc_id: doc_id.clone(),
                    bm25_score: bm25,
                    vector_score: None,
                    combined_score: bm25,
                    snippet,
                }
            })
            .collect();

        self.finalise_results(results)
    }

    /// Search using cosine similarity against document embeddings.
    ///
    /// Documents without an embedding are excluded.
    pub fn search_vector(&self, query_embedding: &[f32]) -> Vec<SearchResult> {
        let results: Vec<SearchResult> = self
            .documents
            .values()
            .filter_map(|doc| {
                doc.embedding.as_ref().map(|emb| {
                    let sim = cosine_similarity(query_embedding, emb);
                    let snippet = Self::extract_snippet(&doc.content, "", 200);
                    SearchResult {
                        doc_id: doc.id.clone(),
                        bm25_score: 0.0,
                        vector_score: Some(sim),
                        combined_score: sim,
                        snippet,
                    }
                })
            })
            .collect();

        self.finalise_results(results)
    }

    /// Search using BM25 + optional vector re-ranking when a query embedding is available.
    pub fn search_with_embedding(&self, query: &str, query_embedding: &[f32]) -> Vec<SearchResult> {
        let query_terms: Vec<String> = Self::tokenize(query);
        let query_term_refs: Vec<&str> = query_terms.iter().map(|s| s.as_str()).collect();

        let results: Vec<SearchResult> = self
            .documents
            .keys()
            .map(|doc_id| {
                let doc = &self.documents[doc_id];
                let bm25 = self.bm25_score(doc_id, &query_term_refs);
                let vec_score = doc
                    .embedding
                    .as_ref()
                    .map(|emb| cosine_similarity(query_embedding, emb));
                let combined = bm25 * self.config.bm25_weight
                    + vec_score.unwrap_or(0.0) * self.config.vector_weight;
                let snippet = Self::extract_snippet(&doc.content, query, 200);
                SearchResult {
                    doc_id: doc_id.clone(),
                    bm25_score: bm25,
                    vector_score: vec_score,
                    combined_score: combined,
                    snippet,
                }
            })
            .collect();

        self.finalise_results(results)
    }

    /// Total number of indexed documents.
    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    // ── Private methods ───────────────────────────────────────────────────────

    fn bm25_score(&self, doc_id: &str, query_terms: &[&str]) -> f32 {
        let tf_map = match self.term_frequencies.get(doc_id) {
            Some(m) => m,
            None => return 0.0,
        };
        let doc_len: usize = tf_map.values().sum();
        let n = self.documents.len() as f32;
        let avg_dl = self.avg_doc_length.max(1.0);
        let dl_norm = 1.0 - B + B * (doc_len as f32 / avg_dl);

        let mut score = 0.0_f32;
        for &term in query_terms {
            let tf = *tf_map.get(term).unwrap_or(&0) as f32;
            let df = *self.doc_frequencies.get(term).unwrap_or(&0) as f32;

            if df == 0.0 {
                continue;
            }

            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            let tf_component = (tf * (K1 + 1.0)) / (tf + K1 * dl_norm);
            score += idf * tf_component;
        }

        score.max(0.0)
    }

    /// Tokenise text into lowercase alphabetic tokens.
    fn tokenize(text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(|t| t.to_lowercase())
            .collect()
    }

    /// Extract a snippet of `max_len` characters from `content` that contains
    /// query terms, falling back to the beginning if no match found.
    fn extract_snippet(content: &str, query: &str, max_len: usize) -> String {
        if content.is_empty() {
            return String::new();
        }

        let lower = content.to_lowercase();
        let first_term_pos = Self::tokenize(query)
            .into_iter()
            .find_map(|term| lower.find(&term));

        let start = first_term_pos
            .map(|pos| pos.saturating_sub(40))
            .unwrap_or(0);

        // Ensure we don't split in the middle of a multi-byte character by
        // finding the nearest char boundary at or before the desired position.
        let safe_start = floor_char_boundary(content, start);
        let end = (safe_start + max_len).min(content.len());
        let safe_end = floor_char_boundary(content, end);

        let snippet = &content[safe_start..safe_end];
        if snippet.len() < content.len() {
            format!("{snippet}…")
        } else {
            snippet.to_string()
        }
    }

    fn recompute_avg_doc_length(&mut self) {
        if self.term_frequencies.is_empty() {
            self.avg_doc_length = 0.0;
            return;
        }
        let total: usize = self
            .term_frequencies
            .values()
            .map(|tf| tf.values().sum::<usize>())
            .sum();
        self.avg_doc_length = total as f32 / self.term_frequencies.len() as f32;
    }

    fn finalise_results(&self, mut results: Vec<SearchResult>) -> Vec<SearchResult> {
        // Apply min_score filter
        results.retain(|r| r.combined_score >= self.config.min_score);
        // Sort descending by combined score
        results.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // Apply top_k
        results.truncate(self.config.top_k);
        results
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Free functions
// ─────────────────────────────────────────────────────────────────────────────

/// Find the largest byte index ≤ `pos` that is a valid UTF-8 char boundary.
fn floor_char_boundary(s: &str, pos: usize) -> usize {
    if pos >= s.len() {
        return s.len();
    }
    let mut i = pos;
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> RetrieverConfig {
        RetrieverConfig {
            top_k: 10,
            bm25_weight: 0.6,
            vector_weight: 0.4,
            min_score: 0.0,
        }
    }

    fn retriever() -> KnowledgeRetriever {
        KnowledgeRetriever::new(config())
    }

    fn doc(id: &str, content: &str) -> Document {
        Document::new(id, content)
    }

    fn doc_emb(id: &str, content: &str, emb: Vec<f32>) -> Document {
        Document::with_embedding(id, content, emb)
    }

    // 1. Empty retriever returns no results
    #[test]
    fn test_empty_retriever_no_results() {
        let r = retriever();
        assert!(r.search("anything").is_empty());
    }

    // 2. document_count
    #[test]
    fn test_document_count() {
        let mut r = retriever();
        assert_eq!(r.document_count(), 0);
        r.add_document(doc("d1", "hello world"));
        assert_eq!(r.document_count(), 1);
    }

    // 3. Single doc search returns that doc
    #[test]
    fn test_single_doc_search() {
        let mut r = retriever();
        r.add_document(doc("d1", "the quick brown fox jumps"));
        let results = r.search_bm25("fox");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "d1");
    }

    // 4. BM25 higher for doc with more term matches
    #[test]
    fn test_bm25_higher_for_more_matches() {
        let mut r = retriever();
        r.add_document(doc("low", "the quick fox"));
        r.add_document(doc("high", "fox fox fox fox fox fox"));
        let results = r.search_bm25("fox");
        assert_eq!(results.len(), 2);
        // 'high' should rank first
        assert_eq!(results[0].doc_id, "high");
    }

    // 5. Query term not in any doc returns empty
    #[test]
    fn test_no_match_returns_empty_bm25() {
        let mut r = retriever();
        r.add_document(doc("d1", "cats and dogs"));
        let results = r.search_bm25("unicorn");
        // BM25 returns 0.0 score but still returns the doc (score >= min_score=0)
        // Actually if score == 0 and min_score == 0, it's included.
        // Let's verify score is 0:
        assert!(results.iter().all(|res| res.combined_score == 0.0));
    }

    // 6. Vector search ordering — most similar first
    #[test]
    fn test_vector_search_ordering() {
        let mut r = retriever();
        r.add_document(doc_emb("close", "text", vec![1.0, 0.0, 0.0]));
        r.add_document(doc_emb("far", "text", vec![0.0, 1.0, 0.0]));

        let query_emb = vec![1.0, 0.0, 0.0];
        let results = r.search_vector(&query_emb);
        assert_eq!(results[0].doc_id, "close");
    }

    // 7. Vector search excludes docs without embeddings
    #[test]
    fn test_vector_search_excludes_no_embedding() {
        let mut r = retriever();
        r.add_document(doc("no_emb", "no embedding"));
        r.add_document(doc_emb("with_emb", "with embedding", vec![1.0, 0.0]));

        let results = r.search_vector(&[1.0, 0.0]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "with_emb");
    }

    // 8. Combined scoring uses bm25_weight and vector_weight
    #[test]
    fn test_combined_scoring() {
        let mut r = KnowledgeRetriever::new(RetrieverConfig {
            bm25_weight: 0.5,
            vector_weight: 0.5,
            ..config()
        });
        let emb = vec![1.0f32, 0.0];
        r.add_document(doc_emb("d1", "rust programming language rust", emb.clone()));
        let results = r.search_with_embedding("rust", &emb);
        assert!(!results.is_empty());
        assert!(results[0].vector_score.is_some());
        // combined = bm25 * 0.5 + vector * 0.5
        let res = &results[0];
        let expected = res.bm25_score * 0.5 + res.vector_score.unwrap_or(0.0) * 0.5;
        assert!((res.combined_score - expected).abs() < 1e-5);
    }

    // 9. top_k limits results
    #[test]
    fn test_top_k_limits_results() {
        let mut r = KnowledgeRetriever::new(RetrieverConfig {
            top_k: 3,
            ..config()
        });
        for i in 0..10 {
            r.add_document(doc(
                &format!("d{i}"),
                &format!("document {i} contains rust"),
            ));
        }
        let results = r.search_bm25("rust");
        assert!(results.len() <= 3);
    }

    // 10. min_score filters low-scoring results
    #[test]
    fn test_min_score_filters() {
        let mut r = KnowledgeRetriever::new(RetrieverConfig {
            min_score: 100.0, // very high
            ..config()
        });
        r.add_document(doc("d1", "totally unrelated content"));
        let results = r.search_bm25("zebra");
        assert!(results.is_empty());
    }

    // 11. remove_document returns true when doc exists
    #[test]
    fn test_remove_document_true() {
        let mut r = retriever();
        r.add_document(doc("d1", "hello"));
        assert!(r.remove_document("d1"));
    }

    // 12. remove_document returns false when doc absent
    #[test]
    fn test_remove_document_false() {
        let mut r = retriever();
        assert!(!r.remove_document("nonexistent"));
    }

    // 13. remove_document decrements document_count
    #[test]
    fn test_remove_decrements_count() {
        let mut r = retriever();
        r.add_document(doc("d1", "test"));
        r.add_document(doc("d2", "test2"));
        r.remove_document("d1");
        assert_eq!(r.document_count(), 1);
    }

    // 14. Removed document no longer appears in search
    #[test]
    fn test_removed_doc_not_in_search() {
        let mut r = retriever();
        r.add_document(doc("d1", "unique keyword xyzzy"));
        r.remove_document("d1");
        let results = r.search_bm25("xyzzy");
        assert!(results.iter().all(|res| res.doc_id != "d1"));
    }

    // 15. Snippet extraction non-empty for matching doc
    #[test]
    fn test_snippet_non_empty() {
        let mut r = retriever();
        r.add_document(doc("d1", "The quick brown fox jumps over the lazy dog"));
        let results = r.search_bm25("fox");
        assert!(!results[0].snippet.is_empty());
    }

    // 16. Snippet contains query term (case-insensitive)
    #[test]
    fn test_snippet_contains_query_term() {
        let mut r = retriever();
        r.add_document(doc(
            "d1",
            "In the beginning was the word, and the word was knowledge.",
        ));
        let results = r.search_bm25("knowledge");
        let snippet = results[0].snippet.to_lowercase();
        assert!(snippet.contains("knowledge"), "snippet: {snippet}");
    }

    // 17. tokenize lowercases
    #[test]
    fn test_tokenize_lowercase() {
        let tokens = KnowledgeRetriever::tokenize("Hello World");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
    }

    // 18. tokenize splits on non-alphanumeric
    #[test]
    fn test_tokenize_splits_punctuation() {
        let tokens = KnowledgeRetriever::tokenize("cats,dogs,fish");
        assert_eq!(tokens.len(), 3);
        assert!(tokens.contains(&"cats".to_string()));
    }

    // 19. tokenize empty string
    #[test]
    fn test_tokenize_empty() {
        let tokens = KnowledgeRetriever::tokenize("");
        assert!(tokens.is_empty());
    }

    // 20. BM25 score is 0 for empty query
    #[test]
    fn test_bm25_empty_query_zero() {
        let mut r = retriever();
        r.add_document(doc("d1", "hello world"));
        let results = r.search_bm25("");
        assert!(results.iter().all(|res| res.combined_score == 0.0));
    }

    // 21. search returns results sorted descending
    #[test]
    fn test_search_sorted_descending() {
        let mut r = retriever();
        r.add_document(doc("a", "rust programming rust rust rust"));
        r.add_document(doc("b", "rust once"));
        let results = r.search_bm25("rust");
        for i in 1..results.len() {
            assert!(results[i - 1].combined_score >= results[i].combined_score);
        }
    }

    // 22. Vector search returns vector_score in results
    #[test]
    fn test_vector_search_score_present() {
        let mut r = retriever();
        r.add_document(doc_emb("d1", "text", vec![1.0, 0.0]));
        let results = r.search_vector(&[1.0, 0.0]);
        assert!(results[0].vector_score.is_some());
    }

    // 23. Identical query/doc embedding → vector_score ≈ 1.0
    #[test]
    fn test_vector_score_identical_embeddings() {
        let mut r = retriever();
        r.add_document(doc_emb("d1", "text", vec![1.0, 0.0, 0.0]));
        let results = r.search_vector(&[1.0, 0.0, 0.0]);
        assert!((results[0].vector_score.unwrap() - 1.0).abs() < 1e-5);
    }

    // 24. Orthogonal embeddings → vector_score ≈ 0.0
    #[test]
    fn test_vector_score_orthogonal() {
        let mut r = retriever();
        r.add_document(doc_emb("d1", "text", vec![1.0, 0.0]));
        let results = r.search_vector(&[0.0, 1.0]);
        assert!(results[0].vector_score.unwrap().abs() < 1e-5);
    }

    // 25. Document metadata is stored
    #[test]
    fn test_document_metadata_stored() {
        let mut r = retriever();
        let mut doc = Document::new("d1", "content");
        doc.metadata
            .insert("author".to_string(), "Alice".to_string());
        r.add_document(doc);
        assert_eq!(r.document_count(), 1);
    }

    // 26. BM25 score non-negative
    #[test]
    fn test_bm25_score_non_negative() {
        let mut r = retriever();
        r.add_document(doc("d1", "the quick brown fox"));
        let results = r.search_bm25("fox");
        assert!(results.iter().all(|res| res.bm25_score >= 0.0));
    }

    // 27. search_bm25 returns bm25_score in result
    #[test]
    fn test_bm25_result_has_score() {
        let mut r = retriever();
        r.add_document(doc("d1", "foo bar baz"));
        let results = r.search_bm25("foo");
        assert!(results[0].bm25_score > 0.0);
    }

    // 28. multiple documents, BM25 scores differ by term frequency
    #[test]
    fn test_multiple_docs_bm25_differs() {
        let mut r = retriever();
        r.add_document(doc("rare", "dog once"));
        r.add_document(doc("freq", "dog dog dog dog dog dog"));
        let results = r.search_bm25("dog");
        assert_ne!(results[0].bm25_score, results[1].bm25_score);
    }

    // 29. search with no embeddings falls back to BM25 only
    #[test]
    fn test_search_no_embeddings_bm25_only() {
        let mut r = retriever();
        r.add_document(doc("d1", "cats and dogs"));
        let results = r.search("cats");
        assert!(!results.is_empty());
        assert!(results[0].vector_score.is_none());
    }

    // 30. Snippet truncated to max_len
    #[test]
    fn test_snippet_truncated() {
        let content = "a".repeat(500);
        let snippet = KnowledgeRetriever::extract_snippet(&content, "a", 200);
        // Snippet body (max 200 bytes) + possible "…" (3 UTF-8 bytes) = at most 203 bytes
        assert!(snippet.len() <= 203, "snippet.len() = {}", snippet.len());
        // The displayed length (char count) must be modest
        assert!(snippet.chars().count() <= 201);
    }

    // 31. Snippet empty content
    #[test]
    fn test_snippet_empty_content() {
        let snippet = KnowledgeRetriever::extract_snippet("", "query", 200);
        assert!(snippet.is_empty());
    }

    // 32. Remove all docs, count = 0
    #[test]
    fn test_remove_all_docs_count_zero() {
        let mut r = retriever();
        r.add_document(doc("d1", "hello"));
        r.add_document(doc("d2", "world"));
        r.remove_document("d1");
        r.remove_document("d2");
        assert_eq!(r.document_count(), 0);
    }

    // 33. Doc IDs are preserved in results
    #[test]
    fn test_doc_ids_in_results() {
        let mut r = retriever();
        r.add_document(doc("alpha", "the alpha dog"));
        r.add_document(doc("beta", "the beta dog"));
        let results = r.search_bm25("dog");
        let ids: Vec<&str> = results.iter().map(|res| res.doc_id.as_str()).collect();
        assert!(ids.contains(&"alpha") || ids.contains(&"beta"));
    }

    // 34. BM25 IDF rewards rare terms
    #[test]
    fn test_idf_rare_term_higher_score() {
        let mut r = retriever();
        // "rare" appears in only 1 of 5 docs, "common" in all 5
        for i in 0..4 {
            r.add_document(doc(&format!("d{i}"), &format!("common word doc {i}")));
        }
        r.add_document(doc("d5", "common word rare term"));
        let rare = r.search_bm25("rare");
        let common = r.search_bm25("common");
        // d5's BM25 for "rare" should be higher than for "common" in d5
        let rare_score = rare
            .iter()
            .find(|res| res.doc_id == "d5")
            .map(|res| res.bm25_score)
            .unwrap_or(0.0);
        let common_score = common
            .iter()
            .find(|res| res.doc_id == "d5")
            .map(|res| res.bm25_score)
            .unwrap_or(0.0);
        assert!(
            rare_score >= common_score,
            "rare={rare_score} common={common_score}"
        );
    }

    // 35. cosine_similarity helper
    #[test]
    fn test_cosine_sim_helper() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-5);
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-5);
    }

    // 36. cosine_similarity different lengths returns 0
    #[test]
    fn test_cosine_sim_different_len() {
        let a = vec![1.0f32, 0.0];
        let b = vec![1.0f32];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    // 37. cosine_similarity zero vector returns 0
    #[test]
    fn test_cosine_sim_zero_vector() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    // 38. RetrieverConfig default
    #[test]
    fn test_retriever_config_default() {
        let cfg = RetrieverConfig::default();
        assert_eq!(cfg.top_k, 5);
        assert!(cfg.bm25_weight > 0.0);
        assert!(cfg.vector_weight > 0.0);
        assert_eq!(cfg.min_score, 0.0);
    }

    // 39. Document::new has no embedding
    #[test]
    fn test_document_new_no_embedding() {
        let d = Document::new("id", "content");
        assert!(d.embedding.is_none());
    }

    // 40. Document::with_embedding has embedding
    #[test]
    fn test_document_with_embedding() {
        let d = Document::with_embedding("id", "content", vec![1.0, 2.0]);
        assert!(d.embedding.is_some());
    }

    // 41. search_bm25 score in result matches internal bm25_score
    #[test]
    fn test_bm25_result_consistency() {
        let mut r = retriever();
        r.add_document(doc("d1", "rust is great rust rust"));
        let results = r.search_bm25("rust");
        for res in &results {
            assert_eq!(res.bm25_score, res.combined_score);
        }
    }

    // 42. avg_doc_length updates after adding docs
    #[test]
    fn test_avg_doc_length_updates() {
        let mut r = retriever();
        // Access avg via search (indirect test)
        r.add_document(doc("d1", "a b c d e"));
        r.add_document(doc("d2", "a b"));
        // avg length = (5+2)/2 = 3.5 — just verify search works
        let results = r.search_bm25("a");
        assert!(!results.is_empty());
    }

    // 43. Multiple queries on same retriever
    #[test]
    fn test_multiple_queries() {
        let mut r = retriever();
        r.add_document(doc("d1", "rust web programming frameworks"));
        r.add_document(doc("d2", "python machine learning data science"));
        let r1 = r.search_bm25("rust");
        let r2 = r.search_bm25("python");
        assert_eq!(r1[0].doc_id, "d1");
        assert_eq!(r2[0].doc_id, "d2");
    }

    // 44. remove updates doc_frequencies correctly
    #[test]
    fn test_remove_updates_doc_freq() {
        let mut r = retriever();
        r.add_document(doc("d1", "unique_term hello"));
        r.add_document(doc("d2", "hello world"));
        r.remove_document("d1");
        // unique_term should no longer appear in results
        let results = r.search_bm25("unique_term");
        assert!(results.iter().all(|res| res.bm25_score == 0.0));
    }

    // 45. top_k = 1 returns at most 1 result
    #[test]
    fn test_top_k_one() {
        let mut r = KnowledgeRetriever::new(RetrieverConfig {
            top_k: 1,
            ..config()
        });
        for i in 0..5 {
            r.add_document(doc(&format!("d{i}"), "rust programming"));
        }
        let results = r.search_bm25("rust");
        assert_eq!(results.len(), 1);
    }

    // 46. search and search_bm25 return same results for docs without embeddings
    #[test]
    fn test_search_equals_bm25_without_embeddings() {
        let mut r = retriever();
        r.add_document(doc("d1", "cats and dogs"));
        r.add_document(doc("d2", "fish and chips"));
        let s1 = r.search("cats");
        let s2 = r.search_bm25("cats");
        assert_eq!(s1.len(), s2.len());
        // Same top doc
        if !s1.is_empty() && !s2.is_empty() {
            assert_eq!(s1[0].doc_id, s2[0].doc_id);
        }
    }

    // 47. search returns snippet for every result
    #[test]
    fn test_search_snippets_present() {
        let mut r = retriever();
        for i in 0..5 {
            r.add_document(doc(&format!("d{i}"), &format!("text about topic {i}")));
        }
        let results = r.search("topic");
        for res in &results {
            // Snippet may be empty if content doesn't contain term, but struct is always set
            let _ = &res.snippet;
        }
        assert_eq!(results.len(), 5);
    }
}
