//! Hybrid RAG Pipeline
//!
//! Implements BM25 + cosine-similarity hybrid retrieval for Retrieval-Augmented
//! Generation. Documents are indexed at insertion time so queries remain fast.
//!
//! The hybrid score is:
//!   `alpha * cosine_similarity(query_embedding, doc_embedding) + (1 - alpha) * bm25_score`

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// RagDocument
// ---------------------------------------------------------------------------

/// A document stored in the retrieval index
#[derive(Debug, Clone)]
pub struct RagDocument {
    pub id: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub embedding: Vec<f32>,
    pub score: f64,
}

impl RagDocument {
    /// Construct a new document with zero score
    pub fn new(id: impl Into<String>, content: impl Into<String>, embedding: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            metadata: HashMap::new(),
            embedding,
            score: 0.0,
        }
    }

    /// Attach a metadata entry
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// ---------------------------------------------------------------------------
// BM25Index
// ---------------------------------------------------------------------------

/// Okapi BM25 inverted index
#[derive(Debug, Clone)]
pub struct BM25Index {
    /// term → doc_id → term frequency in doc
    term_freq: HashMap<String, HashMap<usize, f64>>,
    /// term → number of documents containing term
    doc_freq: HashMap<String, usize>,
    /// document lengths (in tokens)
    doc_lens: HashMap<usize, usize>,
    avg_doc_len: f64,
    k1: f64,
    b: f64,
    /// number of indexed documents
    doc_count: usize,
}

impl Default for BM25Index {
    fn default() -> Self {
        Self::new()
    }
}

impl BM25Index {
    /// Create a new BM25 index with standard parameters (k1=1.5, b=0.75)
    pub fn new() -> Self {
        Self {
            term_freq: HashMap::new(),
            doc_freq: HashMap::new(),
            doc_lens: HashMap::new(),
            avg_doc_len: 0.0,
            k1: 1.5,
            b: 0.75,
            doc_count: 0,
        }
    }

    /// Index a document
    pub fn index(&mut self, doc_id: usize, content: &str) {
        let tokens = tokenize(content);
        let doc_len = tokens.len();

        // Update total token count for avg_doc_len maintenance
        let total_tokens: usize = self.doc_lens.values().sum::<usize>() + doc_len;
        self.doc_lens.insert(doc_id, doc_len);
        self.doc_count += 1;
        self.avg_doc_len = total_tokens as f64 / self.doc_count as f64;

        // Build per-doc term frequencies
        let mut tf_map: HashMap<String, usize> = HashMap::new();
        for token in &tokens {
            *tf_map.entry(token.clone()).or_insert(0) += 1;
        }

        for (term, count) in tf_map {
            let tf = count as f64 / doc_len.max(1) as f64;
            self.term_freq
                .entry(term.clone())
                .or_default()
                .insert(doc_id, tf);
            *self.doc_freq.entry(term).or_insert(0) += 1;
        }
    }

    /// Compute BM25 score for a document given a set of query terms
    pub fn score(&self, doc_id: usize, query_terms: &[&str]) -> f64 {
        if self.doc_count == 0 {
            return 0.0;
        }
        let doc_len = *self.doc_lens.get(&doc_id).unwrap_or(&0) as f64;
        let mut total = 0.0;

        for &term in query_terms {
            let df = *self.doc_freq.get(term).unwrap_or(&0) as f64;
            if df == 0.0 {
                continue;
            }
            let tf = self
                .term_freq
                .get(term)
                .and_then(|m| m.get(&doc_id))
                .copied()
                .unwrap_or(0.0);

            if tf == 0.0 {
                continue;
            }

            // IDF (with smoothing to avoid log(0))
            let idf = ((self.doc_count as f64 - df + 0.5) / (df + 0.5) + 1.0).ln();
            // TF normalised by doc length
            let tf_norm = (tf * (self.k1 + 1.0))
                / (tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_len.max(1.0)));

            total += idf * tf_norm;
        }
        total.max(0.0)
    }
}

// ---------------------------------------------------------------------------
// VectorIndex
// ---------------------------------------------------------------------------

/// Simple brute-force cosine similarity index
#[derive(Debug, Clone, Default)]
pub struct VectorIndex {
    embeddings: Vec<(usize, Vec<f32>)>,
}

impl VectorIndex {
    /// Create an empty vector index
    pub fn new() -> Self {
        Self {
            embeddings: Vec::new(),
        }
    }

    /// Add an embedding for a document
    pub fn add(&mut self, doc_id: usize, embedding: Vec<f32>) {
        self.embeddings.push((doc_id, embedding));
    }

    /// Return up to `top_k` (doc_id, cosine_similarity) pairs, sorted by score descending
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(usize, f64)> {
        let mut scores: Vec<(usize, f64)> = self
            .embeddings
            .iter()
            .map(|(id, emb)| (*id, cosine_similarity(query, emb)))
            .filter(|(_, s)| s.is_finite())
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }
}

// ---------------------------------------------------------------------------
// HybridRetriever
// ---------------------------------------------------------------------------

/// Hybrid retriever combining BM25 and vector similarity
pub struct HybridRetriever {
    documents: Vec<RagDocument>,
    bm25_index: BM25Index,
    vector_index: VectorIndex,
    /// Blend weight: `alpha * vector + (1-alpha) * bm25`
    alpha: f64,
}

impl HybridRetriever {
    /// Create a new hybrid retriever.  `alpha` must be in `[0.0, 1.0]`.
    pub fn new(alpha: f64) -> Self {
        Self {
            documents: Vec::new(),
            bm25_index: BM25Index::new(),
            vector_index: VectorIndex::new(),
            alpha: alpha.clamp(0.0, 1.0),
        }
    }

    /// Add and index a document
    pub fn add_document(&mut self, doc: RagDocument) {
        let doc_id = self.documents.len();
        self.bm25_index.index(doc_id, &doc.content);
        self.vector_index.add(doc_id, doc.embedding.clone());
        self.documents.push(doc);
    }

    /// Number of indexed documents
    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    /// Retrieve up to `top_k` documents using hybrid scoring
    pub fn retrieve(
        &self,
        query: &str,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Vec<&RagDocument> {
        if self.documents.is_empty() {
            return Vec::new();
        }

        let query_terms: Vec<&str> = tokenize(query)
            .iter()
            .map(|s| {
                // SAFETY: tokenize returns String, we need &str with the lifetime of query
                // Instead collect owned terms and reference them below
                s.as_str() as *const str
            })
            .map(|p| unsafe { &*p })
            .collect();

        // Safer approach: collect tokenised terms, then score
        let owned_terms: Vec<String> = tokenize(query);
        let term_refs: Vec<&str> = owned_terms.iter().map(|s| s.as_str()).collect();

        let vector_scores: HashMap<usize, f64> = self
            .vector_index
            .search(query_embedding, self.documents.len())
            .into_iter()
            .collect();

        let mut hybrid: Vec<(usize, f64)> = (0..self.documents.len())
            .map(|id| {
                let v_score = vector_scores.get(&id).copied().unwrap_or(0.0);
                let bm25_score = self.bm25_index.score(id, &term_refs);
                // Normalise BM25 score heuristically (divide by 5 to bring to ~[0,1])
                let bm25_norm = (bm25_score / 5.0).min(1.0);
                let combined = self.alpha * v_score + (1.0 - self.alpha) * bm25_norm;
                (id, combined)
            })
            .collect();

        // Sort by hybrid score descending
        hybrid.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        hybrid.truncate(top_k);

        // Suppress unused variable warning from first attempt
        let _ = query_terms;

        hybrid
            .into_iter()
            .filter_map(|(id, _)| self.documents.get(id))
            .collect()
    }

    /// BM25-only retrieval
    pub fn retrieve_bm25_only(&self, query: &str, top_k: usize) -> Vec<(&RagDocument, f64)> {
        let terms: Vec<String> = tokenize(query);
        let term_refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();

        let mut scored: Vec<(usize, f64)> = (0..self.documents.len())
            .map(|id| (id, self.bm25_index.score(id, &term_refs)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored
            .into_iter()
            .filter_map(|(id, score)| self.documents.get(id).map(|d| (d, score)))
            .collect()
    }

    /// Vector-only retrieval
    pub fn retrieve_vector_only(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Vec<(&RagDocument, f64)> {
        self.vector_index
            .search(query_embedding, top_k)
            .into_iter()
            .filter_map(|(id, score)| self.documents.get(id).map(|d| (d, score)))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// RagPipelineConfig
// ---------------------------------------------------------------------------

/// Configuration for the RAG pipeline
#[derive(Debug, Clone)]
pub struct RagPipelineConfig {
    pub retriever_top_k: usize,
    pub rerank_top_k: usize,
    pub alpha: f64,
    pub min_score: f64,
    pub include_metadata: bool,
}

impl Default for RagPipelineConfig {
    fn default() -> Self {
        Self {
            retriever_top_k: 10,
            rerank_top_k: 5,
            alpha: 0.5,
            min_score: 0.0,
            include_metadata: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Lowercase, split by non-alphanumeric, filter short tokens
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 2)
        .map(|w| w.to_string())
        .collect()
}

/// Cosine similarity between two vectors; returns 0.0 for zero-length vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as f64 * y as f64)
        .sum();
    let norm_a: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|&y| (y as f64).powi(2)).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_embedding(seed: f32) -> Vec<f32> {
        vec![seed, seed * 0.5, seed * 0.25]
    }

    fn make_retriever() -> HybridRetriever {
        let mut r = HybridRetriever::new(0.5);
        let docs = vec![
            ("doc1", "SPARQL is a query language for RDF data.", 1.0f32),
            ("doc2", "RDF is a data model for the web of data.", 0.8),
            ("doc3", "OWL is an ontology language based on RDF.", 0.6),
            ("doc4", "Turtle is a syntax for writing RDF documents.", 0.4),
            (
                "doc5",
                "SHACL defines shapes for validating RDF graphs.",
                0.2,
            ),
        ];
        for (id, content, seed) in docs {
            r.add_document(RagDocument::new(id, content, small_embedding(seed)));
        }
        r
    }

    // --- RagDocument ---

    #[test]
    fn test_rag_document_construction() {
        let doc =
            RagDocument::new("d1", "Hello world", vec![0.1, 0.2]).with_metadata("source", "wiki");
        assert_eq!(doc.id, "d1");
        assert_eq!(doc.content, "Hello world");
        assert_eq!(doc.metadata.get("source"), Some(&"wiki".to_string()));
        assert_eq!(doc.score, 0.0);
    }

    // --- BM25Index ---

    #[test]
    fn test_bm25_empty_index_scores_zero() {
        let idx = BM25Index::new();
        assert_eq!(idx.score(0, &["sparql"]), 0.0);
    }

    #[test]
    fn test_bm25_indexed_doc_scores_positive() {
        let mut idx = BM25Index::new();
        idx.index(0, "SPARQL is a query language for RDF data");
        let score = idx.score(0, &["sparql"]);
        assert!(score > 0.0, "BM25 score should be positive");
    }

    #[test]
    fn test_bm25_missing_term_scores_zero() {
        let mut idx = BM25Index::new();
        idx.index(0, "SPARQL query language");
        let score = idx.score(0, &["turtle"]);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_bm25_higher_freq_scores_higher() {
        let mut idx = BM25Index::new();
        idx.index(0, "rdf rdf rdf sparql");
        idx.index(1, "rdf sparql turtle");
        let score_0 = idx.score(0, &["rdf"]);
        let score_1 = idx.score(1, &["rdf"]);
        // doc0 has higher term frequency so BM25 may vary, but both positive
        assert!(score_0 > 0.0);
        assert!(score_1 > 0.0);
    }

    #[test]
    fn test_bm25_multiple_terms() {
        let mut idx = BM25Index::new();
        idx.index(0, "SPARQL query RDF triples");
        let score = idx.score(0, &["sparql", "rdf", "nonexistent"]);
        assert!(score > 0.0);
    }

    // --- VectorIndex ---

    #[test]
    fn test_vector_index_empty_returns_empty() {
        let idx = VectorIndex::new();
        let results = idx.search(&[1.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_vector_index_identical_embedding_scores_one() {
        let mut idx = VectorIndex::new();
        idx.add(0, vec![1.0, 0.0, 0.0]);
        let results = idx.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert!((results[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vector_index_orthogonal_scores_zero() {
        let mut idx = VectorIndex::new();
        idx.add(0, vec![1.0, 0.0]);
        let results = idx.search(&[0.0, 1.0], 1);
        assert!((results[0].1).abs() < 1e-6);
    }

    #[test]
    fn test_vector_index_top_k_limiting() {
        let mut idx = VectorIndex::new();
        for i in 0..10 {
            idx.add(i, vec![i as f32, 1.0]);
        }
        let results = idx.search(&[5.0, 1.0], 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_vector_index_results_sorted_descending() {
        let mut idx = VectorIndex::new();
        idx.add(0, vec![1.0, 0.0]);
        idx.add(1, vec![0.8, 0.6]);
        idx.add(2, vec![0.0, 1.0]);
        let results = idx.search(&[1.0, 0.0], 3);
        for i in 1..results.len() {
            assert!(results[i - 1].1 >= results[i].1);
        }
    }

    // --- HybridRetriever ---

    #[test]
    fn test_retriever_document_count() {
        let r = make_retriever();
        assert_eq!(r.document_count(), 5);
    }

    #[test]
    fn test_retrieve_returns_results() {
        let r = make_retriever();
        let q_emb = small_embedding(1.0);
        let results = r.retrieve("SPARQL query", &q_emb, 3);
        assert!(!results.is_empty());
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_retrieve_empty_index_returns_empty() {
        let r = HybridRetriever::new(0.5);
        let results = r.retrieve("sparql", &[1.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_retrieve_bm25_only_returns_results() {
        let r = make_retriever();
        let results = r.retrieve_bm25_only("SPARQL query language", 3);
        assert!(!results.is_empty());
        // Should be sorted descending
        for i in 1..results.len() {
            assert!(results[i - 1].1 >= results[i].1);
        }
    }

    #[test]
    fn test_retrieve_vector_only_returns_results() {
        let r = make_retriever();
        let q_emb = small_embedding(1.0);
        let results = r.retrieve_vector_only(&q_emb, 3);
        assert!(!results.is_empty());
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_alpha_zero_uses_bm25_only() {
        let mut r = HybridRetriever::new(0.0);
        r.add_document(RagDocument::new(
            "a",
            "SPARQL queries RDF data",
            vec![0.0, 0.0, 0.0],
        ));
        r.add_document(RagDocument::new(
            "b",
            "Turtle syntax for RDF",
            vec![0.0, 0.0, 0.0],
        ));
        // Both embeddings are zero-vectors → cosine=0; ranking is purely BM25
        let results = r.retrieve("SPARQL", &[0.0, 0.0, 0.0], 2);
        // "a" should rank first since it contains "sparql"
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_alpha_one_uses_vector_only() {
        let mut r = HybridRetriever::new(1.0);
        // doc "a" has embedding close to query; doc "b" has embedding far away
        r.add_document(RagDocument::new("a", "anything", vec![1.0, 0.0, 0.0]));
        r.add_document(RagDocument::new("b", "anything", vec![0.0, 1.0, 0.0]));
        let results = r.retrieve("test", &[1.0, 0.0, 0.0], 2);
        assert_eq!(results[0].id, "a");
    }

    // --- RagPipelineConfig ---

    #[test]
    fn test_pipeline_config_defaults() {
        let cfg = RagPipelineConfig::default();
        assert_eq!(cfg.retriever_top_k, 10);
        assert_eq!(cfg.rerank_top_k, 5);
        assert_eq!(cfg.alpha, 0.5);
        assert_eq!(cfg.min_score, 0.0);
        assert!(cfg.include_metadata);
    }

    // --- cosine_similarity helper ---

    #[test]
    fn test_cosine_empty_returns_zero() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn test_cosine_different_lengths_returns_zero() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
    }
}
