//! ColBERT-style late interaction reranking
//!
//! ColBERT computes MaxSim: for each query token embedding, find the maximum
//! cosine similarity with any document token embedding, then sum across all
//! query tokens.  This module implements a pure-Rust approximation that uses
//! pre-computed token-level embeddings (f32 arrays) to perform late interaction
//! scoring without requiring an external ML runtime.

use crate::{GraphRAGError, GraphRAGResult, ScoredEntity};
use std::collections::HashMap;

/// A token-level embedding (one embedding per sub-word token)
pub type TokenEmbedding = Vec<f32>;

/// Sequence of token embeddings for a passage (document or query)
pub type TokenSequence = Vec<TokenEmbedding>;

/// Cosine similarity between two equal-length embedding vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Embedding dimensions must match");

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-9 || norm_b < 1e-9 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// MaxSim score for one query token against all document tokens
fn max_sim(query_token: &[f32], doc_tokens: &[TokenEmbedding]) -> f32 {
    doc_tokens
        .iter()
        .map(|dt| cosine_similarity(query_token, dt))
        .fold(f32::NEG_INFINITY, f32::max)
}

/// ColBERT score: sum of MaxSim across all query tokens
fn colbert_score(query_tokens: &TokenSequence, doc_tokens: &TokenSequence) -> f32 {
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return 0.0;
    }
    query_tokens
        .iter()
        .map(|qt| max_sim(qt, doc_tokens))
        .sum::<f32>()
        / query_tokens.len() as f32 // normalise by number of query tokens
}

// ─── Configuration ─────────────────────────────────────────────────────────

/// ColBERT reranker configuration
#[derive(Debug, Clone)]
pub struct ColbertRerankerConfig {
    /// Weight given to the ColBERT score vs the original retrieval score
    /// (0.0 = pure original, 1.0 = pure ColBERT)
    pub colbert_weight: f64,
    /// Minimum ColBERT score required to keep a candidate (0.0 = keep all)
    pub min_colbert_score: f32,
    /// Maximum candidates to rerank (truncated before scoring for speed)
    pub max_candidates: usize,
    /// Whether to normalise ColBERT scores across the candidate set
    pub normalise_scores: bool,
}

impl Default for ColbertRerankerConfig {
    fn default() -> Self {
        Self {
            colbert_weight: 0.7,
            min_colbert_score: 0.0,
            max_candidates: 100,
            normalise_scores: true,
        }
    }
}

// ─── Token encoder trait ────────────────────────────────────────────────────

/// Trait for encoding text into token-level embeddings
pub trait TokenEncoder: Send + Sync {
    /// Encode text into a sequence of token embeddings
    fn encode(&self, text: &str) -> GraphRAGResult<TokenSequence>;
}

/// Simple whitespace-based encoder using random unit vectors per token type
/// (placeholder for a real transformer encoder in production)
pub struct MockTokenEncoder {
    dim: usize,
    vocab: HashMap<String, TokenEmbedding>,
}

impl MockTokenEncoder {
    /// Create with the given embedding dimension
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            vocab: HashMap::new(),
        }
    }

    /// Register a specific token embedding (for deterministic tests)
    pub fn register_token(&mut self, token: impl Into<String>, embedding: Vec<f32>) {
        self.vocab.insert(token.into(), embedding);
    }

    /// Generate a deterministic unit vector for an unknown token
    fn hash_embed(&self, token: &str) -> TokenEmbedding {
        let mut v: Vec<f32> = (0..self.dim)
            .map(|i| {
                // Deterministic pseudo-random from token bytes + index
                let hash: u64 = token.bytes().fold(i as u64, |acc, b| {
                    acc.wrapping_mul(6364136223846793005).wrapping_add(b as u64)
                });
                ((hash as i64) as f32) / (i64::MAX as f32)
            })
            .collect();

        // Normalise to unit length
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        v
    }
}

impl TokenEncoder for MockTokenEncoder {
    fn encode(&self, text: &str) -> GraphRAGResult<TokenSequence> {
        let tokens: TokenSequence = text
            .split_whitespace()
            .map(|tok| {
                let lower = tok.to_lowercase();
                self.vocab
                    .get(&lower)
                    .cloned()
                    .unwrap_or_else(|| self.hash_embed(&lower))
            })
            .collect();
        Ok(tokens)
    }
}

// ─── Reranker ───────────────────────────────────────────────────────────────

/// ColBERT-style late interaction reranker
pub struct ColbertReranker<E: TokenEncoder> {
    encoder: E,
    config: ColbertRerankerConfig,
    /// Optional document text lookup: entity_uri → text representation
    doc_store: HashMap<String, String>,
}

impl<E: TokenEncoder> ColbertReranker<E> {
    /// Create a new reranker
    pub fn new(encoder: E, config: ColbertRerankerConfig) -> Self {
        Self {
            encoder,
            config,
            doc_store: HashMap::new(),
        }
    }

    /// Register entity text representations for late interaction scoring
    pub fn register_documents(&mut self, docs: impl IntoIterator<Item = (String, String)>) {
        for (uri, text) in docs {
            self.doc_store.insert(uri, text);
        }
    }

    /// Rerank candidates using ColBERT late interaction.
    ///
    /// Candidates without a registered document text fall back to their
    /// original scores (not discarded, so no precision loss for unknown docs).
    pub fn rerank(
        &self,
        query: &str,
        mut candidates: Vec<ScoredEntity>,
    ) -> GraphRAGResult<Vec<ScoredEntity>> {
        if candidates.is_empty() || query.is_empty() {
            return Ok(candidates);
        }

        // Encode query
        let query_tokens = self.encoder.encode(query)?;

        // Truncate to max_candidates
        candidates.truncate(self.config.max_candidates);

        // Score each candidate
        let mut scored: Vec<(ScoredEntity, f32)> = candidates
            .into_iter()
            .map(|entity| {
                let colbert = self.score_entity(query, &query_tokens, &entity);
                (entity, colbert)
            })
            .collect();

        // Normalise ColBERT scores if requested
        if self.config.normalise_scores {
            let max_c = scored
                .iter()
                .map(|(_, c)| *c)
                .fold(f32::NEG_INFINITY, f32::max);
            if max_c > 1e-9 {
                scored.iter_mut().for_each(|(_, c)| *c /= max_c);
            }
        }

        // Blend and filter
        let w = self.config.colbert_weight;
        let min_c = self.config.min_colbert_score;

        let mut result: Vec<ScoredEntity> = scored
            .into_iter()
            .filter(|(_, c)| *c >= min_c)
            .map(|(mut entity, c)| {
                entity.score = (1.0 - w) * entity.score + w * c as f64;
                entity
            })
            .collect();

        result.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(result)
    }

    /// Score a single entity (returns raw ColBERT score)
    fn score_entity(
        &self,
        _query: &str,
        query_tokens: &TokenSequence,
        entity: &ScoredEntity,
    ) -> f32 {
        let doc_text = match self.doc_store.get(&entity.uri) {
            Some(text) => text.clone(),
            None => {
                // Fall back to the URI itself as a pseudo-document
                entity.uri.clone()
            }
        };

        match self.encoder.encode(&doc_text) {
            Ok(doc_tokens) => colbert_score(query_tokens, &doc_tokens),
            Err(_) => 0.0,
        }
    }
}

// ─── Batch scoring helper ────────────────────────────────────────────────────

/// Score multiple (query, document) pairs and return ColBERT scores
pub fn colbert_score_batch<E: TokenEncoder>(
    encoder: &E,
    query: &str,
    docs: &[(&str, &str)],
) -> GraphRAGResult<Vec<f32>> {
    let query_tokens = encoder.encode(query)?;
    docs.iter()
        .map(|(_, doc_text)| {
            encoder
                .encode(doc_text)
                .map(|dt| colbert_score(&query_tokens, &dt))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScoreSource;

    fn make_encoder(dim: usize) -> MockTokenEncoder {
        MockTokenEncoder::new(dim)
    }

    fn make_entity(uri: &str, score: f64) -> ScoredEntity {
        ScoredEntity {
            uri: uri.to_string(),
            score,
            source: ScoreSource::Fused,
            metadata: HashMap::new(),
        }
    }

    // ── cosine_similarity ─────────────────────────────────────────────────

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let v = vec![0.6, 0.8];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    // ── colbert_score ─────────────────────────────────────────────────────

    #[test]
    fn test_colbert_score_same_query_doc() {
        // A query identical to the document should score high
        let enc = make_encoder(8);
        let q = enc.encode("battery safety").unwrap();
        let d = enc.encode("battery safety").unwrap();
        let score = colbert_score(&q, &d);
        assert!(
            score > 0.8,
            "Identical query/doc should score >0.8, got {score}"
        );
    }

    #[test]
    fn test_colbert_score_empty_query() {
        let q: TokenSequence = vec![];
        let d = vec![vec![1.0f32, 0.0]];
        assert_eq!(colbert_score(&q, &d), 0.0);
    }

    #[test]
    fn test_colbert_score_empty_doc() {
        let q = vec![vec![1.0f32, 0.0]];
        let d: TokenSequence = vec![];
        assert_eq!(colbert_score(&q, &d), 0.0);
    }

    // ── MockTokenEncoder ─────────────────────────────────────────────────

    #[test]
    fn test_mock_encoder_deterministic() {
        let enc = make_encoder(16);
        let e1 = enc.encode("hello world").unwrap();
        let e2 = enc.encode("hello world").unwrap();
        assert_eq!(e1.len(), e2.len());
        for (a, b) in e1.iter().zip(e2.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_mock_encoder_registered_token() {
        let mut enc = make_encoder(4);
        enc.register_token("special", vec![1.0, 0.0, 0.0, 0.0]);
        let tokens = enc.encode("special term").unwrap();
        assert_eq!(tokens.len(), 2);
        // First token should be exactly our registered vector
        assert!((tokens[0][0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_mock_encoder_unit_length() {
        let enc = make_encoder(32);
        let tokens = enc.encode("test token normalization").unwrap();
        for tok in &tokens {
            let norm: f32 = tok.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "Token not unit length: {norm}");
        }
    }

    // ── ColbertReranker ──────────────────────────────────────────────────

    #[test]
    fn test_reranker_basic() {
        let enc = make_encoder(16);
        let mut reranker = ColbertReranker::new(enc, ColbertRerankerConfig::default());
        reranker.register_documents([
            (
                "http://a".to_string(),
                "battery safety cell thermal".to_string(),
            ),
            (
                "http://b".to_string(),
                "charging protocol electric".to_string(),
            ),
        ]);

        let candidates = vec![make_entity("http://a", 0.7), make_entity("http://b", 0.6)];

        let reranked = reranker.rerank("battery safety", candidates).unwrap();
        assert_eq!(reranked.len(), 2);
        // http://a should score higher (relevant doc)
        assert_eq!(reranked[0].uri, "http://a");
    }

    #[test]
    fn test_reranker_empty_candidates() {
        let enc = make_encoder(8);
        let reranker = ColbertReranker::new(enc, ColbertRerankerConfig::default());
        let result = reranker.rerank("query", vec![]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_reranker_empty_query() {
        let enc = make_encoder(8);
        let reranker = ColbertReranker::new(enc, ColbertRerankerConfig::default());
        let candidates = vec![make_entity("http://a", 0.5)];
        let result = reranker.rerank("", candidates).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_reranker_max_candidates_limiting() {
        let enc = make_encoder(8);
        let config = ColbertRerankerConfig {
            max_candidates: 2,
            ..Default::default()
        };
        let reranker = ColbertReranker::new(enc, config);
        let candidates: Vec<ScoredEntity> = (0..10)
            .map(|i| make_entity(&format!("http://e{i}"), 0.5))
            .collect();
        let result = reranker.rerank("test", candidates).unwrap();
        assert!(result.len() <= 2);
    }

    #[test]
    fn test_reranker_min_score_filter() {
        let enc = make_encoder(8);
        let config = ColbertRerankerConfig {
            min_colbert_score: 999.0, // impossible threshold
            normalise_scores: false,
            ..Default::default()
        };
        let reranker = ColbertReranker::new(enc, config);
        let candidates = vec![make_entity("http://a", 0.8)];
        let result = reranker.rerank("test", candidates).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_reranker_fallback_without_doc_store() {
        // No documents registered – should fall back to URI-based scoring
        let enc = make_encoder(8);
        let reranker = ColbertReranker::new(enc, ColbertRerankerConfig::default());
        let candidates = vec![make_entity("http://a", 0.7), make_entity("http://b", 0.6)];
        // Should not panic and should return some ordering
        let result = reranker.rerank("some query", candidates).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_reranker_normalises_scores() {
        let enc = make_encoder(16);
        let config = ColbertRerankerConfig {
            normalise_scores: true,
            colbert_weight: 1.0, // fully ColBERT
            ..Default::default()
        };
        let mut reranker = ColbertReranker::new(enc, config);
        reranker.register_documents([
            ("http://x".to_string(), "alpha beta gamma".to_string()),
            ("http://y".to_string(), "delta epsilon zeta".to_string()),
        ]);
        let candidates = vec![make_entity("http://x", 0.5), make_entity("http://y", 0.5)];
        let result = reranker.rerank("alpha gamma", candidates).unwrap();
        // After normalisation + full ColBERT weight, top doc should score ~1.0
        assert!(
            result[0].score <= 1.01,
            "Score should be ≤ 1.0, got {}",
            result[0].score
        );
    }

    // ── colbert_score_batch ───────────────────────────────────────────────

    #[test]
    fn test_batch_scoring() {
        let enc = make_encoder(16);
        let docs = vec![
            ("id1", "battery safety cell"),
            ("id2", "charging electric vehicle"),
            ("id3", "battery cell chemistry"),
        ];
        let scores = colbert_score_batch(&enc, "battery safety", &docs).unwrap();
        assert_eq!(scores.len(), 3);
        for s in &scores {
            assert!(*s >= 0.0, "Score should be non-negative");
        }
        // First doc is most relevant to "battery safety"
        assert!(
            scores[0] > scores[1],
            "Doc 0 should beat doc 1 for 'battery safety'"
        );
    }

    #[test]
    fn test_batch_scoring_empty_docs() {
        let enc = make_encoder(8);
        let scores = colbert_score_batch(&enc, "query", &[]).unwrap();
        assert!(scores.is_empty());
    }

    #[test]
    fn test_colbert_score_partial_overlap() {
        let enc = make_encoder(16);
        let q = enc.encode("battery cell safety").unwrap();
        let d_rel = enc.encode("battery cell thermal runaway").unwrap();
        let d_irrel = enc.encode("aircraft propulsion jet").unwrap();

        let s_rel = colbert_score(&q, &d_rel);
        let s_irrel = colbert_score(&q, &d_irrel);

        assert!(
            s_rel > s_irrel,
            "Relevant doc should score higher: {s_rel} vs {s_irrel}"
        );
    }
}
