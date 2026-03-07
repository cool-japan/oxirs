//! Batch text encoding pipeline with chunking and pooling strategies.
//!
//! Provides deterministic text-to-embedding conversion with configurable pooling,
//! normalization, and similarity computation — all without external ML dependencies.

use std::f64::consts::PI;

/// Pooling strategy for aggregating token-level embeddings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Mean of all token embeddings
    Mean,
    /// Element-wise maximum across token embeddings
    Max,
    /// Use the first token (CLS-like) embedding
    CLS,
    /// Use the last token embedding
    Last,
}

/// Configuration for the batch encoder.
#[derive(Debug, Clone)]
pub struct EncodingConfig {
    /// Maximum number of tokens per text (truncated if exceeded)
    pub max_length: usize,
    /// Number of texts to process per batch
    pub batch_size: usize,
    /// Pooling strategy to aggregate token embeddings
    pub pooling: PoolingStrategy,
    /// Whether to L2-normalise the final embedding
    pub normalize: bool,
}

impl Default for EncodingConfig {
    fn default() -> Self {
        Self {
            max_length: 128,
            batch_size: 32,
            pooling: PoolingStrategy::Mean,
            normalize: true,
        }
    }
}

/// A tokenised representation of a single text string.
#[derive(Debug, Clone)]
pub struct TokenizedText {
    /// Raw token strings
    pub tokens: Vec<String>,
    /// Sequential integer IDs assigned to each token
    pub ids: Vec<u32>,
    /// Attention mask (1 = real token, 0 = padding — always 1 here)
    pub attention_mask: Vec<u8>,
}

/// The output of encoding a batch of texts.
#[derive(Debug, Clone)]
pub struct EncodedBatch {
    /// One embedding vector per input text
    pub embeddings: Vec<Vec<f32>>,
    /// Number of tokens for each input text (after truncation)
    pub token_counts: Vec<usize>,
    /// The actual number of texts in this batch
    pub batch_size: usize,
}

/// Embedding dimensionality produced by this encoder.
const EMBED_DIM: usize = 128;

/// A large prime used in the deterministic ID hash to spread token IDs.
const HASH_PRIME: u32 = 7919;

/// Batch text encoder: tokenises, embeds, pools, and normalises text.
pub struct BatchEncoder {
    config: EncodingConfig,
    /// Stable token vocabulary built lazily (token string → ID).
    vocab: std::collections::HashMap<String, u32>,
    /// Next available vocabulary ID.
    next_id: u32,
}

impl BatchEncoder {
    /// Create a new encoder with the given configuration.
    pub fn new(config: EncodingConfig) -> Self {
        Self {
            config,
            vocab: std::collections::HashMap::new(),
            next_id: 1, // 0 reserved for unknown/padding
        }
    }

    /// Tokenise `text` by splitting on whitespace, truncating to `max_length`,
    /// and assigning sequential IDs from a growing vocabulary.
    pub fn tokenize(&mut self, text: &str) -> TokenizedText {
        let raw_tokens: Vec<String> = text.split_whitespace().map(|t| t.to_lowercase()).collect();

        let truncated: Vec<String> = raw_tokens
            .into_iter()
            .take(self.config.max_length)
            .collect();

        let ids: Vec<u32> = truncated
            .iter()
            .map(|tok| {
                if let Some(&id) = self.vocab.get(tok) {
                    id
                } else {
                    let id = self.next_id;
                    self.vocab.insert(tok.clone(), id);
                    self.next_id = self.next_id.saturating_add(1);
                    id
                }
            })
            .collect();

        let attention_mask = vec![1u8; truncated.len()];

        TokenizedText {
            tokens: truncated,
            ids,
            attention_mask,
        }
    }

    /// Produce a deterministic 128-dimensional embedding for a single token ID.
    ///
    /// Each dimension `d` is computed as:
    ///   `cos(2π * ((id * HASH_PRIME + d) mod 997) / 997)`
    /// for even dimensions, and the sine counterpart for odd dimensions.
    /// This ensures distinctness across tokens without any randomness.
    fn token_embedding(id: u32) -> Vec<f32> {
        let mut emb = Vec::with_capacity(EMBED_DIM);
        for d in 0..EMBED_DIM {
            let phase = ((id.wrapping_mul(HASH_PRIME).wrapping_add(d as u32)) % 997) as f64 / 997.0
                * 2.0
                * PI;
            let val = if d % 2 == 0 { phase.cos() } else { phase.sin() };
            emb.push(val as f32);
        }
        emb
    }

    /// Encode a single text string into a 128-dimensional embedding.
    ///
    /// Steps: tokenise → produce per-token embeddings → pool → optionally normalise.
    pub fn encode_single(&mut self, text: &str) -> Vec<f32> {
        let tokenized = self.tokenize(text);

        if tokenized.ids.is_empty() {
            // Return zero vector for empty input
            return vec![0.0f32; EMBED_DIM];
        }

        let token_embs: Vec<Vec<f32>> = tokenized
            .ids
            .iter()
            .map(|&id| Self::token_embedding(id))
            .collect();

        let mut pooled = Self::pool(token_embs, &self.config.pooling.clone());

        if self.config.normalize {
            Self::normalize_l2(&mut pooled);
        }

        pooled
    }

    /// Encode a slice of text strings in chunks of `batch_size`.
    pub fn encode_batch(&mut self, texts: &[&str]) -> EncodedBatch {
        let mut embeddings = Vec::with_capacity(texts.len());
        let mut token_counts = Vec::with_capacity(texts.len());

        // Process in chunks of batch_size
        for chunk in texts.chunks(self.config.batch_size) {
            for &text in chunk {
                let tokenized = self.tokenize(text);
                let count = tokenized.ids.len();
                token_counts.push(count);

                if tokenized.ids.is_empty() {
                    embeddings.push(vec![0.0f32; EMBED_DIM]);
                    continue;
                }

                let token_embs: Vec<Vec<f32>> = tokenized
                    .ids
                    .iter()
                    .map(|&id| Self::token_embedding(id))
                    .collect();

                let mut pooled = Self::pool(token_embs, &self.config.pooling.clone());

                if self.config.normalize {
                    Self::normalize_l2(&mut pooled);
                }

                embeddings.push(pooled);
            }
        }

        let batch_size = embeddings.len();
        EncodedBatch {
            embeddings,
            token_counts,
            batch_size,
        }
    }

    /// Aggregate a list of per-token embedding vectors according to `strategy`.
    pub fn pool(token_embeddings: Vec<Vec<f32>>, strategy: &PoolingStrategy) -> Vec<f32> {
        if token_embeddings.is_empty() {
            return vec![0.0f32; EMBED_DIM];
        }

        let dim = token_embeddings[0].len();
        let n = token_embeddings.len();

        match strategy {
            PoolingStrategy::Mean => {
                let mut result = vec![0.0f32; dim];
                for emb in &token_embeddings {
                    for (r, &v) in result.iter_mut().zip(emb.iter()) {
                        *r += v;
                    }
                }
                for r in result.iter_mut() {
                    *r /= n as f32;
                }
                result
            }
            PoolingStrategy::Max => {
                let mut result = vec![f32::NEG_INFINITY; dim];
                for emb in &token_embeddings {
                    for (r, &v) in result.iter_mut().zip(emb.iter()) {
                        if v > *r {
                            *r = v;
                        }
                    }
                }
                result
            }
            PoolingStrategy::CLS => {
                // First token (index 0)
                token_embeddings[0].clone()
            }
            PoolingStrategy::Last => {
                // Last token
                token_embeddings[n - 1].clone()
            }
        }
    }

    /// Normalise a vector in-place to unit L2 norm.
    /// If the norm is zero, the vector is left unchanged.
    pub fn normalize_l2(v: &mut [f32]) {
        let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }

    /// Cosine similarity between two embedding vectors.
    /// Returns 0.0 if either vector has zero norm.
    pub fn similarity(a: &[f32], b: &[f32]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| x as f64 * y as f64)
            .sum();
        let norm_a: f64 = a
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt();
        let norm_b: f64 = b
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt();
        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }

    /// Return the number of unique tokens in the vocabulary so far.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_encoder() -> BatchEncoder {
        BatchEncoder::new(EncodingConfig::default())
    }

    // --- Tokenization tests ---

    #[test]
    fn test_tokenize_basic() {
        let mut enc = default_encoder();
        let t = enc.tokenize("hello world");
        assert_eq!(t.tokens, vec!["hello", "world"]);
        assert_eq!(t.ids.len(), 2);
        assert_eq!(t.attention_mask, vec![1, 1]);
    }

    #[test]
    fn test_tokenize_empty_string() {
        let mut enc = default_encoder();
        let t = enc.tokenize("");
        assert!(t.tokens.is_empty());
        assert!(t.ids.is_empty());
        assert!(t.attention_mask.is_empty());
    }

    #[test]
    fn test_tokenize_single_token() {
        let mut enc = default_encoder();
        let t = enc.tokenize("rust");
        assert_eq!(t.tokens, vec!["rust"]);
        assert_eq!(t.ids.len(), 1);
    }

    #[test]
    fn test_tokenize_lowercases() {
        let mut enc = default_encoder();
        let t = enc.tokenize("Hello WORLD");
        assert_eq!(t.tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_tokenize_truncation() {
        let config = EncodingConfig {
            max_length: 3,
            ..EncodingConfig::default()
        };
        let mut enc = BatchEncoder::new(config);
        let t = enc.tokenize("a b c d e");
        assert_eq!(t.tokens.len(), 3);
        assert_eq!(t.ids.len(), 3);
    }

    #[test]
    fn test_tokenize_max_length_exact() {
        let config = EncodingConfig {
            max_length: 2,
            ..EncodingConfig::default()
        };
        let mut enc = BatchEncoder::new(config);
        let t = enc.tokenize("x y");
        assert_eq!(t.tokens.len(), 2);
    }

    #[test]
    fn test_tokenize_consistent_ids() {
        let mut enc = default_encoder();
        let t1 = enc.tokenize("hello");
        let t2 = enc.tokenize("hello");
        assert_eq!(t1.ids, t2.ids);
    }

    #[test]
    fn test_tokenize_different_words_different_ids() {
        let mut enc = default_encoder();
        let t1 = enc.tokenize("foo");
        let t2 = enc.tokenize("bar");
        assert_ne!(t1.ids[0], t2.ids[0]);
    }

    // --- Encode single tests ---

    #[test]
    fn test_encode_single_returns_128_dim() {
        let mut enc = default_encoder();
        let emb = enc.encode_single("hello world");
        assert_eq!(emb.len(), EMBED_DIM);
    }

    #[test]
    fn test_encode_single_deterministic() {
        let mut enc1 = default_encoder();
        let mut enc2 = default_encoder();
        let e1 = enc1.encode_single("deterministic test");
        let e2 = enc2.encode_single("deterministic test");
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_encode_single_normalized_when_flag_set() {
        let mut enc = default_encoder();
        let emb = enc.encode_single("normalize me please");
        let norm: f32 = emb.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "Expected unit norm, got {norm}");
    }

    #[test]
    fn test_encode_single_no_normalize() {
        let config = EncodingConfig {
            normalize: false,
            ..EncodingConfig::default()
        };
        let mut enc = BatchEncoder::new(config);
        let emb = enc.encode_single("no norm");
        let norm: f32 = emb.iter().map(|&x| x * x).sum::<f32>().sqrt();
        // Not necessarily unit norm
        assert!(norm >= 0.0);
    }

    #[test]
    fn test_encode_single_empty_returns_zeros() {
        let mut enc = default_encoder();
        let emb = enc.encode_single("");
        assert_eq!(emb.len(), EMBED_DIM);
        assert!(emb.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_encode_single_different_texts_different_embeddings() {
        let mut enc = default_encoder();
        let e1 = enc.encode_single("apple banana cherry");
        let e2 = enc.encode_single("dog cat fish");
        // With the same encoder, same tokens get same IDs; different tokens → different embeddings
        assert_ne!(e1, e2);
    }

    // --- Encode batch tests ---

    #[test]
    fn test_encode_batch_count() {
        let mut enc = default_encoder();
        let texts = ["one", "two", "three"];
        let batch = enc.encode_batch(&texts);
        assert_eq!(batch.batch_size, 3);
        assert_eq!(batch.embeddings.len(), 3);
        assert_eq!(batch.token_counts.len(), 3);
    }

    #[test]
    fn test_encode_batch_each_128_dim() {
        let mut enc = default_encoder();
        let texts = ["alpha", "beta gamma", "delta epsilon zeta"];
        let batch = enc.encode_batch(&texts);
        for emb in &batch.embeddings {
            assert_eq!(emb.len(), EMBED_DIM);
        }
    }

    #[test]
    fn test_encode_batch_token_counts_correct() {
        let mut enc = BatchEncoder::new(EncodingConfig {
            max_length: 10,
            ..EncodingConfig::default()
        });
        let texts = ["a b c", "x", "one two three four"];
        let batch = enc.encode_batch(&texts);
        assert_eq!(batch.token_counts[0], 3);
        assert_eq!(batch.token_counts[1], 1);
        assert_eq!(batch.token_counts[2], 4);
    }

    #[test]
    fn test_encode_batch_chunking() {
        let config = EncodingConfig {
            batch_size: 2,
            ..EncodingConfig::default()
        };
        let mut enc = BatchEncoder::new(config);
        let texts: Vec<&str> = (0..5).map(|_| "hello world").collect();
        let batch = enc.encode_batch(&texts);
        assert_eq!(batch.batch_size, 5);
    }

    #[test]
    fn test_encode_batch_empty_texts() {
        let mut enc = default_encoder();
        let texts: Vec<&str> = vec![];
        let batch = enc.encode_batch(&texts);
        assert_eq!(batch.batch_size, 0);
    }

    #[test]
    fn test_encode_batch_single_text() {
        let mut enc = default_encoder();
        let texts = ["only one"];
        let batch = enc.encode_batch(&texts);
        assert_eq!(batch.batch_size, 1);
    }

    // --- Pooling strategy tests ---

    fn sample_token_embeddings() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 0.0, 2.0, -1.0],
            vec![0.0, 3.0, -1.0, 2.0],
            vec![2.0, 1.0, 0.0, 0.5],
        ]
    }

    #[test]
    fn test_pool_mean() {
        let embs = sample_token_embeddings();
        let result = BatchEncoder::pool(embs, &PoolingStrategy::Mean);
        let expected = [1.0, 4.0 / 3.0, 1.0 / 3.0, 0.5];
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-5, "{r} != {e}");
        }
    }

    #[test]
    fn test_pool_max() {
        let embs = sample_token_embeddings();
        let result = BatchEncoder::pool(embs, &PoolingStrategy::Max);
        let expected = vec![2.0f32, 3.0, 2.0, 2.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pool_cls() {
        let embs = sample_token_embeddings();
        let result = BatchEncoder::pool(embs, &PoolingStrategy::CLS);
        assert_eq!(result, vec![1.0, 0.0, 2.0, -1.0]);
    }

    #[test]
    fn test_pool_last() {
        let embs = sample_token_embeddings();
        let result = BatchEncoder::pool(embs, &PoolingStrategy::Last);
        assert_eq!(result, vec![2.0, 1.0, 0.0, 0.5]);
    }

    #[test]
    fn test_pool_empty() {
        let result = BatchEncoder::pool(vec![], &PoolingStrategy::Mean);
        assert_eq!(result.len(), EMBED_DIM);
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_pool_single_token_mean() {
        let embs = vec![vec![1.0, 2.0, 3.0]];
        let result = BatchEncoder::pool(embs.clone(), &PoolingStrategy::Mean);
        assert_eq!(result, embs[0]);
    }

    #[test]
    fn test_pool_single_token_max() {
        let embs = vec![vec![4.0, 5.0, 6.0]];
        let result = BatchEncoder::pool(embs.clone(), &PoolingStrategy::Max);
        assert_eq!(result, embs[0]);
    }

    // --- Normalize tests ---

    #[test]
    fn test_normalize_unit_norm() {
        let mut v = vec![3.0f32, 4.0, 0.0];
        BatchEncoder::normalize_l2(&mut v);
        let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-5);
        assert!((v[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let mut v = vec![0.0f32, 0.0, 0.0];
        BatchEncoder::normalize_l2(&mut v);
        // Should remain zero
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_normalize_already_unit() {
        let mut v = vec![1.0f32, 0.0, 0.0];
        BatchEncoder::normalize_l2(&mut v);
        assert!((v[0] - 1.0).abs() < 1e-6);
    }

    // --- Similarity tests ---

    #[test]
    fn test_similarity_identical_vectors() {
        let v = vec![1.0f32, 0.0, 0.0];
        let sim = BatchEncoder::similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_similarity_orthogonal_vectors() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let sim = BatchEncoder::similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_similarity_opposite_vectors() {
        let a = vec![1.0f32, 0.0];
        let b = vec![-1.0f32, 0.0];
        let sim = BatchEncoder::similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_similarity_zero_vector() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 0.0];
        let sim = BatchEncoder::similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_similarity_mismatched_len() {
        let a = vec![1.0f32, 0.0];
        let b = vec![1.0f32, 0.0, 0.5];
        let sim = BatchEncoder::similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_similarity_empty_vectors() {
        let sim = BatchEncoder::similarity(&[], &[]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_similarity_bounded() {
        let mut enc = default_encoder();
        let e1 = enc.encode_single("semantic similarity test");
        let e2 = enc.encode_single("another sentence here");
        let sim = BatchEncoder::similarity(&e1, &e2);
        assert!((-1.0..=1.0).contains(&sim));
    }

    // --- Vocab tests ---

    #[test]
    fn test_vocab_grows() {
        let mut enc = default_encoder();
        assert_eq!(enc.vocab_size(), 0);
        enc.tokenize("alpha beta gamma");
        assert_eq!(enc.vocab_size(), 3);
        enc.tokenize("alpha delta"); // "alpha" already known
        assert_eq!(enc.vocab_size(), 4);
    }

    #[test]
    fn test_encode_batch_matches_single() {
        let mut enc = default_encoder();
        let texts = ["hello world", "foo bar baz"];
        let e_single_a = enc.encode_single(texts[0]);
        let e_single_b = enc.encode_single(texts[1]);
        let batch = enc.encode_batch(&texts);
        assert_eq!(batch.embeddings[0], e_single_a);
        assert_eq!(batch.embeddings[1], e_single_b);
    }
}
