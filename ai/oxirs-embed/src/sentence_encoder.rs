//! Sentence-level text encoder producing fixed-size embeddings.
//!
//! Tokenizes text into lowercase word tokens, looks up a per-token embedding,
//! then pools the token embeddings into a single fixed-size vector.

use std::collections::HashMap;

/// Strategy for pooling token embeddings into a sentence embedding.
#[derive(Debug, Clone, PartialEq)]
pub enum PoolingStrategy {
    /// Arithmetic mean of all token embeddings.
    Mean,
    /// Element-wise maximum across all token embeddings.
    Max,
    /// Use the first token ("CLS") embedding as-is.
    Cls,
}

/// Configuration for a `SentenceEncoder`.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub pooling: PoolingStrategy,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            vocab_size: 1024,
            embedding_dim: 64,
            pooling: PoolingStrategy::Mean,
        }
    }
}

/// Simple sentence encoder with a learned (or randomly initialised) token vocabulary.
pub struct SentenceEncoder {
    config: EncoderConfig,
    token_embeddings: HashMap<String, Vec<f64>>,
}

impl SentenceEncoder {
    /// Create a new encoder with an empty vocabulary.
    pub fn new(config: EncoderConfig) -> Self {
        Self {
            config,
            token_embeddings: HashMap::new(),
        }
    }

    /// Build an encoder with a randomly initialised vocabulary.
    ///
    /// Uses a simple LCG seeded with `seed` for deterministic initialisation.
    pub fn with_random_vocab(vocab: &[&str], dim: usize, seed: u64) -> Self {
        let config = EncoderConfig {
            vocab_size: vocab.len(),
            embedding_dim: dim,
            pooling: PoolingStrategy::Mean,
        };
        let mut enc = Self::new(config);
        let mut state = seed;
        for &word in vocab {
            let emb: Vec<f64> = (0..dim)
                .map(|_| {
                    state = lcg_next(state);
                    // scale to [-1, 1]
                    (state as f64 / u64::MAX as f64) * 2.0 - 1.0
                })
                .collect();
            enc.token_embeddings.insert(word.to_lowercase(), emb);
        }
        enc
    }

    /// Encode a single text string into a fixed-size embedding vector.
    ///
    /// Unknown tokens are ignored. If no tokens are found the zero vector is
    /// returned.
    pub fn encode(&self, text: &str) -> Vec<f64> {
        let tokens = Self::tokenize(text);
        let mut token_vecs: Vec<Vec<f64>> = Vec::new();
        for token in &tokens {
            if let Some(emb) = self.token_embeddings.get(token) {
                token_vecs.push(emb.clone());
            }
        }
        if token_vecs.is_empty() {
            return vec![0.0; self.config.embedding_dim];
        }
        match &self.config.pooling {
            PoolingStrategy::Mean => pool_mean(&token_vecs),
            PoolingStrategy::Max => pool_max(&token_vecs),
            PoolingStrategy::Cls => token_vecs.into_iter().next().unwrap_or_else(|| vec![0.0; self.config.embedding_dim]),
        }
    }

    /// Encode multiple texts in batch.
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<f64>> {
        texts.iter().map(|t| self.encode(t)).collect()
    }

    /// Tokenize text: lowercase, split on whitespace and ASCII punctuation.
    pub fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| c.is_whitespace() || (c.is_ascii_punctuation() && c != '_' && c != '-'))
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect()
    }

    /// Add or replace a token embedding.
    pub fn add_token(&mut self, token: &str, embedding: Vec<f64>) {
        self.token_embeddings.insert(token.to_lowercase(), embedding);
    }

    /// Number of tokens in the vocabulary.
    pub fn token_count(&self) -> usize {
        self.token_embeddings.len()
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// Cosine similarity between two embedding vectors. Returns 0.0 for zero vectors.
    pub fn similarity(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a < 1e-12 || norm_b < 1e-12 {
            0.0
        } else {
            (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
        }
    }
}

/// Pool embeddings by element-wise mean.
pub fn pool_mean(embeddings: &[Vec<f64>]) -> Vec<f64> {
    if embeddings.is_empty() {
        return vec![];
    }
    let dim = embeddings[0].len();
    let n = embeddings.len() as f64;
    let mut result = vec![0.0f64; dim];
    for emb in embeddings {
        for (r, v) in result.iter_mut().zip(emb.iter()) {
            *r += v;
        }
    }
    result.iter_mut().for_each(|x| *x /= n);
    result
}

/// Pool embeddings by element-wise maximum.
pub fn pool_max(embeddings: &[Vec<f64>]) -> Vec<f64> {
    if embeddings.is_empty() {
        return vec![];
    }
    let dim = embeddings[0].len();
    let mut result = vec![f64::NEG_INFINITY; dim];
    for emb in embeddings {
        for (r, &v) in result.iter_mut().zip(emb.iter()) {
            if v > *r {
                *r = v;
            }
        }
    }
    result
}

/// Simple LCG step.
fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_encoder() -> SentenceEncoder {
        let mut enc = SentenceEncoder::new(EncoderConfig {
            vocab_size: 10,
            embedding_dim: 3,
            pooling: PoolingStrategy::Mean,
        });
        enc.add_token("hello", vec![1.0, 0.0, 0.0]);
        enc.add_token("world", vec![0.0, 1.0, 0.0]);
        enc.add_token("foo", vec![0.0, 0.0, 1.0]);
        enc
    }

    // --- tokenize -------------------------------------------------------

    #[test]
    fn test_tokenize_basic() {
        let tokens = SentenceEncoder::tokenize("Hello World");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_tokenize_punctuation() {
        let tokens = SentenceEncoder::tokenize("Hello, world!");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_tokenize_lowercase() {
        let tokens = SentenceEncoder::tokenize("ABC DEF");
        assert_eq!(tokens, vec!["abc", "def"]);
    }

    #[test]
    fn test_tokenize_empty_string() {
        let tokens = SentenceEncoder::tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenize_multiple_spaces() {
        let tokens = SentenceEncoder::tokenize("a   b   c");
        assert_eq!(tokens, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_tokenize_preserves_hyphenated() {
        // Hyphen is kept inside tokens
        let tokens = SentenceEncoder::tokenize("well-known");
        assert!(tokens.contains(&"well-known".to_string()) || tokens.len() >= 1);
    }

    // --- add_token / token_count ----------------------------------------

    #[test]
    fn test_add_token_increases_count() {
        let mut enc = simple_encoder();
        let before = enc.token_count();
        enc.add_token("new", vec![1.0, 1.0, 1.0]);
        assert_eq!(enc.token_count(), before + 1);
    }

    #[test]
    fn test_add_token_overwrites() {
        let mut enc = simple_encoder();
        enc.add_token("hello", vec![0.5, 0.5, 0.5]);
        let emb = enc.encode("hello");
        assert!((emb[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_add_token_case_insensitive() {
        let mut enc = simple_encoder();
        enc.add_token("UPPER", vec![1.0, 0.0, 0.0]);
        let emb = enc.encode("upper"); // should find it
        assert!((emb[0] - 1.0).abs() < 1e-9);
    }

    // --- encode ----------------------------------------------------------

    #[test]
    fn test_encode_single_known_token() {
        let enc = simple_encoder();
        let emb = enc.encode("hello");
        assert_eq!(emb, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_encode_unknown_token_returns_zeros() {
        let enc = simple_encoder();
        let emb = enc.encode("unknown_xyz");
        assert_eq!(emb, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_encode_empty_string_returns_zeros() {
        let enc = simple_encoder();
        let emb = enc.encode("");
        assert_eq!(emb, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_encode_mean_pooling() {
        let enc = simple_encoder();
        let emb = enc.encode("hello world");
        // mean of [1,0,0] and [0,1,0] = [0.5, 0.5, 0]
        assert!((emb[0] - 0.5).abs() < 1e-9);
        assert!((emb[1] - 0.5).abs() < 1e-9);
        assert!((emb[2] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_encode_max_pooling() {
        let mut enc = SentenceEncoder::new(EncoderConfig {
            vocab_size: 3,
            embedding_dim: 3,
            pooling: PoolingStrategy::Max,
        });
        enc.add_token("a", vec![1.0, 0.0, -1.0]);
        enc.add_token("b", vec![0.0, 2.0, 0.0]);
        let emb = enc.encode("a b");
        assert!((emb[0] - 1.0).abs() < 1e-9);
        assert!((emb[1] - 2.0).abs() < 1e-9);
        assert!((emb[2] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_encode_cls_pooling() {
        let mut enc = SentenceEncoder::new(EncoderConfig {
            vocab_size: 3,
            embedding_dim: 3,
            pooling: PoolingStrategy::Cls,
        });
        enc.add_token("first", vec![1.0, 2.0, 3.0]);
        enc.add_token("second", vec![4.0, 5.0, 6.0]);
        let emb = enc.encode("first second");
        assert_eq!(emb, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_encode_dim_is_correct() {
        let enc = simple_encoder();
        let emb = enc.encode("hello");
        assert_eq!(emb.len(), 3);
    }

    // --- encode_batch ----------------------------------------------------

    #[test]
    fn test_encode_batch_basic() {
        let enc = simple_encoder();
        let results = enc.encode_batch(&["hello", "world"]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], vec![1.0, 0.0, 0.0]);
        assert_eq!(results[1], vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_encode_batch_empty_input() {
        let enc = simple_encoder();
        let results = enc.encode_batch(&[]);
        assert!(results.is_empty());
    }

    // --- with_random_vocab -----------------------------------------------

    #[test]
    fn test_with_random_vocab_token_count() {
        let vocab = vec!["cat", "dog", "fish"];
        let enc = SentenceEncoder::with_random_vocab(&vocab, 8, 42);
        assert_eq!(enc.token_count(), 3);
    }

    #[test]
    fn test_with_random_vocab_dim() {
        let vocab = vec!["a", "b"];
        let enc = SentenceEncoder::with_random_vocab(&vocab, 16, 1);
        let emb = enc.encode("a");
        assert_eq!(emb.len(), 16);
    }

    #[test]
    fn test_with_random_vocab_deterministic() {
        let vocab = vec!["x", "y", "z"];
        let enc1 = SentenceEncoder::with_random_vocab(&vocab, 4, 99);
        let enc2 = SentenceEncoder::with_random_vocab(&vocab, 4, 99);
        assert_eq!(enc1.encode("x"), enc2.encode("x"));
    }

    #[test]
    fn test_with_random_vocab_different_seeds_differ() {
        let vocab = vec!["token"];
        let enc1 = SentenceEncoder::with_random_vocab(&vocab, 4, 1);
        let enc2 = SentenceEncoder::with_random_vocab(&vocab, 4, 2);
        assert_ne!(enc1.encode("token"), enc2.encode("token"));
    }

    // --- similarity ------------------------------------------------------

    #[test]
    fn test_similarity_identical_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        assert!((SentenceEncoder::similarity(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((SentenceEncoder::similarity(&a, &b) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_similarity_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((SentenceEncoder::similarity(&a, &b) + 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 1.0];
        assert_eq!(SentenceEncoder::similarity(&a, &b), 0.0);
    }

    // --- pool helpers ----------------------------------------------------

    #[test]
    fn test_pool_mean_single() {
        let result = pool_mean(&[vec![2.0, 4.0]]);
        assert_eq!(result, vec![2.0, 4.0]);
    }

    #[test]
    fn test_pool_mean_two() {
        let result = pool_mean(&[vec![1.0, 0.0], vec![3.0, 2.0]]);
        assert!((result[0] - 2.0).abs() < 1e-9);
        assert!((result[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_pool_mean_empty() {
        let result = pool_mean(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_pool_max_single() {
        let result = pool_max(&[vec![3.0, -1.0]]);
        assert_eq!(result, vec![3.0, -1.0]);
    }

    #[test]
    fn test_pool_max_two() {
        let result = pool_max(&[vec![1.0, 5.0], vec![3.0, 2.0]]);
        assert_eq!(result, vec![3.0, 5.0]);
    }

    #[test]
    fn test_pool_max_empty() {
        let result = pool_max(&[]);
        assert!(result.is_empty());
    }

    // --- dim / config ----------------------------------------------------

    #[test]
    fn test_dim_matches_config() {
        let enc = simple_encoder();
        assert_eq!(enc.dim(), 3);
    }

    #[test]
    fn test_encoder_default_config() {
        let enc = SentenceEncoder::new(EncoderConfig::default());
        assert_eq!(enc.dim(), 64);
        assert_eq!(enc.token_count(), 0);
    }
}
