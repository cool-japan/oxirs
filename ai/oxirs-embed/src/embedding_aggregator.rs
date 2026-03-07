//! Embedding aggregation strategies for combining token-level embeddings.
//!
//! Provides multiple pooling strategies:
//! - **Mean pooling**: average of all token embeddings
//! - **Max pooling**: element-wise maximum across tokens
//! - **CLS token extraction**: first token (index 0) embedding
//! - **Attention-weighted aggregation**: weighted sum by attention scores
//! - **Hierarchical aggregation**: sentence -> paragraph -> document
//! - Configurable strategy selection, dimension-preserving output, batch support

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Available pooling strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoolingStrategy {
    /// Average of all token embeddings.
    Mean,
    /// Element-wise maximum across all token embeddings.
    Max,
    /// First token (index 0) embedding, typically CLS token in BERT-like models.
    Cls,
    /// Weighted average using attention scores.
    AttentionWeighted,
}

/// Configuration for the embedding aggregator.
#[derive(Debug, Clone)]
pub struct AggregatorConfig {
    /// The default pooling strategy when none is specified.
    pub default_strategy: PoolingStrategy,
    /// Whether to L2-normalise the result after aggregation.
    pub normalize_output: bool,
    /// Epsilon added to norms to prevent division by zero.
    pub eps: f32,
}

impl Default for AggregatorConfig {
    fn default() -> Self {
        Self {
            default_strategy: PoolingStrategy::Mean,
            normalize_output: false,
            eps: 1e-12,
        }
    }
}

/// Result of a single aggregation operation.
#[derive(Debug, Clone)]
pub struct AggregatedEmbedding {
    /// The aggregated vector.
    pub vector: Vec<f32>,
    /// Which strategy was used.
    pub strategy: PoolingStrategy,
    /// How many token embeddings were consumed.
    pub token_count: usize,
}

/// Result of hierarchical aggregation (sentence -> paragraph -> document).
#[derive(Debug, Clone)]
pub struct HierarchicalResult {
    /// Per-sentence aggregated embeddings.
    pub sentence_embeddings: Vec<Vec<f32>>,
    /// Per-paragraph aggregated embeddings (each paragraph = group of sentences).
    pub paragraph_embeddings: Vec<Vec<f32>>,
    /// Document-level embedding (single vector).
    pub document_embedding: Vec<f32>,
}

/// Batch aggregation result.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// One aggregated embedding per input sequence.
    pub embeddings: Vec<AggregatedEmbedding>,
    /// Total number of sequences processed.
    pub sequence_count: usize,
}

// ---------------------------------------------------------------------------
// EmbeddingAggregator
// ---------------------------------------------------------------------------

/// Stateful embedding aggregator that tracks total aggregation operations.
pub struct EmbeddingAggregator {
    config: AggregatorConfig,
    total_aggregations: u64,
}

impl EmbeddingAggregator {
    /// Create a new aggregator with the given configuration.
    pub fn new(config: AggregatorConfig) -> Self {
        Self {
            config,
            total_aggregations: 0,
        }
    }

    /// Aggregate a sequence of token embeddings using the default strategy.
    ///
    /// Each inner `Vec<f32>` is a single token's embedding.
    /// All token embeddings must have the same dimensionality.
    pub fn aggregate(&mut self, tokens: &[Vec<f32>]) -> Option<AggregatedEmbedding> {
        self.aggregate_with(tokens, self.config.default_strategy, None)
    }

    /// Aggregate using a specific strategy.
    ///
    /// `attention_weights` is required when `strategy == AttentionWeighted` and
    /// is ignored for other strategies.
    pub fn aggregate_with(
        &mut self,
        tokens: &[Vec<f32>],
        strategy: PoolingStrategy,
        attention_weights: Option<&[f32]>,
    ) -> Option<AggregatedEmbedding> {
        if tokens.is_empty() {
            return None;
        }
        let dim = tokens[0].len();
        if dim == 0 {
            return None;
        }

        let raw = match strategy {
            PoolingStrategy::Mean => mean_pool(tokens, dim),
            PoolingStrategy::Max => max_pool(tokens, dim),
            PoolingStrategy::Cls => cls_pool(tokens),
            PoolingStrategy::AttentionWeighted => {
                attention_pool(tokens, attention_weights, dim, self.config.eps)
            }
        };

        let vector = if self.config.normalize_output {
            l2_normalize(&raw, self.config.eps)
        } else {
            raw
        };

        self.total_aggregations += 1;

        Some(AggregatedEmbedding {
            vector,
            strategy,
            token_count: tokens.len(),
        })
    }

    /// Aggregate a batch of token sequences using the default strategy.
    pub fn aggregate_batch(&mut self, batch: &[Vec<Vec<f32>>]) -> BatchResult {
        self.aggregate_batch_with(batch, self.config.default_strategy)
    }

    /// Aggregate a batch with a specific strategy.
    pub fn aggregate_batch_with(
        &mut self,
        batch: &[Vec<Vec<f32>>],
        strategy: PoolingStrategy,
    ) -> BatchResult {
        let embeddings: Vec<AggregatedEmbedding> = batch
            .iter()
            .filter_map(|tokens| self.aggregate_with(tokens, strategy, None))
            .collect();
        let sequence_count = embeddings.len();
        BatchResult {
            embeddings,
            sequence_count,
        }
    }

    /// Perform hierarchical aggregation: sentence -> paragraph -> document.
    ///
    /// * `sentences` – each entry is a sequence of token embeddings forming one sentence.
    /// * `paragraph_boundaries` – indices into `sentences` where a new paragraph starts
    ///   (e.g. `[0, 3, 7]` means sentences 0..3 form paragraph 0, 3..7 form paragraph 1, etc.).
    ///
    /// Uses mean pooling at every level.
    pub fn hierarchical_aggregate(
        &mut self,
        sentences: &[Vec<Vec<f32>>],
        paragraph_boundaries: &[usize],
    ) -> Option<HierarchicalResult> {
        if sentences.is_empty() {
            return None;
        }

        // 1. Sentence-level: aggregate each sentence's token embeddings.
        let sentence_embeddings: Vec<Vec<f32>> = sentences
            .iter()
            .filter_map(|tokens| {
                self.aggregate_with(tokens, PoolingStrategy::Mean, None)
                    .map(|agg| agg.vector)
            })
            .collect();

        if sentence_embeddings.is_empty() {
            return None;
        }

        // 2. Paragraph-level: group sentence embeddings by boundaries.
        let paragraph_embeddings =
            aggregate_by_boundaries(&sentence_embeddings, paragraph_boundaries, self.config.eps);

        // 3. Document-level: average of paragraph embeddings.
        let dim = paragraph_embeddings.first().map(|v| v.len()).unwrap_or(0);
        let document_embedding = if paragraph_embeddings.is_empty() || dim == 0 {
            vec![]
        } else {
            mean_pool_refs(&paragraph_embeddings, dim)
        };

        Some(HierarchicalResult {
            sentence_embeddings,
            paragraph_embeddings,
            document_embedding,
        })
    }

    /// Compare two pooling strategies on the same tokens and return both results.
    pub fn compare_strategies(
        &mut self,
        tokens: &[Vec<f32>],
        strategy_a: PoolingStrategy,
        strategy_b: PoolingStrategy,
    ) -> (Option<AggregatedEmbedding>, Option<AggregatedEmbedding>) {
        let a = self.aggregate_with(tokens, strategy_a, None);
        let b = self.aggregate_with(tokens, strategy_b, None);
        (a, b)
    }

    /// Return the total number of individual aggregation operations performed.
    pub fn total_aggregations(&self) -> u64 {
        self.total_aggregations
    }

    /// Return the current configuration.
    pub fn config(&self) -> &AggregatorConfig {
        &self.config
    }

    /// Build a summary of aggregation results per strategy from provided labels.
    pub fn strategy_summary(results: &[AggregatedEmbedding]) -> HashMap<PoolingStrategy, usize> {
        let mut counts: HashMap<PoolingStrategy, usize> = HashMap::new();
        for r in results {
            *counts.entry(r.strategy).or_insert(0) += 1;
        }
        counts
    }
}

// ---------------------------------------------------------------------------
// Free functions – pooling implementations
// ---------------------------------------------------------------------------

/// Mean pooling: element-wise average of all token embeddings.
fn mean_pool(tokens: &[Vec<f32>], dim: usize) -> Vec<f32> {
    let n = tokens.len() as f32;
    let mut result = vec![0.0f32; dim];
    for tok in tokens {
        for (i, &v) in tok.iter().enumerate().take(dim) {
            result[i] += v;
        }
    }
    for v in &mut result {
        *v /= n;
    }
    result
}

/// Mean pool from a slice of references (used in hierarchical aggregation).
fn mean_pool_refs(vectors: &[Vec<f32>], dim: usize) -> Vec<f32> {
    let n = vectors.len() as f32;
    let mut result = vec![0.0f32; dim];
    for vec in vectors {
        for (i, &v) in vec.iter().enumerate().take(dim) {
            result[i] += v;
        }
    }
    for v in &mut result {
        *v /= n;
    }
    result
}

/// Max pooling: element-wise maximum across all token embeddings.
fn max_pool(tokens: &[Vec<f32>], dim: usize) -> Vec<f32> {
    let mut result = vec![f32::NEG_INFINITY; dim];
    for tok in tokens {
        for (i, &v) in tok.iter().enumerate().take(dim) {
            if v > result[i] {
                result[i] = v;
            }
        }
    }
    result
}

/// CLS token extraction: return a clone of the first token's embedding.
fn cls_pool(tokens: &[Vec<f32>]) -> Vec<f32> {
    tokens.first().cloned().unwrap_or_default()
}

/// Attention-weighted pooling: weighted average using attention scores.
///
/// If `weights` is `None` or mismatched in length, falls back to uniform weights.
fn attention_pool(tokens: &[Vec<f32>], weights: Option<&[f32]>, dim: usize, eps: f32) -> Vec<f32> {
    let n = tokens.len();
    let effective_weights: Vec<f32> = match weights {
        Some(w) if w.len() == n => {
            // Softmax-style normalisation (just normalise to sum=1).
            let sum: f32 = w.iter().sum();
            if sum.abs() < eps {
                vec![1.0 / n as f32; n]
            } else {
                w.iter().map(|&v| v / sum).collect()
            }
        }
        _ => vec![1.0 / n as f32; n],
    };

    let mut result = vec![0.0f32; dim];
    for (tok, &weight) in tokens.iter().zip(effective_weights.iter()) {
        for (i, &v) in tok.iter().enumerate().take(dim) {
            result[i] += v * weight;
        }
    }
    result
}

/// L2-normalise a vector in-place.
fn l2_normalize(vec: &[f32], eps: f32) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|&v| v * v).sum::<f32>().sqrt();
    if norm < eps {
        return vec.to_vec();
    }
    vec.iter().map(|&v| v / norm).collect()
}

/// Cosine similarity between two f32 slices.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let dot: f32 = a[..len]
        .iter()
        .zip(b[..len].iter())
        .map(|(x, y)| x * y)
        .sum();
    let na: f32 = a[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// Group vectors by paragraph boundaries and mean-pool each group.
fn aggregate_by_boundaries(vectors: &[Vec<f32>], boundaries: &[usize], _eps: f32) -> Vec<Vec<f32>> {
    if vectors.is_empty() {
        return vec![];
    }
    let dim = vectors[0].len();

    // Determine segment ranges.
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for (i, &start) in boundaries.iter().enumerate() {
        let end = if i + 1 < boundaries.len() {
            boundaries[i + 1]
        } else {
            vectors.len()
        };
        if start < end && start < vectors.len() {
            ranges.push((start, end.min(vectors.len())));
        }
    }

    // If no valid boundaries, treat entire input as one paragraph.
    if ranges.is_empty() {
        ranges.push((0, vectors.len()));
    }

    ranges
        .iter()
        .map(|&(start, end)| mean_pool_refs(&vectors[start..end], dim))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_aggregator() -> EmbeddingAggregator {
        EmbeddingAggregator::new(AggregatorConfig::default())
    }

    fn normalizing_aggregator() -> EmbeddingAggregator {
        EmbeddingAggregator::new(AggregatorConfig {
            normalize_output: true,
            ..AggregatorConfig::default()
        })
    }

    /// Three token embeddings each of dimension 4.
    fn sample_tokens() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ]
    }

    // --- mean pooling ---

    #[test]
    fn test_mean_pool_correct_values() {
        let mut agg = default_aggregator();
        let result = agg
            .aggregate_with(&sample_tokens(), PoolingStrategy::Mean, None)
            .expect("should succeed");
        // mean = (1+5+9)/3, (2+6+10)/3, (3+7+11)/3, (4+8+12)/3 = 5, 6, 7, 8
        assert!((result.vector[0] - 5.0).abs() < 1e-5);
        assert!((result.vector[1] - 6.0).abs() < 1e-5);
        assert!((result.vector[2] - 7.0).abs() < 1e-5);
        assert!((result.vector[3] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean_pool_dimension_preserved() {
        let mut agg = default_aggregator();
        let result = agg
            .aggregate_with(&sample_tokens(), PoolingStrategy::Mean, None)
            .expect("should succeed");
        assert_eq!(result.vector.len(), 4);
    }

    #[test]
    fn test_mean_pool_single_token() {
        let mut agg = default_aggregator();
        let tokens = vec![vec![1.0, 2.0, 3.0]];
        let result = agg
            .aggregate_with(&tokens, PoolingStrategy::Mean, None)
            .expect("should succeed");
        assert!((result.vector[0] - 1.0).abs() < 1e-6);
        assert!((result.vector[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_pool_token_count() {
        let mut agg = default_aggregator();
        let result = agg
            .aggregate_with(&sample_tokens(), PoolingStrategy::Mean, None)
            .expect("should succeed");
        assert_eq!(result.token_count, 3);
    }

    // --- max pooling ---

    #[test]
    fn test_max_pool_correct_values() {
        let mut agg = default_aggregator();
        let result = agg
            .aggregate_with(&sample_tokens(), PoolingStrategy::Max, None)
            .expect("should succeed");
        // max = 9, 10, 11, 12
        assert!((result.vector[0] - 9.0).abs() < 1e-5);
        assert!((result.vector[1] - 10.0).abs() < 1e-5);
        assert!((result.vector[2] - 11.0).abs() < 1e-5);
        assert!((result.vector[3] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_max_pool_with_negatives() {
        let mut agg = default_aggregator();
        let tokens = vec![vec![-1.0, -5.0], vec![-3.0, -2.0]];
        let result = agg
            .aggregate_with(&tokens, PoolingStrategy::Max, None)
            .expect("should succeed");
        assert!((result.vector[0] - (-1.0)).abs() < 1e-6);
        assert!((result.vector[1] - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_max_pool_single_token() {
        let mut agg = default_aggregator();
        let tokens = vec![vec![7.0, 8.0, 9.0]];
        let result = agg
            .aggregate_with(&tokens, PoolingStrategy::Max, None)
            .expect("should succeed");
        assert!((result.vector[0] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_pool_dimension_preserved() {
        let mut agg = default_aggregator();
        let result = agg
            .aggregate_with(&sample_tokens(), PoolingStrategy::Max, None)
            .expect("should succeed");
        assert_eq!(result.vector.len(), 4);
    }

    // --- CLS pooling ---

    #[test]
    fn test_cls_pool_returns_first_token() {
        let mut agg = default_aggregator();
        let result = agg
            .aggregate_with(&sample_tokens(), PoolingStrategy::Cls, None)
            .expect("should succeed");
        assert_eq!(result.vector, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_cls_pool_ignores_subsequent_tokens() {
        let mut agg = default_aggregator();
        let tokens = vec![vec![100.0, 200.0], vec![999.0, 888.0]];
        let result = agg
            .aggregate_with(&tokens, PoolingStrategy::Cls, None)
            .expect("should succeed");
        assert!((result.vector[0] - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_cls_pool_token_count() {
        let mut agg = default_aggregator();
        let tokens = vec![vec![1.0], vec![2.0], vec![3.0]];
        let result = agg
            .aggregate_with(&tokens, PoolingStrategy::Cls, None)
            .expect("should succeed");
        assert_eq!(result.token_count, 3);
    }

    // --- attention-weighted pooling ---

    #[test]
    fn test_attention_pool_uniform_weights_equals_mean() {
        let mut agg = default_aggregator();
        let tokens = sample_tokens();
        let weights = vec![1.0, 1.0, 1.0];
        let attn = agg
            .aggregate_with(&tokens, PoolingStrategy::AttentionWeighted, Some(&weights))
            .expect("should succeed");
        let mean = agg
            .aggregate_with(&tokens, PoolingStrategy::Mean, None)
            .expect("should succeed");
        for (a, m) in attn.vector.iter().zip(mean.vector.iter()) {
            assert!((a - m).abs() < 1e-5, "uniform attn should equal mean");
        }
    }

    #[test]
    fn test_attention_pool_single_nonzero_weight() {
        let mut agg = default_aggregator();
        let tokens = sample_tokens();
        // Only the last token gets weight
        let weights = vec![0.0, 0.0, 1.0];
        let result = agg
            .aggregate_with(&tokens, PoolingStrategy::AttentionWeighted, Some(&weights))
            .expect("should succeed");
        assert!((result.vector[0] - 9.0).abs() < 1e-5);
        assert!((result.vector[1] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_attention_pool_mismatched_weights_falls_back_to_uniform() {
        let mut agg = default_aggregator();
        let tokens = sample_tokens();
        let weights = vec![1.0, 2.0]; // length 2 != 3 tokens
        let result = agg
            .aggregate_with(&tokens, PoolingStrategy::AttentionWeighted, Some(&weights))
            .expect("should succeed");
        // Falls back to uniform = mean pooling
        assert!((result.vector[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_attention_pool_no_weights_falls_back_to_uniform() {
        let mut agg = default_aggregator();
        let tokens = sample_tokens();
        let result = agg
            .aggregate_with(&tokens, PoolingStrategy::AttentionWeighted, None)
            .expect("should succeed");
        assert!((result.vector[0] - 5.0).abs() < 1e-5);
    }

    // --- normalization ---

    #[test]
    fn test_normalized_output_has_unit_norm() {
        let mut agg = normalizing_aggregator();
        let result = agg
            .aggregate_with(&sample_tokens(), PoolingStrategy::Mean, None)
            .expect("should succeed");
        let norm: f32 = result.vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "normalized output should have unit norm"
        );
    }

    #[test]
    fn test_non_normalized_output_not_unit_norm() {
        let mut agg = default_aggregator();
        let result = agg
            .aggregate_with(&sample_tokens(), PoolingStrategy::Mean, None)
            .expect("should succeed");
        let norm: f32 = result.vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        // Mean of (5,6,7,8) has norm > 1
        assert!(norm > 1.0);
    }

    // --- empty / edge cases ---

    #[test]
    fn test_empty_tokens_returns_none() {
        let mut agg = default_aggregator();
        let result = agg.aggregate_with(&[], PoolingStrategy::Mean, None);
        assert!(result.is_none());
    }

    #[test]
    fn test_zero_dim_tokens_returns_none() {
        let mut agg = default_aggregator();
        let tokens: Vec<Vec<f32>> = vec![vec![], vec![]];
        let result = agg.aggregate_with(&tokens, PoolingStrategy::Mean, None);
        assert!(result.is_none());
    }

    // --- default aggregate ---

    #[test]
    fn test_aggregate_uses_default_strategy() {
        let mut agg = EmbeddingAggregator::new(AggregatorConfig {
            default_strategy: PoolingStrategy::Max,
            ..AggregatorConfig::default()
        });
        let result = agg.aggregate(&sample_tokens()).expect("should succeed");
        assert_eq!(result.strategy, PoolingStrategy::Max);
    }

    // --- batch aggregation ---

    #[test]
    fn test_batch_aggregate_count() {
        let mut agg = default_aggregator();
        let batch = vec![sample_tokens(), sample_tokens(), sample_tokens()];
        let result = agg.aggregate_batch(&batch);
        assert_eq!(result.sequence_count, 3);
        assert_eq!(result.embeddings.len(), 3);
    }

    #[test]
    fn test_batch_aggregate_with_empty_sequences() {
        let mut agg = default_aggregator();
        let batch: Vec<Vec<Vec<f32>>> = vec![sample_tokens(), vec![], sample_tokens()];
        let result = agg.aggregate_batch(&batch);
        assert_eq!(
            result.sequence_count, 2,
            "empty sequence should be filtered out"
        );
    }

    #[test]
    fn test_batch_aggregate_strategy_propagates() {
        let mut agg = default_aggregator();
        let batch = vec![sample_tokens()];
        let result = agg.aggregate_batch_with(&batch, PoolingStrategy::Cls);
        assert_eq!(result.embeddings[0].strategy, PoolingStrategy::Cls);
    }

    // --- hierarchical aggregation ---

    #[test]
    fn test_hierarchical_single_sentence() {
        let mut agg = default_aggregator();
        let sentences = vec![sample_tokens()];
        let result = agg
            .hierarchical_aggregate(&sentences, &[0])
            .expect("should succeed");
        assert_eq!(result.sentence_embeddings.len(), 1);
        assert_eq!(result.paragraph_embeddings.len(), 1);
        assert_eq!(result.document_embedding.len(), 4);
    }

    #[test]
    fn test_hierarchical_two_paragraphs() {
        let mut agg = default_aggregator();
        let sentences = vec![
            vec![vec![1.0, 0.0], vec![3.0, 0.0]],  // sentence 0
            vec![vec![5.0, 0.0], vec![7.0, 0.0]],  // sentence 1
            vec![vec![9.0, 0.0], vec![11.0, 0.0]], // sentence 2
        ];
        let boundaries = vec![0, 2]; // paragraph 0 = sentences [0,1], paragraph 1 = sentence [2]
        let result = agg
            .hierarchical_aggregate(&sentences, &boundaries)
            .expect("should succeed");
        assert_eq!(result.paragraph_embeddings.len(), 2);
    }

    #[test]
    fn test_hierarchical_empty_returns_none() {
        let mut agg = default_aggregator();
        let result = agg.hierarchical_aggregate(&[], &[0]);
        assert!(result.is_none());
    }

    #[test]
    fn test_hierarchical_document_is_mean_of_paragraphs() {
        let mut agg = default_aggregator();
        // Two sentences, each with two tokens of dim 2
        let sentences = vec![
            vec![vec![2.0, 4.0], vec![4.0, 6.0]], // sentence 0 → mean = (3, 5)
            vec![vec![6.0, 8.0], vec![8.0, 10.0]], // sentence 1 → mean = (7, 9)
        ];
        // One paragraph encompassing both
        let result = agg
            .hierarchical_aggregate(&sentences, &[0])
            .expect("should succeed");
        // Paragraph mean = (3+7)/2, (5+9)/2 = (5, 7) = document
        assert!((result.document_embedding[0] - 5.0).abs() < 1e-5);
        assert!((result.document_embedding[1] - 7.0).abs() < 1e-5);
    }

    // --- compare strategies ---

    #[test]
    fn test_compare_strategies_returns_both() {
        let mut agg = default_aggregator();
        let (a, b) = agg.compare_strategies(
            &sample_tokens(),
            PoolingStrategy::Mean,
            PoolingStrategy::Max,
        );
        assert!(a.is_some());
        assert!(b.is_some());
        assert_eq!(a.as_ref().map(|r| r.strategy), Some(PoolingStrategy::Mean));
        assert_eq!(b.as_ref().map(|r| r.strategy), Some(PoolingStrategy::Max));
    }

    #[test]
    fn test_compare_strategies_different_results() {
        let mut agg = default_aggregator();
        let (a, b) = agg.compare_strategies(
            &sample_tokens(),
            PoolingStrategy::Mean,
            PoolingStrategy::Max,
        );
        // Mean[0]=5, Max[0]=9
        assert!((a.as_ref().map(|r| r.vector[0]).unwrap_or(0.0) - 5.0).abs() < 1e-5);
        assert!((b.as_ref().map(|r| r.vector[0]).unwrap_or(0.0) - 9.0).abs() < 1e-5);
    }

    // --- total aggregations tracking ---

    #[test]
    fn test_total_aggregations_initially_zero() {
        let agg = default_aggregator();
        assert_eq!(agg.total_aggregations(), 0);
    }

    #[test]
    fn test_total_aggregations_increments() {
        let mut agg = default_aggregator();
        agg.aggregate(&sample_tokens());
        agg.aggregate(&sample_tokens());
        assert_eq!(agg.total_aggregations(), 2);
    }

    #[test]
    fn test_total_aggregations_batch_increments() {
        let mut agg = default_aggregator();
        let batch = vec![sample_tokens(), sample_tokens()];
        agg.aggregate_batch(&batch);
        assert_eq!(agg.total_aggregations(), 2);
    }

    // --- strategy summary ---

    #[test]
    fn test_strategy_summary_counts() {
        let results = vec![
            AggregatedEmbedding {
                vector: vec![1.0],
                strategy: PoolingStrategy::Mean,
                token_count: 1,
            },
            AggregatedEmbedding {
                vector: vec![2.0],
                strategy: PoolingStrategy::Mean,
                token_count: 1,
            },
            AggregatedEmbedding {
                vector: vec![3.0],
                strategy: PoolingStrategy::Max,
                token_count: 1,
            },
        ];
        let summary = EmbeddingAggregator::strategy_summary(&results);
        assert_eq!(summary.get(&PoolingStrategy::Mean), Some(&2));
        assert_eq!(summary.get(&PoolingStrategy::Max), Some(&1));
        assert_eq!(summary.get(&PoolingStrategy::Cls), None);
    }

    // --- cosine similarity ---

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let sim = cosine_similarity(&[], &[]);
        assert_eq!(sim, 0.0);
    }

    // --- config access ---

    #[test]
    fn test_config_accessor() {
        let agg = default_aggregator();
        assert_eq!(agg.config().default_strategy, PoolingStrategy::Mean);
        assert!(!agg.config().normalize_output);
    }

    #[test]
    fn test_aggregator_config_default() {
        let config = AggregatorConfig::default();
        assert_eq!(config.default_strategy, PoolingStrategy::Mean);
        assert!(!config.normalize_output);
        assert!(config.eps > 0.0);
    }
}
