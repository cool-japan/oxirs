//! Scaled dot-product attention for GNN-based entity encoding.
//!
//! This module provides a single-head scaled dot-product attention mechanism
//! that can be stacked on top of message-passing layers to produce
//! context-sensitive entity representations.

/// Single-head scaled dot-product attention.
///
/// Given a query vector and a set of key/value pairs, computes a weighted sum
/// of the value vectors where the weights are proportional to the exponentiated
/// scaled dot products between the query and each key.
///
/// The scaling factor is `1 / sqrt(head_dim)` as in "Attention Is All You Need".
#[derive(Debug, Clone)]
pub struct ScaledDotProductAttention {
    /// Dimensionality of query and key vectors
    pub head_dim: usize,
}

impl ScaledDotProductAttention {
    /// Create a new attention module for the given key/query dimensionality.
    pub fn new(head_dim: usize) -> Self {
        Self { head_dim }
    }

    /// Compute the attention-weighted output vector.
    ///
    /// `query`  — the query vector of length `head_dim`.
    /// `keys`   — a slice of key vectors, each of length `head_dim`.
    /// `values` — a slice of value vectors (same length as `keys`),
    ///            each of length ≥ 0 (can differ from `head_dim`).
    ///
    /// Returns:
    /// - If `keys` is empty: a zero vector of length `head_dim`.
    /// - Otherwise: the softmax-weighted sum of `values`.
    pub fn attend(&self, query: &[f64], keys: &[&[f64]], values: &[&[f64]]) -> Vec<f64> {
        assert_eq!(
            keys.len(),
            values.len(),
            "keys and values must have the same length"
        );

        if keys.is_empty() {
            return vec![0.0; self.head_dim];
        }

        let scale = 1.0 / (self.head_dim as f64).sqrt();

        // Compute raw scores: score_i = dot(query, key_i) * scale
        let scores: Vec<f64> = keys.iter().map(|key| dot(query, key) * scale).collect();

        // Softmax over scores
        let weights = softmax(&scores);

        // Weighted sum of values
        let value_dim = values[0].len();
        let mut output = vec![0.0_f64; value_dim];
        for (weight, value) in weights.iter().zip(values.iter()) {
            for (j, &v) in value.iter().enumerate() {
                output[j] += weight * v;
            }
        }

        output
    }

    /// Compute only the attention weight vector (softmax scores).
    ///
    /// Useful for inspecting where the model is attending.
    pub fn attention_weights(&self, query: &[f64], keys: &[&[f64]]) -> Vec<f64> {
        if keys.is_empty() {
            return Vec::new();
        }

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores: Vec<f64> = keys.iter().map(|key| dot(query, key) * scale).collect();

        softmax(&scores)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Dot product of two vectors (truncated to the shorter).
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Numerically stable softmax: subtract the maximum before exponentiation.
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }

    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();

    if sum < 1e-30 {
        // Degenerate case: return uniform
        let n = logits.len() as f64;
        return vec![1.0 / n; logits.len()];
    }

    exps.into_iter().map(|e| e / sum).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    #[test]
    fn test_attention_weights_sum_to_one() {
        let attn = ScaledDotProductAttention::new(4);
        let query = vec![1.0_f64, 0.0, 0.0, 0.0];
        let k1 = vec![1.0_f64, 0.0, 0.0, 0.0];
        let k2 = vec![0.0_f64, 1.0, 0.0, 0.0];
        let k3 = vec![0.0_f64, 0.0, 1.0, 0.0];
        let v1 = vec![1.0_f64];
        let v2 = vec![2.0_f64];
        let v3 = vec![3.0_f64];

        let output = attn.attend(
            &query,
            &[k1.as_slice(), k2.as_slice(), k3.as_slice()],
            &[v1.as_slice(), v2.as_slice(), v3.as_slice()],
        );

        // Verify the weights sum to 1 by computing them explicitly
        let weights =
            attn.attention_weights(&query, &[k1.as_slice(), k2.as_slice(), k3.as_slice()]);
        let weight_sum: f64 = weights.iter().sum();
        assert!(
            (weight_sum - 1.0).abs() < EPS,
            "Attention weights must sum to 1.0, got {}",
            weight_sum
        );

        // Output must be a valid weighted sum (in [min_val, max_val])
        assert!(output[0] >= 1.0 - EPS);
        assert!(output[0] <= 3.0 + EPS);
    }

    #[test]
    fn test_empty_keys_return_zeros() {
        let attn = ScaledDotProductAttention::new(8);
        let query = vec![1.0_f64; 8];
        let output = attn.attend(&query, &[], &[]);
        assert_eq!(output.len(), 8);
        assert!(
            output.iter().all(|&x| x == 0.0),
            "Empty keys must produce zero output"
        );
    }

    #[test]
    fn test_single_key_is_identity() {
        let attn = ScaledDotProductAttention::new(4);
        let query = vec![0.5_f64, 0.5, 0.5, 0.5];
        let key = vec![1.0_f64, 0.0, 0.0, 0.0];
        let value = vec![7.0_f64, 8.0, 9.0];

        let output = attn.attend(&query, &[key.as_slice()], &[value.as_slice()]);

        // With a single key the attention weight is 1.0, so output == value
        assert_eq!(output.len(), 3);
        assert!((output[0] - 7.0).abs() < EPS);
        assert!((output[1] - 8.0).abs() < EPS);
        assert!((output[2] - 9.0).abs() < EPS);
    }
}
