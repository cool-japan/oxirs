//! Mean-neighbourhood aggregator for GraphSAGE.

/// Compute the element-wise mean of a slice of neighbour embedding vectors.
///
/// Returns a zero-length vector when `neighbour_embeddings` is empty — the
/// caller must substitute the self-embedding (or a zero vector) in that case.
pub fn mean_aggregate(neighbour_embeddings: &[Vec<f64>]) -> Vec<f64> {
    if neighbour_embeddings.is_empty() {
        return Vec::new();
    }
    let dim = neighbour_embeddings[0].len();
    let n = neighbour_embeddings.len() as f64;
    let mut result = vec![0.0_f64; dim];
    for emb in neighbour_embeddings {
        for (i, &v) in emb.iter().enumerate() {
            result[i] += v;
        }
    }
    result.iter_mut().for_each(|x| *x /= n);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_aggregate_basic() {
        let embs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = mean_aggregate(&embs);
        assert!((result[0] - 2.0).abs() < 1e-12);
        assert!((result[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_mean_aggregate_empty() {
        let result = mean_aggregate(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_mean_aggregate_single() {
        let embs = vec![vec![5.0, 6.0, 7.0]];
        let result = mean_aggregate(&embs);
        assert!((result[0] - 5.0).abs() < 1e-12);
        assert!((result[1] - 6.0).abs() < 1e-12);
        assert!((result[2] - 7.0).abs() < 1e-12);
    }
}
