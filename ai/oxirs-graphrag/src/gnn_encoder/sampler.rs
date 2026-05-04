//! Neighbourhood sampler for GraphSAGE.
//!
//! Performs reservoir-sampling to select up to `k` neighbours for a given
//! node from a flat edge list.

use scirs2_core::random::{rand_prelude::StdRng, CoreRandom};

/// Sample up to `k` neighbours of `node_id` from `edges`.
///
/// Edges are directed; any edge `(node_id, dst)` contributes `dst` as a
/// neighbour candidate.  When the candidate set has ≤ k elements all are
/// returned; otherwise `k` are selected by reservoir sampling (Algorithm R).
pub fn sample_neighbours(
    node_id: usize,
    edges: &[(usize, usize)],
    k: usize,
    rng: &mut CoreRandom<StdRng>,
) -> Vec<usize> {
    // Collect all outgoing neighbours.
    let candidates: Vec<usize> = edges
        .iter()
        .filter_map(|&(src, dst)| if src == node_id { Some(dst) } else { None })
        .collect();

    if candidates.len() <= k {
        return candidates;
    }

    // Reservoir sampling (Algorithm R).
    let mut reservoir: Vec<usize> = candidates[..k].to_vec();
    for (i, &item) in candidates[k..].iter().enumerate() {
        // i starts at 0; actual index in candidates is k + i.
        let total = k + i + 1;
        let u = rng.random_range(0.0_f64..1.0_f64);
        let j = (u * total as f64) as usize % total;
        if j < k {
            reservoir[j] = item;
        }
    }
    reservoir
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::seeded_rng;

    #[test]
    fn test_sample_neighbours_fewer_than_k() {
        let edges = vec![(0, 1), (0, 2), (1, 2)];
        let mut rng = seeded_rng(42);
        let result = sample_neighbours(0, &edges, 10, &mut rng);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&1));
        assert!(result.contains(&2));
    }

    #[test]
    fn test_sample_neighbours_exactly_k() {
        let edges: Vec<(usize, usize)> = (1..=5).map(|i| (0, i)).collect();
        let mut rng = seeded_rng(7);
        let result = sample_neighbours(0, &edges, 5, &mut rng);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_sample_neighbours_more_than_k() {
        let edges: Vec<(usize, usize)> = (1..=20).map(|i| (0, i)).collect();
        let mut rng = seeded_rng(0);
        let result = sample_neighbours(0, &edges, 5, &mut rng);
        assert_eq!(result.len(), 5);
        for n in &result {
            assert!((1..=20).contains(n));
        }
    }

    #[test]
    fn test_sample_neighbours_no_outgoing() {
        let edges = vec![(1, 0), (2, 0)];
        let mut rng = seeded_rng(1);
        let result = sample_neighbours(0, &edges, 3, &mut rng);
        assert!(result.is_empty());
    }
}
