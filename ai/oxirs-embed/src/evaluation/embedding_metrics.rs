//! Embedding quality metrics: link prediction, analogy reasoning, clustering.
//!
//! This module provides high-level evaluation primitives for knowledge graph
//! embedding models:
//!
//! - [`EmbeddingEvaluator`] — MRR, Hits\@K, cosine similarity.
//! - [`AnalogicalReasoningBenchmark`] — king − man + woman ≈ queen style tests.
//! - [`EmbeddingClusteringMetrics`] — silhouette score, Davies–Bouldin index.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::{EmbeddingModel, Triple};

// ─────────────────────────────────────────────────────────────────────────────
// Helper: cosine similarity between two dense f64 slices
// ─────────────────────────────────────────────────────────────────────────────

fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// EmbeddingEvaluator
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluates the quality of a knowledge-graph embedding model.
///
/// The evaluator works with any type that implements [`EmbeddingModel`], using
/// the `score_triple` and `get_entities` methods that are already part of the
/// trait.
#[derive(Debug, Clone, Default)]
pub struct EmbeddingEvaluator;

impl EmbeddingEvaluator {
    /// Create a new evaluator.
    pub fn new() -> Self {
        Self
    }

    /// Compute Mean Reciprocal Rank (MRR) for tail-entity prediction.
    ///
    /// For every test triple `(h, r, t)` the evaluator scores all candidate
    /// entities known to `model` as tail replacement, ranks them in descending
    /// score order, and records the reciprocal rank of the correct tail.
    ///
    /// Returns the mean over all test triples.  Returns `0.0` when
    /// `test_triples` is empty.
    pub fn link_prediction_mrr(&self, model: &dyn EmbeddingModel, test_triples: &[Triple]) -> f64 {
        if test_triples.is_empty() {
            return 0.0;
        }

        let entities = model.get_entities();
        if entities.is_empty() {
            return 0.0;
        }

        let reciprocal_ranks: Vec<f64> = test_triples
            .iter()
            .map(|triple| {
                let head = &triple.subject.iri;
                let rel = &triple.predicate.iri;
                let tail = &triple.object.iri;

                // Score every entity as the candidate tail.
                let mut scored: Vec<(String, f64)> = entities
                    .iter()
                    .filter_map(|cand| {
                        model
                            .score_triple(head, rel, cand)
                            .ok()
                            .map(|s| (cand.clone(), s))
                    })
                    .collect();

                // Sort descending by score.
                scored.sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });

                // Find the rank of the correct tail (1-based).
                let rank = scored
                    .iter()
                    .position(|(cand, _)| cand == tail)
                    .map(|pos| pos + 1);

                match rank {
                    Some(r) => 1.0 / r as f64,
                    None => 0.0,
                }
            })
            .collect();

        reciprocal_ranks.iter().sum::<f64>() / reciprocal_ranks.len() as f64
    }

    /// Compute Hits\@K: the fraction of test triples whose correct tail entity
    /// appears within the top `k` predictions.
    ///
    /// Returns `0.0` when `test_triples` is empty or `k == 0`.
    pub fn hits_at_k(&self, model: &dyn EmbeddingModel, test_triples: &[Triple], k: usize) -> f64 {
        if test_triples.is_empty() || k == 0 {
            return 0.0;
        }

        let entities = model.get_entities();
        if entities.is_empty() {
            return 0.0;
        }

        let hits: usize = test_triples
            .iter()
            .filter(|triple| {
                let head = &triple.subject.iri;
                let rel = &triple.predicate.iri;
                let tail = &triple.object.iri;

                let mut scored: Vec<(String, f64)> = entities
                    .iter()
                    .filter_map(|cand| {
                        model
                            .score_triple(head, rel, cand)
                            .ok()
                            .map(|s| (cand.clone(), s))
                    })
                    .collect();

                scored.sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });

                scored.iter().take(k).any(|(cand, _)| cand == tail)
            })
            .count();

        hits as f64 / test_triples.len() as f64
    }

    /// Cosine similarity between two embedding vectors.
    ///
    /// Returns a value in `[-1.0, 1.0]`.  Returns `0.0` for zero-length or
    /// mismatched-length inputs.
    pub fn semantic_similarity(&self, emb1: &[f64], emb2: &[f64]) -> f64 {
        cosine_sim(emb1, emb2)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AnalogicalReasoningBenchmark
// ─────────────────────────────────────────────────────────────────────────────

/// A quadruple for analogy evaluation: `a : b :: c : d`.
///
/// The benchmark tests whether the model can find `d` from `a − b + c`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalogyQuad {
    /// First word in the relation (e.g. "man").
    pub a: String,
    /// Second word in the relation (e.g. "king").
    pub b: String,
    /// Third word (e.g. "woman").
    pub c: String,
    /// Expected answer (e.g. "queen").
    pub expected_d: String,
}

impl AnalogyQuad {
    /// Construct a new `AnalogyQuad`.
    pub fn new(
        a: impl Into<String>,
        b: impl Into<String>,
        c: impl Into<String>,
        expected_d: impl Into<String>,
    ) -> Self {
        Self {
            a: a.into(),
            b: b.into(),
            c: c.into(),
            expected_d: expected_d.into(),
        }
    }
}

/// Evaluates analogy reasoning using the 3CosAdd method:
///
/// `emb(b) − emb(a) + emb(c)` should be closest to `emb(d)`.
#[derive(Debug, Clone, Default)]
pub struct AnalogicalReasoningBenchmark;

impl AnalogicalReasoningBenchmark {
    /// Create a new benchmark.
    pub fn new() -> Self {
        Self
    }

    /// Evaluate a single analogy quad.
    ///
    /// Returns `true` if the entity whose embedding is closest (by cosine
    /// similarity) to `emb(b) - emb(a) + emb(c)` is `expected_d`.
    ///
    /// Entities equal to `a`, `b`, or `c` are excluded from the candidate set
    /// following the standard 3CosAdd protocol.
    ///
    /// Returns `false` on any retrieval error or if the model has not been
    /// trained (embeddings unavailable).
    pub fn evaluate_analogy(&self, model: &dyn EmbeddingModel, quad: &AnalogyQuad) -> bool {
        let get = |name: &str| -> Option<Vec<f64>> {
            model
                .get_entity_embedding(name)
                .ok()
                .map(|v| v.values.iter().map(|&x| x as f64).collect())
        };

        let emb_a = match get(&quad.a) {
            Some(e) => e,
            None => return false,
        };
        let emb_b = match get(&quad.b) {
            Some(e) => e,
            None => return false,
        };
        let emb_c = match get(&quad.c) {
            Some(e) => e,
            None => return false,
        };

        let dim = emb_a.len();
        if dim == 0 || emb_b.len() != dim || emb_c.len() != dim {
            return false;
        }

        // target = emb(b) - emb(a) + emb(c)
        let target: Vec<f64> = (0..dim).map(|i| emb_b[i] - emb_a[i] + emb_c[i]).collect();

        let excluded = [quad.a.as_str(), quad.b.as_str(), quad.c.as_str()];
        let entities = model.get_entities();

        let best = entities
            .iter()
            .filter(|e| !excluded.contains(&e.as_str()))
            .filter_map(|cand| {
                get(cand).map(|emb| {
                    let sim = cosine_sim(&target, &emb);
                    (cand.clone(), sim)
                })
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        match best {
            Some((predicted, _)) => predicted == quad.expected_d,
            None => false,
        }
    }

    /// Evaluate a slice of analogy quads and return the accuracy (fraction correct).
    ///
    /// Returns `0.0` when `quads` is empty.
    pub fn evaluate_analogies(&self, model: &dyn EmbeddingModel, quads: &[AnalogyQuad]) -> f64 {
        if quads.is_empty() {
            return 0.0;
        }
        let correct = quads
            .iter()
            .filter(|q| self.evaluate_analogy(model, q))
            .count();
        correct as f64 / quads.len() as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EmbeddingClusteringMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Cluster quality metrics for embedding spaces.
///
/// All methods assume Euclidean distance between embedding vectors.
#[derive(Debug, Clone, Default)]
pub struct EmbeddingClusteringMetrics;

impl EmbeddingClusteringMetrics {
    /// Create a new metrics calculator.
    pub fn new() -> Self {
        Self
    }

    // ── private helpers ───────────────────────────────────────────────────────

    fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn cluster_centroid(points: &[&Vec<f64>]) -> Vec<f64> {
        if points.is_empty() {
            return Vec::new();
        }
        let dim = points[0].len();
        let n = points.len() as f64;
        let mut centroid = vec![0.0_f64; dim];
        for p in points {
            for (i, v) in p.iter().enumerate() {
                centroid[i] += v;
            }
        }
        centroid.iter_mut().for_each(|v| *v /= n);
        centroid
    }

    // ── public API ────────────────────────────────────────────────────────────

    /// Compute the silhouette score for the given embeddings and cluster assignments.
    ///
    /// The silhouette score ranges from `−1` (incorrect clustering) to `+1`
    /// (dense, well-separated clusters).  A score near `0` indicates overlapping
    /// clusters.
    ///
    /// Returns an error when:
    /// - `embeddings` and `cluster_assignments` have different lengths.
    /// - There are fewer than 2 distinct clusters.
    pub fn silhouette_score(
        &self,
        embeddings: &[Vec<f64>],
        cluster_assignments: &[usize],
    ) -> Result<f64> {
        let n = embeddings.len();
        if n != cluster_assignments.len() {
            return Err(anyhow!(
                "embeddings ({}) and cluster_assignments ({}) must have the same length",
                n,
                cluster_assignments.len()
            ));
        }
        if n < 2 {
            return Err(anyhow!(
                "need at least 2 samples to compute silhouette score"
            ));
        }

        let unique_clusters: std::collections::HashSet<usize> =
            cluster_assignments.iter().copied().collect();
        if unique_clusters.len() < 2 {
            return Err(anyhow!(
                "need at least 2 distinct clusters; found {}",
                unique_clusters.len()
            ));
        }

        let mut scores = Vec::with_capacity(n);

        for i in 0..n {
            let c_i = cluster_assignments[i];
            let dim = embeddings[i].len();
            if dim == 0 {
                continue;
            }

            // a(i): mean intra-cluster distance
            let intra_dists: Vec<f64> = (0..n)
                .filter(|&j| j != i && cluster_assignments[j] == c_i)
                .map(|j| Self::euclidean_dist(&embeddings[i], &embeddings[j]))
                .collect();

            let a_i = if intra_dists.is_empty() {
                0.0
            } else {
                intra_dists.iter().sum::<f64>() / intra_dists.len() as f64
            };

            // b(i): minimum mean inter-cluster distance over all other clusters
            let b_i = unique_clusters
                .iter()
                .filter(|&&c| c != c_i)
                .map(|&c| {
                    let dists: Vec<f64> = (0..n)
                        .filter(|&j| cluster_assignments[j] == c)
                        .map(|j| Self::euclidean_dist(&embeddings[i], &embeddings[j]))
                        .collect();
                    if dists.is_empty() {
                        f64::INFINITY
                    } else {
                        dists.iter().sum::<f64>() / dists.len() as f64
                    }
                })
                .fold(f64::INFINITY, f64::min);

            let s_i = if b_i == f64::INFINITY || (a_i == 0.0 && b_i == 0.0) {
                0.0
            } else {
                let denom = a_i.max(b_i);
                if denom == 0.0 {
                    0.0
                } else {
                    (b_i - a_i) / denom
                }
            };

            scores.push(s_i);
        }

        if scores.is_empty() {
            return Ok(0.0);
        }

        Ok(scores.iter().sum::<f64>() / scores.len() as f64)
    }

    /// Compute the Davies–Bouldin index.
    ///
    /// Lower values indicate better clustering.  Returns an error when:
    /// - `embeddings` and `cluster_assignments` have different lengths.
    /// - There are fewer than 2 distinct clusters.
    pub fn davies_bouldin_index(
        &self,
        embeddings: &[Vec<f64>],
        cluster_assignments: &[usize],
    ) -> Result<f64> {
        let n = embeddings.len();
        if n != cluster_assignments.len() {
            return Err(anyhow!(
                "embeddings ({}) and cluster_assignments ({}) must have the same length",
                n,
                cluster_assignments.len()
            ));
        }
        if n < 2 {
            return Err(anyhow!(
                "need at least 2 samples to compute Davies-Bouldin index"
            ));
        }

        let mut cluster_ids: Vec<usize> = cluster_assignments.to_vec();
        cluster_ids.sort_unstable();
        cluster_ids.dedup();

        if cluster_ids.len() < 2 {
            return Err(anyhow!(
                "need at least 2 distinct clusters; found {}",
                cluster_ids.len()
            ));
        }

        // Build per-cluster point lists.
        let cluster_points: std::collections::HashMap<usize, Vec<&Vec<f64>>> = cluster_ids
            .iter()
            .fold(std::collections::HashMap::new(), |mut acc, &cid| {
                let pts: Vec<&Vec<f64>> = embeddings
                    .iter()
                    .zip(cluster_assignments.iter())
                    .filter(|(_, &a)| a == cid)
                    .map(|(e, _)| e)
                    .collect();
                acc.insert(cid, pts);
                acc
            });

        // Compute per-cluster centroid and scatter (mean distance to centroid).
        let centroids: std::collections::HashMap<usize, Vec<f64>> = cluster_ids
            .iter()
            .map(|&cid| {
                let pts = &cluster_points[&cid];
                let centroid = Self::cluster_centroid(pts);
                (cid, centroid)
            })
            .collect();

        let scatter: std::collections::HashMap<usize, f64> = cluster_ids
            .iter()
            .map(|&cid| {
                let pts = &cluster_points[&cid];
                let c = &centroids[&cid];
                let s = if pts.is_empty() {
                    0.0
                } else {
                    let total: f64 = pts.iter().map(|p| Self::euclidean_dist(p, c)).sum();
                    total / pts.len() as f64
                };
                (cid, s)
            })
            .collect();

        // DB index: mean over clusters of the max R_ij for j != i.
        let k = cluster_ids.len() as f64;
        let db_sum: f64 = cluster_ids
            .iter()
            .map(|&ci| {
                let max_r = cluster_ids
                    .iter()
                    .filter(|&&cj| cj != ci)
                    .map(|&cj| {
                        let d = Self::euclidean_dist(&centroids[&ci], &centroids[&cj]);
                        if d == 0.0 {
                            0.0
                        } else {
                            (scatter[&ci] + scatter[&cj]) / d
                        }
                    })
                    .fold(f64::NEG_INFINITY, f64::max);
                if max_r == f64::NEG_INFINITY {
                    0.0
                } else {
                    max_r
                }
            })
            .sum();

        Ok(db_sum / k)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── cosine_sim helper ─────────────────────────────────────────────────────

    #[test]
    fn test_cosine_sim_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_sim(&v, &v) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_sim_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_sim(&a, &b)).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_sim_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_sim(&a, &b) + 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_sim_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_sim(&a, &b), 0.0);
    }

    // ── EmbeddingEvaluator::semantic_similarity ───────────────────────────────

    #[test]
    fn test_semantic_similarity_via_evaluator() {
        let ev = EmbeddingEvaluator::new();
        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.0];
        assert!((ev.semantic_similarity(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_semantic_similarity_orthogonal() {
        let ev = EmbeddingEvaluator::new();
        assert!((ev.semantic_similarity(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-9);
    }

    #[test]
    fn test_semantic_similarity_empty() {
        let ev = EmbeddingEvaluator::new();
        assert_eq!(ev.semantic_similarity(&[], &[]), 0.0);
    }

    // ── EmbeddingEvaluator::link_prediction_mrr (model-free edge cases) ──────

    #[test]
    fn test_mrr_empty_triples() {
        let ev = EmbeddingEvaluator::new();
        // Use a trivial struct as stand-in — but we can just test the empty branch.
        // We need an EmbeddingModel; use a mock defined below.
        let mock = MockModel::new(vec!["e1".into(), "e2".into()]);
        assert_eq!(ev.link_prediction_mrr(&mock, &[]), 0.0);
    }

    #[test]
    fn test_hits_at_k_empty_triples() {
        let ev = EmbeddingEvaluator::new();
        let mock = MockModel::new(vec!["e1".into()]);
        assert_eq!(ev.hits_at_k(&mock, &[], 3), 0.0);
    }

    #[test]
    fn test_hits_at_k_zero_k() {
        let ev = EmbeddingEvaluator::new();
        let mock = MockModel::new(vec!["e1".into()]);
        let triple = Triple::new(
            crate::NamedNode::new("e1").unwrap(),
            crate::NamedNode::new("r").unwrap(),
            crate::NamedNode::new("e1").unwrap(),
        );
        assert_eq!(ev.hits_at_k(&mock, &[triple], 0), 0.0);
    }

    // ── AnalogyQuad ────────────────────────────────────────────────────────────

    #[test]
    fn test_analogy_quad_construction() {
        let q = AnalogyQuad::new("man", "king", "woman", "queen");
        assert_eq!(q.a, "man");
        assert_eq!(q.expected_d, "queen");
    }

    #[test]
    fn test_analogy_quad_serialization() {
        let q = AnalogyQuad::new("paris", "france", "berlin", "germany");
        let json = serde_json::to_string(&q).expect("serialize");
        let q2: AnalogyQuad = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(q, q2);
    }

    #[test]
    fn test_evaluate_analogies_empty() {
        let bench = AnalogicalReasoningBenchmark::new();
        let mock = MockModel::new(vec![]);
        assert_eq!(bench.evaluate_analogies(&mock, &[]), 0.0);
    }

    // ── EmbeddingClusteringMetrics ────────────────────────────────────────────

    #[test]
    fn test_silhouette_perfect_clusters() {
        let metrics = EmbeddingClusteringMetrics::new();
        // Two tight, well-separated clusters.
        let embeddings = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.1],
        ];
        let assignments = vec![0, 0, 0, 1, 1, 1];
        let score = metrics
            .silhouette_score(&embeddings, &assignments)
            .expect("ok");
        // Well-separated clusters should give a high silhouette score.
        assert!(score > 0.8, "expected high score, got {score}");
    }

    #[test]
    fn test_silhouette_mismatched_lengths() {
        let metrics = EmbeddingClusteringMetrics::new();
        let result = metrics.silhouette_score(&[vec![1.0]], &[0, 1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_silhouette_single_cluster_error() {
        let metrics = EmbeddingClusteringMetrics::new();
        let embeddings = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let assignments = vec![0, 0]; // only one cluster
        assert!(metrics.silhouette_score(&embeddings, &assignments).is_err());
    }

    #[test]
    fn test_davies_bouldin_perfect_clusters() {
        let metrics = EmbeddingClusteringMetrics::new();
        let embeddings = vec![
            vec![0.0, 0.0],
            vec![0.05, 0.0],
            vec![20.0, 20.0],
            vec![20.05, 20.0],
        ];
        let assignments = vec![0, 0, 1, 1];
        let db = metrics
            .davies_bouldin_index(&embeddings, &assignments)
            .expect("ok");
        // Well-separated clusters should give a low DB index.
        assert!(db < 0.1, "expected low DB index, got {db}");
    }

    #[test]
    fn test_davies_bouldin_mismatched_lengths() {
        let metrics = EmbeddingClusteringMetrics::new();
        let result = metrics.davies_bouldin_index(&[vec![1.0]], &[0, 1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_davies_bouldin_single_cluster_error() {
        let metrics = EmbeddingClusteringMetrics::new();
        let embeddings = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let assignments = vec![0, 0];
        assert!(metrics
            .davies_bouldin_index(&embeddings, &assignments)
            .is_err());
    }

    #[test]
    fn test_silhouette_three_clusters() {
        let metrics = EmbeddingClusteringMetrics::new();
        // Three compact, well-separated clusters in 2D.
        let embeddings = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 0.0],
            vec![10.1, 0.1],
            vec![5.0, 8.66],
            vec![5.1, 8.76],
        ];
        let assignments = vec![0, 0, 1, 1, 2, 2];
        let score = metrics
            .silhouette_score(&embeddings, &assignments)
            .expect("ok");
        assert!(score > 0.5, "expected positive silhouette, got {score}");
    }

    #[test]
    fn test_davies_bouldin_three_clusters() {
        let metrics = EmbeddingClusteringMetrics::new();
        let embeddings = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![10.0, 0.0],
            vec![10.1, 0.0],
            vec![5.0, 8.66],
            vec![5.1, 8.66],
        ];
        let assignments = vec![0, 0, 1, 1, 2, 2];
        let db = metrics
            .davies_bouldin_index(&embeddings, &assignments)
            .expect("ok");
        assert!(db < 0.5, "expected low DB index, got {db}");
    }

    // ── Mock EmbeddingModel for unit tests ────────────────────────────────────

    use crate::{ModelConfig, ModelStats, NamedNode, TrainingStats, Vector};
    use anyhow::Result as AResult;
    use async_trait::async_trait;
    use uuid::Uuid;

    struct MockModel {
        entities: Vec<String>,
        id: Uuid,
        config: ModelConfig,
    }

    impl MockModel {
        fn new(entities: Vec<String>) -> Self {
            Self {
                entities,
                id: Uuid::new_v4(),
                config: ModelConfig::default(),
            }
        }
    }

    #[async_trait]
    impl EmbeddingModel for MockModel {
        fn config(&self) -> &ModelConfig {
            &self.config
        }
        fn model_id(&self) -> &Uuid {
            &self.id
        }
        fn model_type(&self) -> &'static str {
            "mock"
        }
        fn add_triple(&mut self, _triple: Triple) -> AResult<()> {
            Ok(())
        }
        async fn train(&mut self, _epochs: Option<usize>) -> AResult<TrainingStats> {
            Ok(TrainingStats::default())
        }
        fn get_entity_embedding(&self, entity: &str) -> AResult<Vector> {
            // Deterministic embedding: hash-based values so analogy tests are meaningful.
            let v: Vec<f32> = entity
                .bytes()
                .take(4)
                .enumerate()
                .map(|(i, b)| (b as f32 + i as f32) / 256.0)
                .collect();
            Ok(Vector::new(v))
        }
        fn get_relation_embedding(&self, _rel: &str) -> AResult<Vector> {
            Ok(Vector::new(vec![0.1, 0.2]))
        }
        fn score_triple(&self, _h: &str, _r: &str, t: &str) -> AResult<f64> {
            // Score based on position in entity list for determinism.
            let score = self
                .entities
                .iter()
                .position(|e| e == t)
                .map(|pos| 1.0 / (pos + 1) as f64)
                .unwrap_or(0.0);
            Ok(score)
        }
        fn predict_objects(&self, _s: &str, _p: &str, k: usize) -> AResult<Vec<(String, f64)>> {
            Ok(self
                .entities
                .iter()
                .take(k)
                .map(|e| (e.clone(), 1.0))
                .collect())
        }
        fn predict_subjects(&self, _p: &str, _o: &str, k: usize) -> AResult<Vec<(String, f64)>> {
            Ok(self
                .entities
                .iter()
                .take(k)
                .map(|e| (e.clone(), 1.0))
                .collect())
        }
        fn predict_relations(&self, _s: &str, _o: &str, k: usize) -> AResult<Vec<(String, f64)>> {
            Ok(self
                .entities
                .iter()
                .take(k)
                .map(|e| (e.clone(), 1.0))
                .collect())
        }
        fn get_entities(&self) -> Vec<String> {
            self.entities.clone()
        }
        fn get_relations(&self) -> Vec<String> {
            vec!["rel".to_string()]
        }
        fn get_stats(&self) -> ModelStats {
            ModelStats::default()
        }
        fn save(&self, _path: &str) -> AResult<()> {
            Ok(())
        }
        fn load(&mut self, _path: &str) -> AResult<()> {
            Ok(())
        }
        fn clear(&mut self) {}
        fn is_trained(&self) -> bool {
            true
        }
        async fn encode(&self, texts: &[String]) -> AResult<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.0f32; 4]).collect())
        }
    }

    // ── Additional coverage tests ─────────────────────────────────────────────

    #[test]
    fn test_mrr_correct_at_rank_1() {
        let ev = EmbeddingEvaluator::new();
        // MockModel scores entity[0] highest, so entity "e1" is rank-1.
        let mock = MockModel::new(vec!["e1".into(), "e2".into(), "e3".into()]);
        let triple = Triple::new(
            NamedNode::new("e1").unwrap(),
            NamedNode::new("r").unwrap(),
            NamedNode::new("e1").unwrap(),
        );
        let mrr = ev.link_prediction_mrr(&mock, &[triple]);
        assert!((mrr - 1.0).abs() < 1e-9, "expected MRR=1 got {mrr}");
    }

    #[test]
    fn test_hits_at_k_correct_first() {
        let ev = EmbeddingEvaluator::new();
        let mock = MockModel::new(vec!["e1".into(), "e2".into(), "e3".into()]);
        let triple = Triple::new(
            NamedNode::new("e1").unwrap(),
            NamedNode::new("r").unwrap(),
            NamedNode::new("e1").unwrap(),
        );
        let h = ev.hits_at_k(&mock, &[triple], 1);
        assert!((h - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_hits_at_k_not_in_top_1() {
        let ev = EmbeddingEvaluator::new();
        // Entity "e3" is last in list (rank 3), so Hits@1 = 0.
        let mock = MockModel::new(vec!["e1".into(), "e2".into(), "e3".into()]);
        let triple = Triple::new(
            NamedNode::new("e1").unwrap(),
            NamedNode::new("r").unwrap(),
            NamedNode::new("e3").unwrap(),
        );
        let h = ev.hits_at_k(&mock, &[triple], 1);
        assert!((h - 0.0).abs() < 1e-9);
    }
}
