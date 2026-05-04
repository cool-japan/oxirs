//! IVF (Inverted File) vector index.

use super::vector_store::{
    compute_similarity, IndexStats, SimilarityMetric, VectorData, VectorIndex,
};
use anyhow::Result;
use dashmap::DashMap;
use scirs2_core::random::Random;
use std::cmp::Ordering;
use std::collections::HashMap;

/// IVF index: clusters vectors into `num_clusters` Voronoi cells; searches
/// only `num_probes` nearest cluster(s) at query time.
///
/// Scoring formula: cosine / dot-product / Euclidean similarity between the
/// query and every vector in the selected cluster(s).
pub struct IVFIndex {
    /// Number of Voronoi clusters (centroids)
    num_clusters: usize,
    /// Number of clusters probed at search time
    num_probes: usize,
    /// Cluster centroids (centroid_idx → centroid vector)
    centroids: Vec<Vec<f32>>,
    /// Inverted lists: centroid_idx → list of (id, vector)
    inverted_lists: Vec<Vec<(String, Vec<f32>)>>,
    /// Flat copy for add/remove operations before a rebuild
    staging: HashMap<String, Vec<f32>>,
    stats: IndexStats,
}

impl IVFIndex {
    /// Create a new IVF index.
    pub fn new(num_clusters: usize, num_probes: usize) -> Self {
        let num_clusters = num_clusters.max(1);
        let num_probes = num_probes.min(num_clusters).max(1);
        Self {
            num_clusters,
            num_probes,
            centroids: Vec::new(),
            inverted_lists: Vec::new(),
            staging: HashMap::new(),
            stats: IndexStats {
                index_type: "IVF".to_string(),
                num_vectors: 0,
                build_time: std::time::Duration::from_secs(0),
                memory_usage: 0,
            },
        }
    }

    /// Run Lloyd's k-means for up to 25 iterations.
    fn kmeans(data: &[(&str, &[f32])], k: usize) -> Vec<Vec<f32>> {
        if data.is_empty() || k == 0 {
            return Vec::new();
        }
        let dim = data[0].1.len();
        let k = k.min(data.len());

        // Forgy initialisation: pick k distinct points at random.
        let mut rng = Random::default();
        let mut indices: Vec<usize> = (0..data.len()).collect();
        for i in (1..indices.len()).rev() {
            let j = rng.random_range(0..=i);
            indices.swap(i, j);
        }
        let mut centroids: Vec<Vec<f32>> =
            indices[..k].iter().map(|&i| data[i].1.to_vec()).collect();

        for _ in 0..25 {
            // Assignment step
            let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); k];
            for (point_idx, (_, vec)) in data.iter().enumerate() {
                let best = Self::nearest_centroid(vec, &centroids);
                clusters[best].push(point_idx);
            }

            // Update step
            let mut changed = false;
            for (c_idx, members) in clusters.iter().enumerate() {
                if members.is_empty() {
                    continue;
                }
                let mut new_centroid = vec![0.0f32; dim];
                for &pt_idx in members {
                    for (d, v) in data[pt_idx].1.iter().enumerate() {
                        new_centroid[d] += v;
                    }
                }
                let n = members.len() as f32;
                for v in &mut new_centroid {
                    *v /= n;
                }
                if new_centroid != centroids[c_idx] {
                    centroids[c_idx] = new_centroid;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        centroids
    }

    /// Return the index of the centroid closest to `vec` (L2 distance).
    fn nearest_centroid(vec: &[f32], centroids: &[Vec<f32>]) -> usize {
        centroids
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let d: f32 = vec
                    .iter()
                    .zip(c.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                (i, d)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Return the `n` closest centroid indices to `query`.
    fn nearest_n_centroids(query: &[f32], centroids: &[Vec<f32>], n: usize) -> Vec<usize> {
        let mut dists: Vec<(usize, f32)> = centroids
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let d: f32 = query
                    .iter()
                    .zip(c.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                (i, d)
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        dists.iter().take(n).map(|(i, _)| *i).collect()
    }
}

#[async_trait::async_trait]
impl VectorIndex for IVFIndex {
    async fn build(&mut self, vectors: &DashMap<String, VectorData>) -> Result<()> {
        let start = std::time::Instant::now();

        let pairs: Vec<(String, Vec<f32>)> = vectors
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().vector.clone()))
            .collect();

        let data_refs: Vec<(&str, &[f32])> = pairs
            .iter()
            .map(|(id, vec)| (id.as_str(), vec.as_slice()))
            .collect();

        self.centroids = Self::kmeans(&data_refs, self.num_clusters);
        self.inverted_lists = vec![Vec::new(); self.centroids.len()];

        for (id, vec) in &pairs {
            let c = Self::nearest_centroid(vec, &self.centroids);
            self.inverted_lists[c].push((id.clone(), vec.clone()));
        }

        self.staging.clear();
        for (id, vec) in pairs {
            self.staging.insert(id, vec);
        }

        self.stats.num_vectors = self.staging.len();
        self.stats.build_time = start.elapsed();
        Ok(())
    }

    async fn search(
        &self,
        query: &[f32],
        k: usize,
        metric: SimilarityMetric,
    ) -> Result<Vec<(String, f32)>> {
        if self.centroids.is_empty() {
            // Fall back to brute-force over staging data
            let mut results: Vec<(String, f32)> = self
                .staging
                .iter()
                .map(|(id, vec)| {
                    let sim = compute_similarity(query, vec, metric).unwrap_or(f32::NEG_INFINITY);
                    (id.clone(), sim)
                })
                .collect();
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            results.truncate(k);
            return Ok(results);
        }

        let probes = Self::nearest_n_centroids(query, &self.centroids, self.num_probes);

        let mut candidates: Vec<(String, f32)> = Vec::new();
        for probe_idx in probes {
            if let Some(list) = self.inverted_lists.get(probe_idx) {
                for (id, vec) in list {
                    let sim = compute_similarity(query, vec, metric)?;
                    candidates.push((id.clone(), sim));
                }
            }
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        candidates.truncate(k);
        Ok(candidates)
    }

    async fn add(&mut self, id: String, vector: Vec<f32>) -> Result<()> {
        if !self.centroids.is_empty() {
            let c = Self::nearest_centroid(&vector, &self.centroids);
            self.inverted_lists[c].push((id.clone(), vector.clone()));
        }
        self.staging.insert(id, vector);
        self.stats.num_vectors = self.staging.len();
        Ok(())
    }

    async fn remove(&mut self, id: &str) -> Result<()> {
        self.staging.remove(id);
        for list in &mut self.inverted_lists {
            list.retain(|(list_id, _)| list_id != id);
        }
        self.stats.num_vectors = self.staging.len();
        Ok(())
    }

    fn get_stats(&self) -> IndexStats {
        self.stats.clone()
    }
}
