//! PQ (Product Quantization) vector index.

use super::vector_store::{IndexStats, SimilarityMetric, VectorData, VectorIndex};
use anyhow::Result;
use dashmap::DashMap;
use scirs2_core::random::Random;
use std::cmp::Ordering;
use std::collections::HashMap;

/// Product Quantization index.
///
/// The vector space is partitioned into `num_subquantizers` equal sub-spaces.
/// Each sub-space has its own k-means codebook with `2^bits_per_subquantizer`
/// codewords.  Vectors are encoded as compact integer codes; ADC (Asymmetric
/// Distance Computation) is used at search time.
pub struct PQIndexLocal {
    num_subquantizers: usize,
    bits_per_subquantizer: usize,
    /// Codebooks: subquantizer × codeword × sub-vector
    codebooks: Vec<Vec<Vec<f32>>>,
    /// Encoded dataset: id → compact code (one codeword per subquantizer)
    codes: HashMap<String, Vec<usize>>,
    /// Full vectors kept for exact re-ranking (optional)
    staging: HashMap<String, Vec<f32>>,
    stats: IndexStats,
}

impl PQIndexLocal {
    /// Create a new PQ index.
    pub fn new(num_subquantizers: usize, bits_per_subquantizer: usize) -> Self {
        Self {
            num_subquantizers,
            bits_per_subquantizer,
            codebooks: Vec::new(),
            codes: HashMap::new(),
            staging: HashMap::new(),
            stats: IndexStats {
                index_type: "PQ".to_string(),
                num_vectors: 0,
                build_time: std::time::Duration::from_secs(0),
                memory_usage: 0,
            },
        }
    }

    /// Number of codewords per sub-space.
    fn num_codewords(&self) -> usize {
        1 << self.bits_per_subquantizer.min(16)
    }

    /// Run k-means on a list of sub-vectors and return centroids.
    fn subspace_kmeans(data: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
        if data.is_empty() || k == 0 {
            return Vec::new();
        }
        let k = k.min(data.len());
        let dim = data[0].len();

        let mut rng = Random::default();
        let mut indices: Vec<usize> = (0..data.len()).collect();
        for i in (1..indices.len()).rev() {
            let j = rng.random_range(0..=i);
            indices.swap(i, j);
        }
        let mut centroids: Vec<Vec<f32>> = indices[..k].iter().map(|&i| data[i].clone()).collect();

        for _ in 0..25 {
            let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); k];
            for (pt_idx, vec) in data.iter().enumerate() {
                let best = Self::nearest(vec, &centroids);
                clusters[best].push(pt_idx);
            }

            let mut changed = false;
            for (c_idx, members) in clusters.iter().enumerate() {
                if members.is_empty() {
                    continue;
                }
                let mut new_c = vec![0.0f32; dim];
                for &pi in members {
                    for (d, v) in data[pi].iter().enumerate() {
                        new_c[d] += v;
                    }
                }
                let n = members.len() as f32;
                for v in &mut new_c {
                    *v /= n;
                }
                if new_c != centroids[c_idx] {
                    centroids[c_idx] = new_c;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        centroids
    }

    /// Return the index of the nearest centroid (L2).
    fn nearest(vec: &[f32], centroids: &[Vec<f32>]) -> usize {
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

    /// Encode a vector into PQ codes.
    fn encode(&self, vec: &[f32]) -> Vec<usize> {
        let sub_dim = (vec.len() / self.num_subquantizers).max(1);
        (0..self.num_subquantizers)
            .map(|m| {
                let start = m * sub_dim;
                let end = if m == self.num_subquantizers - 1 {
                    vec.len()
                } else {
                    start + sub_dim
                };
                let sub = &vec[start..end.min(vec.len())];
                if m < self.codebooks.len() {
                    Self::nearest(sub, &self.codebooks[m])
                } else {
                    0
                }
            })
            .collect()
    }

    /// Compute the ADC (Asymmetric Distance Computation) score.
    ///
    /// Returns negative L2 distance so that higher = more similar,
    /// consistent with the similarity ranking convention used throughout.
    fn adc_score(&self, query: &[f32], code: &[usize]) -> f32 {
        let sub_dim = (query.len() / self.num_subquantizers).max(1);
        let mut total_dist: f32 = 0.0;
        for m in 0..self.num_subquantizers {
            let start = m * sub_dim;
            let end = if m == self.num_subquantizers - 1 {
                query.len()
            } else {
                start + sub_dim
            };
            let q_sub = &query[start..end.min(query.len())];
            let cw_idx = code.get(m).copied().unwrap_or(0);
            let codeword = match self.codebooks.get(m).and_then(|cb| cb.get(cw_idx)) {
                Some(cw) => cw,
                None => continue,
            };
            let dist: f32 = q_sub
                .iter()
                .zip(codeword.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            total_dist += dist;
        }
        -total_dist // negate so higher = better
    }
}

#[async_trait::async_trait]
impl VectorIndex for PQIndexLocal {
    async fn build(&mut self, vectors: &DashMap<String, VectorData>) -> Result<()> {
        let start = std::time::Instant::now();

        let pairs: Vec<(String, Vec<f32>)> = vectors
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().vector.clone()))
            .collect();

        if pairs.is_empty() {
            self.stats.build_time = start.elapsed();
            return Ok(());
        }

        let dim = pairs[0].1.len();
        let sub_dim = (dim / self.num_subquantizers).max(1);
        let k = self.num_codewords();

        // Build codebook for each sub-space
        self.codebooks = (0..self.num_subquantizers)
            .map(|m| {
                let start_d = m * sub_dim;
                let end_d = if m == self.num_subquantizers - 1 {
                    dim
                } else {
                    start_d + sub_dim
                };
                let sub_vecs: Vec<Vec<f32>> = pairs
                    .iter()
                    .map(|(_, vec)| vec[start_d..end_d.min(vec.len())].to_vec())
                    .collect();
                Self::subspace_kmeans(&sub_vecs, k)
            })
            .collect();

        // Encode all vectors
        self.codes.clear();
        self.staging.clear();
        for (id, vec) in &pairs {
            let code = self.encode(vec);
            self.codes.insert(id.clone(), code);
            self.staging.insert(id.clone(), vec.clone());
        }

        self.stats.num_vectors = self.staging.len();
        self.stats.build_time = start.elapsed();
        Ok(())
    }

    async fn search(
        &self,
        query: &[f32],
        k: usize,
        _metric: SimilarityMetric,
    ) -> Result<Vec<(String, f32)>> {
        // Use ADC scoring (approximation of negative L2 distance)
        let mut candidates: Vec<(String, f32)> = self
            .codes
            .iter()
            .map(|(id, code)| {
                let score = self.adc_score(query, code);
                (id.clone(), score)
            })
            .collect();

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        candidates.truncate(k);
        Ok(candidates)
    }

    async fn add(&mut self, id: String, vector: Vec<f32>) -> Result<()> {
        if !self.codebooks.is_empty() {
            let code = self.encode(&vector);
            self.codes.insert(id.clone(), code);
        }
        self.staging.insert(id, vector);
        self.stats.num_vectors = self.staging.len();
        Ok(())
    }

    async fn remove(&mut self, id: &str) -> Result<()> {
        self.staging.remove(id);
        self.codes.remove(id);
        self.stats.num_vectors = self.staging.len();
        Ok(())
    }

    fn get_stats(&self) -> IndexStats {
        self.stats.clone()
    }
}
