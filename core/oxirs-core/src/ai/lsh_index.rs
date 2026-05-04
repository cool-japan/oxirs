//! LSH (Locality Sensitive Hashing) vector index.

use super::vector_store::{
    compute_similarity, IndexStats, SimilarityMetric, VectorData, VectorIndex,
};
use anyhow::Result;
use dashmap::DashMap;
use scirs2_core::random::{Random, RngExt};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

/// Bucket type: hash_code → list of (id, vector)
type LshBucket = HashMap<u64, Vec<(String, Vec<f32>)>>;

/// LSH index using random hyperplane projections.
///
/// Each hash table uses `hash_length` random bit-projections.  Two vectors
/// that are close in cosine similarity are likely to share a hash bucket.
pub struct LSHIndex {
    /// Number of hash tables
    num_tables: usize,
    /// Number of hash bits per table
    hash_length: usize,
    /// Random projection hyperplanes: tables × hash_length × dim
    hyperplanes: Vec<Vec<Vec<f32>>>,
    /// Hash tables: table_idx → (hash_code → list of (id, vector))
    tables: Vec<LshBucket>,
    /// All vectors for exact re-ranking of candidates
    staging: HashMap<String, Vec<f32>>,
    stats: IndexStats,
}

impl LSHIndex {
    /// Create a new LSH index.
    pub fn new(num_tables: usize, hash_length: usize) -> Self {
        Self {
            num_tables,
            hash_length,
            hyperplanes: Vec::new(),
            tables: Vec::new(),
            staging: HashMap::new(),
            stats: IndexStats {
                index_type: "LSH".to_string(),
                num_vectors: 0,
                build_time: std::time::Duration::from_secs(0),
                memory_usage: 0,
            },
        }
    }

    /// Generate a random unit normal value using Box-Muller transform.
    fn random_normal(rng: &mut Random) -> f32 {
        let u1: f32 = rng.random::<f32>().max(1e-10);
        let u2: f32 = rng.random::<f32>();
        let r = (-2.0 * u1.ln()).sqrt();
        r * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// Build random hyperplane projections for the given dimensionality.
    fn build_hyperplanes(&mut self, dim: usize) {
        let mut rng = Random::default();
        self.hyperplanes = (0..self.num_tables)
            .map(|_| {
                (0..self.hash_length)
                    .map(|_| {
                        let mut plane: Vec<f32> =
                            (0..dim).map(|_| Self::random_normal(&mut rng)).collect();
                        // Normalise the hyperplane vector
                        let norm: f32 = plane.iter().map(|x| x * x).sum::<f32>().sqrt();
                        if norm > 1e-10 {
                            for v in &mut plane {
                                *v /= norm;
                            }
                        }
                        plane
                    })
                    .collect()
            })
            .collect();
    }

    /// Compute the hash code for a vector in a given table.
    fn hash_vector(&self, table_idx: usize, vec: &[f32]) -> u64 {
        let planes = &self.hyperplanes[table_idx];
        let mut code: u64 = 0;
        for (bit, plane) in planes.iter().enumerate() {
            let dot: f32 = vec.iter().zip(plane.iter()).map(|(a, b)| a * b).sum();
            if dot >= 0.0 {
                code |= 1 << bit;
            }
        }
        code
    }
}

#[async_trait::async_trait]
impl VectorIndex for LSHIndex {
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
        self.build_hyperplanes(dim);
        self.tables = vec![HashMap::new(); self.num_tables];

        for (id, vec) in &pairs {
            for t in 0..self.num_tables {
                let code = self.hash_vector(t, vec);
                self.tables[t]
                    .entry(code)
                    .or_default()
                    .push((id.clone(), vec.clone()));
            }
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
        metric: SimilarityMetric,
    ) -> Result<Vec<(String, f32)>> {
        if self.hyperplanes.is_empty() {
            // Brute-force fallback when index has not been built yet
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

        // Collect candidates from all tables
        let mut seen: HashSet<String> = HashSet::new();
        let mut candidates: Vec<(String, f32)> = Vec::new();

        for t in 0..self.num_tables {
            let code = self.hash_vector(t, query);
            if let Some(bucket) = self.tables[t].get(&code) {
                for (id, vec) in bucket {
                    if seen.insert(id.clone()) {
                        let sim = compute_similarity(query, vec, metric)?;
                        candidates.push((id.clone(), sim));
                    }
                }
            }
        }

        // Re-rank and truncate
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        candidates.truncate(k);
        Ok(candidates)
    }

    async fn add(&mut self, id: String, vector: Vec<f32>) -> Result<()> {
        if !self.hyperplanes.is_empty() {
            for t in 0..self.num_tables {
                let code = self.hash_vector(t, &vector);
                self.tables[t]
                    .entry(code)
                    .or_default()
                    .push((id.clone(), vector.clone()));
            }
        }
        self.staging.insert(id, vector);
        self.stats.num_vectors = self.staging.len();
        Ok(())
    }

    async fn remove(&mut self, id: &str) -> Result<()> {
        self.staging.remove(id);
        for table in &mut self.tables {
            for bucket in table.values_mut() {
                bucket.retain(|(bid, _)| bid != id);
            }
        }
        self.stats.num_vectors = self.staging.len();
        Ok(())
    }

    fn get_stats(&self) -> IndexStats {
        self.stats.clone()
    }
}
