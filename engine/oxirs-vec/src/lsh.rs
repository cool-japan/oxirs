//! Locality Sensitive Hashing (LSH) for approximate nearest neighbor search
//!
//! This module implements various LSH families including:
//! - Random projection LSH for cosine similarity
//! - MinHash for Jaccard similarity
//! - SimHash for binary vectors
//! - Multi-probe LSH for improved recall

use crate::{Vector, VectorIndex};
use anyhow::{anyhow, Result};
use crate::random_utils::NormalSampler as Normal;
use scirs2_core::random::{Random, Rng};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Configuration for LSH index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LshConfig {
    /// Number of hash tables (L parameter)
    pub num_tables: usize,
    /// Number of hash functions per table (K parameter)
    pub num_hash_functions: usize,
    /// LSH family to use
    pub lsh_family: LshFamily,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Enable multi-probe LSH
    pub multi_probe: bool,
    /// Number of probes for multi-probe LSH
    pub num_probes: usize,
}

impl Default for LshConfig {
    fn default() -> Self {
        Self {
            num_tables: 10,
            num_hash_functions: 8,
            lsh_family: LshFamily::RandomProjection,
            seed: 42,
            multi_probe: true,
            num_probes: 3,
        }
    }
}

/// LSH family types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LshFamily {
    /// Random projection for cosine similarity
    RandomProjection,
    /// MinHash for Jaccard similarity
    MinHash,
    /// SimHash for binary similarity
    SimHash,
    /// P-stable distributions for Lp distance
    PStable(f32), // p value
}

/// Hash function trait
trait HashFunction: Send + Sync {
    /// Compute hash value for a vector
    fn hash(&self, vector: &[f32]) -> u64;

    /// Compute multiple hash values
    fn hash_multi(&self, vector: &[f32], num_hashes: usize) -> Vec<u64> {
        (0..num_hashes).map(|_| self.hash(vector)).collect()
    }
}

/// Random projection hash function for cosine similarity
struct RandomProjectionHash {
    projections: Vec<Vec<f32>>,
    dimensions: usize,
}

impl RandomProjectionHash {
    fn new(dimensions: usize, num_projections: usize, seed: u64) -> Self {
        let mut rng = Random::seed(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let mut projections = Vec::with_capacity(num_projections);
        for _ in 0..num_projections {
            let projection: Vec<f32> = (0..dimensions).map(|_| normal.sample(&mut rng)).collect();
            projections.push(projection);
        }

        Self {
            projections,
            dimensions,
        }
    }
}

impl HashFunction for RandomProjectionHash {
    fn hash(&self, vector: &[f32]) -> u64 {
        let mut hash_value = 0u64;

        for (i, projection) in self.projections.iter().enumerate() {
            // Compute dot product
            use oxirs_core::simd::SimdOps;
            let dot_product = f32::dot(vector, projection);

            // Set bit if positive
            if dot_product > 0.0 {
                hash_value |= 1 << (i % 64);
            }
        }

        hash_value
    }
}

/// MinHash for Jaccard similarity
struct MinHashFunction {
    a: Vec<u64>,
    b: Vec<u64>,
    prime: u64,
}

impl MinHashFunction {
    fn new(num_hashes: usize, seed: u64) -> Self {
        let mut rng = Random::seed(seed);
        let prime = 4294967311u64; // Large prime

        let a: Vec<u64> = (0..num_hashes).map(|_| rng.gen_range(1..prime)).collect();
        let b: Vec<u64> = (0..num_hashes).map(|_| rng.gen_range(0..prime)).collect();

        Self { a, b, prime }
    }

    fn minhash_signature(&self, set_elements: &[u32]) -> Vec<u64> {
        let mut signature = vec![u64::MAX; self.a.len()];

        for &element in set_elements {
            for (i, sig_val) in signature.iter_mut().enumerate().take(self.a.len()) {
                let hash = (self.a[i] * element as u64 + self.b[i]) % self.prime;
                *sig_val = (*sig_val).min(hash);
            }
        }

        signature
    }
}

impl HashFunction for MinHashFunction {
    fn hash(&self, vector: &[f32]) -> u64 {
        // Convert vector to set of indices where value > threshold
        let threshold = 0.0;
        let set_elements: Vec<u32> = vector
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v > threshold)
            .map(|(i, _)| i as u32)
            .collect();

        let signature = self.minhash_signature(&set_elements);

        // Combine signature into single hash
        let mut hash = 0u64;
        for (i, &sig) in signature.iter().enumerate() {
            hash ^= sig.rotate_left((i * 7) as u32);
        }

        hash
    }
}

/// SimHash for binary similarity
struct SimHashFunction {
    random_vectors: Vec<Vec<f32>>,
}

impl SimHashFunction {
    fn new(dimensions: usize, seed: u64) -> Self {
        let mut rng = Random::seed(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let random_vectors: Vec<Vec<f32>> = (0..64)
            .map(|_| (0..dimensions).map(|_| normal.sample(&mut rng)).collect())
            .collect();

        Self { random_vectors }
    }
}

impl HashFunction for SimHashFunction {
    fn hash(&self, vector: &[f32]) -> u64 {
        let mut hash = 0u64;

        for (i, random_vec) in self.random_vectors.iter().enumerate() {
            // Weighted sum
            let mut sum = 0.0;
            for (j, &v) in vector.iter().enumerate() {
                if j < random_vec.len() {
                    sum += v * random_vec[j];
                }
            }

            if sum > 0.0 {
                hash |= 1 << i;
            }
        }

        hash
    }
}

/// P-stable LSH for Lp distance
struct PStableHash {
    projections: Vec<Vec<f32>>,
    offsets: Vec<f32>,
    width: f32,
    p: f32,
}

impl PStableHash {
    fn new(dimensions: usize, num_projections: usize, width: f32, p: f32, seed: u64) -> Self {
        let mut rng = Random::seed(seed);

        // Use Cauchy distribution for L1, Normal for L2
        let projections: Vec<Vec<f32>> = if (p - 1.0).abs() < 0.1 {
            // L1 distance - use Cauchy distribution
            (0..num_projections)
                .map(|_| {
                    (0..dimensions)
                        .map(|_| {
                            let u: f32 = rng
                                .gen_range(-std::f32::consts::PI / 2.0..std::f32::consts::PI / 2.0);
                            u.tan()
                        })
                        .collect()
                })
                .collect()
        } else if (p - 2.0).abs() < 0.1 {
            // L2 distance - use Normal distribution
            let normal = Normal::new(0.0, 1.0).unwrap();
            (0..num_projections)
                .map(|_| (0..dimensions).map(|_| normal.sample(&mut rng)).collect())
                .collect()
        } else {
            // General case - approximate with Normal
            let normal = Normal::new(0.0, 1.0).unwrap();
            (0..num_projections)
                .map(|_| (0..dimensions).map(|_| normal.sample(&mut rng)).collect())
                .collect()
        };

        let offsets: Vec<f32> = (0..num_projections)
            .map(|_| rng.gen_range(0.0..width))
            .collect();

        Self {
            projections,
            offsets,
            width,
            p,
        }
    }
}

impl HashFunction for PStableHash {
    fn hash(&self, vector: &[f32]) -> u64 {
        let mut hash = 0u64;

        for (i, (projection, &offset)) in self.projections.iter().zip(&self.offsets).enumerate() {
            use oxirs_core::simd::SimdOps;
            let dot_product = f32::dot(vector, projection);
            let bucket = ((dot_product + offset) / self.width).floor() as i32;

            // Map bucket to bit position
            if bucket > 0 {
                hash |= 1 << (i % 64);
            }
        }

        hash
    }
}

/// LSH table storing hash buckets
struct LshTable {
    buckets: HashMap<u64, Vec<usize>>,
    hash_function: Box<dyn HashFunction>,
}

impl LshTable {
    fn new(hash_function: Box<dyn HashFunction>) -> Self {
        Self {
            buckets: HashMap::new(),
            hash_function,
        }
    }

    fn insert(&mut self, id: usize, vector: &[f32]) {
        let hash = self.hash_function.hash(vector);
        self.buckets.entry(hash).or_default().push(id);
    }

    fn query(&self, vector: &[f32]) -> Vec<usize> {
        let hash = self.hash_function.hash(vector);
        self.buckets.get(&hash).cloned().unwrap_or_default()
    }

    fn query_multi_probe(&self, vector: &[f32], num_probes: usize) -> Vec<usize> {
        let main_hash = self.hash_function.hash(vector);
        let mut candidates = HashSet::new();

        // Add exact match
        if let Some(ids) = self.buckets.get(&main_hash) {
            candidates.extend(ids);
        }

        // Probe nearby buckets by flipping bits
        for probe in 1..=num_probes {
            for bit in 0..64 {
                let probed_hash = main_hash ^ (1 << bit);
                if let Some(ids) = self.buckets.get(&probed_hash) {
                    candidates.extend(ids);
                }

                // Stop if we have enough probes
                if candidates.len() >= probe * 10 {
                    break;
                }
            }
        }

        candidates.into_iter().collect()
    }
}

/// LSH index implementation
pub struct LshIndex {
    config: LshConfig,
    tables: Vec<LshTable>,
    vectors: Vec<(String, Vector)>,
    uri_to_id: HashMap<String, usize>,
    dimensions: Option<usize>,
}

impl LshIndex {
    /// Create a new LSH index
    pub fn new(config: LshConfig) -> Self {
        let tables = Self::create_tables(&config, 0);

        Self {
            config,
            tables,
            vectors: Vec::new(),
            uri_to_id: HashMap::new(),
            dimensions: None,
        }
    }

    /// Create hash tables based on configuration
    fn create_tables(config: &LshConfig, dimensions: usize) -> Vec<LshTable> {
        let mut tables = Vec::with_capacity(config.num_tables);

        for table_idx in 0..config.num_tables {
            let seed = config.seed.wrapping_add(table_idx as u64);

            let hash_function: Box<dyn HashFunction> = match config.lsh_family {
                LshFamily::RandomProjection => Box::new(RandomProjectionHash::new(
                    dimensions,
                    config.num_hash_functions,
                    seed,
                )),
                LshFamily::MinHash => {
                    Box::new(MinHashFunction::new(config.num_hash_functions, seed))
                }
                LshFamily::SimHash => Box::new(SimHashFunction::new(dimensions, seed)),
                LshFamily::PStable(p) => {
                    Box::new(PStableHash::new(
                        dimensions,
                        config.num_hash_functions,
                        1.0, // Default width
                        p,
                        seed,
                    ))
                }
            };

            tables.push(LshTable::new(hash_function));
        }

        tables
    }

    /// Rebuild tables with known dimensions
    fn rebuild_tables(&mut self) {
        if let Some(dims) = self.dimensions {
            self.tables = Self::create_tables(&self.config, dims);

            // Re-insert all vectors
            for (id, (_, vector)) in self.vectors.iter().enumerate() {
                let vector_f32 = vector.as_f32();
                for table in &mut self.tables {
                    table.insert(id, &vector_f32);
                }
            }
        }
    }

    /// Query for approximate nearest neighbors
    fn query_candidates(&self, vector: &[f32]) -> Vec<usize> {
        let mut candidates = HashSet::new();

        if self.config.multi_probe {
            // Multi-probe LSH
            for table in &self.tables {
                let table_candidates = table.query_multi_probe(vector, self.config.num_probes);
                candidates.extend(table_candidates);
            }
        } else {
            // Standard LSH
            for table in &self.tables {
                let table_candidates = table.query(vector);
                candidates.extend(table_candidates);
            }
        }

        candidates.into_iter().collect()
    }

    /// Get statistics about the index
    pub fn stats(&self) -> LshStats {
        let avg_bucket_size = if self.tables.is_empty() {
            0.0
        } else {
            let total_buckets: usize = self.tables.iter().map(|t| t.buckets.len()).sum();
            let total_items: usize = self
                .tables
                .iter()
                .map(|t| t.buckets.values().map(|v| v.len()).sum::<usize>())
                .sum();

            if total_buckets > 0 {
                total_items as f64 / total_buckets as f64
            } else {
                0.0
            }
        };

        LshStats {
            num_vectors: self.vectors.len(),
            num_tables: self.tables.len(),
            avg_bucket_size,
            memory_usage: self.estimate_memory_usage(),
        }
    }

    fn estimate_memory_usage(&self) -> usize {
        let vector_memory =
            self.vectors.len() * (std::mem::size_of::<String>() + std::mem::size_of::<Vector>());

        let table_memory: usize = self
            .tables
            .iter()
            .map(|t| {
                t.buckets.len() * (std::mem::size_of::<u64>() + std::mem::size_of::<Vec<usize>>())
                    + t.buckets
                        .values()
                        .map(|v| v.len() * std::mem::size_of::<usize>())
                        .sum::<usize>()
            })
            .sum();

        vector_memory + table_memory
    }
}

impl VectorIndex for LshIndex {
    fn insert(&mut self, uri: String, vector: Vector) -> Result<()> {
        // Initialize dimensions if needed
        if self.dimensions.is_none() {
            self.dimensions = Some(vector.dimensions);
            self.rebuild_tables();
        } else if Some(vector.dimensions) != self.dimensions {
            return Err(anyhow!(
                "Vector dimensions ({}) don't match index dimensions ({:?})",
                vector.dimensions,
                self.dimensions
            ));
        }

        let id = self.vectors.len();
        let vector_f32 = vector.as_f32();

        // Insert into all tables
        for table in &mut self.tables {
            table.insert(id, &vector_f32);
        }

        self.uri_to_id.insert(uri.clone(), id);
        self.vectors.push((uri, vector));

        Ok(())
    }

    fn search_knn(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>> {
        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        let query_f32 = query.as_f32();
        let candidates = self.query_candidates(&query_f32);

        // Compute exact distances for candidates
        let mut results: Vec<(usize, f32)> = candidates
            .into_iter()
            .filter_map(|id| {
                self.vectors.get(id).map(|(_, vec)| {
                    let vec_f32 = vec.as_f32();
                    let distance = match self.config.lsh_family {
                        LshFamily::RandomProjection | LshFamily::SimHash => {
                            // Cosine distance
                            use oxirs_core::simd::SimdOps;
                            f32::cosine_distance(&query_f32, &vec_f32)
                        }
                        LshFamily::MinHash => {
                            // Jaccard distance
                            let threshold = 0.0;
                            let set1: HashSet<usize> = query_f32
                                .iter()
                                .enumerate()
                                .filter(|&(_, &v)| v > threshold)
                                .map(|(i, _)| i)
                                .collect();
                            let set2: HashSet<usize> = vec_f32
                                .iter()
                                .enumerate()
                                .filter(|&(_, &v)| v > threshold)
                                .map(|(i, _)| i)
                                .collect();

                            let intersection = set1.intersection(&set2).count();
                            let union = set1.union(&set2).count();

                            if union > 0 {
                                1.0 - (intersection as f32 / union as f32)
                            } else {
                                1.0
                            }
                        }
                        LshFamily::PStable(p) => {
                            // Lp distance
                            use oxirs_core::simd::SimdOps;
                            if (p - 1.0).abs() < 0.1 {
                                f32::manhattan_distance(&query_f32, &vec_f32)
                            } else if (p - 2.0).abs() < 0.1 {
                                f32::euclidean_distance(&query_f32, &vec_f32)
                            } else {
                                // General Minkowski distance
                                query_f32
                                    .iter()
                                    .zip(&vec_f32)
                                    .map(|(a, b)| (a - b).abs().powf(p))
                                    .sum::<f32>()
                                    .powf(1.0 / p)
                            }
                        }
                    };
                    (id, distance)
                })
            })
            .collect();

        // Sort by distance and take top k
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);

        // Convert to final result format
        Ok(results
            .into_iter()
            .map(|(id, dist)| (self.vectors[id].0.clone(), dist))
            .collect())
    }

    fn search_threshold(&self, query: &Vector, threshold: f32) -> Result<Vec<(String, f32)>> {
        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        let query_f32 = query.as_f32();
        let candidates = self.query_candidates(&query_f32);

        // Filter candidates by threshold
        let mut results: Vec<(String, f32)> = candidates
            .into_iter()
            .filter_map(|id| {
                self.vectors.get(id).and_then(|(uri, vec)| {
                    let vec_f32 = vec.as_f32();
                    let distance = match self.config.lsh_family {
                        LshFamily::RandomProjection | LshFamily::SimHash => {
                            use oxirs_core::simd::SimdOps;
                            f32::cosine_distance(&query_f32, &vec_f32)
                        }
                        LshFamily::MinHash => {
                            // Jaccard distance
                            let threshold_val = 0.0;
                            let set1: HashSet<usize> = query_f32
                                .iter()
                                .enumerate()
                                .filter(|&(_, &v)| v > threshold_val)
                                .map(|(i, _)| i)
                                .collect();
                            let set2: HashSet<usize> = vec_f32
                                .iter()
                                .enumerate()
                                .filter(|&(_, &v)| v > threshold_val)
                                .map(|(i, _)| i)
                                .collect();

                            let intersection = set1.intersection(&set2).count();
                            let union = set1.union(&set2).count();

                            if union > 0 {
                                1.0 - (intersection as f32 / union as f32)
                            } else {
                                1.0
                            }
                        }
                        LshFamily::PStable(p) => {
                            use oxirs_core::simd::SimdOps;
                            if (p - 1.0).abs() < 0.1 {
                                f32::manhattan_distance(&query_f32, &vec_f32)
                            } else if (p - 2.0).abs() < 0.1 {
                                f32::euclidean_distance(&query_f32, &vec_f32)
                            } else {
                                query_f32
                                    .iter()
                                    .zip(&vec_f32)
                                    .map(|(a, b)| (a - b).abs().powf(p))
                                    .sum::<f32>()
                                    .powf(1.0 / p)
                            }
                        }
                    };

                    if distance <= threshold {
                        Some((uri.clone(), distance))
                    } else {
                        None
                    }
                })
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(results)
    }

    fn get_vector(&self, uri: &str) -> Option<&Vector> {
        self.uri_to_id
            .get(uri)
            .and_then(|&id| self.vectors.get(id))
            .map(|(_, v)| v)
    }
}

/// LSH index statistics
#[derive(Debug, Clone)]
pub struct LshStats {
    pub num_vectors: usize,
    pub num_tables: usize,
    pub avg_bucket_size: f64,
    pub memory_usage: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_projection_lsh() {
        let config = LshConfig {
            num_tables: 5,
            num_hash_functions: 4,
            lsh_family: LshFamily::RandomProjection,
            seed: 42,
            multi_probe: false,
            num_probes: 0,
        };

        let mut index = LshIndex::new(config);

        // Insert vectors
        let v1 = Vector::new(vec![1.0, 0.0, 0.0]);
        let v2 = Vector::new(vec![0.0, 1.0, 0.0]);
        let v3 = Vector::new(vec![0.0, 0.0, 1.0]);
        let v_similar = Vector::new(vec![0.9, 0.1, 0.0]); // Similar to v1

        index.insert("v1".to_string(), v1.clone()).unwrap();
        index.insert("v2".to_string(), v2.clone()).unwrap();
        index.insert("v3".to_string(), v3.clone()).unwrap();
        index
            .insert("v_similar".to_string(), v_similar.clone())
            .unwrap();

        // Search for similar vectors
        let results = index.search_knn(&v1, 2).unwrap();

        assert!(results.len() <= 2);
        // v1 and v_similar should be the closest
        assert!(results
            .iter()
            .any(|(uri, _)| uri == "v1" || uri == "v_similar"));
    }

    #[test]
    fn test_minhash_lsh() {
        let config = LshConfig {
            num_tables: 3,
            num_hash_functions: 64,
            lsh_family: LshFamily::MinHash,
            seed: 42,
            multi_probe: false,
            num_probes: 0,
        };

        let mut index = LshIndex::new(config);

        // Create sparse binary vectors
        let mut v1 = vec![0.0; 100];
        v1[0] = 1.0;
        v1[10] = 1.0;
        v1[20] = 1.0;

        let mut v2 = vec![0.0; 100];
        v2[0] = 1.0;
        v2[10] = 1.0;
        v2[30] = 1.0; // 2/4 overlap with v1

        let mut v3 = vec![0.0; 100];
        v3[50] = 1.0;
        v3[60] = 1.0;
        v3[70] = 1.0; // No overlap with v1

        index
            .insert("v1".to_string(), Vector::new(v1.clone()))
            .unwrap();
        index.insert("v2".to_string(), Vector::new(v2)).unwrap();
        index.insert("v3".to_string(), Vector::new(v3)).unwrap();

        // Search for similar vectors
        let results = index.search_knn(&Vector::new(v1), 2).unwrap();

        // v1 should be first, v2 should be second (due to overlap)
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "v1");
        if results.len() > 1 {
            assert_eq!(results[1].0, "v2");
        }
    }

    #[test]
    fn test_multi_probe_lsh() {
        let config = LshConfig {
            num_tables: 3,
            num_hash_functions: 4,
            lsh_family: LshFamily::RandomProjection,
            seed: 42,
            multi_probe: true,
            num_probes: 2,
        };

        let mut index = LshIndex::new(config);

        // Insert many vectors
        for i in 0..50 {
            let angle = i as f32 * std::f32::consts::PI / 25.0;
            let vec = Vector::new(vec![angle.cos(), angle.sin(), 0.0]);
            index.insert(format!("v{i}"), vec).unwrap();
        }

        // Search with multi-probe should find more candidates
        let query = Vector::new(vec![1.0, 0.0, 0.0]);
        let results = index.search_knn(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
        // Results should be ordered by distance
        for i in 1..results.len() {
            assert!(results[i - 1].1 <= results[i].1);
        }
    }
}
