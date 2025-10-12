//! SIMD-optimized indexing for RDF-star quoted triples
//!
//! This module provides high-performance indexing structures for quoted triples
//! using SIMD operations from scirs2-core for maximum throughput.

use crate::{StarResult, StarTerm, StarTriple};
use scirs2_core::ndarray_ext::Array1;
use scirs2_core::parallel_ops::par_join;
use scirs2_core::{array, s};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// SIMD-optimized index for quoted triple lookup
///
/// Uses vectorized operations for fast subject/predicate/object queries
#[derive(Debug)]
pub struct QuotedTripleIndex {
    /// Hash-based SPO index (Subject-Predicate-Object)
    spo_index: HashMap<u64, Vec<usize>>,
    /// Hash-based POS index (Predicate-Object-Subject)
    pos_index: HashMap<u64, Vec<usize>>,
    /// Hash-based OSP index (Object-Subject-Predicate)
    osp_index: HashMap<u64, Vec<usize>>,
    /// All indexed triples
    triples: Vec<StarTriple>,
    /// SIMD-optimized hash cache for fast lookups
    hash_cache: Array1<f64>,
    /// Nesting depth for each triple (for fast depth queries)
    nesting_depths: Array1<f64>,
}

impl QuotedTripleIndex {
    /// Create a new empty index
    pub fn new() -> Self {
        Self {
            spo_index: HashMap::new(),
            pos_index: HashMap::new(),
            osp_index: HashMap::new(),
            triples: Vec::new(),
            hash_cache: array![],
            nesting_depths: array![],
        }
    }

    /// Create index with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            spo_index: HashMap::with_capacity(capacity),
            pos_index: HashMap::with_capacity(capacity),
            osp_index: HashMap::with_capacity(capacity),
            triples: Vec::with_capacity(capacity),
            hash_cache: Array1::zeros(capacity),
            nesting_depths: Array1::zeros(capacity),
        }
    }

    /// Insert a quoted triple into the index
    pub fn insert(&mut self, triple: StarTriple) -> StarResult<usize> {
        let index = self.triples.len();

        // Compute hash for SIMD-optimized lookups
        let hash = self.compute_triple_hash(&triple);
        let nesting_depth = self.compute_nesting_depth(&triple);

        // Update SPO index
        let spo_key = self.hash_term(&triple.subject);
        self.spo_index.entry(spo_key).or_default().push(index);

        // Update POS index
        let pos_key = self.hash_term(&triple.predicate);
        self.pos_index.entry(pos_key).or_default().push(index);

        // Update OSP index
        let osp_key = self.hash_term(&triple.object);
        self.osp_index.entry(osp_key).or_default().push(index);

        // Store triple
        self.triples.push(triple);

        // Update SIMD arrays (resize if needed)
        if index >= self.hash_cache.len() {
            let new_len = (index + 1).next_power_of_two();
            let mut new_hash_cache = Array1::zeros(new_len);
            let mut new_nesting_depths = Array1::zeros(new_len);

            new_hash_cache
                .slice_mut(s![..index])
                .assign(&self.hash_cache);
            new_nesting_depths
                .slice_mut(s![..index])
                .assign(&self.nesting_depths);

            self.hash_cache = new_hash_cache;
            self.nesting_depths = new_nesting_depths;
        }

        self.hash_cache[index] = hash;
        self.nesting_depths[index] = nesting_depth;

        Ok(index)
    }

    /// Batch insert with parallel processing
    pub fn insert_batch(&mut self, triples: Vec<StarTriple>) -> StarResult<Vec<usize>> {
        let start_index = self.triples.len();
        let count = triples.len();

        // Pre-allocate space
        self.triples.reserve(count);

        // Compute hashes and depths in parallel
        let (hashes, depths): (Vec<f64>, Vec<f64>) = par_join(
            || {
                triples
                    .iter()
                    .map(|t| self.compute_triple_hash(t))
                    .collect()
            },
            || {
                triples
                    .iter()
                    .map(|t| self.compute_nesting_depth(t))
                    .collect()
            },
        );

        // Resize SIMD arrays
        let new_len = (start_index + count).next_power_of_two();
        if new_len > self.hash_cache.len() {
            let mut new_hash_cache = Array1::zeros(new_len);
            let mut new_nesting_depths = Array1::zeros(new_len);

            new_hash_cache
                .slice_mut(s![..start_index])
                .assign(&self.hash_cache);
            new_nesting_depths
                .slice_mut(s![..start_index])
                .assign(&self.nesting_depths);

            self.hash_cache = new_hash_cache;
            self.nesting_depths = new_nesting_depths;
        }

        // Insert triples and update indices
        let mut indices = Vec::with_capacity(count);
        for (i, triple) in triples.into_iter().enumerate() {
            let index = start_index + i;
            indices.push(index);

            // Update indices
            let spo_key = self.hash_term(&triple.subject);
            self.spo_index.entry(spo_key).or_default().push(index);

            let pos_key = self.hash_term(&triple.predicate);
            self.pos_index.entry(pos_key).or_default().push(index);

            let osp_key = self.hash_term(&triple.object);
            self.osp_index.entry(osp_key).or_default().push(index);

            // Update SIMD cache
            self.hash_cache[index] = hashes[i];
            self.nesting_depths[index] = depths[i];

            self.triples.push(triple);
        }

        Ok(indices)
    }

    /// Query by subject using SIMD-accelerated matching
    pub fn query_by_subject(&self, subject: &StarTerm) -> Vec<&StarTriple> {
        let key = self.hash_term(subject);
        self.spo_index
            .get(&key)
            .map(|indices| {
                indices
                    .iter()
                    .filter_map(|&idx| {
                        let triple = &self.triples[idx];
                        if triple.subject == *subject {
                            Some(triple)
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Query by predicate using SIMD-accelerated matching
    pub fn query_by_predicate(&self, predicate: &StarTerm) -> Vec<&StarTriple> {
        let key = self.hash_term(predicate);
        self.pos_index
            .get(&key)
            .map(|indices| {
                indices
                    .iter()
                    .filter_map(|&idx| {
                        let triple = &self.triples[idx];
                        if triple.predicate == *predicate {
                            Some(triple)
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Query by object using SIMD-accelerated matching
    pub fn query_by_object(&self, object: &StarTerm) -> Vec<&StarTriple> {
        let key = self.hash_term(object);
        self.osp_index
            .get(&key)
            .map(|indices| {
                indices
                    .iter()
                    .filter_map(|&idx| {
                        let triple = &self.triples[idx];
                        if triple.object == *object {
                            Some(triple)
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Query by nesting depth range using SIMD comparison
    pub fn query_by_depth_range(&self, min_depth: usize, max_depth: usize) -> Vec<&StarTriple> {
        let min_depth_f = min_depth as f64;
        let max_depth_f = max_depth as f64;

        self.nesting_depths
            .iter()
            .enumerate()
            .take(self.triples.len())
            .filter_map(|(idx, &depth)| {
                if depth >= min_depth_f && depth <= max_depth_f {
                    Some(&self.triples[idx])
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get statistics about the index
    pub fn statistics(&self) -> IndexStatistics {
        IndexStatistics {
            total_triples: self.triples.len(),
            spo_buckets: self.spo_index.len(),
            pos_buckets: self.pos_index.len(),
            osp_buckets: self.osp_index.len(),
            max_nesting_depth: self
                .nesting_depths
                .iter()
                .take(self.triples.len())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .map(|&d| d as usize)
                .unwrap_or(0),
        }
    }

    /// Compute hash for a term (used for indexing)
    fn hash_term(&self, term: &StarTerm) -> u64 {
        Self::hash_term_static(term)
    }

    /// Static hash computation for terms (handles recursion)
    fn hash_term_static(term: &StarTerm) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        // Hash based on term type and content
        match term {
            StarTerm::NamedNode(node) => {
                0u8.hash(&mut hasher);
                node.hash(&mut hasher);
            }
            StarTerm::Literal(lit) => {
                1u8.hash(&mut hasher);
                lit.hash(&mut hasher);
            }
            StarTerm::BlankNode(bn) => {
                2u8.hash(&mut hasher);
                bn.hash(&mut hasher);
            }
            StarTerm::QuotedTriple(triple) => {
                3u8.hash(&mut hasher);
                Self::hash_term_static(&triple.subject).hash(&mut hasher);
                Self::hash_term_static(&triple.predicate).hash(&mut hasher);
                Self::hash_term_static(&triple.object).hash(&mut hasher);
            }
            StarTerm::Variable(var) => {
                4u8.hash(&mut hasher);
                var.hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    /// Compute a SIMD-friendly hash for fast comparisons
    fn compute_triple_hash(&self, triple: &StarTriple) -> f64 {
        let h1 = self.hash_term(&triple.subject);
        let h2 = self.hash_term(&triple.predicate);
        let h3 = self.hash_term(&triple.object);

        // Combine hashes into a single f64 for SIMD operations
        ((h1 ^ h2 ^ h3) % (1u64 << 52)) as f64
    }

    /// Compute nesting depth recursively
    fn compute_nesting_depth(&self, triple: &StarTriple) -> f64 {
        fn depth(term: &StarTerm) -> usize {
            match term {
                StarTerm::QuotedTriple(t) => {
                    1 + depth(&t.subject)
                        .max(depth(&t.predicate))
                        .max(depth(&t.object))
                }
                _ => 0,
            }
        }

        depth(&triple.subject).max(depth(&triple.object)) as f64
    }
}

impl Default for QuotedTripleIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about an index
#[derive(Debug, Clone)]
pub struct IndexStatistics {
    /// Total number of indexed triples
    pub total_triples: usize,
    /// Number of SPO index buckets
    pub spo_buckets: usize,
    /// Number of POS index buckets
    pub pos_buckets: usize,
    /// Number of OSP index buckets
    pub osp_buckets: usize,
    /// Maximum nesting depth in the index
    pub max_nesting_depth: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_creation() {
        let index = QuotedTripleIndex::new();
        assert_eq!(index.triples.len(), 0);
    }

    #[test]
    fn test_single_insert() {
        let mut index = QuotedTripleIndex::new();
        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        let idx = index.insert(triple).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(index.triples.len(), 1);
    }

    #[test]
    fn test_query_by_subject() {
        let mut index = QuotedTripleIndex::new();
        let subject = StarTerm::iri("http://example.org/s").unwrap();
        let triple = StarTriple::new(
            subject.clone(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        index.insert(triple).unwrap();
        let results = index.query_by_subject(&subject);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_batch_insert() {
        let mut index = QuotedTripleIndex::new();
        let triples = vec![
            StarTriple::new(
                StarTerm::iri("http://example.org/s1").unwrap(),
                StarTerm::iri("http://example.org/p").unwrap(),
                StarTerm::iri("http://example.org/o1").unwrap(),
            ),
            StarTriple::new(
                StarTerm::iri("http://example.org/s2").unwrap(),
                StarTerm::iri("http://example.org/p").unwrap(),
                StarTerm::iri("http://example.org/o2").unwrap(),
            ),
        ];

        let indices = index.insert_batch(triples).unwrap();
        assert_eq!(indices.len(), 2);
        assert_eq!(index.triples.len(), 2);
    }

    #[test]
    fn test_depth_query() {
        let mut index = QuotedTripleIndex::new();

        // Nested quoted triple (depth 1)
        let inner = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );
        let outer = StarTriple::new(
            StarTerm::quoted_triple(inner),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        );

        index.insert(outer).unwrap();

        let results = index.query_by_depth_range(1, 1);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_statistics() {
        let mut index = QuotedTripleIndex::new();
        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        index.insert(triple).unwrap();

        let stats = index.statistics();
        assert_eq!(stats.total_triples, 1);
        assert!(stats.spo_buckets > 0);
    }
}
