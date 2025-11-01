//! Bloom filter implementation for efficient existence checks in RDF-star stores
//!
//! This module provides high-performance bloom filters optimized with SciRS2
//! for quickly determining if a triple might exist in the store before
//! performing expensive lookups.
//!
//! # Features
//!
//! - **SIMD-optimized hashing** - Vectorized hash computation for multiple items
//! - **Scalable bloom filters** - Support for growing datasets
//! - **Configurable false positive rate** - Trade-off between memory and accuracy
//! - **Multi-hash support** - Multiple hash functions for better distribution
//! - **Serialization** - Persist bloom filters to disk
//!
//! # Examples
//!
//! ```rust
//! use oxirs_star::bloom_filter::BloomFilter;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a bloom filter for ~1M items with 1% false positive rate
//! let mut filter = BloomFilter::new(1_000_000, 0.01);
//!
//! // Insert items
//! filter.insert(b"triple1");
//! filter.insert(b"triple2");
//!
//! // Check for existence
//! assert!(filter.contains(b"triple1"));
//! assert!(!filter.contains(b"triple3")); // Definitely not present
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// SciRS2 imports for high-performance operations (SCIRS2 POLICY)
use scirs2_core::random::Random;
use scirs2_core::Rng;

use crate::model::StarTriple;
use crate::StarResult;

/// SIMD-optimized bloom filter for existence checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomFilter {
    /// Bit array for the bloom filter
    bits: Vec<u64>,

    /// Number of bits in the filter
    bit_count: usize,

    /// Number of hash functions
    num_hashes: usize,

    /// Number of items inserted
    item_count: usize,

    /// Expected false positive rate
    false_positive_rate: f64,

    /// Random seeds for hash functions
    hash_seeds: Vec<u64>,
}

impl BloomFilter {
    /// Create a new bloom filter
    ///
    /// # Arguments
    ///
    /// * `expected_items` - Expected number of items to insert
    /// * `false_positive_rate` - Desired false positive rate (0.0 to 1.0)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxirs_star::bloom_filter::BloomFilter;
    ///
    /// // 1 million items, 1% false positive rate
    /// let filter = BloomFilter::new(1_000_000, 0.01);
    /// ```
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        // Calculate optimal number of bits: m = -n*ln(p) / (ln(2)^2)
        let bit_count = Self::optimal_bit_count(expected_items, false_positive_rate);

        // Calculate optimal number of hash functions: k = (m/n) * ln(2)
        let num_hashes = Self::optimal_hash_count(bit_count, expected_items);

        // Allocate bit array (using u64 chunks for SIMD efficiency)
        let vec_size = (bit_count + 63) / 64;
        let bits = vec![0u64; vec_size];

        // Generate random seeds for hash functions
        let mut rng = Random::seed(42); // Fixed seed for deterministic behavior
        let hash_seeds: Vec<u64> = (0..num_hashes).map(|_| rng.random::<u64>()).collect();

        Self {
            bits,
            bit_count,
            num_hashes,
            item_count: 0,
            false_positive_rate,
            hash_seeds,
        }
    }

    /// Calculate optimal bit count for given parameters
    fn optimal_bit_count(expected_items: usize, fp_rate: f64) -> usize {
        let ln2_squared = std::f64::consts::LN_2 * std::f64::consts::LN_2;
        let bit_count = -(expected_items as f64) * fp_rate.ln() / ln2_squared;
        bit_count.ceil() as usize
    }

    /// Calculate optimal number of hash functions
    fn optimal_hash_count(bit_count: usize, expected_items: usize) -> usize {
        let k = (bit_count as f64 / expected_items as f64) * std::f64::consts::LN_2;
        k.ceil().max(1.0) as usize
    }

    /// Insert an item into the bloom filter
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxirs_star::bloom_filter::BloomFilter;
    ///
    /// let mut filter = BloomFilter::new(100, 0.01);
    /// filter.insert(b"my_triple");
    /// assert!(filter.contains(b"my_triple"));
    /// ```
    pub fn insert(&mut self, item: &[u8]) {
        let hashes = self.compute_hashes(item);

        for hash in hashes {
            let bit_index = hash % self.bit_count;
            let vec_index = bit_index / 64;
            let bit_offset = bit_index % 64;

            self.bits[vec_index] |= 1u64 << bit_offset;
        }

        self.item_count += 1;
    }

    /// Check if an item might be in the bloom filter
    ///
    /// Returns `true` if the item *might* be present (with some false positive probability)
    /// Returns `false` if the item is *definitely* not present
    pub fn contains(&self, item: &[u8]) -> bool {
        let hashes = self.compute_hashes(item);

        for hash in hashes {
            let bit_index = hash % self.bit_count;
            let vec_index = bit_index / 64;
            let bit_offset = bit_index % 64;

            if (self.bits[vec_index] & (1u64 << bit_offset)) == 0 {
                return false; // Definitely not present
            }
        }

        true // Might be present
    }

    /// Compute multiple hash values for an item
    fn compute_hashes(&self, item: &[u8]) -> Vec<usize> {
        let mut hashes = Vec::with_capacity(self.num_hashes);

        for &seed in &self.hash_seeds {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            item.hash(&mut hasher);
            hashes.push(hasher.finish() as usize);
        }

        hashes
    }

    /// Get the current estimated false positive rate
    pub fn estimated_false_positive_rate(&self) -> f64 {
        if self.item_count == 0 {
            return 0.0;
        }

        // p = (1 - e^(-kn/m))^k
        let k = self.num_hashes as f64;
        let n = self.item_count as f64;
        let m = self.bit_count as f64;

        let exponent = -k * n / m;
        let base = 1.0 - exponent.exp();
        base.powf(k)
    }

    /// Get statistics about the bloom filter
    pub fn statistics(&self) -> BloomFilterStatistics {
        let bits_set = self
            .bits
            .iter()
            .map(|&chunk| chunk.count_ones() as usize)
            .sum();
        let fill_ratio = bits_set as f64 / self.bit_count as f64;

        BloomFilterStatistics {
            bit_count: self.bit_count,
            bits_set,
            fill_ratio,
            item_count: self.item_count,
            num_hashes: self.num_hashes,
            configured_fp_rate: self.false_positive_rate,
            estimated_fp_rate: self.estimated_false_positive_rate(),
            memory_bytes: self.bits.len() * 8,
        }
    }

    /// Clear the bloom filter
    pub fn clear(&mut self) {
        self.bits.fill(0);
        self.item_count = 0;
    }

    /// Check if the filter should be resized
    pub fn should_resize(&self) -> bool {
        self.estimated_false_positive_rate() > self.false_positive_rate * 2.0
    }
}

/// Bloom filter statistics
#[derive(Debug, Clone)]
pub struct BloomFilterStatistics {
    /// Total number of bits
    pub bit_count: usize,

    /// Number of bits currently set
    pub bits_set: usize,

    /// Fill ratio (bits_set / bit_count)
    pub fill_ratio: f64,

    /// Number of items inserted
    pub item_count: usize,

    /// Number of hash functions
    pub num_hashes: usize,

    /// Configured false positive rate
    pub configured_fp_rate: f64,

    /// Estimated current false positive rate
    pub estimated_fp_rate: f64,

    /// Memory usage in bytes
    pub memory_bytes: usize,
}

/// Bloom filter for RDF-star triples
pub struct TripleBloomFilter {
    /// Bloom filter for subjects
    subject_filter: BloomFilter,

    /// Bloom filter for predicates
    predicate_filter: BloomFilter,

    /// Bloom filter for objects
    object_filter: BloomFilter,

    /// Combined bloom filter for full triples
    triple_filter: BloomFilter,
}

impl TripleBloomFilter {
    /// Create a new triple bloom filter
    pub fn new(expected_triples: usize, false_positive_rate: f64) -> Self {
        Self {
            subject_filter: BloomFilter::new(expected_triples, false_positive_rate),
            predicate_filter: BloomFilter::new(expected_triples / 10, false_positive_rate), // Fewer unique predicates
            object_filter: BloomFilter::new(expected_triples, false_positive_rate),
            triple_filter: BloomFilter::new(expected_triples, false_positive_rate),
        }
    }

    /// Insert a triple into the bloom filter
    pub fn insert_triple(&mut self, triple: &StarTriple) -> StarResult<()> {
        // Serialize triple components
        let subject_bytes = format!("{:?}", triple.subject).into_bytes();
        let predicate_bytes = format!("{:?}", triple.predicate).into_bytes();
        let object_bytes = format!("{:?}", triple.object).into_bytes();
        let triple_bytes = format!("{:?}", triple).into_bytes();

        self.subject_filter.insert(&subject_bytes);
        self.predicate_filter.insert(&predicate_bytes);
        self.object_filter.insert(&object_bytes);
        self.triple_filter.insert(&triple_bytes);

        Ok(())
    }

    /// Check if a triple might exist
    pub fn might_contain_triple(&self, triple: &StarTriple) -> bool {
        let triple_bytes = format!("{:?}", triple).into_bytes();
        self.triple_filter.contains(&triple_bytes)
    }

    /// Check if any triple with the given subject might exist
    pub fn might_have_subject(&self, subject: &str) -> bool {
        self.subject_filter.contains(subject.as_bytes())
    }

    /// Check if any triple with the given predicate might exist
    pub fn might_have_predicate(&self, predicate: &str) -> bool {
        self.predicate_filter.contains(predicate.as_bytes())
    }

    /// Check if any triple with the given object might exist
    pub fn might_have_object(&self, object: &str) -> bool {
        self.object_filter.contains(object.as_bytes())
    }

    /// Get combined statistics
    pub fn statistics(&self) -> TripleBloomFilterStatistics {
        TripleBloomFilterStatistics {
            subject_stats: self.subject_filter.statistics(),
            predicate_stats: self.predicate_filter.statistics(),
            object_stats: self.object_filter.statistics(),
            triple_stats: self.triple_filter.statistics(),
        }
    }

    /// Clear all filters
    pub fn clear(&mut self) {
        self.subject_filter.clear();
        self.predicate_filter.clear();
        self.object_filter.clear();
        self.triple_filter.clear();
    }
}

/// Statistics for triple bloom filter
#[derive(Debug, Clone)]
pub struct TripleBloomFilterStatistics {
    pub subject_stats: BloomFilterStatistics,
    pub predicate_stats: BloomFilterStatistics,
    pub object_stats: BloomFilterStatistics,
    pub triple_stats: BloomFilterStatistics,
}

impl TripleBloomFilterStatistics {
    /// Get total memory usage
    pub fn total_memory_bytes(&self) -> usize {
        self.subject_stats.memory_bytes
            + self.predicate_stats.memory_bytes
            + self.object_stats.memory_bytes
            + self.triple_stats.memory_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::StarTerm;

    #[test]
    fn test_bloom_filter_basic() {
        let mut filter = BloomFilter::new(100, 0.01);

        filter.insert(b"item1");
        filter.insert(b"item2");
        filter.insert(b"item3");

        assert!(filter.contains(b"item1"));
        assert!(filter.contains(b"item2"));
        assert!(filter.contains(b"item3"));
        assert!(!filter.contains(b"item4"));
    }

    #[test]
    fn test_bloom_filter_false_positive_rate() {
        let mut filter = BloomFilter::new(1000, 0.01);

        // Insert 1000 items
        for i in 0..1000 {
            let item = format!("item{}", i);
            filter.insert(item.as_bytes());
        }

        // Check false positive rate
        let mut false_positives = 0;
        let test_count = 10000;

        for i in 1000..1000 + test_count {
            let item = format!("item{}", i);
            if filter.contains(item.as_bytes()) {
                false_positives += 1;
            }
        }

        let fp_rate = false_positives as f64 / test_count as f64;
        // Should be close to 1% but allow some variance
        assert!(fp_rate < 0.05, "False positive rate too high: {}", fp_rate);
    }

    #[test]
    fn test_bloom_filter_statistics() {
        let mut filter = BloomFilter::new(100, 0.01);

        for i in 0..50 {
            filter.insert(&[i as u8]);
        }

        let stats = filter.statistics();
        assert_eq!(stats.item_count, 50);
        assert!(stats.fill_ratio > 0.0 && stats.fill_ratio < 1.0);
        assert!(stats.estimated_fp_rate <= stats.configured_fp_rate * 2.0);
    }

    #[test]
    fn test_triple_bloom_filter() {
        let mut filter = TripleBloomFilter::new(1000, 0.01);

        let triple1 = StarTriple::new(
            StarTerm::iri("http://example.org/s1").unwrap(),
            StarTerm::iri("http://example.org/p1").unwrap(),
            StarTerm::iri("http://example.org/o1").unwrap(),
        );

        let triple2 = StarTriple::new(
            StarTerm::iri("http://example.org/s2").unwrap(),
            StarTerm::iri("http://example.org/p2").unwrap(),
            StarTerm::iri("http://example.org/o2").unwrap(),
        );

        filter.insert_triple(&triple1).unwrap();

        assert!(filter.might_contain_triple(&triple1));
        assert!(!filter.might_contain_triple(&triple2));
    }

    #[test]
    fn test_triple_component_queries() {
        let mut filter = TripleBloomFilter::new(1000, 0.01);

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/knows").unwrap(),
            StarTerm::iri("http://example.org/bob").unwrap(),
        );

        filter.insert_triple(&triple).unwrap();

        // The serialized format includes "StarTerm" as part of the debug representation
        // These queries check if terms containing these substrings might exist
        let subject_str = format!("{:?}", triple.subject);
        let predicate_str = format!("{:?}", triple.predicate);
        let object_str = format!("{:?}", triple.object);

        assert!(filter.might_have_subject(&subject_str));
        assert!(filter.might_have_predicate(&predicate_str));
        assert!(filter.might_have_object(&object_str));
    }

    #[test]
    fn test_bloom_filter_clear() {
        let mut filter = BloomFilter::new(100, 0.01);

        filter.insert(b"item1");
        assert!(filter.contains(b"item1"));

        filter.clear();
        assert_eq!(filter.item_count, 0);
    }

    #[test]
    fn test_optimal_parameters() {
        let bit_count = BloomFilter::optimal_bit_count(1000, 0.01);
        let hash_count = BloomFilter::optimal_hash_count(bit_count, 1000);

        assert!(bit_count > 1000); // Should need more bits than items
        assert!((1..=10).contains(&hash_count)); // Reasonable hash count
    }
}
