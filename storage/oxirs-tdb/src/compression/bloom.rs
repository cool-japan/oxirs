use crate::error::Result;
use oxicode::Decode;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

/// Bloom filter for fast existence checks
///
/// Provides probabilistic membership testing with configurable
/// false positive rate and no false negatives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomFilter {
    /// Bit array
    bits: Vec<bool>,
    /// Number of hash functions
    num_hashes: usize,
    /// Size of bit array
    size: usize,
    /// Number of elements inserted
    count: usize,
}

impl BloomFilter {
    /// Create a new Bloom filter
    ///
    /// # Arguments
    /// * `expected_elements` - Expected number of elements
    /// * `false_positive_rate` - Desired false positive rate (e.g., 0.01 for 1%)
    pub fn new(expected_elements: usize, false_positive_rate: f64) -> Self {
        let size = Self::optimal_size(expected_elements, false_positive_rate);
        let num_hashes = Self::optimal_hash_count(expected_elements, size);

        Self {
            bits: vec![false; size],
            num_hashes,
            size,
            count: 0,
        }
    }

    /// Insert an element
    pub fn insert<T: Hash>(&mut self, item: &T) {
        for i in 0..self.num_hashes {
            let hash = self.hash(item, i);
            let index = (hash % self.size as u64) as usize;
            self.bits[index] = true;
        }
        self.count += 1;
    }

    /// Check if an element might exist (no false negatives)
    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        for i in 0..self.num_hashes {
            let hash = self.hash(item, i);
            let index = (hash % self.size as u64) as usize;
            if !self.bits[index] {
                return false;
            }
        }
        true
    }

    /// Insert a byte slice
    pub fn insert_bytes(&mut self, bytes: &[u8]) {
        self.insert(&bytes)
    }

    /// Check if a byte slice might exist
    pub fn contains_bytes(&self, bytes: &[u8]) -> bool {
        self.contains(&bytes)
    }

    /// Clear the filter
    pub fn clear(&mut self) {
        self.bits.fill(false);
        self.count = 0;
    }

    /// Get current false positive rate estimate
    pub fn false_positive_rate(&self) -> f64 {
        let fraction_set = self.bits.iter().filter(|&&b| b).count() as f64 / self.size as f64;
        fraction_set.powi(self.num_hashes as i32)
    }

    /// Get number of elements inserted
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Calculate optimal bit array size
    fn optimal_size(n: usize, p: f64) -> usize {
        let size = -(n as f64 * p.ln()) / (2f64.ln().powi(2));
        size.ceil() as usize
    }

    /// Calculate optimal number of hash functions
    fn optimal_hash_count(n: usize, m: usize) -> usize {
        let k = (m as f64 / n as f64) * 2f64.ln();
        k.ceil().max(1.0) as usize
    }

    /// Compute hash for item with seed
    fn hash<T: Hash>(&self, item: &T, seed: usize) -> u64 {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        item.hash(&mut hasher);
        hasher.finish()
    }

    /// Get statistics
    pub fn stats(&self) -> BloomFilterStats {
        let bits_set = self.bits.iter().filter(|&&b| b).count();

        BloomFilterStats {
            size: self.size,
            num_hashes: self.num_hashes,
            count: self.count,
            bits_set,
            load_factor: bits_set as f64 / self.size as f64,
            estimated_fpr: self.false_positive_rate(),
        }
    }
}

/// Bloom filter statistics
#[derive(Debug, Clone, Copy)]
pub struct BloomFilterStats {
    /// Total size of bit array
    pub size: usize,
    /// Number of hash functions
    pub num_hashes: usize,
    /// Number of elements inserted
    pub count: usize,
    /// Number of bits set to true
    pub bits_set: usize,
    /// Fraction of bits set (0.0 to 1.0)
    pub load_factor: f64,
    /// Estimated false positive rate
    pub estimated_fpr: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_filter_basic() {
        let mut filter = BloomFilter::new(1000, 0.01);

        filter.insert(&"hello");
        filter.insert(&"world");
        filter.insert(&123);

        assert!(filter.contains(&"hello"));
        assert!(filter.contains(&"world"));
        assert!(filter.contains(&123));
        assert!(!filter.contains(&"not_inserted"));
    }

    #[test]
    fn test_bloom_filter_clear() {
        let mut filter = BloomFilter::new(1000, 0.01);

        filter.insert(&"hello");
        assert!(filter.contains(&"hello"));

        filter.clear();
        assert!(!filter.contains(&"hello"));
        assert_eq!(filter.len(), 0);
    }

    #[test]
    fn test_bloom_filter_false_positives() {
        let mut filter = BloomFilter::new(100, 0.01);

        // Insert 50 elements
        for i in 0..50 {
            filter.insert(&i);
        }

        // Check false positive rate
        let mut false_positives = 0;
        let test_count = 1000;

        for i in 100..(100 + test_count) {
            if filter.contains(&i) {
                false_positives += 1;
            }
        }

        let fpr = false_positives as f64 / test_count as f64;
        // Should be close to 1% (allowing some variance)
        assert!(fpr < 0.05, "False positive rate too high: {}", fpr);
    }

    #[test]
    fn test_bloom_filter_optimal_size() {
        let size = BloomFilter::optimal_size(1000, 0.01);
        assert!(size > 0);
    }

    #[test]
    fn test_bloom_filter_optimal_hash_count() {
        let size = BloomFilter::optimal_size(1000, 0.01);
        let hash_count = BloomFilter::optimal_hash_count(1000, size);
        assert!(hash_count > 0);
    }

    #[test]
    fn test_bloom_filter_stats() {
        let mut filter = BloomFilter::new(1000, 0.01);

        for i in 0..100 {
            filter.insert(&i);
        }

        let stats = filter.stats();
        assert_eq!(stats.count, 100);
        assert!(stats.bits_set > 0);
        assert!(stats.load_factor > 0.0);
        assert!(stats.load_factor < 1.0);
    }

    #[test]
    fn test_bloom_filter_no_false_negatives() {
        let mut filter = BloomFilter::new(100, 0.01);

        let elements: Vec<i32> = (0..100).collect();

        for elem in &elements {
            filter.insert(elem);
        }

        // Check all inserted elements are found (no false negatives)
        for elem in &elements {
            assert!(filter.contains(elem), "False negative for {}", elem);
        }
    }

    #[test]
    fn test_bloom_filter_strings() {
        let mut filter = BloomFilter::new(1000, 0.01);

        let iris = vec![
            "http://example.org/Person",
            "http://example.org/name",
            "http://example.org/age",
        ];

        for iri in &iris {
            filter.insert(iri);
        }

        for iri in &iris {
            assert!(filter.contains(iri));
        }

        assert!(!filter.contains(&"http://example.org/unknown"));
    }

    #[test]
    fn test_bloom_filter_is_empty() {
        let mut filter = BloomFilter::new(1000, 0.01);
        assert!(filter.is_empty());

        filter.insert(&1);
        assert!(!filter.is_empty());

        filter.clear();
        assert!(filter.is_empty());
    }

    #[test]
    fn test_bloom_filter_serialization() {
        let mut filter = BloomFilter::new(1000, 0.01);
        filter.insert(&"test");

        let serialized =
            oxicode::serde::encode_to_vec(&filter, oxicode::config::standard()).unwrap();
        let deserialized: BloomFilter =
            oxicode::serde::decode_from_slice(&serialized, oxicode::config::standard())
                .unwrap()
                .0;

        assert!(deserialized.contains(&"test"));
        assert_eq!(filter.len(), deserialized.len());
    }
}
