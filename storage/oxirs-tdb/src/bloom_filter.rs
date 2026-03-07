//! Space-efficient Bloom filter for triple existence checks.
//!
//! Implements a classic Bloom filter with FNV-1a / DJB2 double hashing
//! to provide fast probabilistic set membership tests for RDF triples.

use std::fmt;

// FNV-1a constants
const FNV_OFFSET: u64 = 14695981039346656037;
const FNV_PRIME: u64 = 1099511628211;

/// Compute the FNV-1a hash of a byte slice.
fn fnv1a(data: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Compute the DJB2 hash of a byte slice.
fn djb2(data: &[u8]) -> u64 {
    let mut hash: u64 = 5381;
    for &byte in data {
        hash = hash
            .wrapping_shl(5)
            .wrapping_add(hash)
            .wrapping_add(byte as u64);
    }
    hash
}

/// Error type for Bloom filter operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BloomError {
    /// Cannot merge two filters with different dimensions.
    DimensionMismatch {
        /// Expected bit count.
        expected: u64,
        /// Actual bit count of the other filter.
        got: u64,
    },
}

impl fmt::Display for BloomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BloomError::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "Bloom filter dimension mismatch: expected {expected} bits, got {got}"
                )
            }
        }
    }
}

impl std::error::Error for BloomError {}

/// A space-efficient probabilistic data structure for membership testing.
///
/// Provides O(k) insert and lookup where k is the number of hash functions.
/// False negatives are impossible; the false-positive rate can be tuned via
/// `expected_items` and `false_positive_rate`.
#[derive(Debug, Clone, PartialEq)]
pub struct BloomFilter {
    /// Bit storage: each `u64` holds 64 bits.
    bits: Vec<u64>,
    /// Number of hash functions to apply per item.
    num_hashes: u32,
    /// Total number of bits in the filter.
    bit_count: u64,
    /// Number of items inserted so far.
    item_count: u64,
}

impl BloomFilter {
    /// Create a Bloom filter optimised for `expected_items` items and the given
    /// false-positive rate (0 < `false_positive_rate` < 1).
    ///
    /// Optimal parameters are computed using the standard formulas:
    /// - `m = -(n * ln(p)) / (ln(2))^2`
    /// - `k = (m/n) * ln(2)`
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        let n = expected_items.max(1) as f64;
        // Clamp to valid range
        let p = false_positive_rate.clamp(1e-15, 1.0 - 1e-15);

        let ln2 = std::f64::consts::LN_2;
        let bit_count_f = -(n * p.ln()) / (ln2 * ln2);
        let bit_count = (bit_count_f.ceil() as u64).max(64);

        let num_hashes_f = (bit_count as f64 / n) * ln2;
        let num_hashes = (num_hashes_f.round() as u32).max(1);

        let word_count = ((bit_count + 63) / 64) as usize;
        Self {
            bits: vec![0u64; word_count],
            num_hashes,
            bit_count,
            item_count: 0,
        }
    }

    /// Create a Bloom filter with explicit parameters (for testing / restore).
    pub fn with_params(bit_count: u64, num_hashes: u32) -> Self {
        let word_count = ((bit_count + 63) / 64) as usize;
        Self {
            bits: vec![0u64; word_count.max(1)],
            num_hashes: num_hashes.max(1),
            bit_count: bit_count.max(64),
            item_count: 0,
        }
    }

    // ── Bit manipulation helpers ───────────────────────────────────────────────

    fn set_bit(&mut self, index: u64) {
        let word = (index / 64) as usize;
        let bit = index % 64;
        if word < self.bits.len() {
            self.bits[word] |= 1u64 << bit;
        }
    }

    fn get_bit(&self, index: u64) -> bool {
        let word = (index / 64) as usize;
        let bit = index % 64;
        if word < self.bits.len() {
            (self.bits[word] >> bit) & 1 == 1
        } else {
            false
        }
    }

    fn bit_index_for(&self, h1: u64, h2: u64, i: u32) -> u64 {
        h1.wrapping_add((i as u64).wrapping_mul(h2)) % self.bit_count
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Insert `item` into the filter.
    pub fn insert(&mut self, item: &str) {
        let bytes = item.as_bytes();
        let h1 = fnv1a(bytes);
        let h2 = djb2(bytes).max(1); // ensure h2 != 0 to avoid degenerate case
        for i in 0..self.num_hashes {
            let idx = self.bit_index_for(h1, h2, i);
            self.set_bit(idx);
        }
        self.item_count += 1;
    }

    /// Test whether `item` is probably in the filter.
    ///
    /// Returns `false` with certainty if the item was never inserted.
    /// May return `true` for items that were not inserted (false positive).
    pub fn contains(&self, item: &str) -> bool {
        let bytes = item.as_bytes();
        let h1 = fnv1a(bytes);
        let h2 = djb2(bytes).max(1);
        for i in 0..self.num_hashes {
            let idx = self.bit_index_for(h1, h2, i);
            if !self.get_bit(idx) {
                return false;
            }
        }
        true
    }

    /// Estimate the current false-positive rate based on inserted item count.
    ///
    /// Uses the formula `(1 - e^(-k*n/m))^k`.
    pub fn estimated_false_positive_rate(&self) -> f64 {
        let k = self.num_hashes as f64;
        let n = self.item_count as f64;
        let m = self.bit_count as f64;
        if m == 0.0 {
            return 1.0;
        }
        let inner = (-k * n / m).exp();
        (1.0 - inner).powf(k)
    }

    /// Return the number of items inserted.
    pub fn item_count(&self) -> u64 {
        self.item_count
    }

    /// Return the total number of bits in the filter.
    pub fn bit_count(&self) -> u64 {
        self.bit_count
    }

    /// Return the fraction of bits that are set.
    pub fn fill_ratio(&self) -> f64 {
        if self.bit_count == 0 {
            return 0.0;
        }
        let set_bits: u64 = self.bits.iter().map(|w| w.count_ones() as u64).sum();
        set_bits as f64 / self.bit_count as f64
    }

    /// Merge two filters with identical dimensions into a new filter.
    ///
    /// Returns `Err(BloomError::DimensionMismatch)` if the filters have different bit counts.
    pub fn merge(&self, other: &BloomFilter) -> Result<BloomFilter, BloomError> {
        if self.bit_count != other.bit_count {
            return Err(BloomError::DimensionMismatch {
                expected: self.bit_count,
                got: other.bit_count,
            });
        }
        let merged_bits: Vec<u64> = self
            .bits
            .iter()
            .zip(other.bits.iter())
            .map(|(a, b)| a | b)
            .collect();
        Ok(BloomFilter {
            bits: merged_bits,
            num_hashes: self.num_hashes,
            bit_count: self.bit_count,
            item_count: self.item_count + other.item_count,
        })
    }

    /// Reset all bits and the item count to zero.
    pub fn clear(&mut self) {
        for word in &mut self.bits {
            *word = 0;
        }
        self.item_count = 0;
    }

    /// Return the number of hash functions used.
    pub fn num_hashes(&self) -> u32 {
        self.num_hashes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Basic construction ─────────────────────────────────────────────────────

    #[test]
    fn test_new_empty_contains_nothing() {
        let bf = BloomFilter::new(1000, 0.01);
        assert!(!bf.contains("hello"));
        assert!(!bf.contains("world"));
        assert_eq!(bf.item_count(), 0);
    }

    #[test]
    fn test_bit_count_reasonable() {
        let bf = BloomFilter::new(1000, 0.01);
        // For 1000 items at 1% FP rate, bit count should be ~9585
        assert!(bf.bit_count() >= 1000);
    }

    #[test]
    fn test_num_hashes_at_least_one() {
        let bf = BloomFilter::new(1, 0.5);
        assert!(bf.num_hashes() >= 1);
    }

    // ── Insert / contains ─────────────────────────────────────────────────────

    #[test]
    fn test_insert_then_contains() {
        let mut bf = BloomFilter::new(1000, 0.01);
        bf.insert("http://example.org/subject");
        assert!(bf.contains("http://example.org/subject"));
    }

    #[test]
    fn test_insert_multiple_items() {
        let mut bf = BloomFilter::new(1000, 0.01);
        let items = ["alpha", "beta", "gamma", "delta", "epsilon"];
        for item in &items {
            bf.insert(item);
        }
        for item in &items {
            assert!(bf.contains(item), "{item} should be in filter");
        }
    }

    #[test]
    fn test_no_false_negatives() {
        let mut bf = BloomFilter::new(5000, 0.001);
        let items: Vec<String> = (0..500).map(|i| format!("item:{i}")).collect();
        for item in &items {
            bf.insert(item);
        }
        for item in &items {
            assert!(bf.contains(item), "False negative for {item}");
        }
    }

    #[test]
    fn test_item_count_tracks_inserts() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert("a");
        bf.insert("b");
        bf.insert("c");
        assert_eq!(bf.item_count(), 3);
    }

    #[test]
    fn test_empty_string_insert() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert("");
        assert!(bf.contains(""));
    }

    #[test]
    fn test_unicode_items() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert("日本語テスト");
        bf.insert("αβγδε");
        assert!(bf.contains("日本語テスト"));
        assert!(bf.contains("αβγδε"));
    }

    // ── False positive rate ───────────────────────────────────────────────────

    #[test]
    fn test_estimated_fpr_zero_items() {
        let bf = BloomFilter::new(1000, 0.01);
        let fpr = bf.estimated_false_positive_rate();
        assert!(fpr < 1e-9, "FPR should be near 0 with no items");
    }

    #[test]
    fn test_estimated_fpr_increases_with_load() {
        let mut bf = BloomFilter::new(100, 0.01);
        let fpr_empty = bf.estimated_false_positive_rate();
        for i in 0..100 {
            bf.insert(&format!("item:{i}"));
        }
        let fpr_full = bf.estimated_false_positive_rate();
        assert!(fpr_full > fpr_empty, "FPR should increase with more items");
    }

    #[test]
    fn test_empirical_false_positive_rate_reasonable() {
        let mut bf = BloomFilter::new(1000, 0.05);
        for i in 0..1000 {
            bf.insert(&format!("inserted:{i}"));
        }
        // Check 10000 items not inserted, count false positives
        let mut false_positives = 0usize;
        let total = 10000usize;
        for i in 10000..10000 + total {
            if bf.contains(&format!("notinserted:{i}")) {
                false_positives += 1;
            }
        }
        let actual_fpr = false_positives as f64 / total as f64;
        // Allow 3x leeway for probabilistic variation
        assert!(actual_fpr < 0.15, "Empirical FPR {actual_fpr} too high");
    }

    // ── fill_ratio ────────────────────────────────────────────────────────────

    #[test]
    fn test_fill_ratio_zero_on_empty() {
        let bf = BloomFilter::new(1000, 0.01);
        assert_eq!(bf.fill_ratio(), 0.0);
    }

    #[test]
    fn test_fill_ratio_increases_with_inserts() {
        let mut bf = BloomFilter::new(1000, 0.01);
        let r0 = bf.fill_ratio();
        bf.insert("item_a");
        let r1 = bf.fill_ratio();
        assert!(r1 > r0);
    }

    #[test]
    fn test_fill_ratio_bounded_by_one() {
        let mut bf = BloomFilter::new(10, 0.5);
        for i in 0..10000 {
            bf.insert(&format!("x:{i}"));
        }
        assert!(bf.fill_ratio() <= 1.0);
    }

    // ── merge ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_merge_compatible_filters() {
        let mut bf1 = BloomFilter::with_params(1024, 3);
        let mut bf2 = BloomFilter::with_params(1024, 3);
        bf1.insert("alpha");
        bf2.insert("beta");
        let merged = bf1.merge(&bf2).unwrap();
        assert!(merged.contains("alpha"));
        assert!(merged.contains("beta"));
    }

    #[test]
    fn test_merge_item_count_sum() {
        let mut bf1 = BloomFilter::with_params(1024, 3);
        let mut bf2 = BloomFilter::with_params(1024, 3);
        bf1.insert("a");
        bf1.insert("b");
        bf2.insert("c");
        let merged = bf1.merge(&bf2).unwrap();
        assert_eq!(merged.item_count(), 3);
    }

    #[test]
    fn test_merge_dimension_mismatch() {
        let bf1 = BloomFilter::with_params(1024, 3);
        let bf2 = BloomFilter::with_params(2048, 3);
        let r = bf1.merge(&bf2);
        assert_eq!(
            r,
            Err(BloomError::DimensionMismatch {
                expected: 1024,
                got: 2048,
            })
        );
    }

    // ── clear ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_clear_resets_filter() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert("hello");
        assert!(bf.contains("hello"));
        bf.clear();
        assert!(!bf.contains("hello"));
        assert_eq!(bf.item_count(), 0);
        assert_eq!(bf.fill_ratio(), 0.0);
    }

    // ── BloomError display ────────────────────────────────────────────────────

    #[test]
    fn test_bloom_error_display() {
        let e = BloomError::DimensionMismatch {
            expected: 1024,
            got: 2048,
        };
        let s = e.to_string();
        assert!(s.contains("1024"));
        assert!(s.contains("2048"));
    }

    // ── with_params ───────────────────────────────────────────────────────────

    #[test]
    fn test_with_params_works() {
        let mut bf = BloomFilter::with_params(512, 5);
        bf.insert("rdf:type");
        assert!(bf.contains("rdf:type"));
        assert!(!bf.contains("rdf:label"));
    }

    // ── RDF triple use case ───────────────────────────────────────────────────

    #[test]
    fn test_rdf_triple_keys() {
        let mut bf = BloomFilter::new(10000, 0.001);
        let triples = [
            "<http://ex.org/s1> <http://ex.org/p1> <http://ex.org/o1>",
            "<http://ex.org/s2> <http://ex.org/p2> \"literal value\"",
            "_:blank1 <http://ex.org/knows> <http://ex.org/alice>",
        ];
        for triple in &triples {
            bf.insert(triple);
        }
        for triple in &triples {
            assert!(bf.contains(triple), "Triple should be found: {triple}");
        }
        // A triple that was not inserted should not be found (with high probability)
        // (not asserting this because of false positives being possible)
    }

    // ── Hash function consistency ──────────────────────────────────────────────

    #[test]
    fn test_same_item_always_found() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert("consistent");
        // Query multiple times - must always return true
        for _ in 0..100 {
            assert!(bf.contains("consistent"));
        }
    }

    #[test]
    fn test_distinct_items_distinct_results() {
        let mut bf = BloomFilter::with_params(65536, 7);
        let items: Vec<String> = (0..1000).map(|i| format!("triple_{i}")).collect();
        for item in &items {
            bf.insert(item);
        }
        // All inserted items must be found (no false negatives)
        for item in &items {
            assert!(bf.contains(item));
        }
    }
}
