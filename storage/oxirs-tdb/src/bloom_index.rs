//! Bloom filter index for fast RDF triple membership tests (v1.1.0 round 16).
//!
//! Implements a space-efficient probabilistic data structure using FNV-1a
//! double-hashing to support `O(k)` insert and lookup with configurable
//! false-positive rates.
//!
//! Reference: Bloom, B.H., "Space/time trade-offs in hash coding with allowable
//! errors", Commun. ACM 13(7), 1970. <https://doi.org/10.1145/362686.362692>

// ──────────────────────────────────────────────────────────────────────────────
// BloomFilter
// ──────────────────────────────────────────────────────────────────────────────

/// A Bloom filter using FNV-1a double hashing.
///
/// Implements the optimal single-array double-hashing technique described
/// by Kirsch & Mitzenmacher (2006).
pub struct BloomFilter {
    /// Bit array stored as 64-bit words.
    bits: Vec<u64>,
    /// Total number of bits in the filter.
    num_bits: usize,
    /// Number of hash functions `k`.
    num_hashes: usize,
    /// Number of items inserted.
    item_count: usize,
}

impl BloomFilter {
    /// Create a new `BloomFilter` tuned for `capacity` items at the given
    /// `false_positive_rate` (must be in `(0, 1)`).
    ///
    /// Optimal bit count: `m = -n * ln(p) / (ln 2)^2`
    /// Optimal hash count: `k = (m/n) * ln 2`
    pub fn new(capacity: usize, false_positive_rate: f64) -> Self {
        let capacity = capacity.max(1);
        let fp = false_positive_rate.clamp(1e-10, 1.0 - 1e-10);

        // Optimal m and k
        let ln2 = core::f64::consts::LN_2;
        let m = (-(capacity as f64) * fp.ln() / (ln2 * ln2)).ceil() as usize;
        let num_bits = m.max(64);
        let k = ((num_bits as f64 / capacity as f64) * ln2).round() as usize;
        let num_hashes = k.max(1);
        let words = (num_bits + 63) / 64;

        Self {
            bits: vec![0u64; words],
            num_bits,
            num_hashes,
            item_count: 0,
        }
    }

    /// Insert `item` into the filter.
    pub fn insert(&mut self, item: &str) {
        let (h1, h2) = Self::double_hash(item, self.num_bits);
        for i in 0..self.num_hashes {
            let bit = (h1.wrapping_add(i.wrapping_mul(h2))) % self.num_bits;
            self.bits[bit / 64] |= 1u64 << (bit % 64);
        }
        self.item_count += 1;
    }

    /// Return `true` if `item` **might** be in the filter.
    ///
    /// False positives are possible; false negatives are not.
    pub fn might_contain(&self, item: &str) -> bool {
        let (h1, h2) = Self::double_hash(item, self.num_bits);
        for i in 0..self.num_hashes {
            let bit = (h1.wrapping_add(i.wrapping_mul(h2))) % self.num_bits;
            if self.bits[bit / 64] & (1u64 << (bit % 64)) == 0 {
                return false;
            }
        }
        true
    }

    /// Number of items inserted.
    pub fn item_count(&self) -> usize {
        self.item_count
    }

    /// Total number of bits allocated.
    pub fn bit_count(&self) -> usize {
        self.num_bits
    }

    /// Estimate the current false-positive rate based on actual fill.
    ///
    /// Uses the formula `(1 - e^(-k*n/m))^k`.
    pub fn estimated_false_positive_rate(&self) -> f64 {
        let k = self.num_hashes as f64;
        let n = self.item_count as f64;
        let m = self.num_bits as f64;
        let fill = (-k * n / m).exp();
        (1.0 - fill).powf(k)
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Compute two independent FNV-1a hashes of `s` mapped into `[0, num_bits)`.
    fn double_hash(s: &str, num_bits: usize) -> (usize, usize) {
        const FNV_PRIME_64: u64 = 1_099_511_628_211;
        const FNV_OFFSET_64: u64 = 14_695_981_039_346_656_037;

        let mut h1 = FNV_OFFSET_64;
        let mut h2 = FNV_OFFSET_64 ^ 0xdead_beef_cafe_1234;

        for byte in s.bytes() {
            h1 ^= byte as u64;
            h1 = h1.wrapping_mul(FNV_PRIME_64);
            h2 ^= (byte as u64).wrapping_add(7);
            h2 = h2.wrapping_mul(FNV_PRIME_64);
        }

        ((h1 as usize) % num_bits, (h2 as usize) % num_bits + 1)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// TripleBloomIndex
// ──────────────────────────────────────────────────────────────────────────────

/// A Bloom-filter-backed index for RDF triples.
///
/// Serialises each triple as `"subject\0predicate\0object"` before passing it
/// to the underlying [`BloomFilter`].
pub struct TripleBloomIndex {
    /// Underlying Bloom filter.
    filter: BloomFilter,
}

impl TripleBloomIndex {
    /// Create a new `TripleBloomIndex` tuned for `capacity` triples at
    /// `fp_rate` false-positive rate.
    pub fn new(capacity: usize, fp_rate: f64) -> Self {
        Self {
            filter: BloomFilter::new(capacity, fp_rate),
        }
    }

    /// Insert a triple into the index.
    pub fn insert(&mut self, subj: &str, pred: &str, obj: &str) {
        let key = Self::triple_key(subj, pred, obj);
        self.filter.insert(&key);
    }

    /// Return `true` if the triple **might** be in the index.
    pub fn might_contain(&self, subj: &str, pred: &str, obj: &str) -> bool {
        let key = Self::triple_key(subj, pred, obj);
        self.filter.might_contain(&key)
    }

    /// Number of triples inserted.
    pub fn item_count(&self) -> usize {
        self.filter.item_count()
    }

    /// Serialise a triple to a canonical key using NUL separators.
    fn triple_key(subj: &str, pred: &str, obj: &str) -> String {
        format!("{}\x00{}\x00{}", subj, pred, obj)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── BloomFilter basics ────────────────────────────────────────────────────

    #[test]
    fn test_bloom_insert_and_contains() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert("hello");
        assert!(bf.might_contain("hello"));
    }

    #[test]
    fn test_bloom_unknown_item_typically_false() {
        let bf = BloomFilter::new(100, 0.01);
        // With no insertions, unknown items must return false.
        assert!(!bf.might_contain("never_inserted"));
    }

    #[test]
    fn test_bloom_item_count_increments() {
        let mut bf = BloomFilter::new(100, 0.01);
        assert_eq!(bf.item_count(), 0);
        bf.insert("a");
        assert_eq!(bf.item_count(), 1);
        bf.insert("b");
        assert_eq!(bf.item_count(), 2);
    }

    #[test]
    fn test_bloom_bit_count_positive() {
        let bf = BloomFilter::new(1000, 0.01);
        assert!(bf.bit_count() >= 64);
    }

    #[test]
    fn test_bloom_bit_count_larger_capacity_more_bits() {
        let bf_small = BloomFilter::new(100, 0.01);
        let bf_large = BloomFilter::new(10_000, 0.01);
        assert!(bf_large.bit_count() > bf_small.bit_count());
    }

    #[test]
    fn test_bloom_fp_rate_zero_when_empty() {
        let bf = BloomFilter::new(100, 0.01);
        // With 0 items the formula returns 0.0
        assert_eq!(bf.estimated_false_positive_rate(), 0.0);
    }

    #[test]
    fn test_bloom_fp_rate_increases_with_fill() {
        let mut bf = BloomFilter::new(10, 0.05);
        let initial = bf.estimated_false_positive_rate();
        for i in 0..10 {
            bf.insert(&format!("item_{}", i));
        }
        let after = bf.estimated_false_positive_rate();
        assert!(after > initial);
    }

    #[test]
    fn test_bloom_multiple_inserts_same_key() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert("dup");
        bf.insert("dup");
        // Count increments twice even for duplicates
        assert_eq!(bf.item_count(), 2);
        assert!(bf.might_contain("dup"));
    }

    #[test]
    fn test_bloom_many_items_all_found() {
        let mut bf = BloomFilter::new(1000, 0.01);
        let items: Vec<String> = (0..100).map(|i| format!("item_{}", i)).collect();
        for item in &items {
            bf.insert(item);
        }
        for item in &items {
            assert!(bf.might_contain(item), "item '{}' not found", item);
        }
    }

    #[test]
    fn test_bloom_empty_string() {
        let mut bf = BloomFilter::new(10, 0.01);
        bf.insert("");
        assert!(bf.might_contain(""));
        assert!(!bf.might_contain("x"));
    }

    // ── TripleBloomIndex ──────────────────────────────────────────────────────

    #[test]
    fn test_triple_bloom_insert_and_contains() {
        let mut idx = TripleBloomIndex::new(100, 0.01);
        idx.insert("http://a.org/s", "http://a.org/p", "http://a.org/o");
        assert!(idx.might_contain("http://a.org/s", "http://a.org/p", "http://a.org/o"));
    }

    #[test]
    fn test_triple_bloom_unknown_triple_false() {
        let idx = TripleBloomIndex::new(100, 0.01);
        assert!(!idx.might_contain("http://a.org/s", "http://a.org/p", "http://a.org/o"));
    }

    #[test]
    fn test_triple_bloom_item_count() {
        let mut idx = TripleBloomIndex::new(100, 0.01);
        assert_eq!(idx.item_count(), 0);
        idx.insert("s", "p", "o");
        assert_eq!(idx.item_count(), 1);
    }

    #[test]
    fn test_triple_bloom_multiple_triples() {
        let mut idx = TripleBloomIndex::new(100, 0.01);
        let triples = vec![("s1", "p1", "o1"), ("s2", "p2", "o2"), ("s3", "p3", "o3")];
        for (s, p, o) in &triples {
            idx.insert(s, p, o);
        }
        assert_eq!(idx.item_count(), 3);
        for (s, p, o) in &triples {
            assert!(idx.might_contain(s, p, o));
        }
    }

    #[test]
    fn test_triple_bloom_same_subject_different_predicate() {
        let mut idx = TripleBloomIndex::new(100, 0.01);
        idx.insert("s", "p1", "o");
        assert!(idx.might_contain("s", "p1", "o"));
        // Different predicate — should be false (absent)
        assert!(!idx.might_contain("s", "p2", "o"));
    }

    #[test]
    fn test_triple_bloom_permutations_distinct() {
        let mut idx = TripleBloomIndex::new(100, 0.01);
        idx.insert("a", "b", "c");
        // (b, a, c) was never inserted so should be absent
        assert!(!idx.might_contain("b", "a", "c"));
    }
}
