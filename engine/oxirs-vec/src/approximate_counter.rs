//! Approximate cardinality counting using the HyperLogLog algorithm.
//!
//! HyperLogLog (HLL) is a probabilistic data structure for estimating the
//! cardinality of large multisets using sub-linear memory.  This implementation
//! follows the original Flajolet et al. 2007 paper with small/large range
//! corrections and 64-bit FNV-1a hashing.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by HyperLogLog and cardinality-estimator operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CounterError {
    /// Two HLL sketches have different register counts and cannot be merged.
    PrecisionMismatch,
    /// No sketch exists for the given stream name.
    StreamNotFound(String),
}

impl std::fmt::Display for CounterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CounterError::PrecisionMismatch => write!(f, "HyperLogLog precision mismatch"),
            CounterError::StreamNotFound(s) => write!(f, "Stream not found: {}", s),
        }
    }
}

impl std::error::Error for CounterError {}

// ---------------------------------------------------------------------------
// FNV-1a 64-bit hash
// ---------------------------------------------------------------------------

const FNV_OFFSET_BASIS_64: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME_64: u64 = 1_099_511_628_211;

fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS_64;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME_64);
    }
    // Apply a Murmur3-style 64-bit finalizer for better bit avalanche.
    // Without this, FNV-1a has weak diffusion across the upper bits.
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51afd7ed558ccd);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xc4ceb9fe1a85ec53);
    hash ^= hash >> 33;
    hash
}

// ---------------------------------------------------------------------------
// HyperLogLog
// ---------------------------------------------------------------------------

/// A single HyperLogLog sketch for estimating cardinality.
///
/// # Parameters
/// * `precision` (b) — controls the number of registers m = 2^b.  Precision
///   must be in the range [4, 16].  Higher precision → less error but more
///   memory.  Typical error rate: ~1.04 / sqrt(m).
#[derive(Debug, Clone)]
pub struct HyperLogLog {
    /// m = 2^precision registers, each storing the maximum leading-zero count.
    pub registers: Vec<u8>,
    /// Number of registers (m = 2^precision).
    pub m: usize,
    /// Bias-correction constant α_m.
    pub alpha: f64,
    /// Precision parameter b where m = 2^b.
    precision: u8,
}

impl HyperLogLog {
    /// Create a new HyperLogLog sketch.
    ///
    /// # Panics
    /// Panics if `precision < 4` or `precision > 16`.
    pub fn new(precision: u8) -> Self {
        assert!(
            (4..=16).contains(&precision),
            "HyperLogLog precision must be in [4, 16]"
        );
        let m = 1usize << precision;
        let alpha = compute_alpha(m);
        Self {
            registers: vec![0u8; m],
            m,
            alpha,
            precision,
        }
    }

    /// Add an item to the sketch.
    pub fn add(&mut self, item: &str) {
        let hash = fnv1a_64(item.as_bytes());
        // Use the top `precision` bits as the register index.
        let idx = (hash >> (64 - self.precision)) as usize;
        // Shift the hash left by `precision` bits to get the "w" bits (the
        // lower 64-precision bits, shifted to the most-significant positions).
        // Count the number of leading zeros in those bits, then add 1.
        // If all w bits are zero (hash << precision == 0) leading_zeros returns
        // 64; clamp to 64-precision to avoid overflows in very small sketches.
        let w = hash << self.precision;
        let rho = if w == 0 {
            (64u32 - self.precision as u32) as u8 + 1
        } else {
            w.leading_zeros() as u8 + 1
        };
        if rho > self.registers[idx] {
            self.registers[idx] = rho;
        }
    }

    /// Estimate the number of distinct items added.
    pub fn count(&self) -> u64 {
        let m = self.m as f64;

        // Harmonic mean of 2^(-M[j])
        let sum: f64 = self
            .registers
            .iter()
            .map(|&r| 2.0_f64.powi(-(r as i32)))
            .sum();

        let raw_estimate = self.alpha * m * m / sum;

        // Small-range correction
        if raw_estimate <= 2.5 * m {
            let zeros = self.registers.iter().filter(|&&r| r == 0).count() as f64;
            if zeros > 0.0 {
                return (m * (m / zeros).ln()).round() as u64;
            }
        }

        // Large-range correction (2^32)
        if raw_estimate > (1u64 << 32) as f64 / 30.0 {
            let correction = -(2.0_f64.powi(32)) * (1.0 - raw_estimate / 2.0_f64.powi(32)).ln();
            return correction.round() as u64;
        }

        raw_estimate.round() as u64
    }

    /// Merge another HLL sketch into this one by taking the per-register max.
    ///
    /// Both sketches must have the same precision.
    pub fn merge(&mut self, other: &HyperLogLog) -> Result<(), CounterError> {
        if self.m != other.m {
            return Err(CounterError::PrecisionMismatch);
        }
        for (a, &b) in self.registers.iter_mut().zip(other.registers.iter()) {
            if b > *a {
                *a = b;
            }
        }
        Ok(())
    }

    /// Reset all registers to zero.
    pub fn clear(&mut self) {
        for r in &mut self.registers {
            *r = 0;
        }
    }

    /// Return the number of registers in this sketch.
    pub fn register_count(&self) -> usize {
        self.m
    }
}

/// Compute the bias-correction constant α_m.
fn compute_alpha(m: usize) -> f64 {
    match m {
        16 => 0.673,
        32 => 0.697,
        64 => 0.709,
        _ => 0.7213 / (1.0 + 1.079 / m as f64),
    }
}

// ---------------------------------------------------------------------------
// CardinalityEstimator
// ---------------------------------------------------------------------------

/// Multi-stream cardinality estimator backed by one HLL sketch per stream.
pub struct CardinalityEstimator {
    streams: HashMap<String, HyperLogLog>,
    precision: u8,
}

impl CardinalityEstimator {
    /// Create a new estimator; all per-stream sketches use `precision`.
    pub fn new(precision: u8) -> Self {
        Self {
            streams: HashMap::new(),
            precision,
        }
    }

    /// Add `item` to the sketch for `stream` (creates the stream if absent).
    pub fn add(&mut self, stream: &str, item: &str) {
        self.streams
            .entry(stream.to_string())
            .or_insert_with(|| HyperLogLog::new(self.precision))
            .add(item);
    }

    /// Estimate the cardinality of a single `stream`.
    ///
    /// Returns 0 if the stream has never been seen.
    pub fn estimate(&self, stream: &str) -> u64 {
        self.streams.get(stream).map(|h| h.count()).unwrap_or(0)
    }

    /// Estimate the cardinality of the union of several `streams` by merging
    /// their sketches.
    ///
    /// Returns 0 when none of the named streams exist.
    pub fn union_estimate(&self, streams: &[&str]) -> u64 {
        let mut merged: Option<HyperLogLog> = None;
        for &name in streams {
            if let Some(hll) = self.streams.get(name) {
                match &mut merged {
                    None => merged = Some(hll.clone()),
                    Some(m) => {
                        // Ignore precision mismatch — shouldn't happen since
                        // all streams use the same precision.
                        let _ = m.merge(hll);
                    }
                }
            }
        }
        merged.map(|h| h.count()).unwrap_or(0)
    }

    /// Return the number of tracked streams.
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // HLL tolerance — HyperLogLog is approximate.
    const TOLERANCE: f64 = 0.30; // 30% is generous but safe

    fn within_tolerance(estimate: u64, expected: u64, tol: f64) -> bool {
        if expected == 0 {
            return estimate == 0;
        }
        let ratio = estimate as f64 / expected as f64;
        ratio >= (1.0 - tol) && ratio <= (1.0 + tol)
    }

    // -----------------------------------------------------------------------
    // Basic construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_precision_4() {
        let hll = HyperLogLog::new(4);
        assert_eq!(hll.m, 16);
        assert_eq!(hll.registers.len(), 16);
    }

    #[test]
    fn test_new_precision_8() {
        let hll = HyperLogLog::new(8);
        assert_eq!(hll.m, 256);
    }

    #[test]
    fn test_new_precision_16() {
        let hll = HyperLogLog::new(16);
        assert_eq!(hll.m, 65536);
    }

    #[test]
    fn test_register_count() {
        let hll = HyperLogLog::new(6);
        assert_eq!(hll.register_count(), 64);
    }

    // -----------------------------------------------------------------------
    // Empty cardinality
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_count_is_zero() {
        let hll = HyperLogLog::new(8);
        assert_eq!(hll.count(), 0);
    }

    // -----------------------------------------------------------------------
    // Single-item cardinality
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_item_count() {
        let mut hll = HyperLogLog::new(10);
        hll.add("hello");
        let c = hll.count();
        assert!(c >= 1, "Expected at least 1, got {}", c);
    }

    // -----------------------------------------------------------------------
    // Monotone increase
    // -----------------------------------------------------------------------

    #[test]
    fn test_count_monotone_increases() {
        let mut hll = HyperLogLog::new(10);
        let mut prev = 0u64;
        for i in 0..100 {
            hll.add(&format!("item_{}", i));
            let curr = hll.count();
            assert!(
                curr >= prev,
                "Count decreased from {} to {} at step {}",
                prev,
                curr,
                i
            );
            prev = curr;
        }
    }

    // -----------------------------------------------------------------------
    // Cardinality accuracy
    // -----------------------------------------------------------------------

    #[test]
    fn test_cardinality_100_items() {
        let mut hll = HyperLogLog::new(10);
        for i in 0..100u64 {
            hll.add(&format!("item_{}", i));
        }
        let est = hll.count();
        assert!(
            within_tolerance(est, 100, TOLERANCE),
            "Estimate {} not within {}% of 100",
            est,
            (TOLERANCE * 100.0) as u32
        );
    }

    #[test]
    fn test_cardinality_1000_items() {
        let mut hll = HyperLogLog::new(10);
        for i in 0..1000u64 {
            hll.add(&format!("element_{}", i));
        }
        let est = hll.count();
        assert!(
            within_tolerance(est, 1000, TOLERANCE),
            "Estimate {} not within {}% of 1000",
            est,
            (TOLERANCE * 100.0) as u32
        );
    }

    #[test]
    fn test_duplicate_items_not_counted() {
        let mut hll = HyperLogLog::new(10);
        for _ in 0..100 {
            hll.add("same_item");
        }
        let est = hll.count();
        // Should be close to 1, not 100
        assert!(est < 10, "Duplicates inflated count to {}", est);
    }

    // -----------------------------------------------------------------------
    // Clear
    // -----------------------------------------------------------------------

    #[test]
    fn test_clear_resets_to_zero() {
        let mut hll = HyperLogLog::new(8);
        for i in 0..50 {
            hll.add(&format!("x{}", i));
        }
        hll.clear();
        assert_eq!(hll.count(), 0);
    }

    #[test]
    fn test_clear_then_reuse() {
        let mut hll = HyperLogLog::new(8);
        for i in 0..50 {
            hll.add(&format!("x{}", i));
        }
        hll.clear();
        hll.add("hello");
        let c = hll.count();
        assert!(c >= 1);
    }

    // -----------------------------------------------------------------------
    // Merge
    // -----------------------------------------------------------------------

    #[test]
    fn test_merge_compatible() {
        let mut hll1 = HyperLogLog::new(10);
        let mut hll2 = HyperLogLog::new(10);
        for i in 0..100 {
            hll1.add(&format!("a{}", i));
        }
        for i in 0..100 {
            hll2.add(&format!("b{}", i));
        }
        let result = hll1.merge(&hll2);
        assert!(result.is_ok());
        let est = hll1.count();
        // Should be roughly 200 unique items
        assert!(
            within_tolerance(est, 200, TOLERANCE),
            "Merged estimate {} not within tolerance of 200",
            est
        );
    }

    #[test]
    fn test_merge_overlapping() {
        let mut hll1 = HyperLogLog::new(10);
        let mut hll2 = HyperLogLog::new(10);
        for i in 0..100 {
            hll1.add(&format!("item{}", i));
            hll2.add(&format!("item{}", i));
        }
        let _ = hll1.merge(&hll2);
        let est = hll1.count();
        // Same items — union cardinality ≈ 100
        assert!(
            within_tolerance(est, 100, TOLERANCE),
            "Overlapping merge estimate {} not within tolerance of 100",
            est
        );
    }

    #[test]
    fn test_merge_precision_mismatch_error() {
        let mut hll1 = HyperLogLog::new(8);
        let hll2 = HyperLogLog::new(10);
        let result = hll1.merge(&hll2);
        assert_eq!(result, Err(CounterError::PrecisionMismatch));
    }

    // -----------------------------------------------------------------------
    // Precision variants
    // -----------------------------------------------------------------------

    #[test]
    fn test_precision_5_accuracy() {
        let mut hll = HyperLogLog::new(5);
        for i in 0..50 {
            hll.add(&format!("p5_{}", i));
        }
        let est = hll.count();
        assert!(within_tolerance(est, 50, TOLERANCE));
    }

    #[test]
    fn test_precision_12_accuracy() {
        let mut hll = HyperLogLog::new(12);
        for i in 0..500 {
            hll.add(&format!("p12_{}", i));
        }
        let est = hll.count();
        assert!(within_tolerance(est, 500, TOLERANCE));
    }

    // -----------------------------------------------------------------------
    // CardinalityEstimator
    // -----------------------------------------------------------------------

    #[test]
    fn test_estimator_single_stream() {
        let mut est = CardinalityEstimator::new(10);
        for i in 0..100 {
            est.add("users", &format!("user_{}", i));
        }
        let c = est.estimate("users");
        assert!(within_tolerance(c, 100, TOLERANCE));
    }

    #[test]
    fn test_estimator_unknown_stream_returns_zero() {
        let est = CardinalityEstimator::new(10);
        assert_eq!(est.estimate("nonexistent"), 0);
    }

    #[test]
    fn test_estimator_multiple_streams() {
        let mut est = CardinalityEstimator::new(10);
        for i in 0..100 {
            est.add("stream_a", &format!("a{}", i));
            est.add("stream_b", &format!("b{}", i));
        }
        assert_eq!(est.stream_count(), 2);
        assert!(within_tolerance(est.estimate("stream_a"), 100, TOLERANCE));
        assert!(within_tolerance(est.estimate("stream_b"), 100, TOLERANCE));
    }

    #[test]
    fn test_estimator_stream_count() {
        let mut est = CardinalityEstimator::new(8);
        assert_eq!(est.stream_count(), 0);
        est.add("x", "item1");
        assert_eq!(est.stream_count(), 1);
        est.add("y", "item2");
        assert_eq!(est.stream_count(), 2);
        est.add("x", "item3"); // same stream
        assert_eq!(est.stream_count(), 2);
    }

    #[test]
    fn test_estimator_union_estimate_two_streams() {
        let mut est = CardinalityEstimator::new(10);
        for i in 0..100 {
            est.add("a", &format!("a{}", i));
        }
        for i in 0..100 {
            est.add("b", &format!("b{}", i));
        }
        let union = est.union_estimate(&["a", "b"]);
        // ~200 unique items
        assert!(within_tolerance(union, 200, TOLERANCE));
    }

    #[test]
    fn test_estimator_union_estimate_nonexistent_streams() {
        let est = CardinalityEstimator::new(10);
        assert_eq!(est.union_estimate(&["no1", "no2"]), 0);
    }

    #[test]
    fn test_estimator_union_estimate_single_stream() {
        let mut est = CardinalityEstimator::new(10);
        for i in 0..50 {
            est.add("only", &format!("i{}", i));
        }
        let union = est.union_estimate(&["only"]);
        assert!(within_tolerance(union, 50, TOLERANCE));
    }

    #[test]
    fn test_estimator_union_mixed_existing_and_missing() {
        let mut est = CardinalityEstimator::new(10);
        for i in 0..80 {
            est.add("real", &format!("r{}", i));
        }
        let union = est.union_estimate(&["real", "missing"]);
        assert!(within_tolerance(union, 80, TOLERANCE));
    }

    #[test]
    fn test_counter_error_display() {
        let e1 = CounterError::PrecisionMismatch;
        assert!(!e1.to_string().is_empty());
        let e2 = CounterError::StreamNotFound("foo".to_string());
        assert!(e2.to_string().contains("foo"));
    }

    #[test]
    fn test_fnv1a_deterministic() {
        assert_eq!(fnv1a_64(b"hello"), fnv1a_64(b"hello"));
        assert_ne!(fnv1a_64(b"hello"), fnv1a_64(b"world"));
    }
}
