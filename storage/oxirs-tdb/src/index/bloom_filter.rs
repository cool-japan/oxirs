//! Bloom filter for index optimization
//!
//! Provides probabilistic membership testing to reduce unnecessary index lookups.
//! Uses counting bloom filters for dynamic updates with SciRS2 integration.
//!
//! ## Features
//!
//! - **Space-efficient** - Configurable false positive rate (default 1%)
//! - **Fast lookups** - O(k) where k is number of hash functions
//! - **Dynamic updates** - Support for insertions and deletions
//! - **SciRS2 profiling** - Performance metrics and optimization tracking
//! - **Persistence** - Serialization for disk storage
//!
//! ## Usage
//!
//! ```rust,no_run
//! use oxirs_tdb::index::bloom_filter::{BloomFilter, BloomFilterConfig};
//!
//! let config = BloomFilterConfig::default();
//! let mut filter = BloomFilter::new(config).unwrap();
//!
//! // Insert elements
//! filter.insert(b"key1");
//! filter.insert(b"key2");
//!
//! // Test membership
//! assert!(filter.contains(b"key1"));
//! assert!(!filter.contains(b"key3")); // Definitely not present
//! ```

use crate::error::{Result, TdbError};
use scirs2_core::metrics::{Counter, Gauge, Histogram, MetricsRegistry};
use scirs2_core::random::{rngs, Random};
use scirs2_core::RngCore;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Bloom filter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomFilterConfig {
    /// Expected number of elements
    pub expected_elements: usize,
    /// Desired false positive rate (0.0 - 1.0)
    pub false_positive_rate: f64,
    /// Number of hash functions (auto-calculated if None)
    pub num_hash_functions: Option<usize>,
    /// Bit array size (auto-calculated if None)
    pub bit_array_size: Option<usize>,
    /// Enable counting bloom filter (supports deletions)
    pub enable_counting: bool,
    /// Enable performance metrics
    pub enable_metrics: bool,
}

impl Default for BloomFilterConfig {
    fn default() -> Self {
        Self {
            expected_elements: 10_000,
            false_positive_rate: 0.01, // 1%
            num_hash_functions: None,
            bit_array_size: None,
            enable_counting: false,
            enable_metrics: true,
        }
    }
}

impl BloomFilterConfig {
    /// Calculate optimal bit array size
    pub fn calculate_bit_array_size(&self) -> usize {
        if let Some(size) = self.bit_array_size {
            return size;
        }

        // m = -(n * ln(p)) / (ln(2)^2)
        // where n = expected elements, p = false positive rate
        let n = self.expected_elements as f64;
        let p = self.false_positive_rate;
        let m = -(n * p.ln()) / (2_f64.ln().powi(2));
        m.ceil() as usize
    }

    /// Calculate optimal number of hash functions
    pub fn calculate_num_hash_functions(&self) -> usize {
        if let Some(k) = self.num_hash_functions {
            return k;
        }

        // k = (m/n) * ln(2)
        let m = self.calculate_bit_array_size() as f64;
        let n = self.expected_elements as f64;
        let k = (m / n) * 2_f64.ln();
        k.ceil().max(1.0) as usize
    }
}

/// Standard bloom filter with bit array
pub struct BloomFilter {
    /// Configuration
    config: BloomFilterConfig,
    /// Bit array
    bits: Vec<AtomicU64>,
    /// Number of bits
    num_bits: usize,
    /// Number of hash functions
    num_hash_functions: usize,
    /// Hash seeds
    hash_seeds: Vec<u64>,
    /// Random number generator for hash seeds
    rng: Random<rngs::StdRng>,
    /// Metrics
    metrics: Option<BloomFilterMetrics>,
    /// Element count estimate
    element_count: AtomicU64,
}

/// Counting bloom filter (supports deletions)
pub struct CountingBloomFilter {
    /// Configuration
    config: BloomFilterConfig,
    /// Counter array (4-bit counters)
    counters: Vec<AtomicU64>,
    /// Number of counters
    num_counters: usize,
    /// Number of hash functions
    num_hash_functions: usize,
    /// Hash seeds
    hash_seeds: Vec<u64>,
    /// Random number generator
    rng: Random<rngs::StdRng>,
    /// Metrics
    metrics: Option<BloomFilterMetrics>,
    /// Element count
    element_count: AtomicU64,
}

/// Bloom filter metrics
struct BloomFilterMetrics {
    /// Registry
    registry: Arc<MetricsRegistry>,
    /// Insert counter
    inserts: Counter,
    /// Delete counter (counting filter only)
    deletes: Counter,
    /// Lookup counter
    lookups: Counter,
    /// True positives
    true_positives: Counter,
    /// False positives
    false_positives: Counter,
    /// Current fill rate
    fill_rate: Gauge,
    /// Lookup latency
    lookup_latency: Histogram,
}

impl BloomFilterMetrics {
    fn new(filter_name: &str) -> Self {
        let registry = Arc::new(MetricsRegistry::new());

        Self {
            registry: registry.clone(),
            inserts: Counter::new(format!("{}.inserts", filter_name)),
            deletes: Counter::new(format!("{}.deletes", filter_name)),
            lookups: Counter::new(format!("{}.lookups", filter_name)),
            true_positives: Counter::new(format!("{}.true_positives", filter_name)),
            false_positives: Counter::new(format!("{}.false_positives", filter_name)),
            fill_rate: Gauge::new(format!("{}.fill_rate", filter_name)),
            lookup_latency: Histogram::new(format!("{}.lookup_latency_us", filter_name)),
        }
    }
}

impl BloomFilter {
    /// Create a new bloom filter
    pub fn new(config: BloomFilterConfig) -> Result<Self> {
        let num_bits = config.calculate_bit_array_size();
        let num_hash_functions = config.calculate_num_hash_functions();

        // Calculate number of u64s needed
        let num_u64s = (num_bits + 63) / 64;

        let mut rng = Random::seed(0);

        // Generate random seeds for hash functions
        let hash_seeds: Vec<u64> = (0..num_hash_functions)
            .map(|_| RngCore::next_u64(&mut rng))
            .collect();

        let metrics = if config.enable_metrics {
            Some(BloomFilterMetrics::new("bloom_filter"))
        } else {
            None
        };

        Ok(Self {
            config,
            bits: (0..num_u64s).map(|_| AtomicU64::new(0)).collect(),
            num_bits,
            num_hash_functions,
            hash_seeds,
            rng,
            metrics,
            element_count: AtomicU64::new(0),
        })
    }

    /// Insert an element into the bloom filter
    pub fn insert<T: Hash>(&mut self, item: &T) {
        let start = std::time::Instant::now();

        for &seed in &self.hash_seeds {
            let hash = self.hash_with_seed(item, seed);
            let bit_index = (hash % self.num_bits as u64) as usize;

            let word_index = bit_index / 64;
            let bit_offset = bit_index % 64;

            self.bits[word_index].fetch_or(1u64 << bit_offset, Ordering::Relaxed);
        }

        self.element_count.fetch_add(1, Ordering::Relaxed);

        if let Some(ref metrics) = self.metrics {
            metrics.inserts.inc();
            self.update_fill_rate();
        }
    }

    /// Test if an element might be in the set
    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        let start = std::time::Instant::now();

        let result = self.hash_seeds.iter().all(|&seed| {
            let hash = self.hash_with_seed(item, seed);
            let bit_index = (hash % self.num_bits as u64) as usize;

            let word_index = bit_index / 64;
            let bit_offset = bit_index % 64;

            let word = self.bits[word_index].load(Ordering::Relaxed);
            (word & (1u64 << bit_offset)) != 0
        });

        if let Some(ref metrics) = self.metrics {
            metrics.lookups.inc();
            let elapsed = start.elapsed();
            metrics.lookup_latency.observe(elapsed.as_micros() as f64);
        }

        result
    }

    /// Clear the bloom filter
    pub fn clear(&mut self) {
        for word in &self.bits {
            word.store(0, Ordering::Relaxed);
        }
        self.element_count.store(0, Ordering::Relaxed);
    }

    /// Get the estimated false positive rate
    pub fn estimated_false_positive_rate(&self) -> f64 {
        let n = self.element_count.load(Ordering::Relaxed) as f64;
        let m = self.num_bits as f64;
        let k = self.num_hash_functions as f64;

        // FPR = (1 - e^(-kn/m))^k
        (1.0 - (-k * n / m).exp()).powf(k)
    }

    /// Get the current fill rate (proportion of bits set)
    pub fn fill_rate(&self) -> f64 {
        let total_bits = self.num_bits;
        let set_bits: u64 = self
            .bits
            .iter()
            .map(|word| word.load(Ordering::Relaxed).count_ones() as u64)
            .sum();

        set_bits as f64 / total_bits as f64
    }

    /// Get statistics
    pub fn stats(&self) -> BloomFilterStats {
        BloomFilterStats {
            num_bits: self.num_bits,
            num_hash_functions: self.num_hash_functions,
            element_count: self.element_count.load(Ordering::Relaxed),
            fill_rate: self.fill_rate(),
            estimated_fpr: self.estimated_false_positive_rate(),
            configured_fpr: self.config.false_positive_rate,
        }
    }

    /// Hash an item with a specific seed
    fn hash_with_seed<T: Hash>(&self, item: &T, seed: u64) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        seed.hash(&mut hasher);
        item.hash(&mut hasher);
        hasher.finish()
    }

    /// Update fill rate metric
    fn update_fill_rate(&self) {
        if let Some(ref metrics) = self.metrics {
            metrics.fill_rate.set(self.fill_rate());
        }
    }
}

impl CountingBloomFilter {
    /// Create a new counting bloom filter
    pub fn new(config: BloomFilterConfig) -> Result<Self> {
        let num_counters = config.calculate_bit_array_size();
        let num_hash_functions = config.calculate_num_hash_functions();

        let mut rng = Random::seed(0);

        let hash_seeds: Vec<u64> = (0..num_hash_functions)
            .map(|_| RngCore::next_u64(&mut rng))
            .collect();

        let metrics = if config.enable_metrics {
            Some(BloomFilterMetrics::new("counting_bloom_filter"))
        } else {
            None
        };

        Ok(Self {
            config,
            counters: (0..num_counters).map(|_| AtomicU64::new(0)).collect(),
            num_counters,
            num_hash_functions,
            hash_seeds,
            rng,
            metrics,
            element_count: AtomicU64::new(0),
        })
    }

    /// Insert an element
    pub fn insert<T: Hash>(&mut self, item: &T) {
        for &seed in &self.hash_seeds {
            let hash = self.hash_with_seed(item, seed);
            let counter_index = (hash % self.num_counters as u64) as usize;

            // Increment counter (saturate at 15 for 4-bit counter)
            let current = self.counters[counter_index].load(Ordering::Relaxed);
            if current < 15 {
                self.counters[counter_index].fetch_add(1, Ordering::Relaxed);
            }
        }

        self.element_count.fetch_add(1, Ordering::Relaxed);

        if let Some(ref metrics) = self.metrics {
            metrics.inserts.inc();
        }
    }

    /// Delete an element
    pub fn delete<T: Hash>(&mut self, item: &T) {
        for &seed in &self.hash_seeds {
            let hash = self.hash_with_seed(item, seed);
            let counter_index = (hash % self.num_counters as u64) as usize;

            // Decrement counter (don't go below 0)
            let current = self.counters[counter_index].load(Ordering::Relaxed);
            if current > 0 {
                self.counters[counter_index].fetch_sub(1, Ordering::Relaxed);
            }
        }

        self.element_count.fetch_sub(1, Ordering::Relaxed);

        if let Some(ref metrics) = self.metrics {
            metrics.deletes.inc();
        }
    }

    /// Test membership
    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        let start = std::time::Instant::now();

        let result = self.hash_seeds.iter().all(|&seed| {
            let hash = self.hash_with_seed(item, seed);
            let counter_index = (hash % self.num_counters as u64) as usize;
            self.counters[counter_index].load(Ordering::Relaxed) > 0
        });

        if let Some(ref metrics) = self.metrics {
            metrics.lookups.inc();
            let elapsed = start.elapsed();
            metrics.lookup_latency.observe(elapsed.as_micros() as f64);
        }

        result
    }

    /// Get statistics
    pub fn stats(&self) -> BloomFilterStats {
        let total_counters = self.num_counters;
        let non_zero_counters = self
            .counters
            .iter()
            .filter(|c| c.load(Ordering::Relaxed) > 0)
            .count();

        let element_count = self.element_count.load(Ordering::Relaxed);

        // Calculate estimated FPR: (1 - e^(-kn/m))^k
        // where k = num_hash_functions, n = element_count, m = num_counters
        let estimated_fpr = if element_count > 0 {
            let k = self.num_hash_functions as f64;
            let n = element_count as f64;
            let m = self.num_counters as f64;
            let exponent = -k * n / m;
            let base = 1.0 - exponent.exp();
            base.powf(k)
        } else {
            0.0
        };

        BloomFilterStats {
            num_bits: self.num_counters, // Use counters instead of bits
            num_hash_functions: self.num_hash_functions,
            element_count,
            fill_rate: non_zero_counters as f64 / total_counters as f64,
            estimated_fpr,
            configured_fpr: self.config.false_positive_rate,
        }
    }

    /// Hash with seed
    fn hash_with_seed<T: Hash>(&self, item: &T, seed: u64) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        seed.hash(&mut hasher);
        item.hash(&mut hasher);
        hasher.finish()
    }
}

/// Bloom filter statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomFilterStats {
    /// Number of bits/counters
    pub num_bits: usize,
    /// Number of hash functions
    pub num_hash_functions: usize,
    /// Estimated element count
    pub element_count: u64,
    /// Fill rate (0.0 - 1.0)
    pub fill_rate: f64,
    /// Estimated false positive rate
    pub estimated_fpr: f64,
    /// Configured false positive rate
    pub configured_fpr: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_filter_basic() -> Result<()> {
        let config = BloomFilterConfig {
            expected_elements: 100,
            false_positive_rate: 0.01,
            ..Default::default()
        };

        let mut filter = BloomFilter::new(config)?;

        // Insert elements
        filter.insert(&"hello");
        filter.insert(&"world");
        filter.insert(&123u64);

        // Test membership
        assert!(filter.contains(&"hello"));
        assert!(filter.contains(&"world"));
        assert!(filter.contains(&123u64));
        assert!(!filter.contains(&"not_present"));

        Ok(())
    }

    #[test]
    fn test_bloom_filter_stats() -> Result<()> {
        let config = BloomFilterConfig::default();
        let mut filter = BloomFilter::new(config)?;

        for i in 0..100 {
            filter.insert(&i);
        }

        let stats = filter.stats();
        assert_eq!(stats.element_count, 100);
        assert!(stats.fill_rate > 0.0);
        assert!(stats.fill_rate < 1.0);

        Ok(())
    }

    #[test]
    fn test_counting_bloom_filter() -> Result<()> {
        let config = BloomFilterConfig {
            expected_elements: 100,
            enable_counting: true,
            ..Default::default()
        };

        let mut filter = CountingBloomFilter::new(config)?;

        // Insert and delete
        filter.insert(&"test");
        assert!(filter.contains(&"test"));

        filter.delete(&"test");
        assert!(!filter.contains(&"test"));

        Ok(())
    }

    #[test]
    fn test_false_positive_rate() -> Result<()> {
        let config = BloomFilterConfig {
            expected_elements: 1000,
            false_positive_rate: 0.01,
            ..Default::default()
        };

        let mut filter = BloomFilter::new(config)?;

        // Insert 1000 elements
        for i in 0..1000 {
            filter.insert(&i);
        }

        // Test with 1000 non-inserted elements
        let mut false_positives = 0;
        for i in 1000..2000 {
            if filter.contains(&i) {
                false_positives += 1;
            }
        }

        let actual_fpr = false_positives as f64 / 1000.0;

        // Should be close to configured rate (with some tolerance)
        assert!(actual_fpr < 0.05); // Less than 5%

        Ok(())
    }

    #[test]
    fn test_clear() -> Result<()> {
        let config = BloomFilterConfig::default();
        let mut filter = BloomFilter::new(config)?;

        filter.insert(&"test");
        assert!(filter.contains(&"test"));

        filter.clear();
        assert!(!filter.contains(&"test"));

        let stats = filter.stats();
        assert_eq!(stats.element_count, 0);
        assert_eq!(stats.fill_rate, 0.0);

        Ok(())
    }

    #[test]
    fn test_config_calculations() {
        let config = BloomFilterConfig {
            expected_elements: 10_000,
            false_positive_rate: 0.01,
            ..Default::default()
        };

        let bits = config.calculate_bit_array_size();
        let hash_funcs = config.calculate_num_hash_functions();

        assert!(bits > 0);
        assert!(hash_funcs > 0);
        assert!(hash_funcs < 20); // Reasonable range
    }

    #[test]
    fn test_multiple_insertions() -> Result<()> {
        let config = BloomFilterConfig::default();
        let mut filter = BloomFilter::new(config)?;

        // Insert same element multiple times (should be idempotent)
        filter.insert(&"test");
        filter.insert(&"test");
        filter.insert(&"test");

        assert!(filter.contains(&"test"));

        Ok(())
    }

    #[test]
    fn test_counting_filter_multiple_ops() -> Result<()> {
        let config = BloomFilterConfig {
            enable_counting: true,
            ..Default::default()
        };

        let mut filter = CountingBloomFilter::new(config)?;

        // Insert multiple times
        filter.insert(&"key");
        filter.insert(&"key");
        assert!(filter.contains(&"key"));

        // Delete once
        filter.delete(&"key");
        assert!(filter.contains(&"key")); // Still there

        // Delete again
        filter.delete(&"key");
        assert!(!filter.contains(&"key")); // Now gone

        Ok(())
    }
}
