//! # Advanced Sampling Techniques for Stream Processing
//!
//! Production-grade probabilistic data structures and sampling algorithms
//! for high-volume streaming scenarios where exact computation is too expensive.
//!
//! ## Features
//!
//! - **Reservoir Sampling**: Fixed-size random samples from unbounded streams
//! - **Stratified Sampling**: Distribution-preserving sampling across categories
//! - **HyperLogLog**: Approximate cardinality estimation with O(1) space
//! - **Count-Min Sketch**: Approximate frequency counting for heavy hitters
//! - **T-Digest**: Approximate percentile calculations for streaming data
//! - **Bloom Filter**: Space-efficient probabilistic membership testing
//!
//! ## Use Cases
//!
//! - Real-time analytics on billion-event streams
//! - Memory-efficient distinct counting
//! - Top-K heavy hitter detection
//! - Approximate quantile tracking
//! - Duplicate detection with minimal memory

use crate::StreamEvent;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Configuration for advanced sampling operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Reservoir sampling size
    pub reservoir_size: usize,
    /// Number of hash functions for Count-Min Sketch
    pub cms_hash_count: usize,
    /// Width of Count-Min Sketch
    pub cms_width: usize,
    /// Number of registers for HyperLogLog (power of 2)
    pub hll_precision: u8,
    /// Compression parameter for T-Digest
    pub tdigest_delta: f64,
    /// Size of Bloom filter in bits
    pub bloom_filter_bits: usize,
    /// Number of hash functions for Bloom filter
    pub bloom_filter_hashes: usize,
    /// Stratified sampling categories
    pub stratified_categories: Vec<String>,
    /// Sample rate per category (0.0 to 1.0)
    pub stratified_sample_rates: HashMap<String, f64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            reservoir_size: 1000,
            cms_hash_count: 4,
            cms_width: 10000,
            hll_precision: 14, // 16K registers
            tdigest_delta: 0.01,
            bloom_filter_bits: 100000,
            bloom_filter_hashes: 7,
            stratified_categories: Vec::new(),
            stratified_sample_rates: HashMap::new(),
        }
    }
}

/// Reservoir Sampler - Maintains uniform random sample from unbounded stream
///
/// Uses Algorithm R (Vitter, 1985) for efficient single-pass sampling.
#[derive(Debug, Clone)]
pub struct ReservoirSampler {
    reservoir: Vec<StreamEvent>,
    capacity: usize,
    count: u64,
}

impl ReservoirSampler {
    /// Create a new reservoir sampler
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of samples to retain
    pub fn new(capacity: usize) -> Self {
        Self {
            reservoir: Vec::with_capacity(capacity),
            capacity,
            count: 0,
        }
    }

    /// Add an event to the reservoir
    ///
    /// Uses Algorithm R for O(1) average insertion time
    pub fn add(&mut self, event: StreamEvent) {
        self.count += 1;

        if self.reservoir.len() < self.capacity {
            // Reservoir not full yet, just add
            self.reservoir.push(event);
        } else {
            // Randomly replace existing event with decreasing probability
            let j = (fastrand::f64() * self.count as f64) as usize;
            if j < self.capacity {
                self.reservoir[j] = event;
            }
        }
    }

    /// Get the current sample
    pub fn sample(&self) -> &[StreamEvent] {
        &self.reservoir
    }

    /// Get the number of events processed
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Clear the reservoir
    pub fn clear(&mut self) {
        self.reservoir.clear();
        self.count = 0;
    }

    /// Get statistics about the sampler
    pub fn stats(&self) -> ReservoirStats {
        ReservoirStats {
            capacity: self.capacity,
            current_size: self.reservoir.len(),
            total_events: self.count,
            sampling_rate: if self.count > 0 {
                self.reservoir.len() as f64 / self.count as f64
            } else {
                0.0
            },
        }
    }
}

/// Statistics for reservoir sampler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReservoirStats {
    pub capacity: usize,
    pub current_size: usize,
    pub total_events: u64,
    pub sampling_rate: f64,
}

/// Stratified Sampler - Preserves distribution across categories
///
/// Maintains separate reservoirs for each category to ensure
/// representative sampling across different event types.
#[derive(Debug, Clone)]
pub struct StratifiedSampler {
    samplers: HashMap<String, ReservoirSampler>,
    sample_rates: HashMap<String, f64>,
    default_capacity: usize,
    category_extractor: fn(&StreamEvent) -> Option<String>,
}

impl StratifiedSampler {
    /// Create a new stratified sampler
    ///
    /// # Arguments
    /// * `default_capacity` - Default reservoir size per category
    /// * `category_extractor` - Function to extract category from event
    pub fn new(
        default_capacity: usize,
        category_extractor: fn(&StreamEvent) -> Option<String>,
    ) -> Self {
        Self {
            samplers: HashMap::new(),
            sample_rates: HashMap::new(),
            default_capacity,
            category_extractor,
        }
    }

    /// Set sample rate for a specific category
    pub fn set_category_rate(&mut self, category: String, rate: f64) {
        assert!((0.0..=1.0).contains(&rate), "Rate must be in [0, 1]");
        self.sample_rates.insert(category, rate);
    }

    /// Add an event to the appropriate category reservoir
    pub fn add(&mut self, event: StreamEvent) {
        if let Some(category) = (self.category_extractor)(&event) {
            // Check if we should sample this category
            let rate = self.sample_rates.get(&category).copied().unwrap_or(1.0);

            if rate <= 0.0 {
                return; // Skip this category
            }

            // Get or create sampler for this category
            let sampler = self.samplers.entry(category.clone()).or_insert_with(|| {
                let capacity = (self.default_capacity as f64 * rate) as usize;
                ReservoirSampler::new(capacity.max(1))
            });

            sampler.add(event);
        }
    }

    /// Get samples for a specific category
    pub fn category_sample(&self, category: &str) -> Option<&[StreamEvent]> {
        self.samplers.get(category).map(|s| s.sample())
    }

    /// Get all samples grouped by category
    pub fn all_samples(&self) -> HashMap<String, Vec<StreamEvent>> {
        self.samplers
            .iter()
            .map(|(cat, sampler)| (cat.clone(), sampler.sample().to_vec()))
            .collect()
    }

    /// Get statistics for all categories
    pub fn stats(&self) -> StratifiedStats {
        let category_stats = self
            .samplers
            .iter()
            .map(|(cat, sampler)| (cat.clone(), sampler.stats()))
            .collect();

        StratifiedStats {
            category_count: self.samplers.len(),
            category_stats,
        }
    }
}

/// Statistics for stratified sampler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratifiedStats {
    pub category_count: usize,
    pub category_stats: HashMap<String, ReservoirStats>,
}

/// HyperLogLog - Approximate cardinality estimator
///
/// Uses probabilistic counting with 2^precision registers to estimate
/// the number of distinct elements with ~1.04/sqrt(m) relative error.
#[derive(Debug, Clone)]
pub struct HyperLogLog {
    registers: Vec<u8>,
    precision: u8,
    alpha: f64,
}

impl HyperLogLog {
    /// Create a new HyperLogLog estimator
    ///
    /// # Arguments
    /// * `precision` - Number of bits for register indexing (4-16)
    ///   Higher precision = more accuracy but more memory
    pub fn new(precision: u8) -> Self {
        assert!(
            (4..=16).contains(&precision),
            "Precision must be between 4 and 16"
        );

        let m = 1 << precision; // 2^precision registers

        // Alpha constant for bias correction
        let alpha = match m {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / m as f64),
        };

        Self {
            registers: vec![0; m],
            precision,
            alpha,
        }
    }

    /// Add an element to the estimator
    pub fn add<T: Hash>(&mut self, element: &T) {
        let hash = self.hash(element);

        // Use first 'precision' bits for register index
        let idx = (hash >> (64 - self.precision)) as usize;

        // Count leading zeros in remaining bits + 1
        let remaining = hash << self.precision;
        let leading_zeros = remaining.leading_zeros() as u8 + 1;

        // Update register with maximum leading zeros seen
        self.registers[idx] = self.registers[idx].max(leading_zeros);
    }

    /// Estimate the cardinality (number of distinct elements)
    pub fn cardinality(&self) -> u64 {
        let m = self.registers.len() as f64;

        // Harmonic mean of 2^register values
        let raw_estimate = self.alpha * m * m
            / self
                .registers
                .iter()
                .map(|&r| 2.0_f64.powi(-(r as i32)))
                .sum::<f64>();

        // Apply bias correction
        if raw_estimate <= 5.0 * m {
            // Small range correction
            let zeros = self.registers.iter().filter(|&&r| r == 0).count() as f64;
            if zeros > 0.0 {
                return (m * (m / zeros).ln()) as u64;
            }
        }

        if raw_estimate <= (1.0 / 30.0) * (1u64 << 32) as f64 {
            // No correction
            raw_estimate as u64
        } else {
            // Large range correction
            let two_32 = (1u64 << 32) as f64;
            (-(two_32) * ((1.0 - raw_estimate / two_32).ln())) as u64
        }
    }

    /// Merge another HyperLogLog into this one
    pub fn merge(&mut self, other: &HyperLogLog) {
        assert_eq!(
            self.precision, other.precision,
            "Cannot merge HyperLogLogs with different precisions"
        );

        for (i, &other_val) in other.registers.iter().enumerate() {
            self.registers[i] = self.registers[i].max(other_val);
        }
    }

    /// Hash function for HyperLogLog
    fn hash<T: Hash>(&self, element: &T) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        element.hash(&mut hasher);
        hasher.finish()
    }

    /// Get statistics about the HyperLogLog
    pub fn stats(&self) -> HyperLogLogStats {
        HyperLogLogStats {
            precision: self.precision,
            register_count: self.registers.len(),
            estimated_cardinality: self.cardinality(),
            memory_bytes: self.registers.len(),
        }
    }
}

/// Statistics for HyperLogLog
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperLogLogStats {
    pub precision: u8,
    pub register_count: usize,
    pub estimated_cardinality: u64,
    pub memory_bytes: usize,
}

/// Count-Min Sketch - Approximate frequency counter
///
/// Space-efficient probabilistic data structure for estimating
/// frequency of elements in a stream with guaranteed error bounds.
#[derive(Debug, Clone)]
pub struct CountMinSketch {
    table: Vec<Vec<u64>>,
    hash_count: usize,
    width: usize,
    total_count: u64,
}

impl CountMinSketch {
    /// Create a new Count-Min Sketch
    ///
    /// # Arguments
    /// * `hash_count` - Number of hash functions (depth)
    /// * `width` - Number of counters per hash function
    ///
    /// Error bounds: ε = e / width, δ = 1 / e^hash_count
    pub fn new(hash_count: usize, width: usize) -> Self {
        Self {
            table: vec![vec![0; width]; hash_count],
            hash_count,
            width,
            total_count: 0,
        }
    }

    /// Add an element with count
    pub fn add<T: Hash>(&mut self, element: &T, count: u64) {
        self.total_count += count;

        for i in 0..self.hash_count {
            let idx = self.hash_i(element, i) % self.width;
            self.table[i][idx] += count;
        }
    }

    /// Estimate the frequency of an element
    pub fn estimate<T: Hash>(&self, element: &T) -> u64 {
        (0..self.hash_count)
            .map(|i| {
                let idx = self.hash_i(element, i) % self.width;
                self.table[i][idx]
            })
            .min()
            .unwrap_or(0)
    }

    /// Get the total count of all elements
    pub fn total_count(&self) -> u64 {
        self.total_count
    }

    /// Hash function with index
    fn hash_i<T: Hash>(&self, element: &T, i: usize) -> usize {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        element.hash(&mut hasher);
        i.hash(&mut hasher);
        hasher.finish() as usize
    }

    /// Get statistics about the Count-Min Sketch
    pub fn stats(&self) -> CountMinSketchStats {
        CountMinSketchStats {
            hash_count: self.hash_count,
            width: self.width,
            total_count: self.total_count,
            memory_bytes: self.hash_count * self.width * std::mem::size_of::<u64>(),
        }
    }
}

/// Statistics for Count-Min Sketch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountMinSketchStats {
    pub hash_count: usize,
    pub width: usize,
    pub total_count: u64,
    pub memory_bytes: usize,
}

/// T-Digest - Approximate quantile estimator
///
/// Provides accurate percentile estimates with compression
/// for streaming data. More accurate at extremes (p0, p100).
#[derive(Debug, Clone)]
pub struct TDigest {
    centroids: Vec<Centroid>,
    delta: f64,
    total_weight: f64,
    max_size: usize,
}

#[derive(Debug, Clone, Copy)]
struct Centroid {
    mean: f64,
    weight: f64,
}

impl TDigest {
    /// Create a new T-Digest
    ///
    /// # Arguments
    /// * `delta` - Compression parameter (0.01 = 1% accuracy)
    pub fn new(delta: f64) -> Self {
        Self {
            centroids: Vec::new(),
            delta,
            total_weight: 0.0,
            max_size: (1.0 / delta) as usize,
        }
    }

    /// Add a value to the digest
    pub fn add(&mut self, value: f64, weight: f64) {
        self.centroids.push(Centroid {
            mean: value,
            weight,
        });
        self.total_weight += weight;

        // Compress if needed
        if self.centroids.len() > self.max_size {
            self.compress();
        }
    }

    /// Estimate the quantile (0.0 to 1.0)
    pub fn quantile(&mut self, q: f64) -> Option<f64> {
        if self.centroids.is_empty() {
            return None;
        }

        if self.centroids.len() > 1 {
            self.compress();
        }

        let index = q * self.total_weight;
        let mut sum = 0.0;

        for centroid in &self.centroids {
            sum += centroid.weight;
            if sum >= index {
                return Some(centroid.mean);
            }
        }

        self.centroids.last().map(|c| c.mean)
    }

    /// Compress centroids
    fn compress(&mut self) {
        if self.centroids.is_empty() {
            return;
        }

        // Sort by mean
        self.centroids
            .sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());

        let mut compressed = Vec::new();
        let mut current = self.centroids[0];

        for &centroid in &self.centroids[1..] {
            // Check if we should merge
            let q = (current.weight + centroid.weight) / self.total_weight;
            let k = self.k_limit(q);

            if current.weight + centroid.weight <= k {
                // Merge centroids
                let total_weight = current.weight + centroid.weight;
                current.mean = (current.mean * current.weight + centroid.mean * centroid.weight)
                    / total_weight;
                current.weight = total_weight;
            } else {
                compressed.push(current);
                current = centroid;
            }
        }
        compressed.push(current);

        self.centroids = compressed;
    }

    /// Compute k limit for compression
    fn k_limit(&self, q: f64) -> f64 {
        4.0 * self.total_weight * self.delta * q * (1.0 - q)
    }

    /// Get statistics about the T-Digest
    pub fn stats(&self) -> TDigestStats {
        TDigestStats {
            centroid_count: self.centroids.len(),
            total_weight: self.total_weight,
            delta: self.delta,
            max_size: self.max_size,
        }
    }
}

/// Statistics for T-Digest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TDigestStats {
    pub centroid_count: usize,
    pub total_weight: f64,
    pub delta: f64,
    pub max_size: usize,
}

/// Bloom Filter - Probabilistic membership test
///
/// Space-efficient set membership test with no false negatives
/// but possible false positives (configurable error rate).
#[derive(Debug, Clone)]
pub struct BloomFilter {
    bits: Vec<bool>,
    hash_count: usize,
    insert_count: u64,
}

impl BloomFilter {
    /// Create a new Bloom filter
    ///
    /// # Arguments
    /// * `size` - Number of bits in the filter
    /// * `hash_count` - Number of hash functions
    ///
    /// Optimal hash_count = (bits / expected_items) * ln(2)
    pub fn new(size: usize, hash_count: usize) -> Self {
        Self {
            bits: vec![false; size],
            hash_count,
            insert_count: 0,
        }
    }

    /// Optimal Bloom filter for expected items and false positive rate
    pub fn optimal(expected_items: usize, false_positive_rate: f64) -> Self {
        let bits = Self::optimal_bits(expected_items, false_positive_rate);
        let hash_count = Self::optimal_hash_count(bits, expected_items);
        Self::new(bits, hash_count)
    }

    /// Calculate optimal number of bits
    fn optimal_bits(n: usize, p: f64) -> usize {
        let numerator = -(n as f64 * p.ln());
        let denominator = 2.0_f64.ln().powi(2);
        (numerator / denominator).ceil() as usize
    }

    /// Calculate optimal number of hash functions
    fn optimal_hash_count(m: usize, n: usize) -> usize {
        ((m as f64 / n as f64) * 2.0_f64.ln()).ceil() as usize
    }

    /// Add an element to the filter
    pub fn add<T: Hash>(&mut self, element: &T) {
        self.insert_count += 1;
        for i in 0..self.hash_count {
            let idx = self.hash_i(element, i) % self.bits.len();
            self.bits[idx] = true;
        }
    }

    /// Check if an element might be in the set
    pub fn contains<T: Hash>(&self, element: &T) -> bool {
        (0..self.hash_count).all(|i| {
            let idx = self.hash_i(element, i) % self.bits.len();
            self.bits[idx]
        })
    }

    /// Hash function with index
    fn hash_i<T: Hash>(&self, element: &T, i: usize) -> usize {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        element.hash(&mut hasher);
        i.hash(&mut hasher);
        hasher.finish() as usize
    }

    /// Estimate current false positive rate
    pub fn false_positive_rate(&self) -> f64 {
        let set_bits = self.bits.iter().filter(|&&b| b).count() as f64;
        let p = set_bits / self.bits.len() as f64;
        p.powi(self.hash_count as i32)
    }

    /// Get statistics about the Bloom filter
    pub fn stats(&self) -> BloomFilterStats {
        let set_bits = self.bits.iter().filter(|&&b| b).count();

        BloomFilterStats {
            size_bits: self.bits.len(),
            hash_count: self.hash_count,
            insert_count: self.insert_count,
            set_bits,
            estimated_fpr: self.false_positive_rate(),
            memory_bytes: self.bits.len() / 8,
        }
    }
}

/// Statistics for Bloom filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomFilterStats {
    pub size_bits: usize,
    pub hash_count: usize,
    pub insert_count: u64,
    pub set_bits: usize,
    pub estimated_fpr: f64,
    pub memory_bytes: usize,
}

/// Unified sampling manager for all sampling techniques
pub struct AdvancedSamplingManager {
    config: SamplingConfig,
    reservoir: ReservoirSampler,
    stratified: Option<StratifiedSampler>,
    hyperloglog: HyperLogLog,
    count_min: CountMinSketch,
    tdigest: TDigest,
    bloom: BloomFilter,
    event_count: u64,
}

impl AdvancedSamplingManager {
    /// Create a new sampling manager with the given configuration
    pub fn new(config: SamplingConfig) -> Self {
        let reservoir = ReservoirSampler::new(config.reservoir_size);
        let hyperloglog = HyperLogLog::new(config.hll_precision);
        let count_min = CountMinSketch::new(config.cms_hash_count, config.cms_width);
        let tdigest = TDigest::new(config.tdigest_delta);
        let bloom = BloomFilter::new(config.bloom_filter_bits, config.bloom_filter_hashes);

        Self {
            config,
            reservoir,
            stratified: None,
            hyperloglog,
            count_min,
            tdigest,
            bloom,
            event_count: 0,
        }
    }

    /// Enable stratified sampling with a category extractor function
    pub fn enable_stratified(&mut self, extractor: fn(&StreamEvent) -> Option<String>) {
        let mut sampler = StratifiedSampler::new(self.config.reservoir_size, extractor);

        // Apply configured sample rates
        for (category, rate) in &self.config.stratified_sample_rates {
            sampler.set_category_rate(category.clone(), *rate);
        }

        self.stratified = Some(sampler);
    }

    /// Process an event through all sampling techniques
    pub fn process_event(&mut self, event: StreamEvent) -> Result<()> {
        self.event_count += 1;

        // Reservoir sampling
        self.reservoir.add(event.clone());

        // Stratified sampling
        if let Some(ref mut stratified) = self.stratified {
            stratified.add(event.clone());
        }

        // HyperLogLog (cardinality)
        let event_id = self.event_id(&event);
        self.hyperloglog.add(&event_id);

        // Count-Min Sketch (frequency)
        self.count_min.add(&event_id, 1);

        // T-Digest (quantiles) - using event timestamp as value
        if let Some(value) = self.extract_numeric_value(&event) {
            self.tdigest.add(value, 1.0);
        }

        // Bloom filter (membership)
        self.bloom.add(&event_id);

        Ok(())
    }

    /// Get reservoir sample
    pub fn reservoir_sample(&self) -> &[StreamEvent] {
        self.reservoir.sample()
    }

    /// Get stratified samples
    pub fn stratified_samples(&self) -> Option<HashMap<String, Vec<StreamEvent>>> {
        self.stratified.as_ref().map(|s| s.all_samples())
    }

    /// Estimate distinct event count
    pub fn distinct_count(&self) -> u64 {
        self.hyperloglog.cardinality()
    }

    /// Estimate frequency of an event
    pub fn event_frequency(&self, event: &StreamEvent) -> u64 {
        let event_id = self.event_id(event);
        self.count_min.estimate(&event_id)
    }

    /// Check if an event was likely seen before
    pub fn likely_seen(&self, event: &StreamEvent) -> bool {
        let event_id = self.event_id(event);
        self.bloom.contains(&event_id)
    }

    /// Estimate quantile (percentile) of numeric values
    pub fn quantile(&mut self, q: f64) -> Option<f64> {
        self.tdigest.quantile(q)
    }

    /// Get comprehensive statistics
    pub fn stats(&self) -> SamplingManagerStats {
        SamplingManagerStats {
            event_count: self.event_count,
            reservoir_stats: self.reservoir.stats(),
            stratified_stats: self.stratified.as_ref().map(|s| s.stats()),
            hyperloglog_stats: self.hyperloglog.stats(),
            count_min_stats: self.count_min.stats(),
            tdigest_stats: self.tdigest.stats(),
            bloom_stats: self.bloom.stats(),
        }
    }

    /// Extract event ID for hashing
    fn event_id(&self, event: &StreamEvent) -> String {
        match event {
            StreamEvent::TripleAdded {
                subject,
                predicate,
                object,
                ..
            } => format!("{}-{}-{}", subject, predicate, object),
            StreamEvent::TripleRemoved {
                subject,
                predicate,
                object,
                ..
            } => format!("{}-{}-{}", subject, predicate, object),
            StreamEvent::GraphCreated { graph, .. } => format!("graph-{}", graph),
            StreamEvent::GraphDeleted { graph, .. } => format!("graph-{}", graph),
            _ => "unknown".to_string(),
        }
    }

    /// Extract numeric value from event for quantile estimation
    fn extract_numeric_value(&self, event: &StreamEvent) -> Option<f64> {
        match event {
            StreamEvent::TripleAdded { metadata, .. }
            | StreamEvent::TripleRemoved { metadata, .. } => {
                Some(metadata.timestamp.timestamp() as f64)
            }
            _ => None,
        }
    }
}

/// Comprehensive statistics for all sampling techniques
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingManagerStats {
    pub event_count: u64,
    pub reservoir_stats: ReservoirStats,
    pub stratified_stats: Option<StratifiedStats>,
    pub hyperloglog_stats: HyperLogLogStats,
    pub count_min_stats: CountMinSketchStats,
    pub tdigest_stats: TDigestStats,
    pub bloom_stats: BloomFilterStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EventMetadata;
    use std::collections::HashMap;

    fn create_test_event(id: &str) -> StreamEvent {
        StreamEvent::TripleAdded {
            subject: format!("http://example.org/{}", id),
            predicate: "http://example.org/prop".to_string(),
            object: "value".to_string(),
            graph: None,
            metadata: EventMetadata {
                event_id: id.to_string(),
                timestamp: chrono::Utc::now(),
                source: "test".to_string(),
                user: None,
                context: None,
                caused_by: None,
                version: "1.0".to_string(),
                properties: HashMap::new(),
                checksum: None,
            },
        }
    }

    #[test]
    fn test_reservoir_sampler() {
        let mut sampler = ReservoirSampler::new(10);

        // Add 100 events
        for i in 0..100 {
            sampler.add(create_test_event(&format!("event-{}", i)));
        }

        let stats = sampler.stats();
        assert_eq!(stats.capacity, 10);
        assert_eq!(stats.current_size, 10);
        assert_eq!(stats.total_events, 100);
        assert_eq!(stats.sampling_rate, 0.1);
    }

    #[test]
    fn test_stratified_sampler() {
        fn category_extractor(event: &StreamEvent) -> Option<String> {
            match event {
                StreamEvent::TripleAdded { metadata, .. } => Some(metadata.source.clone()),
                _ => None,
            }
        }

        let mut sampler = StratifiedSampler::new(10, category_extractor);
        sampler.set_category_rate("source1".to_string(), 0.5);
        sampler.set_category_rate("source2".to_string(), 1.0);

        // Add events from different sources
        for i in 0..50 {
            let mut event = create_test_event(&format!("event-{}", i));
            if let StreamEvent::TripleAdded { metadata, .. } = &mut event {
                metadata.source = if i < 25 {
                    "source1".to_string()
                } else {
                    "source2".to_string()
                };
            }
            sampler.add(event);
        }

        let stats = sampler.stats();
        assert_eq!(stats.category_count, 2);
        assert!(stats.category_stats.contains_key("source1"));
        assert!(stats.category_stats.contains_key("source2"));
    }

    #[test]
    fn test_hyperloglog() {
        let mut hll = HyperLogLog::new(14);

        // Add 1000 distinct elements
        for i in 0..1000 {
            hll.add(&format!("element-{}", i));
        }

        let cardinality = hll.cardinality();

        // HyperLogLog should estimate within ~2% error
        let error = ((cardinality as i64 - 1000).abs() as f64) / 1000.0;
        assert!(error < 0.05, "Error: {}, Estimated: {}", error, cardinality);
    }

    #[test]
    fn test_count_min_sketch() {
        let mut cms = CountMinSketch::new(4, 1000);

        // Add elements with different frequencies
        for _ in 0..100 {
            cms.add(&"frequent", 1);
        }
        for _ in 0..10 {
            cms.add(&"rare", 1);
        }

        let freq_frequent = cms.estimate(&"frequent");
        let freq_rare = cms.estimate(&"rare");

        assert!(freq_frequent >= 100);
        assert!(freq_rare >= 10);
        assert!(freq_frequent > freq_rare);
    }

    #[test]
    fn test_tdigest() {
        let mut digest = TDigest::new(0.01);

        // Add values 1..1000
        for i in 1..=1000 {
            digest.add(i as f64, 1.0);
        }

        // Test median (should be around 500)
        let median = digest.quantile(0.5).unwrap();
        assert!((median - 500.0).abs() < 50.0, "Median: {}", median);

        // Test P90 (should be around 900)
        let p90 = digest.quantile(0.9).unwrap();
        assert!((p90 - 900.0).abs() < 100.0, "P90: {}", p90);
    }

    #[test]
    fn test_bloom_filter() {
        let mut bloom = BloomFilter::optimal(1000, 0.01);

        // Add 500 elements
        for i in 0..500 {
            bloom.add(&format!("element-{}", i));
        }

        // All added elements should be found
        for i in 0..500 {
            assert!(bloom.contains(&format!("element-{}", i)));
        }

        // Test false positive rate
        let mut false_positives = 0;
        for i in 1000..2000 {
            if bloom.contains(&format!("element-{}", i)) {
                false_positives += 1;
            }
        }

        let fpr = false_positives as f64 / 1000.0;
        assert!(fpr < 0.05, "False positive rate too high: {}", fpr);
    }

    #[test]
    fn test_sampling_manager() {
        let config = SamplingConfig::default();
        let mut manager = AdvancedSamplingManager::new(config);

        // Process 100 events
        for i in 0..100 {
            let event = create_test_event(&format!("event-{}", i));
            manager.process_event(event).unwrap();
        }

        let stats = manager.stats();
        assert_eq!(stats.event_count, 100);
        assert!(stats.reservoir_stats.current_size > 0);
        assert!(stats.hyperloglog_stats.estimated_cardinality > 0);
        assert!(stats.count_min_stats.total_count > 0);
    }

    #[test]
    fn test_hyperloglog_merge() {
        let mut hll1 = HyperLogLog::new(14);
        let mut hll2 = HyperLogLog::new(14);

        // Add different elements to each
        for i in 0..500 {
            hll1.add(&format!("element-{}", i));
        }
        for i in 500..1000 {
            hll2.add(&format!("element-{}", i));
        }

        // Merge
        hll1.merge(&hll2);

        let cardinality = hll1.cardinality();

        // Should estimate ~1000 distinct elements
        let error = ((cardinality as i64 - 1000).abs() as f64) / 1000.0;
        assert!(error < 0.05, "Error: {}, Estimated: {}", error, cardinality);
    }

    #[test]
    fn test_bloom_filter_optimal() {
        let bloom = BloomFilter::optimal(10000, 0.01);
        let stats = bloom.stats();

        // Verify optimal sizing
        assert!(stats.size_bits > 0);
        assert!(stats.hash_count > 0);
    }

    #[test]
    fn test_sampling_manager_with_stratified() {
        fn category_extractor(event: &StreamEvent) -> Option<String> {
            match event {
                StreamEvent::TripleAdded { subject, .. } => {
                    if subject.contains("type1") {
                        Some("type1".to_string())
                    } else if subject.contains("type2") {
                        Some("type2".to_string())
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }

        let config = SamplingConfig::default();
        let mut manager = AdvancedSamplingManager::new(config);
        manager.enable_stratified(category_extractor);

        // Process events from different categories
        for i in 0..50 {
            let event = StreamEvent::TripleAdded {
                subject: format!("http://example.org/type1/{}", i),
                predicate: "http://example.org/prop".to_string(),
                object: "value".to_string(),
                graph: None,
                metadata: EventMetadata {
                    event_id: format!("event-{}", i),
                    timestamp: chrono::Utc::now(),
                    source: "test".to_string(),
                    user: None,
                    context: None,
                    caused_by: None,
                    version: "1.0".to_string(),
                    properties: HashMap::new(),
                    checksum: None,
                },
            };
            manager.process_event(event).unwrap();
        }

        for i in 50..100 {
            let event = StreamEvent::TripleAdded {
                subject: format!("http://example.org/type2/{}", i),
                predicate: "http://example.org/prop".to_string(),
                object: "value".to_string(),
                graph: None,
                metadata: EventMetadata {
                    event_id: format!("event-{}", i),
                    timestamp: chrono::Utc::now(),
                    source: "test".to_string(),
                    user: None,
                    context: None,
                    caused_by: None,
                    version: "1.0".to_string(),
                    properties: HashMap::new(),
                    checksum: None,
                },
            };
            manager.process_event(event).unwrap();
        }

        let stats = manager.stats();
        assert_eq!(stats.event_count, 100);
        assert!(stats.stratified_stats.is_some());

        let stratified = stats.stratified_stats.unwrap();
        assert_eq!(stratified.category_count, 2);
    }
}
