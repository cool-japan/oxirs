# Advanced Sampling Techniques - Quick Reference Guide

**Production-grade probabilistic data structures for high-volume RDF stream analytics**

## Overview

The Advanced Sampling module provides memory-efficient algorithms for analyzing billion-event streams with fixed memory footprints and provable error bounds. These techniques enable real-time analytics that would be impossible with exact computation.

## Quick Start

```rust
use oxirs_stream::{AdvancedSamplingManager, SamplingConfig, StreamEvent};

// Create a sampling manager with default configuration
let config = SamplingConfig::default();
let mut manager = AdvancedSamplingManager::new(config);

// Process streaming events
for event in stream {
    manager.process_event(event)?;
}

// Get analytics results with minimal memory
let distinct_count = manager.distinct_count();      // ~50,000
let p99_latency = manager.quantile(0.99)?;          // ~500ms
let event_freq = manager.event_frequency(&event);   // ~1,234
let sample = manager.reservoir_sample();             // 1,000 events
```

## Algorithm Selection Guide

### Use **Reservoir Sampling** when you need:
- Uniform random sample from unbounded stream
- Fixed memory footprint (O(k) space for k-sized sample)
- Statistical analysis on representative subset
- **Example**: Sample 1,000 events from billion-event stream

```rust
use oxirs_stream::ReservoirSampler;

let mut sampler = ReservoirSampler::new(1000);
for event in stream {
    sampler.add(event);
}
let sample = sampler.sample(); // Uniform random 1,000 events
```

### Use **HyperLogLog** when you need:
- Distinct count estimation (cardinality)
- Sub-linear space complexity (1-2 KB for billions)
- Mergeable for distributed counting
- ~1-2% accuracy with 14-bit precision
- **Example**: Count distinct RDF subjects in stream

```rust
use oxirs_stream::HyperLogLog;

let mut hll = HyperLogLog::new(14); // 16K registers, ~16KB memory
for triple in triples {
    hll.add(&triple.subject);
}
let distinct_subjects = hll.cardinality(); // ~1.5% error
```

**Memory vs. Precision Trade-off:**
- Precision 10 (1K registers): ~4KB memory, ~2.6% error
- Precision 12 (4K registers): ~4KB memory, ~1.3% error
- Precision 14 (16K registers): ~16KB memory, ~0.65% error
- Precision 16 (64K registers): ~64KB memory, ~0.33% error

### Use **Count-Min Sketch** when you need:
- Approximate frequency counting
- Heavy hitter detection (top-K queries)
- Configurable error bounds
- **Example**: Find most frequent RDF predicates

```rust
use oxirs_stream::CountMinSketch;

let mut cms = CountMinSketch::new(4, 10_000); // 4 hash functions, 10K width
for triple in triples {
    cms.add(&triple.predicate, 1);
}
let freq = cms.estimate(&"http://schema.org/name"); // ~10,234 occurrences
```

**Error Bounds:**
- ε = e / width (relative error)
- δ = 1 / e^depth (failure probability)
- Example: width=10K, depth=4 → ε=0.027%, δ=1.8%

### Use **T-Digest** when you need:
- Approximate percentile calculations
- Accurate extreme percentiles (p0.1, p99.9)
- Mergeable for distributed quantiles
- **Example**: Track query latency percentiles

```rust
use oxirs_stream::TDigest;

let mut digest = TDigest::new(0.01); // 1% compression
for latency in latencies {
    digest.add(latency, 1.0);
}
let p50 = digest.quantile(0.50)?;  // Median latency
let p99 = digest.quantile(0.99)?;  // P99 latency
```

**Compression Parameter:**
- delta=0.1: ~100 centroids, faster, less accurate
- delta=0.01: ~1,000 centroids, slower, more accurate
- delta=0.001: ~10,000 centroids, slowest, most accurate

### Use **Bloom Filter** when you need:
- Fast membership testing
- No false negatives (100% recall)
- Configurable false positive rate
- **Example**: Duplicate event detection

```rust
use oxirs_stream::BloomFilter;

let mut bloom = BloomFilter::optimal(100_000, 0.01); // 100K items, 1% FPR
for event_id in seen_events {
    bloom.add(&event_id);
}
if bloom.contains(&new_event_id) {
    // Likely seen before (99% confidence)
}
```

**Memory vs. False Positive Rate:**
- 1% FPR: ~10 bits per element
- 0.1% FPR: ~15 bits per element
- 0.01% FPR: ~20 bits per element

### Use **Stratified Sampling** when you need:
- Category-aware sampling
- Preserve distribution across event types
- Per-category sample rates
- **Example**: Sample events proportionally by source

```rust
use oxirs_stream::StratifiedSampler;

fn extract_category(event: &StreamEvent) -> Option<String> {
    // Extract category from event metadata
    Some(event.source.clone())
}

let mut sampler = StratifiedSampler::new(1000, extract_category);
sampler.set_category_rate("high-priority".to_string(), 1.0);  // 100%
sampler.set_category_rate("low-priority".to_string(), 0.1);   // 10%

for event in stream {
    sampler.add(event);
}
let samples = sampler.all_samples(); // Category-aware samples
```

## Configuration Examples

### High-Throughput Configuration
```rust
let config = SamplingConfig {
    reservoir_size: 500,           // Smaller sample
    cms_hash_count: 3,             // Fewer hashes
    cms_width: 5_000,              // Smaller width
    hll_precision: 12,             // 4K registers
    tdigest_delta: 0.05,           // More compression
    bloom_filter_bits: 50_000,     // Smaller filter
    bloom_filter_hashes: 5,
    ..Default::default()
};
```

### High-Accuracy Configuration
```rust
let config = SamplingConfig {
    reservoir_size: 10_000,        // Larger sample
    cms_hash_count: 7,             // More hashes
    cms_width: 100_000,            // Larger width
    hll_precision: 16,             // 64K registers
    tdigest_delta: 0.001,          // Less compression
    bloom_filter_bits: 1_000_000,  // Larger filter
    bloom_filter_hashes: 10,
    ..Default::default()
};
```

### Balanced Configuration (Default)
```rust
let config = SamplingConfig {
    reservoir_size: 1_000,
    cms_hash_count: 4,
    cms_width: 10_000,
    hll_precision: 14,             // ~16KB, ~0.65% error
    tdigest_delta: 0.01,
    bloom_filter_bits: 100_000,
    bloom_filter_hashes: 7,
    ..Default::default()
};
```

## Performance Characteristics

| Algorithm | Space Complexity | Time Complexity | Accuracy |
|-----------|-----------------|-----------------|----------|
| Reservoir | O(k) | O(1) insertion | Exact sample |
| HyperLogLog | O(m) where m=2^p | O(1) insertion, O(m) cardinality | ~1.04/√m error |
| Count-Min | O(wd) | O(d) insertion, O(d) query | ε=e/w, δ=1/e^d |
| T-Digest | O(1/δ) | O(log(1/δ)) | Accurate extremes |
| Bloom Filter | O(n·k) bits | O(k) insertion/query | Configurable FPR |

## Memory Usage Examples

### Processing 1 Billion Events

| Use Case | Exact Approach | Sampling Approach | Savings |
|----------|---------------|-------------------|---------|
| Distinct count | 8GB (HashSet) | 16KB (HyperLogLog p14) | 500,000x |
| Top-K frequencies | 800MB (HashMap) | 320KB (CMS 4x10K) | 2,500x |
| P99 latency | 8GB (sorted array) | 100KB (T-Digest) | 80,000x |
| Duplicate check | 8GB (HashSet) | 1.2MB (Bloom 1% FPR) | 6,667x |
| Random sample | 8GB (full data) | 500KB (Reservoir 1K) | 16,000x |

## Use Case Patterns

### Pattern 1: Real-Time Analytics Dashboard
```rust
let mut manager = AdvancedSamplingManager::new(config);

loop {
    let event = stream.next()?;
    manager.process_event(event)?;

    // Update dashboard every second
    if should_update_dashboard() {
        let stats = manager.stats();
        dashboard.update(
            distinct_users: stats.hyperloglog_stats.estimated_cardinality,
            p50_latency: manager.quantile(0.50)?,
            p99_latency: manager.quantile(0.99)?,
            events_per_sec: calculate_rate(stats.event_count),
        );
    }
}
```

### Pattern 2: Heavy Hitter Detection
```rust
let mut cms = CountMinSketch::new(4, 10_000);
let mut top_k = Vec::new();

for event in stream {
    let key = extract_key(&event);
    cms.add(&key, 1);

    let freq = cms.estimate(&key);
    if freq > THRESHOLD {
        top_k.push((key, freq));
    }
}

top_k.sort_by(|a, b| b.1.cmp(&a.1));
let top_10 = &top_k[..10];
```

### Pattern 3: Distributed Cardinality
```rust
// Worker 1
let mut hll1 = HyperLogLog::new(14);
for event in partition1 {
    hll1.add(&event.subject);
}

// Worker 2
let mut hll2 = HyperLogLog::new(14);
for event in partition2 {
    hll2.add(&event.subject);
}

// Coordinator
hll1.merge(&hll2);
let total_distinct = hll1.cardinality();
```

### Pattern 4: Quality Monitoring
```rust
let mut bloom = BloomFilter::optimal(1_000_000, 0.01);
let mut duplicates = 0;

for event in stream {
    if bloom.contains(&event.id) {
        duplicates += 1;
        log::warn!("Duplicate event detected: {}", event.id);
    } else {
        bloom.add(&event.id);
    }
}

let duplicate_rate = duplicates as f64 / total_events as f64;
```

## Theoretical Foundations

### HyperLogLog
- **Paper**: Flajolet et al. (2007) - "HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm"
- **Error Bound**: σ = 1.04/√m where m = 2^precision
- **Key Insight**: Use harmonic mean of register values for bias correction

### Count-Min Sketch
- **Paper**: Cormode & Muthukrishnan (2005) - "An improved data stream summary: the count-min sketch and its applications"
- **Error Bound**: estimate ≤ true_count + ε·N with probability 1-δ
- **Key Insight**: Multiple hash functions provide probabilistic guarantees

### T-Digest
- **Paper**: Dunning (2013) - "Computing Extremely Accurate Quantiles Using t-Digests"
- **Error Bound**: More accurate at extremes (p0, p100), configurable compression
- **Key Insight**: Variable-size clusters preserve precision where needed

### Bloom Filter
- **Paper**: Bloom (1970) - "Space/time trade-offs in hash coding with allowable errors"
- **Error Bound**: FPR ≈ (1 - e^(-k·n/m))^k
- **Key Insight**: No false negatives, configurable false positive rate

### Reservoir Sampling
- **Paper**: Vitter (1985) - "Random sampling with a reservoir"
- **Error Bound**: Exact uniform sample, no approximation
- **Key Insight**: O(1) per-element processing with decreasing acceptance probability

## Best Practices

1. **Choose the Right Algorithm**
   - Cardinality → HyperLogLog
   - Frequency → Count-Min Sketch
   - Percentiles → T-Digest
   - Membership → Bloom Filter
   - Sampling → Reservoir/Stratified

2. **Configure for Your Workload**
   - High throughput → Lower precision, smaller structures
   - High accuracy → Higher precision, larger structures
   - Memory constrained → Adjust precision/width parameters

3. **Monitor and Validate**
   - Track actual vs. estimated values during development
   - Validate error bounds match your requirements
   - Adjust configuration based on observed accuracy

4. **Combine Algorithms**
   - Use `AdvancedSamplingManager` for comprehensive analytics
   - Combine HyperLogLog + Count-Min for distinct + frequency
   - Use Bloom Filter for deduplication before other algorithms

5. **Handle Edge Cases**
   - Empty streams: All algorithms handle gracefully
   - Low cardinality: HyperLogLog less accurate below ~1000 items
   - Skewed distributions: Count-Min benefits from larger width

## Running Examples and Benchmarks

```bash
# Run comprehensive demo
cargo run --example advanced_sampling_demo --all-features

# Run performance benchmarks
cargo bench --bench sampling_benchmarks --all-features

# Run specific benchmark
cargo bench --bench sampling_benchmarks hyperloglog --all-features
```

## Integration with OxiRS Stream

All sampling techniques integrate seamlessly with the oxirs-stream event processing pipeline:

```rust
use oxirs_stream::{StreamConfig, StreamConsumer, AdvancedSamplingManager};

// Create stream consumer
let config = StreamConfig::memory();
let mut consumer = StreamConsumer::new(config).await?;

// Create sampling manager
let sampling_config = SamplingConfig::default();
let mut sampler = AdvancedSamplingManager::new(sampling_config);

// Process stream with sampling
loop {
    if let Some(event) = consumer.consume().await? {
        sampler.process_event(event)?;
    }

    // Periodically report analytics
    if should_report() {
        let stats = sampler.stats();
        report_analytics(stats);
    }
}
```

## Further Reading

- [Probabilistic Data Structures](https://en.wikipedia.org/wiki/Category:Probabilistic_data_structures)
- [Stream Processing Algorithms](https://en.wikipedia.org/wiki/Streaming_algorithm)
- [Approximate Computing](https://en.wikipedia.org/wiki/Approximate_computing)
- [Big Data Analytics](https://en.wikipedia.org/wiki/Big_data)

## Support

For issues, questions, or contributions related to Advanced Sampling:
- GitHub Issues: https://github.com/cool-japan/oxirs/issues
- Documentation: See module-level docs in `src/advanced_sampling.rs`
- Examples: See `examples/advanced_sampling_demo.rs`
- Benchmarks: See `benches/sampling_benchmarks.rs`
