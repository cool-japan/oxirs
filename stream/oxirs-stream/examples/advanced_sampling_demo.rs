//! # Advanced Sampling Techniques Demo
//!
//! Demonstrates the use of probabilistic data structures for
//! memory-efficient analytics on high-volume RDF streams.
//!
//! Run with: cargo run --example advanced_sampling_demo --all-features

use anyhow::Result;
use oxirs_stream::{
    AdvancedSamplingManager, BloomFilter, CountMinSketch, HyperLogLog, ReservoirSampler,
    SamplingConfig, StreamEvent, TDigest,
};
use std::collections::HashMap;

fn create_sample_event(id: usize, subject_prefix: &str) -> StreamEvent {
    StreamEvent::TripleAdded {
        subject: format!("http://example.org/{}/entity-{}", subject_prefix, id),
        predicate: "http://example.org/prop".to_string(),
        object: format!("value-{}", id),
        graph: None,
        metadata: oxirs_stream::EventMetadata {
            event_id: format!("event-{}", id),
            timestamp: chrono::Utc::now(),
            source: if id % 3 == 0 {
                "source-A".to_string()
            } else if id % 3 == 1 {
                "source-B".to_string()
            } else {
                "source-C".to_string()
            },
            user: Some(format!("user-{}", id % 100)),
            context: None,
            caused_by: None,
            version: "1.0".to_string(),
            properties: HashMap::new(),
            checksum: None,
        },
    }
}

fn demo_reservoir_sampling() -> Result<()> {
    println!("\n=== Reservoir Sampling Demo ===");
    println!("Maintaining uniform random sample from unbounded stream");

    let mut sampler = ReservoirSampler::new(100);

    // Simulate processing 10,000 events
    for i in 0..10_000 {
        sampler.add(create_sample_event(i, "stream"));
    }

    let stats = sampler.stats();
    println!("  Processed: {} events", stats.total_events);
    println!("  Sample size: {} events", stats.current_size);
    println!("  Sampling rate: {:.2}%", stats.sampling_rate * 100.0);
    println!("  Memory usage: ~{} KB", stats.current_size * 500 / 1024);

    Ok(())
}

fn demo_hyperloglog() -> Result<()> {
    println!("\n=== HyperLogLog Demo ===");
    println!("Approximate distinct counting with minimal memory");

    let mut hll = HyperLogLog::new(14); // 16K registers

    // Add 100,000 events with ~50,000 distinct subjects
    for i in 0..100_000 {
        let event = create_sample_event(i % 50_000, "distinct");
        let subject = match &event {
            StreamEvent::TripleAdded { subject, .. } => subject.clone(),
            _ => String::new(),
        };
        hll.add(&subject);
    }

    let stats = hll.stats();
    let actual_distinct = 50_000;
    let estimated = stats.estimated_cardinality;
    let error = ((estimated as i64 - actual_distinct).abs() as f64) / actual_distinct as f64;

    println!("  Actual distinct: {}", actual_distinct);
    println!("  Estimated distinct: {}", estimated);
    println!("  Error: {:.2}%", error * 100.0);
    println!("  Memory usage: {} bytes", stats.memory_bytes);
    println!(
        "  Compression ratio: {:.0}x",
        actual_distinct as f64 / stats.memory_bytes as f64
    );

    Ok(())
}

fn demo_count_min_sketch() -> Result<()> {
    println!("\n=== Count-Min Sketch Demo ===");
    println!("Approximate frequency counting for heavy hitters");

    let mut cms = CountMinSketch::new(4, 10_000);

    // Simulate Zipf distribution - some subjects are much more frequent
    for i in 0..100_000 {
        let subject_id = if i < 10_000 {
            i % 10 // Top 10 subjects appear 1000 times each
        } else if i < 50_000 {
            10 + (i % 100) // Next 100 subjects appear 400 times each
        } else {
            110 + (i % 10_000) // Long tail
        };
        let subject = format!("subject-{}", subject_id);
        cms.add(&subject, 1);
    }

    let stats = cms.stats();
    println!("  Total events: {}", stats.total_count);
    println!("  Memory usage: {} KB", stats.memory_bytes / 1024);

    // Query heavy hitters
    println!("\n  Heavy hitters:");
    for i in 0..5 {
        let subject = format!("subject-{}", i);
        let freq = cms.estimate(&subject);
        println!("    {} → ~{} occurrences", subject, freq);
    }

    Ok(())
}

fn demo_tdigest() -> Result<()> {
    println!("\n=== T-Digest Demo ===");
    println!("Approximate percentile calculation for streaming data");

    let mut digest = TDigest::new(0.01);

    // Add event processing latencies (simulated)
    for i in 0..10_000 {
        // Simulate latency distribution (mostly fast, some slow)
        let latency = if i % 100 == 0 {
            fastrand::f64() * 1000.0 + 100.0 // 1% slow requests
        } else {
            fastrand::f64() * 50.0 + 1.0 // 99% fast requests
        };
        digest.add(latency, 1.0);
    }

    let stats = digest.stats();
    println!("  Total samples: {}", stats.total_weight as u64);
    println!("  Centroids: {}", stats.centroid_count);
    println!(
        "  Compression: {:.1}x",
        stats.total_weight / stats.centroid_count as f64
    );

    println!("\n  Latency percentiles:");
    if let Some(p50) = digest.clone().quantile(0.50) {
        println!("    P50: {:.2}ms", p50);
    }
    if let Some(p90) = digest.clone().quantile(0.90) {
        println!("    P90: {:.2}ms", p90);
    }
    if let Some(p95) = digest.clone().quantile(0.95) {
        println!("    P95: {:.2}ms", p95);
    }
    if let Some(p99) = digest.clone().quantile(0.99) {
        println!("    P99: {:.2}ms", p99);
    }
    if let Some(p999) = digest.clone().quantile(0.999) {
        println!("    P99.9: {:.2}ms", p999);
    }

    Ok(())
}

fn demo_bloom_filter() -> Result<()> {
    println!("\n=== Bloom Filter Demo ===");
    println!("Space-efficient duplicate detection");

    // Create optimal filter for 100K items with 1% FPR
    let mut bloom = BloomFilter::optimal(100_000, 0.01);

    // Add 50K unique subjects
    for i in 0..50_000 {
        let subject = format!("subject-{}", i);
        bloom.add(&subject);
    }

    let stats = bloom.stats();
    println!("  Inserted: {} items", stats.insert_count);
    println!("  Memory usage: {} KB", stats.memory_bytes / 1024);
    println!(
        "  Bits per item: {:.1}",
        stats.size_bits as f64 / stats.insert_count as f64
    );
    println!("  Estimated FPR: {:.2}%", stats.estimated_fpr * 100.0);

    // Test membership
    let mut true_positives = 0;
    let mut false_positives = 0;

    for i in 0..100_000 {
        let subject = format!("subject-{}", i);
        let exists = bloom.contains(&subject);

        if i < 50_000 {
            // Should be in the set
            if exists {
                true_positives += 1;
            }
        } else {
            // Should not be in the set
            if exists {
                false_positives += 1;
            }
        }
    }

    println!("\n  Test results:");
    println!(
        "    True positives: {}/50000 ({:.1}%)",
        true_positives,
        true_positives as f64 / 500.0
    );
    println!(
        "    False positives: {}/50000 ({:.1}%)",
        false_positives,
        false_positives as f64 / 500.0
    );

    Ok(())
}

fn demo_unified_manager() -> Result<()> {
    println!("\n=== Unified Sampling Manager Demo ===");
    println!("Using all sampling techniques together");

    let config = SamplingConfig {
        reservoir_size: 500,
        cms_hash_count: 4,
        cms_width: 5_000,
        hll_precision: 12,
        tdigest_delta: 0.01,
        bloom_filter_bits: 50_000,
        bloom_filter_hashes: 7,
        ..Default::default()
    };

    let mut manager = AdvancedSamplingManager::new(config);

    // Process 50,000 events
    println!("  Processing 50,000 events...");
    for i in 0..50_000 {
        let event = create_sample_event(i, "unified");
        manager.process_event(event)?;
    }

    let stats = manager.stats();
    println!("\n  Results:");
    println!("    Events processed: {}", stats.event_count);
    println!(
        "    Distinct count (HyperLogLog): {}",
        stats.hyperloglog_stats.estimated_cardinality
    );
    println!(
        "    Reservoir sample size: {}",
        stats.reservoir_stats.current_size
    );
    println!(
        "    Total memory usage: ~{} KB",
        (stats.reservoir_stats.current_size * 500
            + stats.hyperloglog_stats.memory_bytes
            + stats.count_min_stats.memory_bytes
            + stats.bloom_stats.memory_bytes)
            / 1024
    );

    // Get a sample event and query its frequency
    if let Some(sample_event) = manager.reservoir_sample().first() {
        let freq = manager.event_frequency(sample_event);
        println!("\n  Sample event frequency: ~{}", freq);
        println!(
            "  Sample event seen before: {}",
            manager.likely_seen(sample_event)
        );
    }

    Ok(())
}

fn main() -> Result<()> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Advanced Sampling Techniques for Stream Processing        ║");
    println!("║  Demonstrating memory-efficient probabilistic algorithms   ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    demo_reservoir_sampling()?;
    demo_hyperloglog()?;
    demo_count_min_sketch()?;
    demo_tdigest()?;
    demo_bloom_filter()?;
    demo_unified_manager()?;

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║  Summary: All algorithms provide sub-linear space          ║");
    println!("║  complexity with provable error bounds, enabling           ║");
    println!("║  real-time analytics on billion-event RDF streams.         ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
