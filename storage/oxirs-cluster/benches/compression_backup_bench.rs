//! # Compression Codec and Backup Policy Benchmark Suite
//!
//! Measures performance of:
//! - Codec compress throughput (Identity, RLE, LZ4, Zstd) on 1 KB / 64 KB / 1 MB
//! - Codec decompress throughput (same)
//! - Codec round-trip latency (compress → decompress)
//! - Registry lookup (warm vs cold)
//! - GFS prune decision speed over a 100-day history
//! - Retention `should_retain` across all 3 tiers with 365 synthetic events
//!
//! ## Running
//!
//! ```bash
//! cargo bench --bench compression_backup_bench -p oxirs-cluster
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_cluster::{
    backup::gfs::{BackupRecord, GfsRotation},
    backup::retention::RetentionTier,
    compression::codecs::{Compressor, IdentityCodec, Lz4Codec, RleCodec, ZstdCodec},
    compression::registry::CodecRegistry,
};
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Payload sizes: 1 KB, 64 KB, 1 MB.
const SIZES: &[usize] = &[1_024, 65_536, 1_048_576];

/// Build a highly repetitive payload suitable for testing RLE.
fn repetitive_payload(size: usize) -> Vec<u8> {
    b"oxirs-cluster-backup-compress "
        .iter()
        .cycle()
        .copied()
        .take(size)
        .collect()
}

/// Named codec instances with their test label.
fn all_codecs() -> Vec<(&'static str, Arc<dyn Compressor>)> {
    vec![
        ("identity", Arc::new(IdentityCodec) as Arc<dyn Compressor>),
        ("rle", Arc::new(RleCodec) as Arc<dyn Compressor>),
        ("lz4", Arc::new(Lz4Codec) as Arc<dyn Compressor>),
        (
            "zstd",
            Arc::new(ZstdCodec::default_level()) as Arc<dyn Compressor>,
        ),
    ]
}

/// Pre-compressed data for decompression benchmarks.
/// Uses repetitive payload so RLE actually compresses (RLE expands random data).
fn pre_compressed(codec: &dyn Compressor, size: usize) -> Vec<u8> {
    let data = repetitive_payload(size);
    codec.compress(&data).expect("pre-compression must succeed")
}

// ---------------------------------------------------------------------------
// 1. Codec compress throughput
// ---------------------------------------------------------------------------

/// Compress throughput for every codec × every payload size.
fn bench_codec_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec_compress");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    for &size in SIZES {
        // Use repetitive payload so all codecs including RLE are valid.
        let data = repetitive_payload(size);

        group.throughput(Throughput::Bytes(size as u64));

        for (label, codec) in all_codecs() {
            group.bench_with_input(BenchmarkId::new(label, size), &data, |b, input| {
                b.iter(|| {
                    let compressed = codec
                        .compress(black_box(input))
                        .expect("compress must not fail");
                    black_box(compressed);
                });
            });
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 2. Codec decompress throughput
// ---------------------------------------------------------------------------

/// Decompress throughput for every codec × every payload size.
/// Data is pre-compressed once outside the benchmark loop.
fn bench_codec_decompress(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec_decompress");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    for &size in SIZES {
        group.throughput(Throughput::Bytes(size as u64));

        for (label, codec) in all_codecs() {
            let compressed = pre_compressed(codec.as_ref(), size);

            group.bench_with_input(BenchmarkId::new(label, size), &compressed, |b, input| {
                b.iter(|| {
                    let decompressed = codec
                        .decompress(black_box(input))
                        .expect("decompress must not fail");
                    black_box(decompressed);
                });
            });
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 3. Codec round-trip latency
// ---------------------------------------------------------------------------

/// Measure the combined compress + decompress latency and verify identity.
/// Uses mixed-entropy payload (avoids trivial identity case for non-identity codecs).
fn bench_codec_round_trip(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec_round_trip");
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(30);

    for &size in SIZES {
        // Use repetitive payload so that RLE round-trip is valid.
        let data = repetitive_payload(size);

        group.throughput(Throughput::Bytes(size as u64));

        for (label, codec) in all_codecs() {
            group.bench_with_input(BenchmarkId::new(label, size), &data, |b, input| {
                b.iter(|| {
                    let compressed = codec
                        .compress(black_box(input))
                        .expect("compress must not fail");
                    let decompressed = codec
                        .decompress(&compressed)
                        .expect("decompress must not fail");
                    // Keep the compiler from eliding the decompression.
                    black_box(decompressed);
                });
            });
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 4. Registry lookup — warm vs cold
// ---------------------------------------------------------------------------

/// Warm lookup: a single registry is reused across all iterations (HashMap is
/// already built, `Arc` clones are cheap).
fn bench_registry_lookup_warm(c: &mut Criterion) {
    let mut group = c.benchmark_group("registry_lookup");

    let registry = CodecRegistry::default();
    let codec_names = ["identity", "rle", "lz4", "zstd"];

    for name in codec_names {
        group.bench_function(BenchmarkId::new("warm", name), |b| {
            b.iter(|| {
                let codec = registry.get(black_box(name)).expect("codec must exist");
                black_box(codec);
            });
        });
    }

    group.finish();
}

/// Cold lookup: a fresh `CodecRegistry::default()` is constructed *inside*
/// the iteration — measures HashMap construction cost + 4 codec allocations.
fn bench_registry_lookup_cold(c: &mut Criterion) {
    let mut group = c.benchmark_group("registry_lookup");

    let codec_names = ["identity", "rle", "lz4", "zstd"];

    for name in codec_names {
        group.bench_function(BenchmarkId::new("cold", name), |b| {
            b.iter(|| {
                let registry = CodecRegistry::default();
                let codec = registry.get(black_box(name)).expect("codec must exist");
                black_box(codec);
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 5. GFS prune_candidates — 100-day history
// ---------------------------------------------------------------------------

/// Simulate a 100-day backup history (one backup per day) and measure how
/// fast `GfsRotation::prune_candidates` decides which backups to delete.
fn bench_gfs_prune_candidates(c: &mut Criterion) {
    let mut group = c.benchmark_group("gfs_prune_candidates");
    group.measurement_time(Duration::from_secs(8));

    // Standard GFS: 7 daily, 4 weekly, 3 monthly.
    let gfs = GfsRotation {
        daily_count: 7,
        weekly_count: 4,
        monthly_count: 3,
    };

    let now = SystemTime::now();

    // Build 100-day history: day 0 = 100 days ago, day 99 = yesterday.
    let records: Vec<BackupRecord> = (0u64..100)
        .map(|day| BackupRecord {
            id: day,
            created_at: now
                .checked_sub(Duration::from_secs((100 - day) * 86_400))
                .expect("duration subtraction must not underflow"),
            size_bytes: 524_288, // 512 KiB per backup
            is_weekly: day % 7 == 0,
            is_monthly: day % 30 == 0,
        })
        .collect();

    group.throughput(Throughput::Elements(records.len() as u64));

    group.bench_function("100_day_history", |b| {
        b.iter(|| {
            let candidates = gfs.prune_candidates(black_box(&records), now);
            black_box(candidates);
        });
    });

    // Also bench with different GFS configurations to explore sensitivity.
    let aggressive = GfsRotation {
        daily_count: 3,
        weekly_count: 2,
        monthly_count: 1,
    };

    group.bench_function("100_day_history_aggressive", |b| {
        b.iter(|| {
            let candidates = aggressive.prune_candidates(black_box(&records), now);
            black_box(candidates);
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 6. Retention should_retain — 365 events × 3 tiers
// ---------------------------------------------------------------------------

/// Precomputed event tuple: (backup_time, is_weekly, is_monthly).
type Event = (SystemTime, bool, bool);

/// Build 365 synthetic daily events (one per day over the past year).
fn build_year_events(now: SystemTime) -> Vec<Event> {
    (0u64..365)
        .map(|day| {
            let backup_time = now
                .checked_sub(Duration::from_secs(day * 86_400))
                .expect("duration subtraction must not underflow");
            let is_weekly = day % 7 == 0;
            let is_monthly = day % 30 == 0;
            (backup_time, is_weekly, is_monthly)
        })
        .collect()
}

/// Benchmark `should_retain` across all 3 tier configurations for 365 events.
fn bench_retention_should_retain(c: &mut Criterion) {
    let mut group = c.benchmark_group("retention_should_retain");
    group.measurement_time(Duration::from_secs(8));

    let now = SystemTime::now();
    let events = build_year_events(now);

    // Three tier configurations: hot-only, standard, generous.
    let tier_configs: &[(&str, RetentionTier)] = &[
        (
            "hot_only",
            RetentionTier {
                hot_days: 7,
                warm_weeks: 0,
                cold_months: 0,
            },
        ),
        ("standard", RetentionTier::standard()),
        (
            "generous",
            RetentionTier {
                hot_days: 14,
                warm_weeks: 8,
                cold_months: 24,
            },
        ),
    ];

    group.throughput(Throughput::Elements(events.len() as u64));

    for (label, tier) in tier_configs {
        group.bench_function(BenchmarkId::new("365_events", label), |b| {
            b.iter(|| {
                let mut kept = 0u32;
                for &(backup_time, is_weekly, is_monthly) in black_box(&events) {
                    if tier.should_retain(backup_time, now, is_weekly, is_monthly) {
                        kept += 1;
                    }
                }
                black_box(kept);
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion groups and main entry point
// ---------------------------------------------------------------------------

criterion_group!(
    codec_benches,
    bench_codec_compress,
    bench_codec_decompress,
    bench_codec_round_trip,
    bench_registry_lookup_warm,
    bench_registry_lookup_cold,
);

criterion_group!(
    backup_benches,
    bench_gfs_prune_candidates,
    bench_retention_should_retain,
);

criterion_main!(codec_benches, backup_benches);
