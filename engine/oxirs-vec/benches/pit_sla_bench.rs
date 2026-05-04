//! Point-in-Time Restore & SLA Admission Control Benchmarks
//!
//! Covers:
//! - `SlaClass` threshold lookup and name parsing (all four tiers)
//! - `AdmissionController::try_admit()` throughput under sustained load
//!   (4 SLA classes × 3 request rates)
//! - `SlaQueryDispatcher` push+pop round-trip with 100 queued queries across
//!   all four priority levels
//! - `PointInTimeRestore::find_base_checkpoint()` for checkpoint counts of
//!   10, 100, and 1 000 (linear scan over pre-populated WAL)
//! - Timestamp resolution: binary-search vs linear-scan approach to locate
//!   the nearest checkpoint before a target timestamp

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_vec::{
    multi_tenancy::{
        admission_controller::AdmissionController, priority_queue::SlaQueryDispatcher,
        sla::SlaClass,
    },
    wal::{WalConfig, WalEntry, WalManager},
    CheckpointRef, PointInTimeRestore,
};
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Duration;
use tempfile::TempDir;

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

const ALL_SLA_CLASSES: [SlaClass; 4] = [
    SlaClass::Bronze,
    SlaClass::Silver,
    SlaClass::Gold,
    SlaClass::Platinum,
];

/// Write `n` checkpoint entries (evenly spaced timestamps 1..=n) to a temp WAL
/// and return the `TempDir` guard (drop → cleanup).
fn write_checkpoint_wal(n: usize) -> TempDir {
    let tmp = TempDir::new().expect("tempdir creation must succeed in bench setup");
    let cfg = WalConfig {
        wal_directory: tmp.path().to_path_buf(),
        checkpoint_interval: u64::MAX,
        sync_on_write: false,
        ..WalConfig::default()
    };
    let mgr = WalManager::new(cfg).expect("WalManager::new must succeed in bench setup");

    for i in 0..n {
        let entry = WalEntry::Checkpoint {
            sequence_number: i as u64,
            timestamp: (i as u64 + 1) * 1_000, // timestamps: 1000, 2000, …
        };
        mgr.append(entry)
            .expect("WAL append must succeed in bench setup");
    }
    mgr.flush().expect("WAL flush must succeed in bench setup");
    tmp
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. SlaClass threshold lookup
// ─────────────────────────────────────────────────────────────────────────────

fn bench_sla_threshold_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("pit_sla/sla_threshold_lookup");
    group.measurement_time(Duration::from_secs(10));

    for &sla in &ALL_SLA_CLASSES {
        group.bench_with_input(
            BenchmarkId::new("thresholds", sla.name()),
            &sla,
            |b, &cls| {
                b.iter(|| {
                    let t = black_box(cls).thresholds();
                    // Touch all fields to prevent the compiler from eliding the call
                    black_box((
                        t.max_latency_p99_ms,
                        t.max_concurrent_queries,
                        t.bandwidth_mb_per_sec,
                        t.token_refill_rate,
                        t.token_bucket_capacity,
                    ))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("dispatch_priority", sla.name()),
            &sla,
            |b, &cls| {
                b.iter(|| black_box(black_box(cls).dispatch_priority()));
            },
        );

        group.bench_with_input(BenchmarkId::new("name", sla.name()), &sla, |b, &cls| {
            b.iter(|| black_box(black_box(cls).name()));
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. AdmissionController throughput
// ─────────────────────────────────────────────────────────────────────────────

/// Request rates as "requests per iteration batch" for the bench inner loop.
/// We use a *batch* approach so Criterion can measure many admissions at once:
/// each batch exercises `try_admit` BATCH_SIZE times, letting us see relative
/// throughput across SLA classes and refill-rate differences.
const BATCH_SIZES: [u64; 3] = [10, 50, 200];

fn bench_admission_controller_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("pit_sla/admission_throughput");
    group.measurement_time(Duration::from_secs(12));

    for &sla in &ALL_SLA_CLASSES {
        // Pre-register a single tenant per SLA class
        let ctrl = AdmissionController::new();
        let tenant_id = format!("tenant_{}", sla.name());
        ctrl.register_tenant(&tenant_id, sla);

        for &batch in &BATCH_SIZES {
            group.throughput(Throughput::Elements(batch));
            group.bench_with_input(BenchmarkId::new(sla.name(), batch), &batch, |b, &n| {
                b.iter(|| {
                    // Re-register before each *iteration* batch to refill tokens
                    // to full capacity; this lets us measure the hot-path
                    // `try_admit` without hitting rate-limit rejections on
                    // low-capacity tiers.
                    ctrl.register_tenant(&tenant_id, sla);

                    let mut admitted = 0u64;
                    let mut rejected = 0u64;
                    for _ in 0..n {
                        match ctrl.try_admit(black_box(&tenant_id)) {
                            Ok(_) => admitted += 1,
                            Err(_) => rejected += 1,
                        }
                    }
                    black_box((admitted, rejected))
                });
            });
        }
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. SlaQueryDispatcher push + pop throughput
// ─────────────────────────────────────────────────────────────────────────────

const DISPATCHER_QUEUE_SIZE: usize = 100;

fn bench_dispatcher_push_pop(c: &mut Criterion) {
    let mut group = c.benchmark_group("pit_sla/dispatcher_push_pop");
    group.measurement_time(Duration::from_secs(12));
    group.throughput(Throughput::Elements(DISPATCHER_QUEUE_SIZE as u64));

    // Pre-build the (tenant_id, SlaClass, payload) tuples for the queue
    let payloads: Vec<(String, SlaClass, u32)> = (0..DISPATCHER_QUEUE_SIZE)
        .map(|i| {
            let sla = ALL_SLA_CLASSES[i % 4];
            let tenant = format!("t_{}", sla.name());
            (tenant, sla, i as u32)
        })
        .collect();

    // Full round-trip: push 100 items then drain all 100
    group.bench_function("100_items_4_priorities", |b| {
        b.iter(|| {
            let mut dispatcher: SlaQueryDispatcher<u32> = SlaQueryDispatcher::new();
            for (tenant, sla, payload) in &payloads {
                dispatcher.enqueue(tenant.clone(), *sla, *payload);
            }
            let mut count = 0usize;
            while let Some(item) = dispatcher.dequeue() {
                count += black_box(item.priority) as usize;
            }
            black_box(count)
        });
    });

    // Isolated push benchmark
    group.bench_function("push_only_100", |b| {
        b.iter(|| {
            let mut dispatcher: SlaQueryDispatcher<u32> = SlaQueryDispatcher::new();
            for (tenant, sla, payload) in &payloads {
                dispatcher.enqueue(
                    black_box(tenant.clone()),
                    black_box(*sla),
                    black_box(*payload),
                );
            }
            black_box(dispatcher.len())
        });
    });

    // drain_ordered benchmark (sorted drain vs individual dequeue)
    group.bench_function("drain_ordered_100", |b| {
        b.iter(|| {
            let mut dispatcher: SlaQueryDispatcher<u32> = SlaQueryDispatcher::new();
            for (tenant, sla, payload) in &payloads {
                dispatcher.enqueue(tenant.clone(), *sla, *payload);
            }
            let drained = dispatcher.drain_ordered();
            black_box(drained.len())
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. PointInTimeRestore checkpoint discovery
// ─────────────────────────────────────────────────────────────────────────────

fn bench_pit_checkpoint_discovery(c: &mut Criterion) {
    let mut group = c.benchmark_group("pit_sla/checkpoint_discovery");
    // I/O-bound: give more time and fewer samples
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(20);

    for &checkpoint_count in &[10usize, 100, 1_000] {
        // Create the WAL once and keep `_tmp` alive for the full group lifetime
        let tmp = write_checkpoint_wal(checkpoint_count);
        let wal_dir: PathBuf = tmp.path().to_path_buf();

        // Target timestamp sits in the middle of the populated range
        let target_ts: u64 = (checkpoint_count as u64 / 2) * 1_000;

        group.bench_with_input(
            BenchmarkId::new("checkpoints", checkpoint_count),
            &checkpoint_count,
            |b, _| {
                b.iter(|| {
                    let pit =
                        PointInTimeRestore::new(black_box(target_ts), black_box(wal_dir.clone()));
                    pit.find_base_checkpoint()
                        .expect("find_base_checkpoint must not error in bench")
                });
            },
        );

        drop(tmp);
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Timestamp resolution: binary search vs linear scan over CheckpointRefs
// ─────────────────────────────────────────────────────────────────────────────

/// Build a sorted `Vec<CheckpointRef>` with `n` checkpoints at evenly spaced
/// timestamps 1 000, 2 000, …, n × 1 000.
fn build_checkpoint_refs(n: usize) -> Vec<CheckpointRef> {
    (0..n)
        .map(|i| CheckpointRef {
            sequence_number: i as u64,
            timestamp: (i as u64 + 1) * 1_000,
        })
        .collect()
}

/// Binary search: find the latest checkpoint whose timestamp ≤ target.
#[inline]
fn binary_search_checkpoint(refs: &[CheckpointRef], target: u64) -> Option<&CheckpointRef> {
    // `partition_point` returns the first index where timestamp > target
    let pos = refs.partition_point(|c| c.timestamp <= target);
    if pos == 0 {
        None
    } else {
        Some(&refs[pos - 1])
    }
}

/// Linear scan: find the latest checkpoint whose timestamp ≤ target (reference
/// implementation matching `PointInTimeRestore::find_base_checkpoint` logic).
#[inline]
fn linear_scan_checkpoint(refs: &[CheckpointRef], target: u64) -> Option<&CheckpointRef> {
    let mut best: Option<&CheckpointRef> = None;
    for cr in refs {
        if cr.timestamp <= target {
            match best {
                None => best = Some(cr),
                Some(prev) if cr.timestamp > prev.timestamp => best = Some(cr),
                _ => {}
            }
        }
    }
    best
}

fn bench_timestamp_resolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("pit_sla/timestamp_resolution");
    group.measurement_time(Duration::from_secs(12));

    for &n in &[10usize, 100, 1_000] {
        let refs = build_checkpoint_refs(n);
        // Target is the midpoint — forces traversal of roughly half the list
        let target: u64 = (n as u64 / 2) * 1_000;

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("binary_search", n), &n, |b, _| {
            b.iter(|| {
                black_box(binary_search_checkpoint(
                    black_box(&refs),
                    black_box(target),
                ))
            });
        });

        group.bench_with_input(BenchmarkId::new("linear_scan", n), &n, |b, _| {
            b.iter(|| black_box(linear_scan_checkpoint(black_box(&refs), black_box(target))));
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Criterion entry points
// ─────────────────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_sla_threshold_lookup,
    bench_admission_controller_throughput,
    bench_dispatcher_push_pop,
    bench_pit_checkpoint_discovery,
    bench_timestamp_resolution,
);
criterion_main!(benches);
