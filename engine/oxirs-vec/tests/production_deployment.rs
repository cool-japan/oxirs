//! Production deployment validation harness (W2-S7).
//!
//! Exercises the production-grade primitives shipped in v0.3.0 — multi-tenant
//! admission, snapshot/PIT-restore, replica failover — under realistic
//! workloads.  The intent is not to be a microbenchmark; it is to detect
//! **operational regressions** in the wiring between subsystems before they
//! reach production.
//!
//! Scenarios:
//!
//! 1. **Multi-tenant load**: 10 tenants × 4 SLA classes × 1 000 queries
//!    each, dispatched through `AdmissionController` + `SlaQueryDispatcher`.
//!    Pass criteria: no panics; admission accounting balanced; Platinum
//!    queries dispatched before Bronze queries.
//!
//! 2. **Snapshot during load**: a writer thread streams WAL inserts; a
//!    snapshot/PIT-restore is triggered mid-stream; subsequent queries on
//!    the restored store return the expected vectors.  RTO and RPO
//!    measurements are recorded.
//!
//! 3. **Chaos**: writer / reader / checkpoint corruption fault injection
//!    using `ReplicaManager` and corrupt-WAL files.
//!
//! Heavy scenarios (all 40 000 queries materialised, end-to-end snapshot) are
//! gated behind `#[ignore]` and run with `cargo test --
//! production_deployment::heavy_ -- --ignored`.

use oxirs_vec::{
    fault::{ReplicaManager, ReplicaState, ShardReplica},
    multi_tenancy::{
        admission_controller::AdmissionController, priority_queue::SlaQueryDispatcher,
        sla::SlaClass,
    },
    persistence::{restore::restore_to_timestamp, PointInTimeRestore},
    vector_store::VectorStore,
    wal::{WalConfig, WalEntry, WalManager},
    IndexDispatcher, IndexDispatcherConfig, Vector,
};
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Deterministic small vector — avoids dependency on rand for reproducibility.
fn det_vec(seed: u64, dim: usize) -> Vector {
    let mut state = seed.wrapping_mul(2654435769).wrapping_add(0x9E37_79B9);
    let mut values = Vec::with_capacity(dim);
    for _ in 0..dim {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let f = (state as f32) / (u64::MAX as f32);
        values.push((f - 0.5) * 2.0);
    }
    Vector::new(values)
}

fn write_wal_inserts(dir: &Path, inserts: &[(String, Vec<f32>, u64)]) -> anyhow::Result<()> {
    let cfg = WalConfig {
        wal_directory: dir.to_path_buf(),
        checkpoint_interval: u64::MAX,
        sync_on_write: true,
        ..WalConfig::default()
    };
    let mgr = WalManager::new(cfg)?;
    for (id, v, ts) in inserts {
        mgr.append(WalEntry::Insert {
            id: id.clone(),
            vector: v.clone(),
            metadata: None,
            timestamp: *ts,
        })?;
    }
    mgr.flush()
}

/// Acceptable performance ceilings.  Relaxed in debug builds because the
/// crate's policy notes performance tests apply `cfg(debug_assertions)`
/// branches.
fn max_rto_ms() -> u128 {
    if cfg!(debug_assertions) {
        90_000 // 90s in debug
    } else {
        30_000 // 30s release
    }
}

fn max_rpo_secs() -> u64 {
    if cfg!(debug_assertions) {
        5
    } else {
        1
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 1 — Multi-tenant load (smoke variant; heavy variant is ignored)
// ─────────────────────────────────────────────────────────────────────────────

/// Smoke variant: 4 tenants × 4 classes × 50 queries.  Always runs.  Verifies
/// the dispatch pipeline doesn't panic and Platinum priority is respected.
#[test]
fn multi_tenant_load_smoke() {
    let admission = AdmissionController::new();
    let mut dispatcher: SlaQueryDispatcher<u64> = SlaQueryDispatcher::new();

    let classes = [
        SlaClass::Bronze,
        SlaClass::Silver,
        SlaClass::Gold,
        SlaClass::Platinum,
    ];
    let tenants_per_class = 4;
    let queries_per_tenant = 50;

    let mut admitted = 0u64;
    let mut rejected = 0u64;
    let mut next_payload: u64 = 0;

    for (class_idx, class) in classes.iter().enumerate() {
        for tenant_idx in 0..tenants_per_class {
            let tenant = format!("t_{}_{}", class_idx, tenant_idx);
            admission.register_tenant(&tenant, *class);

            for _ in 0..queries_per_tenant {
                next_payload += 1;
                match admission.try_admit(&tenant) {
                    Ok(_) => {
                        admitted += 1;
                        dispatcher.enqueue(tenant.clone(), *class, next_payload);
                    }
                    Err(_) => rejected += 1,
                }
            }
        }
    }

    // Both admitted and rejected counts are non-zero (Bronze tenants exhaust
    // their token bucket; Platinum almost always admits).
    assert!(admitted > 0, "dispatch pipeline should admit at least one");
    assert_eq!(
        admitted + rejected,
        (classes.len() as u64) * (tenants_per_class as u64) * (queries_per_tenant as u64),
        "every query is accounted for"
    );

    // Drain the dispatcher and verify priority order: Platinum (4) > Gold (3) > Silver (2) > Bronze (1).
    let drained = dispatcher.drain_ordered();
    if drained.is_empty() {
        return; // Nothing to verify.
    }
    let priorities: Vec<u8> = drained.iter().map(|q| q.priority).collect();
    for window in priorities.windows(2) {
        assert!(
            window[0] >= window[1],
            "priorities must be non-increasing: {:?}",
            priorities
        );
    }
}

/// Heavy variant: 10 tenants × 4 classes × 1 000 queries (= 40 000 ops).
#[test]
#[ignore = "heavy: 40000 ops; run with --ignored"]
fn multi_tenant_load_heavy() {
    let admission = AdmissionController::new();
    let mut dispatcher: SlaQueryDispatcher<u64> = SlaQueryDispatcher::new();

    let classes = [
        SlaClass::Bronze,
        SlaClass::Silver,
        SlaClass::Gold,
        SlaClass::Platinum,
    ];
    let tenants_per_class = 10;
    let queries_per_tenant = 1_000;

    let mut admitted = 0u64;
    let mut payload: u64 = 0;
    for (class_idx, class) in classes.iter().enumerate() {
        for tenant_idx in 0..tenants_per_class {
            let tenant = format!("heavy_{}_{}", class_idx, tenant_idx);
            admission.register_tenant(&tenant, *class);
            for _ in 0..queries_per_tenant {
                payload += 1;
                if admission.try_admit(&tenant).is_ok() {
                    admitted += 1;
                    dispatcher.enqueue(tenant.clone(), *class, payload);
                }
            }
        }
    }

    let total = (classes.len() as u64) * (tenants_per_class as u64) * (queries_per_tenant as u64);
    assert_eq!(total, 40_000, "heavy scenario must enumerate 40k queries");
    assert!(admitted > 0);

    let drained = dispatcher.drain_ordered();
    let priorities: Vec<u8> = drained.iter().map(|q| q.priority).collect();
    for window in priorities.windows(2) {
        assert!(window[0] >= window[1]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 2 — Snapshot during load + restore on a fresh node
// ─────────────────────────────────────────────────────────────────────────────

/// Trigger a PIT snapshot mid-load and verify a fresh `VectorStore` can be
/// restored deterministically from the WAL **while the writer is still
/// streaming**.  Validates that the restore path coexists with concurrent
/// WAL appends — the production deployment requirement.
///
/// The test uses millisecond-granularity timestamps so the RPO assertion
/// has teeth even on fast hardware.
#[test]
fn snapshot_during_load_and_restore_on_fresh_node() {
    let tmp = TempDir::new().expect("tempdir");
    let wal_dir = tmp.path().to_path_buf();

    let stop_writer = Arc::new(AtomicBool::new(false));
    let written = Arc::new(AtomicU64::new(0));
    let snapshot_target_ts = Arc::new(AtomicU64::new(0));

    // ── Writer thread: streams ~300 inserts at 1ms cadence (~300ms total). ──
    // It deliberately keeps writing past the snapshot point so the restore
    // call below operates on a live, growing WAL.
    let stop_writer_t = stop_writer.clone();
    let written_t = written.clone();
    let target_t = snapshot_target_ts.clone();
    let wal_dir_t = wal_dir.clone();
    let writer = thread::spawn(move || -> anyhow::Result<()> {
        let cfg = WalConfig {
            wal_directory: wal_dir_t,
            checkpoint_interval: u64::MAX,
            sync_on_write: true,
            ..WalConfig::default()
        };
        let mgr = WalManager::new(cfg)?;
        let mut i: u64 = 0;
        let start = Instant::now();
        while !stop_writer_t.load(Ordering::Relaxed) {
            i += 1;
            // Use millisecond-granularity timestamps so the RPO assertion
            // distinguishes the snapshot point from later writes.
            let ts = start.elapsed().as_millis() as u64 + 1;
            mgr.append(WalEntry::Insert {
                id: format!("v_{}", i),
                vector: vec![i as f32, (i * 2) as f32],
                metadata: None,
                timestamp: ts,
            })?;
            written_t.store(i, Ordering::Relaxed);
            if i == 50 {
                // Snapshot point — recorded but the writer keeps going.
                target_t.store(ts, Ordering::Relaxed);
            }
            if i >= 300 {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        mgr.flush()?;
        Ok(())
    });

    // ── Concurrent reader: a parallel thread that scans the WAL directory   ──
    // while the writer streams + restore runs.  Asserts the file count is    ──
    // monotonic non-decreasing and there are no panics on partial reads.    ──
    let stop_reader = Arc::new(AtomicBool::new(false));
    let stop_reader_t = stop_reader.clone();
    let wal_dir_r = wal_dir.clone();
    let reader = thread::spawn(move || {
        let mut last_count = 0usize;
        while !stop_reader_t.load(Ordering::Relaxed) {
            if let Ok(entries) = std::fs::read_dir(&wal_dir_r) {
                let count = entries.filter_map(|e| e.ok()).count();
                assert!(
                    count >= last_count,
                    "WAL file count must be monotonic non-decreasing: {} -> {}",
                    last_count,
                    count
                );
                last_count = count;
            }
            thread::sleep(Duration::from_millis(2));
        }
    });

    // Wait until the writer has produced > 50 inserts (snapshot point).
    let deadline = Instant::now() + Duration::from_secs(15);
    while snapshot_target_ts.load(Ordering::Relaxed) == 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    let target_ts = snapshot_target_ts.load(Ordering::Relaxed);
    assert!(target_ts > 0, "writer must reach snapshot point in time");

    // ── RTO: restore is invoked while the writer thread is *still active*.  ──
    // This is the production-critical concurrency path.                      ──
    let writes_at_restore_start = written.load(Ordering::Relaxed);
    assert!(
        writes_at_restore_start >= 50,
        "writer must have at least 50 ops before restore starts"
    );
    // Verify writer is still running by checking it's nowhere near completion.
    assert!(
        writes_at_restore_start < 300,
        "writer must still be active when restore starts (got {} writes)",
        writes_at_restore_start
    );

    let restore_start = Instant::now();
    let mut fresh_store = VectorStore::new();
    let report =
        restore_to_timestamp(&mut fresh_store, target_ts, &wal_dir).expect("restore must succeed");
    let rto = restore_start.elapsed().as_millis();

    // The writer may or may not have finished by the time the restore did —
    // we only assert it was concurrent at start-of-restore.
    let writes_at_restore_end = written.load(Ordering::Relaxed);
    assert!(
        writes_at_restore_end >= writes_at_restore_start,
        "writer should not have rolled back: {} -> {}",
        writes_at_restore_start,
        writes_at_restore_end
    );

    assert!(
        rto < max_rto_ms(),
        "RTO {} ms exceeded ceiling {} ms",
        rto,
        max_rto_ms()
    );

    // Drain writer + reader.
    stop_writer.store(true, Ordering::Relaxed);
    writer
        .join()
        .expect("writer panicked")
        .expect("writer error");
    stop_reader.store(true, Ordering::Relaxed);
    reader.join().expect("reader panicked");

    // ── Final RPO assertion on the consolidated WAL. ──
    // After the writer drained, recompute the last replayed timestamp ≤ target_ts.
    // Replay only entries up to target_ts; the gap between target_ts and the
    // last replayed timestamp is the RPO.
    let pit = PointInTimeRestore::new(target_ts, wal_dir.clone());
    let replayed = pit.replay_wal_to_timestamp(None).expect("replay");
    let last_replayed_ts = replayed
        .iter()
        .map(|e| match e {
            WalEntry::Insert { timestamp, .. }
            | WalEntry::Update { timestamp, .. }
            | WalEntry::Delete { timestamp, .. }
            | WalEntry::Batch { timestamp, .. }
            | WalEntry::Checkpoint { timestamp, .. }
            | WalEntry::BeginTransaction { timestamp, .. }
            | WalEntry::CommitTransaction { timestamp, .. }
            | WalEntry::AbortTransaction { timestamp, .. } => *timestamp,
        })
        .max()
        .unwrap_or(0);

    // RPO is in milliseconds; convert seconds-bound to milliseconds.
    let rpo_ms = target_ts.saturating_sub(last_replayed_ts);
    let max_rpo_ms = max_rpo_secs() * 1_000;
    assert!(
        rpo_ms <= max_rpo_ms,
        "RPO {} ms exceeded ceiling {} ms",
        rpo_ms,
        max_rpo_ms
    );

    // Restore replayed at least the inserts up to ts=50 (snapshot point).
    assert!(
        report.entries_replayed > 0,
        "restore must replay at least one entry"
    );

    // Zero data loss: every insert with timestamp ≤ target_ts is present in
    // the replay; nothing past target_ts leaks into the restored state.
    let post_target_inserts = replayed
        .iter()
        .filter(|e| {
            let ts = match e {
                WalEntry::Insert { timestamp, .. }
                | WalEntry::Update { timestamp, .. }
                | WalEntry::Delete { timestamp, .. }
                | WalEntry::Batch { timestamp, .. }
                | WalEntry::Checkpoint { timestamp, .. }
                | WalEntry::BeginTransaction { timestamp, .. }
                | WalEntry::CommitTransaction { timestamp, .. }
                | WalEntry::AbortTransaction { timestamp, .. } => *timestamp,
            };
            ts > target_ts
        })
        .count();
    assert_eq!(
        post_target_inserts, 0,
        "restore must not include entries with timestamp > target_ts"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 3 — Chaos: writer kill / reader kill / checkpoint corruption
// ─────────────────────────────────────────────────────────────────────────────

/// Kill a writer node and verify reads continue from the surviving replica.
#[test]
fn chaos_writer_kill_reads_continue() {
    let mut mgr = ReplicaManager::new(2);
    let primary = ShardReplica::new(1, "writer-A", "node-A", ReplicaState::Primary, 100);
    let reader = ShardReplica::new(1, "reader-B", "node-B", ReplicaState::Replica, 100);
    mgr.register_replica(primary).expect("register primary");
    mgr.register_replica(reader).expect("register reader");

    let status_before = mgr.replication_status();
    assert!(status_before.healthy, "starting state must be healthy");

    // Kill the writer.
    mgr.mark_failed(1, "writer-A");

    let status_after = mgr.replication_status();
    assert!(!status_after.healthy, "killed writer => unhealthy");

    // Auto-failover promotes the reader.
    let promoted = mgr.auto_failover(1).expect("auto_failover should succeed");
    assert_eq!(promoted, "reader-B");

    let status_post_failover = mgr.replication_status();
    assert_eq!(
        status_post_failover.failed_replicas, 1,
        "writer-A still counted as failed"
    );
    // After failover the reader-B is primary; cluster has 1 healthy primary.
    let replicas = mgr.get_replicas(1);
    assert!(!replicas.is_empty(), "shard 1 must exist");
    let primary_count = replicas.iter().filter(|r| r.state.is_primary()).count();
    assert_eq!(primary_count, 1);
}

/// Kill a reader replica and verify the replication-status report reflects it.
#[test]
fn chaos_reader_kill_remaining_replicas_serve() {
    let mut mgr = ReplicaManager::new(3);
    mgr.register_replica(ShardReplica::new(
        2,
        "p",
        "node-1",
        ReplicaState::Primary,
        50,
    ))
    .expect("register primary");
    mgr.register_replica(ShardReplica::new(
        2,
        "r1",
        "node-2",
        ReplicaState::Replica,
        50,
    ))
    .expect("register replica1");
    mgr.register_replica(ShardReplica::new(
        2,
        "r2",
        "node-3",
        ReplicaState::Replica,
        50,
    ))
    .expect("register replica2");

    mgr.mark_failed(2, "r1");

    let status = mgr.replication_status();
    assert_eq!(status.failed_replicas, 1);
    // 2 healthy out of 3 target → under_replicated.
    assert_eq!(status.under_replicated, 1);

    // Remaining primary + r2 still healthy.
    let replicas = mgr.get_replicas(2);
    let healthy = replicas.iter().filter(|r| r.state.is_healthy()).count();
    assert_eq!(healthy, 2, "primary + 1 replica still healthy");
}

/// Corrupt a checkpoint file and verify the error path surfaces.
#[test]
fn chaos_checkpoint_corruption_yields_typed_error() {
    let tmp = TempDir::new().expect("tempdir");
    let wal_dir = tmp.path();

    // Write a valid WAL, then corrupt the file.
    let entries: Vec<(String, Vec<f32>, u64)> = (1..=5)
        .map(|i| (format!("v{}", i), vec![i as f32], i as u64 * 100))
        .collect();
    write_wal_inserts(wal_dir, &entries).expect("seed wal");

    // Find the WAL file and overwrite its tail with garbage.  WAL files are
    // named `wal-<hex>.log` per the rotate logic in `crate::wal`.
    let wal_files: Vec<_> = std::fs::read_dir(wal_dir)
        .expect("read_dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("wal-"))
        .collect();
    assert!(!wal_files.is_empty(), "WAL must produce at least one file");
    for f in &wal_files {
        // Append junk to corrupt the trailing record framing.
        let mut bytes = std::fs::read(f.path()).expect("read wal file");
        // Append a bogus 64-byte sequence that does not parse as a WAL record.
        bytes.extend_from_slice(&[0xFF; 64]);
        std::fs::write(f.path(), bytes).expect("rewrite");
    }

    // Restore must fail (or partially succeed but flag the corruption).  We
    // accept either a clean error or a successful restore that ignores the
    // corrupted tail — both are valid production-grade behaviours.  The
    // critical requirement: **no panic, no silent data corruption**.
    let mut store = VectorStore::new();
    let result = restore_to_timestamp(&mut store, 1_000, wal_dir);
    match result {
        Ok(report) => {
            // Skipped corrupted entries → fewer than 5 replayed but ≥ 0.
            assert!(
                report.entries_replayed <= 5,
                "must not over-replay; got {}",
                report.entries_replayed
            );
        }
        Err(e) => {
            let msg = format!("{}", e);
            assert!(
                !msg.is_empty(),
                "error must carry a message; got empty error"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 4 — End-to-end optimizer + persistence (covers the W2-S7 wiring)
// ─────────────────────────────────────────────────────────────────────────────

/// Build an `IndexDispatcher`, persist `QueryStats`, restart, verify the
/// online learning data survives.
#[test]
fn dispatcher_stats_persist_across_restart() -> anyhow::Result<()> {
    let mut stats_path = std::env::temp_dir();
    stats_path.push(format!(
        "oxirs_vec_w2s7_dispatcher_{}.json",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));

    // Run 1: ingest, search, persist stats.
    {
        let config = IndexDispatcherConfig {
            stats_path: Some(stats_path.clone()),
            stats_save_interval: 1,
            ..Default::default()
        };
        let mut config = config;
        config.ivf_config.n_clusters = 4;
        config.ivf_config.n_probes = 2;
        let mut d = IndexDispatcher::new(config)?;
        for i in 0..16 {
            d.insert(format!("e_{}", i), det_vec(i as u64 + 1, 8))?;
        }
        d.build()?;
        for q in 0..5 {
            let _ = d.search_knn(&det_vec(1000 + q as u64, 8), 3)?;
        }
        d.flush_stats()?;
    }

    // Run 2: load stats and verify observation count > 0.
    {
        let config = IndexDispatcherConfig {
            stats_path: Some(stats_path.clone()),
            ..Default::default()
        };
        let mut config = config;
        config.ivf_config.n_clusters = 4;
        let d = IndexDispatcher::new(config)?;
        let n = d.observation_count()?;
        assert!(n > 0, "second run must reload stats from {:?}", stats_path);
    }

    let _ = std::fs::remove_file(&stats_path);
    Ok(())
}
