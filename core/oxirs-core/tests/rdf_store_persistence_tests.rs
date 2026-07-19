//! Durability and crash-recovery regression tests for the persistent
//! [`RdfStore`] backend.
//!
//! These tests exercise the append-based persistence, delete/clear compaction,
//! torn-trailing-line recovery, single-fsync bulk insert, and the refusal to
//! rewrite a file whose initial load reported errors. Temporary datasets live
//! under `std::env::temp_dir()` in unique per-test subdirectories.

use oxirs_core::model::{GraphName, Literal, NamedNode, Quad, Triple};
use oxirs_core::rdf_store::{RdfStore, SyncPolicy};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Create a unique temporary directory path (not yet created on disk).
fn unique_dir(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    std::env::temp_dir().join(format!(
        "oxirs_rdf_store_persist_{tag}_{}_{nanos}_{n}",
        std::process::id()
    ))
}

fn triple(s: &str, p: &str, o: &str) -> Triple {
    Triple::new(
        NamedNode::new(s).expect("valid subject IRI"),
        NamedNode::new(p).expect("valid predicate IRI"),
        Literal::new(o),
    )
}

fn named_quad(s: &str, p: &str, o: &str, g: &str) -> Quad {
    Quad::new(
        NamedNode::new(s).expect("valid subject IRI"),
        NamedNode::new(p).expect("valid predicate IRI"),
        Literal::new(o),
        GraphName::NamedNode(NamedNode::new(g).expect("valid graph IRI")),
    )
}

/// Inserts must survive a reopen without any explicit delete (append log
/// durability).
#[test]
fn test_insert_then_reopen_persists() {
    let dir = unique_dir("insert");
    {
        let mut store = RdfStore::open(&dir).expect("open persistent store");
        store
            .insert_triple(triple("http://ex/a", "http://ex/p", "1"))
            .expect("insert");
        store
            .insert_triple(triple("http://ex/b", "http://ex/p", "2"))
            .expect("insert");
        store
            .insert_triple(triple("http://ex/c", "http://ex/p", "3"))
            .expect("insert");
        store.flush().expect("flush");
    } // drop also flushes

    let store = RdfStore::open(&dir).expect("reopen persistent store");
    assert_eq!(
        store.len().expect("len"),
        3,
        "all inserts must survive reopen"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// A deletion must survive a reopen (compaction on flush/Drop). This is the
/// core P0 durability guarantee: deleted quads must not resurrect on restart.
#[test]
fn test_delete_then_reopen_durability() {
    let dir = unique_dir("delete");
    let removed = named_quad("http://ex/b", "http://ex/p", "2", "http://ex/g");
    {
        let mut store = RdfStore::open(&dir).expect("open");
        store
            .insert_quad(named_quad("http://ex/a", "http://ex/p", "1", "http://ex/g"))
            .expect("insert");
        store.insert_quad(removed.clone()).expect("insert");
        store
            .insert_quad(named_quad("http://ex/c", "http://ex/p", "3", "http://ex/g"))
            .expect("insert");
        assert!(store.remove_quad(&removed).expect("remove"));
        store.flush().expect("flush compacts the deletion");
    }

    let store = RdfStore::open(&dir).expect("reopen");
    assert_eq!(
        store.len().expect("len"),
        2,
        "deletion must persist across reopen"
    );
    assert!(
        !store.contains_quad(&removed).expect("contains"),
        "the removed quad must not resurrect"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// Delete durability must also hold when relying solely on `Drop` (no explicit
/// flush call) — graceful shutdown must not lose deletions.
#[test]
fn test_delete_then_reopen_via_drop_only() {
    let dir = unique_dir("delete_drop");
    let removed = triple("http://ex/y", "http://ex/p", "2");
    {
        let mut store = RdfStore::open(&dir).expect("open");
        store
            .insert_triple(triple("http://ex/x", "http://ex/p", "1"))
            .expect("insert");
        store.insert_triple(removed.clone()).expect("insert");
        assert!(store
            .remove_quad(&Quad::from_triple(removed.clone()))
            .expect("remove"));
        // No explicit flush; rely on Drop.
    }

    let store = RdfStore::open(&dir).expect("reopen");
    assert_eq!(
        store.len().expect("len"),
        1,
        "Drop must compact the deletion"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Clearing the store must persist: a reopen sees an empty store.
#[test]
fn test_clear_then_reopen_is_empty() {
    let dir = unique_dir("clear");
    {
        let mut store = RdfStore::open(&dir).expect("open");
        for i in 0..5 {
            store
                .insert_triple(triple("http://ex/s", "http://ex/p", &i.to_string()))
                .expect("insert");
        }
        assert_eq!(store.len().expect("len"), 5);
        store.clear().expect("clear");
        store.flush().expect("flush");
    }

    let store = RdfStore::open(&dir).expect("reopen");
    assert!(
        store.is_empty().expect("is_empty"),
        "clear must persist across reopen"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// A torn trailing line (crash mid-append) must be dropped on load and the file
/// repaired so subsequent appends do not merge into or lose data.
#[test]
fn test_torn_trailing_line_recovery() {
    let dir = unique_dir("torn");
    {
        let mut store = RdfStore::open(&dir).expect("open");
        store
            .insert_triple(triple("http://ex/a", "http://ex/p", "1"))
            .expect("insert");
        store
            .insert_triple(triple("http://ex/b", "http://ex/p", "2"))
            .expect("insert");
        store.flush().expect("flush");
    }

    // Simulate a crash mid-append: append a partial, unterminated line with no
    // trailing newline.
    let data_file = dir.join("data.nq");
    {
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&data_file)
            .expect("open data.nq for append");
        f.write_all(b"<http://ex/c> <http://ex/p> <http://ex/torn")
            .expect("write torn line");
        f.flush().expect("flush torn line");
    }

    // Reopen: the torn line must be dropped and truncated.
    {
        let store = RdfStore::open(&dir).expect("reopen after torn write");
        assert_eq!(
            store.len().expect("len"),
            2,
            "torn trailing line must be dropped"
        );
    }

    // A subsequent append must not merge with the truncated remainder.
    {
        let mut store = RdfStore::open(&dir).expect("reopen");
        store
            .insert_triple(triple("http://ex/d", "http://ex/p", "4"))
            .expect("insert");
        store.flush().expect("flush");
    }
    let store = RdfStore::open(&dir).expect("final reopen");
    assert_eq!(
        store.len().expect("len"),
        3,
        "new insert must be clean, no corruption"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// Bulk insert writes one append batch (single fsync) and returns real
/// per-quad novelty info; every quad must be durable and the file must contain
/// exactly one line per inserted quad.
#[test]
fn test_bulk_insert_single_batch_persists() {
    let dir = unique_dir("bulk");
    let mut quads = Vec::new();
    for i in 0..50 {
        quads.push(named_quad(
            &format!("http://ex/s{i}"),
            "http://ex/p",
            &i.to_string(),
            "http://ex/g",
        ));
    }
    // A duplicate to verify novelty reporting.
    let dup = quads[0].clone();

    {
        let mut store = RdfStore::open(&dir).expect("open");
        let ids = store.bulk_insert_quads(quads.clone()).expect("bulk insert");
        assert_eq!(ids.len(), 50);
        assert!(
            ids.iter().all(|&id| id == 1),
            "all fresh quads report novelty 1"
        );

        let ids2 = store.bulk_insert_quads(vec![dup]).expect("bulk insert dup");
        assert_eq!(ids2, vec![0], "duplicate reports novelty 0");
        assert_eq!(store.len().expect("len"), 50);
        store.flush().expect("flush");
    }

    // Exactly 50 non-empty lines on disk (no per-quad full rewrite duplication).
    let contents = std::fs::read_to_string(dir.join("data.nq")).expect("read data.nq");
    let line_count = contents.lines().filter(|l| !l.trim().is_empty()).count();
    assert_eq!(line_count, 50, "one N-Quads line per inserted quad");

    let store = RdfStore::open(&dir).expect("reopen");
    assert_eq!(
        store.len().expect("len"),
        50,
        "bulk insert must survive reopen"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// When the initial load reports parse errors, the store must refuse the first
/// compaction so unreadable-but-present data is never overwritten.
#[test]
fn test_refuse_compaction_after_load_errors() {
    let dir = unique_dir("loaderr");
    std::fs::create_dir_all(&dir).expect("mkdir");
    let data_file = dir.join("data.nq");

    // Two valid N-Quads lines with a corrupt line in the middle (not trailing).
    let good1 = "<http://ex/a> <http://ex/p> \"1\" <http://ex/g> .";
    let corrupt = "this is not a valid n-quads line";
    let good2 = "<http://ex/b> <http://ex/p> \"2\" <http://ex/g> .";
    std::fs::write(&data_file, format!("{good1}\n{corrupt}\n{good2}\n")).expect("seed data.nq");

    let mut store = RdfStore::open(&dir).expect("open despite corrupt line");
    // The two valid quads loaded; the corrupt line was counted/skipped.
    assert_eq!(store.len().expect("len"), 2);

    // A deletion marks the store dirty; flushing would compact (rewrite) it.
    let removed = Quad::new(
        NamedNode::new("http://ex/a").expect("iri"),
        NamedNode::new("http://ex/p").expect("iri"),
        Literal::new("1"),
        GraphName::NamedNode(NamedNode::new("http://ex/g").expect("iri")),
    );
    assert!(store.remove_quad(&removed).expect("remove"));

    // Compaction must be refused because the load had errors.
    let err = store
        .flush()
        .expect_err("compaction must be refused after load errors");
    let msg = err.to_string();
    assert!(
        msg.contains("Refusing to compact"),
        "unexpected error message: {msg}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// Performance smoke test: inserting 100k quads into the persistent backend must
/// stay linear (append-based), not the previous O(N^2) full-file rewrite. The
/// wall-clock budget is relaxed in debug builds, mirroring the repo's pattern.
#[test]
fn test_persistent_100k_insert_perf_smoke() {
    let dir = unique_dir("perf100k");
    // Use OnFlush so this measures append throughput, not fsync latency.
    let mut store = RdfStore::open_with_sync_policy(&dir, SyncPolicy::OnFlush).expect("open");

    let n = 100_000;
    let start = Instant::now();
    for i in 0..n {
        store
            .insert_triple(triple("http://ex/s", "http://ex/p", &i.to_string()))
            .expect("insert");
    }
    store.flush().expect("flush");
    let elapsed = start.elapsed();

    assert_eq!(store.len().expect("len"), n);

    // O(N^2) would take many minutes for 100k; a linear append path is seconds.
    let budget = if cfg!(debug_assertions) {
        Duration::from_secs(180)
    } else {
        Duration::from_secs(45)
    };
    assert!(
        elapsed < budget,
        "100k persistent inserts took {elapsed:?}, exceeding budget {budget:?} (possible O(N^2) regression)"
    );

    drop(store);
    std::fs::remove_dir_all(&dir).ok();
}
