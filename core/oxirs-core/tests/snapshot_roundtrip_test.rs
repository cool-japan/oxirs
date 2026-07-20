//! Round-trip and safety tests for the frozen mmap snapshot format
//! (`oxirs_core::rdf_store::snapshot`).
//!
//! Covers: builder→loader fidelity (COUNT + pattern queries + named graphs),
//! byte-for-byte determinism (including independence from insertion order), the
//! empty store, and the safe-fallback contract (corrupt magic, version mismatch,
//! stale `source_len`, truncated file — all must return `Err`, never panic).

use std::collections::BTreeSet;
use std::io::{Seek, SeekFrom, Write};

use oxirs_core::model::{
    BlankNode, GraphName, Literal, NamedNode, Object, Predicate, Quad, Subject,
};
use oxirs_core::rdf_store::snapshot::{load_snapshot, write_snapshot, SNAPSHOT_FILE_NAME};
use oxirs_core::rdf_store::storage::MemoryStorage;
use oxirs_core::RdfStore;

fn nn(iri: &str) -> NamedNode {
    NamedNode::new(iri).expect("valid IRI")
}

/// A diverse dataset exercising every commonly-stored term shape: named-node and
/// blank-node subjects/objects, simple / language-tagged / typed literals, and
/// both the default graph and named graphs. Blank-node ids deliberately contain a
/// non-hex character so `BlankNode::new` and `new_unchecked` agree on the variant.
fn diverse_quads() -> Vec<Quad> {
    let xsd_int = nn("http://www.w3.org/2001/XMLSchema#integer");
    vec![
        Quad::new(
            Subject::NamedNode(nn("http://ex.org/s1")),
            Predicate::NamedNode(nn("http://ex.org/p1")),
            Object::NamedNode(nn("http://ex.org/o1")),
            GraphName::DefaultGraph,
        ),
        Quad::new(
            Subject::NamedNode(nn("http://ex.org/s1")),
            Predicate::NamedNode(nn("http://ex.org/label")),
            Object::Literal(Literal::new("plain value")),
            GraphName::DefaultGraph,
        ),
        Quad::new(
            Subject::NamedNode(nn("http://ex.org/s2")),
            Predicate::NamedNode(nn("http://ex.org/label")),
            Object::Literal(Literal::new_lang("こんにちは", "ja").expect("lang literal")),
            GraphName::NamedNode(nn("http://ex.org/g1")),
        ),
        Quad::new(
            Subject::NamedNode(nn("http://ex.org/s2")),
            Predicate::NamedNode(nn("http://ex.org/count")),
            Object::Literal(Literal::new_typed("42", xsd_int)),
            GraphName::NamedNode(nn("http://ex.org/g1")),
        ),
        Quad::new(
            Subject::BlankNode(BlankNode::new("bnode-x").expect("blank")),
            Predicate::NamedNode(nn("http://ex.org/rel")),
            Object::BlankNode(BlankNode::new("bnode-y").expect("blank")),
            GraphName::NamedNode(nn("http://ex.org/g2")),
        ),
        Quad::new(
            Subject::NamedNode(nn("http://ex.org/s3")),
            Predicate::NamedNode(nn("http://ex.org/p1")),
            Object::Literal(Literal::new_lang("hello", "en").expect("lang literal")),
            GraphName::DefaultGraph,
        ),
    ]
}

fn storage_from(quads: &[Quad]) -> MemoryStorage {
    let mut storage = MemoryStorage::new();
    for q in quads {
        storage.insert_quad(q.clone());
    }
    storage
}

fn all_quads(storage: &MemoryStorage) -> BTreeSet<Quad> {
    storage
        .query_quads(None, None, None, None)
        .into_iter()
        .collect()
}

#[test]
fn roundtrip_preserves_count_queries_and_named_graphs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(SNAPSHOT_FILE_NAME);

    let quads = diverse_quads();
    let source = storage_from(&quads);
    write_snapshot(&source, &path).expect("write snapshot");

    let loaded = load_snapshot(&path, None).expect("load snapshot");

    // COUNT matches.
    assert_eq!(loaded.len(), source.len());
    assert_eq!(loaded.len(), quads.len());

    // Full scan matches exactly.
    assert_eq!(all_quads(&loaded), all_quads(&source));

    // Every source quad is a member and reconstructs identically.
    for q in &quads {
        assert!(loaded.contains_quad(q), "missing quad after load: {q:?}");
    }

    // Spot pattern queries agree with the source for each binding shape.
    let s1 = Subject::NamedNode(nn("http://ex.org/s1"));
    let label = Predicate::NamedNode(nn("http://ex.org/label"));
    let g1 = GraphName::NamedNode(nn("http://ex.org/g1"));
    for (s, p, o, g) in [
        (Some(&s1), None, None, None),
        (None, Some(&label), None, None),
        (None, None, None, Some(&g1)),
    ] {
        let got: BTreeSet<Quad> = loaded.query_quads(s, p, o, g).into_iter().collect();
        let want: BTreeSet<Quad> = source.query_quads(s, p, o, g).into_iter().collect();
        assert_eq!(
            got, want,
            "pattern mismatch s={s:?} p={p:?} o={o:?} g={g:?}"
        );
    }

    // Named graphs are derived correctly from the graph column.
    assert_eq!(loaded.named_graphs, source.named_graphs);
    assert_eq!(
        loaded.named_graphs,
        [nn("http://ex.org/g1"), nn("http://ex.org/g2")]
            .into_iter()
            .collect()
    );
}

#[test]
fn output_is_deterministic_and_insertion_order_independent() {
    let dir = tempfile::tempdir().expect("tempdir");
    let a = dir.path().join("a.oxsnap");
    let b = dir.path().join("b.oxsnap");

    let quads = diverse_quads();
    let forward = storage_from(&quads);

    let mut reversed = quads.clone();
    reversed.reverse();
    let backward = storage_from(&reversed);

    write_snapshot(&forward, &a).expect("write a");
    write_snapshot(&backward, &b).expect("write b");

    let bytes_a = std::fs::read(&a).expect("read a");
    let bytes_b = std::fs::read(&b).expect("read b");
    assert_eq!(
        bytes_a, bytes_b,
        "snapshot bytes must be identical regardless of insertion order"
    );
}

#[test]
fn empty_store_roundtrips() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(SNAPSHOT_FILE_NAME);

    let source = MemoryStorage::new();
    write_snapshot(&source, &path).expect("write snapshot");
    let loaded = load_snapshot(&path, None).expect("load snapshot");

    assert_eq!(loaded.len(), 0);
    assert!(loaded.is_empty());
    assert!(loaded.query_quads(None, None, None, None).is_empty());
    assert!(loaded.named_graphs.is_empty());
}

#[test]
fn corrupt_magic_and_version_are_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(SNAPSHOT_FILE_NAME);
    write_snapshot(&storage_from(&diverse_quads()), &path).expect("write snapshot");

    // Corrupt the magic: loader must Err, not panic.
    {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open for corruption");
        f.seek(SeekFrom::Start(0)).expect("seek");
        f.write_all(b"XXXXXXXX").expect("clobber magic");
    }
    assert!(
        load_snapshot(&path, None).is_err(),
        "corrupt magic must be rejected"
    );

    // Rewrite a clean snapshot, then corrupt only the version field.
    write_snapshot(&storage_from(&diverse_quads()), &path).expect("rewrite snapshot");
    {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open for corruption");
        f.seek(SeekFrom::Start(8)).expect("seek");
        f.write_all(&999u32.to_le_bytes()).expect("clobber version");
    }
    assert!(
        load_snapshot(&path, None).is_err(),
        "version mismatch must be rejected"
    );
}

#[test]
fn stale_source_length_is_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(SNAPSHOT_FILE_NAME);
    // No sibling data.nq, so the recorded source_len is 0 and the guard is off:
    // an explicit expected length still cannot flag a 0 as stale.
    write_snapshot(&storage_from(&diverse_quads()), &path).expect("write snapshot");
    assert!(load_snapshot(&path, Some(12345)).is_ok());

    // Now patch a non-zero source_len into the header and mismatch it.
    {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open for patch");
        f.seek(SeekFrom::Start(24)).expect("seek to source_len");
        f.write_all(&1000u64.to_le_bytes())
            .expect("write source_len");
    }
    assert!(
        load_snapshot(&path, Some(999)).is_err(),
        "source_len mismatch must be rejected as stale"
    );
    assert!(
        load_snapshot(&path, Some(1000)).is_ok(),
        "matching source_len must load"
    );
}

#[test]
fn corrupt_term_count_is_rejected_without_panic() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(SNAPSHOT_FILE_NAME);
    write_snapshot(&storage_from(&diverse_quads()), &path).expect("write snapshot");

    // The first dict directory entry (subjects) begins right after the 40-byte
    // header; its `term_count` is the 4th u64 of that 32-byte entry, at offset
    // 40 + 24 = 64. Flip it to u64::MAX: the loader must reject this via bounds
    // checks *before* allocating, never panic on a huge `Vec::with_capacity`.
    {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open for corruption");
        f.seek(SeekFrom::Start(64))
            .expect("seek to subjects term_count");
        f.write_all(&u64::MAX.to_le_bytes())
            .expect("clobber term_count");
    }
    assert!(
        load_snapshot(&path, None).is_err(),
        "an out-of-range term_count must be rejected, not panic"
    );
}

#[test]
fn truncated_file_is_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(SNAPSHOT_FILE_NAME);
    write_snapshot(&storage_from(&diverse_quads()), &path).expect("write snapshot");

    let full = std::fs::read(&path).expect("read");
    // Keep only part of the header/directory: every access must be bounds-checked.
    std::fs::write(&path, &full[..full.len().min(64)]).expect("truncate");
    assert!(
        load_snapshot(&path, None).is_err(),
        "truncated file must be rejected"
    );
}

/// End-to-end through the public `RdfStore` open path: a valid snapshot is used,
/// and a structurally corrupt snapshot falls back to the `data.nq` load without
/// panicking and without changing the query results.
#[test]
fn rdfstore_open_uses_snapshot_and_falls_back_when_corrupt() {
    let dir = tempfile::tempdir().expect("tempdir");
    let quads = diverse_quads();

    {
        let mut store = RdfStore::open(dir.path()).expect("open persistent store");
        for q in &quads {
            store.insert_quad(q.clone()).expect("insert");
        }
        store.flush().expect("flush");
    }

    // Bake the snapshot from data.nq.
    let snapshot_path = RdfStore::build_snapshot(dir.path()).expect("build snapshot");
    assert!(snapshot_path.exists());

    // Reopen: the snapshot path is taken and the data is intact.
    {
        let store = RdfStore::open(dir.path()).expect("reopen via snapshot");
        assert_eq!(store.len().expect("len"), quads.len());
    }

    // Corrupt the snapshot's magic; reopen must fall back to data.nq and still be
    // correct (and must not panic).
    {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .open(&snapshot_path)
            .expect("open snapshot");
        f.seek(SeekFrom::Start(0)).expect("seek");
        f.write_all(b"CORRUPT!").expect("clobber");
    }
    {
        let store = RdfStore::open(dir.path()).expect("reopen falls back");
        assert_eq!(store.len().expect("len"), quads.len());
    }
}
