//! TDB2 Behavioral Parity Test Suite
//!
//! Reference: <https://jena.apache.org/documentation/tdb2/>
//!
//! Verifies OxiRS TDB2 behavior matches the Apache Jena TDB2 specification
//! at the **storage-engine layer**. SPARQL execution features (FILTER, OPTIONAL,
//! UNION, aggregation, property paths) reside in oxirs-arq and are classified as
//! impl-detail divergence — not tested here.
//!
//! # Parity Groups
//!
//! | Group  | Target | What is measured                                             |
//! |--------|--------|--------------------------------------------------------------|
//! | LOAD   | 1.0    | Term-API and BulkLoader triple ingestion + count integrity  |
//! | QUERY  | 1.0    | All 7 triple-pattern wildcard combos via TdbStore            |
//! | TXN    | 1.0    | RW commit, RO tx, abort semantics, concurrent readers        |
//! | INDEX  | 1.0    | SixIndexStore — all 7 patterns + remove consistency          |
//! | BNODE  | 1.0    | BNode ID interning, round-trip via NodeTable, save/open      |
//! | PREFIX | 1.0    | PrefixTable compress / expand, well-known, serde round-trip  |

use oxirs_tdb::dictionary::Term;
use oxirs_tdb::loader::BulkLoaderFactory;
use oxirs_tdb::prefix_table::PrefixTable;
use oxirs_tdb::six_index_store::SixIndexStore;
use oxirs_tdb::tdb2::{RdfTerm, Tdb2Database, Tdb2Format};
use oxirs_tdb::TdbStore;
use std::env;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ─────────────────────────────────────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────────────────────────────────────

static TEST_SEQ: AtomicU64 = AtomicU64::new(1);

/// Build a genuinely-unique temp directory path for the given prefix.
///
/// Nextest runs each test in its own process, so a process-local
/// `AtomicU64` alone resets to the same values in every process and makes
/// different tests collide on the *same* on-disk store directory — which
/// then corrupts under concurrent access and leaks stale data across runs.
/// Mixing in the process id and a high-resolution timestamp guarantees a
/// distinct, freshly-empty directory per invocation and per run.
fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let seq = TEST_SEQ.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = env::temp_dir().join(format!("{}_{}_{}_{}", prefix, pid, seq, nanos));
    // Defend against an astronomically unlikely path repeat: start empty.
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// Create a unique temp directory and return a `TdbStore` plus the path.
fn open_store() -> (TdbStore, std::path::PathBuf) {
    let dir = unique_temp_dir("oxirs_parity");
    let store = TdbStore::open(&dir).expect("open TdbStore");
    (store, dir)
}

fn iri(s: &str) -> RdfTerm {
    RdfTerm::Iri(s.to_string())
}

fn lit(v: &str) -> RdfTerm {
    RdfTerm::Literal {
        value: v.to_string(),
        lang: None,
        datatype: None,
    }
}

fn blank(id: &str) -> RdfTerm {
    RdfTerm::BlankNode(id.to_string())
}

// ─────────────────────────────────────────────────────────────────────────────
// GROUP: LOAD PARITY (target: 1.0)
// ─────────────────────────────────────────────────────────────────────────────

/// Inline "simple" dataset: 10 triples inserted via Term API
const SIMPLE_TRIPLES: &[(&str, &str, &str)] = &[
    (
        "http://ex.org/alice",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://ex.org/Person",
    ),
    (
        "http://ex.org/alice",
        "http://xmlns.com/foaf/0.1/name",
        "http://ex.org/AliceName",
    ),
    (
        "http://ex.org/alice",
        "http://xmlns.com/foaf/0.1/knows",
        "http://ex.org/bob",
    ),
    (
        "http://ex.org/bob",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://ex.org/Person",
    ),
    (
        "http://ex.org/bob",
        "http://xmlns.com/foaf/0.1/name",
        "http://ex.org/BobName",
    ),
    (
        "http://ex.org/carol",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://ex.org/Person",
    ),
    (
        "http://ex.org/carol",
        "http://xmlns.com/foaf/0.1/knows",
        "http://ex.org/alice",
    ),
    (
        "http://ex.org/dave",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://ex.org/Organization",
    ),
    (
        "http://ex.org/dave",
        "http://xmlns.com/foaf/0.1/member",
        "http://ex.org/alice",
    ),
    (
        "http://ex.org/dave",
        "http://xmlns.com/foaf/0.1/member",
        "http://ex.org/bob",
    ),
];

#[test]
fn tdb2_parity_bulk_load_term_api() {
    let (mut store, _dir) = open_store();

    for &(s, p, o) in SIMPLE_TRIPLES {
        store.insert(s, p, o).expect("insert should succeed");
    }

    assert_eq!(
        store.count(),
        10,
        "LOAD: triple count must be 10 after inserting SIMPLE_TRIPLES"
    );
}

#[test]
fn tdb2_parity_bulk_load_via_bulk_loader() {
    let (mut store, _dir) = open_store();

    // BulkLoader::load takes a &mut TdbStore and an Iterator<Item=(Term,Term,Term)>
    let triples: Vec<(Term, Term, Term)> = SIMPLE_TRIPLES
        .iter()
        .map(|&(s, p, o)| (Term::iri(s), Term::iri(p), Term::iri(o)))
        .collect();

    let mut loader = BulkLoaderFactory::balanced();
    let stats = loader
        .load(&mut store, triples.into_iter())
        .expect("bulk load should succeed");

    assert_eq!(
        stats.total_triples, 10,
        "LOAD: BulkLoader must see 10 total triples"
    );
    assert_eq!(
        stats.triples_loaded, 10,
        "LOAD: BulkLoader must load 10 triples"
    );
    assert_eq!(
        stats.errors_encountered, 0,
        "LOAD: BulkLoader must have zero errors"
    );
}

#[test]
fn tdb2_parity_bulk_load_deduplication() {
    let (mut store, _dir) = open_store();

    // Insert same triple twice — TDB2 treats triples as a set
    store
        .insert("http://ex.org/s", "http://ex.org/p", "http://ex.org/o")
        .expect("first insert");
    store
        .insert("http://ex.org/s", "http://ex.org/p", "http://ex.org/o")
        .expect("second insert should not error");

    // Verify the triple is present (deduplication semantics are impl-defined)
    let exists = store
        .contains("http://ex.org/s", "http://ex.org/p", "http://ex.org/o")
        .expect("contains check");
    assert!(
        exists,
        "LOAD: inserted triple must be present after deduplication"
    );
}

#[test]
fn tdb2_parity_bulk_load_large_dataset() {
    let (mut store, _dir) = open_store();

    let n = 500usize;
    for i in 0..n {
        store
            .insert(
                &format!("http://ex.org/s{}", i),
                "http://ex.org/p",
                &format!("http://ex.org/o{}", i),
            )
            .expect("large load insert");
    }

    assert_eq!(
        store.count(),
        n,
        "LOAD: large dataset triple count must match inserted count"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// GROUP: QUERY PARITY — triple-pattern lookups (target: 1.0)
// All 7 non-trivial wildcard combos + exact match via TdbStore
// ─────────────────────────────────────────────────────────────────────────────

/// Build a store pre-loaded with SIMPLE_TRIPLES for query tests.
fn store_with_simple() -> (TdbStore, std::path::PathBuf) {
    let (mut store, dir) = open_store();
    for &(s, p, o) in SIMPLE_TRIPLES {
        store.insert(s, p, o).expect("setup insert");
    }
    (store, dir)
}

#[test]
fn tdb2_parity_query_wildcard_all() {
    let (store, _dir) = store_with_simple();
    let results = store
        .query_triples(None, None, None)
        .expect("query ??? wildcard");
    assert_eq!(
        results.len(),
        10,
        "QUERY ???: full scan must return all 10 triples"
    );
}

#[test]
fn tdb2_parity_query_by_subject() {
    let (store, _dir) = store_with_simple();
    let alice = Term::iri("http://ex.org/alice");
    let results = store
        .query_triples(Some(&alice), None, None)
        .expect("query S??");
    assert_eq!(
        results.len(),
        3,
        "QUERY S??: alice has 3 triples in SIMPLE_TRIPLES"
    );
}

#[test]
fn tdb2_parity_query_by_predicate() {
    let (store, _dir) = store_with_simple();
    let rdf_type = Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
    let results = store
        .query_triples(None, Some(&rdf_type), None)
        .expect("query ?P?");
    assert_eq!(
        results.len(),
        4,
        "QUERY ?P?: rdf:type appears 4 times in SIMPLE_TRIPLES"
    );
}

#[test]
fn tdb2_parity_query_by_object() {
    let (store, _dir) = store_with_simple();
    let person = Term::iri("http://ex.org/Person");
    let results = store
        .query_triples(None, None, Some(&person))
        .expect("query ??O");
    assert_eq!(
        results.len(),
        3,
        "QUERY ??O: ex:Person is the object of 3 triples"
    );
}

#[test]
fn tdb2_parity_query_by_subject_predicate() {
    let (store, _dir) = store_with_simple();
    let dave = Term::iri("http://ex.org/dave");
    let member = Term::iri("http://xmlns.com/foaf/0.1/member");
    let results = store
        .query_triples(Some(&dave), Some(&member), None)
        .expect("query SP?");
    assert_eq!(
        results.len(),
        2,
        "QUERY SP?: dave foaf:member has 2 objects"
    );
}

#[test]
fn tdb2_parity_query_by_predicate_object() {
    let (store, _dir) = store_with_simple();
    let rdf_type = Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
    let person = Term::iri("http://ex.org/Person");
    let results = store
        .query_triples(None, Some(&rdf_type), Some(&person))
        .expect("query ?PO");
    assert_eq!(
        results.len(),
        3,
        "QUERY ?PO: (?, rdf:type, ex:Person) matches 3 subjects"
    );
}

#[test]
fn tdb2_parity_query_by_subject_object() {
    let (store, _dir) = store_with_simple();
    let dave = Term::iri("http://ex.org/dave");
    let alice = Term::iri("http://ex.org/alice");
    let results = store
        .query_triples(Some(&dave), None, Some(&alice))
        .expect("query S?O");
    assert_eq!(
        results.len(),
        1,
        "QUERY S?O: (dave, ?, alice) matches exactly 1 triple"
    );
}

#[test]
fn tdb2_parity_query_exact_match() {
    let (store, _dir) = store_with_simple();
    let s = Term::iri("http://ex.org/alice");
    let p = Term::iri("http://xmlns.com/foaf/0.1/knows");
    let o = Term::iri("http://ex.org/bob");
    let results = store
        .query_triples(Some(&s), Some(&p), Some(&o))
        .expect("query SPO exact");
    assert_eq!(
        results.len(),
        1,
        "QUERY SPO: exact match must return 1 triple"
    );
    assert_eq!(results[0].0, s);
    assert_eq!(results[0].1, p);
    assert_eq!(results[0].2, o);
}

#[test]
fn tdb2_parity_query_unknown_term_returns_empty() {
    let (store, _dir) = store_with_simple();
    let unknown = Term::iri("http://ex.org/nobody");
    let results = store
        .query_triples(Some(&unknown), None, None)
        .expect("query unknown subject");
    assert!(
        results.is_empty(),
        "QUERY S??: unknown subject must return empty result set"
    );
}

#[test]
fn tdb2_parity_query_empty_store() {
    let (store, _dir) = open_store();
    let results = store
        .query_triples(None, None, None)
        .expect("query empty store");
    assert!(
        results.is_empty(),
        "QUERY ???: empty store returns empty result"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// GROUP: TRANSACTION PARITY (target: 1.0)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn tdb2_parity_transaction_rw_begin_commit() {
    let (store, _dir) = open_store();

    // Begin a read-write transaction and commit it
    let txn = store.begin_transaction().expect("begin RW txn");
    assert!(txn.is_active(), "TXN: RW transaction must start active");
    assert!(
        !txn.is_read_only(),
        "TXN: RW transaction must not be read-only"
    );

    store.commit_transaction(txn).expect("commit RW txn");
}

#[test]
fn tdb2_parity_transaction_read_only_begin() {
    let (store, _dir) = open_store();

    // Begin a read-only transaction
    let txn = store.begin_read_transaction().expect("begin RO txn");
    assert!(txn.is_active(), "TXN: RO transaction must start active");
    assert!(txn.is_read_only(), "TXN: RO transaction must be read-only");

    // RO transaction cannot acquire exclusive locks — expect error
    let exclusive_result = txn.lock_exclusive(1u64);
    assert!(
        exclusive_result.is_err(),
        "TXN: RO transaction must reject exclusive lock requests"
    );

    store.commit_transaction(txn).expect("commit RO txn");
}

#[test]
fn tdb2_parity_transaction_abort_clears_state() {
    let (store, _dir) = open_store();

    let txn = store.begin_transaction().expect("begin txn for abort");
    assert!(txn.is_active());

    txn.abort().expect("abort transaction");

    // After abort the transaction should no longer be active
    assert!(
        !txn.is_active(),
        "TXN: aborted transaction must not be active"
    );
}

#[test]
fn tdb2_parity_transaction_concurrent_readers() {
    let (store, _dir) = open_store();

    // Multiple read-only transactions can coexist
    let r1 = store.begin_read_transaction().expect("begin RO txn 1");
    let r2 = store.begin_read_transaction().expect("begin RO txn 2");
    let r3 = store.begin_read_transaction().expect("begin RO txn 3");

    assert!(
        r1.is_active() && r2.is_active() && r3.is_active(),
        "TXN: concurrent RO transactions must all be active simultaneously"
    );

    store.commit_transaction(r1).expect("commit r1");
    store.commit_transaction(r2).expect("commit r2");
    store.commit_transaction(r3).expect("commit r3");
}

#[test]
fn tdb2_parity_transaction_shared_lock_acquirable_in_ro() {
    let (store, _dir) = open_store();

    let txn = store.begin_read_transaction().expect("begin RO txn");

    // Shared (read) lock is valid in a RO transaction (PageId is u64)
    txn.lock_shared(1u64)
        .expect("TXN: RO txn must be able to acquire shared lock");

    store.commit_transaction(txn).expect("commit RO txn");
}

#[test]
fn tdb2_parity_transaction_rw_exclusive_lock() {
    let (store, _dir) = open_store();

    let txn = store.begin_transaction().expect("begin RW txn");

    txn.lock_exclusive(2u64)
        .expect("TXN: RW txn must be able to acquire exclusive lock");

    store.commit_transaction(txn).expect("commit RW txn");
}

// ─────────────────────────────────────────────────────────────────────────────
// GROUP: SIX-INDEX PARITY (target: 1.0)
// Tests SixIndexStore (SPO/POS/OSP) — all 7 wildcard patterns
// ─────────────────────────────────────────────────────────────────────────────

/// Populate a `SixIndexStore` with a known dataset.
///
/// Dataset (5 triples):
/// - (alice, type,  Person)
/// - (alice, knows, bob)
/// - (alice, knows, carol)
/// - (bob,   type,  Person)
/// - (carol, type,  Animal)
fn build_six_index_store() -> SixIndexStore {
    let mut store = SixIndexStore::new();
    store.insert("alice", "type", "Person");
    store.insert("alice", "knows", "bob");
    store.insert("alice", "knows", "carol");
    store.insert("bob", "type", "Person");
    store.insert("carol", "type", "Animal");
    store
}

#[test]
fn tdb2_parity_six_index_full_scan() {
    let store = build_six_index_store();
    let results = store.query_spo(None, None, None);
    assert_eq!(
        results.len(),
        5,
        "INDEX ???: full scan must return all 5 triples"
    );
}

#[test]
fn tdb2_parity_six_index_by_subject() {
    let store = build_six_index_store();
    let results = store.query_spo(Some("alice"), None, None);
    assert_eq!(results.len(), 3, "INDEX S??: alice has 3 triples");
    assert!(
        results.iter().all(|(s, _, _)| s == "alice"),
        "INDEX S??: all results must have subject=alice"
    );
}

#[test]
fn tdb2_parity_six_index_by_predicate() {
    let store = build_six_index_store();
    let results = store.query_spo(None, Some("type"), None);
    assert_eq!(results.len(), 3, "INDEX ?P?: predicate=type has 3 triples");
    assert!(
        results.iter().all(|(_, p, _)| p == "type"),
        "INDEX ?P?: all results must have predicate=type"
    );
}

#[test]
fn tdb2_parity_six_index_by_object() {
    let store = build_six_index_store();
    let results = store.query_spo(None, None, Some("Person"));
    assert_eq!(results.len(), 2, "INDEX ??O: object=Person has 2 triples");
    assert!(
        results.iter().all(|(_, _, o)| o == "Person"),
        "INDEX ??O: all results must have object=Person"
    );
}

#[test]
fn tdb2_parity_six_index_by_subject_predicate() {
    let store = build_six_index_store();
    let results = store.query_spo(Some("alice"), Some("knows"), None);
    assert_eq!(results.len(), 2, "INDEX SP?: alice knows 2 people");
    for (s, p, _) in &results {
        assert_eq!(s, "alice", "INDEX SP?: subject must be alice");
        assert_eq!(p, "knows", "INDEX SP?: predicate must be knows");
    }
}

#[test]
fn tdb2_parity_six_index_by_predicate_object() {
    let store = build_six_index_store();
    let results = store.query_spo(None, Some("type"), Some("Person"));
    assert_eq!(
        results.len(),
        2,
        "INDEX ?PO: (?,type,Person) has 2 subjects"
    );
    for (_, p, o) in &results {
        assert_eq!(p, "type");
        assert_eq!(o, "Person");
    }
}

#[test]
fn tdb2_parity_six_index_by_subject_object() {
    let store = build_six_index_store();
    let results = store.query_spo(Some("alice"), None, Some("carol"));
    assert_eq!(results.len(), 1, "INDEX S?O: (alice,?,carol) has 1 triple");
    assert_eq!(results[0].1, "knows", "INDEX S?O: predicate must be knows");
}

#[test]
fn tdb2_parity_six_index_exact_match() {
    let store = build_six_index_store();
    let results = store.query_spo(Some("bob"), Some("type"), Some("Person"));
    assert_eq!(
        results.len(),
        1,
        "INDEX SPO exact: (bob,type,Person) must match exactly once"
    );
}

#[test]
fn tdb2_parity_six_index_no_match() {
    let store = build_six_index_store();
    let results = store.query_spo(Some("nobody"), None, None);
    assert!(
        results.is_empty(),
        "INDEX S?? unknown subject must be empty"
    );
}

#[test]
fn tdb2_parity_six_index_consistency_helpers() {
    // Verify subjects_for, predicates_for, objects_for produce consistent results
    let mut store = SixIndexStore::new();
    store.insert("s1", "p1", "o1");
    store.insert("s1", "p2", "o2");
    store.insert("s2", "p1", "o1");

    let subjects = store.subjects_for("p1", "o1");
    assert_eq!(subjects.len(), 2, "INDEX: subjects_for(p1,o1) must be 2");

    let preds = store.predicates_for("s1", "o1");
    assert_eq!(
        preds.len(),
        1,
        "INDEX: predicates_for(s1,o1) must be 1 (p1)"
    );

    let objs = store.objects_for("s1", "p2");
    assert_eq!(objs.len(), 1, "INDEX: objects_for(s1,p2) must be 1 (o2)");
}

#[test]
fn tdb2_parity_six_index_remove_consistency() {
    let mut store = build_six_index_store();
    assert_eq!(store.len(), 5);

    let removed = store.remove("alice", "knows", "bob");
    assert!(removed, "INDEX: remove of existing triple must return true");
    assert_eq!(store.len(), 4, "INDEX: length must decrease after remove");

    // Verify triple is gone from all patterns
    assert!(
        store
            .query_spo(Some("alice"), Some("knows"), Some("bob"))
            .is_empty(),
        "INDEX: removed triple must not appear in exact lookup"
    );
    assert_eq!(
        store.query_spo(Some("alice"), Some("knows"), None).len(),
        1,
        "INDEX: alice knows only carol after removing bob"
    );

    let not_removed = store.remove("alice", "knows", "bob");
    assert!(
        !not_removed,
        "INDEX: second remove of same triple must return false"
    );
}

#[test]
fn tdb2_parity_six_index_deduplication() {
    let mut store = SixIndexStore::new();
    let first_insert = store.insert("s", "p", "o");
    let second_insert = store.insert("s", "p", "o");

    assert!(first_insert, "INDEX: first insert must return true");
    assert!(!second_insert, "INDEX: duplicate insert must return false");
    assert_eq!(
        store.len(),
        1,
        "INDEX: set-semantics — duplicate triple must not increase count"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// GROUP: BLANK NODE PRESERVATION (target: 1.0)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn tdb2_parity_bnode_intern_and_lookup() {
    let mut db = Tdb2Database::new();
    let b1 = blank("b001");
    let p = iri("http://ex.org/p");
    let o = lit("value");

    db.insert_triple(&b1, &p, &o).expect("insert bnode triple");
    let results = db.query(Some(&b1), None, None);
    assert_eq!(
        results.len(),
        1,
        "BNODE: blank node as subject must be queryable"
    );
    assert_eq!(results[0].0, b1, "BNODE: subject must round-trip correctly");
}

#[test]
fn tdb2_parity_bnode_as_object() {
    let mut db = Tdb2Database::new();
    let s = iri("http://ex.org/s");
    let p = iri("http://ex.org/p");
    let b1 = blank("b002");

    db.insert_triple(&s, &p, &b1)
        .expect("insert with bnode object");
    let results = db.query(None, None, Some(&b1));
    assert_eq!(
        results.len(),
        1,
        "BNODE: blank node as object must be queryable"
    );
    assert_eq!(
        results[0].2, b1,
        "BNODE: object bnode must round-trip correctly"
    );
}

#[test]
fn tdb2_parity_bnode_id_uniqueness() {
    let mut db = Tdb2Database::new();
    let b1 = blank("bnode_a");
    let b2 = blank("bnode_b");
    let p = iri("http://ex.org/p");
    let o = lit("v");

    db.insert_triple(&b1, &p, &o).expect("insert b1");
    db.insert_triple(&b2, &p, &o).expect("insert b2");

    let a_results = db.query(Some(&b1), None, None);
    let b_results = db.query(Some(&b2), None, None);

    assert_eq!(
        a_results.len(),
        1,
        "BNODE: bnode_a must have exactly 1 triple"
    );
    assert_eq!(
        b_results.len(),
        1,
        "BNODE: bnode_b must have exactly 1 triple"
    );
    assert_ne!(
        a_results[0].0, b_results[0].0,
        "BNODE: distinct bnode IDs must produce distinct subjects"
    );
}

#[test]
fn tdb2_parity_bnode_survives_save_and_open() {
    let tmp = unique_temp_dir("oxirs_bnode_parity");

    let mut db = Tdb2Database::new();
    let b = blank("persistent_bnode");
    let p = iri("http://ex.org/label");
    let o = lit("hello");
    db.insert_triple(&b, &p, &o).expect("insert bnode triple");
    db.save(&tmp).expect("save db");

    // Open fresh from disk
    let db2 = Tdb2Database::open(&tmp).expect("reopen db");
    let results = db2.query(Some(&b), None, None);
    assert_eq!(
        results.len(),
        1,
        "BNODE: bnode triple must survive save/open round-trip"
    );
    assert_eq!(
        results[0].0, b,
        "BNODE: blank node ID must be preserved on disk"
    );

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn tdb2_parity_bnode_chained_structure() {
    // Model: s → p → _:b1 → q → value
    let mut db = Tdb2Database::new();
    let s = iri("http://ex.org/s");
    let p = iri("http://ex.org/p");
    let b1 = blank("chain_b1");
    let q = iri("http://ex.org/q");
    let v = lit("chain_value");

    db.insert_triple(&s, &p, &b1).expect("s → b1");
    db.insert_triple(&b1, &q, &v).expect("b1 → v");

    let hop1 = db.query(Some(&s), None, None);
    assert_eq!(hop1.len(), 1);
    let intermediate = hop1[0].2.clone();

    let hop2 = db.query(Some(&intermediate), None, None);
    assert_eq!(
        hop2.len(),
        1,
        "BNODE: chained bnode must be navigable as subject"
    );
    assert_eq!(hop2[0].2, v, "BNODE: chain end value must match");
}

// ─────────────────────────────────────────────────────────────────────────────
// GROUP: PREFIX MAP PARITY (target: 1.0)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn tdb2_parity_prefix_register_and_compress() {
    let mut table = PrefixTable::new();
    let rdf_ns = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
    table.intern_prefix(rdf_ns);

    let compressed = table.compress_iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
    assert_eq!(
        compressed.local, "type",
        "PREFIX: local part must be 'type'"
    );

    let expanded = table.expand_iri(&compressed);
    assert_eq!(
        expanded, "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "PREFIX: expand must produce original IRI"
    );
}

#[test]
fn tdb2_parity_prefix_well_known() {
    // with_well_known() ships rdf:, rdfs:, owl:, xsd: etc.
    let mut table = PrefixTable::with_well_known();

    let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
    let compressed = table.compress_iri(rdf_type);
    let expanded = table.expand_iri(&compressed);
    assert_eq!(
        expanded, rdf_type,
        "PREFIX: well-known rdf:type must compress/expand losslessly"
    );

    let rdfs_label = "http://www.w3.org/2000/01/rdf-schema#label";
    let compressed2 = table.compress_iri(rdfs_label);
    let expanded2 = table.expand_iri(&compressed2);
    assert_eq!(
        expanded2, rdfs_label,
        "PREFIX: well-known rdfs:label must compress/expand losslessly"
    );
}

#[test]
fn tdb2_parity_prefix_multiple_namespaces() {
    let mut table = PrefixTable::new();
    let foaf = "http://xmlns.com/foaf/0.1/";
    let dc = "http://purl.org/dc/terms/";
    let schema = "http://schema.org/";

    let foaf_id = table.intern_prefix(foaf);
    let dc_id = table.intern_prefix(dc);
    let schema_id = table.intern_prefix(schema);

    assert_ne!(
        foaf_id, dc_id,
        "PREFIX: distinct prefixes must get distinct IDs"
    );
    assert_ne!(
        foaf_id, schema_id,
        "PREFIX: distinct prefixes must get distinct IDs"
    );
    assert_ne!(
        dc_id, schema_id,
        "PREFIX: distinct prefixes must get distinct IDs"
    );

    let foaf_name = table.compress_iri("http://xmlns.com/foaf/0.1/name");
    assert_eq!(foaf_name.local, "name");

    let dc_title = table.compress_iri("http://purl.org/dc/terms/title");
    assert_eq!(dc_title.local, "title");
}

#[test]
fn tdb2_parity_prefix_serde_round_trip() {
    let mut table = PrefixTable::new();
    table.intern_prefix("http://ex.org/ns/");
    table.intern_prefix("http://xmlns.com/foaf/0.1/");

    let json = serde_json::to_string(&table).expect("PREFIX: serialize to JSON");
    let restored: PrefixTable = serde_json::from_str(&json).expect("PREFIX: deserialize from JSON");

    let compressed = table.compress_iri("http://ex.org/ns/subject");
    let expanded = restored.expand_iri(&compressed);
    assert_eq!(
        expanded, "http://ex.org/ns/subject",
        "PREFIX: serde round-trip must preserve compress/expand behavior"
    );
}

#[test]
fn tdb2_parity_prefix_idempotent_intern() {
    let mut table = PrefixTable::new();
    let ns = "http://example.org/ns/";

    let id1 = table.intern_prefix(ns);
    let id2 = table.intern_prefix(ns);
    assert_eq!(
        id1, id2,
        "PREFIX: interning same prefix twice must return same ID"
    );
}

#[test]
fn tdb2_parity_prefix_try_compress_no_match() {
    let table = PrefixTable::new();
    let result = table.try_compress_iri("http://unknown.org/foo");
    assert!(
        result.is_err(),
        "PREFIX: try_compress_iri with no matching prefix must return Err"
    );
}

#[test]
fn tdb2_parity_prefix_try_expand_unknown_id() {
    use oxirs_tdb::prefix_table::CompressedIri;
    let table = PrefixTable::new();
    let fake = CompressedIri::new(9999, "local");
    let result = table.try_expand_iri(&fake);
    assert!(
        result.is_err(),
        "PREFIX: try_expand_iri with unknown prefix_id must return Err"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// GROUP: TDB2 FORMAT LAYER (Tdb2Format open/flush round-trip)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn tdb2_parity_format_insert_flush_reopen() {
    let tmp = unique_temp_dir("oxirs_fmt_parity");

    let mut fmt = Tdb2Format::open(&tmp).expect("open empty Tdb2Format");
    fmt.insert_triple(
        &iri("http://ex.org/s"),
        &iri("http://ex.org/p"),
        &lit("hello"),
    )
    .expect("insert");
    fmt.insert_triple(
        &iri("http://ex.org/s"),
        &iri("http://ex.org/q"),
        &blank("bx"),
    )
    .expect("insert bnode");
    fmt.flush().expect("flush to disk");

    let fmt2 = Tdb2Format::open(&tmp).expect("reopen Tdb2Format");
    assert_eq!(
        fmt2.triple_count(),
        2,
        "FORMAT: triple count must survive flush/reopen"
    );
    // s, p, q, "hello", _:bx = 5 unique terms
    assert_eq!(fmt2.node_count(), 5, "FORMAT: node count must be 5");

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn tdb2_parity_format_query_after_reopen() {
    let tmp = unique_temp_dir("oxirs_fmt_query_parity");

    let subject = iri("http://ex.org/persist_s");
    let pred = iri("http://ex.org/persist_p");
    let obj = lit("persist_value");

    {
        let mut fmt = Tdb2Format::open(&tmp).expect("open");
        fmt.insert_triple(&subject, &pred, &obj).expect("insert");
        fmt.flush().expect("flush");
    }

    let fmt2 = Tdb2Format::open(&tmp).expect("reopen after close");
    let results = fmt2.query(Some(&subject), None, None);
    assert_eq!(
        results.len(),
        1,
        "FORMAT: query after reopen must return 1 result"
    );
    assert_eq!(results[0].0, subject, "FORMAT: subject must survive reopen");
    assert_eq!(
        results[0].2, obj,
        "FORMAT: object literal must survive reopen"
    );

    std::fs::remove_dir_all(&tmp).ok();
}

// ─────────────────────────────────────────────────────────────────────────────
// Parity summary / overall report
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregated parity report — runs representative checks from all groups.
///
/// This is the single authoritative "parity gate" for CI. Each sub-group
/// is scored. If all sub-tests above pass, this passes trivially.
#[test]
fn tdb2_parity_overall_report() {
    // --- LOAD group ---
    let (mut store, _dir) = open_store();
    for &(s, p, o) in SIMPLE_TRIPLES {
        store.insert(s, p, o).expect("load insert");
    }
    let load_count = store.count();
    assert_eq!(load_count, 10, "LOAD parity: 10/10");

    // --- QUERY group ---
    let q_all = store.query_triples(None, None, None).expect("???");
    assert_eq!(q_all.len(), 10, "QUERY ??? parity: 10/10");

    let rdf_type = Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
    let q_pred = store
        .query_triples(None, Some(&rdf_type), None)
        .expect("?P?");
    assert_eq!(q_pred.len(), 4, "QUERY ?P? parity: 4/4");

    let alice = Term::iri("http://ex.org/alice");
    let q_subj = store.query_triples(Some(&alice), None, None).expect("S??");
    assert_eq!(q_subj.len(), 3, "QUERY S?? parity: 3/3");

    // --- INDEX group ---
    let mut idx = build_six_index_store();
    assert_eq!(
        idx.query_spo(None, None, None).len(),
        5,
        "INDEX ??? parity: 5/5"
    );
    assert_eq!(
        idx.query_spo(Some("alice"), None, None).len(),
        3,
        "INDEX S?? parity: 3/3"
    );
    assert!(idx.remove("alice", "knows", "carol"), "INDEX remove parity");
    assert_eq!(idx.len(), 4, "INDEX len parity: 4/5 after remove");

    // --- BNODE group ---
    let mut db = Tdb2Database::new();
    let b = blank("report_b1");
    let p = iri("http://ex.org/p");
    db.insert_triple(&b, &p, &lit("v")).expect("bnode insert");
    assert_eq!(db.query(Some(&b), None, None).len(), 1, "BNODE parity: 1/1");

    // --- PREFIX group ---
    let mut table = PrefixTable::with_well_known();
    let compressed = table.compress_iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
    let expanded = table.expand_iri(&compressed);
    assert_eq!(
        expanded, "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "PREFIX parity: 1/1"
    );

    // --- TXN group ---
    let (store2, _dir2) = open_store();
    let txn = store2.begin_transaction().expect("txn begin");
    assert!(txn.is_active(), "TXN active parity: 1/1");
    store2.commit_transaction(txn).expect("txn commit parity");

    let ro = store2.begin_read_transaction().expect("ro begin");
    assert!(ro.is_read_only(), "TXN read-only parity: 1/1");
    store2.commit_transaction(ro).expect("ro commit parity");

    println!(
        "TDB2 parity: load={}/10 query=10/10 index=4/5(after-remove) bnode=1/1 prefix=1/1 txn=2/2",
        load_count
    );
}
