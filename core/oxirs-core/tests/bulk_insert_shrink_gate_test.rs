//! Integration coverage for the gated dictionary shrink and per-batch reserve
//! added for the R7.5-B ingest-regression fix (adversarial finding P1).
//!
//! Two behaviours are pinned down end to end:
//!
//! * [`MemoryStorage::shrink_to_fit_if_slack`] only reallocates when the term
//!   dictionaries are over-provisioned past 2x their combined slot count (the
//!   gate), and reports whether it actually shrank — so a caller that invokes it
//!   after every batch no longer thrashes the dictionaries with a
//!   shrink→regrow→shrink treadmill or a per-batch `malloc_trim`.
//! * Repeated batched [`Store::bulk_insert_quads`] stays correct — counts,
//!   content, resolve round-trips, and a *bounded* footprint under insert/remove
//!   churn (the observable signature of free-id reuse) — while the per-batch
//!   `reserve_for_bulk_load` pre-sizes each batch.
//!
//! Every store operation goes through the [`Store`] trait explicitly
//! (`Store::bulk_insert_quads(&store, ..)`), because that is the seam the Fuseki
//! ingest path uses and the one carrying the reserve/gate fixes — the inherent
//! `RdfStore::bulk_insert_quads` is a distinct `&mut self` method.

use oxirs_core::model::{GraphName, Literal, NamedNode, Quad};
use oxirs_core::rdf_store::MemoryStorage;
use oxirs_core::{RdfStore, Store};

/// A distinct quad `s{i} p "object-{i}"` in the default graph.
fn quad(i: usize) -> Quad {
    Quad::new(
        NamedNode::new(format!("http://example.org/s{i}")).expect("subject IRI"),
        NamedNode::new("http://example.org/p").expect("predicate IRI"),
        Literal::new_simple_literal(format!("object-{i}")),
        GraphName::DefaultGraph,
    )
}

/// `[lo, hi)` distinct quads as one batch.
fn batch(lo: usize, hi: usize) -> Vec<Quad> {
    (lo..hi).map(quad).collect()
}

// ---------------------------------------------------------------------------
// Gate unit tests — direct on `MemoryStorage` so the 2x threshold is exercised
// deterministically, independent of any higher-level batching policy.
// ---------------------------------------------------------------------------

/// Over-provisioned dictionaries (a large reserve with only a few interned
/// terms) trip the gate, and the reclaim is real; a second call on the now-tight
/// store is a no-op. This second half is the crucial anti-thrash property: after
/// one shrink `capacity == slot_len`, so the gate must not keep firing.
#[test]
fn shrink_gate_fires_once_when_over_provisioned_then_stops() {
    let mut storage = MemoryStorage::new();
    // Reserve room for ~1000 terms but only intern four, so the object/subject
    // columns' capacity dwarfs their slot count.
    storage.reserve_for_bulk_load(1000);
    for i in 0..4 {
        assert!(storage.insert_quad(quad(i)), "distinct quad must be new");
    }

    let before = storage.size_estimate();
    assert!(
        storage.shrink_to_fit_if_slack(),
        "capacity >> slot count must trip the gate"
    );
    let after = storage.size_estimate();
    assert!(
        after < before,
        "a real shrink must reduce the footprint ({after} !< {before})"
    );

    // Immediately after a shrink, capacity == slot count, so the gate is a no-op.
    assert!(
        !storage.shrink_to_fit_if_slack(),
        "a freshly-shrunk (tight) store must not be shrunk again"
    );
    // The data is untouched by the (non-)shrink.
    assert_eq!(storage.len(), 4);
}

/// A dictionary grown by ordinary doubling — no over-reserve — and then made
/// exactly tight is left alone by the gate. This is the steady-state batch case:
/// capacity tracks the live data, so the gate does no work.
#[test]
fn shrink_gate_skips_a_tight_dictionary() {
    let mut storage = MemoryStorage::new();
    for i in 0..200 {
        assert!(storage.insert_quad(quad(i)));
    }
    // Make capacity == slot count with an unconditional shrink, then the gate
    // must report no work regardless of allocator rounding.
    storage.shrink_to_fit();
    assert!(
        !storage.shrink_to_fit_if_slack(),
        "an already-tight dictionary must be left untouched by the gate"
    );
    assert_eq!(storage.len(), 200);
}

// ---------------------------------------------------------------------------
// Batched ingest through the `Store` trait — correctness across many batches,
// with the per-batch reserve and gated shrink in the loop.
// ---------------------------------------------------------------------------

/// Five consecutive `bulk_insert_quads` batches through the trait: the counts
/// add up, every quad resolves back exactly, and a `shrink_to_fit` at the end
/// leaves the data intact.
#[test]
fn multi_batch_bulk_insert_is_correct() {
    let store = RdfStore::new().expect("create store");
    const BATCHES: usize = 5;
    const PER_BATCH: usize = 1000;

    for b in 0..BATCHES {
        let lo = b * PER_BATCH;
        let inserted =
            Store::bulk_insert_quads(&store, batch(lo, lo + PER_BATCH)).expect("bulk insert batch");
        assert_eq!(inserted, PER_BATCH, "every quad in a fresh batch is new");
        // A per-batch gated shrink, exactly as the Fuseki ingest seam issues it.
        Store::shrink_to_fit(&store).expect("gated shrink");
    }

    let total = BATCHES * PER_BATCH;
    assert_eq!(Store::len(&store).expect("len"), total);
    let all = Store::find_quads(&store, None, None, None, None).expect("find all");
    assert_eq!(all.len(), total, "all inserted quads are retrievable");

    // Resolve round-trip: a specific quad from the middle of the run comes back
    // with exactly the term it was interned with (no id-aliasing after shrinks).
    let probe_idx = total / 2 + 7;
    let probe = quad(probe_idx);
    let probe_subject: oxirs_core::model::Subject =
        NamedNode::new(format!("http://example.org/s{probe_idx}"))
            .expect("subject IRI")
            .into();
    let hits =
        Store::find_quads(&store, Some(&probe_subject), None, None, None).expect("find probe");
    assert_eq!(hits.len(), 1, "the probe subject occurs exactly once");
    assert_eq!(hits[0], probe, "resolved quad matches the interned one");

    // A final shrink must not disturb correctness.
    Store::shrink_to_fit(&store).expect("final shrink");
    assert_eq!(Store::len(&store).expect("len after shrink"), total);
    assert_eq!(
        Store::find_quads(&store, None, None, None, None)
            .expect("find all after shrink")
            .len(),
        total
    );

    // Re-inserting the whole set adds nothing new (dedup survives the shrinks).
    let redundant = Store::bulk_insert_quads(&store, batch(0, total)).expect("re-insert all");
    assert_eq!(redundant, 0, "duplicate batch inserts zero new quads");
    assert_eq!(Store::len(&store).expect("len unchanged"), total);
}

/// Insert/remove/re-insert churn: each cycle loads a fresh distinct batch and
/// clears the previous one. Correctness holds every cycle, and the interned
/// footprint stays *bounded* across many cycles — the observable proof that
/// reclaimed term ids are reused rather than leaked (a broken free list would
/// grow the dictionaries linearly in the cycle count).
#[test]
fn insert_remove_churn_reuses_ids_and_stays_bounded() {
    let store = RdfStore::new().expect("create store");
    const CYCLE: usize = 500;
    const CYCLES: usize = 12;

    let mut baseline: Option<usize> = None;
    for c in 0..CYCLES {
        let lo = c * CYCLE;
        // Load a fresh, all-distinct batch.
        let inserted =
            Store::bulk_insert_quads(&store, batch(lo, lo + CYCLE)).expect("insert cycle");
        assert_eq!(inserted, CYCLE);
        assert_eq!(Store::len(&store).expect("len after insert"), CYCLE);

        // The batch resolves back exactly.
        let all = Store::find_quads(&store, None, None, None, None).expect("find cycle");
        assert_eq!(all.len(), CYCLE);

        // Record the footprint after the first steady-state cycle, then assert
        // later cycles never balloon past a generous multiple of it.
        if let Some(size) = store.interned_size_estimate() {
            match baseline {
                None if c == 1 => baseline = Some(size),
                Some(base) => assert!(
                    size <= base * 3,
                    "cycle {c}: interned footprint {size} exceeded 3x the \
                     steady-state baseline {base} — freed term ids are leaking"
                ),
                _ => {}
            }
        }

        // Clear the batch so its term ids are reclaimed onto the free list for
        // the next cycle to reuse.
        for q in all {
            assert!(
                Store::remove_quad(&store, &q).expect("remove"),
                "quad was present"
            );
        }
        assert_eq!(Store::len(&store).expect("len after clear"), 0);
    }
}

/// Coarse ingest-throughput smoke check: 50k-quad batches, five of them, must
/// complete promptly through the trait path (with the reserve + gated shrink in
/// the loop). The bound is deliberately loose — this guards against a gross
/// O(T²/b) regression re-appearing, not against small timing variance, and runs
/// under an unoptimised test build.
#[test]
fn large_repeated_batches_do_not_regress_grossly() {
    use std::time::Instant;

    let store = RdfStore::new().expect("create store");
    const BATCHES: usize = 5;
    const PER_BATCH: usize = 50_000;

    let start = Instant::now();
    for b in 0..BATCHES {
        let lo = b * PER_BATCH;
        let inserted = Store::bulk_insert_quads(&store, batch(lo, lo + PER_BATCH))
            .expect("bulk insert large batch");
        assert_eq!(inserted, PER_BATCH);
        Store::shrink_to_fit(&store).expect("gated shrink");
    }
    let elapsed = start.elapsed();

    assert_eq!(
        Store::len(&store).expect("len"),
        BATCHES * PER_BATCH,
        "all 250k quads inserted"
    );
    // Very loose ceiling: the fixed path handles this in well under a second in
    // release and a few seconds unoptimised; a return of the shrink→regrow
    // treadmill would blow far past this.
    assert!(
        elapsed.as_secs() < 180,
        "50k x5 ingest took {elapsed:?} — suspiciously slow (possible O(T^2/b) regression)"
    );
}
