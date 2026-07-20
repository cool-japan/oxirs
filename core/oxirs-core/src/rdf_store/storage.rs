//! RDF storage backends and implementations.
//!
//! [`MemoryStorage`] uses **term interning** and keeps *no* owned-`Quad` copies
//! at all. Each column (subject, predicate, object, graph name) is interned into
//! a [`ColumnDictionary`] (term <-> `u32`), and every quad lives *only* as four
//! compact `[u32; 4]` id-tuples in the SPOG/POSG/OSPG/GSPO permutation indexes.
//! All reads — `len`/`contains`/pattern queries/full scans/N-Quads export —
//! derive their `Quad` values on the fly by resolving ids back through the
//! dictionaries, so a triple costs four id-tuples plus one interned copy of each
//! *distinct* term per column, rather than a fully materialized owned `Quad`.
//!
//! ## Iteration order
//!
//! [`query_quads`](MemoryStorage::query_quads) still returns results in
//! canonical `Quad` order (it re-sorts through a `BTreeSet<Quad>` at the API
//! boundary), so its observable order is unchanged. The *streaming* scans
//! ([`iter_quads`](MemoryStorage::iter_quads) and
//! [`for_each_quad`](MemoryStorage::for_each_quad)) instead visit quads in
//! *index (term-interning) order* — the order of the leading permutation — so
//! they never materialize the whole graph at once. That order is deterministic
//! but differs from `Quad` order; RDF is unordered and the persistence /
//! serialization consumers of these scans do not depend on a particular order.
//!
//! ## Deletion and the dictionaries
//!
//! Each column dictionary reference counts its ids. A successful `insert_quad`
//! takes one reference on each of the four column ids (only *after* the SPOG
//! novelty gate, so a duplicate insert never double-counts); `remove_quad`
//! drops one reference on each. When a term's last referencing quad is removed
//! its count reaches zero and the id is reclaimed **synchronously** — the value
//! is tombstoned, its lookup entry dropped, and the id pushed onto a free list
//! for reuse by the next new term in that column. Term ids therefore never leak:
//! the dictionary's live footprint tracks the set of terms actually in use, and
//! repeated insert/remove/reinsert churn does not grow it. Round-trips stay
//! correct: a re-inserted term reuses its still-live id, and a term whose id was
//! reclaimed is re-interned into the freed slot on the next insert.

use super::dictionary::ColumnDictionary;
use super::persistence::PersistentState;
use crate::indexing::UltraIndex;
use crate::model::*;
use crate::optimization::RdfArena;
use std::collections::BTreeSet;
use std::sync::{Arc, RwLock};

/// Storage backend for RDF quads
///
/// `Clone` shares the underlying `Arc`-wrapped state (index, arena, persistence
/// handle), so a cloned backend observes the same live store — used by
/// [`PreparedQuery`](super::PreparedQuery) to retain a handle for execution.
#[derive(Debug, Clone)]
pub enum StorageBackend {
    /// Ultra-high performance in-memory storage
    UltraMemory(Arc<UltraIndex>, Arc<RdfArena>),
    /// Legacy in-memory storage using collections
    Memory(Arc<RwLock<MemoryStorage>>),
    /// Durable append-based disk storage. The in-memory `MemoryStorage` is the
    /// query index; [`PersistentState`] owns the append log, dirty flag, and
    /// compaction machinery. The tuple arity is preserved (the second field is
    /// the persistence handle, which also carries the dataset path).
    Persistent(Arc<RwLock<MemoryStorage>>, Arc<PersistentState>),
}

/// In-memory storage implementation backed *entirely* by interned `[u32; 4]`
/// id-tuples — no owned `Quad` copies are retained.
#[derive(Debug, Clone, Default)]
pub struct MemoryStorage {
    /// Per-column term dictionaries (term <-> u32).
    subjects: ColumnDictionary<Subject>,
    predicates: ColumnDictionary<Predicate>,
    objects: ColumnDictionary<Object>,
    graphs: ColumnDictionary<GraphName>,
    /// SPOG permutation: `[s, p, o, g]` — subject / full-scan lookups.
    /// This index is the single source of truth for `len`, membership, and
    /// full iteration.
    spog: BTreeSet<[u32; 4]>,
    /// POSG permutation: `[p, o, s, g]` — predicate lookups.
    posg: BTreeSet<[u32; 4]>,
    /// OSPG permutation: `[o, s, p, g]` — object lookups.
    ospg: BTreeSet<[u32; 4]>,
    /// GSPO permutation: `[g, s, p, o]` — graph lookups.
    gspo: BTreeSet<[u32; 4]>,
    /// Named graphs in the dataset
    pub named_graphs: BTreeSet<NamedNode>,
}

/// `true` when a bound id constraint is absent (`None`) or equals `actual`.
///
/// Spelled as an explicit match rather than `Option::is_none_or` so the crate
/// stays warning-clean under its conservative declared MSRV.
#[inline]
fn id_matches(bound: Option<u32>, actual: u32) -> bool {
    match bound {
        Some(want) => want == actual,
        None => true,
    }
}

/// Pass a dictionary resolution through unchanged, asserting in debug builds
/// that it is `Some`.
///
/// Every id stored in a permutation index is reference counted, so the id is
/// reclaimed only after its last referencing tuple is removed; a `None` here
/// therefore signals an index/dictionary desync (a bug). We assert in debug
/// builds and, in release, fall through the caller's `?` so a corrupt tuple is
/// skipped rather than panicking.
#[inline]
fn resolve_live<T>(resolved: Option<&T>) -> Option<&T> {
    debug_assert!(
        resolved.is_some(),
        "permutation index references an id with no live term (dictionary/index desync)"
    );
    resolved
}

impl MemoryStorage {
    pub fn new() -> Self {
        MemoryStorage {
            subjects: ColumnDictionary::new(),
            predicates: ColumnDictionary::new(),
            objects: ColumnDictionary::new(),
            graphs: ColumnDictionary::new(),
            spog: BTreeSet::new(),
            posg: BTreeSet::new(),
            ospg: BTreeSet::new(),
            gspo: BTreeSet::new(),
            named_graphs: BTreeSet::new(),
        }
    }

    pub fn insert_quad(&mut self, quad: Quad) -> bool {
        // Intern every column first (a no-op that returns the existing id when a
        // term is already present). Novelty is then decided solely by the SPOG
        // index: `BTreeSet::insert` reports whether the id-tuple was new.
        let s = self.subjects.intern(quad.subject());
        let p = self.predicates.intern(quad.predicate());
        let o = self.objects.intern(quad.object());
        let g = self.graphs.intern(quad.graph_name());

        if !self.spog.insert([s, p, o, g]) {
            // Duplicate quad: the id-tuple already exists in every permutation.
            // No reference is taken, so a repeated insert never double-counts.
            return false;
        }

        // Genuinely new quad: take one reference on each column's id so it stays
        // interned for exactly as long as some quad uses it.
        self.subjects.retain(s);
        self.predicates.retain(p);
        self.objects.retain(o);
        self.graphs.retain(g);

        self.posg.insert([p, o, s, g]);
        self.ospg.insert([o, s, p, g]);
        self.gspo.insert([g, s, p, o]);

        if let GraphName::NamedNode(graph_name) = quad.graph_name() {
            self.named_graphs.insert(graph_name.clone());
        }

        true
    }

    pub fn remove_quad(&mut self, quad: &Quad) -> bool {
        // If any component was never interned the quad cannot be present.
        let (s, p, o, g) = match (
            self.subjects.get_id(quad.subject()),
            self.predicates.get_id(quad.predicate()),
            self.objects.get_id(quad.object()),
            self.graphs.get_id(quad.graph_name()),
        ) {
            (Some(s), Some(p), Some(o), Some(g)) => (s, p, o, g),
            _ => return false,
        };

        // The SPOG index is authoritative for membership; if the tuple was not
        // present the quad is not in the store.
        if !self.spog.remove(&[s, p, o, g]) {
            return false;
        }
        self.posg.remove(&[p, o, s, g]);
        self.ospg.remove(&[o, s, p, g]);
        self.gspo.remove(&[g, s, p, o]);

        // Drop one reference on each column's id; the id (and its interned term)
        // is reclaimed synchronously when its last referencing quad is gone.
        self.subjects.release(s);
        self.predicates.release(p);
        self.objects.release(o);
        self.graphs.release(g);

        // Drop the named graph from the set once its last quad is gone.
        if let GraphName::NamedNode(graph_name) = quad.graph_name() {
            let still_present = self
                .gspo
                .range([g, 0, 0, 0]..=[g, u32::MAX, u32::MAX, u32::MAX])
                .next()
                .is_some();
            if !still_present {
                self.named_graphs.remove(graph_name);
            }
        }

        true
    }

    pub fn contains_quad(&self, quad: &Quad) -> bool {
        match (
            self.subjects.get_id(quad.subject()),
            self.predicates.get_id(quad.predicate()),
            self.objects.get_id(quad.object()),
            self.graphs.get_id(quad.graph_name()),
        ) {
            (Some(s), Some(p), Some(o), Some(g)) => self.spog.contains(&[s, p, o, g]),
            _ => false,
        }
    }

    /// Resolve a canonical `(s, p, o, g)` id-tuple into an owned `Quad`.
    fn materialize(&self, s: u32, p: u32, o: u32, g: u32) -> Option<Quad> {
        Some(Quad::new(
            resolve_live(self.subjects.resolve(s))?.clone(),
            resolve_live(self.predicates.resolve(p))?.clone(),
            resolve_live(self.objects.resolve(o))?.clone(),
            resolve_live(self.graphs.resolve(g))?.clone(),
        ))
    }

    /// Scan the best permutation for the given bound-id constraints, yielding
    /// canonical `[s, p, o, g]` id-tuples (with the non-lead constraints applied
    /// as cheap integer filters).
    ///
    /// The leading permutation is chosen by the most selective bound column
    /// (subject, then object, then predicate, then graph); when nothing is bound
    /// the whole SPOG index is scanned. Yielded tuples are in the leading
    /// permutation's order, i.e. term-interning order, *not* `Quad` order.
    fn scan_ids<'a>(
        &'a self,
        sid: Option<u32>,
        pid: Option<u32>,
        oid: Option<u32>,
        gid: Option<u32>,
    ) -> Box<dyn Iterator<Item = [u32; 4]> + 'a> {
        const LO: [u32; 4] = [0, 0, 0, 0];
        let hi = |k: u32| [k, u32::MAX, u32::MAX, u32::MAX];

        let base: Box<dyn Iterator<Item = [u32; 4]> + 'a> = if let Some(s) = sid {
            Box::new(
                self.spog
                    .range([s, LO[1], LO[2], LO[3]]..=hi(s))
                    .map(|t| [t[0], t[1], t[2], t[3]]),
            )
        } else if let Some(o) = oid {
            Box::new(
                self.ospg
                    .range([o, LO[1], LO[2], LO[3]]..=hi(o))
                    .map(|t| [t[1], t[2], t[0], t[3]]),
            )
        } else if let Some(p) = pid {
            Box::new(
                self.posg
                    .range([p, LO[1], LO[2], LO[3]]..=hi(p))
                    .map(|t| [t[2], t[0], t[1], t[3]]),
            )
        } else if let Some(g) = gid {
            Box::new(
                self.gspo
                    .range([g, LO[1], LO[2], LO[3]]..=hi(g))
                    .map(|t| [t[1], t[2], t[3], t[0]]),
            )
        } else {
            Box::new(self.spog.iter().copied())
        };

        Box::new(base.filter(move |q| {
            id_matches(sid, q[0])
                && id_matches(pid, q[1])
                && id_matches(oid, q[2])
                && id_matches(gid, q[3])
        }))
    }

    /// Resolve the four optional bound terms to their ids. Returns `None` if any
    /// bound term is not interned (so the pattern cannot match any quad).
    #[allow(clippy::type_complexity)]
    fn resolve_pattern_ids(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
        graph_name: Option<&GraphName>,
    ) -> Option<(Option<u32>, Option<u32>, Option<u32>, Option<u32>)> {
        let sid = match subject {
            Some(s) => Some(self.subjects.get_id(s)?),
            None => None,
        };
        let pid = match predicate {
            Some(p) => Some(self.predicates.get_id(p)?),
            None => None,
        };
        let oid = match object {
            Some(o) => Some(self.objects.get_id(o)?),
            None => None,
        };
        let gid = match graph_name {
            Some(g) => Some(self.graphs.get_id(g)?),
            None => None,
        };
        Some((sid, pid, oid, gid))
    }

    pub fn query_quads(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
        graph_name: Option<&GraphName>,
    ) -> Vec<Quad> {
        // A bound term that is not interned means no quad can match.
        let (sid, pid, oid, gid) =
            match self.resolve_pattern_ids(subject, predicate, object, graph_name) {
                Some(ids) => ids,
                None => return Vec::new(),
            };

        // Collect into a BTreeSet<Quad> so results stay Quad-ordered and
        // deterministic regardless of which permutation led the scan (this keeps
        // `query_quads`' observable order identical to the previous design).
        let mut results: BTreeSet<Quad> = BTreeSet::new();
        for [s, p, o, g] in self.scan_ids(sid, pid, oid, gid) {
            if let Some(quad) = self.materialize(s, p, o, g) {
                results.insert(quad);
            }
        }
        results.into_iter().collect()
    }

    /// Streaming decode iterator over every stored quad, materialized on the fly
    /// from the SPOG index. Visits quads in index (term-interning) order and
    /// keeps only one `Quad` alive at a time, so a whole graph can be exported
    /// without ever materializing it in full. Backs the persistence compaction
    /// rewrite and any full-graph N-Quads export.
    pub fn iter_quads(&self) -> impl Iterator<Item = Quad> + '_ {
        self.spog
            .iter()
            .filter_map(move |t| self.materialize(t[0], t[1], t[2], t[3]))
    }

    /// Stream every quad matching the pattern to `f` without building a result
    /// `Vec`.
    ///
    /// Selects the best permutation for the bound terms, decodes each matching
    /// id-tuple, and hands the owned `Quad` to `f`, so only one quad is alive at
    /// a time. Quads are visited in the leading permutation's order (term
    /// interning order), which is deterministic but differs from `Quad` order;
    /// see the module-level "Iteration order" note. This backs
    /// [`Store::for_each_quad`](super::Store::for_each_quad).
    pub fn for_each_quad(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
        graph_name: Option<&GraphName>,
        f: &mut dyn FnMut(Quad),
    ) {
        // A bound term that is not interned means nothing matches.
        let (sid, pid, oid, gid) =
            match self.resolve_pattern_ids(subject, predicate, object, graph_name) {
                Some(ids) => ids,
                None => return,
            };
        for [s, p, o, g] in self.scan_ids(sid, pid, oid, gid) {
            if let Some(quad) = self.materialize(s, p, o, g) {
                f(quad);
            }
        }
    }

    /// Pre-size the column dictionaries for a bulk load of roughly `approx_quads`
    /// quads, so the dominant allocations are made once up front instead of
    /// through repeated doubling (each doubling transiently holds both the old and
    /// new backing buffer, inflating peak memory). The object column is by far the
    /// highest-cardinality in typical RDF (close to one distinct literal/IRI per
    /// quad), so it is sized to the full estimate; subjects are lower cardinality
    /// and reserved conservatively; predicates and graphs grow cheaply and are
    /// left to size themselves. Any over-reservation is returned by a later
    /// [`shrink_to_fit`](Self::shrink_to_fit).
    pub fn reserve_for_bulk_load(&mut self, approx_quads: usize) {
        self.objects.reserve(approx_quads);
        self.subjects.reserve(approx_quads / 4);
    }

    /// Release excess reserved capacity in every column dictionary back to the
    /// allocator after a bulk load completes. The permutation indexes are
    /// `BTreeSet`s, which allocate per node and have no slack to trim, so only the
    /// dictionaries (backed by `Vec`/`HashMap` that grow by doubling) are shrunk.
    pub fn shrink_to_fit(&mut self) {
        self.subjects.shrink_to_fit();
        self.predicates.shrink_to_fit();
        self.objects.shrink_to_fit();
        self.graphs.shrink_to_fit();
    }

    /// Gated shrink for the *repeated bulk-ingest* path: release excess dictionary
    /// capacity **only when the dictionaries are genuinely over-provisioned**, i.e.
    /// their combined backing allocation can hold more than twice the slots they
    /// are actually using. Returns `true` when a shrink was performed, `false` when
    /// the store was already tight enough to skip it (a no-op).
    ///
    /// ## Why a gate
    ///
    /// The dictionaries grow by doubling, so a bulk load leaves each column with up
    /// to ~2x slack. A caller that shrinks *unconditionally after every batch* (as
    /// the Fuseki ingest seam did) forces a full-dictionary reallocation and rehash
    /// on each batch, and the very next batch's inserts double the allocation
    /// straight back — an O(T²/b) shrink→regrow→shrink treadmill across `T` total
    /// quads in batches of `b`. Firing only when `capacity > 2 * slot_len` breaks
    /// that treadmill: a steady stream of similarly-sized batches keeps `capacity`
    /// tracking the live data (never crossing the threshold), so no work is done,
    /// while a genuine over-reservation (or post-deletion slack) is still reclaimed.
    ///
    /// ## Why the comparison is *aggregated* across all four columns
    ///
    /// The two low-cardinality columns (predicates, graphs) hold only a handful of
    /// slots, but a `Vec`'s minimum non-zero allocation is 4 slots — so a
    /// single-graph, single-predicate load has `capacity (4) > 2 * slot_len (1..2)`
    /// for those columns *permanently*. A per-column OR gate would therefore trip on
    /// every batch off the back of a 3-slot graph column and re-shrink (and
    /// re-`malloc_trim`) the whole store regardless — reinstating the exact treadmill
    /// this exists to remove. Summing capacity and slots across the four columns
    /// drowns that fixed handful of slack in the dominant object/subject columns, so
    /// the gate only fires when a *large* column is really over-provisioned.
    ///
    /// ## Why `slot_len`, not the live-term count, is the denominator
    ///
    /// [`shrink_to_fit`] shrinks each id vector down to its slot length (live plus
    /// tombstoned slots — tombstones stay for free-list reuse), not down to the
    /// live-term count. Gating on `slot_len` means that right after a shrink
    /// `capacity == slot_len`, so the gate reports "no slack" and does **not** keep
    /// re-shrinking (and re-`malloc_trim`-ing) a tombstone-heavy dictionary batch
    /// after batch — which would reintroduce the very per-batch cost the gate exists
    /// to remove.
    pub fn shrink_to_fit_if_slack(&mut self) -> bool {
        let capacity = self.subjects.capacity_estimate()
            + self.predicates.capacity_estimate()
            + self.objects.capacity_estimate()
            + self.graphs.capacity_estimate();
        let slots = self.subjects.slot_len()
            + self.predicates.slot_len()
            + self.objects.slot_len()
            + self.graphs.slot_len();
        let over_provisioned = capacity > slots.saturating_mul(2);
        if over_provisioned {
            self.shrink_to_fit();
        }
        over_provisioned
    }

    // --- Frozen-snapshot support --------------------------------------------
    //
    // These `pub(crate)` seams let the sibling `snapshot` module read the interned
    // columns and the SPOG permutation to serialize a store, and rebuild a store
    // from deserialized parts, without exposing the private fields or making the
    // dictionaries mutable to the outside.

    /// Borrow the subject column dictionary (snapshot builder).
    pub(crate) fn subjects_dict(&self) -> &ColumnDictionary<Subject> {
        &self.subjects
    }

    /// Borrow the predicate column dictionary (snapshot builder).
    pub(crate) fn predicates_dict(&self) -> &ColumnDictionary<Predicate> {
        &self.predicates
    }

    /// Borrow the object column dictionary (snapshot builder).
    pub(crate) fn objects_dict(&self) -> &ColumnDictionary<Object> {
        &self.objects
    }

    /// Borrow the graph-name column dictionary (snapshot builder).
    pub(crate) fn graphs_dict(&self) -> &ColumnDictionary<GraphName> {
        &self.graphs
    }

    /// Borrow the SPOG permutation. The other three permutations are derivable
    /// from it, so the snapshot builder reads only this one and regenerates the
    /// rest in the target id ordering.
    pub(crate) fn spog_index(&self) -> &BTreeSet<[u32; 4]> {
        &self.spog
    }

    /// Reassemble a storage from snapshot-deserialized parts: the four column
    /// dictionaries (already built with sorted-term ids) and the four permutation
    /// indexes (already sorted in their own permutation order). `named_graphs` is
    /// re-derived from the graph dictionary — every live graph term is, by
    /// construction, in use by some quad, so each `GraphName::NamedNode` entry is
    /// exactly a named graph of the dataset. The caller is responsible for the
    /// ids in the permutations agreeing with the dictionaries.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_snapshot_parts(
        subjects: ColumnDictionary<Subject>,
        predicates: ColumnDictionary<Predicate>,
        objects: ColumnDictionary<Object>,
        graphs: ColumnDictionary<GraphName>,
        spog: BTreeSet<[u32; 4]>,
        posg: BTreeSet<[u32; 4]>,
        ospg: BTreeSet<[u32; 4]>,
        gspo: BTreeSet<[u32; 4]>,
    ) -> Self {
        let mut named_graphs = BTreeSet::new();
        for (_id, graph) in graphs.iter_live_slots() {
            if let GraphName::NamedNode(node) = graph {
                named_graphs.insert(node.clone());
            }
        }
        MemoryStorage {
            subjects,
            predicates,
            objects,
            graphs,
            spog,
            posg,
            ospg,
            gspo,
            named_graphs,
        }
    }

    pub fn len(&self) -> usize {
        self.spog.len()
    }

    pub fn is_empty(&self) -> bool {
        self.spog.is_empty()
    }

    /// Best-effort estimate of the resident heap footprint of the interned
    /// structures, in bytes. Sums the four permutation indexes (compact
    /// `[u32; 4]` tuples) plus every column dictionary (structural capacity plus
    /// the interned term string bytes) plus the named-graph set. Intended for
    /// coarse before/after comparisons, not exact accounting; it deliberately
    /// omits per-node `BTreeSet` bookkeeping overhead.
    pub fn size_estimate(&self) -> usize {
        use std::mem::size_of;
        let tuple = size_of::<[u32; 4]>();
        let perm_bytes =
            (self.spog.len() + self.posg.len() + self.ospg.len() + self.gspo.len()) * tuple;
        let dict_bytes = self.subjects.size_estimate()
            + self.predicates.size_estimate()
            + self.objects.size_estimate()
            + self.graphs.size_estimate();
        let named_graph_bytes = self.named_graphs.len() * size_of::<NamedNode>();
        perm_bytes + dict_bytes + named_graph_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn nn(iri: &str) -> NamedNode {
        NamedNode::new(iri).expect("valid IRI")
    }

    /// Naive reference implementation of pattern matching over a plain
    /// `BTreeSet<Quad>`, used to validate the interned permutation indexes.
    fn naive_query(
        reference: &BTreeSet<Quad>,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
        graph_name: Option<&GraphName>,
    ) -> BTreeSet<Quad> {
        reference
            .iter()
            .filter(|q| match subject {
                Some(s) => q.subject() == s,
                None => true,
            })
            .filter(|q| match predicate {
                Some(p) => q.predicate() == p,
                None => true,
            })
            .filter(|q| match object {
                Some(o) => q.object() == o,
                None => true,
            })
            .filter(|q| match graph_name {
                Some(g) => q.graph_name() == g,
                None => true,
            })
            .cloned()
            .collect()
    }

    fn build_dataset() -> (MemoryStorage, BTreeSet<Quad>) {
        let subjects = [
            Subject::NamedNode(nn("http://ex.org/s1")),
            Subject::NamedNode(nn("http://ex.org/s2")),
            Subject::BlankNode(BlankNode::new("b1").expect("valid blank node")),
        ];
        let predicates = [
            Predicate::NamedNode(nn("http://ex.org/p1")),
            Predicate::NamedNode(nn("http://ex.org/p2")),
        ];
        let objects = [
            Object::NamedNode(nn("http://ex.org/o1")),
            Object::Literal(Literal::new("hello")),
            Object::Literal(Literal::new_lang("hello", "en").expect("valid lang literal")),
            Object::BlankNode(BlankNode::new("b2").expect("valid blank node")),
        ];
        let graphs = [
            GraphName::DefaultGraph,
            GraphName::NamedNode(nn("http://ex.org/g1")),
        ];

        let mut storage = MemoryStorage::new();
        let mut reference = BTreeSet::new();

        // Insert a deterministic, diverse subset of the cross product.
        let mut counter = 0usize;
        for s in &subjects {
            for p in &predicates {
                for o in &objects {
                    for g in &graphs {
                        // Keep the dataset diverse but not the full product.
                        // Use a period (7) coprime with the inner loop sizes so
                        // no single column value is systematically excluded.
                        counter += 1;
                        if counter % 7 == 0 {
                            continue;
                        }
                        let quad = Quad::new(s.clone(), p.clone(), o.clone(), g.clone());
                        assert_eq!(
                            storage.insert_quad(quad.clone()),
                            reference.insert(quad),
                            "insert novelty must agree with the reference set"
                        );
                    }
                }
            }
        }
        (storage, reference)
    }

    #[test]
    fn test_interning_matches_naive_all_16_binding_combinations() {
        let (storage, reference) = build_dataset();

        // Candidate values per column include present terms AND one absent term
        // (to exercise the empty-result path). Prepending `None` makes each
        // column optionally unbound, so the four nested loops cover all 16
        // subject/predicate/object/graph binding combinations.
        let subj_opts: Vec<Option<Subject>> = std::iter::once(None)
            .chain(
                [
                    Subject::NamedNode(nn("http://ex.org/s1")),
                    Subject::NamedNode(nn("http://ex.org/s2")),
                    Subject::BlankNode(BlankNode::new("b1").expect("valid blank node")),
                    Subject::NamedNode(nn("http://ex.org/absent")),
                ]
                .into_iter()
                .map(Some),
            )
            .collect();
        let pred_opts: Vec<Option<Predicate>> = std::iter::once(None)
            .chain(
                [
                    Predicate::NamedNode(nn("http://ex.org/p1")),
                    Predicate::NamedNode(nn("http://ex.org/p2")),
                    Predicate::NamedNode(nn("http://ex.org/absent")),
                ]
                .into_iter()
                .map(Some),
            )
            .collect();
        let obj_opts: Vec<Option<Object>> = std::iter::once(None)
            .chain(
                [
                    Object::NamedNode(nn("http://ex.org/o1")),
                    Object::Literal(Literal::new("hello")),
                    Object::Literal(Literal::new_lang("hello", "en").expect("valid lang literal")),
                    Object::BlankNode(BlankNode::new("b2").expect("valid blank node")),
                    Object::Literal(Literal::new("absent")),
                ]
                .into_iter()
                .map(Some),
            )
            .collect();
        let graph_opts: Vec<Option<GraphName>> = std::iter::once(None)
            .chain(
                [
                    GraphName::DefaultGraph,
                    GraphName::NamedNode(nn("http://ex.org/g1")),
                    GraphName::NamedNode(nn("http://ex.org/absent")),
                ]
                .into_iter()
                .map(Some),
            )
            .collect();

        for so in &subj_opts {
            for po in &pred_opts {
                for oo in &obj_opts {
                    for go in &graph_opts {
                        let got: BTreeSet<Quad> = storage
                            .query_quads(so.as_ref(), po.as_ref(), oo.as_ref(), go.as_ref())
                            .into_iter()
                            .collect();
                        let want = naive_query(
                            &reference,
                            so.as_ref(),
                            po.as_ref(),
                            oo.as_ref(),
                            go.as_ref(),
                        );
                        assert_eq!(
                            got, want,
                            "mismatch for pattern s={so:?} p={po:?} o={oo:?} g={go:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_interning_remove_updates_all_indexes_and_named_graphs() {
        let (mut storage, mut reference) = build_dataset();

        // Remove every quad in a named graph and confirm the named graph is
        // dropped from the set exactly when its last quad is removed.
        let g1 = GraphName::NamedNode(nn("http://ex.org/g1"));
        let g1_quads: Vec<Quad> = reference
            .iter()
            .filter(|q| q.graph_name() == &g1)
            .cloned()
            .collect();
        assert!(!g1_quads.is_empty(), "dataset must contain g1 quads");

        for (i, quad) in g1_quads.iter().enumerate() {
            assert!(storage.remove_quad(quad));
            reference.remove(quad);
            let last = i + 1 == g1_quads.len();
            assert_eq!(
                storage.named_graphs.contains(&nn("http://ex.org/g1")),
                !last,
                "named graph g1 must persist until its last quad is removed"
            );
        }

        // Every remaining pattern query still matches the naive reference.
        let all: BTreeSet<Quad> = storage
            .query_quads(None, None, None, None)
            .into_iter()
            .collect();
        assert_eq!(all, reference);
        assert_eq!(storage.len(), reference.len());
    }

    #[test]
    fn test_interning_duplicate_insert_is_noop() {
        let mut storage = MemoryStorage::new();
        let quad = Quad::new(
            nn("http://ex.org/s"),
            nn("http://ex.org/p"),
            nn("http://ex.org/o"),
            GraphName::DefaultGraph,
        );
        assert!(storage.insert_quad(quad.clone()));
        assert!(!storage.insert_quad(quad.clone()));
        assert_eq!(storage.len(), 1);
        assert!(storage.contains_quad(&quad));
    }

    /// Every binding shape (subject-only, subject+predicate, full triple,
    /// graph-scoped, and the fully-unbound full scan) routes through the
    /// intended permutation and returns exactly the naive reference set. This is
    /// the observable proxy for "the right permutation was selected": the SPOG,
    /// POSG, OSPG, and GSPO decode arms must all reconstruct canonical quads.
    #[test]
    fn test_pattern_queries_select_correct_permutation() {
        let (storage, reference) = build_dataset();

        let s = Subject::NamedNode(nn("http://ex.org/s1"));
        let p = Predicate::NamedNode(nn("http://ex.org/p1"));
        let o = Object::NamedNode(nn("http://ex.org/o1"));
        let g = GraphName::NamedNode(nn("http://ex.org/g1"));

        // subject-only -> SPOG lead
        let got: BTreeSet<Quad> = storage
            .query_quads(Some(&s), None, None, None)
            .into_iter()
            .collect();
        assert_eq!(got, naive_query(&reference, Some(&s), None, None, None));

        // subject + predicate -> SPOG lead, predicate id-filter
        let got: BTreeSet<Quad> = storage
            .query_quads(Some(&s), Some(&p), None, None)
            .into_iter()
            .collect();
        assert_eq!(got, naive_query(&reference, Some(&s), Some(&p), None, None));

        // full triple s+p+o
        let got: BTreeSet<Quad> = storage
            .query_quads(Some(&s), Some(&p), Some(&o), None)
            .into_iter()
            .collect();
        assert_eq!(
            got,
            naive_query(&reference, Some(&s), Some(&p), Some(&o), None)
        );

        // predicate-only -> POSG lead
        let got: BTreeSet<Quad> = storage
            .query_quads(None, Some(&p), None, None)
            .into_iter()
            .collect();
        assert_eq!(got, naive_query(&reference, None, Some(&p), None, None));

        // object-only -> OSPG lead
        let got: BTreeSet<Quad> = storage
            .query_quads(None, None, Some(&o), None)
            .into_iter()
            .collect();
        assert_eq!(got, naive_query(&reference, None, None, Some(&o), None));

        // graph-scoped -> GSPO lead
        let got: BTreeSet<Quad> = storage
            .query_quads(None, None, None, Some(&g))
            .into_iter()
            .collect();
        assert_eq!(got, naive_query(&reference, None, None, None, Some(&g)));

        // full scan (fully unbound) -> SPOG iteration
        let got: BTreeSet<Quad> = storage
            .query_quads(None, None, None, None)
            .into_iter()
            .collect();
        assert_eq!(got, reference);
    }

    /// insert -> remove -> reinsert round-trips cleanly with no owned quad set:
    /// membership, length, and pattern reads all reflect the current state, and
    /// a re-inserted quad is reported novel again after removal.
    #[test]
    fn test_insert_remove_reinsert_round_trip() {
        let mut storage = MemoryStorage::new();
        let quad = Quad::new(
            Subject::NamedNode(nn("http://ex.org/s")),
            Predicate::NamedNode(nn("http://ex.org/p")),
            Object::Literal(Literal::new("v")),
            GraphName::NamedNode(nn("http://ex.org/g")),
        );

        assert!(storage.insert_quad(quad.clone()));
        assert!(storage.contains_quad(&quad));
        assert_eq!(storage.len(), 1);
        assert!(storage.named_graphs.contains(&nn("http://ex.org/g")));

        assert!(storage.remove_quad(&quad));
        assert!(!storage.contains_quad(&quad));
        assert_eq!(storage.len(), 0);
        assert!(storage.is_empty());
        assert!(!storage.named_graphs.contains(&nn("http://ex.org/g")));
        assert!(storage.query_quads(None, None, None, None).is_empty());
        // Removing an absent quad is a no-op.
        assert!(!storage.remove_quad(&quad));

        // Reinsert: reported novel again, fully queryable, named graph restored.
        assert!(storage.insert_quad(quad.clone()));
        assert!(storage.contains_quad(&quad));
        assert_eq!(storage.len(), 1);
        assert!(storage.named_graphs.contains(&nn("http://ex.org/g")));
        let got: Vec<Quad> = storage.query_quads(
            Some(&Subject::NamedNode(nn("http://ex.org/s"))),
            None,
            None,
            None,
        );
        assert_eq!(got, vec![quad]);
    }

    /// `named_graphs` tracks the distinct named graphs incrementally and is
    /// maintained on removal (dropping a graph exactly when its last quad goes),
    /// independent of the total quad count.
    #[test]
    fn test_named_graphs_tracked_incrementally() {
        let mut storage = MemoryStorage::new();
        let g1 = nn("http://ex.org/g1");
        let g2 = nn("http://ex.org/g2");

        // Two quads in g1, one in g2, one in the default graph.
        let q_g1a = Quad::new(
            Subject::NamedNode(nn("http://ex.org/s1")),
            Predicate::NamedNode(nn("http://ex.org/p")),
            Object::Literal(Literal::new("a")),
            GraphName::NamedNode(g1.clone()),
        );
        let q_g1b = Quad::new(
            Subject::NamedNode(nn("http://ex.org/s2")),
            Predicate::NamedNode(nn("http://ex.org/p")),
            Object::Literal(Literal::new("b")),
            GraphName::NamedNode(g1.clone()),
        );
        let q_g2 = Quad::new(
            Subject::NamedNode(nn("http://ex.org/s3")),
            Predicate::NamedNode(nn("http://ex.org/p")),
            Object::Literal(Literal::new("c")),
            GraphName::NamedNode(g2.clone()),
        );
        let q_default = Quad::new(
            Subject::NamedNode(nn("http://ex.org/s4")),
            Predicate::NamedNode(nn("http://ex.org/p")),
            Object::Literal(Literal::new("d")),
            GraphName::DefaultGraph,
        );

        for q in [&q_g1a, &q_g1b, &q_g2, &q_default] {
            assert!(storage.insert_quad(q.clone()));
        }
        // The default graph never contributes a named graph.
        assert_eq!(
            storage.named_graphs,
            [g1.clone(), g2.clone()].into_iter().collect()
        );

        // Removing one of g1's two quads keeps g1 present.
        assert!(storage.remove_quad(&q_g1a));
        assert!(storage.named_graphs.contains(&g1));
        // Removing g1's last quad drops g1 but leaves g2.
        assert!(storage.remove_quad(&q_g1b));
        assert!(!storage.named_graphs.contains(&g1));
        assert!(storage.named_graphs.contains(&g2));
        assert_eq!(storage.named_graphs, [g2].into_iter().collect());
    }

    /// Repeated insert/remove/reinsert of the same quad must reclaim and reuse
    /// its column ids rather than allocate new slots, so neither the per-column
    /// slot counts nor the overall size estimate grow without bound.
    #[test]
    fn test_id_reclamation_bounded_growth() {
        let mut storage = MemoryStorage::new();
        let quad = Quad::new(
            Subject::NamedNode(nn("http://ex.org/s")),
            Predicate::NamedNode(nn("http://ex.org/p")),
            Object::Literal(Literal::new("v")),
            GraphName::NamedNode(nn("http://ex.org/g")),
        );

        // First insert establishes exactly one slot per column.
        assert!(storage.insert_quad(quad.clone()));
        for dict_slots in [
            storage.subjects.slot_count(),
            storage.predicates.slot_count(),
            storage.objects.slot_count(),
            storage.graphs.slot_count(),
        ] {
            assert_eq!(dict_slots, 1);
        }

        // One warmup cycle so the free-list vectors reach their steady-state
        // capacity before the baseline snapshot (the first remove is what first
        // grows each column's free-id stack).
        assert!(storage.remove_quad(&quad));
        assert!(storage.insert_quad(quad.clone()));
        let baseline = storage.size_estimate();

        // Many churn cycles: each remove reclaims the ids (live -> 0, one free
        // slot) and each reinsert must reuse the freed slot.
        for _ in 0..1_000 {
            assert!(storage.remove_quad(&quad));
            assert_eq!(storage.subjects.live_count(), 0);
            assert_eq!(storage.subjects.free_count(), 1);
            assert_eq!(storage.graphs.live_count(), 0);
            assert!(storage.insert_quad(quad.clone()));
        }

        // No growth: still one slot per column and an unchanged size estimate.
        assert_eq!(storage.subjects.slot_count(), 1);
        assert_eq!(storage.predicates.slot_count(), 1);
        assert_eq!(storage.objects.slot_count(), 1);
        assert_eq!(storage.graphs.slot_count(), 1);
        assert_eq!(storage.size_estimate(), baseline);
        assert_eq!(storage.len(), 1);
        assert!(storage.contains_quad(&quad));
    }

    /// Interleaving inserts and removes so ids get reclaimed and reused must
    /// leave the interned store exactly equal to a naive `BTreeSet<Quad>` model.
    #[test]
    fn test_id_reuse_round_trip_matches_naive_model() {
        let mk = |i: usize| {
            Quad::new(
                Subject::NamedNode(nn(&format!("http://ex.org/s{i}"))),
                Predicate::NamedNode(nn("http://ex.org/p")),
                Object::Literal(Literal::new(format!("v{i}"))),
                GraphName::DefaultGraph,
            )
        };

        let mut storage = MemoryStorage::new();
        let mut reference: BTreeSet<Quad> = BTreeSet::new();

        // Insert 0..20.
        for i in 0..20 {
            let q = mk(i);
            assert_eq!(storage.insert_quad(q.clone()), reference.insert(q));
        }
        // Remove the even-indexed quads, freeing their subject/object ids.
        for i in (0..20).step_by(2) {
            let q = mk(i);
            assert_eq!(storage.remove_quad(&q), reference.remove(&q));
        }
        // Insert a fresh batch 20..30 — these must reuse the freed id slots.
        let slots_before = storage.subjects.slot_count();
        for i in 20..30 {
            let q = mk(i);
            assert_eq!(storage.insert_quad(q.clone()), reference.insert(q));
        }
        // The new subjects reused reclaimed slots, so the slot count did not grow
        // by the full ten (ten ids were freed, ten reinserted).
        assert_eq!(storage.subjects.slot_count(), slots_before);

        // Full scan and every membership probe agree with the naive model.
        let got: BTreeSet<Quad> = storage
            .query_quads(None, None, None, None)
            .into_iter()
            .collect();
        assert_eq!(got, reference);
        for q in &reference {
            assert!(storage.contains_quad(q));
        }
        // A removed quad is really gone and its subject reclaimed.
        assert!(!storage.contains_quad(&mk(0)));
    }

    /// A term shared by several distinct quads (a duplicate insert takes no extra
    /// reference) must stay interned until the *last* quad referencing it is
    /// removed — guarding both the shared-term refcount and the duplicate no-op.
    #[test]
    fn test_shared_term_retained_until_last_quad_removed() {
        let mut storage = MemoryStorage::new();
        let s = Subject::NamedNode(nn("http://ex.org/s"));
        let p = Predicate::NamedNode(nn("http://ex.org/p"));
        let q1 = Quad::new(
            s.clone(),
            p.clone(),
            Object::Literal(Literal::new("o1")),
            GraphName::DefaultGraph,
        );
        let q2 = Quad::new(
            s.clone(),
            p.clone(),
            Object::Literal(Literal::new("o2")),
            GraphName::DefaultGraph,
        );

        assert!(storage.insert_quad(q1.clone()));
        assert!(storage.insert_quad(q2.clone()));
        // Duplicate insert must NOT take a second reference on the shared terms.
        assert!(!storage.insert_quad(q1.clone()));

        let s_id = storage.subjects.get_id(&s).expect("subject interned");

        // Removing q1 leaves the shared subject live — q2 still references it.
        assert!(storage.remove_quad(&q1));
        assert_eq!(storage.subjects.get_id(&s), Some(s_id));
        assert!(storage.subjects.resolve(s_id).is_some());
        assert_eq!(storage.subjects.live_count(), 1);

        // Removing q2 (the last referencing quad) reclaims the subject term. If
        // the duplicate insert had double-counted, the term would survive here.
        assert!(storage.remove_quad(&q2));
        assert_eq!(storage.subjects.get_id(&s), None);
        assert_eq!(storage.subjects.live_count(), 0);
        assert_eq!(storage.subjects.free_count(), 1);
        assert!(storage.is_empty());
    }

    /// Clearing a whole graph (as `RdfStore::clear_graph` / `drop_graph` do, by
    /// removing every quad in it) reclaims every column id it used and drops the
    /// graph from `named_graphs`.
    #[test]
    fn test_clearing_a_graph_reclaims_its_terms() {
        let mut storage = MemoryStorage::new();
        let g = GraphName::NamedNode(nn("http://ex.org/g"));
        let quads: Vec<Quad> = (0..5)
            .map(|i| {
                Quad::new(
                    Subject::NamedNode(nn(&format!("http://ex.org/s{i}"))),
                    Predicate::NamedNode(nn("http://ex.org/p")),
                    Object::Literal(Literal::new(format!("o{i}"))),
                    g.clone(),
                )
            })
            .collect();
        for q in &quads {
            assert!(storage.insert_quad(q.clone()));
        }
        assert!(storage.named_graphs.contains(&nn("http://ex.org/g")));
        let graph_slots = storage.graphs.slot_count();
        assert_eq!(storage.graphs.live_count(), 1);

        // Emulate clear_graph / drop_graph: remove every quad in graph g.
        for q in &quads {
            assert!(storage.remove_quad(q));
        }

        // The graph name is dropped and every column has reclaimed all its ids.
        assert!(!storage.named_graphs.contains(&nn("http://ex.org/g")));
        assert_eq!(storage.graphs.live_count(), 0);
        assert_eq!(storage.graphs.free_count(), graph_slots);
        assert_eq!(storage.subjects.live_count(), 0);
        assert_eq!(storage.predicates.live_count(), 0);
        assert_eq!(storage.objects.live_count(), 0);
        assert!(storage.is_empty());
    }

    /// `named_graphs` bookkeeping stays correct across id reclamation: dropping a
    /// graph's last quad reclaims its graph-name id and removes it from the set,
    /// and a brand-new graph reuses the freed id slot without corrupting the set.
    #[test]
    fn test_named_graphs_bookkeeping_with_id_reclamation() {
        let mut storage = MemoryStorage::new();
        let g1 = nn("http://ex.org/g1");
        let g2 = nn("http://ex.org/g2");
        let mk = |g: &NamedNode, o: &str| {
            Quad::new(
                Subject::NamedNode(nn("http://ex.org/s")),
                Predicate::NamedNode(nn("http://ex.org/p")),
                Object::Literal(Literal::new(o)),
                GraphName::NamedNode(g.clone()),
            )
        };

        let q1 = mk(&g1, "a");
        let q2 = mk(&g2, "b");
        assert!(storage.insert_quad(q1.clone()));
        assert!(storage.insert_quad(q2.clone()));
        assert_eq!(
            storage.named_graphs,
            [g1.clone(), g2.clone()].into_iter().collect()
        );

        // Drop g1's only quad: its graph-name id is reclaimed and g1 leaves the
        // set, while g2 stays.
        assert!(storage.remove_quad(&q1));
        assert!(!storage.named_graphs.contains(&g1));
        assert!(storage.named_graphs.contains(&g2));
        assert_eq!(
            storage.graphs.get_id(&GraphName::NamedNode(g1.clone())),
            None
        );

        // A brand-new graph g3 reuses g1's freed graph-id slot; the set tracks it
        // by value, so no slot growth and the membership stays exact.
        let g3 = nn("http://ex.org/g3");
        let q3 = mk(&g3, "c");
        assert!(storage.insert_quad(q3.clone()));
        assert_eq!(
            storage.named_graphs,
            [g2.clone(), g3.clone()].into_iter().collect()
        );
        assert_eq!(storage.graphs.slot_count(), 2);
        assert_eq!(storage.graphs.live_count(), 2);

        // Full-scan correctness after all the churn.
        let got: BTreeSet<Quad> = storage
            .query_quads(None, None, None, None)
            .into_iter()
            .collect();
        assert_eq!(got, [q2, q3].into_iter().collect());
    }

    /// Best-effort memory-footprint measurement for the interned-only design.
    /// Reports the estimated resident bytes for the current structures versus
    /// the previous design (which additionally held a full owned
    /// `BTreeSet<Quad>`), for 100k synthetic triples. Ignored by default; run
    /// with `cargo nextest run -p oxirs-core --run-ignored all
    /// memory_footprint_100k_triples` (or `cargo test ... -- --ignored`).
    #[test]
    #[ignore = "manual memory-footprint measurement"]
    fn memory_footprint_100k_triples() {
        let mut storage = MemoryStorage::new();
        for i in 0..100_000u32 {
            let quad = Quad::new(
                Subject::NamedNode(nn(&format!("http://ex.org/s{}", i % 10_000))),
                Predicate::NamedNode(nn(&format!("http://ex.org/p{}", i % 100))),
                Object::Literal(Literal::new(format!("value-{i}"))),
                GraphName::DefaultGraph,
            );
            storage.insert_quad(quad);
        }
        assert_eq!(storage.len(), 100_000);

        let after = storage.size_estimate();

        // The previous design also kept one owned `Quad` per triple in a
        // `BTreeSet<Quad>`; approximate that removed overhead (struct size plus
        // the term string bytes it duplicated) for the before/after ratio.
        let owned_quad_bytes: usize = storage
            .iter_quads()
            .map(|q| {
                std::mem::size_of::<Quad>()
                    + q.subject().to_string().len()
                    + q.predicate().to_string().len()
                    + q.object().to_string().len()
                    + q.graph_name().to_string().len()
            })
            .sum();
        let before = after + owned_quad_bytes;

        eprintln!(
            "memory_footprint(100k triples): interned-only ~= {after} bytes \
             ({:.1} MiB); previous design with owned quad set ~= {before} bytes \
             ({:.1} MiB); dropping the owned set saves ~= {} bytes ({:.2}x smaller)",
            after as f64 / (1024.0 * 1024.0),
            before as f64 / (1024.0 * 1024.0),
            owned_quad_bytes,
            before as f64 / after as f64,
        );
        assert!(after < before, "interned-only must be smaller than before");
    }
}
