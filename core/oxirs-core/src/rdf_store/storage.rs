//! RDF storage backends and implementations.
//!
//! [`MemoryStorage`] uses **term interning** to avoid the previous five full
//! owned-`Quad` copies per triple. Each column (subject, predicate, object,
//! graph name) is interned into a [`ColumnDictionary`], and quads are stored as
//! compact `[u32; 4]` id-tuples in four permutation indexes (SPOG/POSG/OSPG/
//! GSPO) that support `O(log n + k)` range-scan lookups without cloning whole
//! index sets. The canonical owned set `quads` is retained so query results can
//! be materialized deterministically at the API boundary and so the file stays
//! a straightforward N-Quads export.

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

/// In-memory storage implementation backed by interned `[u32; 4]` id-tuples.
#[derive(Debug, Clone, Default)]
pub struct MemoryStorage {
    /// Canonical set of quads in owned form (SPOG-ordered).
    ///
    /// Retained for deterministic iteration, `contains`/`len`, N-Quads export,
    /// and to satisfy read-only consumers that iterate quads directly. The four
    /// permutation indexes below hold only compact id-tuples, so this is the
    /// single owned copy per triple rather than the previous five.
    pub quads: BTreeSet<Quad>,
    /// Per-column term dictionaries (term <-> u32).
    subjects: ColumnDictionary<Subject>,
    predicates: ColumnDictionary<Predicate>,
    objects: ColumnDictionary<Object>,
    graphs: ColumnDictionary<GraphName>,
    /// SPOG permutation: `[s, p, o, g]` — subject and full lookups.
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

/// Lead index chosen for a lookup, encoding how a scanned tuple maps back to
/// canonical `(s, p, o, g)` ids.
enum LeadIndex {
    Spog,
    Posg,
    Ospg,
    Gspo,
}

impl MemoryStorage {
    pub fn new() -> Self {
        MemoryStorage {
            quads: BTreeSet::new(),
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
        // BTreeSet::insert reports novelty and keeps the canonical owned copy.
        if !self.quads.insert(quad.clone()) {
            return false;
        }

        let s = self.subjects.intern(quad.subject());
        let p = self.predicates.intern(quad.predicate());
        let o = self.objects.intern(quad.object());
        let g = self.graphs.intern(quad.graph_name());

        self.spog.insert([s, p, o, g]);
        self.posg.insert([p, o, s, g]);
        self.ospg.insert([o, s, p, g]);
        self.gspo.insert([g, s, p, o]);

        if let GraphName::NamedNode(graph_name) = quad.graph_name() {
            self.named_graphs.insert(graph_name.clone());
        }

        true
    }

    pub fn remove_quad(&mut self, quad: &Quad) -> bool {
        if !self.quads.remove(quad) {
            return false;
        }

        // The quad was present, so every component is already interned.
        if let (Some(s), Some(p), Some(o), Some(g)) = (
            self.subjects.get_id(quad.subject()),
            self.predicates.get_id(quad.predicate()),
            self.objects.get_id(quad.object()),
            self.graphs.get_id(quad.graph_name()),
        ) {
            self.spog.remove(&[s, p, o, g]);
            self.posg.remove(&[p, o, s, g]);
            self.ospg.remove(&[o, s, p, g]);
            self.gspo.remove(&[g, s, p, o]);

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
        }

        true
    }

    pub fn contains_quad(&self, quad: &Quad) -> bool {
        self.quads.contains(quad)
    }

    #[allow(dead_code)]
    fn iter_quads(&self) -> impl Iterator<Item = &Quad> {
        self.quads.iter()
    }

    /// Resolve a canonical `(s, p, o, g)` id-tuple into an owned `Quad`.
    fn materialize(&self, s: u32, p: u32, o: u32, g: u32) -> Option<Quad> {
        Some(Quad::new(
            self.subjects.resolve(s)?.clone(),
            self.predicates.resolve(p)?.clone(),
            self.objects.resolve(o)?.clone(),
            self.graphs.resolve(g)?.clone(),
        ))
    }

    pub fn query_quads(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
        graph_name: Option<&GraphName>,
    ) -> Vec<Quad> {
        // Resolve each bound term to its id. A bound term that is not interned
        // means no quad can match, so return immediately.
        let sid = match subject {
            Some(s) => match self.subjects.get_id(s) {
                Some(id) => Some(id),
                None => return Vec::new(),
            },
            None => None,
        };
        let pid = match predicate {
            Some(p) => match self.predicates.get_id(p) {
                Some(id) => Some(id),
                None => return Vec::new(),
            },
            None => None,
        };
        let oid = match object {
            Some(o) => match self.objects.get_id(o) {
                Some(id) => Some(id),
                None => return Vec::new(),
            },
            None => None,
        };
        let gid = match graph_name {
            Some(g) => match self.graphs.get_id(g) {
                Some(id) => Some(id),
                None => return Vec::new(),
            },
            None => None,
        };

        // Fully unbound: return every quad (result-sized clone, deterministic).
        if sid.is_none() && pid.is_none() && oid.is_none() && gid.is_none() {
            return self.quads.iter().cloned().collect();
        }

        // Pick the lead permutation by the most selective bound column
        // (subject, then object, then predicate, then graph).
        let (lead, index, key): (LeadIndex, &BTreeSet<[u32; 4]>, u32) = if let Some(s) = sid {
            (LeadIndex::Spog, &self.spog, s)
        } else if let Some(o) = oid {
            (LeadIndex::Ospg, &self.ospg, o)
        } else if let Some(p) = pid {
            (LeadIndex::Posg, &self.posg, p)
        } else if let Some(g) = gid {
            (LeadIndex::Gspo, &self.gspo, g)
        } else {
            // Unreachable given the guard above, but stay safe.
            return self.quads.iter().cloned().collect();
        };

        // Collect into a BTreeSet<Quad> so results stay Quad-ordered and
        // deterministic regardless of which permutation led the scan.
        let mut results: BTreeSet<Quad> = BTreeSet::new();
        let lower = [key, 0, 0, 0];
        let upper = [key, u32::MAX, u32::MAX, u32::MAX];
        for tuple in index.range(lower..=upper) {
            let (s, p, o, g) = match lead {
                LeadIndex::Spog => (tuple[0], tuple[1], tuple[2], tuple[3]),
                LeadIndex::Posg => (tuple[2], tuple[0], tuple[1], tuple[3]),
                LeadIndex::Ospg => (tuple[1], tuple[2], tuple[0], tuple[3]),
                LeadIndex::Gspo => (tuple[1], tuple[2], tuple[3], tuple[0]),
            };

            // Apply the remaining bound filters via cheap integer comparisons.
            if let Some(want) = sid {
                if s != want {
                    continue;
                }
            }
            if let Some(want) = pid {
                if p != want {
                    continue;
                }
            }
            if let Some(want) = oid {
                if o != want {
                    continue;
                }
            }
            if let Some(want) = gid {
                if g != want {
                    continue;
                }
            }

            if let Some(quad) = self.materialize(s, p, o, g) {
                results.insert(quad);
            }
        }

        results.into_iter().collect()
    }

    /// Stream every quad matching the pattern to `f` without allocating a
    /// result `Vec`.
    ///
    /// Iterates the canonical owned `quads` set (which is `Quad`-ordered) and
    /// applies the bound-term filters inline, so quads are visited in exactly
    /// the same deterministic order as [`query_quads`](Self::query_quads)
    /// returns them, but only one quad clone is alive at a time (the one handed
    /// to `f`). This backs [`Store::for_each_quad`](super::Store::for_each_quad)
    /// so consumers can serialize a whole graph incrementally without ever
    /// materializing it in full.
    pub fn for_each_quad(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
        graph_name: Option<&GraphName>,
        f: &mut dyn FnMut(Quad),
    ) {
        for quad in &self.quads {
            if let Some(s) = subject {
                if quad.subject() != s {
                    continue;
                }
            }
            if let Some(p) = predicate {
                if quad.predicate() != p {
                    continue;
                }
            }
            if let Some(o) = object {
                if quad.object() != o {
                    continue;
                }
            }
            if let Some(g) = graph_name {
                if quad.graph_name() != g {
                    continue;
                }
            }
            f(quad.clone());
        }
    }

    pub fn len(&self) -> usize {
        self.quads.len()
    }

    pub fn is_empty(&self) -> bool {
        self.quads.is_empty()
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
            .filter(|q| subject.is_none_or(|s| q.subject() == s))
            .filter(|q| predicate.is_none_or(|p| q.predicate() == p))
            .filter(|q| object.is_none_or(|o| q.object() == o))
            .filter(|q| graph_name.is_none_or(|g| q.graph_name() == g))
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
}
