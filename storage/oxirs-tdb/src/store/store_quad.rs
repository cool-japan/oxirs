//! Named-graph (quad) API for the TDB store (F4).
//!
//! A dataset is a default (unnamed) graph plus zero or more named graphs. This
//! module layers a quad API over the store:
//!
//! - **Default graph** quads are stored in the existing SPO/POS/OSP triple
//!   indexes, so the triple API ([`insert`](crate::store::TdbStore::insert),
//!   [`query_triples`](crate::store::TdbStore::query_triples), …) and the
//!   default-graph quad API operate on the same data and both round-trip
//!   through reopen exactly as before.
//! - **Named graphs** are stored in the GSPO/GPOS/GOSP quad indexes
//!   ([`QuadIndexes`](crate::index::QuadIndexes)); the graph name is interned in
//!   the shared dictionary just like any subject/predicate/object term (this is
//!   the "graph column" — a [`Term`] is a [`Term`], so no separate dictionary is
//!   needed).
//!
//! Quad-index roots and the named-graph quad count are persisted in the
//! superblock and fsynced on [`sync`](crate::store::TdbStore::sync), so quads
//! survive a `drop` + reopen just like triples.

use crate::dictionary::{Dictionary, Term};
use crate::error::{Result, TdbError};
use crate::index::{Quad, QuadIndexes, QuadScan, Triple, TripleScan};
use crate::store::store_impl::TdbStore;
use crate::store::store_stream::decode_triple_terms;
use crate::store::store_wal::StoreOp;

/// The graph a decoded quad belongs to.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GraphName {
    /// The unnamed default graph (backed by the triple indexes).
    DefaultGraph,
    /// A named graph identified by a [`Term`] (an IRI or blank node).
    Named(Term),
}

/// Selects which graph(s) a quad scan targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphTarget<'a> {
    /// Match quads in any graph — the default graph and every named graph.
    AnyGraph,
    /// Match quads only in the default (unnamed) graph.
    DefaultGraph,
    /// Match quads only in the named graph identified by this term.
    Named(&'a Term),
}

/// A decoded quad returned by the quad scan API.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuadResult {
    /// The graph the quad belongs to.
    pub graph: GraphName,
    /// Subject term.
    pub subject: Term,
    /// Predicate term.
    pub predicate: Term,
    /// Object term.
    pub object: Term,
}

/// Decode a node-encoded named-graph [`Quad`] into a [`QuadResult`], failing
/// loudly if any id (including the graph id) is absent from the dictionary.
fn decode_quad_terms(dictionary: &Dictionary, quad: Quad) -> Result<QuadResult> {
    let graph = dictionary
        .decode(quad.graph)?
        .ok_or_else(|| TdbError::Other("Graph id not found in dictionary".to_string()))?;
    let subject = dictionary
        .decode(quad.subject)?
        .ok_or_else(|| TdbError::Other("Subject id not found in dictionary".to_string()))?;
    let predicate = dictionary
        .decode(quad.predicate)?
        .ok_or_else(|| TdbError::Other("Predicate id not found in dictionary".to_string()))?;
    let object = dictionary
        .decode(quad.object)?
        .ok_or_else(|| TdbError::Other("Object id not found in dictionary".to_string()))?;
    Ok(QuadResult {
        graph: GraphName::Named(graph),
        subject,
        predicate,
        object,
    })
}

/// A lazy, streaming iterator over decoded quads (F5).
///
/// Created by [`TdbStore::quad_iter`]. It first drains the default-graph part
/// (a triple scan, tagged [`GraphName::DefaultGraph`]) and then the named-graph
/// part (a quad scan, tagged [`GraphName::Named`]); either part may be absent
/// depending on the [`GraphTarget`]. The full result set is never materialized
/// in memory at once — only the underlying B+Tree leaf pages are buffered by
/// the scans.
pub struct QuadTermIter<'a> {
    /// Dictionary used to decode node ids back into terms.
    dictionary: &'a Dictionary,
    /// Default-graph triple scan (yields [`GraphName::DefaultGraph`] quads).
    default_scan: Option<TripleScan>,
    /// Named-graph quad scan (yields [`GraphName::Named`] quads).
    named_scan: Option<QuadScan>,
}

impl<'a> QuadTermIter<'a> {
    /// An iterator that yields nothing (used when a pattern term is unknown).
    fn empty(dictionary: &'a Dictionary) -> Self {
        Self {
            dictionary,
            default_scan: None,
            named_scan: None,
        }
    }
}

impl Iterator for QuadTermIter<'_> {
    type Item = Result<QuadResult>;

    fn next(&mut self) -> Option<Self::Item> {
        // Drain the default-graph part first.
        if let Some(scan) = self.default_scan.as_mut() {
            match scan.next() {
                Some(Ok(triple)) => {
                    return Some(decode_triple_terms(self.dictionary, triple).map(
                        |(subject, predicate, object)| QuadResult {
                            graph: GraphName::DefaultGraph,
                            subject,
                            predicate,
                            object,
                        },
                    ));
                }
                Some(Err(e)) => return Some(Err(e)),
                None => self.default_scan = None,
            }
        }

        // Then the named-graph part.
        if let Some(scan) = self.named_scan.as_mut() {
            match scan.next() {
                Some(Ok(quad)) => return Some(decode_quad_terms(self.dictionary, quad)),
                Some(Err(e)) => return Some(Err(e)),
                None => self.named_scan = None,
            }
        }

        None
    }
}

impl TdbStore {
    /// Number of named-graph quads currently stored.
    ///
    /// Default-graph triples are counted separately by
    /// [`count`](crate::store::TdbStore::count); the whole-dataset size is
    /// [`dataset_len`](TdbStore::dataset_len).
    pub fn quad_count(&self) -> usize {
        self.quad_count
    }

    /// Total number of statements across the whole dataset: default-graph
    /// triples plus named-graph quads.
    pub fn dataset_len(&self) -> usize {
        self.triple_count + self.quad_count
    }

    /// Whether named-graph (quad) support is available on this store handle.
    pub fn quads_enabled(&self) -> bool {
        self.quad_indexes.is_some()
    }

    /// Insert a quad `(graph, subject, predicate, object)`.
    ///
    /// `graph` is `None` for the default graph (routed to the triple indexes)
    /// or `Some(term)` for a named graph (routed to the quad indexes, interning
    /// the graph term). Returns `true` if the quad was newly added, `false` if
    /// it already existed. Named-graph inserts fail with
    /// [`TdbError::Unsupported`] when quad support is disabled.
    pub fn insert_quad(
        &mut self,
        graph: Option<&Term>,
        subject: &Term,
        predicate: &Term,
        object: &Term,
    ) -> Result<bool> {
        match graph {
            None => self.insert_default_graph_quad(subject, predicate, object),
            Some(graph_term) => {
                if !self.quads_writable {
                    return Err(TdbError::Unsupported(
                        "named-graph writes are disabled (open with enable_quad_indexes = true)"
                            .to_string(),
                    ));
                }
                let g_id = self.dictionary.encode(graph_term)?;
                let s_id = self.dictionary.encode(subject)?;
                let p_id = self.dictionary.encode(predicate)?;
                let o_id = self.dictionary.encode(object)?;
                let quad = Quad::new(g_id, s_id, p_id, o_id);
                let is_new = {
                    let quad_indexes = self.quad_indexes.as_mut().ok_or_else(|| {
                        TdbError::Unsupported("quad indexes are not initialized".to_string())
                    })?;
                    quad_indexes.insert(quad)?
                };
                if is_new {
                    self.quad_count += 1;
                    // Log the committed named-graph insert so it survives a crash
                    // before the next checkpoint.
                    self.wal_log_op(StoreOp::InsertQuad {
                        graph: graph_term.clone(),
                        subject: subject.clone(),
                        predicate: predicate.clone(),
                        object: object.clone(),
                    })?;
                }
                Ok(is_new)
            }
        }
    }

    /// Insert a default-graph quad into the triple indexes (shared with the
    /// triple API), maintaining the bloom filter and triple count.
    fn insert_default_graph_quad(
        &mut self,
        subject: &Term,
        predicate: &Term,
        object: &Term,
    ) -> Result<bool> {
        let s_id = self.dictionary.encode(subject)?;
        let p_id = self.dictionary.encode(predicate)?;
        let o_id = self.dictionary.encode(object)?;
        let triple = Triple::new(s_id, p_id, o_id);
        let is_new = self.indexes.insert(triple)?;
        if let Some(ref mut bloom) = self.bloom_filter {
            bloom.insert(&triple);
        }
        if is_new {
            self.triple_count += 1;
            // A default-graph quad is a triple: log it as a triple insert (the
            // triple indexes back both the triple API and the default graph).
            self.wal_log_op(StoreOp::InsertTriple {
                subject: subject.clone(),
                predicate: predicate.clone(),
                object: object.clone(),
            })?;
        }
        // The default graph changed, so cached triple-query results are stale.
        self.query_cache.clear();
        Ok(is_new)
    }

    /// Bulk-insert quads `(graph, subject, predicate, object)` with a sorted,
    /// sequential-leaf-append B+Tree build (F6, Phase A).
    ///
    /// Default-graph quads (`graph == None`) are routed to the SPO/POS/OSP triple
    /// indexes; named-graph quads to the GSPO/GPOS/GOSP quad indexes. Each index
    /// is fed its batch pre-sorted in its own key order (see
    /// [`TripleIndexes::insert_sorted`](crate::index::TripleIndexes::insert_sorted)
    /// and [`QuadIndexes::insert_sorted`](crate::index::QuadIndexes::insert_sorted)),
    /// so the trees append to their right-most leaves instead of splitting in
    /// random order.
    ///
    /// Coordinating with F3, the whole batch is one WAL transaction and one
    /// checkpoint. Returns the number of genuinely new statements (default-graph
    /// triples plus named-graph quads) actually added.
    ///
    /// Fails loudly *before any mutation* if a subject is a literal, or if any
    /// named-graph quad is present while named-graph writes are disabled
    /// ([`TdbError::Unsupported`]).
    pub fn insert_quads_bulk(
        &mut self,
        quads: &[(Option<Term>, Term, Term, Term)],
    ) -> Result<usize> {
        // Validate the whole batch first so a malformed input never leaves a
        // half-applied store.
        let mut has_named = false;
        for (graph, subject, _predicate, _object) in quads {
            if matches!(subject, Term::Literal { .. }) {
                return Err(TdbError::Other("Subject cannot be a literal".to_string()));
            }
            if graph.is_some() {
                has_named = true;
            }
        }
        if has_named && !self.quads_writable {
            return Err(TdbError::Unsupported(
                "named-graph writes are disabled (open with enable_quad_indexes = true)"
                    .to_string(),
            ));
        }
        if quads.is_empty() {
            return Ok(0);
        }

        // (1)+(2) intern + encode, splitting default-graph triples from named
        // quads and building the one-transaction WAL op list in input order.
        let mut encoded_triples: Vec<Triple> = Vec::new();
        let mut encoded_quads: Vec<Quad> = Vec::new();
        let mut ops: Vec<StoreOp> = Vec::with_capacity(quads.len());
        for (graph, subject, predicate, object) in quads {
            let s_id = self.dictionary.encode(subject)?;
            let p_id = self.dictionary.encode(predicate)?;
            let o_id = self.dictionary.encode(object)?;
            match graph {
                None => {
                    encoded_triples.push(Triple::new(s_id, p_id, o_id));
                    ops.push(StoreOp::InsertTriple {
                        subject: subject.clone(),
                        predicate: predicate.clone(),
                        object: object.clone(),
                    });
                }
                Some(graph_term) => {
                    let g_id = self.dictionary.encode(graph_term)?;
                    encoded_quads.push(Quad::new(g_id, s_id, p_id, o_id));
                    ops.push(StoreOp::InsertQuad {
                        graph: graph_term.clone(),
                        subject: subject.clone(),
                        predicate: predicate.clone(),
                        object: object.clone(),
                    });
                }
            }
        }

        // (3)+(4) sorted bulk build for the default-graph triple indexes.
        let mut new_count = self.indexes.insert_sorted(&encoded_triples)?;
        if let Some(ref mut bloom) = self.bloom_filter {
            for triple in &encoded_triples {
                bloom.insert(triple);
            }
        }
        self.triple_count += new_count;

        // Sorted bulk build for the named-graph quad indexes (materialized on
        // demand — only reachable when quads are writable, guarded above).
        if !encoded_quads.is_empty() {
            if self.quad_indexes.is_none() {
                self.quad_indexes = Some(QuadIndexes::new(self.buffer_pool.clone()));
                self.quads_writable = true;
            }
            let quad_indexes = self.quad_indexes.as_mut().ok_or_else(|| {
                TdbError::Unsupported("quad indexes are not initialized".to_string())
            })?;
            let new_quads = quad_indexes.insert_sorted(&encoded_quads)?;
            self.quad_count += new_quads;
            new_count += new_quads;
        }

        // The default graph changed, so cached triple-query results are stale.
        self.query_cache.clear();

        // One WAL transaction + one durable checkpoint for the whole batch (F3).
        self.wal_log_batch(&ops)?;
        self.sync()?;

        Ok(new_count)
    }

    /// Delete a quad `(graph, subject, predicate, object)`.
    ///
    /// Returns `true` if a quad was removed. Returns `Ok(false)` (not an error)
    /// when the quad — or any of its terms — is not present. Named-graph deletes
    /// fail with [`TdbError::Unsupported`] when quad support is disabled.
    pub fn delete_quad(
        &mut self,
        graph: Option<&Term>,
        subject: &Term,
        predicate: &Term,
        object: &Term,
    ) -> Result<bool> {
        match graph {
            None => {
                let (s_id, p_id, o_id) = match self.lookup_spo(subject, predicate, object)? {
                    Some(ids) => ids,
                    None => return Ok(false),
                };
                let triple = Triple::new(s_id, p_id, o_id);
                let deleted = self.indexes.delete(&triple)?;
                if deleted {
                    self.triple_count = self.triple_count.saturating_sub(1);
                    self.query_cache.clear();
                    // A default-graph quad delete is a triple delete.
                    self.wal_log_op(StoreOp::DeleteTriple {
                        subject: subject.clone(),
                        predicate: predicate.clone(),
                        object: object.clone(),
                    })?;
                }
                Ok(deleted)
            }
            Some(graph_term) => {
                if !self.quads_writable {
                    return Err(TdbError::Unsupported(
                        "named-graph writes are disabled (open with enable_quad_indexes = true)"
                            .to_string(),
                    ));
                }
                let g_id = match self.dictionary.lookup(graph_term)? {
                    Some(id) => id,
                    None => return Ok(false),
                };
                let (s_id, p_id, o_id) = match self.lookup_spo(subject, predicate, object)? {
                    Some(ids) => ids,
                    None => return Ok(false),
                };
                let quad = Quad::new(g_id, s_id, p_id, o_id);
                let deleted = {
                    let quad_indexes = self.quad_indexes.as_mut().ok_or_else(|| {
                        TdbError::Unsupported("quad indexes are not initialized".to_string())
                    })?;
                    quad_indexes.delete(quad)?
                };
                if deleted {
                    self.quad_count = self.quad_count.saturating_sub(1);
                    // Log the committed named-graph delete.
                    self.wal_log_op(StoreOp::DeleteQuad {
                        graph: graph_term.clone(),
                        subject: subject.clone(),
                        predicate: predicate.clone(),
                        object: object.clone(),
                    })?;
                }
                Ok(deleted)
            }
        }
    }

    /// Check whether a quad `(graph, subject, predicate, object)` exists.
    pub fn contains_quad(
        &self,
        graph: Option<&Term>,
        subject: &Term,
        predicate: &Term,
        object: &Term,
    ) -> Result<bool> {
        match graph {
            None => {
                let (s_id, p_id, o_id) = match self.lookup_spo(subject, predicate, object)? {
                    Some(ids) => ids,
                    None => return Ok(false),
                };
                self.indexes.contains(&Triple::new(s_id, p_id, o_id))
            }
            Some(graph_term) => {
                let g_id = match self.dictionary.lookup(graph_term)? {
                    Some(id) => id,
                    None => return Ok(false),
                };
                let (s_id, p_id, o_id) = match self.lookup_spo(subject, predicate, object)? {
                    Some(ids) => ids,
                    None => return Ok(false),
                };
                match &self.quad_indexes {
                    Some(quad_indexes) => quad_indexes.contains(&Quad::new(g_id, s_id, p_id, o_id)),
                    None => Ok(false),
                }
            }
        }
    }

    /// Look up the node ids of a `(subject, predicate, object)` triple, or
    /// `None` if any term is absent from the dictionary.
    fn lookup_spo(
        &self,
        subject: &Term,
        predicate: &Term,
        object: &Term,
    ) -> Result<
        Option<(
            crate::dictionary::NodeId,
            crate::dictionary::NodeId,
            crate::dictionary::NodeId,
        )>,
    > {
        let s_id = match self.dictionary.lookup(subject)? {
            Some(id) => id,
            None => return Ok(None),
        };
        let p_id = match self.dictionary.lookup(predicate)? {
            Some(id) => id,
            None => return Ok(None),
        };
        let o_id = match self.dictionary.lookup(object)? {
            Some(id) => id,
            None => return Ok(None),
        };
        Ok(Some((s_id, p_id, o_id)))
    }

    /// Open a lazy, streaming iterator over the quads matching a pattern (F5).
    ///
    /// `graph` selects the default graph, a named graph, or any graph; each of
    /// `subject`/`predicate`/`object` is `None` for a wildcard. Decoded quads
    /// are produced one at a time without materializing the whole result set. A
    /// pattern component whose term is unknown yields an empty iterator.
    pub fn quad_iter(
        &self,
        graph: GraphTarget<'_>,
        subject: Option<&Term>,
        predicate: Option<&Term>,
        object: Option<&Term>,
    ) -> Result<QuadTermIter<'_>> {
        // Resolve pattern terms to ids; an unknown term matches nothing.
        let s_id = match self.resolve_pattern(subject)? {
            Some(id) => id,
            None => return Ok(QuadTermIter::empty(&self.dictionary)),
        };
        let p_id = match self.resolve_pattern(predicate)? {
            Some(id) => id,
            None => return Ok(QuadTermIter::empty(&self.dictionary)),
        };
        let o_id = match self.resolve_pattern(object)? {
            Some(id) => id,
            None => return Ok(QuadTermIter::empty(&self.dictionary)),
        };

        match graph {
            GraphTarget::DefaultGraph => {
                let default_scan = self.indexes.scan(s_id, p_id, o_id)?;
                Ok(QuadTermIter {
                    dictionary: &self.dictionary,
                    default_scan: Some(default_scan),
                    named_scan: None,
                })
            }
            GraphTarget::Named(graph_term) => {
                let g_id = match self.dictionary.lookup(graph_term)? {
                    Some(id) => id,
                    None => return Ok(QuadTermIter::empty(&self.dictionary)),
                };
                match &self.quad_indexes {
                    Some(quad_indexes) => {
                        let named_scan = quad_indexes.scan(Some(g_id), s_id, p_id, o_id)?;
                        Ok(QuadTermIter {
                            dictionary: &self.dictionary,
                            default_scan: None,
                            named_scan: Some(named_scan),
                        })
                    }
                    None => Ok(QuadTermIter::empty(&self.dictionary)),
                }
            }
            GraphTarget::AnyGraph => {
                let default_scan = Some(self.indexes.scan(s_id, p_id, o_id)?);
                let named_scan = match &self.quad_indexes {
                    Some(quad_indexes) => Some(quad_indexes.scan(None, s_id, p_id, o_id)?),
                    None => None,
                };
                Ok(QuadTermIter {
                    dictionary: &self.dictionary,
                    default_scan,
                    named_scan,
                })
            }
        }
    }

    /// Resolve an optional pattern term to `Some(Some(id))` when bound and
    /// present, `Some(None)` when a wildcard, and `None` when the term is bound
    /// but absent from the dictionary (so the whole scan yields nothing).
    fn resolve_pattern(
        &self,
        term: Option<&Term>,
    ) -> Result<Option<Option<crate::dictionary::NodeId>>> {
        match term {
            None => Ok(Some(None)),
            Some(t) => match self.dictionary.lookup(t)? {
                Some(id) => Ok(Some(Some(id))),
                None => Ok(None),
            },
        }
    }

    /// Scan quads matching a pattern, materializing the decoded results.
    ///
    /// A convenience `Vec`-collecting wrapper around [`TdbStore::quad_iter`];
    /// prefer `quad_iter` (or [`TdbStore::for_each_quad`]) for large result
    /// sets to avoid buffering everything in memory.
    pub fn scan_quads(
        &self,
        graph: GraphTarget<'_>,
        subject: Option<&Term>,
        predicate: Option<&Term>,
        object: Option<&Term>,
    ) -> Result<Vec<QuadResult>> {
        self.quad_iter(graph, subject, predicate, object)?.collect()
    }

    /// Invoke `f` for each quad matching the pattern, streaming (never
    /// materializing the whole result). Stops early and propagates the error if
    /// `f` returns `Err`.
    pub fn for_each_quad<F>(
        &self,
        graph: GraphTarget<'_>,
        subject: Option<&Term>,
        predicate: Option<&Term>,
        object: Option<&Term>,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut(QuadResult) -> Result<()>,
    {
        for item in self.quad_iter(graph, subject, predicate, object)? {
            f(item?)?;
        }
        Ok(())
    }
}
