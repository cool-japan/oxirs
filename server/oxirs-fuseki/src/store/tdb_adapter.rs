//! On-disk TDB2 backend adapter implementing the [`oxirs_core::Store`] trait.
//!
//! Fuseki's [`Store`](crate::store::Store) delegates every RDF operation to a
//! concrete backend held behind `Arc<RwLock<dyn oxirs_core::Store>>`. Until now
//! the only backend was [`oxirs_core::RdfStore`]; a dataset configured as
//! `StoreType::TDB2` / `dataset_type = "tdb2"` was an inert label.
//!
//! [`TdbStoreAdapter`] closes that gap: it wraps an [`oxirs_tdb::TdbStore`]
//! (real durable B+Tree persistence with named-graph quad support) and presents
//! it through the same `oxirs_core::Store` trait, so every existing handler —
//! SPARQL query/update, Graph Store Protocol, `/upload` — works against a real
//! on-disk TDB2 dataset with **zero handler changes**.
//!
//! ## Term model bridging
//!
//! `oxirs-tdb` does **not** depend on `oxirs-core`; it has its own
//! [`oxirs_tdb::dictionary::Term`] enum (`Iri` / `Literal { value, language,
//! datatype }` / `BlankNode`). This module converts faithfully in both
//! directions between that enum and the richer `oxirs_core` model
//! (`NamedNode` / `BlankNode` / typed & language-tagged `Literal` /
//! `Subject` / `Predicate` / `Object` / `GraphName` / `Quad`). Round-tripping
//! preserves literal datatype and language tag, blank-node identity, and the
//! default-vs-named graph distinction (the default graph maps to TDB's reserved
//! default-graph id, i.e. `graph = None`).
//!
//! ## Interior mutability
//!
//! [`TdbStore`]'s mutating methods take `&mut self`, but the `oxirs_core::Store`
//! trait is `&self` throughout (it is used as `dyn Store`). The adapter
//! therefore holds the store behind an [`RwLock`]: read patterns take a shared
//! lock (concurrent scans), mutations take the exclusive lock.

use std::path::Path;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use oxirs_core::model::{
    BlankNode, GraphName, Literal, NamedNode, Object, Predicate, Quad, Subject,
};
use oxirs_core::query::{QueryEngine, QueryResult as CoreQueryResult};
use oxirs_core::rdf_store::{OxirsQueryResults, PreparedQuery, VariableBinding};
use oxirs_core::{OxirsError, Result, Store as CoreStore};

use oxirs_tdb::dictionary::Term as TdbTerm;
use oxirs_tdb::{GraphName as TdbGraphName, GraphTarget, QuadResult, TdbStore};

/// The `xsd:string` datatype IRI. A simple (untyped, unlanguaged) literal has
/// this datatype implicitly in the `oxirs-core` model; on the TDB side it is
/// represented as a plain literal with no datatype, so the two round-trip.
const XSD_STRING: &str = "http://www.w3.org/2001/XMLSchema#string";

/// Wrap any backend (TDB) error as an `oxirs_core` store error.
fn tdb_err<E: std::fmt::Display>(e: E) -> OxirsError {
    OxirsError::Store(format!("TDB2 backend error: {e}"))
}

/// When the adapter fsyncs the underlying TDB store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TdbSyncPolicy {
    /// fsync after every individual mutation (durable per write; the safe
    /// default for a server that must survive a crash between operations).
    PerWrite,
    /// fsync only on explicit [`flush`](CoreStore) / bulk load / `Drop`. Faster
    /// for write-heavy workloads that tolerate losing the last few unsynced
    /// operations after a crash.
    OnFlush,
}

/// A durable on-disk TDB2 backend presented through the `oxirs_core::Store`
/// trait. See the [module documentation](self) for the design.
pub struct TdbStoreAdapter {
    /// The wrapped store. `RwLock` provides the interior mutability the `&self`
    /// trait methods need over `TdbStore`'s `&mut self` mutators.
    inner: RwLock<TdbStore>,
    /// fsync policy applied after single-quad mutations.
    sync_policy: TdbSyncPolicy,
}

impl std::fmt::Debug for TdbStoreAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TdbStoreAdapter")
            .field("sync_policy", &self.sync_policy)
            .finish_non_exhaustive()
    }
}

impl TdbStoreAdapter {
    /// Open (or create) a TDB2 store rooted at `dir`, syncing after every write
    /// ([`TdbSyncPolicy::PerWrite`]).
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_sync_policy(dir, TdbSyncPolicy::PerWrite)
    }

    /// Open (or create) a TDB2 store rooted at `dir` with an explicit sync
    /// policy.
    pub fn open_with_sync_policy(
        dir: impl AsRef<Path>,
        sync_policy: TdbSyncPolicy,
    ) -> Result<Self> {
        let store = TdbStore::open(dir.as_ref()).map_err(tdb_err)?;
        Ok(Self {
            inner: RwLock::new(store),
            sync_policy,
        })
    }

    /// Acquire a shared read lock on the wrapped store.
    fn read(&self) -> Result<RwLockReadGuard<'_, TdbStore>> {
        self.inner
            .read()
            .map_err(|e| OxirsError::Store(format!("TDB2 read lock poisoned: {e}")))
    }

    /// Acquire the exclusive write lock on the wrapped store.
    fn write(&self) -> Result<RwLockWriteGuard<'_, TdbStore>> {
        self.inner
            .write()
            .map_err(|e| OxirsError::Store(format!("TDB2 write lock poisoned: {e}")))
    }

    /// fsync after a single mutation when the policy demands it.
    fn maybe_sync(&self, store: &TdbStore) -> Result<()> {
        if self.sync_policy == TdbSyncPolicy::PerWrite {
            store.sync().map_err(tdb_err)?;
        }
        Ok(())
    }

    /// Durably flush all buffered pages and the catalog to disk.
    pub fn sync(&self) -> Result<()> {
        self.read()?.sync().map_err(tdb_err)
    }
}

// ---------------------------------------------------------------------------
// oxirs-core model  ->  oxirs-tdb Term conversion (write / lookup direction)
// ---------------------------------------------------------------------------

/// Convert an `oxirs-core` [`Subject`] into a TDB [`TdbTerm`]. A variable or
/// RDF-star quoted triple cannot be stored and fails loudly.
fn subject_to_tdb(subject: &Subject) -> Result<TdbTerm> {
    match subject {
        Subject::NamedNode(n) => Ok(TdbTerm::iri(n.as_str())),
        Subject::BlankNode(b) => Ok(TdbTerm::blank_node(b.as_str())),
        Subject::Variable(_) => Err(OxirsError::Store(
            "cannot store an unbound variable as a subject in TDB2".to_string(),
        )),
        Subject::QuotedTriple(_) => Err(OxirsError::NotSupported(
            "RDF-star quoted triples are not supported by the TDB2 backend".to_string(),
        )),
    }
}

/// Convert an `oxirs-core` [`Predicate`] into a TDB [`TdbTerm`].
fn predicate_to_tdb(predicate: &Predicate) -> Result<TdbTerm> {
    match predicate {
        Predicate::NamedNode(n) => Ok(TdbTerm::iri(n.as_str())),
        Predicate::Variable(_) => Err(OxirsError::Store(
            "cannot store an unbound variable as a predicate in TDB2".to_string(),
        )),
    }
}

/// Convert an `oxirs-core` [`Object`] into a TDB [`TdbTerm`].
fn object_to_tdb(object: &Object) -> Result<TdbTerm> {
    match object {
        Object::NamedNode(n) => Ok(TdbTerm::iri(n.as_str())),
        Object::BlankNode(b) => Ok(TdbTerm::blank_node(b.as_str())),
        Object::Literal(l) => Ok(literal_to_tdb(l)),
        Object::Variable(_) => Err(OxirsError::Store(
            "cannot store an unbound variable as an object in TDB2".to_string(),
        )),
        Object::QuotedTriple(_) => Err(OxirsError::NotSupported(
            "RDF-star quoted triples are not supported by the TDB2 backend".to_string(),
        )),
    }
}

/// Convert an `oxirs-core` [`Literal`] into a TDB literal [`TdbTerm`],
/// preserving the language tag or datatype. A simple `xsd:string` literal is
/// stored as a plain TDB literal (no datatype) so it round-trips exactly.
fn literal_to_tdb(literal: &Literal) -> TdbTerm {
    if let Some(lang) = literal.language() {
        TdbTerm::literal_with_lang(literal.value(), lang)
    } else {
        let datatype = literal.datatype();
        if datatype.as_str() == XSD_STRING {
            TdbTerm::literal(literal.value())
        } else {
            TdbTerm::literal_with_datatype(literal.value(), datatype.as_str())
        }
    }
}

/// Convert an `oxirs-core` [`GraphName`] into the TDB graph term: `None` for the
/// default graph, `Some(term)` for a named graph.
fn graph_to_tdb(graph: &GraphName) -> Result<Option<TdbTerm>> {
    match graph {
        GraphName::DefaultGraph => Ok(None),
        GraphName::NamedNode(n) => Ok(Some(TdbTerm::iri(n.as_str()))),
        GraphName::BlankNode(b) => Ok(Some(TdbTerm::blank_node(b.as_str()))),
        GraphName::Variable(_) => Err(OxirsError::Store(
            "cannot store an unbound variable as a graph name in TDB2".to_string(),
        )),
    }
}

// ---------------------------------------------------------------------------
// oxirs-tdb Term  ->  oxirs-core model conversion (read / decode direction)
// ---------------------------------------------------------------------------

/// Decode a TDB [`TdbTerm`] into an `oxirs-core` [`Subject`].
fn tdb_to_subject(term: &TdbTerm) -> Result<Subject> {
    match term {
        TdbTerm::Iri(iri) => Ok(Subject::NamedNode(NamedNode::new(iri).map_err(tdb_err)?)),
        TdbTerm::BlankNode(id) => Ok(Subject::BlankNode(BlankNode::new(id).map_err(tdb_err)?)),
        TdbTerm::Literal { .. } => Err(OxirsError::Store(
            "TDB2 returned a literal in subject position (corrupt data)".to_string(),
        )),
    }
}

/// Decode a TDB [`TdbTerm`] into an `oxirs-core` [`Predicate`].
fn tdb_to_predicate(term: &TdbTerm) -> Result<Predicate> {
    match term {
        TdbTerm::Iri(iri) => Ok(Predicate::NamedNode(NamedNode::new(iri).map_err(tdb_err)?)),
        TdbTerm::BlankNode(_) | TdbTerm::Literal { .. } => Err(OxirsError::Store(
            "TDB2 returned a non-IRI in predicate position (corrupt data)".to_string(),
        )),
    }
}

/// Decode a TDB [`TdbTerm`] into an `oxirs-core` [`Object`].
fn tdb_to_object(term: &TdbTerm) -> Result<Object> {
    match term {
        TdbTerm::Iri(iri) => Ok(Object::NamedNode(NamedNode::new(iri).map_err(tdb_err)?)),
        TdbTerm::BlankNode(id) => Ok(Object::BlankNode(BlankNode::new(id).map_err(tdb_err)?)),
        TdbTerm::Literal {
            value,
            language,
            datatype,
        } => Ok(Object::Literal(tdb_to_literal(value, language, datatype)?)),
    }
}

/// Rebuild an `oxirs-core` [`Literal`] from TDB's decomposed literal fields.
fn tdb_to_literal(
    value: &str,
    language: &Option<String>,
    datatype: &Option<String>,
) -> Result<Literal> {
    if let Some(lang) = language {
        Literal::new_lang(value, lang.as_str()).map_err(tdb_err)
    } else if let Some(dt) = datatype {
        Ok(Literal::new_typed(
            value,
            NamedNode::new(dt).map_err(tdb_err)?,
        ))
    } else {
        Ok(Literal::new_simple_literal(value))
    }
}

/// Decode a TDB [`TdbGraphName`] into an `oxirs-core` [`GraphName`].
fn tdb_to_graph(graph: &TdbGraphName) -> Result<GraphName> {
    match graph {
        TdbGraphName::DefaultGraph => Ok(GraphName::DefaultGraph),
        TdbGraphName::Named(TdbTerm::Iri(iri)) => {
            Ok(GraphName::NamedNode(NamedNode::new(iri).map_err(tdb_err)?))
        }
        TdbGraphName::Named(TdbTerm::BlankNode(id)) => {
            Ok(GraphName::BlankNode(BlankNode::new(id).map_err(tdb_err)?))
        }
        TdbGraphName::Named(TdbTerm::Literal { .. }) => Err(OxirsError::Store(
            "TDB2 returned a literal as a graph name (corrupt data)".to_string(),
        )),
    }
}

/// Decode a whole [`QuadResult`] into an `oxirs-core` [`Quad`].
fn quad_result_to_core(qr: QuadResult) -> Result<Quad> {
    let subject = tdb_to_subject(&qr.subject)?;
    let predicate = tdb_to_predicate(&qr.predicate)?;
    let object = tdb_to_object(&qr.object)?;
    let graph = tdb_to_graph(&qr.graph)?;
    Ok(Quad::new(subject, predicate, object, graph))
}

// ---------------------------------------------------------------------------
// Pattern conversion (find / scan direction). A `None` component is a wildcard;
// a bound variable is also treated as a wildcard.
// ---------------------------------------------------------------------------

/// An owned graph pattern selector. It owns the graph term (if any) so the
/// borrowed [`GraphTarget`] handed to the scan can outlive a temporary.
enum GraphSelector {
    Any,
    Default,
    Named(TdbTerm),
}

impl GraphSelector {
    /// Borrow this selector as a [`GraphTarget`] for a scan call.
    fn as_target(&self) -> GraphTarget<'_> {
        match self {
            GraphSelector::Any => GraphTarget::AnyGraph,
            GraphSelector::Default => GraphTarget::DefaultGraph,
            GraphSelector::Named(term) => GraphTarget::Named(term),
        }
    }
}

/// Build the graph selector for a scan from an optional `oxirs-core` graph
/// pattern (`None` = any graph, a bound variable = any graph).
fn graph_selector(graph: Option<&GraphName>) -> Result<GraphSelector> {
    match graph {
        None | Some(GraphName::Variable(_)) => Ok(GraphSelector::Any),
        Some(GraphName::DefaultGraph) => Ok(GraphSelector::Default),
        Some(GraphName::NamedNode(n)) => Ok(GraphSelector::Named(TdbTerm::iri(n.as_str()))),
        Some(GraphName::BlankNode(b)) => Ok(GraphSelector::Named(TdbTerm::blank_node(b.as_str()))),
    }
}

/// Convert an optional subject pattern to an optional TDB term (variable =
/// wildcard).
fn subject_pattern(subject: Option<&Subject>) -> Result<Option<TdbTerm>> {
    match subject {
        None | Some(Subject::Variable(_)) => Ok(None),
        Some(other) => subject_to_tdb(other).map(Some),
    }
}

/// Convert an optional predicate pattern to an optional TDB term.
fn predicate_pattern(predicate: Option<&Predicate>) -> Result<Option<TdbTerm>> {
    match predicate {
        None | Some(Predicate::Variable(_)) => Ok(None),
        Some(other) => predicate_to_tdb(other).map(Some),
    }
}

/// Convert an optional object pattern to an optional TDB term.
fn object_pattern(object: Option<&Object>) -> Result<Option<TdbTerm>> {
    match object {
        None | Some(Object::Variable(_)) => Ok(None),
        Some(other) => object_to_tdb(other).map(Some),
    }
}

/// Convert an `oxirs-core` [`QueryResult`](CoreQueryResult) produced by the
/// shared [`QueryEngine`] into the [`OxirsQueryResults`] the `Store` trait
/// returns.
fn core_query_result_to_oxirs(result: CoreQueryResult) -> OxirsQueryResults {
    match result {
        CoreQueryResult::Select {
            variables,
            bindings,
        } => {
            let rows = bindings
                .into_iter()
                .map(|bindings| VariableBinding { bindings })
                .collect();
            OxirsQueryResults::from_bindings(rows, variables)
        }
        CoreQueryResult::Ask(value) => OxirsQueryResults::from_boolean(value),
        CoreQueryResult::Construct(triples) => {
            OxirsQueryResults::from_graph(triples.into_iter().map(Quad::from_triple).collect())
        }
    }
}

// ---------------------------------------------------------------------------
// oxirs_core::Store trait implementation
// ---------------------------------------------------------------------------

impl CoreStore for TdbStoreAdapter {
    fn insert_quad(&self, quad: Quad) -> Result<bool> {
        let graph = graph_to_tdb(quad.graph_name())?;
        let subject = subject_to_tdb(quad.subject())?;
        let predicate = predicate_to_tdb(quad.predicate())?;
        let object = object_to_tdb(quad.object())?;
        let mut store = self.write()?;
        let inserted = store
            .insert_quad(graph.as_ref(), &subject, &predicate, &object)
            .map_err(tdb_err)?;
        if inserted {
            self.maybe_sync(&store)?;
        }
        Ok(inserted)
    }

    fn remove_quad(&self, quad: &Quad) -> Result<bool> {
        let graph = graph_to_tdb(quad.graph_name())?;
        let subject = subject_to_tdb(quad.subject())?;
        let predicate = predicate_to_tdb(quad.predicate())?;
        let object = object_to_tdb(quad.object())?;
        let mut store = self.write()?;
        let removed = store
            .delete_quad(graph.as_ref(), &subject, &predicate, &object)
            .map_err(tdb_err)?;
        if removed {
            self.maybe_sync(&store)?;
        }
        Ok(removed)
    }

    fn find_quads(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
        graph_name: Option<&GraphName>,
    ) -> Result<Vec<Quad>> {
        let s = subject_pattern(subject)?;
        let p = predicate_pattern(predicate)?;
        let o = object_pattern(object)?;
        let selector = graph_selector(graph_name)?;
        let store = self.read()?;
        let results = store
            .scan_quads(selector.as_target(), s.as_ref(), p.as_ref(), o.as_ref())
            .map_err(tdb_err)?;
        results.into_iter().map(quad_result_to_core).collect()
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn len(&self) -> Result<usize> {
        Ok(self.read()?.dataset_len())
    }

    fn is_empty(&self) -> Result<bool> {
        Ok(self.read()?.dataset_len() == 0)
    }

    fn query(&self, sparql: &str) -> Result<OxirsQueryResults> {
        // Run the shared SPARQL engine over this store; it pulls data through
        // the `find_quads` seam implemented above, so it works against TDB with
        // no engine-side changes.
        let engine = QueryEngine::new();
        let result = engine
            .query(sparql, self)
            .map_err(|e| OxirsError::Query(format!("TDB2 query failed: {e}")))?;
        Ok(core_query_result_to_oxirs(result))
    }

    fn prepare_query(&self, sparql: &str) -> Result<PreparedQuery> {
        // The TDB backend has no `StorageBackend` handle to bind a prepared
        // query to, so this returns the backend-less form (which fails loudly on
        // `exec`). Callers wanting execution should use [`query`](Self::query),
        // which the Fuseki query path uses.
        Ok(PreparedQuery::new(sparql.to_string()))
    }

    /// Bulk insert: one write lock, one fsync for the whole batch (instead of a
    /// per-quad fsync loop). All quads are converted up front so a malformed
    /// quad fails loudly before any write happens.
    fn bulk_insert_quads(&self, quads: Vec<Quad>) -> Result<usize> {
        let mut converted = Vec::with_capacity(quads.len());
        for quad in &quads {
            let graph = graph_to_tdb(quad.graph_name())?;
            let subject = subject_to_tdb(quad.subject())?;
            let predicate = predicate_to_tdb(quad.predicate())?;
            let object = object_to_tdb(quad.object())?;
            converted.push((graph, subject, predicate, object));
        }
        let mut store = self.write()?;
        let mut inserted = 0usize;
        for (graph, subject, predicate, object) in &converted {
            if store
                .insert_quad(graph.as_ref(), subject, predicate, object)
                .map_err(tdb_err)?
            {
                inserted += 1;
            }
        }
        // Single durable sync for the whole batch, regardless of sync policy.
        store.sync().map_err(tdb_err)?;
        Ok(inserted)
    }

    /// Streaming scan: pulls decoded quads one at a time from TDB's lazy
    /// `quad_iter` and never materializes the whole matching set.
    fn for_each_quad(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
        graph_name: Option<&GraphName>,
        f: &mut dyn FnMut(Quad),
    ) -> Result<()> {
        let s = subject_pattern(subject)?;
        let p = predicate_pattern(predicate)?;
        let o = object_pattern(object)?;
        let selector = graph_selector(graph_name)?;
        let store = self.read()?;
        for item in store
            .quad_iter(selector.as_target(), s.as_ref(), p.as_ref(), o.as_ref())
            .map_err(tdb_err)?
        {
            let qr = item.map_err(tdb_err)?;
            f(quad_result_to_core(qr)?);
        }
        Ok(())
    }

    /// Clear every graph (default + named) and return the number of statements
    /// removed, then fsync so the empty state is durable.
    fn clear_all(&self) -> Result<usize> {
        let mut store = self.write()?;
        let removed = store.dataset_len();
        store.clear().map_err(tdb_err)?;
        store.sync().map_err(tdb_err)?;
        Ok(removed)
    }

    /// List the IRI-named graphs present in the dataset. TDB has no direct
    /// "list graphs" API, so this scans the named-graph quads and collects the
    /// distinct graph IRIs (blank-node graph labels are not `NamedNode`s and
    /// are skipped, matching the `Vec<NamedNode>` contract).
    fn named_graphs(&self) -> Result<Vec<NamedNode>> {
        let store = self.read()?;
        let mut seen = std::collections::BTreeSet::new();
        for item in store
            .quad_iter(GraphTarget::AnyGraph, None, None, None)
            .map_err(tdb_err)?
        {
            let qr = item.map_err(tdb_err)?;
            if let TdbGraphName::Named(TdbTerm::Iri(iri)) = &qr.graph {
                seen.insert(iri.clone());
            }
        }
        seen.into_iter()
            .map(|iri| NamedNode::new(&iri).map_err(tdb_err))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::Store as CoreStore;

    /// A fresh, unique temp directory for one test's TDB store.
    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("oxirs-fuseki-tdb-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn iri_subject(iri: &str) -> Subject {
        Subject::NamedNode(NamedNode::new(iri).expect("iri"))
    }

    fn iri_predicate(iri: &str) -> Predicate {
        Predicate::NamedNode(NamedNode::new(iri).expect("iri"))
    }

    fn named_graph(iri: &str) -> GraphName {
        GraphName::NamedNode(NamedNode::new(iri).expect("iri"))
    }

    #[test]
    fn roundtrip_across_default_and_named_graphs() {
        let dir = temp_dir("roundtrip");

        // Insert into the default graph and two named graphs through the
        // `oxirs_core::Store` trait.
        {
            let adapter = TdbStoreAdapter::open(&dir).expect("open adapter");
            let store: &dyn CoreStore = &adapter;

            store
                .insert_quad(Quad::new(
                    iri_subject("http://ex/s1"),
                    iri_predicate("http://ex/p"),
                    Object::NamedNode(NamedNode::new("http://ex/o1").expect("iri")),
                    GraphName::DefaultGraph,
                ))
                .expect("insert default");
            store
                .insert_quad(Quad::new(
                    iri_subject("http://ex/s2"),
                    iri_predicate("http://ex/p"),
                    Object::Literal(Literal::new_simple_literal("hello")),
                    named_graph("http://ex/g1"),
                ))
                .expect("insert g1");
            store
                .insert_quad(Quad::new(
                    iri_subject("http://ex/s3"),
                    iri_predicate("http://ex/p"),
                    Object::NamedNode(NamedNode::new("http://ex/o3").expect("iri")),
                    named_graph("http://ex/g2"),
                ))
                .expect("insert g2");

            assert_eq!(store.len().expect("len"), 3);
            // Drop syncs to disk (sync_on_drop is on by default).
        }

        // Reopen the same on-disk store and verify every quad survived.
        {
            let adapter = TdbStoreAdapter::open(&dir).expect("reopen adapter");
            let store: &dyn CoreStore = &adapter;

            assert_eq!(store.len().expect("len"), 3);

            let all = store.find_quads(None, None, None, None).expect("find all");
            assert_eq!(all.len(), 3);

            // Default-graph-only scan returns exactly the default quad.
            let default_quads = store
                .find_quads(None, None, None, Some(&GraphName::DefaultGraph))
                .expect("find default");
            assert_eq!(default_quads.len(), 1);
            assert_eq!(default_quads[0].graph_name(), &GraphName::DefaultGraph);

            // Named-graph scan is isolated to that graph.
            let g1 = named_graph("http://ex/g1");
            let g1_quads = store
                .find_quads(None, None, None, Some(&g1))
                .expect("find g1");
            assert_eq!(g1_quads.len(), 1);
            assert_eq!(g1_quads[0].graph_name(), &g1);
            assert_eq!(
                g1_quads[0].object(),
                &Object::Literal(Literal::new_simple_literal("hello"))
            );

            // named_graphs() lists both named graphs (not the default graph).
            let mut graphs: Vec<String> = store
                .named_graphs()
                .expect("named graphs")
                .into_iter()
                .map(|n| n.as_str().to_string())
                .collect();
            graphs.sort();
            assert_eq!(graphs, vec!["http://ex/g1", "http://ex/g2"]);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn term_conversion_fidelity() {
        let dir = temp_dir("fidelity");
        let adapter = TdbStoreAdapter::open(&dir).expect("open adapter");
        let store: &dyn CoreStore = &adapter;

        let subj = iri_subject("http://ex/s");
        let pred = iri_predicate("http://ex/p");

        // 1. Typed literal (xsd:integer).
        let typed = Literal::new_typed(
            "42",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").expect("iri"),
        );
        // 2. Language-tagged literal.
        let lang = Literal::new_lang("bonjour", "fr").expect("lang literal");
        // 3. Simple literal (implicit xsd:string).
        let simple = Literal::new_simple_literal("plain");
        // 4. Blank-node object.
        let blank = BlankNode::new("b42").expect("blank");

        store
            .insert_quad(Quad::new(
                subj.clone(),
                pred.clone(),
                Object::Literal(typed.clone()),
                named_graph("http://ex/g"),
            ))
            .expect("insert typed");
        store
            .insert_quad(Quad::new(
                subj.clone(),
                pred.clone(),
                Object::Literal(lang.clone()),
                GraphName::DefaultGraph,
            ))
            .expect("insert lang");
        store
            .insert_quad(Quad::new(
                subj.clone(),
                pred.clone(),
                Object::Literal(simple.clone()),
                GraphName::DefaultGraph,
            ))
            .expect("insert simple");
        store
            .insert_quad(Quad::new(
                subj.clone(),
                pred.clone(),
                Object::BlankNode(blank.clone()),
                GraphName::DefaultGraph,
            ))
            .expect("insert blank");

        // Typed literal round-trips with its datatype and lives in the named graph.
        let g = named_graph("http://ex/g");
        let typed_back = store
            .find_quads(None, None, None, Some(&g))
            .expect("find typed");
        assert_eq!(typed_back.len(), 1);
        match typed_back[0].object() {
            Object::Literal(l) => {
                assert_eq!(l.value(), "42");
                assert_eq!(
                    l.datatype().as_str(),
                    "http://www.w3.org/2001/XMLSchema#integer"
                );
                assert!(l.language().is_none());
            }
            other => panic!("expected typed literal, got {other:?}"),
        }

        // The default graph holds the lang, simple, and blank objects.
        let default_objs: Vec<Object> = store
            .find_quads(None, None, None, Some(&GraphName::DefaultGraph))
            .expect("find default")
            .into_iter()
            .map(|q| q.object().clone())
            .collect();
        assert_eq!(default_objs.len(), 3);
        assert!(default_objs.contains(&Object::Literal(lang)));
        assert!(default_objs.contains(&Object::Literal(simple)));
        assert!(default_objs.contains(&Object::BlankNode(blank)));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn remove_and_bulk_insert() {
        let dir = temp_dir("bulk");
        let adapter = TdbStoreAdapter::open(&dir).expect("open adapter");
        let store: &dyn CoreStore = &adapter;

        let quads: Vec<Quad> = (0..5)
            .map(|i| {
                Quad::new(
                    iri_subject(&format!("http://ex/s{i}")),
                    iri_predicate("http://ex/p"),
                    Object::NamedNode(NamedNode::new(format!("http://ex/o{i}")).expect("iri")),
                    if i % 2 == 0 {
                        GraphName::DefaultGraph
                    } else {
                        named_graph("http://ex/g")
                    },
                )
            })
            .collect();

        let inserted = store.bulk_insert_quads(quads.clone()).expect("bulk insert");
        assert_eq!(inserted, 5);
        assert_eq!(store.len().expect("len"), 5);

        // Re-inserting the same batch inserts nothing new.
        let again = store
            .bulk_insert_quads(quads.clone())
            .expect("bulk insert 2");
        assert_eq!(again, 0);
        assert_eq!(store.len().expect("len"), 5);

        // Remove one named-graph quad.
        assert!(store.remove_quad(&quads[1]).expect("remove"));
        assert_eq!(store.len().expect("len"), 4);
        assert!(!store.remove_quad(&quads[1]).expect("remove missing"));

        // clear_all empties everything and reports the prior count.
        let removed = store.clear_all().expect("clear all");
        assert_eq!(removed, 4);
        assert!(store.is_empty().expect("is empty"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sparql_query_over_tdb_backend() {
        let dir = temp_dir("sparql");
        let adapter = TdbStoreAdapter::open(&dir).expect("open adapter");
        let store: &dyn CoreStore = &adapter;

        for i in 0..3 {
            store
                .insert_quad(Quad::new(
                    iri_subject(&format!("http://ex/s{i}")),
                    iri_predicate("http://ex/p"),
                    Object::Literal(Literal::new_simple_literal(format!("v{i}"))),
                    GraphName::DefaultGraph,
                ))
                .expect("insert");
        }

        // The SPARQL engine runs over the adapter's `find_quads` seam.
        let results = store
            .query("SELECT ?s ?o WHERE { ?s <http://ex/p> ?o }")
            .expect("query");
        assert_eq!(results.len(), 3);
    }
}
