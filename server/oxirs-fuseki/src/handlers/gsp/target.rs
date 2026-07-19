//! Graph Target Determination and Access

use super::types::{GraphTarget, GspError};
use oxirs_core::model::{GraphName, NamedNode, Quad, Triple};
use oxirs_core::Store;
use std::sync::Arc;

/// Represents access to a specific graph in the store
pub struct GraphAccess {
    target: GraphTarget,
    exists: bool,
}

impl GraphAccess {
    /// Create a new graph access
    pub fn new(target: GraphTarget, store: &dyn Store) -> Self {
        let exists = Self::check_exists(&target, store);
        Self { target, exists }
    }

    /// Check if the graph exists in the store
    fn check_exists(target: &GraphTarget, store: &dyn Store) -> bool {
        match target {
            GraphTarget::Default => {
                // Default graph always exists
                true
            }
            GraphTarget::Union => {
                // Union graph is a virtual view, always exists
                true
            }
            GraphTarget::Named(uri) => {
                // Check if any quads exist in this named graph
                if let Ok(node) = NamedNode::new(uri) {
                    let graph_name = GraphName::NamedNode(node);
                    // Try to get at least one quad from this graph
                    store
                        .find_quads(None, None, None, Some(&graph_name))
                        .map(|quads| !quads.is_empty())
                        .unwrap_or(false)
                } else {
                    false
                }
            }
        }
    }

    /// Check if this graph exists
    pub fn exists(&self) -> bool {
        self.exists
    }

    /// Get the graph target
    pub fn target(&self) -> &GraphTarget {
        &self.target
    }

    /// Get label for logging/errors
    pub fn label(&self) -> String {
        self.target.label()
    }

    /// Check if this graph is writable
    pub fn is_writable(&self) -> bool {
        self.target.is_writable()
    }

    /// Get all triples from this graph
    pub fn get_triples(&self, store: &dyn Store) -> Result<Vec<Triple>, GspError> {
        match &self.target {
            GraphTarget::Default => {
                // Get all triples from default graph
                let quads = store
                    .find_quads(None, None, None, Some(&GraphName::DefaultGraph))
                    .map_err(|e| GspError::StoreError(e.to_string()))?;

                let triples: Vec<Triple> = quads
                    .into_iter()
                    .map(|quad| {
                        Triple::new(
                            quad.subject().clone(),
                            quad.predicate().clone(),
                            quad.object().clone(),
                        )
                    })
                    .collect();
                Ok(triples)
            }
            GraphTarget::Union => {
                // Get all triples from all graphs (union)
                let quads = store
                    .find_quads(None, None, None, None)
                    .map_err(|e| GspError::StoreError(e.to_string()))?;

                let triples: Vec<Triple> = quads
                    .into_iter()
                    .map(|quad| {
                        Triple::new(
                            quad.subject().clone(),
                            quad.predicate().clone(),
                            quad.object().clone(),
                        )
                    })
                    .collect();
                Ok(triples)
            }
            GraphTarget::Named(uri) => {
                // Get all triples from named graph
                let node = NamedNode::new(uri)
                    .map_err(|e| GspError::BadRequest(format!("Invalid graph URI: {}", e)))?;
                let graph_name = GraphName::NamedNode(node);

                let quads = store
                    .find_quads(None, None, None, Some(&graph_name))
                    .map_err(|e| GspError::StoreError(e.to_string()))?;

                let triples: Vec<Triple> = quads
                    .into_iter()
                    .map(|quad| {
                        Triple::new(
                            quad.subject().clone(),
                            quad.predicate().clone(),
                            quad.object().clone(),
                        )
                    })
                    .collect();
                Ok(triples)
            }
        }
    }

    /// Replace all triples in this graph.
    ///
    /// This is documented (and used by GSP PUT) as an atomic replace. Because
    /// the `Store` trait exposes no multi-statement transaction, atomicity is
    /// achieved with explicit rollback: the existing graph contents are
    /// snapshotted first, then removed, then the new triples are inserted; if
    /// ANY step fails midway, every partial change is undone so the graph is
    /// restored to its exact pre-call state before the error is returned. A
    /// failed PUT therefore never leaves a half-cleared / half-populated graph.
    pub fn replace_triples(
        &self,
        store: &dyn Store,
        triples: Vec<Triple>,
    ) -> Result<usize, GspError> {
        if !self.is_writable() {
            return Err(GspError::MethodNotAllowed(format!(
                "Cannot write to {}",
                self.label()
            )));
        }

        let graph_name = match &self.target {
            GraphTarget::Default => GraphName::DefaultGraph,
            GraphTarget::Named(uri) => {
                let node = NamedNode::new(uri)
                    .map_err(|e| GspError::BadRequest(format!("Invalid graph URI: {}", e)))?;
                GraphName::NamedNode(node)
            }
            GraphTarget::Union => {
                return Err(GspError::MethodNotAllowed(
                    "Cannot write to union graph".to_string(),
                ))
            }
        };

        Self::replace_graph_atomic(store, &graph_name, triples)
    }

    /// Atomically replace the contents of `graph_name` with `triples`, rolling
    /// back all partial changes on any store error.
    fn replace_graph_atomic(
        store: &dyn Store,
        graph_name: &GraphName,
        triples: Vec<Triple>,
    ) -> Result<usize, GspError> {
        // 1. Snapshot the current graph contents (needed for rollback).
        let existing = store
            .find_quads(None, None, None, Some(graph_name))
            .map_err(|e| GspError::StoreError(e.to_string()))?;

        // 2. Remove existing quads. On failure, re-insert the ones already
        //    removed and abort — the graph is back to its original state.
        for (i, quad) in existing.iter().enumerate() {
            if let Err(e) = store.remove_quad(quad) {
                for restored in &existing[..i] {
                    if let Err(re) = store.insert_quad(restored.clone()) {
                        tracing::warn!("GSP replace rollback (re-insert) failed: {re}");
                    }
                }
                return Err(GspError::StoreError(format!(
                    "replace failed while clearing graph (changes rolled back): {e}"
                )));
            }
        }

        // 3. Insert the new triples. On failure, remove the ones already
        //    inserted and re-insert the full original contents.
        for (idx, triple) in triples.iter().enumerate() {
            let quad = Quad::new(
                triple.subject().clone(),
                triple.predicate().clone(),
                triple.object().clone(),
                graph_name.clone(),
            );
            if let Err(e) = store.insert_quad(quad) {
                for undo in &triples[..idx] {
                    let undo_quad = Quad::new(
                        undo.subject().clone(),
                        undo.predicate().clone(),
                        undo.object().clone(),
                        graph_name.clone(),
                    );
                    if let Err(re) = store.remove_quad(&undo_quad) {
                        tracing::warn!("GSP replace rollback (remove) failed: {re}");
                    }
                }
                for restored in &existing {
                    if let Err(re) = store.insert_quad(restored.clone()) {
                        tracing::warn!("GSP replace rollback (restore) failed: {re}");
                    }
                }
                return Err(GspError::StoreError(format!(
                    "replace failed while inserting (changes rolled back): {e}"
                )));
            }
        }

        Ok(triples.len())
    }

    /// Add triples to this graph
    pub fn add_triples(&self, store: &dyn Store, triples: Vec<Triple>) -> Result<usize, GspError> {
        if !self.is_writable() {
            return Err(GspError::MethodNotAllowed(format!(
                "Cannot write to {}",
                self.label()
            )));
        }

        let graph_name = match &self.target {
            GraphTarget::Default => GraphName::DefaultGraph,
            GraphTarget::Named(uri) => {
                let node = NamedNode::new(uri)
                    .map_err(|e| GspError::BadRequest(format!("Invalid graph URI: {}", e)))?;
                GraphName::NamedNode(node)
            }
            GraphTarget::Union => {
                return Err(GspError::MethodNotAllowed(
                    "Cannot write to union graph".to_string(),
                ))
            }
        };

        // Accumulate the target-graph quads and insert the whole batch through
        // the single batched ingest path instead of a per-triple loop.
        let count = triples.len();
        let batch: Vec<Quad> = triples
            .into_iter()
            .map(|triple| {
                Quad::new(
                    triple.subject().clone(),
                    triple.predicate().clone(),
                    triple.object().clone(),
                    graph_name.clone(),
                )
            })
            .collect();
        crate::store::bulk_insert_quads(store, batch)
            .map_err(|e| GspError::StoreError(e.to_string()))?;

        Ok(count)
    }

    /// Delete all triples from this graph
    pub fn delete_graph(&self, store: &dyn Store) -> Result<usize, GspError> {
        if !self.is_writable() {
            return Err(GspError::MethodNotAllowed(format!(
                "Cannot delete {}",
                self.label()
            )));
        }

        let graph_name = match &self.target {
            GraphTarget::Default => GraphName::DefaultGraph,
            GraphTarget::Named(uri) => {
                let node = NamedNode::new(uri)
                    .map_err(|e| GspError::BadRequest(format!("Invalid graph URI: {}", e)))?;
                GraphName::NamedNode(node)
            }
            GraphTarget::Union => {
                return Err(GspError::MethodNotAllowed(
                    "Cannot delete union graph".to_string(),
                ))
            }
        };

        // Get all quads in this graph
        let existing = store
            .find_quads(None, None, None, Some(&graph_name))
            .map_err(|e| GspError::StoreError(e.to_string()))?;

        let count = existing.len();

        // Delete all quads
        for quad in existing {
            store
                .remove_quad(&quad)
                .map_err(|e| GspError::StoreError(e.to_string()))?;
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::Literal;
    use oxirs_core::rdf_store::ConcreteStore;

    fn setup_test_store() -> ConcreteStore {
        let store = ConcreteStore::new().unwrap();

        // Add some test data to default graph
        let s1 = NamedNode::new("http://example.org/s1").unwrap();
        let p1 = NamedNode::new("http://example.org/p1").unwrap();
        let o1 = Literal::new_simple_literal("value1");
        let triple1 = Triple::new(s1, p1, o1);
        store.insert_triple(triple1).unwrap();

        store
    }

    #[test]
    fn test_graph_access_default() {
        let store = setup_test_store();
        let target = GraphTarget::Default;
        let access = GraphAccess::new(target, &store);

        assert!(access.exists());
        assert!(access.is_writable());
    }

    #[test]
    fn test_graph_access_union() {
        let store = setup_test_store();
        let target = GraphTarget::Union;
        let access = GraphAccess::new(target, &store);

        assert!(access.exists());
        assert!(!access.is_writable());
    }

    #[test]
    fn test_get_triples_default_graph() {
        let store = setup_test_store();
        let target = GraphTarget::Default;
        let access = GraphAccess::new(target, &store);

        let triples = access.get_triples(&store).unwrap();
        assert!(!triples.is_empty());
    }

    #[test]
    fn test_replace_triples() {
        let store = setup_test_store();
        let target = GraphTarget::Default;
        let access = GraphAccess::new(target, &store);

        let s = NamedNode::new("http://example.org/new").unwrap();
        let p = NamedNode::new("http://example.org/pred").unwrap();
        let o = Literal::new_simple_literal("new value");
        let triple = Triple::new(s, p, o);

        let count = access.replace_triples(&store, vec![triple]).unwrap();
        assert_eq!(count, 1);

        let triples = access.get_triples(&store).unwrap();
        assert_eq!(triples.len(), 1);
    }

    /// Regression: a store error partway through a GSP PUT replace must roll
    /// back so the graph is restored to its exact pre-call contents, never left
    /// half-cleared / half-populated.
    #[test]
    fn test_replace_triples_atomic_rollback_on_insert_error() {
        use oxirs_core::model::{Object, Predicate, Subject};
        use oxirs_core::rdf_store::{ConcreteStore, OxirsQueryResults, PreparedQuery};
        use oxirs_core::OxirsError;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct FailingStore {
            inner: ConcreteStore,
            insert_calls: AtomicUsize,
            fail_at: usize,
        }

        impl Store for FailingStore {
            fn insert_quad(&self, quad: Quad) -> oxirs_core::Result<bool> {
                let n = self.insert_calls.fetch_add(1, Ordering::SeqCst);
                if n == self.fail_at {
                    return Err(OxirsError::Store("injected insert failure".to_string()));
                }
                self.inner.insert_quad(quad)
            }
            fn remove_quad(&self, quad: &Quad) -> oxirs_core::Result<bool> {
                self.inner.remove_quad(quad)
            }
            fn find_quads(
                &self,
                s: Option<&Subject>,
                p: Option<&Predicate>,
                o: Option<&Object>,
                g: Option<&GraphName>,
            ) -> oxirs_core::Result<Vec<Quad>> {
                self.inner.find_quads(s, p, o, g)
            }
            fn is_ready(&self) -> bool {
                self.inner.is_ready()
            }
            fn len(&self) -> oxirs_core::Result<usize> {
                self.inner.len()
            }
            fn is_empty(&self) -> oxirs_core::Result<bool> {
                self.inner.is_empty()
            }
            fn query(&self, sparql: &str) -> oxirs_core::Result<OxirsQueryResults> {
                self.inner.query(sparql)
            }
            fn prepare_query(&self, sparql: &str) -> oxirs_core::Result<PreparedQuery> {
                self.inner.prepare_query(sparql)
            }
        }

        // Seed the default graph with two original triples.
        let inner = ConcreteStore::new().unwrap();
        for i in 0..2 {
            let s = NamedNode::new(format!("http://example.org/orig{i}")).unwrap();
            let p = NamedNode::new("http://example.org/p").unwrap();
            let o = Literal::new_simple_literal(format!("v{i}"));
            inner.insert_triple(Triple::new(s, p, o)).unwrap();
        }
        // Fail on the second NEW insert (index 1).
        let failing = FailingStore {
            inner,
            insert_calls: AtomicUsize::new(0),
            fail_at: 1,
        };

        let new_triples: Vec<Triple> = (0..3)
            .map(|i| {
                let s = NamedNode::new(format!("http://example.org/new{i}")).unwrap();
                let p = NamedNode::new("http://example.org/p").unwrap();
                let o = Literal::new_simple_literal(format!("n{i}"));
                Triple::new(s, p, o)
            })
            .collect();

        let access = GraphAccess::new(GraphTarget::Default, &failing);
        let result = access.replace_triples(&failing, new_triples);
        assert!(result.is_err(), "replace must fail when an insert errors");

        // Rollback must have restored exactly the original two triples.
        let remaining = access.get_triples(&failing).unwrap();
        assert_eq!(
            remaining.len(),
            2,
            "graph must be restored to original contents after rollback"
        );
        assert!(
            remaining
                .iter()
                .all(|t| t.subject().to_string().contains("orig")),
            "only the original triples should remain after rollback"
        );
    }
}
