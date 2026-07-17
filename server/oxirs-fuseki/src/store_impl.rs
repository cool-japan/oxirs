//! Implement oxirs_core::Store trait for fuseki Store

use crate::store::Store;
use oxirs_core::model::{GraphName, Object, Predicate, Quad, Subject, Triple};
use oxirs_core::rdf_store::{OxirsQueryResults, PreparedQuery};
use oxirs_core::{OxirsError, Result};

impl oxirs_core::Store for Store {
    fn insert_quad(&self, quad: Quad) -> Result<bool> {
        let store = self
            .default_store
            .write()
            .map_err(|e| OxirsError::Store(format!("Lock error: {}", e)))?;
        store.insert_quad(quad)
    }

    fn remove_quad(&self, quad: &Quad) -> Result<bool> {
        let store = self
            .default_store
            .write()
            .map_err(|e| OxirsError::Store(format!("Lock error: {}", e)))?;
        store.remove_quad(quad)
    }

    fn find_quads(
        &self,
        subject: Option<&oxirs_core::model::Subject>,
        predicate: Option<&oxirs_core::model::Predicate>,
        object: Option<&oxirs_core::model::Object>,
        graph: Option<&GraphName>,
    ) -> Result<Vec<Quad>> {
        let store = self
            .default_store
            .read()
            .map_err(|e| OxirsError::Store(format!("Lock error: {}", e)))?;
        store.find_quads(subject, predicate, object, graph)
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn len(&self) -> Result<usize> {
        let store = self
            .default_store
            .read()
            .map_err(|e| OxirsError::Store(format!("Lock error: {}", e)))?;
        store.len()
    }

    fn is_empty(&self) -> Result<bool> {
        let store = self
            .default_store
            .read()
            .map_err(|e| OxirsError::Store(format!("Lock error: {}", e)))?;
        store.is_empty()
    }

    fn query(&self, sparql: &str) -> Result<OxirsQueryResults> {
        let store = self
            .default_store
            .read()
            .map_err(|e| OxirsError::Store(format!("Lock error: {}", e)))?;
        store.query(sparql)
    }

    fn prepare_query(&self, sparql: &str) -> Result<PreparedQuery> {
        let store = self
            .default_store
            .read()
            .map_err(|e| OxirsError::Store(format!("Lock error: {}", e)))?;
        store.prepare_query(sparql)
    }

    /// Delegate the batch-ingest to the backend so durable backends (RdfStore's
    /// persistent mode, the TDB2 adapter) take a single write lock and a single
    /// fsync for the whole batch instead of a per-quad fsync loop.
    fn bulk_insert_quads(&self, quads: Vec<Quad>) -> Result<usize> {
        let store = self
            .default_store
            .write()
            .map_err(|e| OxirsError::Store(format!("Lock error: {}", e)))?;
        store.bulk_insert_quads(quads)
    }

    /// Delegate the streaming scan to the backend so large graph scans (GSP
    /// downloads) pull quads one at a time from the backend's native iterator
    /// instead of materializing the whole graph. The backend's read lock is
    /// held for the scan's duration, yielding a consistent snapshot.
    fn for_each_quad(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
        graph_name: Option<&GraphName>,
        f: &mut dyn FnMut(Quad),
    ) -> Result<()> {
        let store = self
            .default_store
            .read()
            .map_err(|e| OxirsError::Store(format!("Lock error: {}", e)))?;
        store.for_each_quad(subject, predicate, object, graph_name, f)
    }
}
