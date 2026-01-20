//! Implement oxirs_core::Store trait for fuseki Store

use crate::store::Store;
use oxirs_core::model::{GraphName, Quad, Triple};
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
}
