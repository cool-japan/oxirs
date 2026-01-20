//! Async I/O wrapper for RDF store operations
//!
//! This module provides async/await compatible interfaces for RDF store operations.
//!
//! # Examples
//!
//! ```no_run
//! use oxirs_core::rdf_store::{RdfStore, AsyncRdfStore};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let store = RdfStore::new()?;
//!     let async_store = AsyncRdfStore::new(store);
//!     
//!     // Use async operations
//!     let count = async_store.len_async().await?;
//!     println!("Store contains {} quads", count);
//!     Ok(())
//! }
//! ```

use super::{OxirsQueryResults, RdfStore};
use crate::model::{NamedNode, Quad};
use crate::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::task;

/// Async wrapper for RdfStore providing non-blocking I/O operations
pub struct AsyncRdfStore {
    store: Arc<RwLock<RdfStore>>,
}

impl AsyncRdfStore {
    /// Create a new async RDF store wrapper
    pub fn new(store: RdfStore) -> Self {
        Self {
            store: Arc::new(RwLock::new(store)),
        }
    }

    /// Get a read lock on the store
    pub async fn store(&self) -> tokio::sync::RwLockReadGuard<'_, RdfStore> {
        self.store.read().await
    }

    /// Get a write lock on the store
    pub async fn store_mut(&self) -> tokio::sync::RwLockWriteGuard<'_, RdfStore> {
        self.store.write().await
    }

    /// Execute SPARQL query asynchronously
    pub async fn query_async(&self, sparql: &str) -> Result<OxirsQueryResults> {
        let sparql = sparql.to_string();
        let store = self.store.clone();

        task::spawn_blocking(move || {
            let store = futures::executor::block_on(store.read());
            store.query(&sparql)
        })
        .await
        .map_err(|e| crate::OxirsError::Io(format!("Task error: {}", e)))?
    }

    /// Insert quad asynchronously
    pub async fn insert_quad_async(&self, quad: Quad) -> Result<bool> {
        let mut store = self.store.write().await;
        store.insert_quad(quad)
    }

    /// Remove quad asynchronously
    pub async fn remove_quad_async(&self, quad: &Quad) -> Result<bool> {
        let mut store = self.store.write().await;
        store.remove_quad(quad)
    }

    /// Get quad count asynchronously
    pub async fn len_async(&self) -> Result<usize> {
        let store = self.store.read().await;
        store.len()
    }

    /// Check if store is empty asynchronously
    pub async fn is_empty_async(&self) -> Result<bool> {
        let store = self.store.read().await;
        store.is_empty()
    }

    /// Clear store asynchronously
    pub async fn clear_async(&self) -> Result<()> {
        let mut store = self.store.write().await;
        store.clear()
    }

    /// Get all quads asynchronously
    pub async fn quads_async(&self) -> Result<Vec<Quad>> {
        let store = self.store.read().await;
        store.quads()
    }

    /// Get graph names asynchronously
    pub async fn graphs_async(&self) -> Result<Vec<NamedNode>> {
        let store = self.store.read().await;
        store.graphs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::*;

    #[tokio::test]
    async fn test_async_store_creation() {
        let store = RdfStore::new().unwrap();
        let _async_store = AsyncRdfStore::new(store);
    }

    #[tokio::test]
    async fn test_async_insert_and_query() {
        let store = RdfStore::new().unwrap();
        let async_store = AsyncRdfStore::new(store);

        let quad = Quad::new(
            NamedNode::new("http://example.org/s").unwrap(),
            NamedNode::new("http://example.org/p").unwrap(),
            NamedNode::new("http://example.org/o").unwrap(),
            GraphName::DefaultGraph,
        );

        async_store.insert_quad_async(quad).await.unwrap();

        let count = async_store.len_async().await.unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_async_query() {
        let store = RdfStore::new().unwrap();
        let async_store = AsyncRdfStore::new(store);

        let quad = Quad::new(
            NamedNode::new("http://example.org/s").unwrap(),
            NamedNode::new("http://example.org/p").unwrap(),
            NamedNode::new("http://example.org/o").unwrap(),
            GraphName::DefaultGraph,
        );

        async_store.insert_quad_async(quad).await.unwrap();

        let _results = async_store
            .query_async("SELECT * WHERE { ?s ?p ?o }")
            .await
            .unwrap();

        // Query executed successfully
    }
}
