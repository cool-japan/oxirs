//! Backend selection: build the concrete RDF store for a dataset from its
//! configured [`StoreType`].
//!
//! Fuseki datasets are configured either in-memory ([`StoreType::InMemory`],
//! backed by [`oxirs_core::RdfStore`]) or on-disk TDB2 ([`StoreType::TDB2`],
//! backed by [`TdbStoreAdapter`]). Both satisfy the same `oxirs_core::Store`
//! trait, so the rest of the server — every handler, the query engine, GSP,
//! `/upload` — is backend-agnostic and needs no changes.
//!
//! An unknown storage-type string is a hard configuration error: the factory
//! [fails loud](StoreFactory::store_type_from_str) rather than silently falling
//! back to a volatile in-memory store (which would lose a user's data on
//! restart without warning).

use std::sync::{Arc, RwLock};

use oxirs_core::{RdfStore, Store as CoreStore};

use crate::dataset_manager::StoreType;
use crate::error::{FusekiError, FusekiResult};
use crate::store::tdb_adapter::TdbStoreAdapter;

/// Constructs backend stores for datasets according to their [`StoreType`].
pub struct StoreFactory;

impl StoreFactory {
    /// Parse a textual dataset storage type (e.g. from a config file's
    /// `dataset_type` / an API request's `storage_type`) into a [`StoreType`].
    ///
    /// Recognised (case-insensitive): `""`/`mem`/`memory`/`in-memory` →
    /// [`StoreType::InMemory`]; `tdb2`/`tdb` → [`StoreType::TDB2`]. Any other
    /// value is a configuration error — the factory never silently defaults an
    /// unknown type to memory.
    pub fn store_type_from_str(kind: &str) -> FusekiResult<StoreType> {
        match kind.trim().to_ascii_lowercase().as_str() {
            "" | "mem" | "memory" | "in-memory" | "inmemory" => Ok(StoreType::InMemory),
            "tdb2" | "tdb" => Ok(StoreType::TDB2),
            other => Err(FusekiError::configuration(format!(
                "Unsupported dataset storage type '{other}'; expected 'mem' (in-memory) \
                 or 'tdb2' (on-disk TDB2)"
            ))),
        }
    }

    /// Build the backend store for `store_type` at `location`.
    ///
    /// - [`StoreType::InMemory`]: an [`RdfStore`]; an empty `location` is a pure
    ///   volatile store, a non-empty one is an append-only persistent file.
    /// - [`StoreType::TDB2`]: a durable [`TdbStoreAdapter`] rooted at the
    ///   `location` directory (required — a TDB2 dataset must have a path).
    /// - [`StoreType::External`]: rejected — an external SPARQL endpoint is not
    ///   a local backing store.
    ///
    /// The returned handle is `Arc<RwLock<dyn Store>>`, exactly the shape
    /// Fuseki's [`Store`](crate::store::Store) already holds for its default and
    /// named datasets, so callers drop it straight in with no further glue.
    pub fn create_backend(
        store_type: &StoreType,
        location: &str,
    ) -> FusekiResult<Arc<RwLock<dyn CoreStore>>> {
        match store_type {
            StoreType::InMemory => {
                let store = if location.is_empty() {
                    RdfStore::new()
                } else {
                    RdfStore::open(location)
                }
                .map_err(|e| FusekiError::store(format!("Failed to open in-memory store: {e}")))?;
                let backend: Arc<RwLock<dyn CoreStore>> = Arc::new(RwLock::new(store));
                Ok(backend)
            }
            StoreType::TDB2 => {
                if location.is_empty() {
                    return Err(FusekiError::configuration(
                        "A TDB2 dataset requires a non-empty 'location' directory".to_string(),
                    ));
                }
                let adapter = TdbStoreAdapter::open(location).map_err(|e| {
                    FusekiError::store(format!("Failed to open TDB2 store at '{location}': {e}"))
                })?;
                let backend: Arc<RwLock<dyn CoreStore>> = Arc::new(RwLock::new(adapter));
                Ok(backend)
            }
            StoreType::External(endpoint) => Err(FusekiError::configuration(format!(
                "External store backend '{endpoint}' cannot be opened as a local dataset store"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::{GraphName, NamedNode, Object, Predicate, Quad, Subject};
    use oxirs_core::Store as CoreStore;

    fn temp_base(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "oxirs-fuseki-factory-{tag}-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp base");
        dir
    }

    fn sample_quad(i: usize, graph: GraphName) -> Quad {
        Quad::new(
            Subject::NamedNode(NamedNode::new(format!("http://ex/s{i}")).expect("iri")),
            Predicate::NamedNode(NamedNode::new("http://ex/p").expect("iri")),
            Object::NamedNode(NamedNode::new(format!("http://ex/o{i}")).expect("iri")),
            graph,
        )
    }

    #[test]
    fn unknown_storage_type_fails_loud() {
        let err = StoreFactory::store_type_from_str("rocksdb").expect_err("must reject");
        assert!(err.to_string().to_lowercase().contains("unsupported"));

        assert_eq!(
            StoreFactory::store_type_from_str("tdb2").expect("tdb2"),
            StoreType::TDB2
        );
        assert_eq!(
            StoreFactory::store_type_from_str("").expect("empty"),
            StoreType::InMemory
        );
        assert_eq!(
            StoreFactory::store_type_from_str("MEM").expect("mem"),
            StoreType::InMemory
        );
    }

    #[test]
    fn tdb2_dataset_requires_location() {
        // `Arc<RwLock<dyn Store>>` is not `Debug`, so match instead of `expect_err`.
        let err = match StoreFactory::create_backend(&StoreType::TDB2, "") {
            Ok(_) => panic!("TDB2 with an empty location must be rejected"),
            Err(e) => e,
        };
        assert!(err.to_string().to_lowercase().contains("location"));
    }

    #[test]
    fn tdb2_backend_persists_across_simulated_restart() {
        let base = temp_base("persist");
        let location = base.join("mydataset");
        let location_str = location.to_str().expect("utf-8 path").to_string();

        // First "process": create a TDB2-backed dataset backend via the factory,
        // write quads across the default and a named graph, then drop it (which
        // fsyncs — a clean shutdown).
        {
            let backend = StoreFactory::create_backend(&StoreType::TDB2, &location_str)
                .expect("open tdb2 backend");
            let guard = backend.read().expect("read lock");
            guard
                .insert_quad(sample_quad(0, GraphName::DefaultGraph))
                .expect("insert default");
            guard
                .insert_quad(sample_quad(
                    1,
                    GraphName::NamedNode(NamedNode::new("http://ex/g").expect("iri")),
                ))
                .expect("insert named");
            assert_eq!(guard.len().expect("len"), 2);
            drop(guard);
            // Backend Arc dropped here -> TdbStore dropped -> synced to disk.
        }

        // Second "process": reopen the same StoreType::TDB2 dataset at the same
        // location and confirm the data survived the restart.
        {
            let backend = StoreFactory::create_backend(&StoreType::TDB2, &location_str)
                .expect("reopen tdb2 backend");
            let guard = backend.read().expect("read lock");
            assert_eq!(guard.len().expect("len"), 2);

            let all = guard.find_quads(None, None, None, None).expect("find all");
            assert_eq!(all.len(), 2);

            let named = guard
                .find_quads(
                    None,
                    None,
                    None,
                    Some(&GraphName::NamedNode(
                        NamedNode::new("http://ex/g").expect("iri"),
                    )),
                )
                .expect("find named");
            assert_eq!(named.len(), 1);
        }

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn in_memory_backend_is_volatile_but_functional() {
        let backend =
            StoreFactory::create_backend(&StoreType::InMemory, "").expect("open mem backend");
        let guard = backend.read().expect("read lock");
        guard
            .insert_quad(sample_quad(0, GraphName::DefaultGraph))
            .expect("insert");
        assert_eq!(guard.len().expect("len"), 1);
    }
}
