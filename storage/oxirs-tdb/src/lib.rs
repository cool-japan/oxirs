//! # OxiRS TDB
//!
//! MVCC layer and assembler grammar with TDB2 parity for persistent RDF storage.
//!
//! This crate provides advanced persistent storage capabilities for RDF data,
//! including multi-version concurrency control and transaction support.

use anyhow::Result;
use std::path::Path;

pub mod assembler;
pub mod btree;
pub mod mvcc;
pub mod nodes;
pub mod page;
pub mod storage;
pub mod transactions;
pub mod triple_store;
pub mod wal;

// Re-export main types for convenience
pub use mvcc::{TransactionId, Version};
pub use nodes::{NodeId, NodeTable, Term};
pub use triple_store::{Quad, Triple, TripleStore, TripleStoreConfig, TripleStoreStats};

/// TDB storage engine configuration
#[derive(Debug, Clone)]
pub struct TdbConfig {
    pub location: String,
    pub cache_size: usize,
    pub enable_transactions: bool,
    pub enable_mvcc: bool,
}

impl Default for TdbConfig {
    fn default() -> Self {
        Self {
            location: "./tdb".to_string(),
            cache_size: 1024 * 1024 * 100, // 100MB
            enable_transactions: true,
            enable_mvcc: true,
        }
    }
}

/// TDB storage engine - now wraps the TripleStore implementation
pub struct TdbStore {
    config: TdbConfig,
    triple_store: TripleStore,
}

impl TdbStore {
    /// Create a new TDB store
    pub fn new(config: TdbConfig) -> Result<Self> {
        let triple_store_config = TripleStoreConfig {
            storage_path: config.location.clone().into(),
            buffer_config: crate::page::BufferPoolConfig {
                max_pages: config.cache_size / crate::page::PAGE_SIZE,
                ..Default::default()
            },
            ..Default::default()
        };

        let triple_store = TripleStore::with_config(triple_store_config)?;

        Ok(Self {
            config,
            triple_store,
        })
    }

    /// Open an existing TDB store
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config = TdbConfig {
            location: path.as_ref().to_string_lossy().to_string(),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Begin a new transaction
    pub fn begin_transaction(&self) -> Result<Transaction> {
        let tx_id = self.triple_store.begin_transaction()?;
        Ok(Transaction::new(tx_id))
    }

    /// Begin a read-only transaction
    pub fn begin_read_transaction(&self) -> Result<Transaction> {
        let tx_id = self.triple_store.begin_read_transaction()?;
        Ok(Transaction::new(tx_id))
    }

    /// Commit a transaction
    pub fn commit_transaction(&self, transaction: Transaction) -> Result<Version> {
        self.triple_store.commit_transaction(transaction.tx_id)
    }

    /// Rollback a transaction
    pub fn rollback_transaction(&self, transaction: Transaction) -> Result<()> {
        self.triple_store.abort_transaction(transaction.tx_id)
    }

    /// Insert a triple
    pub fn insert_triple(&self, subject: &Term, predicate: &Term, object: &Term) -> Result<()> {
        let subject_id = self.triple_store.store_term(subject)?;
        let predicate_id = self.triple_store.store_term(predicate)?;
        let object_id = self.triple_store.store_term(object)?;

        let triple = Triple::new(subject_id, predicate_id, object_id);
        self.triple_store.insert_triple(&triple)
    }

    /// Insert a quad
    pub fn insert_quad(
        &self,
        subject: &Term,
        predicate: &Term,
        object: &Term,
        graph: Option<&Term>,
    ) -> Result<()> {
        let subject_id = self.triple_store.store_term(subject)?;
        let predicate_id = self.triple_store.store_term(predicate)?;
        let object_id = self.triple_store.store_term(object)?;
        let graph_id = if let Some(g) = graph {
            Some(self.triple_store.store_term(g)?)
        } else {
            Some(self.triple_store.default_graph())
        };

        let quad = Quad::new(subject_id, predicate_id, object_id, graph_id);
        self.triple_store.insert_quad(&quad)
    }

    /// Delete a triple
    pub fn delete_triple(&self, subject: &Term, predicate: &Term, object: &Term) -> Result<bool> {
        let subject_id = self.triple_store.get_node_id(subject)?.unwrap_or(0);
        let predicate_id = self.triple_store.get_node_id(predicate)?.unwrap_or(0);
        let object_id = self.triple_store.get_node_id(object)?.unwrap_or(0);

        if subject_id == 0 || predicate_id == 0 || object_id == 0 {
            return Ok(false); // Triple doesn't exist
        }

        let triple = Triple::new(subject_id, predicate_id, object_id);
        self.triple_store.delete_triple(&triple)
    }

    /// Query triples
    pub fn query_triples(
        &self,
        subject: Option<&Term>,
        predicate: Option<&Term>,
        object: Option<&Term>,
    ) -> Result<Vec<(Term, Term, Term)>> {
        let subject_id = if let Some(s) = subject {
            self.triple_store.get_node_id(s)?
        } else {
            None
        };
        let predicate_id = if let Some(p) = predicate {
            self.triple_store.get_node_id(p)?
        } else {
            None
        };
        let object_id = if let Some(o) = object {
            self.triple_store.get_node_id(o)?
        } else {
            None
        };

        let triples = self
            .triple_store
            .query_triples(subject_id, predicate_id, object_id)?;

        // Convert back to terms
        let mut result = Vec::new();
        for triple in triples {
            let subject_term = self
                .triple_store
                .get_term(triple.subject)?
                .unwrap_or_else(|| Term::iri("unknown"));
            let predicate_term = self
                .triple_store
                .get_term(triple.predicate)?
                .unwrap_or_else(|| Term::iri("unknown"));
            let object_term = self
                .triple_store
                .get_term(triple.object)?
                .unwrap_or_else(|| Term::iri("unknown"));

            result.push((subject_term, predicate_term, object_term));
        }

        Ok(result)
    }

    /// Get statistics
    pub fn get_stats(&self) -> Result<TripleStoreStats> {
        self.triple_store.get_stats()
    }

    /// Get the number of triples
    pub fn len(&self) -> Result<u64> {
        self.triple_store.len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> Result<bool> {
        self.triple_store.is_empty()
    }

    /// Compact the store
    pub fn compact(&self) -> Result<()> {
        self.triple_store.compact()
    }

    /// Clear all data
    pub fn clear(&self) -> Result<()> {
        self.triple_store.clear()
    }

    /// Access the underlying triple store
    pub fn triple_store(&self) -> &TripleStore {
        &self.triple_store
    }
}

/// Database transaction
pub struct Transaction {
    tx_id: TransactionId,
}

impl Transaction {
    fn new(tx_id: TransactionId) -> Self {
        Self { tx_id }
    }

    /// Get the transaction ID
    pub fn id(&self) -> TransactionId {
        self.tx_id
    }
}
