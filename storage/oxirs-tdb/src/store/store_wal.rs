//! WAL-integrated writes and crash-recovery replay for the TDB store (F3).
//!
//! Every mutating store operation is bracketed in a WAL transaction
//! (`Begin -> DataOp -> Commit`) so that a *committed* write survives a crash
//! that happens *before the next* [`sync`](crate::store::TdbStore::sync). On
//! reopen the committed operations found in the WAL are replayed on top of the
//! last checkpoint (the superblock), reconstructing the exact committed state.
//!
//! # Why operation-level (logical) redo rather than full-page images
//!
//! The [`LogRecord::DataOp`] records written here carry a compact,
//! operation-level redo payload (the interned triple/quad and its terms), not
//! full 4 KiB page before/after images. That is a deliberate design choice for
//! this storage engine, forced by three properties of the surrounding code:
//!
//! 1. **Per-commit durability is required.** The crash-recovery contract
//!    demands that committed writes survive a reopen *without* an intervening
//!    `sync()`. That forces the committing transaction's effect to be in the
//!    WAL at commit time — page-image logging would therefore have to log every
//!    page a transaction dirtied, on every commit.
//! 2. **The catalog lives only in the superblock.** B+Tree roots, counts and
//!    the next dictionary id are persisted only by `sync()`. Operation-level
//!    redo sidesteps this entirely: replaying the logical operation on top of
//!    the reconstructed catalog rebuilds the in-memory indexes directly, so no
//!    per-page catalog surgery is needed during recovery.
//! 3. **Bounded WAL size and memory.** [`WriteAheadLog`] retains every appended
//!    entry in an in-memory buffer, and [`crate::storage::FileManager`] fsyncs
//!    on every page write. Full-page-image-per-commit logging of the existing
//!    10k-triple round-trip workloads would produce ~gigabyte WALs and an
//!    equally large in-memory buffer. Operation-level redo keeps each record at
//!    ~100 bytes, so the WAL stays small and the suite stays fast.
//!
//! Recovery is **redo-only** and transaction-atomic: each store operation is a
//! complete, independently-committed transaction, so there is never a partial
//! transaction to undo — a crash either finds a `Commit` for a `DataOp` (replay
//! it) or does not (skip it). The full-page-image [`LogRecord::Update`] path and
//! the lower-level [`crate::recovery::RecoveryManager`] remain available for
//! physical page redo and are unaffected by this module.

use crate::dictionary::Term;
use crate::error::{Result, TdbError};
use crate::index::{Quad, QuadIndexes, Triple};
use crate::store::store_impl::TdbStore;
use crate::transaction::wal::{LogRecord, TxnId};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// A committed, replayable store mutation (operation-level redo).
///
/// Serialized into the opaque payload of a [`LogRecord::DataOp`]. Default-graph
/// quad operations are recorded as the corresponding triple operation (they act
/// on the same triple indexes); only *named-graph* operations carry a graph
/// term.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum StoreOp {
    /// Insert a default-graph triple.
    InsertTriple {
        /// Subject term.
        subject: Term,
        /// Predicate term.
        predicate: Term,
        /// Object term.
        object: Term,
    },
    /// Delete a default-graph triple.
    DeleteTriple {
        /// Subject term.
        subject: Term,
        /// Predicate term.
        predicate: Term,
        /// Object term.
        object: Term,
    },
    /// Insert a named-graph quad.
    InsertQuad {
        /// Named-graph term.
        graph: Term,
        /// Subject term.
        subject: Term,
        /// Predicate term.
        predicate: Term,
        /// Object term.
        object: Term,
    },
    /// Delete a named-graph quad.
    DeleteQuad {
        /// Named-graph term.
        graph: Term,
        /// Subject term.
        subject: Term,
        /// Predicate term.
        predicate: Term,
        /// Object term.
        object: Term,
    },
}

/// Serialize a [`StoreOp`] into a WAL `DataOp` payload.
pub(crate) fn encode_store_op(op: &StoreOp) -> Result<Vec<u8>> {
    oxicode::serde::encode_to_vec(op, oxicode::config::standard())
        .map_err(|e| TdbError::Serialization(e.to_string()))
}

/// Deserialize a [`StoreOp`] from a WAL `DataOp` payload.
fn decode_store_op(bytes: &[u8]) -> Result<StoreOp> {
    oxicode::serde::decode_from_slice(bytes, oxicode::config::standard())
        .map(|(op, _)| op)
        .map_err(|e| TdbError::Deserialization(e.to_string()))
}

impl TdbStore {
    /// Allocate the next WAL transaction id for this store handle.
    ///
    /// Seeded past the highest id seen during recovery (see
    /// [`TdbStore::recover_from_wal`]) so ids never collide with un-truncated
    /// records left by a previous, crashed session.
    fn next_wal_txn(&mut self) -> TxnId {
        self.wal_txn_counter += 1;
        TxnId::new(self.wal_txn_counter)
    }

    /// Log a single committed mutating operation to the WAL as a self-contained
    /// transaction (`Begin -> DataOp -> Commit`).
    ///
    /// A no-op when WAL logging is disabled. The commit is fsynced only when
    /// [`TdbConfig::wal_sync_on_commit`](crate::store::TdbConfig) is set;
    /// otherwise the append is buffered and made durable by the next
    /// `sync()`/checkpoint (or process exit via `Drop`).
    pub(crate) fn wal_log_op(&mut self, op: StoreOp) -> Result<()> {
        if !self.config.enable_wal {
            return Ok(());
        }
        let txn_id = self.next_wal_txn();
        let payload = encode_store_op(&op)?;
        let wal = self.txn_manager.wal();
        wal.append(LogRecord::Begin { txn_id })?;
        wal.append(LogRecord::DataOp { txn_id, payload })?;
        wal.append(LogRecord::Commit { txn_id })?;
        if self.config.wal_sync_on_commit {
            wal.flush()?;
        }
        Ok(())
    }

    /// Log a batch of committed mutating operations as a *single* WAL
    /// transaction (`Begin -> DataOp* -> Commit`).
    ///
    /// Used by the bulk write path so a bulk load is one atomic transaction with
    /// no per-element fsync; the caller flushes/checkpoints afterwards.
    pub(crate) fn wal_log_batch(&mut self, ops: &[StoreOp]) -> Result<()> {
        if !self.config.enable_wal || ops.is_empty() {
            return Ok(());
        }
        let txn_id = self.next_wal_txn();
        let wal = self.txn_manager.wal();
        wal.append(LogRecord::Begin { txn_id })?;
        for op in ops {
            let payload = encode_store_op(op)?;
            wal.append(LogRecord::DataOp { txn_id, payload })?;
        }
        wal.append(LogRecord::Commit { txn_id })?;
        Ok(())
    }

    /// Replay committed WAL operations on top of the reconstructed catalog.
    ///
    /// Called on open, *after* the indexes/dictionary have been rebuilt from the
    /// superblock and the bloom filter repopulated, and *before* the store
    /// serves any read. Only `DataOp` records whose transaction has a matching
    /// `Commit` are applied (torn/uncommitted operations are skipped). Replay is
    /// idempotent: re-applying an operation already reflected in the catalog is a
    /// no-op (encode is lookup-or-insert; index insert/delete are idempotent),
    /// so a crash between the superblock write and the WAL truncation in
    /// `sync()` recovers correctly.
    ///
    /// Returns the number of operations replayed.
    pub(crate) fn recover_from_wal(&mut self) -> Result<usize> {
        let entries = self.txn_manager.wal().all_entries();
        if entries.is_empty() {
            return Ok(0);
        }

        // A transaction's operations are only redone if it committed.
        let committed: HashSet<TxnId> = entries
            .iter()
            .filter_map(|e| match &e.record {
                LogRecord::Commit { txn_id } => Some(*txn_id),
                _ => None,
            })
            .collect();

        let mut max_txn = self.wal_txn_counter;
        let mut applied = 0usize;
        for entry in &entries {
            match &entry.record {
                LogRecord::Begin { txn_id }
                | LogRecord::Commit { txn_id }
                | LogRecord::Abort { txn_id } => {
                    max_txn = max_txn.max(txn_id.as_u64());
                }
                LogRecord::DataOp { txn_id, payload } => {
                    max_txn = max_txn.max(txn_id.as_u64());
                    if committed.contains(txn_id) {
                        let op = decode_store_op(payload)?;
                        self.apply_store_op(op)?;
                        applied += 1;
                    }
                }
                // Full-page `Update` and `Checkpoint` records are not part of
                // the operation-level redo path handled here.
                _ => {}
            }
        }

        // Continue the transaction-id space past anything left in the WAL so a
        // subsequent write cannot reuse a pre-crash id in the same segment.
        self.wal_txn_counter = max_txn;
        Ok(applied)
    }

    /// Apply one decoded [`StoreOp`] during recovery (redo).
    fn apply_store_op(&mut self, op: StoreOp) -> Result<()> {
        match op {
            StoreOp::InsertTriple {
                subject,
                predicate,
                object,
            } => self.replay_insert_triple(&subject, &predicate, &object),
            StoreOp::DeleteTriple {
                subject,
                predicate,
                object,
            } => self.replay_delete_triple(&subject, &predicate, &object),
            StoreOp::InsertQuad {
                graph,
                subject,
                predicate,
                object,
            } => self.replay_insert_named_quad(&graph, &subject, &predicate, &object),
            StoreOp::DeleteQuad {
                graph,
                subject,
                predicate,
                object,
            } => self.replay_delete_named_quad(&graph, &subject, &predicate, &object),
        }
    }

    /// Redo a default-graph triple insert (index + dictionary + bloom + count).
    fn replay_insert_triple(&mut self, s: &Term, p: &Term, o: &Term) -> Result<()> {
        let s_id = self.dictionary.encode(s)?;
        let p_id = self.dictionary.encode(p)?;
        let o_id = self.dictionary.encode(o)?;
        let triple = Triple::new(s_id, p_id, o_id);
        let is_new = self.indexes.insert(triple)?;
        if let Some(ref mut bloom) = self.bloom_filter {
            bloom.insert(&triple);
        }
        if is_new {
            self.triple_count += 1;
        }
        Ok(())
    }

    /// Redo a default-graph triple delete. A term absent from the dictionary
    /// means the triple was never present, so there is nothing to redo.
    fn replay_delete_triple(&mut self, s: &Term, p: &Term, o: &Term) -> Result<()> {
        let (s_id, p_id, o_id) = match (
            self.dictionary.lookup(s)?,
            self.dictionary.lookup(p)?,
            self.dictionary.lookup(o)?,
        ) {
            (Some(s_id), Some(p_id), Some(o_id)) => (s_id, p_id, o_id),
            _ => return Ok(()),
        };
        let triple = Triple::new(s_id, p_id, o_id);
        if self.indexes.delete(&triple)? {
            self.triple_count = self.triple_count.saturating_sub(1);
        }
        Ok(())
    }

    /// Redo a named-graph quad insert, materializing the quad indexes on demand
    /// if the current handle opened without them (so committed named-graph
    /// writes are never lost even when reopened with quad support disabled).
    fn replay_insert_named_quad(
        &mut self,
        graph: &Term,
        s: &Term,
        p: &Term,
        o: &Term,
    ) -> Result<()> {
        let g_id = self.dictionary.encode(graph)?;
        let s_id = self.dictionary.encode(s)?;
        let p_id = self.dictionary.encode(p)?;
        let o_id = self.dictionary.encode(o)?;
        if self.quad_indexes.is_none() {
            self.quad_indexes = Some(QuadIndexes::new(self.buffer_pool.clone()));
            self.quads_writable = true;
        }
        let quad_indexes = self.quad_indexes.as_mut().ok_or_else(|| {
            TdbError::Other("quad indexes unexpectedly absent during WAL replay".to_string())
        })?;
        let quad = Quad::new(g_id, s_id, p_id, o_id);
        if quad_indexes.insert(quad)? {
            self.quad_count += 1;
        }
        Ok(())
    }

    /// Redo a named-graph quad delete. A term (or the graph) absent from the
    /// dictionary means the quad was never present, so there is nothing to redo.
    fn replay_delete_named_quad(
        &mut self,
        graph: &Term,
        s: &Term,
        p: &Term,
        o: &Term,
    ) -> Result<()> {
        let (g_id, s_id, p_id, o_id) = match (
            self.dictionary.lookup(graph)?,
            self.dictionary.lookup(s)?,
            self.dictionary.lookup(p)?,
            self.dictionary.lookup(o)?,
        ) {
            (Some(g_id), Some(s_id), Some(p_id), Some(o_id)) => (g_id, s_id, p_id, o_id),
            _ => return Ok(()),
        };
        if let Some(quad_indexes) = self.quad_indexes.as_mut() {
            let quad = Quad::new(g_id, s_id, p_id, o_id);
            if quad_indexes.delete(quad)? {
                self.quad_count = self.quad_count.saturating_sub(1);
            }
        }
        Ok(())
    }
}
