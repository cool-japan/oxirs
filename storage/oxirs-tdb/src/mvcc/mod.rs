//! MVCC (Multi-Version Concurrency Control) Transaction Manager for OxiRS TDB
//!
//! Provides snapshot isolation, repeatable read, and full serializability
//! via Serializable Snapshot Isolation (SSI).
//!
//! ## Design
//! - Each transaction gets a unique monotonically increasing TxId
//! - Writes create new versions (never in-place updates)
//! - Reads see the snapshot at their `start_tx_id`
//! - Vacuum removes versions no longer visible to any active transaction
//!
//! ## Isolation Levels
//! - `ReadCommitted` — reads the latest committed version
//! - `RepeatableRead` — snapshot at transaction start (prevents phantom reads via gap locks)
//! - `Serializable` — full serializability via SSI conflict detection
//!
//! ## Module Layout
//!
//! - `transaction` — `TxId`, `MvccError`, `IsolationLevel`, `TransactionState`,
//!   `VersionedKey`, `VersionedEntry`, `TransactionContext`
//! - `snapshot` — `MvccStats` and snapshot-read helpers
//! - `gc` — version-store garbage collection (`vacuum_versions`)
//! - `mvcc_tests` — unit-test suite

pub mod deadlock;
pub mod gc;
pub mod mvcc_tests;
pub mod snapshot;
pub mod transaction;

use crate::error::{Result, TdbError};
use deadlock::DeadlockDetector;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

// Flat re-exports so external callers can use `mvcc::Foo` directly.
pub use snapshot::MvccStats;
pub use transaction::{
    IsolationLevel, MvccError, TransactionContext, TransactionState, TxId, VersionedEntry,
    VersionedKey, TX_ID_COMMITTED,
};

/// Core MVCC manager
///
/// Thread-safe; clone the `Arc<MvccManager>` to share across tasks/threads.
///
/// # Lock ordering (deadlock avoidance)
///
/// `MvccManager` holds three `std::sync::RwLock`s that are sometimes acquired
/// together. `std::sync::RwLock` read locks are NOT deadlock-immune once a writer
/// is queued, so every site that takes more than one of these locks at once MUST
/// acquire them in this fixed, total order:
///
/// 1. `active_txns`
/// 2. `committed_txns`
/// 3. `version_store`
///
/// (No site takes `active_txns` and `committed_txns` together, so their relative
/// order is only nominal; the load-bearing invariant is that `version_store` is
/// always acquired LAST.) Acquiring in a different order between two sites — as
/// `check_write_conflict` once did versus `stats` — allows an AB-BA deadlock and
/// is a bug. `terminate_transaction` never holds `version_store` and
/// `active_txns` simultaneously (it drops the former before taking the latter),
/// so it does not participate in this ordering.
pub struct MvccManager {
    /// Global transaction counter (next TxId to hand out)
    next_tx_id: AtomicU64,
    /// Active transactions: tx_id -> context
    active_txns: Arc<RwLock<HashMap<TxId, Arc<Mutex<TransactionContext>>>>>,
    /// Committed transaction IDs (used for visibility checks)
    committed_txns: Arc<RwLock<HashSet<TxId>>>,
    /// Version store: key -> versions ordered oldest-first
    version_store: Arc<RwLock<HashMap<Vec<u8>, VecDeque<VersionedEntry>>>>,
    /// Low watermark: all active transactions have start_tx_id >= watermark
    watermark: AtomicU64,
    /// Counters
    stat_committed: AtomicU64,
    stat_rolled_back: AtomicU64,
    stat_aborted: AtomicU64,
    stat_vacuumed: AtomicU64,
    stat_serial_conflicts: AtomicU64,
    stat_write_conflicts: AtomicU64,
    stat_deadlocks: AtomicU64,
    /// Deadlock detector
    deadlock_detector: Arc<deadlock::DeadlockDetector>,
}

impl MvccManager {
    /// Create a new `MvccManager`. TxId counter starts at 1.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            next_tx_id: AtomicU64::new(1),
            active_txns: Arc::new(RwLock::new(HashMap::new())),
            committed_txns: Arc::new(RwLock::new(HashSet::new())),
            version_store: Arc::new(RwLock::new(HashMap::new())),
            watermark: AtomicU64::new(0),
            stat_committed: AtomicU64::new(0),
            stat_rolled_back: AtomicU64::new(0),
            stat_aborted: AtomicU64::new(0),
            stat_vacuumed: AtomicU64::new(0),
            stat_serial_conflicts: AtomicU64::new(0),
            stat_write_conflicts: AtomicU64::new(0),
            stat_deadlocks: AtomicU64::new(0),
            deadlock_detector: DeadlockDetector::new(),
        })
    }

    // -------------------------------------------------------------------------
    // Transaction lifecycle
    // -------------------------------------------------------------------------

    /// Begin a new transaction and return its TxId.
    pub fn begin_transaction(&self, isolation: IsolationLevel) -> TxId {
        let tx_id = self.next_tx_id.fetch_add(1, Ordering::SeqCst);
        let start_tx_id = match isolation {
            IsolationLevel::ReadCommitted => tx_id,
            IsolationLevel::RepeatableRead | IsolationLevel::Serializable => tx_id,
        };

        let ctx = TransactionContext::new(tx_id, start_tx_id, isolation);
        let ctx_arc = Arc::new(Mutex::new(ctx));

        self.active_txns
            .write()
            .expect("active_txns write lock")
            .insert(tx_id, ctx_arc);

        self.update_watermark();
        tx_id
    }

    /// Commit a transaction. For Serializable level, performs SSI conflict check.
    pub fn commit(&self, tx_id: TxId) -> Result<()> {
        let ctx_arc = self.get_active_context(tx_id)?;

        {
            let mut ctx = ctx_arc.lock().expect("tx ctx lock");
            ctx.ensure_active()?;

            if ctx.isolation_level == IsolationLevel::Serializable {
                self.check_ssi_conflicts(&ctx)?;
            }

            ctx.state = TransactionState::Committed;
        }

        self.committed_txns
            .write()
            .expect("committed_txns write lock")
            .insert(tx_id);

        self.active_txns
            .write()
            .expect("active_txns write lock")
            .remove(&tx_id);

        self.deadlock_detector.remove_transaction(tx_id);
        self.stat_committed.fetch_add(1, Ordering::Relaxed);
        self.update_watermark();
        Ok(())
    }

    /// Rollback a transaction explicitly (user-initiated).
    pub fn rollback(&self, tx_id: TxId) -> Result<()> {
        self.terminate_transaction(tx_id, TransactionState::RolledBack)?;
        self.stat_rolled_back.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Abort a transaction (system-initiated, e.g., deadlock victim).
    pub fn abort(&self, tx_id: TxId) -> Result<()> {
        self.terminate_transaction(tx_id, TransactionState::Aborted)?;
        self.stat_aborted.fetch_add(1, Ordering::Relaxed);
        self.stat_rolled_back.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Read / Write / Delete
    // -------------------------------------------------------------------------

    /// Read the value of `key` as seen by `tx_id`.
    pub fn read(&self, tx_id: TxId, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let ctx_arc = self.get_active_context(tx_id)?;
        let (snapshot, isolation) = {
            let ctx = ctx_arc.lock().expect("tx ctx lock");
            ctx.ensure_active()?;
            let snapshot = match ctx.isolation_level {
                IsolationLevel::ReadCommitted => self.current_committed_watermark(),
                _ => ctx.start_tx_id,
            };
            (snapshot, ctx.isolation_level)
        };

        let result = self.snapshot_read(snapshot, key)?;

        if isolation == IsolationLevel::Serializable {
            let mut ctx = ctx_arc.lock().expect("tx ctx lock");
            ctx.read_set.insert(VersionedKey::new(key));
        }

        Ok(result)
    }

    /// Write (insert or update) a value under `key` within `tx_id`.
    pub fn write(&self, tx_id: TxId, key: &[u8], value: &[u8]) -> Result<()> {
        let ctx_arc = self.get_active_context(tx_id)?;
        let isolation = {
            let mut ctx = ctx_arc.lock().expect("tx ctx lock");
            ctx.ensure_active()?;
            ctx.write_set.insert(VersionedKey::new(key));
            ctx.isolation_level
        };

        if isolation != IsolationLevel::Serializable {
            self.check_write_conflict(tx_id, key)?;
        }

        let entry = VersionedEntry {
            created_tx_id: tx_id,
            deleted_tx_id: None,
            value: value.to_vec(),
        };

        let mut store = self
            .version_store
            .write()
            .expect("version_store write lock");
        store.entry(key.to_vec()).or_default().push_back(entry);
        Ok(())
    }

    /// Delete `key` within `tx_id` by marking the latest visible version as deleted.
    pub fn delete(&self, tx_id: TxId, key: &[u8]) -> Result<()> {
        let ctx_arc = self.get_active_context(tx_id)?;
        let snapshot = {
            let mut ctx = ctx_arc.lock().expect("tx ctx lock");
            ctx.ensure_active()?;
            ctx.write_set.insert(VersionedKey::new(key));
            ctx.start_tx_id
        };

        self.check_write_conflict(tx_id, key)?;

        let committed = self
            .committed_txns
            .read()
            .expect("committed_txns read lock");

        let mut store = self
            .version_store
            .write()
            .expect("version_store write lock");
        if let Some(versions) = store.get_mut(key) {
            for entry in versions.iter_mut().rev() {
                if entry.deleted_tx_id.is_none() && entry.is_visible_at(snapshot, &committed) {
                    entry.deleted_tx_id = Some(tx_id);
                    break;
                }
            }
        }
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Snapshot read (public, for explicit snapshot queries)
    // -------------------------------------------------------------------------

    /// Read the value of `key` at an explicit snapshot (identified by `snapshot_tx_id`).
    pub fn snapshot_read(&self, snapshot_tx_id: TxId, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let committed = self
            .committed_txns
            .read()
            .expect("committed_txns read lock");

        let store = self.version_store.read().expect("version_store read lock");
        let Some(versions) = store.get(key) else {
            return Ok(None);
        };

        for entry in versions.iter().rev() {
            if entry.is_visible_at(snapshot_tx_id, &committed) {
                return Ok(Some(entry.value.clone()));
            }
        }
        Ok(None)
    }

    // -------------------------------------------------------------------------
    // Garbage collection
    // -------------------------------------------------------------------------

    /// Remove old versions that are no longer visible to any active transaction.
    ///
    /// Returns the number of version entries removed.
    pub fn vacuum(&self) -> Result<usize> {
        let watermark = self.watermark.load(Ordering::Acquire);
        let committed = self
            .committed_txns
            .read()
            .expect("committed_txns read lock");

        let mut store = self
            .version_store
            .write()
            .expect("version_store write lock");
        let mut removed = 0usize;

        for versions in store.values_mut() {
            removed += gc::vacuum_versions(versions, watermark, &committed);
        }

        store.retain(|_, v| !v.is_empty());

        self.stat_vacuumed
            .fetch_add(removed as u64, Ordering::Relaxed);
        Ok(removed)
    }

    // -------------------------------------------------------------------------
    // Statistics
    // -------------------------------------------------------------------------

    /// Return a statistics snapshot.
    pub fn stats(&self) -> MvccStats {
        let active_txns = self.active_txns.read().expect("active_txns read lock");
        let version_store = self.version_store.read().expect("version_store read lock");
        let total_versions: usize = version_store.values().map(|v| v.len()).sum();

        MvccStats {
            total_begun: self.next_tx_id.load(Ordering::Relaxed).saturating_sub(1),
            total_committed: self.stat_committed.load(Ordering::Relaxed),
            total_rolled_back: self.stat_rolled_back.load(Ordering::Relaxed),
            total_aborted: self.stat_aborted.load(Ordering::Relaxed),
            active_count: active_txns.len(),
            total_versions,
            versions_vacuumed: self.stat_vacuumed.load(Ordering::Relaxed),
            serialization_conflicts: self.stat_serial_conflicts.load(Ordering::Relaxed),
            write_conflicts: self.stat_write_conflicts.load(Ordering::Relaxed),
            deadlocks_detected: self.stat_deadlocks.load(Ordering::Relaxed),
            watermark: self.watermark.load(Ordering::Acquire),
        }
    }

    /// Return the current low watermark.
    pub fn low_water_mark(&self) -> TxId {
        self.watermark.load(Ordering::Acquire)
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    fn get_active_context(&self, tx_id: TxId) -> Result<Arc<Mutex<TransactionContext>>> {
        self.active_txns
            .read()
            .expect("active_txns read lock")
            .get(&tx_id)
            .cloned()
            .ok_or_else(|| TdbError::from(MvccError::UnknownTransaction { tx_id }))
    }

    fn terminate_transaction(&self, tx_id: TxId, new_state: TransactionState) -> Result<()> {
        let ctx_arc = self.get_active_context(tx_id)?;
        {
            let mut ctx = ctx_arc.lock().expect("tx ctx lock");
            ctx.ensure_active()?;
            ctx.state = new_state;
        }

        let write_keys: Vec<Vec<u8>> = {
            let ctx = ctx_arc.lock().expect("tx ctx lock");
            ctx.write_set.iter().map(|k| k.0.clone()).collect()
        };

        {
            let mut store = self
                .version_store
                .write()
                .expect("version_store write lock");
            for key in &write_keys {
                if let Some(versions) = store.get_mut(key) {
                    versions.retain(|e| e.created_tx_id != tx_id);
                    for entry in versions.iter_mut() {
                        if entry.deleted_tx_id == Some(tx_id) {
                            entry.deleted_tx_id = None;
                        }
                    }
                }
            }
            store.retain(|_, v| !v.is_empty());
        }

        self.active_txns
            .write()
            .expect("active_txns write lock")
            .remove(&tx_id);

        self.deadlock_detector.remove_transaction(tx_id);
        self.update_watermark();
        Ok(())
    }

    /// Update the low-water mark: min(start_tx_id) across all active transactions.
    fn update_watermark(&self) {
        let active = self.active_txns.read().expect("active_txns read lock");
        let new_wm = active
            .values()
            .filter_map(|ctx_arc| ctx_arc.lock().ok().map(|ctx| ctx.start_tx_id))
            .min()
            .unwrap_or_else(|| self.next_tx_id.load(Ordering::Acquire));

        self.watermark.store(new_wm, Ordering::Release);
    }

    /// Return the highest committed TxId (used by ReadCommitted snapshot logic).
    fn current_committed_watermark(&self) -> TxId {
        let committed = self
            .committed_txns
            .read()
            .expect("committed_txns read lock");
        snapshot::committed_watermark(&committed)
    }

    /// Check for write-write conflict: another ACTIVE transaction wrote `key`.
    ///
    /// Acquires `active_txns` before `version_store` to honor the canonical lock
    /// order (see the module note on lock ordering); acquiring them in the
    /// opposite order here (as this function previously did) created an AB-BA
    /// deadlock risk against [`Self::stats`], which takes them in canonical order.
    fn check_write_conflict(&self, our_tx_id: TxId, key: &[u8]) -> Result<()> {
        let active = self.active_txns.read().expect("active_txns read lock");
        let store = self.version_store.read().expect("version_store read lock");

        if let Some(versions) = store.get(key) {
            for entry in versions.iter() {
                let creator = entry.created_tx_id;
                if creator != our_tx_id && active.contains_key(&creator) {
                    self.stat_write_conflicts.fetch_add(1, Ordering::Relaxed);
                    return Err(MvccError::WriteWriteConflict {
                        existing_tx: creator,
                    }
                    .into());
                }
            }
        }
        Ok(())
    }

    /// SSI conflict check on commit.
    fn check_ssi_conflicts(&self, ctx: &TransactionContext) -> Result<()> {
        let snapshot = ctx.start_tx_id;
        let committed = self
            .committed_txns
            .read()
            .expect("committed_txns read lock");
        let store = self.version_store.read().expect("version_store read lock");

        for read_key in &ctx.read_set {
            let Some(versions) = store.get(&read_key.0) else {
                continue;
            };
            for entry in versions.iter() {
                let creator = entry.created_tx_id;
                if creator != ctx.tx_id && creator > snapshot && committed.contains(&creator) {
                    self.stat_serial_conflicts.fetch_add(1, Ordering::Relaxed);
                    return Err(MvccError::SerializationConflict {
                        snapshot_tx: snapshot,
                        writer_tx: creator,
                    }
                    .into());
                }
            }
        }
        Ok(())
    }
}

impl Default for MvccManager {
    fn default() -> Self {
        Arc::try_unwrap(Self::new()).unwrap_or_else(|_| unreachable!())
    }
}
