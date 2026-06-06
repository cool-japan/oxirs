//! Transaction context, state, isolation, and version entry types for MVCC.

use crate::error::{Result, TdbError};
use std::collections::HashSet;

/// Transaction identifier — monotonically increasing u64
pub type TxId = u64;

/// Sentinel value: no transaction (committed data visible to all)
pub const TX_ID_COMMITTED: TxId = 0;

/// Error variants specific to MVCC
#[derive(Debug, thiserror::Error)]
pub enum MvccError {
    /// A concurrent transaction modified a key this transaction read (SSI conflict)
    #[error(
        "Serialization conflict on key: snapshot tx={snapshot_tx}, conflicting_writer={writer_tx}"
    )]
    SerializationConflict {
        /// Snapshot TxId of the failing transaction
        snapshot_tx: TxId,
        /// TxId that committed a conflicting write
        writer_tx: TxId,
    },

    /// Write-write conflict: another transaction already wrote this key
    #[error("Write-write conflict on key: existing writer tx={existing_tx}")]
    WriteWriteConflict {
        /// TxId that holds the conflicting write lock
        existing_tx: TxId,
    },

    /// The transaction is not known (may have been cleaned up or never existed)
    #[error("Unknown transaction: {tx_id}")]
    UnknownTransaction {
        /// The unknown transaction ID
        tx_id: TxId,
    },

    /// The transaction is not in Active state
    #[error("Transaction {tx_id} is not active (state: {state})")]
    TransactionNotActive {
        /// Transaction ID
        tx_id: TxId,
        /// Current state name
        state: &'static str,
    },

    /// Deadlock detected — transaction was chosen as victim
    #[error("Deadlock detected; transaction {victim_tx} aborted as victim")]
    DeadlockVictim {
        /// The transaction chosen as deadlock victim
        victim_tx: TxId,
    },

    /// Lock acquisition failed (timeout or conflict)
    #[error("Lock acquisition failed for transaction {tx_id}")]
    LockFailed {
        /// Transaction that failed to acquire the lock
        tx_id: TxId,
    },
}

impl From<MvccError> for TdbError {
    fn from(e: MvccError) -> Self {
        TdbError::Transaction(e.to_string())
    }
}

/// Transaction isolation level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationLevel {
    /// Reads see the latest committed version at the time of each read
    ReadCommitted,
    /// Reads always see the snapshot taken at transaction start
    RepeatableRead,
    /// Full serializability via SSI; may abort with `SerializationConflict`
    Serializable,
}

/// Lifecycle state of a transaction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionState {
    /// Transaction is executing
    Active,
    /// Transaction has committed successfully
    Committed,
    /// Transaction was rolled back explicitly
    RolledBack,
    /// Transaction was aborted (by system, deadlock, or conflict)
    Aborted,
}

impl TransactionState {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::Active => "Active",
            Self::Committed => "Committed",
            Self::RolledBack => "RolledBack",
            Self::Aborted => "Aborted",
        }
    }
}

/// A key that has been versioned (opaque byte key)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VersionedKey(pub Vec<u8>);

impl VersionedKey {
    /// Create a new versioned key from a byte slice
    pub fn new(key: &[u8]) -> Self {
        Self(key.to_vec())
    }
}

/// A single version of a value stored under a key
#[derive(Debug, Clone)]
pub struct VersionedEntry {
    /// The TxId that created (wrote/updated) this version
    pub created_tx_id: TxId,
    /// The TxId that deleted this version, or `None` if still live
    pub deleted_tx_id: Option<TxId>,
    /// The serialized value bytes
    pub value: Vec<u8>,
}

impl VersionedEntry {
    /// Returns `true` if this version is a tombstone (logical delete)
    pub fn is_tombstone(&self) -> bool {
        self.value.is_empty() && self.deleted_tx_id.is_some()
    }

    /// Check whether this version is visible to a reader with `reader_snapshot_tx`.
    ///
    /// Visibility rule:
    ///   - The version must have been created by a committed tx <= reader_snapshot_tx
    ///   - The version must NOT have been deleted by a committed tx <= reader_snapshot_tx
    pub fn is_visible_at(&self, reader_snapshot_tx: TxId, committed: &HashSet<TxId>) -> bool {
        // created_tx_id == TX_ID_COMMITTED means bootstrapped / always-committed
        let creator_committed = self.created_tx_id == TX_ID_COMMITTED
            || (self.created_tx_id <= reader_snapshot_tx
                && committed.contains(&self.created_tx_id));

        if !creator_committed {
            return false;
        }

        match self.deleted_tx_id {
            None => true,
            Some(del_tx) => {
                // Deleted version is invisible if the deleter committed before/at snapshot
                !(del_tx <= reader_snapshot_tx && committed.contains(&del_tx))
            }
        }
    }
}

/// Per-transaction context, protected behind a `Mutex` when shared
pub struct TransactionContext {
    /// Unique transaction identifier
    pub tx_id: TxId,
    /// The TxId watermark at which the snapshot was taken
    pub start_tx_id: TxId,
    /// Isolation level for this transaction
    pub isolation_level: IsolationLevel,
    /// Current lifecycle state
    pub state: TransactionState,
    /// Keys written (or deleted) in this transaction (for SSI tracking)
    pub write_set: HashSet<VersionedKey>,
    /// Keys read in this transaction (for SSI anti-dependency tracking)
    pub read_set: HashSet<VersionedKey>,
}

impl TransactionContext {
    pub(crate) fn new(tx_id: TxId, start_tx_id: TxId, isolation_level: IsolationLevel) -> Self {
        Self {
            tx_id,
            start_tx_id,
            isolation_level,
            state: TransactionState::Active,
            write_set: HashSet::new(),
            read_set: HashSet::new(),
        }
    }

    pub(crate) fn ensure_active(&self) -> Result<()> {
        if self.state != TransactionState::Active {
            Err(MvccError::TransactionNotActive {
                tx_id: self.tx_id,
                state: self.state.as_str(),
            }
            .into())
        } else {
            Ok(())
        }
    }
}
