//! Atomic batch write operations for TDB store.
//!
//! Provides `WriteBatch` for accumulating insert/delete/update operations
//! and applying them atomically.  Each batch produces a WAL entry on commit
//! and supports rollback on failure.  Concurrent compatible batches can be
//! merged before commit via `WriteBatch::merge`.

use std::collections::HashMap;

// ── Triple representation ───────────────────────────────────────────────────

/// A triple (subject, predicate, object) with optional named graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BatchTriple {
    /// Subject IRI or blank node.
    pub subject: String,
    /// Predicate IRI.
    pub predicate: String,
    /// Object value (IRI, blank node, or literal).
    pub object: String,
    /// Optional named graph IRI.
    pub graph: Option<String>,
}

impl BatchTriple {
    /// Create a new triple for the default graph.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            graph: None,
        }
    }

    /// Create a new quad (triple in a named graph).
    pub fn quad(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
        graph: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            graph: Some(graph.into()),
        }
    }
}

// ── Operation ───────────────────────────────────────────────────────────────

/// A single operation within a write batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchOperation {
    /// Insert a triple.
    Insert(BatchTriple),
    /// Delete a triple.
    Delete(BatchTriple),
}

impl BatchOperation {
    /// Return the triple affected by this operation.
    pub fn triple(&self) -> &BatchTriple {
        match self {
            BatchOperation::Insert(t) | BatchOperation::Delete(t) => t,
        }
    }

    /// Return `true` if this is an insert.
    pub fn is_insert(&self) -> bool {
        matches!(self, BatchOperation::Insert(_))
    }

    /// Return `true` if this is a delete.
    pub fn is_delete(&self) -> bool {
        matches!(self, BatchOperation::Delete(_))
    }
}

// ── Batch status ────────────────────────────────────────────────────────────

/// Lifecycle state of a write batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchStatus {
    /// The batch is open and accepting new operations.
    Pending,
    /// The batch has been committed successfully.
    Committed,
    /// The batch has been rolled back.
    RolledBack,
}

// ── WAL entry ───────────────────────────────────────────────────────────────

/// A write-ahead log entry produced by committing a batch.
#[derive(Debug, Clone)]
pub struct WalBatchEntry {
    /// Unique batch identifier.
    pub batch_id: u64,
    /// Serialised operations contained in this entry.
    pub operations: Vec<BatchOperation>,
    /// Simple FNV-1a checksum of the operations.
    pub checksum: u32,
}

// ── Batch configuration ─────────────────────────────────────────────────────

/// Configuration for write batch limits.
#[derive(Debug, Clone)]
pub struct WriteBatchConfig {
    /// Maximum number of operations allowed per batch (0 = unlimited).
    pub max_batch_size: usize,
}

impl Default for WriteBatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 100_000,
        }
    }
}

impl WriteBatchConfig {
    /// Create a config with the given max batch size.
    pub fn with_max_size(max_batch_size: usize) -> Self {
        Self { max_batch_size }
    }

    /// Create a config with no size limit.
    pub fn unlimited() -> Self {
        Self { max_batch_size: 0 }
    }
}

// ── Batch error ─────────────────────────────────────────────────────────────

/// Errors that can occur during batch operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchError {
    /// The batch size limit has been reached.
    BatchSizeLimitReached {
        /// Current number of operations.
        current: usize,
        /// Maximum allowed.
        max: usize,
    },
    /// The batch is not in the expected status.
    InvalidStatus {
        /// Expected status.
        expected: BatchStatus,
        /// Actual status.
        actual: BatchStatus,
    },
    /// Merge conflict: overlapping operations on the same triple.
    MergeConflict {
        /// Description of the conflicting triple.
        description: String,
    },
    /// Generic error.
    Other(String),
}

impl std::fmt::Display for BatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchError::BatchSizeLimitReached { current, max } => {
                write!(f, "batch size limit reached: {current}/{max}")
            }
            BatchError::InvalidStatus { expected, actual } => {
                write!(
                    f,
                    "invalid batch status: expected {expected:?}, got {actual:?}"
                )
            }
            BatchError::MergeConflict { description } => {
                write!(f, "merge conflict: {description}")
            }
            BatchError::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for BatchError {}

/// Result alias for batch operations.
pub type BatchResult<T> = std::result::Result<T, BatchError>;

// ── Batch statistics ────────────────────────────────────────────────────────

/// Statistics snapshot of a write batch.
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Number of insert operations.
    pub inserts: usize,
    /// Number of delete operations.
    pub deletes: usize,
    /// Current status.
    pub status: Option<BatchStatus>,
}

// ── WriteBatch ──────────────────────────────────────────────────────────────

/// An atomic batch of write operations against the TDB store.
///
/// Operations are accumulated via [`insert`](Self::insert) and
/// [`delete`](Self::delete) calls, then applied atomically via
/// [`commit`](Self::commit).
pub struct WriteBatch {
    batch_id: u64,
    operations: Vec<BatchOperation>,
    status: BatchStatus,
    config: WriteBatchConfig,
    /// WAL entries produced by committed batches.
    wal_entries: Vec<WalBatchEntry>,
    /// Monotonically increasing batch id generator.
    next_batch_id: u64,
}

impl WriteBatch {
    /// Create a new empty write batch with default config.
    pub fn new(batch_id: u64) -> Self {
        Self {
            batch_id,
            operations: Vec::new(),
            status: BatchStatus::Pending,
            config: WriteBatchConfig::default(),
            wal_entries: Vec::new(),
            next_batch_id: batch_id + 1,
        }
    }

    /// Create a new write batch with custom config.
    pub fn with_config(batch_id: u64, config: WriteBatchConfig) -> Self {
        Self {
            batch_id,
            operations: Vec::new(),
            status: BatchStatus::Pending,
            config,
            wal_entries: Vec::new(),
            next_batch_id: batch_id + 1,
        }
    }

    /// Return the batch identifier.
    pub fn id(&self) -> u64 {
        self.batch_id
    }

    /// Return the current status.
    pub fn status(&self) -> BatchStatus {
        self.status
    }

    /// Return the number of operations in the batch.
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Return `true` if the batch contains no operations.
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Return a snapshot of batch statistics.
    pub fn stats(&self) -> BatchStats {
        let inserts = self.operations.iter().filter(|o| o.is_insert()).count();
        let deletes = self.operations.iter().filter(|o| o.is_delete()).count();
        BatchStats {
            inserts,
            deletes,
            status: Some(self.status),
        }
    }

    /// Return a slice of all operations in submission order.
    pub fn operations(&self) -> &[BatchOperation] {
        &self.operations
    }

    /// Return the list of WAL entries produced by commits.
    pub fn wal_entries(&self) -> &[WalBatchEntry] {
        &self.wal_entries
    }

    // ── mutation ─────────────────────────────────────────────────────────────

    /// Add an insert operation for the given triple.
    pub fn insert(&mut self, triple: BatchTriple) -> BatchResult<()> {
        self.ensure_pending()?;
        self.check_size_limit()?;
        self.operations.push(BatchOperation::Insert(triple));
        Ok(())
    }

    /// Add a delete operation for the given triple.
    pub fn delete(&mut self, triple: BatchTriple) -> BatchResult<()> {
        self.ensure_pending()?;
        self.check_size_limit()?;
        self.operations.push(BatchOperation::Delete(triple));
        Ok(())
    }

    /// Add an atomic update: delete `old` then insert `new`.
    pub fn update(&mut self, old: BatchTriple, new: BatchTriple) -> BatchResult<()> {
        self.ensure_pending()?;
        // Check that we have room for two operations.
        if self.config.max_batch_size > 0 && self.operations.len() + 2 > self.config.max_batch_size
        {
            return Err(BatchError::BatchSizeLimitReached {
                current: self.operations.len(),
                max: self.config.max_batch_size,
            });
        }
        self.operations.push(BatchOperation::Delete(old));
        self.operations.push(BatchOperation::Insert(new));
        Ok(())
    }

    /// Batch insert: add multiple triples atomically.
    pub fn insert_many(&mut self, triples: Vec<BatchTriple>) -> BatchResult<()> {
        self.ensure_pending()?;
        if self.config.max_batch_size > 0
            && self.operations.len() + triples.len() > self.config.max_batch_size
        {
            return Err(BatchError::BatchSizeLimitReached {
                current: self.operations.len(),
                max: self.config.max_batch_size,
            });
        }
        for t in triples {
            self.operations.push(BatchOperation::Insert(t));
        }
        Ok(())
    }

    /// Batch delete: remove multiple triples atomically.
    pub fn delete_many(&mut self, triples: Vec<BatchTriple>) -> BatchResult<()> {
        self.ensure_pending()?;
        if self.config.max_batch_size > 0
            && self.operations.len() + triples.len() > self.config.max_batch_size
        {
            return Err(BatchError::BatchSizeLimitReached {
                current: self.operations.len(),
                max: self.config.max_batch_size,
            });
        }
        for t in triples {
            self.operations.push(BatchOperation::Delete(t));
        }
        Ok(())
    }

    // ── commit / rollback ───────────────────────────────────────────────────

    /// Commit the batch, producing a WAL entry.
    ///
    /// Transitions the status from `Pending` to `Committed`.
    pub fn commit(&mut self) -> BatchResult<WalBatchEntry> {
        self.ensure_pending()?;
        let checksum = compute_operations_checksum(&self.operations);
        let entry = WalBatchEntry {
            batch_id: self.batch_id,
            operations: self.operations.clone(),
            checksum,
        };
        self.wal_entries.push(entry.clone());
        self.status = BatchStatus::Committed;
        Ok(entry)
    }

    /// Rollback the batch, discarding all operations.
    pub fn rollback(&mut self) -> BatchResult<()> {
        self.ensure_pending()?;
        self.operations.clear();
        self.status = BatchStatus::RolledBack;
        Ok(())
    }

    // ── merge ───────────────────────────────────────────────────────────────

    /// Merge another pending batch into this one.
    ///
    /// Both batches must be in the `Pending` state.  If any triple appears in
    /// both batches with conflicting intent (one insert, one delete for the
    /// same triple), an error is returned.
    pub fn merge(&mut self, other: &WriteBatch) -> BatchResult<()> {
        self.ensure_pending()?;
        if other.status != BatchStatus::Pending {
            return Err(BatchError::InvalidStatus {
                expected: BatchStatus::Pending,
                actual: other.status,
            });
        }

        // Build a map of triple -> is_insert for self operations.
        let mut intent_map: HashMap<&BatchTriple, bool> = HashMap::new();
        for op in &self.operations {
            intent_map.insert(op.triple(), op.is_insert());
        }

        // Check for conflicts.
        for op in &other.operations {
            if let Some(&existing_is_insert) = intent_map.get(op.triple()) {
                if existing_is_insert != op.is_insert() {
                    return Err(BatchError::MergeConflict {
                        description: format!(
                            "({}, {}, {})",
                            op.triple().subject,
                            op.triple().predicate,
                            op.triple().object,
                        ),
                    });
                }
            }
        }

        // Check size limit.
        let combined = self.operations.len() + other.operations.len();
        if self.config.max_batch_size > 0 && combined > self.config.max_batch_size {
            return Err(BatchError::BatchSizeLimitReached {
                current: combined,
                max: self.config.max_batch_size,
            });
        }

        self.operations.extend(other.operations.iter().cloned());
        Ok(())
    }

    // ── serialization ───────────────────────────────────────────────────────

    /// Serialize the batch operations into a compact byte representation.
    ///
    /// Format per operation:
    ///   `[op_byte][s_len:u32][s_bytes][p_len:u32][p_bytes][o_len:u32][o_bytes][graph_flag][g_len:u32?][g_bytes?]`
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // Header: batch_id (8 bytes) + operation count (4 bytes)
        buf.extend_from_slice(&self.batch_id.to_le_bytes());
        buf.extend_from_slice(&(self.operations.len() as u32).to_le_bytes());

        for op in &self.operations {
            let (op_byte, triple) = match op {
                BatchOperation::Insert(t) => (0u8, t),
                BatchOperation::Delete(t) => (1u8, t),
            };
            buf.push(op_byte);
            write_string(&mut buf, &triple.subject);
            write_string(&mut buf, &triple.predicate);
            write_string(&mut buf, &triple.object);
            match &triple.graph {
                Some(g) => {
                    buf.push(1);
                    write_string(&mut buf, g);
                }
                None => {
                    buf.push(0);
                }
            }
        }
        buf
    }

    /// Deserialize a batch from bytes produced by [`serialize`](Self::serialize).
    pub fn deserialize(data: &[u8]) -> BatchResult<Self> {
        if data.len() < 12 {
            return Err(BatchError::Other("data too short for header".to_string()));
        }
        let batch_id = u64::from_le_bytes(
            data[0..8]
                .try_into()
                .map_err(|_| BatchError::Other("bad batch_id bytes".to_string()))?,
        );
        let op_count = u32::from_le_bytes(
            data[8..12]
                .try_into()
                .map_err(|_| BatchError::Other("bad op count bytes".to_string()))?,
        ) as usize;

        let mut cursor = 12;
        let mut operations = Vec::with_capacity(op_count);

        for _ in 0..op_count {
            if cursor >= data.len() {
                return Err(BatchError::Other("unexpected end of data".to_string()));
            }
            let op_byte = data[cursor];
            cursor += 1;

            let subject = read_string(data, &mut cursor)?;
            let predicate = read_string(data, &mut cursor)?;
            let object = read_string(data, &mut cursor)?;

            if cursor >= data.len() {
                return Err(BatchError::Other(
                    "unexpected end of data (graph flag)".to_string(),
                ));
            }
            let graph_flag = data[cursor];
            cursor += 1;
            let graph = if graph_flag == 1 {
                Some(read_string(data, &mut cursor)?)
            } else {
                None
            };

            let triple = BatchTriple {
                subject,
                predicate,
                object,
                graph,
            };
            let op = if op_byte == 0 {
                BatchOperation::Insert(triple)
            } else {
                BatchOperation::Delete(triple)
            };
            operations.push(op);
        }

        let mut batch = WriteBatch::new(batch_id);
        batch.operations = operations;
        Ok(batch)
    }

    // ── helpers (new batch creation) ─────────────────────────────────────────

    /// Create a fresh batch from this writer's id sequence.
    pub fn next_batch(&mut self) -> WriteBatch {
        let id = self.next_batch_id;
        self.next_batch_id += 1;
        WriteBatch::new(id)
    }

    // ── internal ─────────────────────────────────────────────────────────────

    fn ensure_pending(&self) -> BatchResult<()> {
        if self.status != BatchStatus::Pending {
            return Err(BatchError::InvalidStatus {
                expected: BatchStatus::Pending,
                actual: self.status,
            });
        }
        Ok(())
    }

    fn check_size_limit(&self) -> BatchResult<()> {
        if self.config.max_batch_size > 0 && self.operations.len() >= self.config.max_batch_size {
            return Err(BatchError::BatchSizeLimitReached {
                current: self.operations.len(),
                max: self.config.max_batch_size,
            });
        }
        Ok(())
    }
}

// ── serialization helpers ────────────────────────────────────────────────────

fn write_string(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(bytes);
}

fn read_string(data: &[u8], cursor: &mut usize) -> BatchResult<String> {
    if *cursor + 4 > data.len() {
        return Err(BatchError::Other(
            "unexpected end of data (string len)".to_string(),
        ));
    }
    let len = u32::from_le_bytes(
        data[*cursor..*cursor + 4]
            .try_into()
            .map_err(|_| BatchError::Other("bad string len bytes".to_string()))?,
    ) as usize;
    *cursor += 4;
    if *cursor + len > data.len() {
        return Err(BatchError::Other(
            "unexpected end of data (string body)".to_string(),
        ));
    }
    let s = String::from_utf8(data[*cursor..*cursor + len].to_vec())
        .map_err(|e| BatchError::Other(format!("invalid utf-8: {e}")))?;
    *cursor += len;
    Ok(s)
}

// ── checksum ─────────────────────────────────────────────────────────────────

/// FNV-1a checksum of operations for WAL integrity.
fn compute_operations_checksum(operations: &[BatchOperation]) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for op in operations {
        let discriminant: u8 = if op.is_insert() { 0 } else { 1 };
        hash ^= discriminant as u32;
        hash = hash.wrapping_mul(0x0100_0193);
        for byte in op.triple().subject.as_bytes() {
            hash ^= *byte as u32;
            hash = hash.wrapping_mul(0x0100_0193);
        }
        for byte in op.triple().predicate.as_bytes() {
            hash ^= *byte as u32;
            hash = hash.wrapping_mul(0x0100_0193);
        }
        for byte in op.triple().object.as_bytes() {
            hash ^= *byte as u32;
            hash = hash.wrapping_mul(0x0100_0193);
        }
        if let Some(g) = &op.triple().graph {
            for byte in g.as_bytes() {
                hash ^= *byte as u32;
                hash = hash.wrapping_mul(0x0100_0193);
            }
        }
    }
    hash
}

/// Verify that a WAL entry's checksum matches its operations.
pub fn verify_wal_entry(entry: &WalBatchEntry) -> bool {
    compute_operations_checksum(&entry.operations) == entry.checksum
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn triple(s: &str, p: &str, o: &str) -> BatchTriple {
        BatchTriple::new(s, p, o)
    }

    fn quad(s: &str, p: &str, o: &str, g: &str) -> BatchTriple {
        BatchTriple::quad(s, p, o, g)
    }

    // ── BatchTriple construction ────────────────────────────────────────────

    #[test]
    fn test_triple_new() {
        let t = triple("s", "p", "o");
        assert_eq!(t.subject, "s");
        assert_eq!(t.predicate, "p");
        assert_eq!(t.object, "o");
        assert!(t.graph.is_none());
    }

    #[test]
    fn test_quad_new() {
        let t = quad("s", "p", "o", "g");
        assert_eq!(t.graph, Some("g".to_string()));
    }

    // ── BatchOperation ──────────────────────────────────────────────────────

    #[test]
    fn test_operation_is_insert() {
        let op = BatchOperation::Insert(triple("s", "p", "o"));
        assert!(op.is_insert());
        assert!(!op.is_delete());
    }

    #[test]
    fn test_operation_is_delete() {
        let op = BatchOperation::Delete(triple("s", "p", "o"));
        assert!(op.is_delete());
        assert!(!op.is_insert());
    }

    #[test]
    fn test_operation_triple_ref() {
        let t = triple("s", "p", "o");
        let op = BatchOperation::Insert(t.clone());
        assert_eq!(op.triple(), &t);
    }

    // ── WriteBatch basics ───────────────────────────────────────────────────

    #[test]
    fn test_new_batch_is_pending() {
        let b = WriteBatch::new(1);
        assert_eq!(b.status(), BatchStatus::Pending);
        assert!(b.is_empty());
        assert_eq!(b.len(), 0);
    }

    #[test]
    fn test_batch_id() {
        let b = WriteBatch::new(42);
        assert_eq!(b.id(), 42);
    }

    #[test]
    fn test_insert_increments_len() {
        let mut b = WriteBatch::new(1);
        b.insert(triple("s", "p", "o")).ok();
        assert_eq!(b.len(), 1);
        assert!(!b.is_empty());
    }

    #[test]
    fn test_delete_increments_len() {
        let mut b = WriteBatch::new(1);
        b.delete(triple("s", "p", "o")).ok();
        assert_eq!(b.len(), 1);
    }

    // ── batch insert/delete many ────────────────────────────────────────────

    #[test]
    fn test_insert_many() {
        let mut b = WriteBatch::new(1);
        let triples = vec![
            triple("s1", "p", "o"),
            triple("s2", "p", "o"),
            triple("s3", "p", "o"),
        ];
        b.insert_many(triples).ok();
        assert_eq!(b.len(), 3);
        assert!(b.operations().iter().all(|o| o.is_insert()));
    }

    #[test]
    fn test_delete_many() {
        let mut b = WriteBatch::new(1);
        let triples = vec![triple("s1", "p", "o"), triple("s2", "p", "o")];
        b.delete_many(triples).ok();
        assert_eq!(b.len(), 2);
        assert!(b.operations().iter().all(|o| o.is_delete()));
    }

    // ── update ──────────────────────────────────────────────────────────────

    #[test]
    fn test_update_produces_delete_then_insert() {
        let mut b = WriteBatch::new(1);
        b.update(triple("old_s", "p", "o"), triple("new_s", "p", "o"))
            .ok();
        assert_eq!(b.len(), 2);
        assert!(b.operations()[0].is_delete());
        assert!(b.operations()[1].is_insert());
    }

    // ── commit ──────────────────────────────────────────────────────────────

    #[test]
    fn test_commit_transitions_to_committed() {
        let mut b = WriteBatch::new(1);
        b.insert(triple("s", "p", "o")).ok();
        let _entry = b.commit();
        assert_eq!(b.status(), BatchStatus::Committed);
    }

    #[test]
    fn test_commit_produces_wal_entry() {
        let mut b = WriteBatch::new(1);
        b.insert(triple("s", "p", "o")).ok();
        let entry = b.commit().expect("commit should succeed");
        assert_eq!(entry.batch_id, 1);
        assert_eq!(entry.operations.len(), 1);
    }

    #[test]
    fn test_commit_wal_entry_stored() {
        let mut b = WriteBatch::new(1);
        b.insert(triple("s", "p", "o")).ok();
        b.commit().ok();
        assert_eq!(b.wal_entries().len(), 1);
    }

    #[test]
    fn test_commit_on_committed_fails() {
        let mut b = WriteBatch::new(1);
        b.insert(triple("s", "p", "o")).ok();
        b.commit().ok();
        let result = b.commit();
        assert!(result.is_err());
    }

    // ── rollback ────────────────────────────────────────────────────────────

    #[test]
    fn test_rollback_transitions_to_rolledback() {
        let mut b = WriteBatch::new(1);
        b.insert(triple("s", "p", "o")).ok();
        b.rollback().ok();
        assert_eq!(b.status(), BatchStatus::RolledBack);
    }

    #[test]
    fn test_rollback_clears_operations() {
        let mut b = WriteBatch::new(1);
        b.insert(triple("s", "p", "o")).ok();
        b.rollback().ok();
        assert!(b.is_empty());
    }

    #[test]
    fn test_rollback_on_committed_fails() {
        let mut b = WriteBatch::new(1);
        b.commit().ok();
        let result = b.rollback();
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_after_rollback_fails() {
        let mut b = WriteBatch::new(1);
        b.rollback().ok();
        let result = b.insert(triple("s", "p", "o"));
        assert!(result.is_err());
    }

    // ── size limits ─────────────────────────────────────────────────────────

    #[test]
    fn test_size_limit_enforced_on_insert() {
        let config = WriteBatchConfig::with_max_size(2);
        let mut b = WriteBatch::with_config(1, config);
        b.insert(triple("s1", "p", "o")).ok();
        b.insert(triple("s2", "p", "o")).ok();
        let result = b.insert(triple("s3", "p", "o"));
        assert!(result.is_err());
    }

    #[test]
    fn test_size_limit_enforced_on_delete() {
        let config = WriteBatchConfig::with_max_size(1);
        let mut b = WriteBatch::with_config(1, config);
        b.delete(triple("s1", "p", "o")).ok();
        let result = b.delete(triple("s2", "p", "o"));
        assert!(result.is_err());
    }

    #[test]
    fn test_size_limit_enforced_on_insert_many() {
        let config = WriteBatchConfig::with_max_size(2);
        let mut b = WriteBatch::with_config(1, config);
        let triples = vec![
            triple("a", "b", "c"),
            triple("d", "e", "f"),
            triple("g", "h", "i"),
        ];
        let result = b.insert_many(triples);
        assert!(result.is_err());
    }

    #[test]
    fn test_size_limit_enforced_on_update() {
        let config = WriteBatchConfig::with_max_size(1);
        let mut b = WriteBatch::with_config(1, config);
        let result = b.update(triple("old", "p", "o"), triple("new", "p", "o"));
        assert!(result.is_err());
    }

    #[test]
    fn test_unlimited_config_allows_many() {
        let config = WriteBatchConfig::unlimited();
        let mut b = WriteBatch::with_config(1, config);
        for i in 0..1000 {
            b.insert(triple(&format!("s{i}"), "p", "o")).ok();
        }
        assert_eq!(b.len(), 1000);
    }

    // ── merge ───────────────────────────────────────────────────────────────

    #[test]
    fn test_merge_compatible_batches() {
        let mut b1 = WriteBatch::new(1);
        b1.insert(triple("s1", "p", "o")).ok();
        let mut b2 = WriteBatch::new(2);
        b2.insert(triple("s2", "p", "o")).ok();
        b1.merge(&b2).ok();
        assert_eq!(b1.len(), 2);
    }

    #[test]
    fn test_merge_conflict_detected() {
        let mut b1 = WriteBatch::new(1);
        b1.insert(triple("s", "p", "o")).ok();
        let mut b2 = WriteBatch::new(2);
        b2.delete(triple("s", "p", "o")).ok();
        let result = b1.merge(&b2);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_same_intent_succeeds() {
        let mut b1 = WriteBatch::new(1);
        b1.insert(triple("s", "p", "o")).ok();
        let mut b2 = WriteBatch::new(2);
        b2.insert(triple("s", "p", "o")).ok();
        assert!(b1.merge(&b2).is_ok());
    }

    #[test]
    fn test_merge_committed_other_fails() {
        let mut b1 = WriteBatch::new(1);
        let mut b2 = WriteBatch::new(2);
        b2.commit().ok();
        let result = b1.merge(&b2);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_respects_size_limit() {
        let config = WriteBatchConfig::with_max_size(3);
        let mut b1 = WriteBatch::with_config(1, config);
        b1.insert(triple("s1", "p", "o")).ok();
        b1.insert(triple("s2", "p", "o")).ok();
        let mut b2 = WriteBatch::new(2);
        b2.insert(triple("s3", "p", "o")).ok();
        b2.insert(triple("s4", "p", "o")).ok();
        let result = b1.merge(&b2);
        assert!(result.is_err());
    }

    // ── stats ───────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_insert_count() {
        let mut b = WriteBatch::new(1);
        b.insert(triple("s1", "p", "o")).ok();
        b.insert(triple("s2", "p", "o")).ok();
        let s = b.stats();
        assert_eq!(s.inserts, 2);
        assert_eq!(s.deletes, 0);
    }

    #[test]
    fn test_stats_delete_count() {
        let mut b = WriteBatch::new(1);
        b.delete(triple("s1", "p", "o")).ok();
        let s = b.stats();
        assert_eq!(s.inserts, 0);
        assert_eq!(s.deletes, 1);
    }

    #[test]
    fn test_stats_mixed() {
        let mut b = WriteBatch::new(1);
        b.insert(triple("s1", "p", "o")).ok();
        b.delete(triple("s2", "p", "o")).ok();
        b.insert(triple("s3", "p", "o")).ok();
        let s = b.stats();
        assert_eq!(s.inserts, 2);
        assert_eq!(s.deletes, 1);
        assert_eq!(s.status, Some(BatchStatus::Pending));
    }

    // ── WAL checksum ────────────────────────────────────────────────────────

    #[test]
    fn test_wal_entry_checksum_valid() {
        let mut b = WriteBatch::new(1);
        b.insert(triple("s", "p", "o")).ok();
        let entry = b.commit().expect("commit ok");
        assert!(verify_wal_entry(&entry));
    }

    #[test]
    fn test_wal_entry_checksum_corrupted_fails() {
        let mut b = WriteBatch::new(1);
        b.insert(triple("s", "p", "o")).ok();
        let mut entry = b.commit().expect("commit ok");
        entry.checksum = entry.checksum.wrapping_add(1);
        assert!(!verify_wal_entry(&entry));
    }

    #[test]
    fn test_wal_checksum_deterministic() {
        let ops1 = vec![BatchOperation::Insert(triple("s", "p", "o"))];
        let ops2 = vec![BatchOperation::Insert(triple("s", "p", "o"))];
        assert_eq!(
            compute_operations_checksum(&ops1),
            compute_operations_checksum(&ops2)
        );
    }

    #[test]
    fn test_wal_checksum_differs_for_different_ops() {
        let ops1 = vec![BatchOperation::Insert(triple("s1", "p", "o"))];
        let ops2 = vec![BatchOperation::Insert(triple("s2", "p", "o"))];
        assert_ne!(
            compute_operations_checksum(&ops1),
            compute_operations_checksum(&ops2)
        );
    }

    // ── serialization / deserialization ──────────────────────────────────────

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let mut b = WriteBatch::new(42);
        b.insert(triple(
            "http://example.org/s",
            "http://example.org/p",
            "hello",
        ))
        .ok();
        b.delete(triple(
            "http://example.org/old",
            "http://example.org/p",
            "world",
        ))
        .ok();
        let data = b.serialize();
        let restored = WriteBatch::deserialize(&data).expect("deserialize ok");
        assert_eq!(restored.id(), 42);
        assert_eq!(restored.len(), 2);
        assert!(restored.operations()[0].is_insert());
        assert!(restored.operations()[1].is_delete());
    }

    #[test]
    fn test_serialize_deserialize_quad() {
        let mut b = WriteBatch::new(1);
        b.insert(quad("s", "p", "o", "g")).ok();
        let data = b.serialize();
        let restored = WriteBatch::deserialize(&data).expect("deserialize ok");
        assert_eq!(
            restored.operations()[0].triple().graph,
            Some("g".to_string())
        );
    }

    #[test]
    fn test_deserialize_empty_data_fails() {
        let result = WriteBatch::deserialize(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_truncated_data_fails() {
        let result = WriteBatch::deserialize(&[0u8; 5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_serialize_empty_batch() {
        let b = WriteBatch::new(1);
        let data = b.serialize();
        let restored = WriteBatch::deserialize(&data).expect("deserialize ok");
        assert!(restored.is_empty());
    }

    // ── next_batch ──────────────────────────────────────────────────────────

    #[test]
    fn test_next_batch_increments_id() {
        let mut b = WriteBatch::new(10);
        let b2 = b.next_batch();
        assert_eq!(b2.id(), 11);
        let b3 = b.next_batch();
        assert_eq!(b3.id(), 12);
    }

    // ── edge cases ──────────────────────────────────────────────────────────

    #[test]
    fn test_commit_empty_batch_succeeds() {
        let mut b = WriteBatch::new(1);
        let entry = b.commit().expect("commit ok");
        assert!(entry.operations.is_empty());
    }

    #[test]
    fn test_insert_after_commit_fails() {
        let mut b = WriteBatch::new(1);
        b.commit().ok();
        let result = b.insert(triple("s", "p", "o"));
        assert!(result.is_err());
    }

    #[test]
    fn test_delete_after_commit_fails() {
        let mut b = WriteBatch::new(1);
        b.commit().ok();
        let result = b.delete(triple("s", "p", "o"));
        assert!(result.is_err());
    }

    #[test]
    fn test_delete_many_after_commit_fails() {
        let mut b = WriteBatch::new(1);
        b.commit().ok();
        let result = b.delete_many(vec![triple("s", "p", "o")]);
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_many_after_commit_fails() {
        let mut b = WriteBatch::new(1);
        b.commit().ok();
        let result = b.insert_many(vec![triple("s", "p", "o")]);
        assert!(result.is_err());
    }

    #[test]
    fn test_update_after_commit_fails() {
        let mut b = WriteBatch::new(1);
        b.commit().ok();
        let result = b.update(triple("old", "p", "o"), triple("new", "p", "o"));
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_into_committed_fails() {
        let mut b1 = WriteBatch::new(1);
        b1.commit().ok();
        let b2 = WriteBatch::new(2);
        let result = b1.merge(&b2);
        assert!(result.is_err());
    }

    #[test]
    fn test_default_config() {
        let config = WriteBatchConfig::default();
        assert_eq!(config.max_batch_size, 100_000);
    }

    #[test]
    fn test_batch_status_in_stats() {
        let mut b = WriteBatch::new(1);
        assert_eq!(b.stats().status, Some(BatchStatus::Pending));
        b.commit().ok();
        assert_eq!(b.stats().status, Some(BatchStatus::Committed));
    }

    #[test]
    fn test_rollback_status_in_stats() {
        let mut b = WriteBatch::new(1);
        b.rollback().ok();
        assert_eq!(b.stats().status, Some(BatchStatus::RolledBack));
    }
}
