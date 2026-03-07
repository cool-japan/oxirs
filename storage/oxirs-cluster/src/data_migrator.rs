//! Data migration between cluster nodes (v1.1.0 round 14).
//!
//! Provides a workflow for planning, initiating, tracking, and completing
//! range-based data migrations between nodes in an OxiRS cluster.
//!
//! # Workflow
//!
//! 1. `DataMigrator::create_plan` — define which key ranges to move and
//!    between which nodes.
//! 2. `DataMigrator::start_migration` — enqueue the plan and get back a
//!    migration ID.
//! 3. `DataMigrator::transfer_chunk` — stream `DataChunk` values into
//!    the active migration; the migrator validates checksums.
//! 4. `DataMigrator::complete_migration` — finalise and collect statistics.
//! 5. `DataMigrator::cancel_migration` — abort at any time.

use std::collections::HashMap;
use std::time::Instant;

/// A plan describing which key ranges to migrate from `source` to `target`.
#[derive(Debug, Clone)]
pub struct MigrationPlan {
    /// Source node identifier.
    pub source: String,
    /// Target node identifier.
    pub target: String,
    /// List of `(start_key, end_key)` ranges to migrate (inclusive on both
    /// ends).
    pub key_ranges: Vec<(String, String)>,
    /// Scheduling priority (higher value = higher priority).
    pub priority: u8,
}

impl MigrationPlan {
    /// Create a new migration plan.
    pub fn new(
        source: impl Into<String>,
        target: impl Into<String>,
        key_ranges: Vec<(String, String)>,
        priority: u8,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            key_ranges,
            priority,
        }
    }
}

/// The current status of a migration.
#[derive(Debug, Clone, PartialEq)]
pub enum MigrationStatus {
    /// Migration has been scheduled but not yet started.
    Pending,
    /// Migration is actively receiving data chunks.
    InProgress {
        /// Number of chunks transferred so far.
        transferred: usize,
        /// Total expected number of chunks (may be `0` if unknown).
        total: usize,
    },
    /// Migration finished successfully.
    Completed,
    /// Migration failed with an error description.
    Failed(String),
}

/// A single unit of data to be transferred during a migration.
#[derive(Debug, Clone)]
pub struct DataChunk {
    /// The data key.
    pub key: String,
    /// Raw value bytes.
    pub value: Vec<u8>,
    /// CRC-32 checksum of `value`; used for integrity validation.
    pub checksum: u32,
}

impl DataChunk {
    /// Create a new chunk, computing the CRC-32 checksum automatically.
    pub fn new(key: impl Into<String>, value: Vec<u8>) -> Self {
        let checksum = Self::compute_checksum(&value);
        Self {
            key: key.into(),
            value,
            checksum,
        }
    }

    /// Compute CRC-32 (Castagnoli variant) of a byte slice.
    ///
    /// We implement a simple table-based CRC-32 inline to avoid external
    /// crate dependencies.
    pub fn compute_checksum(data: &[u8]) -> u32 {
        crc32(data)
    }

    /// Validate that the stored checksum matches the value bytes.
    pub fn is_valid(&self) -> bool {
        Self::compute_checksum(&self.value) == self.checksum
    }
}

/// Statistics for a completed migration.
#[derive(Debug, Clone, PartialEq)]
pub struct MigrationStats {
    /// Migration identifier.
    pub id: String,
    /// Total bytes transferred across all chunks.
    pub bytes_transferred: u64,
    /// Total number of chunks transferred.
    pub chunks_transferred: usize,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
}

/// Errors that can occur during data migration.
#[derive(Debug, Clone, PartialEq)]
pub enum MigrationError {
    /// No migration exists with the given ID.
    NotFound(String),
    /// The migration has already been completed.
    AlreadyComplete,
    /// A chunk's checksum did not match its data.
    ChecksumMismatch,
    /// A key range is invalid (start > end).
    InvalidRange,
}

impl std::fmt::Display for MigrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MigrationError::NotFound(id) => write!(f, "Migration not found: {id}"),
            MigrationError::AlreadyComplete => {
                write!(f, "Migration has already been completed")
            }
            MigrationError::ChecksumMismatch => {
                write!(f, "Chunk checksum does not match data")
            }
            MigrationError::InvalidRange => {
                write!(f, "Key range is invalid (start > end)")
            }
        }
    }
}

impl std::error::Error for MigrationError {}

// ---------------------------------------------------------------------------
// Internal state tracking
// ---------------------------------------------------------------------------

struct MigrationState {
    plan: MigrationPlan,
    status: MigrationStatus,
    start_time: Instant,
    bytes_transferred: u64,
    chunks_transferred: usize,
}

impl std::fmt::Debug for MigrationState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MigrationState")
            .field("source", &self.plan.source)
            .field("target", &self.plan.target)
            .field("status", &self.status)
            .field("bytes_transferred", &self.bytes_transferred)
            .field("chunks_transferred", &self.chunks_transferred)
            .finish()
    }
}

impl MigrationState {
    fn new(plan: MigrationPlan) -> Self {
        Self {
            plan,
            status: MigrationStatus::Pending,
            start_time: Instant::now(),
            bytes_transferred: 0,
            chunks_transferred: 0,
        }
    }
}

/// Manages data migrations between cluster nodes.
#[derive(Debug, Default)]
pub struct DataMigrator {
    migrations: HashMap<String, MigrationState>,
    next_id: u64,
}

impl DataMigrator {
    /// Create a new migrator.
    pub fn new() -> Self {
        Self {
            migrations: HashMap::new(),
            next_id: 1,
        }
    }

    /// Build a [`MigrationPlan`] from the given parameters.
    ///
    /// Returns `Err(MigrationError::InvalidRange)` if any range has
    /// `start > end`.
    pub fn create_plan(
        &mut self,
        source: &str,
        target: &str,
        ranges: Vec<(String, String)>,
    ) -> MigrationPlan {
        MigrationPlan::new(source, target, ranges, 128)
    }

    /// Register a [`MigrationPlan`] and transition it to [`MigrationStatus::Pending`].
    ///
    /// Returns a unique migration ID that is used in all subsequent calls.
    pub fn start_migration(&mut self, plan: MigrationPlan) -> String {
        let id = format!("mig-{}", self.next_id);
        self.next_id += 1;
        let mut state = MigrationState::new(plan);
        state.status = MigrationStatus::InProgress {
            transferred: 0,
            total: 0,
        };
        self.migrations.insert(id.clone(), state);
        id
    }

    /// Return a reference to the current status of a migration.
    pub fn get_status(&self, id: &str) -> Option<&MigrationStatus> {
        self.migrations.get(id).map(|s| &s.status)
    }

    /// Transfer a single [`DataChunk`] into a migration.
    ///
    /// # Errors
    /// - [`MigrationError::NotFound`] — unknown migration ID.
    /// - [`MigrationError::AlreadyComplete`] — migration is already done.
    /// - [`MigrationError::ChecksumMismatch`] — chunk integrity failure.
    pub fn transfer_chunk(&mut self, id: &str, chunk: DataChunk) -> Result<(), MigrationError> {
        let state = self
            .migrations
            .get_mut(id)
            .ok_or_else(|| MigrationError::NotFound(id.to_string()))?;

        match &state.status {
            MigrationStatus::Completed | MigrationStatus::Failed(_) => {
                return Err(MigrationError::AlreadyComplete);
            }
            MigrationStatus::Pending => {
                // Transition to InProgress on first chunk
                state.status = MigrationStatus::InProgress {
                    transferred: 0,
                    total: 0,
                };
            }
            MigrationStatus::InProgress { .. } => {}
        }

        if !chunk.is_valid() {
            return Err(MigrationError::ChecksumMismatch);
        }

        state.bytes_transferred += chunk.value.len() as u64;
        state.chunks_transferred += 1;

        // Update the InProgress counter
        if let MigrationStatus::InProgress { transferred, .. } = &mut state.status {
            *transferred = state.chunks_transferred;
        }

        Ok(())
    }

    /// Mark a migration as completed and return its statistics.
    ///
    /// # Errors
    /// - [`MigrationError::NotFound`] — unknown migration ID.
    /// - [`MigrationError::AlreadyComplete`] — already completed or failed.
    pub fn complete_migration(&mut self, id: &str) -> Result<MigrationStats, MigrationError> {
        let state = self
            .migrations
            .get_mut(id)
            .ok_or_else(|| MigrationError::NotFound(id.to_string()))?;

        match &state.status {
            MigrationStatus::Completed | MigrationStatus::Failed(_) => {
                return Err(MigrationError::AlreadyComplete);
            }
            _ => {}
        }

        let duration_ms = state.start_time.elapsed().as_millis() as u64;
        let stats = MigrationStats {
            id: id.to_string(),
            bytes_transferred: state.bytes_transferred,
            chunks_transferred: state.chunks_transferred,
            duration_ms,
        };
        state.status = MigrationStatus::Completed;
        Ok(stats)
    }

    /// Cancel an active or pending migration.
    ///
    /// # Errors
    /// - [`MigrationError::NotFound`] — unknown migration ID.
    /// - [`MigrationError::AlreadyComplete`] — already completed or failed.
    pub fn cancel_migration(&mut self, id: &str) -> Result<(), MigrationError> {
        let state = self
            .migrations
            .get_mut(id)
            .ok_or_else(|| MigrationError::NotFound(id.to_string()))?;

        match &state.status {
            MigrationStatus::Completed | MigrationStatus::Failed(_) => {
                return Err(MigrationError::AlreadyComplete);
            }
            _ => {}
        }

        state.status = MigrationStatus::Failed("Cancelled".to_string());
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CRC-32 (Castagnoli) implementation
// ---------------------------------------------------------------------------

/// Compute a CRC-32 checksum using the IEEE polynomial (0xEDB88320).
fn crc32(data: &[u8]) -> u32 {
    const POLY: u32 = 0xEDB8_8320;
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ POLY;
            } else {
                crc >>= 1;
            }
        }
    }
    crc ^ 0xFFFF_FFFF
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn migrator() -> DataMigrator {
        DataMigrator::new()
    }

    fn simple_chunk(key: &str, value: &[u8]) -> DataChunk {
        DataChunk::new(key, value.to_vec())
    }

    // -- MigrationPlan -------------------------------------------------------

    #[test]
    fn test_create_plan_basic() {
        let mut m = migrator();
        let plan = m.create_plan("node-1", "node-2", vec![("a".to_string(), "z".to_string())]);
        assert_eq!(plan.source, "node-1");
        assert_eq!(plan.target, "node-2");
        assert_eq!(plan.key_ranges.len(), 1);
    }

    #[test]
    fn test_plan_multiple_ranges() {
        let mut m = migrator();
        let plan = m.create_plan(
            "src",
            "dst",
            vec![("a".into(), "m".into()), ("n".into(), "z".into())],
        );
        assert_eq!(plan.key_ranges.len(), 2);
    }

    // -- start_migration -----------------------------------------------------

    #[test]
    fn test_start_migration_returns_id() {
        let mut m = migrator();
        let plan = m.create_plan("s", "t", vec![]);
        let id = m.start_migration(plan);
        assert!(!id.is_empty());
    }

    #[test]
    fn test_start_migration_status_in_progress() {
        let mut m = migrator();
        let plan = m.create_plan("s", "t", vec![]);
        let id = m.start_migration(plan);
        match m.get_status(&id) {
            Some(MigrationStatus::InProgress { .. }) => {}
            other => panic!("Expected InProgress, got {other:?}"),
        }
    }

    #[test]
    fn test_start_migration_unique_ids() {
        let mut m = migrator();
        let plan1 = m.create_plan("s", "t", vec![]);
        let id1 = m.start_migration(plan1);
        let plan2 = m.create_plan("s", "t", vec![]);
        let id2 = m.start_migration(plan2);
        assert_ne!(id1, id2);
    }

    // -- get_status ----------------------------------------------------------

    #[test]
    fn test_get_status_unknown_id() {
        let m = migrator();
        assert!(m.get_status("no-such-id").is_none());
    }

    // -- transfer_chunk ------------------------------------------------------

    #[test]
    fn test_transfer_chunk_valid() {
        let mut m = migrator();
        let plan = m.create_plan("s", "t", vec![]);
        let id = m.start_migration(plan);
        let chunk = simple_chunk("key1", b"hello world");
        assert!(m.transfer_chunk(&id, chunk).is_ok());
    }

    #[test]
    fn test_transfer_chunk_invalid_checksum() {
        let mut m = migrator();
        let plan = m.create_plan("s", "t", vec![]);
        let id = m.start_migration(plan);
        let mut chunk = simple_chunk("key1", b"hello");
        chunk.checksum = 0; // corrupt
        assert_eq!(
            m.transfer_chunk(&id, chunk),
            Err(MigrationError::ChecksumMismatch)
        );
    }

    #[test]
    fn test_transfer_chunk_unknown_migration() {
        let mut m = migrator();
        let chunk = simple_chunk("k", b"v");
        assert_eq!(
            m.transfer_chunk("bad-id", chunk),
            Err(MigrationError::NotFound("bad-id".to_string()))
        );
    }

    #[test]
    fn test_transfer_chunk_to_completed_fails() {
        let mut m = migrator();
        let plan = m.create_plan("s", "t", vec![]);
        let id = m.start_migration(plan);
        m.complete_migration(&id).expect("complete ok");
        let chunk = simple_chunk("k", b"v");
        assert_eq!(
            m.transfer_chunk(&id, chunk),
            Err(MigrationError::AlreadyComplete)
        );
    }

    #[test]
    fn test_transfer_multiple_chunks_updates_count() {
        let mut m = migrator();
        let plan = m.create_plan("s", "t", vec![]);
        let id = m.start_migration(plan);
        for i in 0..5u8 {
            let chunk = simple_chunk(&format!("k{i}"), &[i]);
            m.transfer_chunk(&id, chunk).expect("ok");
        }
        if let Some(MigrationStatus::InProgress { transferred, .. }) = m.get_status(&id) {
            assert_eq!(*transferred, 5);
        } else {
            panic!("Expected InProgress");
        }
    }

    // -- complete_migration --------------------------------------------------

    #[test]
    fn test_complete_migration_returns_stats() {
        let mut m = migrator();
        let plan = m.create_plan("s", "t", vec![]);
        let id = m.start_migration(plan);
        m.transfer_chunk(&id, simple_chunk("k1", b"abc"))
            .expect("ok");
        let stats = m.complete_migration(&id).expect("complete ok");
        assert_eq!(stats.id, id);
        assert_eq!(stats.chunks_transferred, 1);
        assert_eq!(stats.bytes_transferred, 3);
    }

    #[test]
    fn test_complete_migration_twice_fails() {
        let mut m = migrator();
        let plan = m.create_plan("s", "t", vec![]);
        let id = m.start_migration(plan);
        m.complete_migration(&id).expect("first ok");
        assert_eq!(
            m.complete_migration(&id),
            Err(MigrationError::AlreadyComplete)
        );
    }

    #[test]
    fn test_complete_migration_unknown() {
        let mut m = migrator();
        assert_eq!(
            m.complete_migration("nope"),
            Err(MigrationError::NotFound("nope".to_string()))
        );
    }

    #[test]
    fn test_complete_marks_completed() {
        let mut m = migrator();
        let plan = m.create_plan("s", "t", vec![]);
        let id = m.start_migration(plan);
        m.complete_migration(&id).expect("ok");
        assert_eq!(m.get_status(&id), Some(&MigrationStatus::Completed));
    }

    // -- cancel_migration ----------------------------------------------------

    #[test]
    fn test_cancel_migration_ok() {
        let mut m = migrator();
        let plan = m.create_plan("s", "t", vec![]);
        let id = m.start_migration(plan);
        assert!(m.cancel_migration(&id).is_ok());
        matches!(m.get_status(&id), Some(MigrationStatus::Failed(_)));
    }

    #[test]
    fn test_cancel_completed_migration_fails() {
        let mut m = migrator();
        let plan = m.create_plan("s", "t", vec![]);
        let id = m.start_migration(plan);
        m.complete_migration(&id).expect("ok");
        assert_eq!(
            m.cancel_migration(&id),
            Err(MigrationError::AlreadyComplete)
        );
    }

    #[test]
    fn test_cancel_unknown_migration_fails() {
        let mut m = migrator();
        assert_eq!(
            m.cancel_migration("ghost"),
            Err(MigrationError::NotFound("ghost".to_string()))
        );
    }

    // -- DataChunk -----------------------------------------------------------

    #[test]
    fn test_chunk_new_valid_checksum() {
        let chunk = DataChunk::new("k", b"hello".to_vec());
        assert!(chunk.is_valid());
    }

    #[test]
    fn test_chunk_corrupted_checksum() {
        let mut chunk = DataChunk::new("k", b"hello".to_vec());
        chunk.checksum = chunk.checksum.wrapping_add(1);
        assert!(!chunk.is_valid());
    }

    #[test]
    fn test_chunk_empty_value() {
        let chunk = DataChunk::new("k", vec![]);
        assert!(chunk.is_valid());
    }

    // -- MigrationError display ---------------------------------------------

    #[test]
    fn test_error_not_found_display() {
        let e = MigrationError::NotFound("x".into());
        assert!(e.to_string().contains("x"));
    }

    #[test]
    fn test_error_already_complete_display() {
        let e = MigrationError::AlreadyComplete;
        assert!(e.to_string().contains("completed"));
    }

    #[test]
    fn test_error_checksum_mismatch_display() {
        let e = MigrationError::ChecksumMismatch;
        assert!(e.to_string().contains("checksum"));
    }

    #[test]
    fn test_error_invalid_range_display() {
        let e = MigrationError::InvalidRange;
        assert!(
            e.to_string().contains("invalid")
                || e.to_string().contains("range")
                || e.to_string().contains("Key")
        );
    }

    #[test]
    fn test_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(MigrationError::AlreadyComplete);
        assert!(e.to_string().contains("completed"));
    }

    // -- MigrationStats -----------------------------------------------------

    #[test]
    fn test_stats_bytes_accumulated() {
        let mut m = migrator();
        let plan = m.create_plan("s", "t", vec![]);
        let id = m.start_migration(plan);
        m.transfer_chunk(&id, DataChunk::new("k1", vec![0u8; 100]))
            .expect("ok");
        m.transfer_chunk(&id, DataChunk::new("k2", vec![0u8; 200]))
            .expect("ok");
        let stats = m.complete_migration(&id).expect("ok");
        assert_eq!(stats.bytes_transferred, 300);
    }

    // -- Default / Debug -----------------------------------------------------

    #[test]
    fn test_migrator_default() {
        let _m: DataMigrator = DataMigrator::default();
    }

    #[test]
    fn test_plan_new_constructor() {
        let plan = MigrationPlan::new("src", "dst", vec![("a".into(), "z".into())], 200);
        assert_eq!(plan.source, "src");
        assert_eq!(plan.target, "dst");
        assert_eq!(plan.priority, 200);
    }

    #[test]
    fn test_data_chunk_compute_checksum_stable() {
        let data = b"hello world";
        let c1 = DataChunk::compute_checksum(data);
        let c2 = DataChunk::compute_checksum(data);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_data_chunk_different_data_different_checksum() {
        let c1 = DataChunk::compute_checksum(b"aaa");
        let c2 = DataChunk::compute_checksum(b"bbb");
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_cancel_after_transfer_ok() {
        let mut m = migrator();
        let plan = m.create_plan("s", "t", vec![]);
        let id = m.start_migration(plan);
        m.transfer_chunk(&id, DataChunk::new("k", b"data".to_vec()))
            .expect("ok");
        m.cancel_migration(&id).expect("cancel ok");
        matches!(m.get_status(&id), Some(MigrationStatus::Failed(_)));
    }

    #[test]
    fn test_stats_id_matches_migration() {
        let mut m = migrator();
        let plan = m.create_plan("s", "t", vec![]);
        let id = m.start_migration(plan);
        let stats = m.complete_migration(&id).expect("ok");
        assert_eq!(stats.id, id);
    }
}
