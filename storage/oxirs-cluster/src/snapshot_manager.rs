//! Raft snapshot management with retention policy and FNV-1a checksum validation.
//!
//! Manages the lifecycle of Raft snapshots: creation, installation, retention
//! pruning, and integrity verification. Snapshots capture the full state machine
//! state at a given log index so that old log entries can be discarded.

/// Metadata associated with a snapshot.
#[derive(Debug, Clone)]
pub struct SnapshotMetadata {
    /// The log index at which the snapshot was taken.
    pub index: u64,
    /// The term at that index.
    pub term: u64,
    /// Creation time in milliseconds since epoch (use 0 in tests).
    pub created_at_ms: u64,
    /// Size of the snapshot data in bytes.
    pub size_bytes: usize,
    /// FNV-1a checksum of the data bytes.
    pub checksum: u64,
}

/// A Raft snapshot: metadata + raw data bytes.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub metadata: SnapshotMetadata,
    pub data: Vec<u8>,
}

/// Policy controlling when new snapshots are automatically triggered.
#[derive(Debug, Clone)]
pub enum SnapshotPolicy {
    /// Take a snapshot every N log entries since the last snapshot.
    EveryN(u64),
    /// Take a snapshot when the estimated log byte size exceeds this threshold.
    SizeThreshold(usize),
    /// Never take automatic snapshots.
    Never,
}

/// Errors that can occur during snapshot operations.
#[derive(Debug)]
pub enum SnapshotError {
    InvalidChecksum,
    IndexTooOld(u64),
    EmptyData,
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SnapshotError::InvalidChecksum => write!(f, "Snapshot checksum mismatch"),
            SnapshotError::IndexTooOld(idx) => {
                write!(f, "Snapshot index {} is older than current state", idx)
            }
            SnapshotError::EmptyData => write!(f, "Snapshot data must not be empty"),
        }
    }
}

impl std::error::Error for SnapshotError {}

/// Manages the full lifecycle of Raft snapshots.
pub struct SnapshotManager {
    policy: SnapshotPolicy,
    /// Snapshots sorted ascending by index.
    snapshots: Vec<Snapshot>,
    max_retained: usize,
    current_index: u64,
    current_term: u64,
    entries_since_snapshot: u64,
    log_size_bytes: usize,
}

impl SnapshotManager {
    /// Create a new manager with the given policy and maximum number of retained snapshots.
    pub fn new(policy: SnapshotPolicy, max_retained: usize) -> Self {
        SnapshotManager {
            policy,
            snapshots: Vec::new(),
            max_retained,
            current_index: 0,
            current_term: 0,
            entries_since_snapshot: 0,
            log_size_bytes: 0,
        }
    }

    /// Create a new snapshot from the given data at the current log index and term.
    ///
    /// Returns an error if `data` is empty.
    /// Enforces the `max_retained` limit by evicting the oldest snapshot(s).
    pub fn create_snapshot(&mut self, data: Vec<u8>) -> Result<Snapshot, SnapshotError> {
        if data.is_empty() {
            return Err(SnapshotError::EmptyData);
        }
        let checksum = Self::fnv1a(&data);
        let metadata = SnapshotMetadata {
            index: self.current_index,
            term: self.current_term,
            created_at_ms: 0,
            size_bytes: data.len(),
            checksum,
        };
        let snapshot = Snapshot { metadata, data };

        self.snapshots.push(snapshot.clone());
        // Keep sorted by index
        self.snapshots.sort_by_key(|s| s.metadata.index);

        // Enforce max_retained
        if self.max_retained > 0 {
            while self.snapshots.len() > self.max_retained {
                self.snapshots.remove(0); // remove oldest
            }
        }

        // Reset entries-since-snapshot counter
        self.entries_since_snapshot = 0;
        self.log_size_bytes = 0;

        Ok(snapshot)
    }

    /// Check whether the current policy dictates that a new snapshot should be taken.
    pub fn should_snapshot(&self) -> bool {
        match &self.policy {
            SnapshotPolicy::EveryN(n) => *n > 0 && self.entries_since_snapshot >= *n,
            SnapshotPolicy::SizeThreshold(threshold) => self.log_size_bytes >= *threshold,
            SnapshotPolicy::Never => false,
        }
    }

    /// Install a snapshot received from the Raft leader.
    ///
    /// Validates the checksum and updates the current index/term.
    /// Returns `IndexTooOld` if the snapshot's index is not newer than the current state.
    pub fn install_snapshot(&mut self, snapshot: Snapshot) -> Result<(), SnapshotError> {
        if !Self::verify_checksum(&snapshot) {
            return Err(SnapshotError::InvalidChecksum);
        }
        // A snapshot must advance the state machine
        if snapshot.metadata.index <= self.current_index && self.current_index > 0 {
            return Err(SnapshotError::IndexTooOld(snapshot.metadata.index));
        }

        self.current_index = snapshot.metadata.index;
        self.current_term = snapshot.metadata.term;
        self.entries_since_snapshot = 0;
        self.log_size_bytes = 0;

        self.snapshots.push(snapshot);
        self.snapshots.sort_by_key(|s| s.metadata.index);

        if self.max_retained > 0 {
            while self.snapshots.len() > self.max_retained {
                self.snapshots.remove(0);
            }
        }

        Ok(())
    }

    /// The most recently created snapshot (highest index), if any.
    pub fn latest(&self) -> Option<&Snapshot> {
        self.snapshots.last()
    }

    /// Total number of snapshots currently retained.
    pub fn count(&self) -> usize {
        self.snapshots.len()
    }

    /// All retained snapshots in ascending index order.
    pub fn all_snapshots(&self) -> &[Snapshot] {
        &self.snapshots
    }

    /// Advance the log state by recording a new log entry.
    ///
    /// This updates the `current_index`, `current_term`, `entries_since_snapshot`,
    /// and `log_size_bytes` counters used by `should_snapshot()`.
    pub fn advance_log(&mut self, index: u64, term: u64, entry_size_bytes: usize) {
        self.current_index = index;
        self.current_term = term;
        self.entries_since_snapshot += 1;
        self.log_size_bytes += entry_size_bytes;
    }

    /// Remove all snapshots with index strictly less than `index`.
    pub fn truncate_before(&mut self, index: u64) {
        self.snapshots.retain(|s| s.metadata.index >= index);
    }

    /// Verify that a snapshot's stored checksum matches the FNV-1a hash of its data.
    pub fn verify_checksum(snapshot: &Snapshot) -> bool {
        Self::fnv1a(&snapshot.data) == snapshot.metadata.checksum
    }

    /// FNV-1a 64-bit hash of a byte slice.
    fn fnv1a(data: &[u8]) -> u64 {
        const FNV_PRIME: u64 = 0x00000100_000001b3;
        const FNV_OFFSET: u64 = 0xcbf29ce4_84222325;
        let mut hash = FNV_OFFSET;
        for &byte in data {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }
}

// Expose fnv1a for tests
impl SnapshotManager {
    pub fn fnv1a_pub(data: &[u8]) -> u64 {
        Self::fnv1a(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(content: &[u8]) -> Vec<u8> {
        content.to_vec()
    }

    // ── New manager ───────────────────────────────────────────────────────────

    #[test]
    fn test_new_manager_empty() {
        let mgr = SnapshotManager::new(SnapshotPolicy::Never, 3);
        assert_eq!(mgr.count(), 0);
        assert!(mgr.latest().is_none());
    }

    #[test]
    fn test_new_manager_should_not_snapshot_never() {
        let mgr = SnapshotManager::new(SnapshotPolicy::Never, 3);
        assert!(!mgr.should_snapshot());
    }

    // ── create_snapshot ───────────────────────────────────────────────────────

    #[test]
    fn test_create_snapshot_basic() {
        let mut mgr = SnapshotManager::new(SnapshotPolicy::Never, 5);
        mgr.advance_log(1, 1, 100);
        let snap = mgr.create_snapshot(make_data(b"state")).unwrap();
        assert_eq!(snap.metadata.index, 1);
        assert_eq!(snap.metadata.term, 1);
        assert_eq!(snap.metadata.size_bytes, 5);
        assert_eq!(mgr.count(), 1);
    }

    #[test]
    fn test_create_snapshot_empty_data_error() {
        let mut mgr = SnapshotManager::new(SnapshotPolicy::Never, 5);
        let result = mgr.create_snapshot(vec![]);
        assert!(result.is_err());
        match result.unwrap_err() {
            SnapshotError::EmptyData => {}
            e => panic!("wrong error: {:?}", e),
        }
    }

    #[test]
    fn test_create_snapshot_checksum_correct() {
        let mut mgr = SnapshotManager::new(SnapshotPolicy::Never, 5);
        mgr.advance_log(1, 1, 10);
        let snap = mgr.create_snapshot(make_data(b"hello")).unwrap();
        assert!(SnapshotManager::verify_checksum(&snap));
    }

    #[test]
    fn test_create_multiple_snapshots() {
        let mut mgr = SnapshotManager::new(SnapshotPolicy::Never, 10);
        mgr.advance_log(1, 1, 10);
        mgr.create_snapshot(make_data(b"snap1")).unwrap();
        mgr.advance_log(2, 1, 10);
        mgr.create_snapshot(make_data(b"snap2")).unwrap();
        mgr.advance_log(3, 1, 10);
        mgr.create_snapshot(make_data(b"snap3")).unwrap();
        assert_eq!(mgr.count(), 3);
    }

    // ── Checksum verification ─────────────────────────────────────────────────

    #[test]
    fn test_verify_checksum_ok() {
        let mut mgr = SnapshotManager::new(SnapshotPolicy::Never, 5);
        mgr.advance_log(1, 1, 10);
        let snap = mgr.create_snapshot(make_data(b"data")).unwrap();
        assert!(SnapshotManager::verify_checksum(&snap));
    }

    #[test]
    fn test_verify_checksum_tampered_data_fails() {
        let mut mgr = SnapshotManager::new(SnapshotPolicy::Never, 5);
        mgr.advance_log(1, 1, 10);
        let mut snap = mgr.create_snapshot(make_data(b"data")).unwrap();
        // Tamper with the data
        snap.data[0] ^= 0xFF;
        assert!(!SnapshotManager::verify_checksum(&snap));
    }

    #[test]
    fn test_verify_checksum_tampered_checksum_fails() {
        let mut mgr = SnapshotManager::new(SnapshotPolicy::Never, 5);
        mgr.advance_log(1, 1, 10);
        let mut snap = mgr.create_snapshot(make_data(b"data")).unwrap();
        snap.metadata.checksum ^= 0x1; // flip one bit
        assert!(!SnapshotManager::verify_checksum(&snap));
    }

    // ── max_retained eviction ─────────────────────────────────────────────────

    #[test]
    fn test_max_retained_evicts_oldest() {
        let mut mgr = SnapshotManager::new(SnapshotPolicy::Never, 2);
        mgr.advance_log(1, 1, 10);
        mgr.create_snapshot(make_data(b"snap1")).unwrap();
        mgr.advance_log(2, 1, 10);
        mgr.create_snapshot(make_data(b"snap2")).unwrap();
        mgr.advance_log(3, 1, 10);
        mgr.create_snapshot(make_data(b"snap3")).unwrap();

        assert_eq!(mgr.count(), 2);
        // Oldest (index=1) should be evicted
        let indices: Vec<u64> = mgr
            .all_snapshots()
            .iter()
            .map(|s| s.metadata.index)
            .collect();
        assert!(!indices.contains(&1));
        assert!(indices.contains(&2));
        assert!(indices.contains(&3));
    }

    #[test]
    fn test_max_retained_one() {
        let mut mgr = SnapshotManager::new(SnapshotPolicy::Never, 1);
        mgr.advance_log(1, 1, 10);
        mgr.create_snapshot(make_data(b"snap1")).unwrap();
        mgr.advance_log(2, 1, 10);
        mgr.create_snapshot(make_data(b"snap2")).unwrap();
        assert_eq!(mgr.count(), 1);
        assert_eq!(mgr.latest().unwrap().metadata.index, 2);
    }

    // ── EveryN policy ─────────────────────────────────────────────────────────

    #[test]
    fn test_every_n_policy_not_triggered_yet() {
        let mut mgr = SnapshotManager::new(SnapshotPolicy::EveryN(5), 10);
        mgr.advance_log(1, 1, 10);
        mgr.advance_log(2, 1, 10);
        assert!(!mgr.should_snapshot());
    }

    #[test]
    fn test_every_n_policy_triggers_at_n() {
        let mut mgr = SnapshotManager::new(SnapshotPolicy::EveryN(3), 10);
        mgr.advance_log(1, 1, 10);
        mgr.advance_log(2, 1, 10);
        mgr.advance_log(3, 1, 10);
        assert!(mgr.should_snapshot());
    }

    #[test]
    fn test_every_n_policy_resets_after_snapshot() {
        let mut mgr = SnapshotManager::new(SnapshotPolicy::EveryN(3), 10);
        mgr.advance_log(1, 1, 10);
        mgr.advance_log(2, 1, 10);
        mgr.advance_log(3, 1, 10);
        assert!(mgr.should_snapshot());
        mgr.create_snapshot(make_data(b"state")).unwrap();
        // After snapshot, counter resets
        assert!(!mgr.should_snapshot());
    }

    #[test]
    fn test_every_n_zero_never_triggers() {
        let mut mgr = SnapshotManager::new(SnapshotPolicy::EveryN(0), 10);
        for i in 1..=100 {
            mgr.advance_log(i, 1, 10);
        }
        assert!(!mgr.should_snapshot());
    }

    // ── SizeThreshold policy ──────────────────────────────────────────────────

    #[test]
    fn test_size_threshold_not_triggered() {
        let mut mgr = SnapshotManager::new(SnapshotPolicy::SizeThreshold(1000), 10);
        mgr.advance_log(1, 1, 100);
        assert!(!mgr.should_snapshot());
    }

    #[test]
    fn test_size_threshold_triggers() {
        let mut mgr = SnapshotManager::new(SnapshotPolicy::SizeThreshold(500), 10);
        for i in 1..=6 {
            mgr.advance_log(i, 1, 100); // 600 bytes total
        }
        assert!(mgr.should_snapshot());
    }

    #[test]
    fn test_size_threshold_resets_after_snapshot() {
        let mut mgr = SnapshotManager::new(SnapshotPolicy::SizeThreshold(200), 10);
        mgr.advance_log(1, 1, 100);
        mgr.advance_log(2, 1, 100);
        assert!(mgr.should_snapshot());
        mgr.create_snapshot(make_data(b"state")).unwrap();
        assert!(!mgr.should_snapshot());
    }

    // ── install_snapshot ──────────────────────────────────────────────────────

    #[test]
    fn test_install_snapshot_valid() {
        let mut mgr = SnapshotManager::new(SnapshotPolicy::Never, 5);
        let data = make_data(b"full state");
        let checksum = SnapshotManager::fnv1a_pub(&data);
        let snap = Snapshot {
            metadata: SnapshotMetadata {
                index: 10,
                term: 2,
                created_at_ms: 0,
                size_bytes: data.len(),
                checksum,
            },
            data,
        };
        mgr.install_snapshot(snap).unwrap();
        assert_eq!(mgr.current_index, 10);
        assert_eq!(mgr.current_term, 2);
        assert_eq!(mgr.count(), 1);
    }

    #[test]
    fn test_install_snapshot_invalid_checksum() {
        let mut mgr = SnapshotManager::new(SnapshotPolicy::Never, 5);
        let snap = Snapshot {
            metadata: SnapshotMetadata {
                index: 10,
                term: 1,
                created_at_ms: 0,
                size_bytes: 4,
                checksum: 0xDEADBEEF, // wrong checksum
            },
            data: make_data(b"data"),
        };
        let result = mgr.install_snapshot(snap);
        assert!(result.is_err());
        match result.unwrap_err() {
            SnapshotError::InvalidChecksum => {}
            e => panic!("wrong error: {:?}", e),
        }
    }

    #[test]
    fn test_install_snapshot_too_old() {
        let mut mgr = SnapshotManager::new(SnapshotPolicy::Never, 5);
        mgr.advance_log(20, 3, 100);

        let data = make_data(b"old");
        let checksum = SnapshotManager::fnv1a_pub(&data);
        let snap = Snapshot {
            metadata: SnapshotMetadata {
                index: 5, // older than current_index=20
                term: 1,
                created_at_ms: 0,
                size_bytes: data.len(),
                checksum,
            },
            data,
        };
        let result = mgr.install_snapshot(snap);
        assert!(result.is_err());
        match result.unwrap_err() {
            SnapshotError::IndexTooOld(5) => {}
            e => panic!("wrong error: {:?}", e),
        }
    }

    // ── truncate_before ───────────────────────────────────────────────────────

    #[test]
    fn test_truncate_before_removes_old() {
        let mut mgr = SnapshotManager::new(SnapshotPolicy::Never, 10);
        for i in 1..=5u64 {
            mgr.advance_log(i, 1, 10);
            mgr.create_snapshot(make_data(b"s")).unwrap();
        }
        mgr.truncate_before(3);
        let indices: Vec<u64> = mgr
            .all_snapshots()
            .iter()
            .map(|s| s.metadata.index)
            .collect();
        assert!(!indices.contains(&1));
        assert!(!indices.contains(&2));
        assert!(indices.contains(&3));
        assert!(indices.contains(&5));
    }

    #[test]
    fn test_truncate_before_removes_nothing_when_all_newer() {
        let mut mgr = SnapshotManager::new(SnapshotPolicy::Never, 10);
        mgr.advance_log(5, 1, 10);
        mgr.create_snapshot(make_data(b"s")).unwrap();
        mgr.truncate_before(3);
        assert_eq!(mgr.count(), 1); // index=5 >= 3, kept
    }

    #[test]
    fn test_truncate_before_removes_all() {
        let mut mgr = SnapshotManager::new(SnapshotPolicy::Never, 10);
        for i in 1..=3u64 {
            mgr.advance_log(i, 1, 10);
            mgr.create_snapshot(make_data(b"s")).unwrap();
        }
        mgr.truncate_before(100);
        assert_eq!(mgr.count(), 0);
    }

    // ── advance_log ───────────────────────────────────────────────────────────

    #[test]
    fn test_advance_log_updates_index_and_term() {
        let mut mgr = SnapshotManager::new(SnapshotPolicy::Never, 5);
        mgr.advance_log(42, 7, 256);
        assert_eq!(mgr.current_index, 42);
        assert_eq!(mgr.current_term, 7);
    }

    #[test]
    fn test_advance_log_increments_entries_counter() {
        let mut mgr = SnapshotManager::new(SnapshotPolicy::EveryN(5), 5);
        mgr.advance_log(1, 1, 100);
        mgr.advance_log(2, 1, 100);
        assert_eq!(mgr.entries_since_snapshot, 2);
    }

    // ── latest ────────────────────────────────────────────────────────────────

    #[test]
    fn test_latest_none_when_empty() {
        let mgr = SnapshotManager::new(SnapshotPolicy::Never, 5);
        assert!(mgr.latest().is_none());
    }

    #[test]
    fn test_latest_returns_highest_index() {
        let mut mgr = SnapshotManager::new(SnapshotPolicy::Never, 10);
        mgr.advance_log(1, 1, 10);
        mgr.create_snapshot(make_data(b"a")).unwrap();
        mgr.advance_log(2, 1, 10);
        mgr.create_snapshot(make_data(b"b")).unwrap();
        let latest = mgr.latest().unwrap();
        assert_eq!(latest.metadata.index, 2);
    }

    // ── FNV-1a consistency ────────────────────────────────────────────────────

    #[test]
    fn test_fnv1a_deterministic() {
        let data = b"hello world";
        let h1 = SnapshotManager::fnv1a_pub(data);
        let h2 = SnapshotManager::fnv1a_pub(data);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fnv1a_different_data_different_hash() {
        let h1 = SnapshotManager::fnv1a_pub(b"hello");
        let h2 = SnapshotManager::fnv1a_pub(b"world");
        assert_ne!(h1, h2);
    }
}
