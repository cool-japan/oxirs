//! Modbus register change detector.
//!
//! Tracks sequential register snapshots and emits `RegisterChange` records whenever
//! a register value differs from the previous snapshot.  A ring-buffer style change
//! log is maintained with a configurable maximum size.

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/// A snapshot of contiguous Modbus register values taken at a specific instant.
#[derive(Debug, Clone)]
pub struct RegisterSnapshot {
    /// The starting Modbus address of the first register in `values`.
    pub address_start: u16,
    /// Register values in address order.
    pub values: Vec<u16>,
    /// Timestamp when this snapshot was captured, in milliseconds since the epoch.
    pub timestamp_ms: u64,
}

impl RegisterSnapshot {
    /// Create a new snapshot.
    pub fn new(address_start: u16, values: Vec<u16>, timestamp_ms: u64) -> Self {
        Self {
            address_start,
            values,
            timestamp_ms,
        }
    }

    /// Return the address of the last register in this snapshot, or `None` if empty.
    pub fn address_end(&self) -> Option<u16> {
        if self.values.is_empty() {
            None
        } else {
            Some(self.address_start + (self.values.len() as u16) - 1)
        }
    }

    /// Return the value of the register at `address`, or `None` if not contained.
    pub fn get(&self, address: u16) -> Option<u16> {
        if address < self.address_start {
            return None;
        }
        let idx = (address - self.address_start) as usize;
        self.values.get(idx).copied()
    }
}

/// A single detected change in a register value between two snapshots.
#[derive(Debug, Clone, PartialEq)]
pub struct RegisterChange {
    /// The Modbus register address that changed.
    pub address: u16,
    /// The register value in the previous snapshot.
    pub old_value: u16,
    /// The register value in the new snapshot.
    pub new_value: u16,
    /// Timestamp of the new snapshot where the change was detected, in milliseconds.
    pub timestamp_ms: u64,
}

// ────────────────────────────────────────────────────────────────────────────
// RegisterWatcher
// ────────────────────────────────────────────────────────────────────────────

/// Watches Modbus register snapshots and detects value changes.
///
/// When `update` is called with a new `RegisterSnapshot`, the watcher compares each
/// register in the new snapshot against the corresponding register in the previous
/// snapshot and records any differences in an internal change log.
///
/// The change log is a bounded ring-buffer: once it reaches `max_log_size` entries
/// the oldest entries are dropped to make room for new ones.
pub struct RegisterWatcher {
    /// The most recently ingested snapshot.
    last_snapshot: Option<RegisterSnapshot>,
    /// Ring-buffer of detected changes (oldest entries dropped when full).
    change_log: Vec<RegisterChange>,
    /// Maximum number of entries retained in `change_log`.
    max_log_size: usize,
}

impl RegisterWatcher {
    /// Create a new watcher that retains at most `max_log_size` change entries.
    ///
    /// A `max_log_size` of `0` disables the log (no entries are retained).
    pub fn new(max_log_size: usize) -> Self {
        Self {
            last_snapshot: None,
            change_log: Vec::new(),
            max_log_size,
        }
    }

    /// Ingest a new snapshot and return any changes detected versus the previous snapshot.
    ///
    /// On the very first call there is no previous snapshot, so no changes are detected
    /// and an empty `Vec` is returned.  The new snapshot is stored for subsequent comparisons.
    pub fn update(&mut self, snapshot: RegisterSnapshot) -> Vec<RegisterChange> {
        let mut changes = Vec::new();

        if let Some(ref prev) = self.last_snapshot {
            // Walk every register in the new snapshot and compare against the previous.
            for (i, &new_val) in snapshot.values.iter().enumerate() {
                let address = snapshot.address_start + i as u16;
                if let Some(old_val) = prev.get(address) {
                    if old_val != new_val {
                        changes.push(RegisterChange {
                            address,
                            old_value: old_val,
                            new_value: new_val,
                            timestamp_ms: snapshot.timestamp_ms,
                        });
                    }
                }
            }
        }

        // Persist new changes into the log, respecting the size cap.
        for change in &changes {
            if self.max_log_size == 0 {
                break;
            }
            if self.change_log.len() >= self.max_log_size {
                self.change_log.remove(0);
            }
            self.change_log.push(change.clone());
        }

        self.last_snapshot = Some(snapshot);
        changes
    }

    /// Return the most recent `n` changes from the log (most recent last).
    ///
    /// If `n` is larger than the number of logged changes, all changes are returned.
    pub fn recent_changes(&self, n: usize) -> Vec<&RegisterChange> {
        let start = self.change_log.len().saturating_sub(n);
        self.change_log[start..].iter().collect()
    }

    /// Return all logged changes for the given register `address`.
    pub fn changes_for(&self, address: u16) -> Vec<&RegisterChange> {
        self.change_log
            .iter()
            .filter(|c| c.address == address)
            .collect()
    }

    /// Return the total number of changes currently in the log.
    pub fn change_count(&self) -> usize {
        self.change_log.len()
    }

    /// Clear all entries from the change log.
    pub fn clear(&mut self) {
        self.change_log.clear();
    }

    /// Return `true` if any logged change is for the given register `address`.
    pub fn has_changed(&self, address: u16) -> bool {
        self.change_log.iter().any(|c| c.address == address)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────

    fn snap(addr: u16, values: Vec<u16>, ts: u64) -> RegisterSnapshot {
        RegisterSnapshot::new(addr, values, ts)
    }

    // ── RegisterSnapshot helpers ──────────────────────────────────────────

    #[test]
    fn test_snapshot_get_in_range() {
        let s = snap(100, vec![10, 20, 30], 0);
        assert_eq!(s.get(100), Some(10));
        assert_eq!(s.get(101), Some(20));
        assert_eq!(s.get(102), Some(30));
    }

    #[test]
    fn test_snapshot_get_out_of_range() {
        let s = snap(100, vec![10, 20], 0);
        assert_eq!(s.get(99), None);
        assert_eq!(s.get(102), None);
    }

    #[test]
    fn test_snapshot_address_end_non_empty() {
        let s = snap(10, vec![1, 2, 3], 0);
        assert_eq!(s.address_end(), Some(12));
    }

    #[test]
    fn test_snapshot_address_end_empty() {
        let s = snap(10, vec![], 0);
        assert_eq!(s.address_end(), None);
    }

    #[test]
    fn test_snapshot_single_register() {
        let s = snap(200, vec![42], 1000);
        assert_eq!(s.get(200), Some(42));
        assert_eq!(s.address_end(), Some(200));
    }

    // ── First update — no previous snapshot ─────────────────────────────

    #[test]
    fn test_first_update_no_changes() {
        let mut w = RegisterWatcher::new(100);
        let changes = w.update(snap(0, vec![1, 2, 3], 1000));
        assert!(changes.is_empty());
        assert_eq!(w.change_count(), 0);
    }

    #[test]
    fn test_first_update_stores_snapshot() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(0, vec![100, 200], 1000));
        // No crash; last_snapshot is set internally.
        assert_eq!(w.change_count(), 0);
    }

    // ── Second update with changes ────────────────────────────────────────

    #[test]
    fn test_second_update_detects_change() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(0, vec![10, 20, 30], 1000));
        let changes = w.update(snap(0, vec![10, 99, 30], 2000));
        assert_eq!(changes.len(), 1);
        let c = &changes[0];
        assert_eq!(c.address, 1);
        assert_eq!(c.old_value, 20);
        assert_eq!(c.new_value, 99);
        assert_eq!(c.timestamp_ms, 2000);
    }

    #[test]
    fn test_second_update_detects_multiple_changes() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(0, vec![1, 2, 3, 4], 1000));
        let changes = w.update(snap(0, vec![9, 2, 8, 4], 2000));
        assert_eq!(changes.len(), 2);
        let addrs: Vec<u16> = changes.iter().map(|c| c.address).collect();
        assert!(addrs.contains(&0));
        assert!(addrs.contains(&2));
    }

    // ── No-change update ──────────────────────────────────────────────────

    #[test]
    fn test_no_change_update_returns_empty() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(0, vec![7, 8, 9], 1000));
        let changes = w.update(snap(0, vec![7, 8, 9], 2000));
        assert!(changes.is_empty());
        assert_eq!(w.change_count(), 0);
    }

    // ── Change log grows ──────────────────────────────────────────────────

    #[test]
    fn test_change_log_grows_across_updates() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(0, vec![0], 1000));
        w.update(snap(0, vec![1], 2000));
        w.update(snap(0, vec![2], 3000));
        w.update(snap(0, vec![3], 4000));
        assert_eq!(w.change_count(), 3);
    }

    // ── max_log_size overflow ─────────────────────────────────────────────

    #[test]
    fn test_max_log_size_overflow_drops_oldest() {
        let mut w = RegisterWatcher::new(3);
        w.update(snap(0, vec![0], 1000));
        w.update(snap(0, vec![1], 2000)); // change log: [addr0 old=0 new=1]
        w.update(snap(0, vec![2], 3000)); // change log: [addr0 old=1 new=2]
        w.update(snap(0, vec![3], 4000)); // change log: [addr0 old=2 new=3]
                                          // Now overflow: next change should push out oldest
        w.update(snap(0, vec![4], 5000));
        assert_eq!(w.change_count(), 3);
        // The oldest entry (old=0→new=1) should be gone; newest is new=4
        let last = w.recent_changes(1)[0];
        assert_eq!(last.new_value, 4);
    }

    #[test]
    fn test_max_log_size_zero_no_entries_stored() {
        let mut w = RegisterWatcher::new(0);
        w.update(snap(0, vec![0], 1000));
        w.update(snap(0, vec![1], 2000));
        assert_eq!(w.change_count(), 0);
    }

    #[test]
    fn test_max_log_size_one_keeps_latest_only() {
        let mut w = RegisterWatcher::new(1);
        w.update(snap(0, vec![0], 1000));
        w.update(snap(0, vec![1], 2000));
        w.update(snap(0, vec![2], 3000));
        assert_eq!(w.change_count(), 1);
        assert_eq!(w.recent_changes(1)[0].new_value, 2);
    }

    // ── recent_changes ────────────────────────────────────────────────────

    #[test]
    fn test_recent_changes_returns_last_n() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(0, vec![0], 1000));
        for i in 1_u16..=10 {
            w.update(snap(0, vec![i], (i as u64 + 1) * 1000));
        }
        let recent = w.recent_changes(3);
        assert_eq!(recent.len(), 3);
        // Most-recent 3: new_value = 10, 9, 8 in order
        assert_eq!(recent[2].new_value, 10);
        assert_eq!(recent[1].new_value, 9);
        assert_eq!(recent[0].new_value, 8);
    }

    #[test]
    fn test_recent_changes_n_larger_than_log() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(0, vec![0], 1000));
        w.update(snap(0, vec![1], 2000));
        // Only 1 change logged
        let recent = w.recent_changes(50);
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn test_recent_changes_empty_log() {
        let w = RegisterWatcher::new(100);
        let recent = w.recent_changes(10);
        assert!(recent.is_empty());
    }

    #[test]
    fn test_recent_changes_n_zero() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(0, vec![0], 1000));
        w.update(snap(0, vec![1], 2000));
        let recent = w.recent_changes(0);
        assert!(recent.is_empty());
    }

    // ── changes_for ───────────────────────────────────────────────────────

    #[test]
    fn test_changes_for_specific_address() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(10, vec![0, 0], 1000));
        w.update(snap(10, vec![1, 0], 2000)); // addr 10 changed
        w.update(snap(10, vec![1, 5], 3000)); // addr 11 changed
        w.update(snap(10, vec![2, 5], 4000)); // addr 10 changed again
        let for_10 = w.changes_for(10);
        assert_eq!(for_10.len(), 2);
        let for_11 = w.changes_for(11);
        assert_eq!(for_11.len(), 1);
    }

    #[test]
    fn test_changes_for_unknown_address() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(0, vec![1], 1000));
        w.update(snap(0, vec![2], 2000));
        let for_99 = w.changes_for(99);
        assert!(for_99.is_empty());
    }

    // ── change_count ─────────────────────────────────────────────────────

    #[test]
    fn test_change_count_initial_zero() {
        let w = RegisterWatcher::new(50);
        assert_eq!(w.change_count(), 0);
    }

    #[test]
    fn test_change_count_after_no_change_update() {
        let mut w = RegisterWatcher::new(50);
        w.update(snap(0, vec![7], 1000));
        w.update(snap(0, vec![7], 2000));
        assert_eq!(w.change_count(), 0);
    }

    // ── clear ─────────────────────────────────────────────────────────────

    #[test]
    fn test_clear_empties_log() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(0, vec![0], 1000));
        w.update(snap(0, vec![1], 2000));
        assert_eq!(w.change_count(), 1);
        w.clear();
        assert_eq!(w.change_count(), 0);
    }

    #[test]
    fn test_clear_then_update_works() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(0, vec![0], 1000));
        w.update(snap(0, vec![1], 2000));
        w.clear();
        // After clear, the last_snapshot is still intact so next update still compares.
        let changes = w.update(snap(0, vec![2], 3000));
        assert_eq!(changes.len(), 1);
        assert_eq!(w.change_count(), 1);
    }

    #[test]
    fn test_clear_on_empty_log_is_safe() {
        let mut w = RegisterWatcher::new(100);
        w.clear(); // should not panic
        assert_eq!(w.change_count(), 0);
    }

    // ── has_changed ───────────────────────────────────────────────────────

    #[test]
    fn test_has_changed_true_after_change() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(5, vec![0], 1000));
        w.update(snap(5, vec![1], 2000));
        assert!(w.has_changed(5));
    }

    #[test]
    fn test_has_changed_false_no_change() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(5, vec![10], 1000));
        w.update(snap(5, vec![10], 2000));
        assert!(!w.has_changed(5));
    }

    #[test]
    fn test_has_changed_false_for_untracked_address() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(5, vec![0], 1000));
        w.update(snap(5, vec![1], 2000));
        assert!(!w.has_changed(99));
    }

    #[test]
    fn test_has_changed_false_after_clear() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(0, vec![0], 1000));
        w.update(snap(0, vec![1], 2000));
        w.clear();
        assert!(!w.has_changed(0));
    }

    // ── Multiple registers in a single snapshot ───────────────────────────

    #[test]
    fn test_multiple_register_changes_logged_correctly() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(100, vec![0, 0, 0, 0, 0], 1000));
        let changes = w.update(snap(100, vec![1, 0, 2, 0, 3], 2000));
        assert_eq!(changes.len(), 3);
        let addrs: Vec<u16> = changes.iter().map(|c| c.address).collect();
        assert!(addrs.contains(&100));
        assert!(addrs.contains(&102));
        assert!(addrs.contains(&104));
    }

    #[test]
    fn test_change_values_recorded_correctly() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(0, vec![0xABCD, 0x1234], 1000));
        let changes = w.update(snap(0, vec![0xABCD, 0x5678], 2000));
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].address, 1);
        assert_eq!(changes[0].old_value, 0x1234);
        assert_eq!(changes[0].new_value, 0x5678);
    }

    // ── Timestamp propagation ─────────────────────────────────────────────

    #[test]
    fn test_change_timestamp_matches_snapshot() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(0, vec![10], 1000));
        let changes = w.update(snap(0, vec![20], 9999));
        assert_eq!(changes[0].timestamp_ms, 9999);
    }

    #[test]
    fn test_multiple_updates_timestamp_per_change() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(0, vec![1], 1000));
        let c1 = w.update(snap(0, vec![2], 2000));
        let c2 = w.update(snap(0, vec![3], 3000));
        assert_eq!(c1[0].timestamp_ms, 2000);
        assert_eq!(c2[0].timestamp_ms, 3000);
    }

    // ── RegisterChange equality ───────────────────────────────────────────

    #[test]
    fn test_register_change_equality() {
        let c1 = RegisterChange {
            address: 5,
            old_value: 1,
            new_value: 2,
            timestamp_ms: 100,
        };
        let c2 = RegisterChange {
            address: 5,
            old_value: 1,
            new_value: 2,
            timestamp_ms: 100,
        };
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_register_change_inequality_address() {
        let c1 = RegisterChange {
            address: 5,
            old_value: 1,
            new_value: 2,
            timestamp_ms: 100,
        };
        let c2 = RegisterChange {
            address: 6,
            old_value: 1,
            new_value: 2,
            timestamp_ms: 100,
        };
        assert_ne!(c1, c2);
    }

    // ── Snapshot with high addresses ──────────────────────────────────────

    #[test]
    fn test_high_address_registers() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(0xFFF0, vec![100, 200], 1000));
        let changes = w.update(snap(0xFFF0, vec![100, 201], 2000));
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].address, 0xFFF1);
        assert_eq!(changes[0].old_value, 200);
        assert_eq!(changes[0].new_value, 201);
    }

    // ── returns_changes_but_does_not_log_when_size_zero ───────────────────

    #[test]
    fn test_changes_returned_even_when_log_size_zero() {
        let mut w = RegisterWatcher::new(0);
        w.update(snap(0, vec![1], 1000));
        let changes = w.update(snap(0, vec![2], 2000));
        // Changes are returned to caller even if not logged
        assert_eq!(changes.len(), 1);
        assert_eq!(w.change_count(), 0);
    }

    // ── All registers change ──────────────────────────────────────────────

    #[test]
    fn test_all_registers_change_simultaneously() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(0, vec![0, 0, 0], 1000));
        let changes = w.update(snap(0, vec![1, 2, 3], 2000));
        assert_eq!(changes.len(), 3);
    }

    // ── Snapshot with zero values ─────────────────────────────────────────

    #[test]
    fn test_change_from_nonzero_to_zero() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(0, vec![0xFFFF], 1000));
        let changes = w.update(snap(0, vec![0], 2000));
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].old_value, 0xFFFF);
        assert_eq!(changes[0].new_value, 0);
    }

    // ── Watcher with no snapshots returns no log data ─────────────────────

    #[test]
    fn test_watcher_no_updates_has_no_changed_addr() {
        let w = RegisterWatcher::new(100);
        assert!(!w.has_changed(0));
        assert_eq!(w.change_count(), 0);
    }

    // ── changes_for returns references in log order ───────────────────────

    #[test]
    fn test_changes_for_returns_in_insertion_order() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(0, vec![1], 1000));
        w.update(snap(0, vec![2], 2000));
        w.update(snap(0, vec![3], 3000));
        let for_0 = w.changes_for(0);
        assert_eq!(for_0.len(), 2);
        assert_eq!(for_0[0].new_value, 2);
        assert_eq!(for_0[1].new_value, 3);
    }

    // ── RegisterSnapshot new stores all fields ────────────────────────────

    #[test]
    fn test_snapshot_new_stores_all_fields() {
        let s = RegisterSnapshot::new(10, vec![1, 2, 3], 9876);
        assert_eq!(s.address_start, 10);
        assert_eq!(s.values, vec![1, 2, 3]);
        assert_eq!(s.timestamp_ms, 9876);
    }

    #[test]
    fn test_snapshot_get_boundary_values() {
        let s = snap(0, vec![0, 0xFFFF], 0);
        assert_eq!(s.get(0), Some(0));
        assert_eq!(s.get(1), Some(0xFFFF));
    }

    // ── max_log_size with multiple changes per update ─────────────────────

    #[test]
    fn test_max_log_size_with_multi_change_update() {
        let mut w = RegisterWatcher::new(2);
        w.update(snap(0, vec![0, 0], 1000));
        let changes = w.update(snap(0, vec![1, 2], 2000));
        assert_eq!(changes.len(), 2);
        assert_eq!(w.change_count(), 2);
    }

    // ── Value wraps around u16 max ────────────────────────────────────────

    #[test]
    fn test_value_wraps_u16_max() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(0, vec![0xFFFE], 1000));
        let changes = w.update(snap(0, vec![0xFFFF], 2000));
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].old_value, 0xFFFE);
        assert_eq!(changes[0].new_value, 0xFFFF);
    }

    // ── has_changed only for addresses that actually changed ──────────────

    #[test]
    fn test_has_changed_only_for_changed_addresses() {
        let mut w = RegisterWatcher::new(100);
        w.update(snap(0, vec![0, 0, 0], 1000));
        w.update(snap(0, vec![0, 1, 0], 2000));
        assert!(!w.has_changed(0));
        assert!(w.has_changed(1));
        assert!(!w.has_changed(2));
    }

    // ── Partial address overlap ───────────────────────────────────────────

    #[test]
    fn test_partial_overlap_only_shared_registers_compared() {
        let mut w = RegisterWatcher::new(100);
        // First snapshot covers addresses 5, 6, 7
        w.update(snap(5, vec![10, 20, 30], 1000));
        // New snapshot covers addresses 6, 7 only
        let changes = w.update(snap(6, vec![99, 30], 2000));
        // Address 6 changed (20→99); address 7 unchanged (30→30)
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].address, 6);
        assert_eq!(changes[0].old_value, 20);
        assert_eq!(changes[0].new_value, 99);
    }
}
