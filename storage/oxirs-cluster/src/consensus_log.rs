//! Simplified Raft-style consensus log.
//!
//! Implements the Raft log abstraction: append-only entries with monotonically
//! increasing indexes, term tracking, commit-index advancement, and sequential
//! application of committed entries.

/// A single entry in the consensus log.
#[derive(Debug, Clone, PartialEq)]
pub struct LogEntry {
    /// 1-based index within the log.
    pub index: u64,
    /// Raft term in which this entry was created.
    pub term: u64,
    /// Serialised command payload.
    pub command: Vec<u8>,
    /// Whether the leader has committed this entry (quorum acknowledged).
    pub is_committed: bool,
}

/// Result of an `append_entries` RPC call.
#[derive(Debug, Clone, PartialEq)]
pub struct AppendResult {
    /// Whether the append was accepted.
    pub success: bool,
    /// If `success` is `false`, the index of the conflicting entry.
    pub conflict_index: Option<u64>,
}

/// A Raft-style append-only log with commit tracking.
///
/// Indexes are 1-based (`last_index()` returns 0 for an empty log).
pub struct ConsensusLog {
    entries: Vec<LogEntry>,
    commit_index: u64,
    last_applied: u64,
    current_term: u64,
}

impl Default for ConsensusLog {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsensusLog {
    /// Create an empty log at term 0.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            commit_index: 0,
            last_applied: 0,
            current_term: 0,
        }
    }

    /// Append a new entry created by the leader in `term`.
    ///
    /// Returns the 1-based index assigned to the new entry.
    pub fn append(&mut self, term: u64, command: Vec<u8>) -> u64 {
        if term > self.current_term {
            self.current_term = term;
        }
        let index = self.entries.len() as u64 + 1;
        self.entries.push(LogEntry {
            index,
            term,
            command,
            is_committed: false,
        });
        index
    }

    /// Raft `AppendEntries` RPC handler.
    ///
    /// * `prev_index` — log index of the entry immediately preceding the new
    ///   entries (0 means "before the first entry").
    /// * `prev_term` — term of that entry.
    /// * `entries` — the new entries to append.
    ///
    /// Returns `AppendResult { success: false, conflict_index }` if the log
    /// doesn't contain an entry at `prev_index` with term `prev_term`.
    pub fn append_entries(
        &mut self,
        prev_index: u64,
        prev_term: u64,
        entries: Vec<LogEntry>,
    ) -> AppendResult {
        // Verify the predecessor entry
        if prev_index > 0 {
            match self.entry_at(prev_index) {
                None => {
                    return AppendResult {
                        success: false,
                        conflict_index: Some(prev_index),
                    };
                }
                Some(e) if e.term != prev_term => {
                    // Term mismatch — find the first index of the conflicting term
                    let conflict = self
                        .entries
                        .iter()
                        .find(|e2| e2.term == e.term)
                        .map(|e2| e2.index)
                        .unwrap_or(prev_index);
                    return AppendResult {
                        success: false,
                        conflict_index: Some(conflict),
                    };
                }
                _ => {}
            }
        }

        // Truncate any conflicting suffix
        for new_entry in &entries {
            if let Some(existing) = self.entry_at(new_entry.index) {
                if existing.term != new_entry.term {
                    // Truncate from this index onwards
                    let truncate_at = (new_entry.index - 1) as usize;
                    self.entries.truncate(truncate_at);
                    // Also reset commit_index if we truncated past it
                    if self.commit_index >= new_entry.index {
                        self.commit_index = truncate_at as u64;
                    }
                    break;
                }
            }
        }

        // Append entries that are not already present
        for new_entry in entries {
            if new_entry.index > self.last_index() {
                if new_entry.term > self.current_term {
                    self.current_term = new_entry.term;
                }
                self.entries.push(new_entry);
            }
        }

        AppendResult {
            success: true,
            conflict_index: None,
        }
    }

    /// Advance the commit index to `index` (or as far as the log allows).
    ///
    /// Returns the count of entries newly committed.
    pub fn commit_up_to(&mut self, index: u64) -> usize {
        let target = index.min(self.last_index());
        if target <= self.commit_index {
            return 0;
        }
        let prev = self.commit_index;
        for entry in self.entries.iter_mut() {
            if entry.index > prev && entry.index <= target {
                entry.is_committed = true;
            }
        }
        self.commit_index = target;
        (target - prev) as usize
    }

    /// Apply the next committed-but-not-yet-applied entry.
    ///
    /// Returns `None` if all committed entries have already been applied.
    pub fn apply_next(&mut self) -> Option<&LogEntry> {
        if self.last_applied >= self.commit_index {
            return None;
        }
        self.last_applied += 1;
        let idx = self.last_applied;
        self.entries.iter().find(|e| e.index == idx)
    }

    /// Index of the last entry in the log, or 0 if empty.
    pub fn last_index(&self) -> u64 {
        self.entries.last().map(|e| e.index).unwrap_or(0)
    }

    /// Term of the last entry in the log, or 0 if empty.
    pub fn last_term(&self) -> u64 {
        self.entries.last().map(|e| e.term).unwrap_or(0)
    }

    /// Current commit index (0 = nothing committed yet).
    pub fn commit_index(&self) -> u64 {
        self.commit_index
    }

    /// Index of the last applied entry (0 = nothing applied yet).
    pub fn last_applied(&self) -> u64 {
        self.last_applied
    }

    /// Return the entry at the given 1-based `index`, or `None`.
    pub fn entry_at(&self, index: u64) -> Option<&LogEntry> {
        if index == 0 || index > self.entries.len() as u64 {
            return None;
        }
        self.entries.get((index - 1) as usize)
    }

    /// Return a slice of all entries starting from 1-based `from_index`.
    /// Returns an empty slice if `from_index` is beyond the log.
    pub fn entries_from(&self, from_index: u64) -> &[LogEntry] {
        if from_index == 0 || from_index > self.entries.len() as u64 + 1 {
            return &[];
        }
        let start = (from_index - 1) as usize;
        if start >= self.entries.len() {
            &[]
        } else {
            &self.entries[start..]
        }
    }

    /// Return `true` if this log is at least as up-to-date as a peer whose
    /// last entry is `(other_index, other_term)`.
    ///
    /// "Up-to-date" per Raft §5.4.1:
    /// - Higher last term wins.
    /// - Equal terms: longer log wins.
    pub fn is_up_to_date(&self, other_index: u64, other_term: u64) -> bool {
        let my_term = self.last_term();
        let my_index = self.last_index();
        if my_term != other_term {
            my_term > other_term
        } else {
            my_index >= other_index
        }
    }

    /// Return the current term.
    pub fn current_term(&self) -> u64 {
        self.current_term
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(index: u64, term: u64, cmd: &str) -> LogEntry {
        LogEntry {
            index,
            term,
            command: cmd.as_bytes().to_vec(),
            is_committed: false,
        }
    }

    // --- new / empty state ---
    #[test]
    fn test_new_is_empty() {
        let log = ConsensusLog::new();
        assert_eq!(log.last_index(), 0);
        assert_eq!(log.last_term(), 0);
        assert_eq!(log.commit_index(), 0);
        assert_eq!(log.last_applied(), 0);
    }

    // --- append ---
    #[test]
    fn test_append_returns_index() {
        let mut log = ConsensusLog::new();
        let i1 = log.append(1, b"cmd1".to_vec());
        let i2 = log.append(1, b"cmd2".to_vec());
        assert_eq!(i1, 1);
        assert_eq!(i2, 2);
    }

    #[test]
    fn test_append_updates_last_index() {
        let mut log = ConsensusLog::new();
        log.append(1, b"cmd".to_vec());
        assert_eq!(log.last_index(), 1);
    }

    #[test]
    fn test_append_updates_last_term() {
        let mut log = ConsensusLog::new();
        log.append(3, b"cmd".to_vec());
        assert_eq!(log.last_term(), 3);
    }

    #[test]
    fn test_append_multiple() {
        let mut log = ConsensusLog::new();
        for i in 1u64..=5 {
            log.append(1, vec![i as u8]);
        }
        assert_eq!(log.last_index(), 5);
    }

    // --- entry_at ---
    #[test]
    fn test_entry_at_valid() {
        let mut log = ConsensusLog::new();
        log.append(2, b"hello".to_vec());
        let e = log.entry_at(1).expect("entry 1");
        assert_eq!(e.term, 2);
        assert_eq!(e.command, b"hello");
    }

    #[test]
    fn test_entry_at_zero_is_none() {
        let log = ConsensusLog::new();
        assert!(log.entry_at(0).is_none());
    }

    #[test]
    fn test_entry_at_beyond_is_none() {
        let mut log = ConsensusLog::new();
        log.append(1, b"x".to_vec());
        assert!(log.entry_at(2).is_none());
    }

    // --- entries_from ---
    #[test]
    fn test_entries_from_start() {
        let mut log = ConsensusLog::new();
        log.append(1, b"a".to_vec());
        log.append(1, b"b".to_vec());
        let slice = log.entries_from(1);
        assert_eq!(slice.len(), 2);
    }

    #[test]
    fn test_entries_from_middle() {
        let mut log = ConsensusLog::new();
        for _ in 0..4 {
            log.append(1, b"x".to_vec());
        }
        let slice = log.entries_from(3);
        assert_eq!(slice.len(), 2);
    }

    #[test]
    fn test_entries_from_beyond_is_empty() {
        let mut log = ConsensusLog::new();
        log.append(1, b"x".to_vec());
        assert!(log.entries_from(2).is_empty());
    }

    #[test]
    fn test_entries_from_zero_empty() {
        let log = ConsensusLog::new();
        assert!(log.entries_from(0).is_empty());
    }

    // --- commit_up_to ---
    #[test]
    fn test_commit_up_to_basic() {
        let mut log = ConsensusLog::new();
        log.append(1, b"a".to_vec());
        log.append(1, b"b".to_vec());
        log.append(1, b"c".to_vec());
        let n = log.commit_up_to(2);
        assert_eq!(n, 2);
        assert_eq!(log.commit_index(), 2);
    }

    #[test]
    fn test_commit_up_to_marks_committed() {
        let mut log = ConsensusLog::new();
        log.append(1, b"a".to_vec());
        log.commit_up_to(1);
        assert!(log.entry_at(1).unwrap().is_committed);
    }

    #[test]
    fn test_commit_up_to_beyond_log() {
        let mut log = ConsensusLog::new();
        log.append(1, b"a".to_vec());
        let n = log.commit_up_to(999);
        assert_eq!(n, 1);
        assert_eq!(log.commit_index(), 1);
    }

    #[test]
    fn test_commit_up_to_no_regression() {
        let mut log = ConsensusLog::new();
        log.append(1, b"a".to_vec());
        log.append(1, b"b".to_vec());
        log.commit_up_to(2);
        let n = log.commit_up_to(1); // lower than current
        assert_eq!(n, 0);
        assert_eq!(log.commit_index(), 2);
    }

    // --- apply_next ---
    #[test]
    fn test_apply_next_returns_none_if_nothing_committed() {
        let mut log = ConsensusLog::new();
        log.append(1, b"a".to_vec());
        assert!(log.apply_next().is_none());
    }

    #[test]
    fn test_apply_next_after_commit() {
        let mut log = ConsensusLog::new();
        log.append(1, b"cmd1".to_vec());
        log.commit_up_to(1);
        let entry = log.apply_next().expect("should apply");
        assert_eq!(entry.index, 1);
        assert_eq!(log.last_applied(), 1);
    }

    #[test]
    fn test_apply_next_sequential() {
        let mut log = ConsensusLog::new();
        for i in 0..3 {
            log.append(1, vec![i]);
        }
        log.commit_up_to(3);
        for expected_idx in 1u64..=3 {
            let e = log.apply_next().expect("entry");
            assert_eq!(e.index, expected_idx);
        }
        assert!(log.apply_next().is_none());
    }

    // --- append_entries ---
    #[test]
    fn test_append_entries_empty_log() {
        let mut log = ConsensusLog::new();
        let entries = vec![make_entry(1, 1, "cmd1"), make_entry(2, 1, "cmd2")];
        let result = log.append_entries(0, 0, entries);
        assert!(result.success);
        assert_eq!(log.last_index(), 2);
    }

    #[test]
    fn test_append_entries_consistency_check_fails() {
        let mut log = ConsensusLog::new();
        log.append(1, b"x".to_vec());
        // prev_index=5 but log only has 1 entry
        let result = log.append_entries(5, 1, vec![]);
        assert!(!result.success);
        assert!(result.conflict_index.is_some());
    }

    #[test]
    fn test_append_entries_term_mismatch() {
        let mut log = ConsensusLog::new();
        log.append(1, b"x".to_vec()); // term=1, index=1
        // prev says term=2 but actual is term=1
        let result = log.append_entries(1, 2, vec![]);
        assert!(!result.success);
    }

    #[test]
    fn test_append_entries_valid_extension() {
        let mut log = ConsensusLog::new();
        log.append(1, b"x".to_vec()); // index=1, term=1
        let entries = vec![make_entry(2, 1, "y")];
        let result = log.append_entries(1, 1, entries);
        assert!(result.success);
        assert_eq!(log.last_index(), 2);
    }

    // --- is_up_to_date ---
    #[test]
    fn test_is_up_to_date_higher_term() {
        let mut log = ConsensusLog::new();
        log.append(3, b"x".to_vec());
        assert!(log.is_up_to_date(100, 2)); // our term=3, peer term=2
    }

    #[test]
    fn test_is_up_to_date_lower_term() {
        let mut log = ConsensusLog::new();
        log.append(1, b"x".to_vec());
        assert!(!log.is_up_to_date(1, 2)); // our term=1, peer term=2
    }

    #[test]
    fn test_is_up_to_date_same_term_longer() {
        let mut log = ConsensusLog::new();
        log.append(1, b"a".to_vec());
        log.append(1, b"b".to_vec());
        assert!(log.is_up_to_date(1, 1)); // same term, we are longer
    }

    #[test]
    fn test_is_up_to_date_same_term_shorter() {
        let mut log = ConsensusLog::new();
        log.append(1, b"a".to_vec());
        assert!(!log.is_up_to_date(5, 1)); // same term, peer is longer
    }

    #[test]
    fn test_is_up_to_date_empty_log() {
        let log = ConsensusLog::new();
        assert!(!log.is_up_to_date(1, 1));
        assert!(log.is_up_to_date(0, 0));
    }

    // --- current_term updated by append ---
    #[test]
    fn test_current_term_updates() {
        let mut log = ConsensusLog::new();
        log.append(1, b"x".to_vec());
        assert_eq!(log.current_term(), 1);
        log.append(5, b"y".to_vec());
        assert_eq!(log.current_term(), 5);
    }

    // --- commit does not re-commit already committed ---
    #[test]
    fn test_commit_idempotent() {
        let mut log = ConsensusLog::new();
        log.append(1, b"a".to_vec());
        log.append(1, b"b".to_vec());
        log.commit_up_to(2);
        let n = log.commit_up_to(2);
        assert_eq!(n, 0);
    }
}
