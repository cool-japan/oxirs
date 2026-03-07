//! Raft log replication state machine.
//!
//! Implements an in-memory Raft log supporting append, commit, apply,
//! truncation, and the `AppendEntries` RPC handler described in the Raft paper.

// ────────────────────────────────────────────────────────────────────────────
// Entry types
// ────────────────────────────────────────────────────────────────────────────

/// The type of a log entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntryType {
    /// Regular application command.
    Normal,
    /// Cluster membership / configuration change.
    Configuration,
    /// No-op entry appended by a new leader to commit previous-term entries.
    Noop,
}

/// A single entry in the Raft log.
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// 1-based position in the log.
    pub index: u64,
    /// Term in which this entry was created.
    pub term: u64,
    /// Opaque command bytes.
    pub data: Vec<u8>,
    pub entry_type: EntryType,
}

// ────────────────────────────────────────────────────────────────────────────
// Replication state for a peer
// ────────────────────────────────────────────────────────────────────────────

/// Leader's view of a follower's replication progress.
#[derive(Debug, Clone)]
pub struct ReplicationState {
    pub peer_id: String,
    /// Next index to send to this peer.
    pub next_index: u64,
    /// Highest index known to be replicated on this peer.
    pub match_index: u64,
    /// Number of in-flight AppendEntries RPCs.
    pub in_flight: usize,
}

// ────────────────────────────────────────────────────────────────────────────
// AppendEntries RPC
// ────────────────────────────────────────────────────────────────────────────

/// Arguments for the `AppendEntries` RPC.
#[derive(Debug, Clone)]
pub struct AppendEntriesRequest {
    pub leader_id: String,
    pub term: u64,
    /// Index of the log entry immediately before the new ones.
    pub prev_log_index: u64,
    /// Term of `prev_log_index`.
    pub prev_log_term: u64,
    pub entries: Vec<LogEntry>,
    /// Highest log index the leader has committed.
    pub leader_commit: u64,
}

/// Response to an `AppendEntries` RPC.
#[derive(Debug, Clone)]
pub struct AppendEntriesResponse {
    pub term: u64,
    pub success: bool,
    /// Highest index that the follower now has in its log (on success).
    pub match_index: u64,
    /// Hint for the leader: the earliest conflicting index (on failure).
    pub conflict_index: Option<u64>,
}

// ────────────────────────────────────────────────────────────────────────────
// Raft log
// ────────────────────────────────────────────────────────────────────────────

/// An in-memory Raft log with commit / apply tracking.
///
/// Log indices are **1-based** (index 0 is a sentinel meaning "nothing").
/// Entries are stored in `self.entries` starting at index 0 (which corresponds
/// to log index 1).
pub struct RaftLog {
    entries: Vec<LogEntry>,
    commit_index: u64,
    last_applied: u64,
    current_term: u64,
}

impl Default for RaftLog {
    fn default() -> Self {
        Self::new()
    }
}

impl RaftLog {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            commit_index: 0,
            last_applied: 0,
            current_term: 0,
        }
    }

    // ── append ───────────────────────────────────────────────────────────────

    /// Append a new entry to the log.  Returns the 1-based log index.
    pub fn append(&mut self, data: Vec<u8>, term: u64, entry_type: EntryType) -> u64 {
        let index = self.entries.len() as u64 + 1;
        if term > self.current_term {
            self.current_term = term;
        }
        self.entries.push(LogEntry {
            index,
            term,
            data,
            entry_type,
        });
        index
    }

    // ── getters ──────────────────────────────────────────────────────────────

    /// Retrieve the entry at the given 1-based log index.
    pub fn get(&self, index: u64) -> Option<&LogEntry> {
        if index == 0 {
            return None;
        }
        self.entries.get((index - 1) as usize)
    }

    /// 1-based index of the last entry, or 0 if the log is empty.
    pub fn last_index(&self) -> u64 {
        self.entries.len() as u64
    }

    /// Term of the last entry, or 0 if the log is empty.
    pub fn last_term(&self) -> u64 {
        self.entries.last().map(|e| e.term).unwrap_or(0)
    }

    pub fn commit_index(&self) -> u64 {
        self.commit_index
    }

    pub fn last_applied(&self) -> u64 {
        self.last_applied
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // ── commit / apply ───────────────────────────────────────────────────────

    /// Advance `commit_index` to `min(index, last_index)`.
    ///
    /// Returns references to newly committed entries (those between the old
    /// commit index and the new one, inclusive).
    pub fn commit(&mut self, index: u64) -> Vec<&LogEntry> {
        let capped = index.min(self.last_index());
        if capped <= self.commit_index {
            return vec![];
        }
        let old = self.commit_index;
        self.commit_index = capped;
        self.entries[(old as usize)..(self.commit_index as usize)]
            .iter()
            .collect()
    }

    /// Advance `last_applied` to `commit_index`, returning entries that were applied.
    pub fn apply(&mut self) -> Vec<&LogEntry> {
        if self.last_applied >= self.commit_index {
            return vec![];
        }
        let old = self.last_applied;
        self.last_applied = self.commit_index;
        self.entries[(old as usize)..(self.last_applied as usize)]
            .iter()
            .collect()
    }

    // ── truncation ───────────────────────────────────────────────────────────

    /// Remove all entries at index `>= from_index` (1-based).
    ///
    /// This is used when a follower discovers a conflict during `AppendEntries`.
    pub fn truncate_from(&mut self, index: u64) {
        if index == 0 || index > self.last_index() {
            return;
        }
        let pos = (index - 1) as usize;
        self.entries.truncate(pos);
        // Adjust commit_index and last_applied to not exceed the new last_index
        self.commit_index = self.commit_index.min(self.last_index());
        self.last_applied = self.last_applied.min(self.commit_index);
    }

    // ── AppendEntries RPC ────────────────────────────────────────────────────

    /// Handle an `AppendEntries` request from a Raft leader.
    ///
    /// Returns a response indicating success or failure, with a conflict hint.
    pub fn handle_append_entries(
        &mut self,
        req: AppendEntriesRequest,
    ) -> AppendEntriesResponse {
        // Term check
        if req.term < self.current_term {
            return AppendEntriesResponse {
                term: self.current_term,
                success: false,
                match_index: 0,
                conflict_index: None,
            };
        }
        if req.term > self.current_term {
            self.current_term = req.term;
        }

        // Consistency check: verify prev_log_index / prev_log_term
        if req.prev_log_index > 0 {
            match self.get(req.prev_log_index) {
                None => {
                    // Log too short
                    return AppendEntriesResponse {
                        term: self.current_term,
                        success: false,
                        match_index: self.last_index(),
                        conflict_index: Some(self.last_index() + 1),
                    };
                }
                Some(e) if e.term != req.prev_log_term => {
                    // Term mismatch — find first entry of the conflicting term
                    let conflict_term = e.term;
                    let conflict_index = self
                        .entries
                        .iter()
                        .find(|x| x.term == conflict_term)
                        .map(|x| x.index)
                        .unwrap_or(req.prev_log_index);
                    self.truncate_from(req.prev_log_index);
                    return AppendEntriesResponse {
                        term: self.current_term,
                        success: false,
                        match_index: self.last_index(),
                        conflict_index: Some(conflict_index),
                    };
                }
                _ => {} // ok
            }
        }

        // Append new entries (overwrite conflicting tail if needed)
        let mut insert_pos = req.prev_log_index + 1; // 1-based
        for entry in req.entries {
            match self.get(insert_pos) {
                Some(existing) if existing.term != entry.term => {
                    // Conflict — truncate from here and append
                    self.truncate_from(insert_pos);
                    self.entries.push(LogEntry {
                        index: self.entries.len() as u64 + 1,
                        ..entry
                    });
                }
                None => {
                    // Past end of log — just push
                    self.entries.push(LogEntry {
                        index: self.entries.len() as u64 + 1,
                        ..entry
                    });
                }
                _ => {} // entry already present with matching term; skip
            }
            insert_pos += 1;
        }

        // Advance commit index
        if req.leader_commit > self.commit_index {
            self.commit_index = req.leader_commit.min(self.last_index());
        }

        AppendEntriesResponse {
            term: self.current_term,
            success: true,
            match_index: self.last_index(),
            conflict_index: None,
        }
    }

    // ── slice ────────────────────────────────────────────────────────────────

    /// Returns a slice of entries starting from the given 1-based index.
    pub fn entries_from(&self, from_index: u64) -> &[LogEntry] {
        if from_index == 0 || from_index > self.last_index() {
            return &[];
        }
        &self.entries[(from_index - 1) as usize..]
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn make_req(
        prev_index: u64,
        prev_term: u64,
        entries: Vec<(Vec<u8>, u64)>,
        leader_commit: u64,
    ) -> AppendEntriesRequest {
        AppendEntriesRequest {
            leader_id: "leader1".to_string(),
            term: 1,
            prev_log_index: prev_index,
            prev_log_term: prev_term,
            entries: entries
                .into_iter()
                .enumerate()
                .map(|(i, (data, term))| LogEntry {
                    index: prev_index + i as u64 + 1,
                    term,
                    data,
                    entry_type: EntryType::Normal,
                })
                .collect(),
            leader_commit,
        }
    }

    // ── append / basic accessors ──────────────────────────────────────────────

    #[test]
    fn test_append_returns_increasing_index() {
        let mut log = RaftLog::new();
        let i1 = log.append(b"cmd1".to_vec(), 1, EntryType::Normal);
        let i2 = log.append(b"cmd2".to_vec(), 1, EntryType::Normal);
        assert_eq!(i1, 1);
        assert_eq!(i2, 2);
    }

    #[test]
    fn test_empty_log_properties() {
        let log = RaftLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
        assert_eq!(log.last_index(), 0);
        assert_eq!(log.last_term(), 0);
        assert_eq!(log.commit_index(), 0);
        assert_eq!(log.last_applied(), 0);
    }

    #[test]
    fn test_get_by_index() {
        let mut log = RaftLog::new();
        log.append(b"hello".to_vec(), 1, EntryType::Normal);
        let e = log.get(1).expect("entry at index 1");
        assert_eq!(e.data, b"hello");
        assert_eq!(e.term, 1);
    }

    #[test]
    fn test_get_out_of_bounds_returns_none() {
        let log = RaftLog::new();
        assert!(log.get(1).is_none());
        assert!(log.get(0).is_none());
    }

    #[test]
    fn test_last_index_and_term() {
        let mut log = RaftLog::new();
        log.append(b"a".to_vec(), 1, EntryType::Normal);
        log.append(b"b".to_vec(), 2, EntryType::Normal);
        assert_eq!(log.last_index(), 2);
        assert_eq!(log.last_term(), 2);
    }

    #[test]
    fn test_append_noop_entry() {
        let mut log = RaftLog::new();
        let idx = log.append(vec![], 1, EntryType::Noop);
        assert_eq!(idx, 1);
        let e = log.get(1).expect("noop entry");
        assert_eq!(e.entry_type, EntryType::Noop);
    }

    #[test]
    fn test_append_configuration_entry() {
        let mut log = RaftLog::new();
        let idx = log.append(b"new_config".to_vec(), 1, EntryType::Configuration);
        assert_eq!(idx, 1);
        let e = log.get(1).expect("config entry");
        assert_eq!(e.entry_type, EntryType::Configuration);
    }

    // ── commit ────────────────────────────────────────────────────────────────

    #[test]
    fn test_commit_advances_commit_index() {
        let mut log = RaftLog::new();
        log.append(b"a".to_vec(), 1, EntryType::Normal);
        log.append(b"b".to_vec(), 1, EntryType::Normal);
        log.commit(2);
        assert_eq!(log.commit_index(), 2);
    }

    #[test]
    fn test_commit_capped_at_last_index() {
        let mut log = RaftLog::new();
        log.append(b"a".to_vec(), 1, EntryType::Normal);
        log.commit(100);
        assert_eq!(log.commit_index(), 1);
    }

    #[test]
    fn test_commit_returns_newly_committed_entries() {
        let mut log = RaftLog::new();
        log.append(b"a".to_vec(), 1, EntryType::Normal);
        log.append(b"b".to_vec(), 1, EntryType::Normal);
        log.append(b"c".to_vec(), 1, EntryType::Normal);
        let committed = log.commit(3);
        assert_eq!(committed.len(), 3);
    }

    #[test]
    fn test_commit_no_new_entries_when_already_committed() {
        let mut log = RaftLog::new();
        log.append(b"a".to_vec(), 1, EntryType::Normal);
        log.commit(1);
        let committed = log.commit(1);
        assert!(committed.is_empty());
    }

    // ── apply ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_apply_advances_last_applied() {
        let mut log = RaftLog::new();
        log.append(b"a".to_vec(), 1, EntryType::Normal);
        log.commit(1);
        log.apply();
        assert_eq!(log.last_applied(), 1);
    }

    #[test]
    fn test_apply_returns_applied_entries() {
        let mut log = RaftLog::new();
        log.append(b"x".to_vec(), 1, EntryType::Normal);
        log.append(b"y".to_vec(), 1, EntryType::Normal);
        log.commit(2);
        let applied = log.apply();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_apply_nothing_when_not_committed() {
        let mut log = RaftLog::new();
        log.append(b"a".to_vec(), 1, EntryType::Normal);
        let applied = log.apply();
        assert!(applied.is_empty());
    }

    // ── truncate_from ─────────────────────────────────────────────────────────

    #[test]
    fn test_truncate_removes_tail() {
        let mut log = RaftLog::new();
        log.append(b"a".to_vec(), 1, EntryType::Normal);
        log.append(b"b".to_vec(), 1, EntryType::Normal);
        log.append(b"c".to_vec(), 1, EntryType::Normal);
        log.truncate_from(2);
        assert_eq!(log.last_index(), 1);
        assert!(log.get(2).is_none());
    }

    #[test]
    fn test_truncate_out_of_range_is_noop() {
        let mut log = RaftLog::new();
        log.append(b"a".to_vec(), 1, EntryType::Normal);
        log.truncate_from(10); // beyond log
        assert_eq!(log.last_index(), 1);
    }

    #[test]
    fn test_truncate_adjusts_commit_index() {
        let mut log = RaftLog::new();
        log.append(b"a".to_vec(), 1, EntryType::Normal);
        log.append(b"b".to_vec(), 1, EntryType::Normal);
        log.commit(2);
        log.truncate_from(1);
        assert_eq!(log.commit_index(), 0);
    }

    // ── entries_from ──────────────────────────────────────────────────────────

    #[test]
    fn test_entries_from_beginning() {
        let mut log = RaftLog::new();
        log.append(b"a".to_vec(), 1, EntryType::Normal);
        log.append(b"b".to_vec(), 1, EntryType::Normal);
        let slice = log.entries_from(1);
        assert_eq!(slice.len(), 2);
    }

    #[test]
    fn test_entries_from_middle() {
        let mut log = RaftLog::new();
        for i in 0..5u8 {
            log.append(vec![i], 1, EntryType::Normal);
        }
        let slice = log.entries_from(3);
        assert_eq!(slice.len(), 3);
        assert_eq!(slice[0].data, vec![2]);
    }

    #[test]
    fn test_entries_from_beyond_log_is_empty() {
        let mut log = RaftLog::new();
        log.append(b"a".to_vec(), 1, EntryType::Normal);
        assert!(log.entries_from(5).is_empty());
    }

    #[test]
    fn test_entries_from_zero_is_empty() {
        let mut log = RaftLog::new();
        log.append(b"a".to_vec(), 1, EntryType::Normal);
        assert!(log.entries_from(0).is_empty());
    }

    // ── handle_append_entries ────────────────────────────────────────────────

    #[test]
    fn test_append_entries_empty_to_empty_log() {
        let mut log = RaftLog::new();
        let req = make_req(0, 0, vec![], 0);
        let resp = log.handle_append_entries(req);
        assert!(resp.success);
        assert_eq!(resp.match_index, 0);
    }

    #[test]
    fn test_append_entries_adds_new_entries() {
        let mut log = RaftLog::new();
        let req = make_req(0, 0, vec![(b"a".to_vec(), 1), (b"b".to_vec(), 1)], 0);
        let resp = log.handle_append_entries(req);
        assert!(resp.success);
        assert_eq!(resp.match_index, 2);
        assert_eq!(log.last_index(), 2);
    }

    #[test]
    fn test_append_entries_prev_mismatch_fails() {
        let mut log = RaftLog::new();
        log.append(b"a".to_vec(), 1, EntryType::Normal);
        // prev_log_index=2 but log only has 1 entry
        let req = make_req(2, 1, vec![], 0);
        let resp = log.handle_append_entries(req);
        assert!(!resp.success);
        assert!(resp.conflict_index.is_some());
    }

    #[test]
    fn test_append_entries_term_mismatch_fails() {
        let mut log = RaftLog::new();
        log.append(b"a".to_vec(), 2, EntryType::Normal); // term 2
        // Leader says prev was at term 1, but we have term 2
        let req = AppendEntriesRequest {
            leader_id: "L".to_string(),
            term: 3,
            prev_log_index: 1,
            prev_log_term: 1, // mismatch: actual is 2
            entries: vec![],
            leader_commit: 0,
        };
        let resp = log.handle_append_entries(req);
        assert!(!resp.success);
        assert!(resp.conflict_index.is_some());
    }

    #[test]
    fn test_append_entries_advances_commit_index() {
        let mut log = RaftLog::new();
        log.append(b"a".to_vec(), 1, EntryType::Normal);
        log.append(b"b".to_vec(), 1, EntryType::Normal);
        let req = make_req(2, 1, vec![], 2);
        log.handle_append_entries(req);
        assert_eq!(log.commit_index(), 2);
    }

    #[test]
    fn test_append_entries_stale_term_rejected() {
        let mut log = RaftLog::new();
        log.append(b"a".to_vec(), 5, EntryType::Normal);
        // Request with lower term
        let req = AppendEntriesRequest {
            leader_id: "old_leader".to_string(),
            term: 3, // < current_term=5
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
        };
        let resp = log.handle_append_entries(req);
        assert!(!resp.success);
        assert_eq!(resp.term, 5);
    }

    #[test]
    fn test_conflict_index_returned_on_short_log() {
        let mut log = RaftLog::new();
        let req = make_req(5, 1, vec![], 0);
        let resp = log.handle_append_entries(req);
        assert!(!resp.success);
        let ci = resp.conflict_index.expect("conflict_index should be set");
        assert!(ci > 0);
    }

    // ── len / is_empty ────────────────────────────────────────────────────────

    #[test]
    fn test_len_increases_with_appends() {
        let mut log = RaftLog::new();
        assert_eq!(log.len(), 0);
        log.append(b"x".to_vec(), 1, EntryType::Normal);
        assert_eq!(log.len(), 1);
        log.append(b"y".to_vec(), 1, EntryType::Normal);
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_is_empty_false_after_append() {
        let mut log = RaftLog::new();
        log.append(b"a".to_vec(), 1, EntryType::Normal);
        assert!(!log.is_empty());
    }

    #[test]
    fn test_full_lifecycle() {
        let mut log = RaftLog::new();
        // Append
        log.append(b"write_key=1".to_vec(), 1, EntryType::Normal);
        log.append(b"write_key=2".to_vec(), 1, EntryType::Normal);
        log.append(b"write_key=3".to_vec(), 1, EntryType::Normal);
        // Commit up to 2
        let committed = log.commit(2);
        assert_eq!(committed.len(), 2);
        // Apply
        let applied = log.apply();
        assert_eq!(applied.len(), 2);
        assert_eq!(log.last_applied(), 2);
        // Commit the rest
        log.commit(3);
        log.apply();
        assert_eq!(log.last_applied(), 3);
    }
}
