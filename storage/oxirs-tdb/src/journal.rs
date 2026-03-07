//! Write-ahead journal for crash recovery
//!
//! Implements a simple append-only journal that records Begin/Write/Commit/Abort/
//! Checkpoint entries. On recovery, the journal replays only the writes belonging
//! to committed transactions that occurred after the last checkpoint.

use std::collections::{HashMap, HashSet};

// ─── JournalEntry ─────────────────────────────────────────────────────────────

/// The payload of a single journal record
#[derive(Debug, Clone)]
pub enum JournalEntry {
    /// Begin a transaction with the given ID
    Begin(u64),
    /// Write a triple (insert or delete) under a transaction
    Write {
        /// Owning transaction ID
        tx_id: u64,
        /// (subject, predicate, object)
        triple: (String, String, String),
        /// Named graph (None = default graph)
        graph: Option<String>,
        /// true = insert, false = delete
        is_insert: bool,
    },
    /// Commit a transaction
    Commit(u64),
    /// Abort (roll back) a transaction
    Abort(u64),
    /// Checkpoint: all preceding committed state is durable
    Checkpoint(u64),
}

// ─── JournalRecord ────────────────────────────────────────────────────────────

/// A full journal record with sequence number and checksum
#[derive(Debug, Clone)]
pub struct JournalRecord {
    /// Log sequence number (monotonically increasing)
    pub lsn: u64,
    /// The journal entry payload
    pub entry: JournalEntry,
    /// CRC32-like checksum of the serialised entry
    pub checksum: u32,
}

// ─── JournalStats ─────────────────────────────────────────────────────────────

/// Snapshot statistics about the journal
#[derive(Debug, Clone, Default)]
pub struct JournalStats {
    /// Total records written
    pub total_records: usize,
    /// Number of transactions that reached Commit
    pub committed_txs: usize,
    /// Number of transactions that reached Abort
    pub aborted_txs: usize,
    /// Transactions that have been started but not yet committed or aborted
    pub pending_txs: usize,
}

// ─── Journal ─────────────────────────────────────────────────────────────────

/// An in-memory write-ahead journal
pub struct Journal {
    records: Vec<JournalRecord>,
    next_lsn: u64,
    committed_lsn_val: u64,
    /// LSN of the most recent Checkpoint record (0 if none)
    checkpoint_lsn: u64,
}

impl Default for Journal {
    fn default() -> Self {
        Self::new()
    }
}

impl Journal {
    /// Create an empty journal
    pub fn new() -> Self {
        Journal {
            records: Vec::new(),
            next_lsn: 1,
            committed_lsn_val: 0,
            checkpoint_lsn: 0,
        }
    }

    /// Append a journal entry; returns the LSN assigned to this record
    pub fn append(&mut self, entry: JournalEntry) -> u64 {
        let lsn = self.next_lsn;
        self.next_lsn += 1;
        let checksum = compute_entry_checksum(lsn, &entry);
        self.records.push(JournalRecord {
            lsn,
            entry,
            checksum,
        });
        lsn
    }

    /// Mark a transaction as committed; returns `true` if a matching Begin was found
    pub fn commit(&mut self, tx_id: u64) -> bool {
        let has_begin = self
            .records
            .iter()
            .any(|r| matches!(&r.entry, JournalEntry::Begin(tid) if *tid == tx_id));
        if !has_begin {
            return false;
        }
        let lsn = self.append(JournalEntry::Commit(tx_id));
        self.committed_lsn_val = lsn;
        true
    }

    /// Mark a transaction as aborted; returns `true` if a matching Begin was found
    pub fn abort(&mut self, tx_id: u64) -> bool {
        let has_begin = self
            .records
            .iter()
            .any(|r| matches!(&r.entry, JournalEntry::Begin(tid) if *tid == tx_id));
        if !has_begin {
            return false;
        }
        self.append(JournalEntry::Abort(tx_id));
        true
    }

    /// Write a Checkpoint record; returns the checkpoint LSN
    pub fn checkpoint(&mut self) -> u64 {
        let lsn = self.append(JournalEntry::Checkpoint(self.next_lsn));
        self.checkpoint_lsn = lsn;
        lsn
    }

    /// Replay committed writes since the last checkpoint.
    ///
    /// Returns all `(subject, predicate, object, graph)` tuples for Write entries
    /// where `is_insert == true` and the transaction was committed, occurring after
    /// the most recent Checkpoint.
    pub fn recover(&self) -> Vec<(String, String, String, Option<String>)> {
        // Find the position in records for the last checkpoint
        let start_pos = self
            .records
            .iter()
            .rposition(|r| matches!(r.entry, JournalEntry::Checkpoint(_)))
            .map(|p| p + 1)
            .unwrap_or(0);

        let post_checkpoint = &self.records[start_pos..];

        // Build the set of committed transaction IDs
        let committed: HashSet<u64> = post_checkpoint
            .iter()
            .filter_map(|r| {
                if let JournalEntry::Commit(tid) = r.entry {
                    Some(tid)
                } else {
                    None
                }
            })
            .collect();

        // Collect all inserts from committed transactions in order
        post_checkpoint
            .iter()
            .filter_map(|r| {
                if let JournalEntry::Write {
                    tx_id,
                    triple,
                    graph,
                    is_insert,
                } = &r.entry
                {
                    if *is_insert && committed.contains(tx_id) {
                        return Some((
                            triple.0.clone(),
                            triple.1.clone(),
                            triple.2.clone(),
                            graph.clone(),
                        ));
                    }
                }
                None
            })
            .collect()
    }

    /// Returns the IDs of transactions that have a Begin but no Commit or Abort
    pub fn pending_transactions(&self) -> Vec<u64> {
        let mut started: Vec<u64> = Vec::new();
        let mut finished: HashSet<u64> = HashSet::new();
        for r in &self.records {
            match r.entry {
                JournalEntry::Begin(tid) => started.push(tid),
                JournalEntry::Commit(tid) | JournalEntry::Abort(tid) => {
                    finished.insert(tid);
                }
                _ => {}
            }
        }
        started
            .into_iter()
            .filter(|id| !finished.contains(id))
            .collect::<std::collections::HashSet<u64>>()
            .into_iter()
            .collect()
    }

    /// Collect statistics about this journal
    pub fn stats(&self) -> JournalStats {
        let mut begun: HashMap<u64, bool> = HashMap::new();
        let mut committed_count = 0usize;
        let mut aborted_count = 0usize;

        for r in &self.records {
            match r.entry {
                JournalEntry::Begin(tid) => {
                    begun.entry(tid).or_insert(false);
                }
                JournalEntry::Commit(tid) => {
                    if begun.contains_key(&tid) {
                        *begun.get_mut(&tid).unwrap_or(&mut false) = true;
                        committed_count += 1;
                    }
                }
                JournalEntry::Abort(tid) => {
                    if begun.contains_key(&tid) {
                        begun.remove(&tid);
                        aborted_count += 1;
                    }
                }
                _ => {}
            }
        }
        let pending = begun.values().filter(|committed| !**committed).count();
        JournalStats {
            total_records: self.records.len(),
            committed_txs: committed_count,
            aborted_txs: aborted_count,
            pending_txs: pending,
        }
    }

    /// All records whose LSN is >= `lsn`
    pub fn records_since(&self, lsn: u64) -> &[JournalRecord] {
        let pos = self.records.partition_point(|r| r.lsn < lsn);
        &self.records[pos..]
    }

    /// Verify the checksum of a single record
    pub fn verify_checksum(record: &JournalRecord) -> bool {
        let expected = compute_entry_checksum(record.lsn, &record.entry);
        record.checksum == expected
    }

    /// The LSN of the most recently committed transaction record
    pub fn committed_lsn(&self) -> u64 {
        self.committed_lsn_val
    }

    /// Total number of records in the journal
    pub fn record_count(&self) -> usize {
        self.records.len()
    }
}

// ─── CRC32 checksum ───────────────────────────────────────────────────────────

/// Compute a CRC32-like checksum (Castagnoli polynomial 0xEDB88320)
pub fn crc32_checksum(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xffff_ffff;
    for byte in data {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            if crc & 1 == 1 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Compute checksum for a journal entry given its LSN
fn compute_entry_checksum(lsn: u64, entry: &JournalEntry) -> u32 {
    let mut data = Vec::with_capacity(64);
    // Encode LSN
    data.extend_from_slice(&lsn.to_le_bytes());
    // Encode the entry discriminant + key fields
    match entry {
        JournalEntry::Begin(tid) => {
            data.push(0);
            data.extend_from_slice(&tid.to_le_bytes());
        }
        JournalEntry::Write {
            tx_id,
            triple: (s, p, o),
            graph,
            is_insert,
        } => {
            data.push(1);
            data.extend_from_slice(&tx_id.to_le_bytes());
            data.extend_from_slice(s.as_bytes());
            data.extend_from_slice(p.as_bytes());
            data.extend_from_slice(o.as_bytes());
            if let Some(g) = graph {
                data.extend_from_slice(g.as_bytes());
            }
            data.push(u8::from(*is_insert));
        }
        JournalEntry::Commit(tid) => {
            data.push(2);
            data.extend_from_slice(&tid.to_le_bytes());
        }
        JournalEntry::Abort(tid) => {
            data.push(3);
            data.extend_from_slice(&tid.to_le_bytes());
        }
        JournalEntry::Checkpoint(val) => {
            data.push(4);
            data.extend_from_slice(&val.to_le_bytes());
        }
    }
    crc32_checksum(&data)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn triple(s: &str, p: &str, o: &str) -> (String, String, String) {
        (s.to_string(), p.to_string(), o.to_string())
    }

    // ── append / basic LSN ───────────────────────────────────────────────────

    #[test]
    fn test_append_returns_sequential_lsn() {
        let mut j = Journal::new();
        let l1 = j.append(JournalEntry::Begin(1));
        let l2 = j.append(JournalEntry::Begin(2));
        assert_eq!(l1, 1);
        assert_eq!(l2, 2);
    }

    #[test]
    fn test_record_count_increases() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(1));
        j.append(JournalEntry::Begin(2));
        assert_eq!(j.record_count(), 2);
    }

    // ── commit / abort ───────────────────────────────────────────────────────

    #[test]
    fn test_commit_returns_true_for_known_tx() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(42));
        assert!(j.commit(42));
    }

    #[test]
    fn test_commit_returns_false_for_unknown_tx() {
        let mut j = Journal::new();
        assert!(!j.commit(99));
    }

    #[test]
    fn test_abort_returns_true_for_known_tx() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(7));
        assert!(j.abort(7));
    }

    #[test]
    fn test_abort_returns_false_for_unknown_tx() {
        let mut j = Journal::new();
        assert!(!j.abort(99));
    }

    #[test]
    fn test_committed_lsn_updated_after_commit() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(1));
        j.commit(1);
        assert!(j.committed_lsn() > 0);
    }

    // ── recover ──────────────────────────────────────────────────────────────

    #[test]
    fn test_recover_empty_journal() {
        let j = Journal::new();
        let recovered = j.recover();
        assert!(recovered.is_empty());
    }

    #[test]
    fn test_recover_committed_insert() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(1));
        j.append(JournalEntry::Write {
            tx_id: 1,
            triple: triple("s", "p", "o"),
            graph: None,
            is_insert: true,
        });
        j.commit(1);
        let recovered = j.recover();
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].0, "s");
    }

    #[test]
    fn test_recover_skips_aborted_tx() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(1));
        j.append(JournalEntry::Write {
            tx_id: 1,
            triple: triple("s1", "p1", "o1"),
            graph: None,
            is_insert: true,
        });
        j.abort(1);
        let recovered = j.recover();
        assert!(recovered.is_empty());
    }

    #[test]
    fn test_recover_skips_pending_tx() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(1));
        j.append(JournalEntry::Write {
            tx_id: 1,
            triple: triple("s1", "p1", "o1"),
            graph: None,
            is_insert: true,
        });
        // No commit or abort
        let recovered = j.recover();
        assert!(recovered.is_empty());
    }

    #[test]
    fn test_recover_skips_deletes() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(1));
        j.append(JournalEntry::Write {
            tx_id: 1,
            triple: triple("s", "p", "o"),
            graph: None,
            is_insert: false, // delete
        });
        j.commit(1);
        let recovered = j.recover();
        assert!(recovered.is_empty());
    }

    #[test]
    fn test_recover_multiple_committed_txs() {
        let mut j = Journal::new();
        for tid in 1u64..=3 {
            j.append(JournalEntry::Begin(tid));
            j.append(JournalEntry::Write {
                tx_id: tid,
                triple: triple(&format!("s{tid}"), "p", "o"),
                graph: None,
                is_insert: true,
            });
            j.commit(tid);
        }
        let recovered = j.recover();
        assert_eq!(recovered.len(), 3);
    }

    #[test]
    fn test_recover_with_named_graph() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(1));
        j.append(JournalEntry::Write {
            tx_id: 1,
            triple: triple("s", "p", "o"),
            graph: Some("http://graph.example/".to_string()),
            is_insert: true,
        });
        j.commit(1);
        let recovered = j.recover();
        assert_eq!(recovered[0].3, Some("http://graph.example/".to_string()));
    }

    // ── checkpoint clears pre-checkpoint data from recovery ──────────────────

    #[test]
    fn test_checkpoint_clears_pre_checkpoint_from_recovery() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(1));
        j.append(JournalEntry::Write {
            tx_id: 1,
            triple: triple("pre", "p", "o"),
            graph: None,
            is_insert: true,
        });
        j.commit(1);
        // Now checkpoint
        j.checkpoint();
        // After checkpoint, new tx
        j.append(JournalEntry::Begin(2));
        j.append(JournalEntry::Write {
            tx_id: 2,
            triple: triple("post", "p", "o"),
            graph: None,
            is_insert: true,
        });
        j.commit(2);
        let recovered = j.recover();
        // Only the post-checkpoint write should appear
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].0, "post");
    }

    #[test]
    fn test_checkpoint_returns_valid_lsn() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(1));
        let cp_lsn = j.checkpoint();
        assert!(cp_lsn >= 2); // At least 2 records
    }

    // ── pending_transactions ─────────────────────────────────────────────────

    #[test]
    fn test_pending_transactions_empty() {
        let j = Journal::new();
        assert!(j.pending_transactions().is_empty());
    }

    #[test]
    fn test_pending_transactions_one_pending() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(5));
        let pending = j.pending_transactions();
        assert!(pending.contains(&5));
    }

    #[test]
    fn test_pending_transactions_committed_not_pending() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(1));
        j.commit(1);
        assert!(j.pending_transactions().is_empty());
    }

    #[test]
    fn test_pending_transactions_aborted_not_pending() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(1));
        j.abort(1);
        assert!(j.pending_transactions().is_empty());
    }

    #[test]
    fn test_pending_transactions_multiple() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(1));
        j.append(JournalEntry::Begin(2));
        j.append(JournalEntry::Begin(3));
        j.commit(2);
        let pending = j.pending_transactions();
        assert_eq!(pending.len(), 2);
        assert!(pending.contains(&1) && pending.contains(&3));
    }

    // ── stats ────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial() {
        let j = Journal::new();
        let s = j.stats();
        assert_eq!(s.total_records, 0);
        assert_eq!(s.committed_txs, 0);
        assert_eq!(s.aborted_txs, 0);
        assert_eq!(s.pending_txs, 0);
    }

    #[test]
    fn test_stats_total_records() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(1));
        j.commit(1);
        let s = j.stats();
        assert_eq!(s.total_records, 2); // Begin + Commit
    }

    #[test]
    fn test_stats_committed_count() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(1));
        j.commit(1);
        j.append(JournalEntry::Begin(2));
        j.commit(2);
        assert_eq!(j.stats().committed_txs, 2);
    }

    #[test]
    fn test_stats_aborted_count() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(1));
        j.abort(1);
        assert_eq!(j.stats().aborted_txs, 1);
    }

    #[test]
    fn test_stats_pending_count() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(1));
        j.append(JournalEntry::Begin(2));
        j.commit(1);
        assert_eq!(j.stats().pending_txs, 1);
    }

    // ── verify_checksum ──────────────────────────────────────────────────────

    #[test]
    fn test_verify_checksum_valid_record() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(1));
        let record = &j.records[0];
        assert!(Journal::verify_checksum(record));
    }

    #[test]
    fn test_verify_checksum_corrupted_fails() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(1));
        let mut record = j.records[0].clone();
        record.checksum = record.checksum.wrapping_add(1); // corrupt
        assert!(!Journal::verify_checksum(&record));
    }

    #[test]
    fn test_verify_checksum_all_records() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(1));
        j.append(JournalEntry::Write {
            tx_id: 1,
            triple: triple("s", "p", "o"),
            graph: None,
            is_insert: true,
        });
        j.commit(1);
        for record in &j.records {
            assert!(Journal::verify_checksum(record));
        }
    }

    // ── records_since ────────────────────────────────────────────────────────

    #[test]
    fn test_records_since_all() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(1));
        j.append(JournalEntry::Begin(2));
        let slice = j.records_since(1);
        assert_eq!(slice.len(), 2);
    }

    #[test]
    fn test_records_since_partial() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(1)); // LSN 1
        j.append(JournalEntry::Begin(2)); // LSN 2
        j.append(JournalEntry::Begin(3)); // LSN 3
        let slice = j.records_since(2);
        assert_eq!(slice.len(), 2);
        assert_eq!(slice[0].lsn, 2);
    }

    #[test]
    fn test_records_since_beyond_end() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(1));
        let slice = j.records_since(100);
        assert!(slice.is_empty());
    }

    // ── interleaved transactions ──────────────────────────────────────────────

    #[test]
    fn test_interleaved_txs_recovery() {
        let mut j = Journal::new();
        // TX 1 and TX 2 interleaved; only TX 1 commits
        j.append(JournalEntry::Begin(1));
        j.append(JournalEntry::Begin(2));
        j.append(JournalEntry::Write {
            tx_id: 1,
            triple: triple("s1", "p", "o"),
            graph: None,
            is_insert: true,
        });
        j.append(JournalEntry::Write {
            tx_id: 2,
            triple: triple("s2", "p", "o"),
            graph: None,
            is_insert: true,
        });
        j.commit(1);
        j.abort(2);
        let recovered = j.recover();
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].0, "s1");
    }

    #[test]
    fn test_abort_does_not_appear_in_recovery() {
        let mut j = Journal::new();
        j.append(JournalEntry::Begin(1));
        j.append(JournalEntry::Write {
            tx_id: 1,
            triple: triple("bad", "p", "o"),
            graph: None,
            is_insert: true,
        });
        j.abort(1);
        j.append(JournalEntry::Begin(2));
        j.append(JournalEntry::Write {
            tx_id: 2,
            triple: triple("good", "p", "o"),
            graph: None,
            is_insert: true,
        });
        j.commit(2);
        let recovered = j.recover();
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].0, "good");
    }

    // ── crc32 ────────────────────────────────────────────────────────────────

    #[test]
    fn test_crc32_deterministic() {
        let a = crc32_checksum(b"hello world");
        let b = crc32_checksum(b"hello world");
        assert_eq!(a, b);
    }

    #[test]
    fn test_crc32_differs_for_different_input() {
        let a = crc32_checksum(b"foo");
        let b = crc32_checksum(b"bar");
        assert_ne!(a, b);
    }

    #[test]
    fn test_crc32_empty_input() {
        let v = crc32_checksum(b"");
        // Well-defined result for empty slice
        assert_eq!(v, 0);
    }
}
