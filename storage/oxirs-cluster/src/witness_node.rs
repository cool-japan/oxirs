//! Witness nodes — consensus participants without full log storage.
//!
//! A witness node:
//! - Participates in leader election (votes for candidates)
//! - Confirms `AppendEntries` for entries in its tail window
//! - Rejects `AppendEntries` for entries older than its tail window
//!   (the leader must fall back to a full member for historical queries)
//! - Does **not** serve client reads (has no complete log)
//!
//! ## Why witnesses?
//!
//! A 1000-node cluster where 200 nodes are witnesses needs only
//! `800 × full_log_size` disk, not `1000 × full_log_size`, while still
//! keeping those 200 nodes as quorum participants.
//!
//! ## Raft correctness properties
//!
//! - **Term monotonicity**: if a higher term is observed on any RPC, `current_term`
//!   is updated immediately and `voted_for` is reset before proceeding.
//! - **Vote safety**: at most one vote granted per term.
//! - **Log currency check**: candidate's log must be at least as up-to-date as the
//!   witness's tail (compared lexicographically on `(last_term, last_index)`).
//! - **Tail window eviction**: entries older than `tail_window` are evicted from the
//!   front of the circular buffer; the leader must escalate to a full member for
//!   those entries.

use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────────────────
// Log entry stub
// ─────────────────────────────────────────────────────────────────────────────

/// A log entry stub retained by witness nodes.
///
/// Witnesses store the index, term, and a CRC32 checksum of the full entry
/// content. They can verify that an entry was received but cannot reconstruct
/// the payload.
#[derive(Debug, Clone)]
pub struct WitnessLogEntry {
    /// 1-based log index.
    pub index: u64,
    /// Raft term in which this entry was created.
    pub term: u64,
    /// CRC32 of the full entry bytes (witnesses verify, not store, content).
    pub checksum: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// RPC types
// ─────────────────────────────────────────────────────────────────────────────

/// Arguments for a Raft vote request, as seen by a witness node.
#[derive(Debug, Clone)]
pub struct VoteRequest {
    /// ID of the node requesting the vote.
    pub candidate_id: String,
    /// Candidate's current term.
    pub term: u64,
    /// Index of the last entry in the candidate's log.
    pub last_log_index: u64,
    /// Term of the last entry in the candidate's log.
    pub last_log_term: u64,
}

/// Response to a vote request.
#[derive(Debug, Clone)]
pub struct VoteResponse {
    /// The witness's current term (so the candidate can update itself if stale).
    pub term: u64,
    /// Whether the vote was granted.
    pub vote_granted: bool,
    /// Human-readable rejection reason (only set when `vote_granted == false`).
    pub reason: Option<String>,
}

/// An `AppendEntries` RPC addressed to a witness node.
///
/// Unlike the full `AppendEntriesRequest` (which carries `Vec<LogEntry>` with
/// opaque payload bytes), this variant carries [`WitnessLogEntry`] stubs —
/// the witness only needs to confirm index/term/checksum.
#[derive(Debug, Clone)]
pub struct WitnessAppendRequest {
    /// ID of the current leader.
    pub leader_id: String,
    /// Leader's current term.
    pub term: u64,
    /// Index of the log entry immediately preceding the new entries.
    pub prev_log_index: u64,
    /// Term of the entry at `prev_log_index`.
    pub prev_log_term: u64,
    /// New entries to append (may be empty for a heartbeat).
    pub entries: Vec<WitnessLogEntry>,
    /// Highest index that the leader has committed.
    pub leader_commit: u64,
}

/// Response to a witness `AppendEntries` RPC.
#[derive(Debug, Clone)]
pub struct WitnessAppendResponse {
    /// The witness's current term (so the leader can step down if stale).
    pub term: u64,
    /// Whether the append was accepted.
    pub success: bool,
    /// The highest log index the witness now has confirmed (on success).
    pub match_index: u64,
    /// `false` when `prev_log_index` (or the entries themselves) fall outside
    /// the witness's tail window. The leader must then contact a full member.
    pub in_window: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Witness node state machine
// ─────────────────────────────────────────────────────────────────────────────

/// Witness node state — minimal Raft participant.
///
/// A witness participates in leader election and confirms log replication
/// for recent entries, but holds only the last `tail_window` entries.
#[derive(Debug)]
pub struct WitnessNode {
    node_id: String,
    current_term: u64,
    voted_for: Option<String>,
    /// Circular buffer of recent log entries, oldest at front, newest at back.
    tail: VecDeque<WitnessLogEntry>,
    tail_window: usize,
    commit_index: u64,
}

impl WitnessNode {
    /// Create a new witness node with an empty tail buffer.
    ///
    /// # Arguments
    /// * `node_id`     — unique cluster identifier for this witness.
    /// * `tail_window` — how many recent log entries to retain (> 0).
    pub fn new(node_id: &str, tail_window: usize) -> Self {
        assert!(tail_window > 0, "tail_window must be > 0");
        Self {
            node_id: node_id.to_owned(),
            current_term: 0,
            voted_for: None,
            tail: VecDeque::with_capacity(tail_window),
            tail_window,
            commit_index: 0,
        }
    }

    // ── Public accessors ─────────────────────────────────────────────────────

    /// The unique node ID of this witness.
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// The witness's current Raft term.
    pub fn current_term(&self) -> u64 {
        self.current_term
    }

    /// The number of entries currently retained in the tail buffer.
    pub fn tail_len(&self) -> usize {
        self.tail.len()
    }

    /// The highest log index the witness has committed.
    pub fn commit_index(&self) -> u64 {
        self.commit_index
    }

    /// Returns `true` if the entry at `index` is present in the tail window.
    pub fn has_entry(&self, index: u64) -> bool {
        self.tail.iter().any(|e| e.index == index)
    }

    /// Returns the highest log index the witness has confirmed in its tail,
    /// or `0` if the tail is empty.
    pub fn last_confirmed_index(&self) -> u64 {
        self.tail.back().map(|e| e.index).unwrap_or(0)
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// The witness's "last log" position: `(last_log_term, last_log_index)`.
    /// Returns `(0, 0)` when the tail is empty.
    fn last_log_position(&self) -> (u64, u64) {
        self.tail
            .back()
            .map(|e| (e.term, e.index))
            .unwrap_or((0, 0))
    }

    /// The smallest log index currently in the tail window, or `0` if empty.
    fn oldest_index(&self) -> u64 {
        self.tail.front().map(|e| e.index).unwrap_or(0)
    }

    /// Advance `current_term` and reset `voted_for` when a higher term is
    /// observed.  Must be called **before** any grant/accept decision.
    fn maybe_advance_term(&mut self, incoming_term: u64) {
        if incoming_term > self.current_term {
            self.current_term = incoming_term;
            self.voted_for = None;
        }
    }

    // ── Raft RPCs ────────────────────────────────────────────────────────────

    /// Handle an incoming vote request.
    ///
    /// Implements §5.2 of the Raft paper with the log-currency check:
    /// - Term monotonicity: if `req.term > current_term`, update and reset vote.
    /// - Reject if `req.term < current_term`.
    /// - Reject if already voted for a *different* candidate this term.
    /// - Reject if candidate's log is less up-to-date than the witness's tail.
    pub fn handle_vote_request(&mut self, req: &VoteRequest) -> VoteResponse {
        // Step 1: advance term on higher incoming term (clears voted_for).
        self.maybe_advance_term(req.term);

        // Step 2: reject if the candidate's term is still stale.
        if req.term < self.current_term {
            return VoteResponse {
                term: self.current_term,
                vote_granted: false,
                reason: Some(format!(
                    "stale term: candidate={}, witness={}",
                    req.term, self.current_term
                )),
            };
        }

        // Step 3: check whether we have already voted for someone else this term.
        if let Some(ref already_voted) = self.voted_for {
            if *already_voted != req.candidate_id {
                return VoteResponse {
                    term: self.current_term,
                    vote_granted: false,
                    reason: Some(format!(
                        "already voted for {} in term {}",
                        already_voted, self.current_term
                    )),
                };
            }
        }

        // Step 4: log-currency check — candidate's log must be ≥ witness's tail.
        // Use lexicographic comparison on (last_log_term, last_log_index).
        let (my_last_term, my_last_index) = self.last_log_position();
        let candidate_more_uptodate =
            (req.last_log_term, req.last_log_index) >= (my_last_term, my_last_index);
        if !candidate_more_uptodate {
            return VoteResponse {
                term: self.current_term,
                vote_granted: false,
                reason: Some(format!(
                    "candidate log ({},{}) less current than witness ({},{})",
                    req.last_log_term, req.last_log_index, my_last_term, my_last_index
                )),
            };
        }

        // Grant vote.
        self.voted_for = Some(req.candidate_id.clone());
        VoteResponse {
            term: self.current_term,
            vote_granted: true,
            reason: None,
        }
    }

    /// Handle an incoming `AppendEntries` RPC.
    ///
    /// Implements §5.3 of the Raft paper adapted for witness (tail-window) storage:
    /// - Term monotonicity: if `req.term > current_term`, update and reset vote.
    /// - Reject if `req.term < current_term` (`in_window = true`; the leader is stale).
    /// - If `prev_log_index == 0`: accept unconditionally (bootstrapping).
    /// - If `prev_log_index` is in the tail window: verify the entry's term matches;
    ///   reject with `success=false, in_window=true` on mismatch.
    /// - If `prev_log_index` is **before** the tail window: return
    ///   `success=false, in_window=false` — leader must contact a full member.
    /// - On success: append entries (evicting oldest beyond `tail_window`), advance
    ///   `commit_index` to `min(leader_commit, last_appended_index)`.
    pub fn handle_append_entries(&mut self, req: &WitnessAppendRequest) -> WitnessAppendResponse {
        // Step 1: advance term on higher incoming term (clears voted_for).
        self.maybe_advance_term(req.term);

        // Step 2: reject stale leader.
        if req.term < self.current_term {
            return WitnessAppendResponse {
                term: self.current_term,
                success: false,
                match_index: self.last_confirmed_index(),
                in_window: true, // the witness is fine; leader is stale
            };
        }

        // Step 3: consistency check on prev_log_index.
        if req.prev_log_index > 0 {
            let oldest = self.oldest_index();

            if oldest > 0 && req.prev_log_index < oldest {
                // prev entry is older than our tail window → leader must use a full member
                return WitnessAppendResponse {
                    term: self.current_term,
                    success: false,
                    match_index: self.last_confirmed_index(),
                    in_window: false,
                };
            }

            // If the tail is non-empty, look for prev_log_index and verify its term.
            if let Some(prev_entry) = self.tail.iter().find(|e| e.index == req.prev_log_index) {
                if prev_entry.term != req.prev_log_term {
                    return WitnessAppendResponse {
                        term: self.current_term,
                        success: false,
                        match_index: self.last_confirmed_index(),
                        in_window: true,
                    };
                }
            } else if !self.tail.is_empty() {
                // prev_log_index is within the window range but not found:
                // could be a gap or a log conflict — reject.
                // (If tail is empty, oldest_index() == 0 and we fall through to accept.)
                return WitnessAppendResponse {
                    term: self.current_term,
                    success: false,
                    match_index: self.last_confirmed_index(),
                    in_window: true,
                };
            }
            // tail is empty (oldest == 0) and prev_log_index > 0 → we have no history
            // to verify against; we trust the leader (standard Raft behavior for a
            // freshly joined member that has not yet received any entries).
        }

        // Step 4: append new entries.
        for entry in &req.entries {
            // If we already have this index, overwrite (log truncation).
            if let Some(pos) = self.tail.iter().position(|e| e.index == entry.index) {
                self.tail.drain(pos..);
            }
            self.tail.push_back(entry.clone());

            // Evict oldest if window exceeded.
            while self.tail.len() > self.tail_window {
                self.tail.pop_front();
            }
        }

        // Step 5: advance commit_index.
        let last_appended = self.last_confirmed_index();
        if req.leader_commit > self.commit_index {
            self.commit_index = req.leader_commit.min(last_appended);
        }

        WitnessAppendResponse {
            term: self.current_term,
            success: true,
            match_index: last_appended,
            in_window: true,
        }
    }
}
