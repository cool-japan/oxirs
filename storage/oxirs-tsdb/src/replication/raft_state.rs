//! Basic Raft consensus state machine for TSDB replication
//!
//! Implements the core state transitions described in:
//! "In Search of an Understandable Consensus Algorithm (Extended Version)"
//! Diego Ongaro and John Ousterhout, 2014.
//! <https://raft.github.io/raft.pdf>
//!
//! ## Scope
//!
//! This module provides the **state machine** (the pure logic layer) without
//! any network I/O or timers.  It is the caller's responsibility to:
//!
//! - Deliver `AppendEntries` / `RequestVote` RPCs from the network layer.
//! - Trigger elections on timeout.
//! - Send RPC responses back to peers.
//! - Persist `current_term` and `voted_for` to durable storage before
//!   applying state transitions that modify them.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use oxirs_tsdb::replication::{RaftState, TsdbCommand};
//!
//! let mut state = RaftState::new("node-1".to_string());
//! // On election timeout:
//! let vote_args = state.become_candidate();
//! // ... broadcast vote_args to peers ...
//! ```

use crate::error::TsdbError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Metadata associated with a named time series in the distributed cluster.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeriesMetadata {
    /// Human-readable series name.
    pub name: String,
    /// Tag key-value pairs (e.g. `{"host": "server-1", "region": "eu-west"}`).
    pub tags: HashMap<String, String>,
    /// Optional retention period in days. `None` means retain forever.
    pub retention_days: Option<u32>,
}

impl SeriesMetadata {
    /// Create metadata with only a name (no tags, no retention limit).
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            tags: HashMap::new(),
            retention_days: None,
        }
    }

    /// Builder: add a tag.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    /// Builder: set retention.
    pub fn with_retention(mut self, days: u32) -> Self {
        self.retention_days = Some(days);
        self
    }
}

/// A command that can be replicated via Raft to all cluster nodes.
///
/// Each variant represents a state-machine transition applied to the TSDB.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TsdbCommand {
    /// Write a single data-point to a series.
    WriteDatapoint {
        /// Series identifier (UUID-style string).
        series_id: String,
        /// Timestamp in milliseconds since Unix epoch.
        timestamp: i64,
        /// Measured value.
        value: f64,
    },

    /// Create a new time series.
    CreateSeries {
        /// Series identifier.
        series_id: String,
        /// Descriptive metadata.
        metadata: SeriesMetadata,
    },

    /// Permanently delete a series and all its data.
    DeleteSeries {
        /// Series identifier.
        series_id: String,
    },

    /// Compact (expire) data older than the given timestamp.
    Compact {
        /// Series identifier.
        series_id: String,
        /// Remove data with `timestamp <= up_to_timestamp`.
        up_to_timestamp: i64,
    },
}

/// A single entry in the Raft replicated log.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogEntry {
    /// The Raft term in which this entry was created.
    pub term: u64,
    /// 1-based log index.
    pub index: u64,
    /// The command to be applied to the state machine.
    pub command: TsdbCommand,
}

// ---------------------------------------------------------------------------
// RPC argument/reply types
// ---------------------------------------------------------------------------

/// Arguments for the RequestVote RPC (§5.2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestVoteArgs {
    /// Candidate's current term.
    pub term: u64,
    /// Candidate's node identifier.
    pub candidate_id: String,
    /// Index of candidate's last log entry.
    pub last_log_index: u64,
    /// Term of candidate's last log entry.
    pub last_log_term: u64,
}

/// Reply to a RequestVote RPC.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestVoteReply {
    /// Current term, for the candidate to update itself.
    pub term: u64,
    /// `true` means candidate received vote.
    pub vote_granted: bool,
}

/// Arguments for the AppendEntries RPC (§5.3, also used as heartbeat when
/// `entries` is empty).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppendEntriesArgs {
    /// Leader's current term.
    pub term: u64,
    /// Leader's node identifier (so followers can redirect clients).
    pub leader_id: String,
    /// Index of log entry immediately preceding the new entries.
    pub prev_log_index: u64,
    /// Term of `prev_log_index` entry.
    pub prev_log_term: u64,
    /// New log entries to append (empty for heartbeat).
    pub entries: Vec<LogEntry>,
    /// Leader's `commit_index`.
    pub leader_commit: u64,
}

/// Reply to an AppendEntries RPC.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppendEntriesReply {
    /// Current term, for the leader to update itself.
    pub term: u64,
    /// `true` if follower contained entry matching `prev_log_index` and
    /// `prev_log_term`.
    pub success: bool,
    /// Optimised conflict resolution (§5.3, "fast backup"):
    /// the first index with a conflicting term, if any.
    pub conflict_index: Option<u64>,
    /// The term of the conflicting entry, if any.
    pub conflict_term: Option<u64>,
}

// ---------------------------------------------------------------------------
// Raft role
// ---------------------------------------------------------------------------

/// The three roles a Raft node can occupy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RaftRole {
    /// Passive: forwards client requests to leader and responds to RPCs.
    Follower,
    /// Transitional: actively seeking votes to become leader.
    Candidate,
    /// Active: manages log replication to all followers.
    Leader,
}

impl std::fmt::Display for RaftRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Follower => write!(f, "follower"),
            Self::Candidate => write!(f, "candidate"),
            Self::Leader => write!(f, "leader"),
        }
    }
}

// ---------------------------------------------------------------------------
// Raft errors
// ---------------------------------------------------------------------------

/// Errors specific to the Raft state machine.
#[derive(Debug, thiserror::Error)]
pub enum RaftError {
    /// The operation requires leadership but this node is not the leader.
    #[error("not the leader (current role: {role}, known leader: {current_leader:?})")]
    NotLeader {
        /// Current role of this node.
        role: String,
        /// Last known leader ID, if any.
        current_leader: Option<String>,
    },

    /// A stale term was observed; the caller should retry after updating.
    #[error("stale term: received {received}, current {current}")]
    StaleTerm {
        /// Term received in the incoming message.
        received: u64,
        /// This node's current term.
        current: u64,
    },

    /// The log is in an inconsistent state (programming error).
    #[error("log invariant violated: {0}")]
    LogInvariant(String),

    /// Wraps the general TSDB error type.
    #[error("tsdb error: {0}")]
    Tsdb(#[from] TsdbError),
}

// ---------------------------------------------------------------------------
// RaftState
// ---------------------------------------------------------------------------

/// The complete Raft state for a single cluster node.
///
/// All fields match the Raft paper's terminology exactly.
///
/// # Persistence contract
///
/// The caller **must** durably persist these fields before responding to
/// any RPC:
/// - `current_term`
/// - `voted_for`
/// - `log`
///
/// The remaining fields are volatile and may be safely reinitialised to their
/// documented defaults on node restart.
#[derive(Debug)]
pub struct RaftState {
    // ---- Persistent state (must be written to stable storage before RPCs) --
    /// Latest term this server has seen (initialised to 0 on first boot,
    /// monotonically increasing).
    pub current_term: u64,
    /// `candidate_id` that received vote in current term, or `None`.
    pub voted_for: Option<String>,
    /// The replicated command log.  Index 0 is a sentinel empty entry with
    /// `term = 0` and `index = 0`; real entries start at index 1.
    pub log: Vec<LogEntry>,

    // ---- Volatile state (reinitialise to defaults on restart) --------------
    /// Index of highest log entry known to be committed (≥ 0, initialised 0).
    pub commit_index: u64,
    /// Index of highest log entry applied to state machine (≥ 0, initialised 0).
    pub last_applied: u64,
    /// This node's current role.
    pub role: RaftRole,
    /// This node's unique identifier.
    pub node_id: String,
    /// Last known leader ID (used to redirect client requests).
    pub current_leader: Option<String>,

    // ---- Votes received during current candidate term (volatile) -----------
    /// Peer IDs that granted a vote in the current election.
    votes_received: std::collections::HashSet<String>,

    // ---- Leader-only volatile state (re-initialised after each election) ---
    /// For each peer: next log index to send (initialised to leader's
    /// `last_log_index + 1`).
    pub next_index: HashMap<String, u64>,
    /// For each peer: highest log index known to be replicated on that peer.
    pub match_index: HashMap<String, u64>,
}

impl RaftState {
    /// Create a new Raft node in the `Follower` role with term 0.
    pub fn new(node_id: impl Into<String>) -> Self {
        // Sentinel log entry at index 0 (never applied)
        let sentinel = LogEntry {
            term: 0,
            index: 0,
            command: TsdbCommand::Compact {
                series_id: String::new(),
                up_to_timestamp: 0,
            },
        };
        Self {
            current_term: 0,
            voted_for: None,
            log: vec![sentinel],
            commit_index: 0,
            last_applied: 0,
            role: RaftRole::Follower,
            node_id: node_id.into(),
            current_leader: None,
            votes_received: std::collections::HashSet::new(),
            next_index: HashMap::new(),
            match_index: HashMap::new(),
        }
    }

    // ---- Log helpers -------------------------------------------------------

    /// Return the index of the last log entry (0 if only sentinel present).
    pub fn last_log_index(&self) -> u64 {
        self.log.last().map(|e| e.index).unwrap_or(0)
    }

    /// Return the term of the last log entry (0 if only sentinel present).
    pub fn last_log_term(&self) -> u64 {
        self.log.last().map(|e| e.term).unwrap_or(0)
    }

    /// Retrieve a log entry by 1-based index, or `None` if out of range.
    pub fn get_entry(&self, index: u64) -> Option<&LogEntry> {
        // Our sentinel is at position 0 with index 0; real entries follow.
        self.log.get(index as usize)
    }

    /// Term of the entry at `index`, or `None` if index is invalid.
    fn entry_term(&self, index: u64) -> Option<u64> {
        self.get_entry(index).map(|e| e.term)
    }

    // ---- Role transitions --------------------------------------------------

    /// Transition to `Follower` for the given term.
    ///
    /// Clears `voted_for` if the term increases.
    pub fn become_follower(&mut self, term: u64) {
        if term > self.current_term {
            self.current_term = term;
            self.voted_for = None;
        }
        self.role = RaftRole::Follower;
        self.next_index.clear();
        self.match_index.clear();
        self.votes_received.clear();
    }

    /// Transition to `Candidate` and start a new election.
    ///
    /// Increments `current_term`, votes for self, and returns the
    /// [`RequestVoteArgs`] to broadcast to peers.
    pub fn become_candidate(&mut self) -> RequestVoteArgs {
        self.current_term += 1;
        self.role = RaftRole::Candidate;
        self.voted_for = Some(self.node_id.clone());
        self.current_leader = None;
        self.votes_received.clear();
        self.votes_received.insert(self.node_id.clone()); // vote for self

        RequestVoteArgs {
            term: self.current_term,
            candidate_id: self.node_id.clone(),
            last_log_index: self.last_log_index(),
            last_log_term: self.last_log_term(),
        }
    }

    /// Transition to `Leader`.
    ///
    /// Initialises `next_index` and `match_index` for each peer and appends a
    /// no-op entry to the log (§8, "no-op commitment").
    pub fn become_leader(&mut self, peers: &[String]) {
        debug_assert_eq!(self.role, RaftRole::Candidate);
        self.role = RaftRole::Leader;
        self.current_leader = Some(self.node_id.clone());

        let next = self.last_log_index() + 1;
        for peer in peers {
            self.next_index.insert(peer.clone(), next);
            self.match_index.insert(peer.clone(), 0);
        }

        // Append a no-op so that any uncommitted entries from prior terms
        // get committed indirectly (Raft §5.4.2).
        let noop_index = next;
        self.log.push(LogEntry {
            term: self.current_term,
            index: noop_index,
            command: TsdbCommand::Compact {
                series_id: "__noop__".to_string(),
                up_to_timestamp: 0,
            },
        });
    }

    // ---- RPC handlers ------------------------------------------------------

    /// Handle an incoming `RequestVote` RPC.
    ///
    /// Implements the grant logic from Figure 2 of the Raft paper.
    pub fn handle_vote_request(&mut self, args: &RequestVoteArgs) -> RequestVoteReply {
        // Step down if we see a higher term
        if args.term > self.current_term {
            self.become_follower(args.term);
        }

        let vote_granted = if args.term < self.current_term {
            // Stale term: deny
            false
        } else {
            // Grant only if we haven't voted for someone else in this term
            let already_voted = self
                .voted_for
                .as_ref()
                .map(|v| v != &args.candidate_id)
                .unwrap_or(false);

            // Candidate's log must be at least as up-to-date as ours (§5.4.1)
            let candidate_log_ok = args.last_log_term > self.last_log_term()
                || (args.last_log_term == self.last_log_term()
                    && args.last_log_index >= self.last_log_index());

            !already_voted && candidate_log_ok
        };

        if vote_granted {
            self.voted_for = Some(args.candidate_id.clone());
        }

        RequestVoteReply {
            term: self.current_term,
            vote_granted,
        }
    }

    /// Process a `RequestVoteReply` from a peer.
    ///
    /// Returns `true` if this node has just achieved a quorum and become
    /// leader (so the caller can proceed with `become_leader`).
    ///
    /// `total_peers` is the total number of nodes in the cluster (including
    /// this one).
    pub fn handle_vote_reply(
        &mut self,
        from: &str,
        reply: &RequestVoteReply,
        total_peers: usize,
    ) -> bool {
        if reply.term > self.current_term {
            self.become_follower(reply.term);
            return false;
        }
        if self.role != RaftRole::Candidate || reply.term != self.current_term {
            return false;
        }

        if reply.vote_granted {
            self.votes_received.insert(from.to_owned());
        }

        let quorum = total_peers / 2 + 1;
        self.votes_received.len() >= quorum
    }

    /// Handle an incoming `AppendEntries` RPC (also serves as heartbeat).
    ///
    /// Implements Figure 2, "AppendEntries RPC" receiver logic.
    pub fn handle_append_entries(&mut self, args: &AppendEntriesArgs) -> AppendEntriesReply {
        // 1. Reply false if term < currentTerm (§5.1)
        if args.term < self.current_term {
            return AppendEntriesReply {
                term: self.current_term,
                success: false,
                conflict_index: None,
                conflict_term: None,
            };
        }

        // Update term and role if needed
        if args.term > self.current_term {
            self.become_follower(args.term);
        } else if self.role == RaftRole::Candidate {
            // A valid leader emerged in the same term
            self.role = RaftRole::Follower;
        }

        self.current_leader = Some(args.leader_id.clone());

        // 2. Reply false if log doesn't contain entry at prev_log_index with
        //    prev_log_term (§5.3)
        match self.entry_term(args.prev_log_index) {
            None => {
                // Our log is shorter than expected
                return AppendEntriesReply {
                    term: self.current_term,
                    success: false,
                    conflict_index: Some(self.last_log_index() + 1),
                    conflict_term: None,
                };
            }
            Some(t) if t != args.prev_log_term => {
                // Term mismatch at prev_log_index – fast backup (§5.3)
                let conflict_term = t;
                // Find the first index in our log with this conflicting term
                let conflict_index = self
                    .log
                    .iter()
                    .find(|e| e.term == conflict_term)
                    .map(|e| e.index)
                    .unwrap_or(args.prev_log_index);
                // Truncate conflicting entries
                self.log.truncate(args.prev_log_index as usize);
                return AppendEntriesReply {
                    term: self.current_term,
                    success: false,
                    conflict_index: Some(conflict_index),
                    conflict_term: Some(conflict_term),
                };
            }
            Some(_) => {} // prev entry matches – continue
        }

        // 3. If an existing entry conflicts with a new one (same index,
        //    different terms), delete the existing entry and all that follow.
        // 4. Append any new entries not already in the log.
        for new_entry in &args.entries {
            let idx = new_entry.index as usize;
            if idx < self.log.len() {
                if self.log[idx].term != new_entry.term {
                    // Conflict – truncate and append
                    self.log.truncate(idx);
                    self.log.push(new_entry.clone());
                }
                // else entry already present and matches – skip
            } else {
                self.log.push(new_entry.clone());
            }
        }

        // 5. If leaderCommit > commitIndex, set commitIndex to min(leaderCommit,
        //    index of last new entry)
        if args.leader_commit > self.commit_index {
            let last_new = args
                .entries
                .last()
                .map(|e| e.index)
                .unwrap_or(self.last_log_index());
            self.commit_index = args.leader_commit.min(last_new);
        }

        AppendEntriesReply {
            term: self.current_term,
            success: true,
            conflict_index: None,
            conflict_term: None,
        }
    }

    /// Leader operation: append a new command to the log.
    ///
    /// Returns the log index assigned to the new entry, or [`RaftError::NotLeader`]
    /// if this node is not currently the leader.
    pub fn propose_command(&mut self, command: TsdbCommand) -> Result<u64, RaftError> {
        if self.role != RaftRole::Leader {
            return Err(RaftError::NotLeader {
                role: self.role.to_string(),
                current_leader: self.current_leader.clone(),
            });
        }
        let index = self.last_log_index() + 1;
        self.log.push(LogEntry {
            term: self.current_term,
            index,
            command,
        });
        Ok(index)
    }

    /// Advance `commit_index` based on quorum acknowledgement.
    ///
    /// Leaders should call this after updating `match_index` entries.
    /// `total_peers` is the cluster size (including this node).
    ///
    /// Returns `true` if `commit_index` was advanced.
    pub fn try_advance_commit_index(&mut self, total_peers: usize) -> bool {
        if self.role != RaftRole::Leader {
            return false;
        }
        let quorum = total_peers / 2 + 1;
        let mut advanced = false;

        // Check potential commit indices from high to low
        for n in (self.commit_index + 1..=self.last_log_index()).rev() {
            // Only commit entries from the current term (§5.4.2)
            let entry_term = match self.entry_term(n) {
                Some(t) => t,
                None => continue,
            };
            if entry_term != self.current_term {
                continue;
            }

            // Count nodes that have replicated up to index n
            let replicated = self.match_index.values().filter(|&&m| m >= n).count() + 1; // +1 for leader itself

            if replicated >= quorum {
                self.commit_index = n;
                advanced = true;
                break; // commit_index is monotonic, only need the highest
            }
        }
        advanced
    }

    /// Apply all committed but not-yet-applied entries to the state machine.
    ///
    /// Returns the list of commands that were applied (in order).  The caller
    /// is responsible for executing the commands against the actual TSDB
    /// storage.
    pub fn apply_committed_entries(&mut self) -> Vec<TsdbCommand> {
        let mut applied = Vec::new();
        while self.last_applied < self.commit_index {
            self.last_applied += 1;
            if let Some(entry) = self.get_entry(self.last_applied) {
                applied.push(entry.command.clone());
            }
        }
        applied
    }

    /// Build `AppendEntries` args for a specific peer.
    ///
    /// Returns `Err` if this node is not the leader, or if log state is
    /// inconsistent.
    pub fn build_append_entries(&self, peer_id: &str) -> Result<AppendEntriesArgs, RaftError> {
        if self.role != RaftRole::Leader {
            return Err(RaftError::NotLeader {
                role: self.role.to_string(),
                current_leader: self.current_leader.clone(),
            });
        }
        let next = *self
            .next_index
            .get(peer_id)
            .ok_or_else(|| RaftError::LogInvariant(format!("unknown peer: {}", peer_id)))?;
        let prev_index = next.saturating_sub(1);
        let prev_term = self.entry_term(prev_index).unwrap_or(0);

        let entries: Vec<LogEntry> = self
            .log
            .iter()
            .filter(|e| e.index >= next)
            .cloned()
            .collect();

        Ok(AppendEntriesArgs {
            term: self.current_term,
            leader_id: self.node_id.clone(),
            prev_log_index: prev_index,
            prev_log_term: prev_term,
            entries,
            leader_commit: self.commit_index,
        })
    }

    /// Update `next_index` and `match_index` for a peer after a successful
    /// `AppendEntries` reply.
    pub fn handle_append_success(&mut self, peer_id: &str, match_index: u64) {
        if self.role != RaftRole::Leader {
            return;
        }
        self.match_index.insert(peer_id.to_owned(), match_index);
        self.next_index.insert(peer_id.to_owned(), match_index + 1);
    }

    /// Update `next_index` for a peer after a failed `AppendEntries` reply,
    /// using the fast-backup optimisation from §5.3.
    pub fn handle_append_failure(&mut self, peer_id: &str, reply: &AppendEntriesReply) {
        if self.role != RaftRole::Leader {
            return;
        }

        if reply.term > self.current_term {
            self.become_follower(reply.term);
            return;
        }

        let new_next = if let (Some(conflict_term), Some(conflict_index)) =
            (reply.conflict_term, reply.conflict_index)
        {
            // Find the last entry in our log with the conflicting term.
            // If found, next = last matching index + 1;
            // if not, next = conflict_index.
            let last_in_term = self
                .log
                .iter()
                .rev()
                .find(|e| e.term == conflict_term)
                .map(|e| e.index);
            match last_in_term {
                Some(idx) => idx + 1,
                None => conflict_index,
            }
        } else if let Some(conflict_index) = reply.conflict_index {
            conflict_index
        } else {
            // Fallback: decrement by one
            self.next_index
                .get(peer_id)
                .copied()
                .unwrap_or(1)
                .saturating_sub(1)
                .max(1)
        };

        self.next_index.insert(peer_id.to_owned(), new_next);
    }

    // ---- Diagnostic helpers ------------------------------------------------

    /// Returns `true` if this node believes it is the current leader.
    pub fn is_leader(&self) -> bool {
        self.role == RaftRole::Leader
    }

    /// Returns `true` if this node has voted in the current term.
    pub fn has_voted(&self) -> bool {
        self.voted_for.is_some()
    }

    /// Number of entries in the log (including the sentinel at index 0).
    pub fn log_len(&self) -> usize {
        self.log.len()
    }

    /// Number of pending votes received so far in the current election.
    pub fn votes_received_count(&self) -> usize {
        self.votes_received.len()
    }
}

// ---------------------------------------------------------------------------
// RaftError: extend TsdbResult mapping
// ---------------------------------------------------------------------------

impl From<RaftError> for TsdbError {
    fn from(e: RaftError) -> Self {
        TsdbError::Integration(e.to_string())
    }
}

// Provide a more descriptive error variant for NotLeader with current_leader
// We need to handle the renamed field (current_leader vs leader)
impl RaftError {
    /// Convenience constructor for `NotLeader`.
    pub fn not_leader(role: impl Into<String>, current_leader: Option<String>) -> Self {
        Self::NotLeader {
            role: role.into(),
            current_leader,
        }
    }
}

/// Alias for Raft operation results.
pub type RaftResult<T> = Result<T, RaftError>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(id: &str) -> RaftState {
        RaftState::new(id.to_string())
    }

    // ---- Initial state -----------------------------------------------------

    #[test]
    fn test_initial_state() {
        let state = make_node("node-1");
        assert_eq!(state.current_term, 0);
        assert!(state.voted_for.is_none());
        assert_eq!(state.role, RaftRole::Follower);
        assert_eq!(state.commit_index, 0);
        assert_eq!(state.last_applied, 0);
        assert_eq!(state.last_log_index(), 0); // sentinel only
        assert!(!state.is_leader());
    }

    // ---- become_candidate -------------------------------------------------

    #[test]
    fn test_become_candidate_increments_term() {
        let mut state = make_node("node-1");
        let args = state.become_candidate();
        assert_eq!(state.current_term, 1);
        assert_eq!(state.role, RaftRole::Candidate);
        assert_eq!(state.voted_for.as_deref(), Some("node-1"));
        assert_eq!(args.term, 1);
        assert_eq!(args.candidate_id, "node-1");
        assert_eq!(state.votes_received_count(), 1); // voted for self
    }

    #[test]
    fn test_become_candidate_multiple_elections() {
        let mut state = make_node("n");
        state.become_candidate(); // term 1
        state.become_follower(1); // lose election
        state.become_candidate(); // term 2
        assert_eq!(state.current_term, 2);
    }

    // ---- become_follower --------------------------------------------------

    #[test]
    fn test_become_follower_clears_voted_for_on_term_advance() {
        let mut state = make_node("node-1");
        state.voted_for = Some("node-2".to_string());
        state.current_term = 3;
        state.become_follower(5);
        assert_eq!(state.current_term, 5);
        assert!(state.voted_for.is_none());
        assert_eq!(state.role, RaftRole::Follower);
    }

    #[test]
    fn test_become_follower_same_term_preserves_voted_for() {
        let mut state = make_node("node-1");
        state.current_term = 3;
        state.voted_for = Some("node-2".to_string());
        state.become_follower(3); // same term
        assert_eq!(state.voted_for.as_deref(), Some("node-2"));
    }

    // ---- handle_vote_request ----------------------------------------------

    #[test]
    fn test_vote_granted_to_up_to_date_candidate() {
        let mut follower = make_node("follower");
        let args = RequestVoteArgs {
            term: 1,
            candidate_id: "candidate".to_string(),
            last_log_index: 0,
            last_log_term: 0,
        };
        let reply = follower.handle_vote_request(&args);
        assert!(reply.vote_granted);
        assert_eq!(reply.term, 1);
        assert_eq!(follower.voted_for.as_deref(), Some("candidate"));
    }

    #[test]
    fn test_vote_denied_stale_term() {
        let mut follower = make_node("follower");
        follower.current_term = 5;
        let args = RequestVoteArgs {
            term: 3, // older term
            candidate_id: "candidate".to_string(),
            last_log_index: 0,
            last_log_term: 0,
        };
        let reply = follower.handle_vote_request(&args);
        assert!(!reply.vote_granted);
    }

    #[test]
    fn test_vote_denied_already_voted_for_other() {
        let mut follower = make_node("follower");
        follower.current_term = 1;
        follower.voted_for = Some("other-candidate".to_string());
        let args = RequestVoteArgs {
            term: 1,
            candidate_id: "new-candidate".to_string(),
            last_log_index: 0,
            last_log_term: 0,
        };
        let reply = follower.handle_vote_request(&args);
        assert!(!reply.vote_granted);
    }

    #[test]
    fn test_vote_denied_candidate_log_behind() {
        let mut follower = make_node("follower");
        // Give follower a log entry at index 1 in term 2
        follower.current_term = 2;
        follower.log.push(LogEntry {
            term: 2,
            index: 1,
            command: TsdbCommand::WriteDatapoint {
                series_id: "s1".to_string(),
                timestamp: 1000,
                value: 1.0,
            },
        });

        let args = RequestVoteArgs {
            term: 3,
            candidate_id: "behind-candidate".to_string(),
            last_log_index: 0, // candidate has less log
            last_log_term: 0,
        };
        let reply = follower.handle_vote_request(&args);
        assert!(
            !reply.vote_granted,
            "should deny vote to candidate with older log"
        );
    }

    // ---- handle_vote_reply ------------------------------------------------

    #[test]
    fn test_quorum_achieved() {
        let mut candidate = make_node("c");
        candidate.become_candidate(); // term 1, 1 vote (self)

        // 3-node cluster; need 2 votes
        let reply = RequestVoteReply {
            term: 1,
            vote_granted: true,
        };
        let achieved = candidate.handle_vote_reply("peer1", &reply, 3);
        assert!(
            achieved,
            "quorum (2/3) should be achieved after one more vote"
        );
    }

    #[test]
    fn test_quorum_not_yet_achieved_5_nodes() {
        let mut candidate = make_node("c");
        candidate.become_candidate(); // 1 vote (self)

        let reply = RequestVoteReply {
            term: 1,
            vote_granted: true,
        };
        let achieved = candidate.handle_vote_reply("peer1", &reply, 5);
        assert!(!achieved, "need 3/5 votes, only have 2");
    }

    #[test]
    fn test_vote_reply_higher_term_causes_step_down() {
        let mut candidate = make_node("c");
        candidate.become_candidate();

        let reply = RequestVoteReply {
            term: 10,
            vote_granted: false,
        };
        let achieved = candidate.handle_vote_reply("peer", &reply, 3);
        assert!(!achieved);
        assert_eq!(candidate.role, RaftRole::Follower);
        assert_eq!(candidate.current_term, 10);
    }

    // ---- become_leader ----------------------------------------------------

    #[test]
    fn test_become_leader_initialises_indices() {
        let mut state = make_node("leader");
        state.become_candidate();
        state.become_leader(&["peer1".to_string(), "peer2".to_string()]);

        assert_eq!(state.role, RaftRole::Leader);
        assert!(state.is_leader());
        // next_index should be initialized for each peer
        assert!(state.next_index.contains_key("peer1"));
        assert!(state.next_index.contains_key("peer2"));
        // match_index should be 0 for each peer
        assert_eq!(state.match_index["peer1"], 0);
        assert_eq!(state.match_index["peer2"], 0);
    }

    // ---- handle_append_entries --------------------------------------------

    #[test]
    fn test_heartbeat_accepted() {
        let mut follower = make_node("follower");
        let args = AppendEntriesArgs {
            term: 1,
            leader_id: "leader".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
        };
        let reply = follower.handle_append_entries(&args);
        assert!(reply.success);
        assert_eq!(follower.current_leader.as_deref(), Some("leader"));
    }

    #[test]
    fn test_append_entries_rejected_stale_term() {
        let mut follower = make_node("follower");
        follower.current_term = 5;
        let args = AppendEntriesArgs {
            term: 3,
            leader_id: "old-leader".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
        };
        let reply = follower.handle_append_entries(&args);
        assert!(!reply.success);
        assert_eq!(reply.term, 5);
    }

    #[test]
    fn test_append_entries_appends_to_log() {
        let mut follower = make_node("follower");
        let entry = LogEntry {
            term: 1,
            index: 1,
            command: TsdbCommand::WriteDatapoint {
                series_id: "temp".to_string(),
                timestamp: 1000,
                value: 22.5,
            },
        };
        let args = AppendEntriesArgs {
            term: 1,
            leader_id: "leader".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![entry.clone()],
            leader_commit: 0,
        };
        let reply = follower.handle_append_entries(&args);
        assert!(reply.success);
        assert_eq!(follower.log_len(), 2); // sentinel + 1 entry
        assert_eq!(follower.last_log_index(), 1);
    }

    #[test]
    fn test_commit_index_advanced_on_append() {
        let mut follower = make_node("follower");
        let entry = LogEntry {
            term: 1,
            index: 1,
            command: TsdbCommand::CreateSeries {
                series_id: "s1".to_string(),
                metadata: SeriesMetadata::new("sensor-1"),
            },
        };
        let args = AppendEntriesArgs {
            term: 1,
            leader_id: "leader".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![entry],
            leader_commit: 1, // leader says index 1 is committed
        };
        follower.handle_append_entries(&args);
        assert_eq!(follower.commit_index, 1);
    }

    // ---- propose_command --------------------------------------------------

    #[test]
    fn test_propose_command_not_leader_returns_error() {
        let mut follower = make_node("follower");
        let result = follower.propose_command(TsdbCommand::DeleteSeries {
            series_id: "x".to_string(),
        });
        assert!(result.is_err());
        match result {
            Err(RaftError::NotLeader { .. }) => {}
            other => panic!("expected NotLeader, got {:?}", other),
        }
    }

    #[test]
    fn test_propose_command_as_leader() {
        let mut state = make_node("leader");
        state.become_candidate();
        state.become_leader(&[]);

        let cmd = TsdbCommand::WriteDatapoint {
            series_id: "ts1".to_string(),
            timestamp: 12345,
            value: 99.9,
        };
        let idx = state
            .propose_command(cmd)
            .expect("should succeed as leader");
        assert!(idx >= 1);
    }

    // ---- apply_committed_entries ------------------------------------------

    #[test]
    fn test_apply_committed_entries() {
        let mut state = make_node("leader");
        state.become_candidate();
        state.become_leader(&[]);

        // Propose 3 commands
        for i in 0..3u64 {
            state
                .propose_command(TsdbCommand::WriteDatapoint {
                    series_id: "s".to_string(),
                    timestamp: i as i64 * 1000,
                    value: i as f64,
                })
                .expect("propose ok");
        }

        // Manually advance commit_index (normally driven by quorum)
        state.commit_index = state.last_log_index();
        let applied = state.apply_committed_entries();

        // Should have applied entries up to commit_index.
        // Note: the no-op entry from become_leader is also included.
        assert!(!applied.is_empty());
        // last_applied should now equal commit_index
        assert_eq!(state.last_applied, state.commit_index);
    }

    // ---- try_advance_commit_index -----------------------------------------

    #[test]
    fn test_advance_commit_index_with_quorum() {
        let mut leader = make_node("leader");
        leader.become_candidate();
        leader.become_leader(&["p1".to_string(), "p2".to_string()]);

        leader
            .propose_command(TsdbCommand::DeleteSeries {
                series_id: "old".to_string(),
            })
            .expect("ok");

        // Simulate both peers acknowledging the entry
        let last = leader.last_log_index();
        leader.handle_append_success("p1", last);
        leader.handle_append_success("p2", last);

        let advanced = leader.try_advance_commit_index(3); // 3-node cluster
        assert!(advanced);
        assert!(leader.commit_index >= 1);
    }

    // ---- build_append_entries ---------------------------------------------

    #[test]
    fn test_build_append_entries_not_leader_errors() {
        let follower = make_node("follower");
        let result = follower.build_append_entries("peer");
        assert!(result.is_err());
    }

    #[test]
    fn test_build_append_entries_leader() {
        let mut leader = make_node("leader");
        leader.become_candidate();
        leader.become_leader(&["peer1".to_string()]);

        let args = leader
            .build_append_entries("peer1")
            .expect("should succeed");
        assert_eq!(args.leader_id, "leader");
        assert_eq!(args.term, leader.current_term);
    }

    // ---- SeriesMetadata builder -------------------------------------------

    #[test]
    fn test_series_metadata_builder() {
        let meta = SeriesMetadata::new("temperature")
            .with_tag("host", "server-1")
            .with_retention(90);
        assert_eq!(meta.name, "temperature");
        assert_eq!(meta.tags["host"], "server-1");
        assert_eq!(meta.retention_days, Some(90));
    }

    // ---- Edge cases -------------------------------------------------------

    #[test]
    fn test_candidate_steps_down_on_valid_append_entries() {
        let mut candidate = make_node("c");
        candidate.become_candidate();
        assert_eq!(candidate.role, RaftRole::Candidate);

        // Receive valid AppendEntries from a legitimate leader in same term
        let args = AppendEntriesArgs {
            term: candidate.current_term,
            leader_id: "real-leader".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
        };
        let reply = candidate.handle_append_entries(&args);
        assert!(reply.success);
        assert_eq!(candidate.role, RaftRole::Follower);
    }

    #[test]
    fn test_log_index_monotone_after_multiple_proposes() {
        let mut leader = make_node("leader");
        leader.become_candidate();
        leader.become_leader(&[]);

        let mut indices = Vec::new();
        for i in 0..10 {
            let idx = leader
                .propose_command(TsdbCommand::Compact {
                    series_id: "s".to_string(),
                    up_to_timestamp: i * 1000,
                })
                .expect("propose ok");
            indices.push(idx);
        }
        // Indices must be strictly increasing
        for w in indices.windows(2) {
            assert!(w[1] > w[0], "log indices must be monotone increasing");
        }
    }
}
