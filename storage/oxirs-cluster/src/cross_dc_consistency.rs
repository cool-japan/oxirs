//! # Cross-DC Consistency Enforcer
//!
//! Provides `CrossDcConsistencyEnforcer` for enforcing configurable consistency
//! levels (Eventual, Strong, Causal) across datacenter boundaries.
//!
//! Uses vector clocks for causal tracking and offers conflict resolution strategies
//! including Last-Write-Wins, Causal ordering, and Strong (quorum-based) consistency.

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

// ---------------------------------------------------------------------------
// Dc-level consistency level (distinct from cross_dc::ConsistencyLevel)
// ---------------------------------------------------------------------------

/// The consistency model enforced by `CrossDcConsistencyEnforcer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DcConsistencyLevel {
    /// Updates propagate asynchronously; no ordering guarantees.
    Eventual,
    /// All DCs must acknowledge before write completes; linearizable.
    Strong,
    /// Causal ordering is preserved via vector clocks; reads respect causal history.
    Causal,
}

impl Default for DcConsistencyLevel {
    fn default() -> Self {
        Self::Eventual
    }
}

impl std::fmt::Display for DcConsistencyLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Eventual => write!(f, "Eventual"),
            Self::Strong => write!(f, "Strong"),
            Self::Causal => write!(f, "Causal"),
        }
    }
}

// ---------------------------------------------------------------------------
// Vector clock
// ---------------------------------------------------------------------------

/// Per-node logical clock for causal consistency.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct DcVectorClock {
    pub clocks: BTreeMap<String, u64>,
}

impl DcVectorClock {
    /// Create a new, empty vector clock.
    pub fn new() -> Self {
        Self::default()
    }

    /// Increment the local entry for `dc_id`.
    pub fn tick(&mut self, dc_id: &str) {
        let v = self.clocks.entry(dc_id.to_string()).or_insert(0);
        *v += 1;
    }

    /// Merge (element-wise max) another clock into this one.
    pub fn merge(&mut self, other: &DcVectorClock) {
        for (dc, &t) in &other.clocks {
            let v = self.clocks.entry(dc.clone()).or_insert(0);
            if t > *v {
                *v = t;
            }
        }
    }

    /// Return a merged copy without mutating self.
    pub fn merged_with(&self, other: &DcVectorClock) -> DcVectorClock {
        let mut result = self.clone();
        result.merge(other);
        result
    }

    /// True if `self` causally precedes `other` (happens-before).
    pub fn happens_before(&self, other: &DcVectorClock) -> bool {
        let all_keys: std::collections::HashSet<_> =
            self.clocks.keys().chain(other.clocks.keys()).collect();
        let mut at_least_one_less = false;
        for k in all_keys {
            let a = self.clocks.get(k).copied().unwrap_or(0);
            let b = other.clocks.get(k).copied().unwrap_or(0);
            if a > b {
                return false;
            }
            if a < b {
                at_least_one_less = true;
            }
        }
        at_least_one_less
    }

    /// True if neither clock happens-before the other (concurrent).
    pub fn concurrent_with(&self, other: &DcVectorClock) -> bool {
        !self.happens_before(other) && !other.happens_before(self) && self != other
    }
}

// ---------------------------------------------------------------------------
// Versioned write entry
// ---------------------------------------------------------------------------

/// A single write operation tagged with a vector clock and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedWrite {
    /// Globally unique identifier (UUID v4 string).
    pub id: String,
    /// Originating datacenter identifier.
    pub origin_dc: String,
    /// Vector clock at the moment of origin.
    pub clock: DcVectorClock,
    /// Unix epoch in milliseconds.
    pub timestamp_ms: u64,
    /// Serialised payload (e.g. RDF triple(s) as JSON bytes).
    pub payload: Vec<u8>,
    /// Arbitrary key affected by this write (used for conflict detection).
    pub key: String,
}

impl VersionedWrite {
    /// Create a new `VersionedWrite`.
    pub fn new(
        id: impl Into<String>,
        origin_dc: impl Into<String>,
        clock: DcVectorClock,
        key: impl Into<String>,
        payload: Vec<u8>,
    ) -> Self {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;
        Self {
            id: id.into(),
            origin_dc: origin_dc.into(),
            clock,
            timestamp_ms: ts,
            payload,
            key: key.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Conflict resolution result
// ---------------------------------------------------------------------------

/// The outcome after resolving a conflict between two concurrent writes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictOutcome {
    /// Accept `winner_id`, discard `loser_id`.
    Resolved { winner_id: String, loser_id: String },
    /// Both writes are concurrent but non-conflicting; keep both.
    MergeKeepBoth { id_a: String, id_b: String },
    /// Manual intervention required.
    ManualRequired {
        id_a: String,
        id_b: String,
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// Pending write queue
// ---------------------------------------------------------------------------

/// Internal bookkeeping for writes awaiting remote acknowledgement.
#[derive(Debug)]
struct PendingWrite {
    #[allow(dead_code)]
    write: VersionedWrite,
    enqueued_at: Instant,
    acks_received: usize,
    acks_required: usize,
}

// ---------------------------------------------------------------------------
// Enforcer configuration
// ---------------------------------------------------------------------------

/// Configuration for `CrossDcConsistencyEnforcer`.
#[derive(Debug, Clone)]
pub struct EnforcerConfig {
    /// Local datacenter identifier.
    pub local_dc: String,
    /// All known datacenter identifiers (including local).
    pub all_dcs: Vec<String>,
    /// Desired consistency level.
    pub consistency: DcConsistencyLevel,
    /// Maximum time to wait for remote acks (Strong/Causal only).
    pub ack_timeout: Duration,
    /// Capacity of the in-memory write history used for causal reads.
    pub history_capacity: usize,
}

impl EnforcerConfig {
    /// Create a default config for the given DC and consistency level.
    pub fn new(
        local_dc: impl Into<String>,
        all_dcs: Vec<String>,
        consistency: DcConsistencyLevel,
    ) -> Self {
        Self {
            local_dc: local_dc.into(),
            all_dcs,
            consistency,
            ack_timeout: Duration::from_secs(5),
            history_capacity: 1024,
        }
    }
}

// ---------------------------------------------------------------------------
// CrossDcConsistencyEnforcer
// ---------------------------------------------------------------------------

/// Enforces cross-datacenter consistency for distributed writes.
///
/// # Consistency levels
///
/// | Level    | Guarantee                                   | Latency |
/// |----------|---------------------------------------------|---------|
/// | Eventual | Writes propagate eventually, no ordering    | Lowest  |
/// | Causal   | Causal order preserved via vector clocks    | Medium  |
/// | Strong   | All DCs ack before write is confirmed       | Highest |
pub struct CrossDcConsistencyEnforcer {
    config: EnforcerConfig,
    /// Logical clock maintained by this DC.
    local_clock: Arc<RwLock<DcVectorClock>>,
    /// Pending writes waiting for remote acknowledgements.
    pending: Arc<RwLock<HashMap<String, PendingWrite>>>,
    /// Ordered write history for causal-read enforcement.
    history: Arc<RwLock<VecDeque<VersionedWrite>>>,
    /// Remote DCs that have acknowledged writes (dc -> set of write IDs).
    remote_acks: Arc<RwLock<HashMap<String, std::collections::HashSet<String>>>>,
    /// Number of writes successfully enforced.
    writes_enforced: Arc<RwLock<u64>>,
    /// Number of conflicts resolved.
    conflicts_resolved: Arc<RwLock<u64>>,
}

impl CrossDcConsistencyEnforcer {
    /// Create a new enforcer with the given configuration.
    pub fn new(config: EnforcerConfig) -> Self {
        Self {
            config,
            local_clock: Arc::new(RwLock::new(DcVectorClock::new())),
            pending: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(VecDeque::new())),
            remote_acks: Arc::new(RwLock::new(HashMap::new())),
            writes_enforced: Arc::new(RwLock::new(0)),
            conflicts_resolved: Arc::new(RwLock::new(0)),
        }
    }

    /// Return a reference to the enforcer configuration.
    pub fn config(&self) -> &EnforcerConfig {
        &self.config
    }

    /// Return the current local vector clock snapshot.
    pub async fn local_clock(&self) -> DcVectorClock {
        self.local_clock.read().await.clone()
    }

    // -----------------------------------------------------------------------
    // Write submission
    // -----------------------------------------------------------------------

    /// Submit a local write.
    ///
    /// Ticks the local clock, wraps the payload in a `VersionedWrite`, adds it
    /// to the pending queue (for Strong/Causal), and appends it to history.
    ///
    /// Returns the write ID.
    pub async fn submit_write(
        &self,
        write_id: impl Into<String>,
        key: impl Into<String>,
        payload: Vec<u8>,
    ) -> Result<VersionedWrite> {
        // Advance local clock.
        let clock = {
            let mut lc = self.local_clock.write().await;
            lc.tick(&self.config.local_dc);
            lc.clone()
        };

        let acks_required = self.acks_required_for_level();
        let vw = VersionedWrite::new(write_id, &self.config.local_dc, clock, key, payload);

        // Track in history.
        self.append_history(vw.clone()).await;

        if acks_required > 0 {
            let pw = PendingWrite {
                write: vw.clone(),
                enqueued_at: Instant::now(),
                acks_received: 0,
                acks_required,
            };
            self.pending.write().await.insert(vw.id.clone(), pw);
        }

        *self.writes_enforced.write().await += 1;
        Ok(vw)
    }

    /// Acknowledge a write from a remote DC.
    ///
    /// Returns `true` if the write is now fully committed (all acks received).
    pub async fn acknowledge(&self, write_id: &str, from_dc: &str) -> Result<bool> {
        // Record the ack.
        self.remote_acks
            .write()
            .await
            .entry(from_dc.to_string())
            .or_default()
            .insert(write_id.to_string());

        // Update pending tracker.
        let mut pending = self.pending.write().await;
        if let Some(pw) = pending.get_mut(write_id) {
            pw.acks_received += 1;
            if pw.acks_received >= pw.acks_required {
                pending.remove(write_id);
                return Ok(true);
            }
            return Ok(false);
        }
        // Write not in pending (already resolved or eventual consistency).
        Ok(true)
    }

    /// Receive a write from a remote DC and integrate it into local history.
    ///
    /// The local clock is advanced to be causally after the remote write.
    /// Returns a conflict outcome if a concurrent write to the same key exists.
    pub async fn receive_remote_write(
        &self,
        remote: VersionedWrite,
    ) -> Result<Option<ConflictOutcome>> {
        // Advance local clock.
        {
            let mut lc = self.local_clock.write().await;
            lc.merge(&remote.clock);
            lc.tick(&self.config.local_dc);
        }

        // Check for conflicts with existing history.
        let conflict = self.detect_conflict(&remote).await;

        // Append to history.
        self.append_history(remote.clone()).await;

        if let Some(outcome) = conflict {
            *self.conflicts_resolved.write().await += 1;
            return Ok(Some(outcome));
        }

        Ok(None)
    }

    // -----------------------------------------------------------------------
    // Conflict resolution
    // -----------------------------------------------------------------------

    /// Resolve a conflict between two concurrent `VersionedWrite`s.
    ///
    /// Strategy:
    /// - Strong consistency: use timestamp as tie-breaker (LWW).
    /// - Causal: earlier causal write wins; concurrent → LWW.
    /// - Eventual: LWW (last-write-wins by timestamp).
    pub fn resolve_conflict(&self, a: &VersionedWrite, b: &VersionedWrite) -> ConflictOutcome {
        match self.config.consistency {
            DcConsistencyLevel::Strong => self.last_write_wins(a, b),
            DcConsistencyLevel::Causal => {
                if a.clock.happens_before(&b.clock) {
                    // a is causally before b → b wins
                    ConflictOutcome::Resolved {
                        winner_id: b.id.clone(),
                        loser_id: a.id.clone(),
                    }
                } else if b.clock.happens_before(&a.clock) {
                    ConflictOutcome::Resolved {
                        winner_id: a.id.clone(),
                        loser_id: b.id.clone(),
                    }
                } else {
                    // Concurrent → fallback to LWW
                    self.last_write_wins(a, b)
                }
            }
            DcConsistencyLevel::Eventual => self.last_write_wins(a, b),
        }
    }

    // -----------------------------------------------------------------------
    // Status & metrics
    // -----------------------------------------------------------------------

    /// Returns the number of writes currently pending remote acknowledgements.
    pub async fn pending_count(&self) -> usize {
        self.pending.read().await.len()
    }

    /// Returns the number of writes that have been submitted.
    pub async fn writes_enforced(&self) -> u64 {
        *self.writes_enforced.read().await
    }

    /// Returns the number of conflicts that have been resolved.
    pub async fn conflicts_resolved(&self) -> u64 {
        *self.conflicts_resolved.read().await
    }

    /// Returns a snapshot of the write history (most recent last).
    pub async fn history_snapshot(&self) -> Vec<VersionedWrite> {
        self.history.read().await.iter().cloned().collect()
    }

    /// Check whether all pending writes have been acknowledged.
    pub async fn is_fully_acknowledged(&self) -> bool {
        self.pending.read().await.is_empty()
    }

    /// Expire pending writes that have exceeded the ack timeout.
    ///
    /// Returns the IDs of expired writes.
    pub async fn expire_timed_out_pending(&self) -> Vec<String> {
        let now = Instant::now();
        let timeout = self.config.ack_timeout;
        let mut pending = self.pending.write().await;
        let expired: Vec<String> = pending
            .iter()
            .filter(|(_, pw)| now.duration_since(pw.enqueued_at) > timeout)
            .map(|(id, _)| id.clone())
            .collect();
        for id in &expired {
            pending.remove(id);
        }
        expired
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn acks_required_for_level(&self) -> usize {
        let remote_dcs = self.config.all_dcs.len().saturating_sub(1);
        match self.config.consistency {
            DcConsistencyLevel::Eventual => 0,
            DcConsistencyLevel::Causal => {
                // At least one remote DC must ack for causal durability.
                if remote_dcs > 0 {
                    1
                } else {
                    0
                }
            }
            DcConsistencyLevel::Strong => remote_dcs,
        }
    }

    async fn append_history(&self, vw: VersionedWrite) {
        let mut hist = self.history.write().await;
        if hist.len() >= self.config.history_capacity {
            hist.pop_front();
        }
        hist.push_back(vw);
    }

    async fn detect_conflict(&self, incoming: &VersionedWrite) -> Option<ConflictOutcome> {
        let hist = self.history.read().await;
        // Look for a write to the same key from a different origin that is potentially
        // conflicting.  A conflict exists whenever the incoming write does NOT causally
        // dominate the existing write, i.e. the existing write is not known to the
        // incoming writer.  This covers the common cross-DC case where two DCs write
        // to the same key without observing each other's writes first:
        //   - existing clock: {dc-a: 1}  (local write)
        //   - incoming clock: {}          (remote write created without seeing dc-a)
        // Here {} does NOT causally dominate {dc-a: 1}, so we declare a conflict.
        // We additionally exclude the trivial case where both clocks are identical
        // (which would mean the same write is being replayed).
        for existing in hist.iter().rev() {
            if existing.key == incoming.key
                && existing.origin_dc != incoming.origin_dc
                && existing.clock != incoming.clock
                && !existing.clock.happens_before(&incoming.clock)
            {
                return Some(self.resolve_conflict(existing, incoming));
            }
        }
        None
    }

    fn last_write_wins(&self, a: &VersionedWrite, b: &VersionedWrite) -> ConflictOutcome {
        if a.timestamp_ms >= b.timestamp_ms {
            ConflictOutcome::Resolved {
                winner_id: a.id.clone(),
                loser_id: b.id.clone(),
            }
        } else {
            ConflictOutcome::Resolved {
                winner_id: b.id.clone(),
                loser_id: a.id.clone(),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Causal read barrier
// ---------------------------------------------------------------------------

/// Ensures that reads observe a causal snapshot consistent with a given clock.
pub struct CausalReadBarrier {
    enforcer: Arc<CrossDcConsistencyEnforcer>,
}

impl CausalReadBarrier {
    /// Create a new barrier backed by the given enforcer.
    pub fn new(enforcer: Arc<CrossDcConsistencyEnforcer>) -> Self {
        Self { enforcer }
    }

    /// Return the writes in history that causally precede or equal `clock`.
    pub async fn reads_consistent_with(&self, clock: &DcVectorClock) -> Vec<VersionedWrite> {
        self.enforcer
            .history_snapshot()
            .await
            .into_iter()
            .filter(|w| w.clock.happens_before(clock) || &w.clock == clock)
            .collect()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::runtime::Runtime;

    fn make_enforcer(level: DcConsistencyLevel) -> CrossDcConsistencyEnforcer {
        let config = EnforcerConfig::new(
            "dc-a",
            vec!["dc-a".into(), "dc-b".into(), "dc-c".into()],
            level,
        );
        CrossDcConsistencyEnforcer::new(config)
    }

    fn rt() -> Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime")
    }

    // -----------------------------------------------------------------------
    // DcVectorClock
    // -----------------------------------------------------------------------

    #[test]
    fn test_vector_clock_tick() {
        let mut vc = DcVectorClock::new();
        vc.tick("dc-a");
        vc.tick("dc-a");
        assert_eq!(vc.clocks["dc-a"], 2);
    }

    #[test]
    fn test_vector_clock_merge() {
        let mut a = DcVectorClock::new();
        a.tick("dc-a");
        a.tick("dc-a");

        let mut b = DcVectorClock::new();
        b.tick("dc-b");
        b.tick("dc-a"); // dc-a=1

        a.merge(&b);
        assert_eq!(a.clocks["dc-a"], 2); // max(2,1)
        assert_eq!(a.clocks["dc-b"], 1);
    }

    #[test]
    fn test_happens_before() {
        let mut a = DcVectorClock::new();
        a.tick("dc-a");

        let mut b = DcVectorClock::new();
        b.tick("dc-a");
        b.tick("dc-b");

        assert!(a.happens_before(&b));
        assert!(!b.happens_before(&a));
    }

    #[test]
    fn test_concurrent_clocks() {
        let mut a = DcVectorClock::new();
        a.tick("dc-a");

        let mut b = DcVectorClock::new();
        b.tick("dc-b");

        assert!(a.concurrent_with(&b));
        assert!(b.concurrent_with(&a));
    }

    #[test]
    fn test_equal_clocks_not_concurrent() {
        let mut a = DcVectorClock::new();
        a.tick("dc-a");
        let b = a.clone();
        assert!(!a.concurrent_with(&b));
    }

    #[test]
    fn test_merged_with_does_not_mutate() {
        let mut a = DcVectorClock::new();
        a.tick("dc-a");
        let mut b = DcVectorClock::new();
        b.tick("dc-b");

        let merged = a.merged_with(&b);
        assert_eq!(merged.clocks.len(), 2);
        assert_eq!(a.clocks.len(), 1); // not mutated
    }

    // -----------------------------------------------------------------------
    // Consistency level default / display
    // -----------------------------------------------------------------------

    #[test]
    fn test_consistency_level_default() {
        assert_eq!(DcConsistencyLevel::default(), DcConsistencyLevel::Eventual);
    }

    #[test]
    fn test_consistency_level_display() {
        assert_eq!(DcConsistencyLevel::Strong.to_string(), "Strong");
        assert_eq!(DcConsistencyLevel::Causal.to_string(), "Causal");
        assert_eq!(DcConsistencyLevel::Eventual.to_string(), "Eventual");
    }

    // -----------------------------------------------------------------------
    // Submit and track writes
    // -----------------------------------------------------------------------

    #[test]
    fn test_submit_write_eventual() {
        let rt = rt();
        rt.block_on(async {
            let enforcer = make_enforcer(DcConsistencyLevel::Eventual);
            let vw = enforcer
                .submit_write("w1", "key:foo", b"payload".to_vec())
                .await
                .expect("submit");
            assert_eq!(vw.origin_dc, "dc-a");
            assert_eq!(vw.key, "key:foo");
            assert_eq!(enforcer.writes_enforced().await, 1);
        });
    }

    #[test]
    fn test_submit_write_advances_local_clock() {
        let rt = rt();
        rt.block_on(async {
            let enforcer = make_enforcer(DcConsistencyLevel::Causal);
            enforcer
                .submit_write("w1", "key:foo", vec![])
                .await
                .expect("submit");
            let clock = enforcer.local_clock().await;
            assert_eq!(clock.clocks["dc-a"], 1);
        });
    }

    #[test]
    fn test_eventual_writes_have_no_pending() {
        let rt = rt();
        rt.block_on(async {
            let enforcer = make_enforcer(DcConsistencyLevel::Eventual);
            enforcer
                .submit_write("w1", "k", vec![1])
                .await
                .expect("submit");
            assert_eq!(enforcer.pending_count().await, 0);
        });
    }

    #[test]
    fn test_strong_writes_have_pending_until_ack() {
        let rt = rt();
        rt.block_on(async {
            let enforcer = make_enforcer(DcConsistencyLevel::Strong);
            let vw = enforcer
                .submit_write("w1", "k", vec![1])
                .await
                .expect("submit");
            assert_eq!(enforcer.pending_count().await, 1);

            // Ack from dc-b
            let done = enforcer.acknowledge(&vw.id, "dc-b").await.expect("ack");
            assert!(!done); // still needs dc-c

            // Ack from dc-c
            let done = enforcer.acknowledge(&vw.id, "dc-c").await.expect("ack");
            assert!(done);
            assert_eq!(enforcer.pending_count().await, 0);
        });
    }

    #[test]
    fn test_causal_write_pending_until_one_ack() {
        let rt = rt();
        rt.block_on(async {
            let enforcer = make_enforcer(DcConsistencyLevel::Causal);
            let vw = enforcer
                .submit_write("w1", "k", vec![])
                .await
                .expect("submit");
            assert_eq!(enforcer.pending_count().await, 1);

            // One ack suffices for Causal
            let done = enforcer.acknowledge(&vw.id, "dc-b").await.expect("ack");
            assert!(done);
            assert_eq!(enforcer.pending_count().await, 0);
        });
    }

    // -----------------------------------------------------------------------
    // Receive remote writes
    // -----------------------------------------------------------------------

    #[test]
    fn test_receive_remote_write_no_conflict() {
        let rt = rt();
        rt.block_on(async {
            let enforcer = make_enforcer(DcConsistencyLevel::Causal);
            let mut clock = DcVectorClock::new();
            clock.tick("dc-b");
            let remote = VersionedWrite::new("rw1", "dc-b", clock, "key:bar", vec![2]);
            let outcome = enforcer
                .receive_remote_write(remote)
                .await
                .expect("receive");
            assert!(outcome.is_none());
        });
    }

    #[test]
    fn test_receive_remote_write_advances_local_clock() {
        let rt = rt();
        rt.block_on(async {
            let enforcer = make_enforcer(DcConsistencyLevel::Causal);
            let mut remote_clock = DcVectorClock::new();
            remote_clock.tick("dc-b");
            remote_clock.tick("dc-b");
            let remote = VersionedWrite::new("rw2", "dc-b", remote_clock, "k", vec![]);
            enforcer.receive_remote_write(remote).await.expect("recv");
            let lc = enforcer.local_clock().await;
            // Local should have dc-b=2 merged in
            assert_eq!(lc.clocks.get("dc-b").copied().unwrap_or(0), 2);
        });
    }

    // -----------------------------------------------------------------------
    // Conflict detection and resolution
    // -----------------------------------------------------------------------

    #[test]
    fn test_conflict_detected_concurrent_same_key() {
        let rt = rt();
        rt.block_on(async {
            let enforcer = make_enforcer(DcConsistencyLevel::Eventual);

            // Local write
            enforcer
                .submit_write("w-local", "key:x", vec![1])
                .await
                .expect("submit");

            // Concurrent remote write to same key
            let remote_clock = DcVectorClock::new(); // clock is empty → concurrent with local
            let remote = VersionedWrite::new("w-remote", "dc-b", remote_clock, "key:x", vec![2]);
            let outcome = enforcer.receive_remote_write(remote).await.expect("recv");
            assert!(outcome.is_some(), "Expected conflict");
            assert!(enforcer.conflicts_resolved().await >= 1);
        });
    }

    #[test]
    fn test_no_conflict_different_keys() {
        let rt = rt();
        rt.block_on(async {
            let enforcer = make_enforcer(DcConsistencyLevel::Causal);
            enforcer
                .submit_write("w1", "key:alpha", vec![])
                .await
                .expect("submit");
            let remote =
                VersionedWrite::new("r1", "dc-b", DcVectorClock::new(), "key:beta", vec![]);
            let outcome = enforcer.receive_remote_write(remote).await.expect("recv");
            assert!(outcome.is_none());
        });
    }

    #[test]
    fn test_resolve_conflict_causal_earlier_wins() {
        let enforcer = make_enforcer(DcConsistencyLevel::Causal);
        let mut clock_a = DcVectorClock::new();
        clock_a.tick("dc-a");

        let mut clock_b = clock_a.clone();
        clock_b.tick("dc-b"); // b causally after a

        let a = VersionedWrite::new("a", "dc-a", clock_a, "k", vec![]);
        let b = VersionedWrite::new("b", "dc-b", clock_b, "k", vec![]);

        // b happens-after a → b wins
        let outcome = enforcer.resolve_conflict(&a, &b);
        assert_eq!(
            outcome,
            ConflictOutcome::Resolved {
                winner_id: "b".into(),
                loser_id: "a".into(),
            }
        );
    }

    #[test]
    fn test_resolve_conflict_strong_lww() {
        let enforcer = make_enforcer(DcConsistencyLevel::Strong);
        let mut a = VersionedWrite::new("a", "dc-a", DcVectorClock::new(), "k", vec![]);
        let mut b = VersionedWrite::new("b", "dc-b", DcVectorClock::new(), "k", vec![]);
        // Force timestamps
        a.timestamp_ms = 1000;
        b.timestamp_ms = 2000;

        let outcome = enforcer.resolve_conflict(&a, &b);
        assert_eq!(
            outcome,
            ConflictOutcome::Resolved {
                winner_id: "b".into(),
                loser_id: "a".into(),
            }
        );
    }

    #[test]
    fn test_resolve_conflict_eventual_lww() {
        let enforcer = make_enforcer(DcConsistencyLevel::Eventual);
        let mut a = VersionedWrite::new("a", "dc-a", DcVectorClock::new(), "k", vec![]);
        let mut b = VersionedWrite::new("b", "dc-b", DcVectorClock::new(), "k", vec![]);
        a.timestamp_ms = 5000;
        b.timestamp_ms = 3000;
        let outcome = enforcer.resolve_conflict(&a, &b);
        assert_eq!(
            outcome,
            ConflictOutcome::Resolved {
                winner_id: "a".into(),
                loser_id: "b".into(),
            }
        );
    }

    // -----------------------------------------------------------------------
    // History and causal reads
    // -----------------------------------------------------------------------

    #[test]
    fn test_history_grows_with_writes() {
        let rt = rt();
        rt.block_on(async {
            let enforcer = make_enforcer(DcConsistencyLevel::Eventual);
            for i in 0..5u32 {
                enforcer
                    .submit_write(format!("w{i}"), "k", vec![i as u8])
                    .await
                    .expect("submit");
            }
            assert_eq!(enforcer.history_snapshot().await.len(), 5);
        });
    }

    #[test]
    fn test_history_capacity_capped() {
        let rt = rt();
        rt.block_on(async {
            let mut config =
                EnforcerConfig::new("dc-a", vec!["dc-a".into()], DcConsistencyLevel::Eventual);
            config.history_capacity = 3;
            let enforcer = CrossDcConsistencyEnforcer::new(config);
            for i in 0..10u32 {
                enforcer
                    .submit_write(format!("w{i}"), "k", vec![])
                    .await
                    .expect("submit");
            }
            assert_eq!(enforcer.history_snapshot().await.len(), 3);
        });
    }

    #[test]
    fn test_causal_read_barrier_filters_by_clock() {
        let rt = rt();
        rt.block_on(async {
            let enforcer = Arc::new(make_enforcer(DcConsistencyLevel::Causal));
            let barrier = CausalReadBarrier::new(Arc::clone(&enforcer));

            // Submit two writes, each advancing clock
            let w1 = enforcer
                .submit_write("w1", "k1", vec![1])
                .await
                .expect("submit");
            let _w2 = enforcer
                .submit_write("w2", "k2", vec![2])
                .await
                .expect("submit");

            // Reads consistent with w1's clock should only include w1.
            let reads = barrier.reads_consistent_with(&w1.clock).await;
            // w1 clock happens-before w2 clock, so only w1 qualifies.
            assert!(reads.iter().any(|w| w.id == "w1"));
            // w2 clock is strictly after w1's clock → should not be included.
            assert!(!reads.iter().any(|w| w.id == "w2"));
        });
    }

    // -----------------------------------------------------------------------
    // Pending expiry
    // -----------------------------------------------------------------------

    #[test]
    fn test_expire_timed_out_strong_write() {
        let rt = rt();
        rt.block_on(async {
            let mut config = EnforcerConfig::new(
                "dc-a",
                vec!["dc-a".into(), "dc-b".into()],
                DcConsistencyLevel::Strong,
            );
            config.ack_timeout = Duration::from_millis(1); // very short
            let enforcer = CrossDcConsistencyEnforcer::new(config);
            let _vw = enforcer
                .submit_write("w-timeout", "k", vec![])
                .await
                .expect("submit");
            assert_eq!(enforcer.pending_count().await, 1);

            // Sleep to exceed timeout
            tokio::time::sleep(Duration::from_millis(5)).await;
            let expired = enforcer.expire_timed_out_pending().await;
            assert_eq!(expired.len(), 1);
            assert_eq!(enforcer.pending_count().await, 0);
        });
    }

    // -----------------------------------------------------------------------
    // Fully acknowledged helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_fully_acknowledged_eventual() {
        let rt = rt();
        rt.block_on(async {
            let enforcer = make_enforcer(DcConsistencyLevel::Eventual);
            enforcer
                .submit_write("w1", "k", vec![])
                .await
                .expect("submit");
            assert!(enforcer.is_fully_acknowledged().await);
        });
    }

    #[test]
    fn test_acks_required_no_remote_dcs() {
        let rt = rt();
        rt.block_on(async {
            let config =
                EnforcerConfig::new("dc-a", vec!["dc-a".into()], DcConsistencyLevel::Strong);
            let enforcer = CrossDcConsistencyEnforcer::new(config);
            // With only one DC, strong writes require 0 remote acks.
            let vw = enforcer
                .submit_write("w1", "k", vec![])
                .await
                .expect("submit");
            assert_eq!(enforcer.pending_count().await, 0);
            assert_eq!(vw.origin_dc, "dc-a");
        });
    }

    #[test]
    fn test_multiple_conflicts_counted() {
        let rt = rt();
        rt.block_on(async {
            let enforcer = make_enforcer(DcConsistencyLevel::Eventual);
            // Two local writes to same key
            enforcer
                .submit_write("local1", "k", vec![1])
                .await
                .expect("s");
            enforcer
                .submit_write("local2", "k", vec![2])
                .await
                .expect("s");

            // Remote concurrent write
            let r = VersionedWrite::new("r1", "dc-b", DcVectorClock::new(), "k", vec![3]);
            enforcer.receive_remote_write(r).await.expect("recv");

            // Should have resolved at least one conflict
            assert!(enforcer.conflicts_resolved().await >= 1);
        });
    }
}
