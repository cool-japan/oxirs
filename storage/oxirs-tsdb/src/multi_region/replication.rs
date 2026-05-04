//! Cross-region asynchronous replication and conflict resolution.
//!
//! Each home-region commit produces one [`RegionWriteRecord`]; the
//! replicator buffers per-target fanout work in a queue and exposes:
//!
//! - [`CrossRegionReplicator::enqueue_fanout`] — schedule a record for
//!   replication to one or more peer regions.
//! - [`CrossRegionReplicator::drain_pending`] — apply all buffered records
//!   into the per-region authoritative map under the configured
//!   conflict-resolution policy.
//! - [`CrossRegionReplicator::record_heard_from`] — record an in-bound
//!   replication ack and let the caller observe per-region replication lag.
//!
//! The default conflict resolver is last-writer-wins keyed on
//! `(timestamp_ms, region_id)`. The lexicographic region tie-breaker
//! guarantees deterministic convergence across all regions without
//! coordination.

use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::routing::RegionId;
use crate::replication::WriteEntry;

// ─────────────────────────────────────────────────────────────────────────────
// Records and outcomes
// ─────────────────────────────────────────────────────────────────────────────

/// Authoritative per-key record produced by a home-region commit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegionWriteRecord {
    /// Region whose Raft group accepted this write.
    pub region: RegionId,
    /// Per-region Raft log index.
    pub log_index: u64,
    /// Logical key being mutated. Defaults to the metric name from the
    /// originating [`WriteEntry`].
    pub key: String,
    /// Wall-clock timestamp from the originating write.
    pub timestamp_ms: i64,
    /// Observed value.
    pub value: f64,
    /// Tag map serialised as a JSON object string for compactness on the
    /// wire. Stored as text so the replicator stays oxicode/serde-free.
    pub tags_json: String,
    /// Time at which the replicator first observed the record (epoch ms).
    pub observed_at_ms: u128,
}

impl RegionWriteRecord {
    /// Build a record from a [`WriteEntry`] that just committed in `region`
    /// at the given Raft log index.
    pub fn from_entry(region: &RegionId, entry: &WriteEntry, log_index: u64) -> Self {
        let tags_json = serde_json::to_string(&entry.tags).unwrap_or_else(|_| "{}".to_string());
        let observed_at_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or_default();
        Self {
            region: region.clone(),
            log_index,
            key: entry.metric_name.clone(),
            timestamp_ms: entry.timestamp,
            value: entry.value,
            tags_json,
            observed_at_ms,
        }
    }
}

/// Outcome of a single replicator step.
#[derive(Debug, Clone, PartialEq)]
pub enum FanoutOutcome {
    /// Record was scheduled for fanout to the listed peer regions.
    Enqueued {
        /// Peer regions waiting for the record.
        targets: Vec<RegionId>,
    },
    /// Record was rejected because a strictly newer record already exists.
    Rejected {
        /// Existing record's region (the winner).
        winner_region: RegionId,
        /// Existing record's timestamp.
        winner_timestamp_ms: i64,
    },
}

/// One pending fanout work item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FanoutEntry {
    /// Authoritative record to be replicated.
    pub record: RegionWriteRecord,
    /// Target region the record still needs to be applied to.
    pub target: RegionId,
}

/// Statistics returned by [`CrossRegionReplicator::stats`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FanoutQueueStats {
    /// Number of pending fanout entries per target region.
    pub pending_per_region: HashMap<RegionId, usize>,
    /// Total committed records observed by the replicator.
    pub total_committed: u64,
    /// Total records rejected by the conflict resolver.
    pub total_rejected: u64,
}

/// Final apply decision for one record at one peer region.
#[derive(Debug, Clone, PartialEq)]
pub enum FanoutResolution {
    /// Record was applied as the new authoritative value.
    Applied,
    /// Record was rejected — newer or causally-later value already in
    /// place.
    Rejected {
        /// Region that owns the existing record.
        winner_region: RegionId,
        /// Existing record's timestamp.
        winner_timestamp_ms: i64,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Conflict resolver
// ─────────────────────────────────────────────────────────────────────────────

/// Conflict-resolution policy used by the replicator. The default
/// implementation is `LastWriterWins` keyed on `(timestamp_ms, region_id)`.
#[derive(Debug, Clone)]
pub struct LwwConflictResolver;

impl Default for LwwConflictResolver {
    fn default() -> Self {
        Self
    }
}

impl LwwConflictResolver {
    /// Compare a candidate record against an existing record. Returns
    /// `true` when `candidate` should win.
    pub fn candidate_wins(
        &self,
        candidate: &RegionWriteRecord,
        existing: &RegionWriteRecord,
    ) -> bool {
        if candidate.timestamp_ms != existing.timestamp_ms {
            return candidate.timestamp_ms > existing.timestamp_ms;
        }
        // Lexicographic region tie-break gives every region the same
        // deterministic answer.
        candidate.region > existing.region
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Replicator
// ─────────────────────────────────────────────────────────────────────────────

/// Cross-region replicator: holds per-region authoritative views of every
/// key and queues outbound fanout work per peer region.
///
/// The replicator tracks one logical state-machine per region; the home
/// region writes into its own view first, then schedules fanout to peer
/// regions where conflict resolution runs again at the destination.
#[derive(Debug)]
pub struct CrossRegionReplicator {
    regions: Vec<RegionId>,
    resolver: LwwConflictResolver,
    /// Per-region authoritative per-key record map.
    /// `region_views[region][key]` is the current value at that region.
    region_views: HashMap<RegionId, HashMap<String, RegionWriteRecord>>,
    /// Per-target FIFO queue of records still to be applied.
    pending: HashMap<RegionId, VecDeque<FanoutEntry>>,
    /// Per-region last-heard observation timestamp (epoch ms).
    last_heard_ms: HashMap<RegionId, u128>,
    total_committed: u64,
    total_rejected: u64,
}

impl CrossRegionReplicator {
    /// Build a replicator that ships records to `regions`.
    pub fn new(regions: Vec<RegionId>) -> Self {
        let pending = regions
            .iter()
            .map(|r| (r.clone(), VecDeque::new()))
            .collect();
        let region_views = regions
            .iter()
            .map(|r| (r.clone(), HashMap::new()))
            .collect();
        Self {
            regions,
            resolver: LwwConflictResolver,
            region_views,
            pending,
            last_heard_ms: HashMap::new(),
            total_committed: 0,
            total_rejected: 0,
        }
    }

    /// Returns the configured peer regions.
    pub fn regions(&self) -> &[RegionId] {
        &self.regions
    }

    /// Total committed records observed by the replicator.
    pub fn total_committed(&self) -> u64 {
        self.total_committed
    }

    /// Total records rejected by the conflict resolver so far.
    pub fn total_rejected(&self) -> u64 {
        self.total_rejected
    }

    /// Pending fanout count for `target`. Unknown regions return `0`.
    pub fn pending_for(&self, target: &RegionId) -> usize {
        self.pending.get(target).map(|q| q.len()).unwrap_or(0)
    }

    /// Read-only access to the per-region authoritative view of `region`.
    pub fn view_of(&self, region: &RegionId) -> Option<&HashMap<String, RegionWriteRecord>> {
        self.region_views.get(region)
    }

    /// Returns the value of `key` as observed by `region`'s view.
    pub fn record_in(&self, region: &RegionId, key: &str) -> Option<&RegionWriteRecord> {
        self.region_views.get(region).and_then(|v| v.get(key))
    }

    /// Apply a [`RegionWriteRecord`] locally to its home region's view and
    /// schedule fanout to `targets`.
    ///
    /// `targets` should not contain the record's home region; it is filtered
    /// out defensively.
    pub fn enqueue_fanout(
        &mut self,
        record: RegionWriteRecord,
        targets: Vec<RegionId>,
    ) -> FanoutOutcome {
        // Apply locally first.
        let local_resolution = self.apply_remote(&record.region, &record);
        if let FanoutResolution::Rejected {
            winner_region,
            winner_timestamp_ms,
        } = local_resolution
        {
            return FanoutOutcome::Rejected {
                winner_region,
                winner_timestamp_ms,
            };
        }

        let mut shipped = Vec::with_capacity(targets.len());
        for t in targets {
            if t == record.region {
                continue; // never ship to ourselves.
            }
            if let Some(q) = self.pending.get_mut(&t) {
                q.push_back(FanoutEntry {
                    record: record.clone(),
                    target: t.clone(),
                });
                shipped.push(t);
            }
        }
        FanoutOutcome::Enqueued { targets: shipped }
    }

    /// Drain the queue and "apply" each pending [`FanoutEntry`] at its
    /// destination region. Returns the per-region number of *applied*
    /// (non-rejected) entries.
    ///
    /// In a real deployment, "apply" would send the record over the network
    /// and wait for an ack; this single-process replicator simulates the
    /// destination's view by re-using the same resolver against its own
    /// per-region map.
    pub fn drain_pending(&mut self) -> HashMap<RegionId, usize> {
        let mut applied: HashMap<RegionId, usize> = HashMap::new();
        for region in self.regions.clone() {
            let queue = match self.pending.get_mut(&region) {
                Some(q) => std::mem::take(q),
                None => continue,
            };
            let mut count = 0usize;
            for entry in queue {
                let resolution = self.apply_remote(&region, &entry.record);
                if let FanoutResolution::Applied = resolution {
                    count += 1;
                }
            }
            applied.insert(region, count);
        }
        applied
    }

    /// Apply a single record at `target_region`. Used internally by
    /// [`drain_pending`] but exposed so callers can simulate finer-grained
    /// drains in tests.
    pub fn apply_remote(
        &mut self,
        target_region: &RegionId,
        record: &RegionWriteRecord,
    ) -> FanoutResolution {
        // Compute resolution against the destination region's current view.
        let resolution = match self.region_views.get(target_region) {
            Some(view) => match view.get(&record.key) {
                Some(existing) => {
                    if record == existing {
                        // Idempotent re-apply (e.g. fanout of our own record).
                        // Treat as Applied without counting; the prior local
                        // apply already counted this commit.
                        return FanoutResolution::Applied;
                    }
                    if self.resolver.candidate_wins(record, existing) {
                        FanoutResolution::Applied
                    } else {
                        FanoutResolution::Rejected {
                            winner_region: existing.region.clone(),
                            winner_timestamp_ms: existing.timestamp_ms,
                        }
                    }
                }
                None => FanoutResolution::Applied,
            },
            None => {
                // Unknown target region — treat as no-op rejection so callers
                // can detect misrouted writes.
                return FanoutResolution::Rejected {
                    winner_region: record.region.clone(),
                    winner_timestamp_ms: record.timestamp_ms,
                };
            }
        };

        match &resolution {
            FanoutResolution::Applied => {
                if let Some(view) = self.region_views.get_mut(target_region) {
                    view.insert(record.key.clone(), record.clone());
                }
                // Local commit (target == record.region) is the canonical
                // moment we count "total committed".
                if target_region == &record.region {
                    self.total_committed += 1;
                }
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0);
                self.last_heard_ms.insert(record.region.clone(), now);
                self.last_heard_ms
                    .entry(target_region.clone())
                    .or_insert(now);
            }
            FanoutResolution::Rejected { .. } => {
                self.total_rejected += 1;
            }
        }
        resolution
    }

    /// Record a "heard from" event for `region`.  Useful when the caller has
    /// out-of-band knowledge (e.g. a cross-region heartbeat) of region
    /// liveness independent of fanout traffic.
    pub fn record_heard_from(&mut self, region: &RegionId) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        self.last_heard_ms.insert(region.clone(), now);
    }

    /// Returns the last-heard timestamp (epoch ms) for `region`, if known.
    pub fn last_heard(&self, region: &RegionId) -> Option<u128> {
        self.last_heard_ms.get(region).copied()
    }

    /// Compose statistics for monitoring.
    pub fn stats(&self) -> FanoutQueueStats {
        let mut pending: HashMap<RegionId, usize> = HashMap::new();
        for (r, q) in &self.pending {
            pending.insert(r.clone(), q.len());
        }
        FanoutQueueStats {
            pending_per_region: pending,
            total_committed: self.total_committed,
            total_rejected: self.total_rejected,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replication::WriteEntry;

    fn rec(region: &str, key: &str, ts: i64, val: f64) -> RegionWriteRecord {
        RegionWriteRecord {
            region: region.into(),
            log_index: 1,
            key: key.into(),
            timestamp_ms: ts,
            value: val,
            tags_json: "{}".into(),
            observed_at_ms: 0,
        }
    }

    #[test]
    fn enqueue_simple_fanout() {
        let mut r = CrossRegionReplicator::new(vec!["us".into(), "eu".into(), "ap".into()]);
        let record = rec("us", "metrics.cpu", 1_000, 1.5);
        let out = r.enqueue_fanout(record, vec!["eu".into(), "ap".into()]);
        match out {
            FanoutOutcome::Enqueued { targets } => {
                assert!(targets.contains(&"eu".to_string()));
                assert!(targets.contains(&"ap".to_string()));
            }
            other => panic!("unexpected outcome: {other:?}"),
        }
        assert_eq!(r.pending_for(&"eu".to_string()), 1);
        assert_eq!(r.pending_for(&"ap".to_string()), 1);
        assert_eq!(r.total_committed(), 1);
    }

    #[test]
    fn lww_rejects_older_record() {
        let mut r = CrossRegionReplicator::new(vec!["us".into(), "eu".into()]);
        // First write commits in "us" *and* fans out to "eu".
        r.enqueue_fanout(rec("us", "k", 200, 1.0), vec!["eu".into()]);
        r.drain_pending();
        // A subsequent write in "eu" with an older timestamp must be
        // rejected at the "eu" view.
        let outcome = r.enqueue_fanout(rec("eu", "k", 100, 99.0), vec!["us".into()]);
        match outcome {
            FanoutOutcome::Rejected {
                winner_region,
                winner_timestamp_ms,
            } => {
                assert_eq!(winner_region, "us");
                assert_eq!(winner_timestamp_ms, 200);
            }
            other => panic!("expected rejection, got {other:?}"),
        }
        assert_eq!(r.total_rejected(), 1);
        assert_eq!(r.total_committed(), 1);
    }

    #[test]
    fn lww_tiebreak_by_lexicographic_region() {
        let mut r = CrossRegionReplicator::new(vec!["us".into(), "eu".into()]);
        r.enqueue_fanout(rec("eu", "k", 100, 1.0), vec!["us".into()]);
        // Drain so the eu→us fanout actually applies in the us view.
        r.drain_pending();
        // Same timestamp but us > eu so us wins.
        let outcome = r.enqueue_fanout(rec("us", "k", 100, 2.0), vec!["eu".into()]);
        if let FanoutOutcome::Enqueued { .. } = outcome {
            // pass
        } else {
            panic!("us should win the tiebreak: got {outcome:?}");
        }
        let auth = r
            .record_in(&"us".to_string(), "k")
            .cloned()
            .expect("us view");
        assert_eq!(auth.region, "us");
    }

    #[test]
    fn drain_applies_to_targets() {
        let mut r = CrossRegionReplicator::new(vec!["us".into(), "eu".into(), "ap".into()]);
        r.enqueue_fanout(rec("us", "k", 200, 1.0), vec!["eu".into(), "ap".into()]);
        let applied = r.drain_pending();
        assert_eq!(applied.get("eu").copied(), Some(1));
        assert_eq!(applied.get("ap").copied(), Some(1));
    }

    #[test]
    fn drain_skips_self_region() {
        let mut r = CrossRegionReplicator::new(vec!["us".into(), "eu".into()]);
        // Even when caller passes us, we should not enqueue to ourselves.
        let out = r.enqueue_fanout(rec("us", "k", 1, 0.0), vec!["us".into(), "eu".into()]);
        match out {
            FanoutOutcome::Enqueued { targets } => {
                assert_eq!(targets, vec!["eu".to_string()]);
            }
            _ => panic!(),
        }
        assert_eq!(r.pending_for(&"us".to_string()), 0);
        assert_eq!(r.pending_for(&"eu".to_string()), 1);
    }

    #[test]
    fn record_heard_from_records_timestamp() {
        let mut r = CrossRegionReplicator::new(vec!["us".into(), "eu".into()]);
        let r_id = "eu".to_string();
        assert!(r.last_heard(&r_id).is_none());
        r.record_heard_from(&r_id);
        assert!(r.last_heard(&r_id).is_some());
    }

    #[test]
    fn stats_reports_pending() {
        let mut r = CrossRegionReplicator::new(vec!["a".into(), "b".into()]);
        r.enqueue_fanout(rec("a", "k", 1, 0.0), vec!["b".into()]);
        let s = r.stats();
        assert_eq!(s.total_committed, 1);
        assert_eq!(s.pending_per_region.get("b").copied(), Some(1));
    }

    #[test]
    fn fanout_from_entry() {
        let entry = WriteEntry::new(1_700_000_000, "metrics.cpu", 0.7).with_tag("host", "h1");
        let record = RegionWriteRecord::from_entry(&"us".to_string(), &entry, 5);
        assert_eq!(record.key, "metrics.cpu");
        assert_eq!(record.timestamp_ms, 1_700_000_000);
        assert!((record.value - 0.7).abs() < 1e-12);
        assert!(record.tags_json.contains("h1"));
        assert_eq!(record.region, "us");
        assert_eq!(record.log_index, 5);
    }

    #[test]
    fn apply_remote_sets_authoritative() {
        let mut r = CrossRegionReplicator::new(vec!["us".into(), "eu".into()]);
        let res = r.apply_remote(&"eu".to_string(), &rec("us", "k", 100, 1.0));
        assert_eq!(res, FanoutResolution::Applied);
        assert_eq!(r.record_in(&"eu".to_string(), "k").unwrap().region, "us");
    }

    #[test]
    fn apply_remote_rejects_stale() {
        let mut r = CrossRegionReplicator::new(vec!["us".into(), "eu".into()]);
        // Establish ts=200 on the "us" view.
        r.apply_remote(&"us".to_string(), &rec("us", "k", 200, 1.0));
        // Now deliver an older record with origin "eu" to the "us" view.
        let res = r.apply_remote(&"us".to_string(), &rec("eu", "k", 100, 9.0));
        assert!(matches!(res, FanoutResolution::Rejected { .. }));
        assert_eq!(r.total_rejected(), 1);
    }

    #[test]
    fn fanout_to_unknown_region_silently_dropped() {
        let mut r = CrossRegionReplicator::new(vec!["us".into()]);
        let out = r.enqueue_fanout(rec("us", "k", 1, 0.0), vec!["void".into()]);
        if let FanoutOutcome::Enqueued { targets } = out {
            assert!(targets.is_empty());
        } else {
            panic!();
        }
        assert_eq!(r.pending_for(&"void".to_string()), 0);
    }
}
