//! Active-active geo replication geometry.
//!
//! Composes with the existing [`crate::cross_dc::CrossDcReplicationManager`]
//! to provide an active-active topology where multiple regions accept local
//! writes simultaneously. Each region runs its own Raft group and the
//! replicator ships committed log entries asynchronously to peer regions.
//!
//! # Overview
//!
//! * [`ActiveActiveGeoConfig`] declares the participating regions, the
//!   primary tier mapping, and a list of [`RegionRoutingRule`]s used to
//!   classify incoming writes.
//! * Each region is represented by a [`RegionRaftGroup`] which owns its
//!   own per-region Raft consensus identifier, sequence counter, and
//!   conflict-resolution policy.
//! * [`ActiveActiveReplicator`] sits on top of these per-region groups and
//!   dispatches writes through the configured routing rules. Conflicts
//!   between concurrent regional writes are resolved with one of the
//!   [`ConflictResolutionMode`]s — by default a CRDT-style last-writer-wins
//!   over (timestamp, region-id) — falling back to vector-clock causal
//!   resolution when explicitly configured.
//!
//! The module never bypasses Raft: every write is first committed in its
//! origin region's Raft log. Cross-region propagation is asynchronous and
//! eventually consistent.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::conflict_resolution::VectorClock;
use crate::cross_dc::ConsistencyLevel;
use crate::error::{ClusterError, Result};
use crate::raft::OxirsNodeId;

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// Region identifier (e.g. `"us-east-1"`, `"eu-west-1"`).
pub type RegionId = String;

/// Conflict resolution mode for concurrent cross-region writes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictResolutionMode {
    /// CRDT last-writer-wins over (timestamp_ms, region_id) — default.
    ///
    /// Two concurrent writes keep the one with the highest timestamp; ties
    /// are broken deterministically by lexicographic region id so that all
    /// regions converge on the same winner without coordination.
    LastWriterWins,
    /// Vector-clock causality: drop writes that happen-before an existing
    /// observation, accept those that strictly happen-after, and apply the
    /// LWW tiebreak only to *concurrent* writes.
    VectorClock,
}

impl Default for ConflictResolutionMode {
    fn default() -> Self {
        ConflictResolutionMode::LastWriterWins
    }
}

/// Routing decision for a single inbound write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteRoutingDecision {
    /// Region whose Raft group should accept the write.
    pub primary_region: RegionId,
    /// Regions that must receive an asynchronous copy of the committed log
    /// entry.
    pub fanout_regions: Vec<RegionId>,
    /// Consistency level honoured by the cross-region replicator.
    pub consistency: ConsistencyLevel,
}

/// Rule that maps a write to a primary region.
///
/// Rules are evaluated in declaration order; the first one whose
/// `match_predicate` returns `true` wins. Common predicates include
/// "subject namespace prefix", "tenant id", or "graph IRI".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionRoutingRule {
    /// Human-readable name of the rule (used for diagnostics).
    pub name: String,
    /// Optional namespace prefix that the write subject must start with for
    /// the rule to fire. `None` matches everything.
    pub subject_prefix: Option<String>,
    /// Optional tenant identifier that the write must carry.
    pub tenant_id: Option<String>,
    /// Region selected when the rule matches.
    pub target_region: RegionId,
    /// Consistency level demanded by writes routed via this rule.
    pub consistency: ConsistencyLevel,
}

impl RegionRoutingRule {
    /// Convenience constructor for a "match every write" default rule.
    pub fn default_to(region: impl Into<RegionId>, consistency: ConsistencyLevel) -> Self {
        Self {
            name: "default".into(),
            subject_prefix: None,
            tenant_id: None,
            target_region: region.into(),
            consistency,
        }
    }

    /// Returns `true` when the rule matches the given (subject, tenant) pair.
    pub fn matches(&self, subject: &str, tenant: Option<&str>) -> bool {
        if let Some(prefix) = &self.subject_prefix {
            if !subject.starts_with(prefix) {
                return false;
            }
        }
        if let Some(want_tenant) = &self.tenant_id {
            match tenant {
                Some(t) if t == want_tenant => {}
                _ => return false,
            }
        }
        true
    }
}

/// Active-active geo configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveActiveGeoConfig {
    /// All regions that participate in the deployment.
    pub regions: Vec<RegionId>,
    /// Region that hosts the locally running node — used as the default
    /// origin when no rule matches a particular write.
    pub local_region: RegionId,
    /// Map from logical "tier" (e.g. `"primary"`, `"secondary"`) to the
    /// regions that fulfil it. Higher tiers are tried first when a region
    /// fails over.
    pub primary_tier: BTreeMap<String, Vec<RegionId>>,
    /// Ordered routing rules. First match wins; if none match, the default
    /// region is `local_region` with `LocalAsync` consistency.
    pub routing_rules: Vec<RegionRoutingRule>,
    /// Conflict resolution mode for concurrent writes.
    pub conflict_mode: ConflictResolutionMode,
    /// Maximum acceptable cross-region replication lag before the
    /// replicator marks a stream as stale.
    pub max_lag: Duration,
}

impl ActiveActiveGeoConfig {
    /// Construct a minimal configuration with only the local region.
    pub fn single_region(local: impl Into<RegionId>) -> Self {
        let local = local.into();
        Self {
            regions: vec![local.clone()],
            local_region: local.clone(),
            primary_tier: BTreeMap::new(),
            routing_rules: vec![RegionRoutingRule::default_to(
                local,
                ConsistencyLevel::LocalAsync,
            )],
            conflict_mode: ConflictResolutionMode::default(),
            max_lag: Duration::from_secs(5),
        }
    }

    /// Build a multi-region configuration with the given regions, marking
    /// `local` as the local region and giving every region the same routing
    /// behaviour.
    pub fn multi_region(local: impl Into<RegionId>, regions: Vec<RegionId>) -> Self {
        let local = local.into();
        let mut all = regions;
        if !all.contains(&local) {
            all.insert(0, local.clone());
        }
        let mut primary_tier = BTreeMap::new();
        primary_tier.insert("primary".to_string(), all.clone());
        Self {
            regions: all,
            local_region: local.clone(),
            primary_tier,
            routing_rules: vec![RegionRoutingRule::default_to(
                local,
                ConsistencyLevel::LocalAsync,
            )],
            conflict_mode: ConflictResolutionMode::default(),
            max_lag: Duration::from_secs(5),
        }
    }

    /// Decide where a given write should be routed.
    pub fn route(&self, subject: &str, tenant: Option<&str>) -> WriteRoutingDecision {
        for rule in &self.routing_rules {
            if rule.matches(subject, tenant) {
                let fanout = self
                    .regions
                    .iter()
                    .filter(|r| **r != rule.target_region)
                    .cloned()
                    .collect();
                return WriteRoutingDecision {
                    primary_region: rule.target_region.clone(),
                    fanout_regions: fanout,
                    consistency: rule.consistency.clone(),
                };
            }
        }
        // Fallback: route to the local region.
        let fanout = self
            .regions
            .iter()
            .filter(|r| **r != self.local_region)
            .cloned()
            .collect();
        WriteRoutingDecision {
            primary_region: self.local_region.clone(),
            fanout_regions: fanout,
            consistency: ConsistencyLevel::LocalAsync,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-region Raft group
// ─────────────────────────────────────────────────────────────────────────────

/// Identifies the Raft group running in a single region.
///
/// Each region runs its own independent Raft instance (separate term space,
/// separate log indices); this struct gives the active-active replicator the
/// metadata it needs without needing a live `Raft` handle (which is built up
/// elsewhere via [`crate::consensus::ConsensusManager`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionRaftGroup {
    /// Region identifier.
    pub region_id: RegionId,
    /// Stable group identifier shared by all members of the per-region Raft
    /// instance.
    pub group_id: u64,
    /// Node ids that are voting members of this region's Raft group.
    pub members: Vec<OxirsNodeId>,
    /// Highest sequence number committed in this region.
    pub committed_seq: u64,
}

impl RegionRaftGroup {
    /// Build a new group definition.
    pub fn new(region_id: impl Into<RegionId>, group_id: u64, members: Vec<OxirsNodeId>) -> Self {
        Self {
            region_id: region_id.into(),
            group_id,
            members,
            committed_seq: 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Active-active outcome
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome of a single active-active write attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeoWriteOutcome {
    /// Write was committed in the primary region; `seq` is the per-region
    /// sequence number assigned to the entry.
    Committed { region: RegionId, seq: u64 },
    /// Write was rejected by the active-active layer because another region
    /// already committed a strictly newer (or causally later) entry for the
    /// same key.
    RejectedByConflict {
        region: RegionId,
        winner_region: RegionId,
        winner_timestamp_ms: u64,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Replicator state
// ─────────────────────────────────────────────────────────────────────────────

/// Snapshot of one record's authoritative state in the replicator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeoRecord {
    /// Region that owns the current value.
    pub region: RegionId,
    /// Wall-clock timestamp (milliseconds since UNIX epoch).
    pub timestamp_ms: u64,
    /// Vector clock at the time of the write — used in `VectorClock` mode.
    pub clock: VectorClock,
    /// The value itself (opaque payload).
    pub value: String,
}

/// Active-active geo replicator.
///
/// Tracks the authoritative value of every key it has observed and applies
/// the configured conflict-resolution policy to incoming writes.
#[derive(Debug)]
pub struct ActiveActiveReplicator {
    config: ActiveActiveGeoConfig,
    state: Arc<Mutex<ReplicatorState>>,
}

#[derive(Debug)]
struct ReplicatorState {
    /// Per-region Raft groups.
    groups: HashMap<RegionId, RegionRaftGroup>,
    /// Per-key authoritative record.
    records: HashMap<String, GeoRecord>,
    /// Per-region sequence counter (increments on every committed local
    /// write originating in that region).
    region_seq: HashMap<RegionId, u64>,
    /// Per-region pending fanout queue (entries that still need to be
    /// shipped to peer regions).
    pending_fanout: HashMap<RegionId, Vec<PendingFanout>>,
}

/// A committed local write awaiting fanout to peer regions.
#[derive(Debug, Clone)]
struct PendingFanout {
    key: String,
    record: GeoRecord,
    seq: u64,
    targets: Vec<RegionId>,
}

impl ActiveActiveReplicator {
    /// Build a replicator with the given configuration and per-region Raft
    /// groups.
    pub fn new(config: ActiveActiveGeoConfig, groups: Vec<RegionRaftGroup>) -> Result<Self> {
        if !config.regions.contains(&config.local_region) {
            return Err(ClusterError::Config(format!(
                "local_region '{}' is not in the configured regions list",
                config.local_region
            )));
        }
        for g in &groups {
            if !config.regions.contains(&g.region_id) {
                return Err(ClusterError::Config(format!(
                    "Raft group region '{}' is not in the configured regions list",
                    g.region_id
                )));
            }
        }
        let region_seq = config
            .regions
            .iter()
            .map(|r| (r.clone(), 0u64))
            .collect::<HashMap<_, _>>();
        let pending_fanout = config
            .regions
            .iter()
            .map(|r| (r.clone(), Vec::new()))
            .collect::<HashMap<_, _>>();
        let groups_map = groups
            .into_iter()
            .map(|g| (g.region_id.clone(), g))
            .collect();
        let state = ReplicatorState {
            groups: groups_map,
            records: HashMap::new(),
            region_seq,
            pending_fanout,
        };
        Ok(Self {
            config,
            state: Arc::new(Mutex::new(state)),
        })
    }

    /// Returns the configuration in use.
    pub fn config(&self) -> &ActiveActiveGeoConfig {
        &self.config
    }

    /// Apply a *local* write that just committed in the named region's Raft
    /// log. Returns the assigned sequence number on success or a conflict
    /// outcome.
    pub fn apply_local_write(
        &self,
        region: &RegionId,
        key: &str,
        value: &str,
        timestamp_ms: u64,
        clock: VectorClock,
    ) -> Result<GeoWriteOutcome> {
        let mut st = self.lock_state()?;
        if !st.region_seq.contains_key(region) {
            return Err(ClusterError::Config(format!(
                "Region '{}' is not part of the active-active deployment",
                region
            )));
        }
        let new_record = GeoRecord {
            region: region.clone(),
            timestamp_ms,
            clock,
            value: value.to_owned(),
        };
        match self.resolve(&st, key, &new_record) {
            Resolution::Accept => {
                let seq = {
                    let counter = st.region_seq.entry(region.clone()).or_insert(0);
                    *counter += 1;
                    *counter
                };
                if let Some(group) = st.groups.get_mut(region) {
                    group.committed_seq = group.committed_seq.max(seq);
                }
                let targets: Vec<RegionId> = self
                    .config
                    .regions
                    .iter()
                    .filter(|r| r != &region)
                    .cloned()
                    .collect();
                let pending = PendingFanout {
                    key: key.to_string(),
                    record: new_record.clone(),
                    seq,
                    targets,
                };
                st.records.insert(key.to_string(), new_record);
                if let Some(queue) = st.pending_fanout.get_mut(region) {
                    queue.push(pending);
                }
                Ok(GeoWriteOutcome::Committed {
                    region: region.clone(),
                    seq,
                })
            }
            Resolution::Reject(winner) => Ok(GeoWriteOutcome::RejectedByConflict {
                region: region.clone(),
                winner_region: winner.region.clone(),
                winner_timestamp_ms: winner.timestamp_ms,
            }),
        }
    }

    /// Apply a *remote* write received from a peer region's fanout. Resolves
    /// the conflict with the existing authoritative record (if any) using
    /// the configured policy.
    pub fn apply_remote_write(
        &self,
        origin_region: &RegionId,
        key: &str,
        value: &str,
        timestamp_ms: u64,
        clock: VectorClock,
    ) -> Result<GeoWriteOutcome> {
        let mut st = self.lock_state()?;
        if !st.region_seq.contains_key(origin_region) {
            return Err(ClusterError::Config(format!(
                "Origin region '{}' is not part of the active-active deployment",
                origin_region
            )));
        }
        let new_record = GeoRecord {
            region: origin_region.clone(),
            timestamp_ms,
            clock,
            value: value.to_owned(),
        };
        match self.resolve(&st, key, &new_record) {
            Resolution::Accept => {
                st.records.insert(key.to_string(), new_record);
                // Bump the origin region's seq pointer to track remote
                // writes we have observed.
                let counter = st.region_seq.entry(origin_region.clone()).or_insert(0);
                *counter += 1;
                let seq = *counter;
                Ok(GeoWriteOutcome::Committed {
                    region: origin_region.clone(),
                    seq,
                })
            }
            Resolution::Reject(winner) => Ok(GeoWriteOutcome::RejectedByConflict {
                region: origin_region.clone(),
                winner_region: winner.region.clone(),
                winner_timestamp_ms: winner.timestamp_ms,
            }),
        }
    }

    /// Drain pending fanout entries that the named source region has
    /// queued up for the given target region.
    pub fn drain_fanout(
        &self,
        source_region: &RegionId,
        target_region: &RegionId,
    ) -> Result<Vec<(String, GeoRecord, u64)>> {
        let mut st = self.lock_state()?;
        let mut keep: Vec<PendingFanout> = Vec::new();
        let mut emit: Vec<(String, GeoRecord, u64)> = Vec::new();
        let queue = st.pending_fanout.remove(source_region).unwrap_or_default();
        for mut entry in queue {
            if let Some(idx) = entry.targets.iter().position(|t| t == target_region) {
                emit.push((entry.key.clone(), entry.record.clone(), entry.seq));
                entry.targets.remove(idx);
            }
            if !entry.targets.is_empty() {
                keep.push(entry);
            }
        }
        st.pending_fanout.insert(source_region.clone(), keep);
        Ok(emit)
    }

    /// Look up the current authoritative record for `key`.
    pub fn get_record(&self, key: &str) -> Result<Option<GeoRecord>> {
        let st = self.lock_state()?;
        Ok(st.records.get(key).cloned())
    }

    /// Number of records currently tracked.
    pub fn record_count(&self) -> Result<usize> {
        let st = self.lock_state()?;
        Ok(st.records.len())
    }

    /// Snapshot the per-region committed sequence counters.
    pub fn region_sequences(&self) -> Result<BTreeMap<RegionId, u64>> {
        let st = self.lock_state()?;
        Ok(st.region_seq.iter().map(|(k, v)| (k.clone(), *v)).collect())
    }

    /// Number of pending fanout entries for `source_region`.
    pub fn pending_fanout_len(&self, source_region: &RegionId) -> Result<usize> {
        let st = self.lock_state()?;
        Ok(st
            .pending_fanout
            .get(source_region)
            .map(|v| v.len())
            .unwrap_or(0))
    }

    fn resolve<'r>(
        &self,
        st: &'r ReplicatorState,
        key: &str,
        new_record: &GeoRecord,
    ) -> Resolution<'r> {
        let existing = match st.records.get(key) {
            Some(r) => r,
            None => return Resolution::Accept,
        };
        match self.config.conflict_mode {
            ConflictResolutionMode::LastWriterWins => lww_resolve(existing, new_record),
            ConflictResolutionMode::VectorClock => vector_clock_resolve(existing, new_record),
        }
    }

    fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, ReplicatorState>> {
        self.state
            .lock()
            .map_err(|e| ClusterError::Lock(format!("active-active state lock poisoned: {}", e)))
    }
}

enum Resolution<'r> {
    Accept,
    Reject(&'r GeoRecord),
}

fn lww_resolve<'r>(existing: &'r GeoRecord, new_record: &GeoRecord) -> Resolution<'r> {
    if new_record.timestamp_ms > existing.timestamp_ms {
        Resolution::Accept
    } else if new_record.timestamp_ms == existing.timestamp_ms {
        // Tie-break by lexicographic region id; the higher region id wins.
        if new_record.region > existing.region {
            Resolution::Accept
        } else {
            Resolution::Reject(existing)
        }
    } else {
        Resolution::Reject(existing)
    }
}

fn vector_clock_resolve<'r>(existing: &'r GeoRecord, new_record: &GeoRecord) -> Resolution<'r> {
    if existing.clock.happens_before(&new_record.clock) {
        // New strictly happens-after existing — accept.
        Resolution::Accept
    } else if new_record.clock.happens_before(&existing.clock) {
        // Existing already saw a causally-later write — reject.
        Resolution::Reject(existing)
    } else {
        // Concurrent — fall back to LWW for deterministic convergence.
        lww_resolve(existing, new_record)
    }
}

/// Helper: current wall-clock time in milliseconds since UNIX epoch.
pub fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn three_region_config() -> ActiveActiveGeoConfig {
        ActiveActiveGeoConfig::multi_region(
            "us-east-1",
            vec![
                "us-east-1".to_string(),
                "eu-west-1".to_string(),
                "ap-northeast-1".to_string(),
            ],
        )
    }

    fn replicator() -> ActiveActiveReplicator {
        let config = three_region_config();
        let groups = config
            .regions
            .iter()
            .enumerate()
            .map(|(i, r)| RegionRaftGroup::new(r.clone(), 1_000 + i as u64, vec![1, 2, 3]))
            .collect();
        ActiveActiveReplicator::new(config, groups).expect("valid config")
    }

    #[test]
    fn route_default_uses_local_region_for_unmatched_writes() {
        let cfg = three_region_config();
        let decision = cfg.route("http://example.org/s", None);
        assert_eq!(decision.primary_region, "us-east-1");
        assert!(!decision.fanout_regions.contains(&"us-east-1".to_string()));
        assert_eq!(decision.fanout_regions.len(), 2);
    }

    #[test]
    fn route_matches_subject_prefix_rule() {
        let mut cfg = three_region_config();
        cfg.routing_rules.insert(
            0,
            RegionRoutingRule {
                name: "eu-tenant".into(),
                subject_prefix: Some("https://eu.example.org/".into()),
                tenant_id: None,
                target_region: "eu-west-1".into(),
                consistency: ConsistencyLevel::EachQuorum,
            },
        );
        let decision = cfg.route("https://eu.example.org/foo", None);
        assert_eq!(decision.primary_region, "eu-west-1");
        assert_eq!(decision.consistency, ConsistencyLevel::EachQuorum);
    }

    #[test]
    fn local_write_assigns_monotonic_seq() {
        let r = replicator();
        let region: RegionId = "us-east-1".into();
        for i in 0..5 {
            let outcome = r
                .apply_local_write(
                    &region,
                    &format!("key-{}", i),
                    &format!("v{}", i),
                    current_timestamp_ms(),
                    VectorClock::new(),
                )
                .expect("apply");
            match outcome {
                GeoWriteOutcome::Committed { seq, .. } => assert_eq!(seq, (i + 1) as u64),
                _ => panic!("unexpected outcome"),
            }
        }
        assert_eq!(r.record_count().expect("count"), 5);
    }

    #[test]
    fn lww_newer_timestamp_wins() {
        let r = replicator();
        let us: RegionId = "us-east-1".into();
        let eu: RegionId = "eu-west-1".into();
        r.apply_local_write(&us, "k", "from-us", 100, VectorClock::new())
            .expect("first");
        let outcome = r
            .apply_remote_write(&eu, "k", "from-eu", 200, VectorClock::new())
            .expect("second");
        assert!(matches!(outcome, GeoWriteOutcome::Committed { .. }));
        let rec = r.get_record("k").expect("get").expect("present");
        assert_eq!(rec.value, "from-eu");
        assert_eq!(rec.region, "eu-west-1");
    }

    #[test]
    fn lww_older_timestamp_rejected() {
        let r = replicator();
        let us: RegionId = "us-east-1".into();
        let eu: RegionId = "eu-west-1".into();
        r.apply_local_write(&us, "k", "from-us", 200, VectorClock::new())
            .expect("first");
        let outcome = r
            .apply_remote_write(&eu, "k", "from-eu", 100, VectorClock::new())
            .expect("second");
        assert!(matches!(
            outcome,
            GeoWriteOutcome::RejectedByConflict { .. }
        ));
        let rec = r.get_record("k").expect("get").expect("present");
        assert_eq!(rec.value, "from-us");
    }

    #[test]
    fn lww_tie_breaks_on_region_id() {
        let r = replicator();
        let us: RegionId = "us-east-1".into();
        let eu: RegionId = "eu-west-1".into();
        r.apply_local_write(&us, "k", "from-us", 100, VectorClock::new())
            .expect("first");
        let outcome = r
            .apply_remote_write(&eu, "k", "from-eu", 100, VectorClock::new())
            .expect("second");
        // "us-east-1" > "eu-west-1" lexicographically, so the eu remote
        // write loses despite identical timestamp.
        assert!(matches!(
            outcome,
            GeoWriteOutcome::RejectedByConflict { .. }
        ));
        let rec = r.get_record("k").expect("get").expect("present");
        assert_eq!(rec.value, "from-us");
    }

    #[test]
    fn vector_clock_mode_accepts_strictly_later() {
        let mut cfg = three_region_config();
        cfg.conflict_mode = ConflictResolutionMode::VectorClock;
        let groups = cfg
            .regions
            .iter()
            .enumerate()
            .map(|(i, r)| RegionRaftGroup::new(r.clone(), 5_000 + i as u64, vec![1, 2, 3]))
            .collect();
        let r = ActiveActiveReplicator::new(cfg, groups).expect("valid");
        let mut clk1 = VectorClock::new();
        clk1.increment(1);
        r.apply_local_write(&"us-east-1".into(), "k", "v1", 100, clk1.clone())
            .expect("first");

        let mut clk2 = clk1.clone();
        clk2.increment(2);
        let outcome = r
            .apply_remote_write(&"eu-west-1".into(), "k", "v2", 50, clk2)
            .expect("second");
        // Even though timestamp is older, vector clock strictly happens-after,
        // so it should be accepted.
        assert!(matches!(outcome, GeoWriteOutcome::Committed { .. }));
        let rec = r.get_record("k").expect("get").expect("present");
        assert_eq!(rec.value, "v2");
    }

    #[test]
    fn fanout_drain_routes_only_to_target() {
        let r = replicator();
        let us: RegionId = "us-east-1".into();
        r.apply_local_write(&us, "k", "v", 100, VectorClock::new())
            .expect("apply");
        assert_eq!(r.pending_fanout_len(&us).expect("len"), 1);
        let drained_eu = r.drain_fanout(&us, &"eu-west-1".into()).expect("drain eu");
        assert_eq!(drained_eu.len(), 1);
        assert_eq!(drained_eu[0].0, "k");
        // After draining for eu, the entry is still pending for ap.
        assert_eq!(r.pending_fanout_len(&us).expect("len"), 1);
        let drained_ap = r
            .drain_fanout(&us, &"ap-northeast-1".into())
            .expect("drain ap");
        assert_eq!(drained_ap.len(), 1);
        assert_eq!(r.pending_fanout_len(&us).expect("len"), 0);
    }

    #[test]
    fn config_rejects_unknown_local_region() {
        let cfg = ActiveActiveGeoConfig {
            regions: vec!["us-east-1".into()],
            local_region: "eu-west-1".into(),
            primary_tier: BTreeMap::new(),
            routing_rules: Vec::new(),
            conflict_mode: ConflictResolutionMode::default(),
            max_lag: Duration::from_secs(5),
        };
        let res = ActiveActiveReplicator::new(cfg, Vec::new());
        assert!(res.is_err());
    }
}
