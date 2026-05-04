//! Active-active multi-region deployment for `oxirs-tsdb`.
//!
//! Provides a thin geometry layer that composes one
//! [`ReplicationGroup`](crate::replication::ReplicationGroup) per region with:
//!
//! 1. [`routing`] — write-routing policy that picks a *home region* for each
//!    incoming write based on subject prefix / tenant id rules with a
//!    deterministic fallback.
//! 2. [`health_probe`] — heartbeat tracker with timeout-based unreachable
//!    detection, exposing a [`RegionStatus`] map that the routing layer can
//!    consult to skip dead regions.
//! 3. [`replication`] — async cross-region fanout queue with last-writer-wins
//!    conflict resolution keyed on `(timestamp_ms, region_id)`.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │              ActiveActiveMultiRegion (this module)               │
//! │                                                                  │
//! │  ┌─────────┐    ┌─────────┐    ┌─────────┐                       │
//! │  │ region  │    │ region  │    │ region  │ ← routing.rs picks    │
//! │  │us-east-1│    │eu-west-1│    │ap-south │   the home region     │
//! │  │  Raft   │    │  Raft   │    │  Raft   │                       │
//! │  │  group  │    │  group  │    │  group  │                       │
//! │  └────┬────┘    └────┬────┘    └────┬────┘                       │
//! │       │              │              │                            │
//! │       └─────── async fanout queue ──┘ ← replication.rs           │
//! │              (LWW conflict resolution)                           │
//! │                                                                  │
//! │   health_probe.rs heartbeats every region, marks dead ones       │
//! │   Suspect after `failure_threshold` missed beats.                │
//! └──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! The module is **always compiled** (no feature gate) and depends only on
//! `oxirs-tsdb`'s existing replication primitives — no new external deps.
//!
//! ## References
//!
//! - The cross-DC primitives in `oxirs-cluster::replication::active_active_geo`
//!   (W2-S5) inspired the LWW + region-tier design; this module re-implements
//!   the small subset required for a TSDB-shaped active-active topology
//!   without taking a cross-crate dep.

pub mod health_probe;
pub mod replication;
pub mod routing;

pub use health_probe::{HealthConfig, HealthProbe, RegionHealthSnapshot, RegionStatus};
pub use replication::{
    CrossRegionReplicator, FanoutEntry, FanoutOutcome, FanoutQueueStats, FanoutResolution,
    LwwConflictResolver, RegionWriteRecord,
};
pub use routing::{
    RegionId, RouteContext, RouteDecision, RoutingError, RoutingTable, WriteRoutingRule,
};

use crate::error::{TsdbError, TsdbResult};
use crate::replication::{ReplicationGroup, WriteEntry};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Top-level deployment
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for an active-active multi-region TSDB deployment.
#[derive(Debug, Clone)]
pub struct ActiveActiveConfig {
    /// All regions that participate in the deployment.
    pub regions: Vec<RegionId>,
    /// Region that hosts the locally running node.
    pub local_region: RegionId,
    /// Number of nodes in each per-region Raft group (must be ≥ 1).
    pub nodes_per_region: usize,
    /// Routing policy for incoming writes.
    pub routing: RoutingTable,
    /// Health-probe configuration shared by all regions.
    pub health: HealthConfig,
}

impl ActiveActiveConfig {
    /// Build a minimal config that contains only `local`.
    pub fn single_region(local: impl Into<RegionId>, nodes: usize) -> TsdbResult<Self> {
        if nodes == 0 {
            return Err(TsdbError::Config(
                "nodes_per_region must be at least 1".into(),
            ));
        }
        let local = local.into();
        Ok(Self {
            regions: vec![local.clone()],
            local_region: local.clone(),
            nodes_per_region: nodes,
            routing: RoutingTable::default_to(local),
            health: HealthConfig::default(),
        })
    }

    /// Build a multi-region config; `local` is always inserted into `regions`
    /// when missing.
    pub fn multi_region(
        local: impl Into<RegionId>,
        regions: Vec<RegionId>,
        nodes: usize,
    ) -> TsdbResult<Self> {
        if nodes == 0 {
            return Err(TsdbError::Config(
                "nodes_per_region must be at least 1".into(),
            ));
        }
        let local = local.into();
        let mut all = regions;
        if !all.contains(&local) {
            all.insert(0, local.clone());
        }
        if all.is_empty() {
            return Err(TsdbError::Config("regions list must not be empty".into()));
        }
        Ok(Self {
            regions: all,
            local_region: local.clone(),
            nodes_per_region: nodes,
            routing: RoutingTable::default_to(local),
            health: HealthConfig::default(),
        })
    }
}

/// The top-level orchestrator that wires together routing, per-region Raft
/// groups, health probing, and cross-region replication.
pub struct ActiveActiveMultiRegion {
    config: ActiveActiveConfig,
    groups: HashMap<RegionId, ReplicationGroup>,
    probe: HealthProbe,
    replicator: CrossRegionReplicator,
}

impl std::fmt::Debug for ActiveActiveMultiRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActiveActiveMultiRegion")
            .field("regions", &self.config.regions)
            .field("local_region", &self.config.local_region)
            .field("nodes_per_region", &self.config.nodes_per_region)
            .field(
                "group_node_counts",
                &self
                    .groups
                    .iter()
                    .map(|(r, g)| (r.clone(), g.cluster_size()))
                    .collect::<std::collections::BTreeMap<_, _>>(),
            )
            .finish()
    }
}

impl ActiveActiveMultiRegion {
    /// Construct an active-active deployment from `cfg`.
    ///
    /// Spawns one [`ReplicationGroup`] per region using
    /// `cfg.nodes_per_region` nodes and labels them `<region>-n0`,
    /// `<region>-n1`, … so node ids are globally unique.
    pub fn new(cfg: ActiveActiveConfig) -> TsdbResult<Self> {
        if !cfg.regions.contains(&cfg.local_region) {
            return Err(TsdbError::Config(format!(
                "local_region '{}' is not in the regions list",
                cfg.local_region
            )));
        }
        let mut groups = HashMap::with_capacity(cfg.regions.len());
        for region in &cfg.regions {
            let node_ids: Vec<String> = (0..cfg.nodes_per_region)
                .map(|i| format!("{region}-n{i}"))
                .collect();
            let timeouts: Vec<u32> = (0..cfg.nodes_per_region)
                .map(|i| 8 + i as u32 * 2)
                .collect();
            let id_refs: Vec<&str> = node_ids.iter().map(String::as_str).collect();
            let group = ReplicationGroup::new(&id_refs, Some(&timeouts));
            groups.insert(region.clone(), group);
        }
        let probe = HealthProbe::new(cfg.regions.clone(), cfg.health.clone());
        let replicator = CrossRegionReplicator::new(cfg.regions.clone());
        Ok(Self {
            config: cfg,
            groups,
            probe,
            replicator,
        })
    }

    /// Returns the deployment configuration.
    pub fn config(&self) -> &ActiveActiveConfig {
        &self.config
    }

    /// Returns the per-region [`ReplicationGroup`]s.
    pub fn groups(&self) -> &HashMap<RegionId, ReplicationGroup> {
        &self.groups
    }

    /// Mutable access to the per-region [`ReplicationGroup`]s.
    pub fn groups_mut(&mut self) -> &mut HashMap<RegionId, ReplicationGroup> {
        &mut self.groups
    }

    /// Mutable access to the health probe (used to record heartbeats and
    /// trigger tick advancement in tests).
    pub fn health_probe_mut(&mut self) -> &mut HealthProbe {
        &mut self.probe
    }

    /// Read-only health probe handle.
    pub fn health_probe(&self) -> &HealthProbe {
        &self.probe
    }

    /// Mutable access to the replicator.
    pub fn replicator_mut(&mut self) -> &mut CrossRegionReplicator {
        &mut self.replicator
    }

    /// Read-only access to the replicator.
    pub fn replicator(&self) -> &CrossRegionReplicator {
        &self.replicator
    }

    /// Drive every per-region Raft group forward by `ticks` simulated
    /// election ticks. Each tick also records a synthetic heartbeat for
    /// every locally hosted region so the embedded health probe stays
    /// `Healthy` while the test cluster runs. To simulate a network
    /// partition use [`HealthProbe::force_status`] via
    /// [`Self::health_probe_mut`] before driving more ticks.
    pub fn tick_all(&mut self, ticks: u32) {
        for group in self.groups.values_mut() {
            for _ in 0..ticks {
                group.tick();
            }
        }
        // Health probe tick uses logical ticks for unreachable detection.
        // We feed heartbeats for every region we own a Raft group for so
        // routing decisions stay healthy in the absence of explicit failure
        // injection.
        let regions: Vec<RegionId> = self.groups.keys().cloned().collect();
        for _ in 0..ticks {
            self.probe.tick();
            for r in &regions {
                self.probe.record_heartbeat(r);
            }
        }
    }

    /// Drive ticks on per-region Raft groups *without* feeding heartbeats —
    /// useful for partition / failure-injection tests where you want the
    /// health probe to age out a region.
    pub fn tick_all_silent(&mut self, ticks: u32) {
        for group in self.groups.values_mut() {
            for _ in 0..ticks {
                group.tick();
            }
        }
        for _ in 0..ticks {
            self.probe.tick();
        }
    }

    /// Submit a write to the deployment.
    ///
    /// 1. The routing table picks a home region (skipping any region the
    ///    health probe currently reports as `Failed`).
    /// 2. The write is appended to that region's Raft group via
    ///    [`ReplicationGroup::propose_and_commit`].
    /// 3. The replicator schedules an asynchronous fanout to peer regions.
    ///
    /// `commit_ticks` bounds how many simulated ticks we drive while waiting
    /// for the per-region quorum.
    ///
    /// Returns the chosen home region and the per-region Raft log index of
    /// the new entry.
    pub fn submit_write(
        &mut self,
        ctx: &RouteContext,
        entry: WriteEntry,
        commit_ticks: u32,
    ) -> TsdbResult<WriteOutcome> {
        let snapshot = self.probe.snapshot();
        let decision = self
            .config
            .routing
            .route(ctx, &snapshot)
            .map_err(|e| TsdbError::Replication(format!("routing failed: {e}")))?;

        let group = self.groups.get_mut(&decision.region).ok_or_else(|| {
            TsdbError::Replication(format!(
                "routing produced unknown region '{}'",
                decision.region
            ))
        })?;
        let (_leader, index) = group
            .propose_and_commit(&entry, commit_ticks)
            .map_err(|e| {
                TsdbError::Replication(format!("region '{}' commit failed: {e}", decision.region))
            })?;

        let peers: Vec<RegionId> = self
            .config
            .regions
            .iter()
            .filter(|r| **r != decision.region)
            .cloned()
            .collect();
        let record = RegionWriteRecord::from_entry(&decision.region, &entry, index);
        let outcome = self.replicator.enqueue_fanout(record, peers.clone());

        Ok(WriteOutcome {
            home_region: decision.region,
            log_index: index,
            fanout_targets: peers,
            replication_outcome: outcome,
        })
    }

    /// Drain the replicator queue and apply records to the destination region
    /// state. Returns the per-region count of successful applies.
    pub fn drain_replicator(&mut self) -> HashMap<RegionId, usize> {
        self.replicator.drain_pending()
    }
}

/// Outcome of [`ActiveActiveMultiRegion::submit_write`].
#[derive(Debug, Clone)]
pub struct WriteOutcome {
    /// Region that accepted the write into its Raft log.
    pub home_region: RegionId,
    /// Per-region Raft log index assigned to the entry.
    pub log_index: u64,
    /// Regions that will receive an asynchronous fanout copy.
    pub fanout_targets: Vec<RegionId>,
    /// Initial outcome reported by the cross-region replicator (whether the
    /// record was newer than any conflicting existing record).
    pub replication_outcome: FanoutOutcome,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replication::WriteEntry;
    use std::collections::HashMap as StdHashMap;

    #[test]
    fn build_multi_region_initialises_groups() {
        let cfg = ActiveActiveConfig::multi_region(
            "us-east",
            vec!["us-east".into(), "eu-west".into(), "ap-south".into()],
            3,
        )
        .expect("cfg");
        let mr = ActiveActiveMultiRegion::new(cfg).expect("mr");
        assert_eq!(mr.groups().len(), 3);
        assert!(mr.groups().contains_key("us-east"));
        assert!(mr.groups().contains_key("eu-west"));
        assert!(mr.groups().contains_key("ap-south"));
    }

    #[test]
    fn single_region_works() {
        let cfg = ActiveActiveConfig::single_region("solo", 3).expect("cfg");
        let mr = ActiveActiveMultiRegion::new(cfg).expect("mr");
        assert_eq!(mr.groups().len(), 1);
    }

    #[test]
    fn submit_write_routes_to_home_region() {
        let cfg = ActiveActiveConfig::multi_region(
            "us-east",
            vec!["us-east".into(), "eu-west".into()],
            3,
        )
        .expect("cfg");
        let mut mr = ActiveActiveMultiRegion::new(cfg).expect("mr");
        // Drive tick to elect leaders in every region.
        mr.tick_all(40);
        let entry = WriteEntry::new(1_000, "cpu", 50.0).with_tag("host", "h1");
        let ctx = RouteContext {
            subject: "cpu".into(),
            tenant: None,
            timestamp_ms: 1_000,
        };
        let outcome = mr.submit_write(&ctx, entry, 50).expect("submit");
        assert!(["us-east", "eu-west"].contains(&outcome.home_region.as_str()));
        assert!(outcome.log_index >= 1);
        assert_eq!(outcome.fanout_targets.len(), 1);
    }

    #[test]
    fn config_rejects_zero_nodes_per_region() {
        let err = ActiveActiveConfig::multi_region("a", vec!["a".into()], 0).err();
        assert!(err.is_some());
    }

    #[test]
    fn config_inserts_local_when_missing_from_regions() {
        let cfg = ActiveActiveConfig::multi_region("local", vec!["x".into(), "y".into()], 1)
            .expect("cfg");
        assert!(cfg.regions.contains(&"local".to_string()));
        assert!(cfg.regions.contains(&"x".to_string()));
        assert!(cfg.regions.contains(&"y".to_string()));
    }

    #[test]
    fn drain_replicator_returns_per_region_counts() {
        let cfg = ActiveActiveConfig::multi_region(
            "us-east",
            vec!["us-east".into(), "eu-west".into(), "ap-south".into()],
            3,
        )
        .expect("cfg");
        let mut mr = ActiveActiveMultiRegion::new(cfg).expect("mr");
        mr.tick_all(80);
        let entry = WriteEntry::new(2_000, "metrics.cpu", 22.0);
        let ctx = RouteContext {
            subject: "metrics.cpu".into(),
            tenant: None,
            timestamp_ms: 2_000,
        };
        let _ = mr.submit_write(&ctx, entry, 80).expect("submit");
        let drained = mr.drain_replicator();
        // The home region does not need fanout to itself.
        let total: usize = drained.values().sum();
        assert!(total >= 1);
        let _ = StdHashMap::<RegionId, usize>::new(); // silence unused import on some platforms.
    }
}
