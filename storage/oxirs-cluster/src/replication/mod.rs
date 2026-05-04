//! # Data Replication
//!
//! High-level data replication management for distributed RDF storage.
//!
//! The `manager` submodule contains the long-standing intra-cluster
//! [`ReplicationManager`] used for Raft-backed replicas. The new W2-S5
//! submodules add active-active geo geometry on top of `cross_dc`:
//!
//! * [`active_active_geo`] — `ActiveActiveGeoConfig`, region routing rules,
//!   per-region Raft group identifiers, and the active-active replicator
//!   that tracks LWW vector-clock conflict resolution across regions.
//! * [`region_failover`] — finite state machine that demotes an unreachable
//!   primary region and promotes a healthy secondary, replaying outstanding
//!   writes on recovery.
//! * [`cross_region_anti_entropy`] — extends the existing intra-cluster
//!   `MerkleTree` to compare per-region trees and produce a divergence
//!   report consumed by `active_active_geo` for catch-up replication.

pub mod active_active_geo;
pub mod cross_region_anti_entropy;
pub mod manager;
pub mod region_failover;

pub use manager::{
    ReplicaInfo, ReplicationError, ReplicationManager, ReplicationStats, ReplicationStrategy,
};

pub use active_active_geo::{
    ActiveActiveGeoConfig, ActiveActiveReplicator, ConflictResolutionMode, GeoWriteOutcome,
    RegionRaftGroup, RegionRoutingRule, WriteRoutingDecision,
};
pub use cross_region_anti_entropy::{
    CrossRegionAntiEntropy, CrossRegionDivergence, RegionDivergence,
};
pub use region_failover::{
    FailoverEvent, RegionFailoverController, RegionFailoverError, RegionRole, RegionState,
};
