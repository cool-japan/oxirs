//! Cluster topology management
//!
//! Provides hierarchical topology (Region → AZ → Rack → Node) and consistent
//! hash-based shard placement via a virtual node ring.

pub mod hierarchy;
pub mod placement;
pub mod scale;

// Re-export primary types
pub use hierarchy::{
    AvailabilityZone, ClusterNode, ClusterTopology, NodeCapacity, NodeRole, NodeState, Rack,
    Region, TopologyDigest,
};
pub use placement::{plan_rebalance, RebalancePlan, ShardMove, VNodeRing};
pub use scale::{
    ClusterScaleManager, ConsistentHashRing, NodeGroup, NodeMeta, TopologyAwarePlacement,
    VirtualNode,
};
