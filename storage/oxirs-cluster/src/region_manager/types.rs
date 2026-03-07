//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::raft::OxirsNodeId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Geographic coordinates for latency estimation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GeoCoordinates {
    pub latitude: f64,
    pub longitude: f64,
}
/// Cross-region replication strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CrossRegionStrategy {
    /// Asynchronous replication to all regions
    AsyncAll,
    /// Replication to specific backup regions
    SelectiveSync { target_regions: Vec<String> },
    /// Eventual consistency with conflict resolution
    EventualConsistency { reconciliation_interval_ms: u64 },
    /// Chain replication across regions
    ChainReplication { replication_chain: Vec<String> },
}
/// Metadata for eventual consistency replication
#[derive(Debug, Clone)]
pub struct EventualConsistencyMetadata {
    pub timestamp: SystemTime,
    pub vector_clock: VectorClock,
    pub source_region: String,
    pub reconciliation_interval: Duration,
}
/// Region-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegionConfig {
    /// Preferred replication factor within region
    pub local_replication_factor: usize,
    /// Cross-region replication factor
    pub cross_region_replication_factor: usize,
    /// Maximum acceptable latency for regional operations (ms)
    pub max_regional_latency_ms: u64,
    /// Enable region-local leader preference
    pub prefer_local_leader: bool,
    /// Enable cross-region backup consensus
    pub enable_cross_region_backup: bool,
    /// Enable relay routing for high-latency paths
    pub enable_relay: bool,
    /// Relay latency threshold (ms) - use relay if improvement > threshold
    pub relay_latency_threshold_ms: f64,
    /// Enable compression for cross-region data
    pub enable_compression: bool,
    /// Enable read-local routing (route reads to nearest replica)
    pub enable_read_local: bool,
    /// Routing strategy to use
    pub routing_strategy: RoutingStrategy,
    /// Custom region properties
    pub properties: HashMap<String, String>,
}
/// Performance metrics for multi-region operations
#[derive(Debug)]
pub struct RegionPerformanceMetrics {
    /// Latency measurements between regions
    pub inter_region_latencies: HashMap<(String, String), LatencyStats>,
    /// Throughput metrics per region
    pub region_throughput: HashMap<String, ThroughputStats>,
    /// Error rates per region
    pub region_error_rates: HashMap<String, ErrorRateStats>,
    /// Last update timestamp
    pub last_updated: SystemTime,
    /// Whether monitoring is currently active
    pub monitoring_enabled: bool,
}
/// Routing strategy for cross-region communication
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RoutingStrategy {
    /// Always use direct connection
    Direct,
    /// Use Dijkstra for minimum latency path
    LatencyAware,
    /// Consider both latency and bandwidth
    BandwidthAware,
    /// Minimize cross-region data transfer costs
    CostAware,
}
/// Route between regions
#[derive(Debug, Clone)]
pub struct Route {
    /// Hops in the route (e.g., [us-west, us-east, eu-central])
    pub hops: Vec<String>,
    /// Total latency of the route in milliseconds
    pub total_latency: f64,
    /// Whether to use compression for this route
    pub use_compression: bool,
}
impl Route {
    /// Create a direct route between two regions
    pub fn direct(source: String, dest: String) -> Self {
        Self {
            hops: vec![source, dest],
            total_latency: 0.0,
            use_compression: true,
        }
    }
}
/// Vector clock for distributed consistency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorClock {
    pub clocks: HashMap<String, u64>,
}
impl VectorClock {
    /// Create a new empty vector clock
    pub fn new() -> Self {
        Self {
            clocks: HashMap::new(),
        }
    }
    /// Compare two vector clocks for ordering
    pub fn compare(&self, other: &VectorClock) -> VectorClockOrdering {
        let mut self_greater = false;
        let mut other_greater = false;
        let all_keys: std::collections::HashSet<_> =
            self.clocks.keys().chain(other.clocks.keys()).collect();
        for key in all_keys {
            let self_value = self.clocks.get(key).unwrap_or(&0);
            let other_value = other.clocks.get(key).unwrap_or(&0);
            if self_value > other_value {
                self_greater = true;
            } else if other_value > self_value {
                other_greater = true;
            }
        }
        match (self_greater, other_greater) {
            (true, false) => VectorClockOrdering::Greater,
            (false, true) => VectorClockOrdering::Less,
            (false, false) => VectorClockOrdering::Equal,
            (true, true) => VectorClockOrdering::Concurrent,
        }
    }
}
/// Throughput statistics
#[derive(Debug, Clone)]
pub struct ThroughputStats {
    pub operations_per_second: f64,
    pub bytes_per_second: f64,
    pub peak_ops_per_second: f64,
    pub last_measurement: SystemTime,
}
/// Region status enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum RegionStatus {
    Healthy,
    Degraded,
    Unavailable,
}
/// Status of connectivity between regions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConnectivityStatus {
    /// Full connectivity with low latency
    Optimal,
    /// Connectivity with elevated latency
    Degraded { latency_ms: u64 },
    /// Partial connectivity or intermittent issues
    Unstable { error_rate: f64 },
    /// No connectivity
    Disconnected,
}
/// Conflict resolution strategy for multi-region
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConflictResolutionStrategy {
    /// Last writer wins with timestamp
    LastWriterWins,
    /// Vector clock based resolution
    VectorClock,
    /// Application-defined custom resolution
    Custom { resolution_function: String },
    /// Manual resolution required
    Manual,
}
/// Latency statistics
#[derive(Debug, Default)]
pub struct LatencyStats {
    pub min_ms: f64,
    pub max_ms: f64,
    pub avg_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub sample_count: u64,
}
/// Multi-region deployment topology
#[derive(Debug, Clone)]
pub struct RegionTopology {
    /// All regions in the deployment
    pub regions: HashMap<String, Region>,
    /// Node placement mapping
    pub node_placements: HashMap<OxirsNodeId, NodePlacement>,
    /// Inter-region latency matrix (in milliseconds, f64 for precision)
    pub latency_matrix: HashMap<(String, String), f64>,
    /// Region-to-region connectivity status
    pub connectivity_status: HashMap<(String, String), ConnectivityStatus>,
}
/// Availability zone within a region
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct AvailabilityZone {
    /// AZ identifier (e.g., "us-east-1a")
    pub id: String,
    /// Human-readable AZ name
    pub name: String,
    /// Region this AZ belongs to
    pub region_id: String,
}
/// Package containing data and metadata for replication
#[derive(Debug, Clone)]
pub struct ReplicationPackage {
    pub data: Vec<u8>,
    pub metadata: EventualConsistencyMetadata,
}
/// Error rate statistics
#[derive(Debug, Clone)]
pub struct ErrorRateStats {
    pub total_operations: u64,
    pub failed_operations: u64,
    pub error_rate: f64,
    pub last_error: Option<SystemTime>,
}
/// Region health information
#[derive(Debug, Clone)]
pub struct RegionHealth {
    pub region_id: String,
    pub total_nodes: usize,
    pub healthy_nodes: usize,
    pub throughput: ThroughputStats,
    pub error_rate: ErrorRateStats,
    pub status: RegionStatus,
}
/// Replication strategy for multi-region deployment
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MultiRegionReplicationStrategy {
    /// Strategy for intra-region replication
    pub intra_region: IntraRegionStrategy,
    /// Strategy for cross-region replication
    pub cross_region: CrossRegionStrategy,
    /// Conflict resolution approach
    pub conflict_resolution: ConflictResolutionStrategy,
}
/// Multi-region consensus strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConsensusStrategy {
    /// Standard Raft across all regions
    GlobalRaft,
    /// Regional Raft with cross-region coordination
    RegionalRaft {
        /// Primary consensus region
        primary_region: String,
        /// Backup regions for failover
        backup_regions: Vec<String>,
    },
    /// Byzantine fault tolerant consensus for high security
    ByzantineConsensus {
        /// Required byzantine quorum size
        byzantine_quorum: usize,
    },
    /// Hybrid approach with regional leaders
    HybridConsensus {
        /// Regional leader preferences
        region_preferences: HashMap<String, f64>,
    },
}
/// Vector clock comparison result
#[derive(Debug, Clone, PartialEq)]
pub enum VectorClockOrdering {
    Less,
    Greater,
    Equal,
    Concurrent,
}
/// Geographic region configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Region {
    /// Region identifier (e.g., "us-east-1", "eu-west-1")
    pub id: String,
    /// Human-readable region name
    pub name: String,
    /// Geographic coordinates for latency calculations
    pub coordinates: Option<GeoCoordinates>,
    /// Availability zones within this region
    pub availability_zones: Vec<AvailabilityZone>,
    /// Region-specific configuration
    pub config: RegionConfig,
}
/// Node placement information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NodePlacement {
    /// Node identifier
    pub node_id: OxirsNodeId,
    /// Region where the node is located
    pub region_id: String,
    /// Availability zone where the node is located
    pub availability_zone_id: String,
    /// Data center or specific location within AZ
    pub data_center: Option<String>,
    /// Rack identifier for fine-grained placement
    pub rack: Option<String>,
}
/// Intra-region replication strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IntraRegionStrategy {
    /// Synchronous replication within region
    Synchronous { min_replicas: usize },
    /// Asynchronous replication with batching
    Asynchronous {
        batch_size: usize,
        batch_timeout_ms: u64,
    },
    /// Quorum-based replication
    Quorum { quorum_size: usize },
}
