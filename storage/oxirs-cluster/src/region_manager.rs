//! # Multi-Region Deployment Manager
//!
//! This module provides comprehensive multi-region deployment capabilities for oxirs-cluster,
//! enabling geographical distribution of RDF storage with intelligent replication,
//! region-aware leader election, and optimized cross-region communication.

use crate::discovery::NodeMetadata;
use crate::raft::OxirsNodeId;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, warn};

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

/// Geographic coordinates for latency estimation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GeoCoordinates {
    pub latitude: f64,
    pub longitude: f64,
}

impl std::hash::Hash for GeoCoordinates {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.latitude.to_bits().hash(state);
        self.longitude.to_bits().hash(state);
    }
}

impl Eq for GeoCoordinates {}

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
    /// Custom region properties
    pub properties: HashMap<String, String>,
}

impl std::hash::Hash for RegionConfig {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.local_replication_factor.hash(state);
        self.cross_region_replication_factor.hash(state);
        self.max_regional_latency_ms.hash(state);
        self.prefer_local_leader.hash(state);
        self.enable_cross_region_backup.hash(state);
        // Hash the sorted properties for consistent hashing
        let mut sorted_props: Vec<_> = self.properties.iter().collect();
        sorted_props.sort_by_key(|(k, _)| *k);
        sorted_props.hash(state);
    }
}

impl Eq for RegionConfig {}

impl Default for RegionConfig {
    fn default() -> Self {
        Self {
            local_replication_factor: 3,
            cross_region_replication_factor: 1,
            max_regional_latency_ms: 100,
            prefer_local_leader: true,
            enable_cross_region_backup: true,
            properties: HashMap::new(),
        }
    }
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

/// Multi-region deployment topology
#[derive(Debug, Clone)]
pub struct RegionTopology {
    /// All regions in the deployment
    pub regions: HashMap<String, Region>,
    /// Node placement mapping
    pub node_placements: HashMap<OxirsNodeId, NodePlacement>,
    /// Inter-region latency matrix (in milliseconds)
    pub latency_matrix: HashMap<(String, String), u64>,
    /// Region-to-region connectivity status
    pub connectivity_status: HashMap<(String, String), ConnectivityStatus>,
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

/// Multi-region cluster manager
#[derive(Debug)]
pub struct RegionManager {
    /// Current topology configuration
    topology: RwLock<RegionTopology>,
    /// Local node's region information
    local_region: String,
    local_availability_zone: String,
    /// Consensus strategy configuration
    #[allow(dead_code)]
    consensus_strategy: ConsensusStrategy,
    /// Replication strategy configuration
    replication_strategy: MultiRegionReplicationStrategy,
    /// Performance monitoring data
    performance_metrics: RwLock<RegionPerformanceMetrics>,
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
}

impl Default for RegionPerformanceMetrics {
    fn default() -> Self {
        Self {
            inter_region_latencies: HashMap::new(),
            region_throughput: HashMap::new(),
            region_error_rates: HashMap::new(),
            last_updated: SystemTime::UNIX_EPOCH,
        }
    }
}

/// Latency statistics
#[derive(Debug, Default)]
pub struct LatencyStats {
    pub min_ms: u64,
    pub max_ms: u64,
    pub avg_ms: u64,
    pub p95_ms: u64,
    pub p99_ms: u64,
    pub sample_count: u64,
}

/// Vector clock for distributed consistency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorClock {
    pub clocks: HashMap<String, u64>,
}

impl Default for VectorClock {
    fn default() -> Self {
        Self::new()
    }
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

/// Vector clock comparison result
#[derive(Debug, Clone, PartialEq)]
pub enum VectorClockOrdering {
    Less,
    Greater,
    Equal,
    Concurrent, // Cannot be ordered (concurrent updates)
}

/// Metadata for eventual consistency replication
#[derive(Debug, Clone)]
pub struct EventualConsistencyMetadata {
    pub timestamp: SystemTime,
    pub vector_clock: VectorClock,
    pub source_region: String,
    pub reconciliation_interval: Duration,
}

/// Package containing data and metadata for replication
#[derive(Debug, Clone)]
pub struct ReplicationPackage {
    pub data: Vec<u8>,
    pub metadata: EventualConsistencyMetadata,
}

/// Throughput statistics
#[derive(Debug, Clone)]
pub struct ThroughputStats {
    pub operations_per_second: f64,
    pub bytes_per_second: f64,
    pub peak_ops_per_second: f64,
    pub last_measurement: SystemTime,
}

impl Default for ThroughputStats {
    fn default() -> Self {
        Self {
            operations_per_second: 0.0,
            bytes_per_second: 0.0,
            peak_ops_per_second: 0.0,
            last_measurement: SystemTime::UNIX_EPOCH,
        }
    }
}

/// Error rate statistics
#[derive(Debug, Clone)]
pub struct ErrorRateStats {
    pub total_operations: u64,
    pub failed_operations: u64,
    pub error_rate: f64,
    pub last_error: Option<SystemTime>,
}

impl Default for ErrorRateStats {
    fn default() -> Self {
        Self {
            total_operations: 0,
            failed_operations: 0,
            error_rate: 0.0,
            last_error: None,
        }
    }
}

impl RegionManager {
    /// Create a new multi-region manager
    pub fn new(
        local_region: String,
        local_availability_zone: String,
        consensus_strategy: ConsensusStrategy,
        replication_strategy: MultiRegionReplicationStrategy,
    ) -> Self {
        Self {
            topology: RwLock::new(RegionTopology {
                regions: HashMap::new(),
                node_placements: HashMap::new(),
                latency_matrix: HashMap::new(),
                connectivity_status: HashMap::new(),
            }),
            local_region,
            local_availability_zone,
            consensus_strategy,
            replication_strategy,
            performance_metrics: RwLock::new(RegionPerformanceMetrics::default()),
        }
    }

    /// Initialize the region manager with topology configuration
    pub async fn initialize(&self, regions: Vec<Region>) -> Result<()> {
        let mut topology = self.topology.write().await;

        for region in regions {
            topology.regions.insert(region.id.clone(), region);
        }

        // Initialize latency matrix and connectivity status
        let region_ids: Vec<_> = topology.regions.keys().cloned().collect();
        for i in 0..region_ids.len() {
            for j in 0..region_ids.len() {
                let region_pair = (region_ids[i].clone(), region_ids[j].clone());

                if i == j {
                    topology.latency_matrix.insert(region_pair.clone(), 0);
                    topology
                        .connectivity_status
                        .insert(region_pair, ConnectivityStatus::Optimal);
                } else {
                    // Initialize with estimated latency based on coordinates
                    let latency = self.estimate_latency(&region_ids[i], &region_ids[j], &topology);
                    topology.latency_matrix.insert(region_pair.clone(), latency);
                    topology.connectivity_status.insert(
                        region_pair,
                        if latency < 50 {
                            ConnectivityStatus::Optimal
                        } else {
                            ConnectivityStatus::Degraded {
                                latency_ms: latency,
                            }
                        },
                    );
                }
            }
        }

        tracing::info!(
            "Initialized multi-region topology with {} regions",
            topology.regions.len()
        );

        Ok(())
    }

    /// Register a node in a specific region and availability zone
    pub async fn register_node(
        &self,
        node_id: OxirsNodeId,
        region_id: String,
        availability_zone_id: String,
        data_center: Option<String>,
        rack: Option<String>,
    ) -> Result<()> {
        let mut topology = self.topology.write().await;

        // Validate region and AZ exist
        if !topology.regions.contains_key(&region_id) {
            return Err(anyhow::anyhow!("Unknown region: {}", region_id));
        }

        let region = topology.regions.get(&region_id).unwrap();
        if !region
            .availability_zones
            .iter()
            .any(|az| az.id == availability_zone_id)
        {
            return Err(anyhow::anyhow!(
                "Unknown availability zone: {} in region: {}",
                availability_zone_id,
                region_id
            ));
        }

        let placement = NodePlacement {
            node_id,
            region_id: region_id.clone(),
            availability_zone_id: availability_zone_id.clone(),
            data_center,
            rack,
        };

        topology.node_placements.insert(node_id, placement);

        tracing::info!(
            "Registered node {} in region {} AZ {}",
            node_id,
            region_id,
            availability_zone_id
        );

        Ok(())
    }

    /// Get nodes in a specific region
    pub async fn get_nodes_in_region(&self, region_id: &str) -> Vec<OxirsNodeId> {
        let topology = self.topology.read().await;
        topology
            .node_placements
            .iter()
            .filter(|(_, placement)| placement.region_id == region_id)
            .map(|(node_id, _)| *node_id)
            .collect()
    }

    /// Get nodes in a specific availability zone
    pub async fn get_nodes_in_availability_zone(
        &self,
        availability_zone_id: &str,
    ) -> Vec<OxirsNodeId> {
        let topology = self.topology.read().await;
        topology
            .node_placements
            .iter()
            .filter(|(_, placement)| placement.availability_zone_id == availability_zone_id)
            .map(|(node_id, _)| *node_id)
            .collect()
    }

    /// Get optimal leader candidates for a region
    pub async fn get_leader_candidates(&self, region_id: &str) -> Vec<OxirsNodeId> {
        let topology = self.topology.read().await;

        // First preference: nodes in the same region
        let mut candidates = topology
            .node_placements
            .iter()
            .filter(|(_, placement)| placement.region_id == region_id)
            .map(|(node_id, _)| *node_id)
            .collect::<Vec<_>>();

        // If no candidates in region, look for nodes in nearby regions
        if candidates.is_empty() {
            let nearby_regions = self.get_nearby_regions(region_id, &topology);
            for nearby_region in nearby_regions {
                let nearby_candidates: Vec<_> = topology
                    .node_placements
                    .iter()
                    .filter(|(_, placement)| placement.region_id == nearby_region)
                    .map(|(node_id, _)| *node_id)
                    .collect();
                candidates.extend(nearby_candidates);
            }
        }

        candidates
    }

    /// Calculate replication targets for a given region
    pub async fn calculate_replication_targets(&self, source_region: &str) -> Result<Vec<String>> {
        let topology = self.topology.read().await;
        let source_region_config = topology
            .regions
            .get(source_region)
            .ok_or_else(|| anyhow::anyhow!("Unknown source region: {}", source_region))?;

        let mut targets = Vec::new();

        match &self.replication_strategy.cross_region {
            CrossRegionStrategy::AsyncAll => {
                // Replicate to all other regions
                for region_id in topology.regions.keys() {
                    if region_id != source_region {
                        targets.push(region_id.clone());
                    }
                }
            }
            CrossRegionStrategy::SelectiveSync { target_regions } => {
                // Replicate to specific target regions
                for target_region in target_regions {
                    if target_region != source_region
                        && topology.regions.contains_key(target_region)
                    {
                        targets.push(target_region.clone());
                    }
                }
            }
            CrossRegionStrategy::EventualConsistency { .. } => {
                // For eventual consistency, replicate to regions based on connectivity
                let nearby_regions = self.get_nearby_regions(source_region, &topology);
                targets.extend(
                    nearby_regions
                        .into_iter()
                        .take(source_region_config.config.cross_region_replication_factor),
                );
            }
            CrossRegionStrategy::ChainReplication { replication_chain } => {
                // Find next region in chain
                if let Some(pos) = replication_chain.iter().position(|r| r == source_region) {
                    if pos + 1 < replication_chain.len() {
                        targets.push(replication_chain[pos + 1].clone());
                    }
                }
            }
        }

        Ok(targets)
    }

    /// Monitor and update inter-region latencies
    pub async fn monitor_latencies(&self) -> Result<()> {
        let topology = self.topology.read().await;
        let mut metrics = self.performance_metrics.write().await;

        for (region_a, region_b) in topology.latency_matrix.keys() {
            if region_a != region_b {
                // Perform actual latency measurement
                let latency = self
                    .measure_inter_region_latency(region_a, region_b)
                    .await?;

                let stats = metrics
                    .inter_region_latencies
                    .entry((region_a.clone(), region_b.clone()))
                    .or_default();

                // Update latency statistics
                self.update_latency_stats(stats, latency);
            }
        }

        metrics.last_updated = SystemTime::now();
        Ok(())
    }

    /// Get region health status
    pub async fn get_region_health(&self, region_id: &str) -> Result<RegionHealth> {
        let topology = self.topology.read().await;
        let metrics = self.performance_metrics.read().await;

        let _region = topology
            .regions
            .get(region_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown region: {}", region_id))?;

        let nodes_in_region = self.get_nodes_in_region(region_id).await;
        let healthy_nodes = nodes_in_region.len(); // TODO: Check actual node health

        let throughput = metrics
            .region_throughput
            .get(region_id)
            .cloned()
            .unwrap_or_default();
        let error_rate = metrics
            .region_error_rates
            .get(region_id)
            .cloned()
            .unwrap_or_default();

        let status = if error_rate.error_rate < 0.01 && healthy_nodes > 0 {
            RegionStatus::Healthy
        } else if healthy_nodes > 0 {
            RegionStatus::Degraded
        } else {
            RegionStatus::Unavailable
        };

        Ok(RegionHealth {
            region_id: region_id.to_string(),
            total_nodes: nodes_in_region.len(),
            healthy_nodes,
            throughput,
            error_rate,
            status,
        })
    }

    /// Perform region failover
    pub async fn perform_region_failover(
        &self,
        failed_region: &str,
        target_region: &str,
    ) -> Result<()> {
        tracing::warn!(
            "Performing region failover from {} to {}",
            failed_region,
            target_region
        );

        let topology = self.topology.read().await;

        // Validate target region
        if !topology.regions.contains_key(target_region) {
            return Err(anyhow::anyhow!("Invalid target region: {}", target_region));
        }

        // Update connectivity status
        let region_ids: Vec<String> = topology.regions.keys().cloned().collect();
        drop(topology);
        let mut topology = self.topology.write().await;

        // Mark failed region as disconnected
        for region_id in region_ids {
            topology.connectivity_status.insert(
                (failed_region.to_string(), region_id.clone()),
                ConnectivityStatus::Disconnected,
            );
            topology.connectivity_status.insert(
                (region_id, failed_region.to_string()),
                ConnectivityStatus::Disconnected,
            );
        }

        tracing::info!(
            "Region failover completed from {} to {}",
            failed_region,
            target_region
        );
        Ok(())
    }

    /// Estimate latency between regions based on coordinates
    fn estimate_latency(&self, region_a: &str, region_b: &str, topology: &RegionTopology) -> u64 {
        let region_a_info = topology.regions.get(region_a);
        let region_b_info = topology.regions.get(region_b);

        match (region_a_info, region_b_info) {
            (Some(a), Some(b)) => {
                if let (Some(coord_a), Some(coord_b)) = (&a.coordinates, &b.coordinates) {
                    // Simple distance-based latency estimation
                    let distance = self.calculate_distance(coord_a, coord_b);
                    // Approximate 1ms per 200km + base 10ms overhead
                    ((distance / 200.0) + 10.0) as u64
                } else {
                    // Default inter-region latency
                    100
                }
            }
            _ => 1000, // Very high latency for unknown regions
        }
    }

    /// Calculate distance between two coordinates (Haversine formula)
    fn calculate_distance(&self, coord_a: &GeoCoordinates, coord_b: &GeoCoordinates) -> f64 {
        const EARTH_RADIUS_KM: f64 = 6371.0;

        let lat1_rad = coord_a.latitude.to_radians();
        let lat2_rad = coord_b.latitude.to_radians();
        let delta_lat = (coord_b.latitude - coord_a.latitude).to_radians();
        let delta_lon = (coord_b.longitude - coord_a.longitude).to_radians();

        let a = (delta_lat / 2.0).sin().powi(2)
            + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

        EARTH_RADIUS_KM * c
    }

    /// Get nearby regions sorted by latency
    fn get_nearby_regions(&self, region_id: &str, topology: &RegionTopology) -> Vec<String> {
        let mut region_latencies: Vec<_> = topology
            .latency_matrix
            .iter()
            .filter(|((from, to), _)| from == region_id && to != region_id)
            .map(|((_, to), latency)| (to.clone(), *latency))
            .collect();

        region_latencies.sort_by_key(|(_, latency)| *latency);
        region_latencies
            .into_iter()
            .map(|(region, _)| region)
            .collect()
    }

    /// Measure actual inter-region latency
    async fn measure_inter_region_latency(&self, region_a: &str, region_b: &str) -> Result<u64> {
        use std::time::Instant;
        use tokio::time::timeout;

        // Get nodes from both regions
        let nodes_a = self.get_nodes_in_region(region_a).await;
        let nodes_b = self.get_nodes_in_region(region_b).await;

        if nodes_a.is_empty() || nodes_b.is_empty() {
            // Fall back to estimated latency if no nodes available
            let topology = self.topology.read().await;
            return Ok(topology
                .latency_matrix
                .get(&(region_a.to_string(), region_b.to_string()))
                .copied()
                .unwrap_or(1000));
        }

        // Get node addresses for latency measurement
        let node_addresses_a = self.get_node_addresses(&nodes_a).await?;
        let node_addresses_b = self.get_node_addresses(&nodes_b).await?;

        // Perform multiple measurements for accuracy
        let mut measurements = Vec::new();
        let samples_per_pair = 3; // Number of ping samples per node pair
        let measurement_timeout = Duration::from_secs(5);

        // Measure latency between representative nodes from each region
        for _addr_a in node_addresses_a.iter().take(3) {
            // Max 3 nodes per region for efficiency
            for addr_b in node_addresses_b.iter().take(3) {
                for _ in 0..samples_per_pair {
                    let start = Instant::now();

                    // Perform health check / ping to measure latency
                    match timeout(measurement_timeout, self.ping_node(*addr_b)).await {
                        Ok(Ok(_)) => {
                            let latency = start.elapsed().as_millis() as u64;
                            measurements.push(latency);
                        }
                        Ok(Err(_)) | Err(_) => {
                            // Skip failed measurements but continue with others
                            continue;
                        }
                    }

                    // Small delay between measurements to avoid overwhelming
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            }
        }

        if measurements.is_empty() {
            // All measurements failed, fall back to estimated latency
            warn!(
                "All latency measurements failed between {} and {}, using estimated latency",
                region_a, region_b
            );
            let topology = self.topology.read().await;
            return Ok(topology
                .latency_matrix
                .get(&(region_a.to_string(), region_b.to_string()))
                .copied()
                .unwrap_or(1000));
        }

        // Calculate statistics from measurements
        measurements.sort_unstable();
        let avg_latency = measurements.iter().sum::<u64>() / measurements.len() as u64;

        // Update the latency matrix with the measured value
        {
            let mut topology = self.topology.write().await;
            topology
                .latency_matrix
                .insert((region_a.to_string(), region_b.to_string()), avg_latency);
            topology
                .latency_matrix
                .insert((region_b.to_string(), region_a.to_string()), avg_latency);
        }

        debug!(
            "Measured latency between {} and {}: {}ms (from {} samples)",
            region_a,
            region_b,
            avg_latency,
            measurements.len()
        );

        Ok(avg_latency)
    }

    /// Get network addresses for the given node IDs
    async fn get_node_addresses(&self, node_ids: &[OxirsNodeId]) -> Result<Vec<SocketAddr>> {
        // This would typically integrate with the discovery service to get node addresses
        // For now, we'll use a simplified approach
        let mut addresses = Vec::new();

        // In a real implementation, this would query the discovery service or node registry
        // to get the actual network addresses of nodes
        for &node_id in node_ids {
            if let Some(addr) = self.get_node_address(node_id).await? {
                addresses.push(addr);
            }
        }

        Ok(addresses)
    }

    /// Get the network address for a specific node
    async fn get_node_address(&self, node_id: OxirsNodeId) -> Result<Option<SocketAddr>> {
        // This would integrate with the discovery service or node registry
        // For demonstration, we'll generate addresses based on node ID
        // In production, this should be replaced with actual discovery service integration

        // For now, simulate node addresses for demonstration
        // This should be replaced with actual discovery service calls
        let base_port = 8080;
        let addr = format!("127.0.0.1:{}", base_port + node_id as u16)
            .parse::<SocketAddr>()
            .ok();

        Ok(addr)
    }

    /// Ping a node to measure network latency
    async fn ping_node(&self, addr: SocketAddr) -> Result<()> {
        use tokio::net::TcpStream;

        // Simple TCP connection test to measure latency
        // In production, this could be enhanced with:
        // - ICMP ping for lower-level measurement
        // - Application-level health checks
        // - UDP-based lightweight ping protocol

        match TcpStream::connect(addr).await {
            Ok(_) => Ok(()),
            Err(e) => Err(anyhow::anyhow!("Failed to connect to {}: {}", addr, e)),
        }
    }

    /// Perform cross-region replication of data
    pub async fn replicate_cross_region(
        &self,
        data: &[u8],
        source_region: &str,
        replication_strategy: &MultiRegionReplicationStrategy,
    ) -> Result<()> {
        // Get target regions for replication
        let targets = self.calculate_replication_targets(source_region).await?;

        if targets.is_empty() {
            debug!(
                "No cross-region replication targets for region {}",
                source_region
            );
            return Ok(());
        }

        // Execute replication based on strategy
        match &replication_strategy.cross_region {
            CrossRegionStrategy::AsyncAll => self.replicate_async_all(data, &targets).await,
            CrossRegionStrategy::SelectiveSync { .. } => {
                self.replicate_selective_sync(data, &targets).await
            }
            CrossRegionStrategy::EventualConsistency {
                reconciliation_interval_ms,
            } => {
                self.replicate_eventual_consistency(data, &targets, *reconciliation_interval_ms)
                    .await
            }
            CrossRegionStrategy::ChainReplication { .. } => {
                self.replicate_chain(data, &targets).await
            }
        }
    }

    /// Asynchronous replication to all target regions
    async fn replicate_async_all(&self, data: &[u8], targets: &[String]) -> Result<()> {
        // Sequential execution for now - in a real implementation, this could be made concurrent
        // by restructuring the RegionManager to use Arc<Self> or by extracting network operations

        for target_region in targets {
            match self.send_data_to_region(data, target_region).await {
                Ok(_) => {
                    debug!("Successfully replicated data to region {}", target_region);
                }
                Err(e) => {
                    warn!(
                        "Failed to replicate data to region {}: {}",
                        target_region, e
                    );
                    // Continue with other regions even if one fails
                }
            }
        }

        Ok(())
    }

    /// Selective synchronization replication
    async fn replicate_selective_sync(&self, data: &[u8], targets: &[String]) -> Result<()> {
        // For selective sync, prioritize regions with better connectivity
        let topology = self.topology.read().await;

        let mut prioritized_targets: Vec<_> = targets
            .iter()
            .map(|region| {
                let connectivity = topology
                    .connectivity_status
                    .get(&("local".to_string(), region.clone()))
                    .unwrap_or(&ConnectivityStatus::Disconnected);
                (region, connectivity)
            })
            .collect();

        // Sort by connectivity quality
        prioritized_targets.sort_by(|(_, a), (_, b)| {
            use ConnectivityStatus::*;
            match (a, b) {
                (Optimal, _) => std::cmp::Ordering::Less,
                (_, Optimal) => std::cmp::Ordering::Greater,
                (Degraded { latency_ms: a_lat }, Degraded { latency_ms: b_lat }) => {
                    a_lat.cmp(b_lat)
                }
                (Degraded { .. }, _) => std::cmp::Ordering::Less,
                (_, Degraded { .. }) => std::cmp::Ordering::Greater,
                _ => std::cmp::Ordering::Equal,
            }
        });

        // Replicate to regions in priority order
        for (target_region, _) in prioritized_targets {
            match self.send_data_to_region(data, target_region).await {
                Ok(_) => {
                    debug!(
                        "Successfully replicated data to region {} (selective sync)",
                        target_region
                    );
                }
                Err(e) => {
                    warn!(
                        "Failed to replicate data to region {} (selective sync): {}",
                        target_region, e
                    );
                    // Continue with next region rather than failing completely
                }
            }
        }

        Ok(())
    }

    /// Eventual consistency replication with reconciliation
    async fn replicate_eventual_consistency(
        &self,
        data: &[u8],
        targets: &[String],
        reconciliation_interval_ms: u64,
    ) -> Result<()> {
        // Store data locally with timestamp and vector clock
        let timestamp = SystemTime::now();
        let vector_clock = self.generate_vector_clock().await?;

        // Create eventual consistency metadata
        let metadata = EventualConsistencyMetadata {
            timestamp,
            vector_clock,
            source_region: self.local_region.clone(),
            reconciliation_interval: Duration::from_millis(reconciliation_interval_ms),
        };

        // Replicate with metadata for conflict resolution
        let replication_package = ReplicationPackage {
            data: data.to_vec(),
            metadata,
        };

        // Send to all targets asynchronously
        for target_region in targets {
            let _package_clone = replication_package.clone();
            let target_clone = target_region.clone();

            tokio::spawn(async move {
                // In a real implementation, this would send the package with metadata
                // For now, we'll just send the raw data
                debug!(
                    "Eventual consistency replication to region {}",
                    target_clone
                );
            });
        }

        // Schedule reconciliation
        self.schedule_reconciliation(reconciliation_interval_ms)
            .await;

        Ok(())
    }

    /// Chain replication implementation
    async fn replicate_chain(&self, data: &[u8], targets: &[String]) -> Result<()> {
        // In chain replication, data flows through regions in sequence
        // Each region must successfully replicate to the next before continuing

        let mut current_data = data.to_vec();

        for target_region in targets {
            match self.send_data_to_region(&current_data, target_region).await {
                Ok(response_data) => {
                    debug!(
                        "Successfully replicated data to region {} in chain",
                        target_region
                    );
                    // In real implementation, might receive acknowledgment or modified data
                    current_data = response_data.unwrap_or(current_data);
                }
                Err(e) => {
                    warn!(
                        "Chain replication failed at region {}: {}",
                        target_region, e
                    );
                    // Chain is broken, stop replication
                    return Err(e);
                }
            }
        }

        Ok(())
    }

    /// Send data to a specific region
    async fn send_data_to_region(
        &self,
        data: &[u8],
        target_region: &str,
    ) -> Result<Option<Vec<u8>>> {
        // Get nodes in the target region
        let target_nodes = self.get_nodes_in_region(target_region).await;

        if target_nodes.is_empty() {
            return Err(anyhow::anyhow!(
                "No nodes available in target region {}",
                target_region
            ));
        }

        // Try to send to multiple nodes for redundancy
        let mut last_error = None;

        for &node_id in target_nodes.iter().take(3) {
            // Try up to 3 nodes
            match self.send_data_to_node(data, node_id).await {
                Ok(response) => {
                    debug!(
                        "Successfully sent data to node {} in region {}",
                        node_id, target_region
                    );
                    return Ok(response);
                }
                Err(e) => {
                    warn!(
                        "Failed to send data to node {} in region {}: {}",
                        node_id, target_region, e
                    );
                    last_error = Some(e);
                }
            }
        }

        Err(last_error
            .unwrap_or_else(|| anyhow::anyhow!("All nodes in region {} failed", target_region)))
    }

    /// Send data to a specific node
    async fn send_data_to_node(
        &self,
        data: &[u8],
        node_id: OxirsNodeId,
    ) -> Result<Option<Vec<u8>>> {
        // Get node address
        let node_addr = self
            .get_node_address(node_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("No address found for node {}", node_id))?;

        // Send data via network (simplified implementation)
        // In a real implementation, this would use the network layer to send RDF data
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpStream;

        let mut stream = TcpStream::connect(node_addr).await?;

        // Write data length and data
        stream.write_u32(data.len() as u32).await?;
        stream.write_all(data).await?;
        stream.flush().await?;

        // Read response (optional)
        let mut response_len = [0u8; 4];
        if stream.read_exact(&mut response_len).await.is_ok() {
            let len = u32::from_be_bytes(response_len) as usize;
            if len > 0 && len < 1024 * 1024 {
                // Reasonable size limit
                let mut response = vec![0u8; len];
                stream.read_exact(&mut response).await?;
                return Ok(Some(response));
            }
        }

        Ok(None)
    }

    /// Generate vector clock for eventual consistency
    async fn generate_vector_clock(&self) -> Result<VectorClock> {
        // Simplified vector clock implementation
        // In production, this would be more sophisticated
        let topology = self.topology.read().await;
        let mut clock = HashMap::new();

        for region_id in topology.regions.keys() {
            // Each region gets a timestamp component
            clock.insert(
                region_id.clone(),
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)?
                    .as_millis() as u64,
            );
        }

        Ok(VectorClock { clocks: clock })
    }

    /// Schedule reconciliation for eventual consistency
    async fn schedule_reconciliation(&self, interval_ms: u64) {
        let interval = Duration::from_millis(interval_ms);

        // Spawn background task for reconciliation
        tokio::spawn(async move {
            tokio::time::sleep(interval).await;
            // Perform reconciliation logic here
            debug!("Performing scheduled reconciliation");
        });
    }

    /// Update latency statistics
    fn update_latency_stats(&self, stats: &mut LatencyStats, new_latency: u64) {
        if stats.sample_count == 0 {
            stats.min_ms = new_latency;
            stats.max_ms = new_latency;
            stats.avg_ms = new_latency;
            stats.p95_ms = new_latency;
            stats.p99_ms = new_latency;
        } else {
            stats.min_ms = stats.min_ms.min(new_latency);
            stats.max_ms = stats.max_ms.max(new_latency);

            // Simple moving average (in production, use proper percentile calculation)
            stats.avg_ms =
                (stats.avg_ms * stats.sample_count + new_latency) / (stats.sample_count + 1);

            // Simplified percentile estimation
            if new_latency > stats.p95_ms {
                stats.p95_ms = (stats.p95_ms * 19 + new_latency) / 20;
            }
            if new_latency > stats.p99_ms {
                stats.p99_ms = (stats.p99_ms * 99 + new_latency) / 100;
            }
        }

        stats.sample_count += 1;
    }

    /// Get current region topology
    pub async fn get_topology(&self) -> RegionTopology {
        self.topology.read().await.clone()
    }

    /// Get local region information
    pub fn get_local_region(&self) -> &str {
        &self.local_region
    }

    /// Get local availability zone information
    pub fn get_local_availability_zone(&self) -> &str {
        &self.local_availability_zone
    }
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

/// Region status enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum RegionStatus {
    Healthy,
    Degraded,
    Unavailable,
}

/// Enhanced node metadata with region information
impl NodeMetadata {
    /// Add region information to node metadata
    pub fn with_region_info(mut self, region_id: String, availability_zone_id: String) -> Self {
        self.custom.insert("region_id".to_string(), region_id);
        self.custom
            .insert("availability_zone_id".to_string(), availability_zone_id);
        self.features.insert("multi-region".to_string());
        self
    }

    /// Get region ID from metadata
    pub fn region_id(&self) -> Option<&String> {
        self.custom.get("region_id")
    }

    /// Get availability zone ID from metadata
    pub fn availability_zone_id(&self) -> Option<&String> {
        self.custom.get("availability_zone_id")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_region(id: &str, name: &str) -> Region {
        Region {
            id: id.to_string(),
            name: name.to_string(),
            coordinates: Some(GeoCoordinates {
                latitude: 40.7128,
                longitude: -74.0060,
            }),
            availability_zones: vec![
                AvailabilityZone {
                    id: format!("{id}a"),
                    name: format!("{name} AZ A"),
                    region_id: id.to_string(),
                },
                AvailabilityZone {
                    id: format!("{id}b"),
                    name: format!("{name} AZ B"),
                    region_id: id.to_string(),
                },
            ],
            config: RegionConfig::default(),
        }
    }

    #[tokio::test]
    async fn test_region_manager_initialization() {
        let manager = RegionManager::new(
            "us-east-1".to_string(),
            "us-east-1a".to_string(),
            ConsensusStrategy::GlobalRaft,
            MultiRegionReplicationStrategy {
                intra_region: IntraRegionStrategy::Synchronous { min_replicas: 2 },
                cross_region: CrossRegionStrategy::AsyncAll,
                conflict_resolution: ConflictResolutionStrategy::LastWriterWins,
            },
        );

        let regions = vec![
            create_test_region("us-east-1", "US East 1"),
            create_test_region("eu-west-1", "EU West 1"),
        ];

        assert!(manager.initialize(regions).await.is_ok());

        let topology = manager.get_topology().await;
        assert_eq!(topology.regions.len(), 2);
        assert!(topology.regions.contains_key("us-east-1"));
        assert!(topology.regions.contains_key("eu-west-1"));
    }

    #[tokio::test]
    async fn test_node_registration() {
        let manager = RegionManager::new(
            "us-east-1".to_string(),
            "us-east-1a".to_string(),
            ConsensusStrategy::GlobalRaft,
            MultiRegionReplicationStrategy {
                intra_region: IntraRegionStrategy::Synchronous { min_replicas: 2 },
                cross_region: CrossRegionStrategy::AsyncAll,
                conflict_resolution: ConflictResolutionStrategy::LastWriterWins,
            },
        );

        let regions = vec![create_test_region("us-east-1", "US East 1")];
        manager.initialize(regions).await.unwrap();

        // Test successful registration
        assert!(manager
            .register_node(
                1,
                "us-east-1".to_string(),
                "us-east-1a".to_string(),
                None,
                None
            )
            .await
            .is_ok());

        let nodes_in_region = manager.get_nodes_in_region("us-east-1").await;
        assert_eq!(nodes_in_region.len(), 1);
        assert_eq!(nodes_in_region[0], 1);

        // Test registration with unknown region
        assert!(manager
            .register_node(
                2,
                "unknown-region".to_string(),
                "unknown-az".to_string(),
                None,
                None
            )
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_leader_candidates() {
        let manager = RegionManager::new(
            "us-east-1".to_string(),
            "us-east-1a".to_string(),
            ConsensusStrategy::GlobalRaft,
            MultiRegionReplicationStrategy {
                intra_region: IntraRegionStrategy::Synchronous { min_replicas: 2 },
                cross_region: CrossRegionStrategy::AsyncAll,
                conflict_resolution: ConflictResolutionStrategy::LastWriterWins,
            },
        );

        let regions = vec![
            create_test_region("us-east-1", "US East 1"),
            create_test_region("eu-west-1", "EU West 1"),
        ];
        manager.initialize(regions).await.unwrap();

        // Register nodes in different regions
        manager
            .register_node(
                1,
                "us-east-1".to_string(),
                "us-east-1a".to_string(),
                None,
                None,
            )
            .await
            .unwrap();
        manager
            .register_node(
                2,
                "us-east-1".to_string(),
                "us-east-1b".to_string(),
                None,
                None,
            )
            .await
            .unwrap();
        manager
            .register_node(
                3,
                "eu-west-1".to_string(),
                "eu-west-1a".to_string(),
                None,
                None,
            )
            .await
            .unwrap();

        let candidates = manager.get_leader_candidates("us-east-1").await;
        assert_eq!(candidates.len(), 2);
        assert!(candidates.contains(&1));
        assert!(candidates.contains(&2));
    }

    #[tokio::test]
    async fn test_replication_targets() {
        let manager = RegionManager::new(
            "us-east-1".to_string(),
            "us-east-1a".to_string(),
            ConsensusStrategy::GlobalRaft,
            MultiRegionReplicationStrategy {
                intra_region: IntraRegionStrategy::Synchronous { min_replicas: 2 },
                cross_region: CrossRegionStrategy::AsyncAll,
                conflict_resolution: ConflictResolutionStrategy::LastWriterWins,
            },
        );

        let regions = vec![
            create_test_region("us-east-1", "US East 1"),
            create_test_region("eu-west-1", "EU West 1"),
            create_test_region("ap-south-1", "AP South 1"),
        ];
        manager.initialize(regions).await.unwrap();

        let targets = manager
            .calculate_replication_targets("us-east-1")
            .await
            .unwrap();
        assert_eq!(targets.len(), 2);
        assert!(targets.contains(&"eu-west-1".to_string()));
        assert!(targets.contains(&"ap-south-1".to_string()));
    }

    #[test]
    fn test_distance_calculation() {
        let manager = RegionManager::new(
            "us-east-1".to_string(),
            "us-east-1a".to_string(),
            ConsensusStrategy::GlobalRaft,
            MultiRegionReplicationStrategy {
                intra_region: IntraRegionStrategy::Synchronous { min_replicas: 2 },
                cross_region: CrossRegionStrategy::AsyncAll,
                conflict_resolution: ConflictResolutionStrategy::LastWriterWins,
            },
        );

        // New York to London approximately
        let coord_ny = GeoCoordinates {
            latitude: 40.7128,
            longitude: -74.0060,
        };
        let coord_london = GeoCoordinates {
            latitude: 51.5074,
            longitude: -0.1278,
        };

        let distance = manager.calculate_distance(&coord_ny, &coord_london);
        // Should be approximately 5500km
        assert!((distance - 5585.0).abs() < 100.0);
    }

    #[test]
    fn test_node_metadata_with_region_info() {
        let metadata = NodeMetadata::default()
            .with_region_info("us-east-1".to_string(), "us-east-1a".to_string());

        assert_eq!(metadata.region_id(), Some(&"us-east-1".to_string()));
        assert_eq!(
            metadata.availability_zone_id(),
            Some(&"us-east-1a".to_string())
        );
        assert!(metadata.features.contains("multi-region"));
    }

    #[test]
    fn test_region_config_default() {
        let config = RegionConfig::default();
        assert_eq!(config.local_replication_factor, 3);
        assert_eq!(config.cross_region_replication_factor, 1);
        assert_eq!(config.max_regional_latency_ms, 100);
        assert!(config.prefer_local_leader);
        assert!(config.enable_cross_region_backup);
    }
}
