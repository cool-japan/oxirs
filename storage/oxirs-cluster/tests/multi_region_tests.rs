//! Multi-Region Integration Tests for OxiRS Cluster
//!
//! Simplified tests for multi-region functionality without complex setup.
//!
//! Run with: cargo test --test multi_region_tests -- --nocapture

use oxirs_cluster::conflict_resolution::ResolutionStrategy;
use oxirs_cluster::region_manager::{AvailabilityZone, Region, RegionConfig};
use oxirs_cluster::region_manager::{
    ConflictResolutionStrategy as MRConflictStrategy, ConsensusStrategy, CrossRegionStrategy,
    IntraRegionStrategy, MultiRegionReplicationStrategy,
};
use oxirs_cluster::{ClusterNode, MultiRegionConfig, NodeConfig};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

/// Create a simple region configuration
fn create_test_region(region_id: &str) -> Region {
    Region {
        id: region_id.to_string(),
        name: format!("Test Region {}", region_id),
        coordinates: None,
        availability_zones: vec![
            AvailabilityZone {
                id: format!("{}-a", region_id),
                name: format!("{} Zone A", region_id),
                region_id: region_id.to_string(),
            },
            AvailabilityZone {
                id: format!("{}-b", region_id),
                name: format!("{} Zone B", region_id),
                region_id: region_id.to_string(),
            },
        ],
        config: RegionConfig::default(),
    }
}

/// Create a node with multi-region configuration
async fn create_multi_region_node(
    node_id: u64,
    port: u16,
    region_id: &str,
    az_id: &str,
    regions: Vec<Region>,
) -> anyhow::Result<ClusterNode> {
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), port);
    let data_dir =
        std::env::temp_dir().join(format!("oxirs-mr-test-{}-node-{}", region_id, node_id));

    // Clean up old data
    if data_dir.exists() {
        let _ = std::fs::remove_dir_all(&data_dir);
    }

    let mut node_config = NodeConfig::new(node_id, addr);
    node_config.data_dir = data_dir.to_string_lossy().to_string();

    // Configure multi-region settings
    let region_config = MultiRegionConfig {
        region_id: region_id.to_string(),
        availability_zone_id: az_id.to_string(),
        data_center: Some(format!("dc-{}", region_id)),
        rack: Some("rack-1".to_string()),
        regions,
        consensus_strategy: ConsensusStrategy::GlobalRaft,
        replication_strategy: MultiRegionReplicationStrategy {
            intra_region: IntraRegionStrategy::Quorum { quorum_size: 2 },
            cross_region: CrossRegionStrategy::AsyncAll,
            conflict_resolution: MRConflictStrategy::LastWriterWins,
        },
        conflict_resolution_strategy: ResolutionStrategy::LastWriterWins,
        edge_config: None,
        enable_monitoring: true,
    };

    node_config.region_config = Some(region_config);

    ClusterNode::new(node_config).await.map_err(Into::into)
}

/// Test 1: Multi-region configuration
#[tokio::test]
async fn test_multi_region_configuration() -> anyhow::Result<()> {
    println!("🌍 Testing multi-region configuration");

    // Create regions
    let regions = vec![
        create_test_region("us-east-1"),
        create_test_region("eu-west-1"),
    ];

    // Create node with multi-region config
    let node =
        create_multi_region_node(1, 10000, "us-east-1", "us-east-1-a", regions.clone()).await?;

    // Verify multi-region is enabled
    assert!(node.is_multi_region_enabled());
    println!("   ✓ Multi-region is enabled");

    // Verify region ID
    assert_eq!(node.get_region_id(), Some("us-east-1".to_string()));
    println!("   ✓ Region ID: us-east-1");

    // Verify AZ ID
    assert_eq!(
        node.get_availability_zone_id(),
        Some("us-east-1-a".to_string())
    );
    println!("   ✓ Availability Zone: us-east-1-a");

    Ok(())
}

/// Test 2: Region manager initialization
#[tokio::test]
async fn test_region_manager_initialization() -> anyhow::Result<()> {
    println!("🌍 Testing region manager initialization");

    let regions = vec![
        create_test_region("us-east-1"),
        create_test_region("eu-west-1"),
        create_test_region("ap-south-1"),
    ];

    let node =
        create_multi_region_node(2, 10001, "us-east-1", "us-east-1-a", regions.clone()).await?;

    // Verify region manager exists
    assert!(node.region_manager().is_some());
    println!("   ✓ Region manager initialized");

    // Get topology
    let topology = node.get_region_topology().await?;
    assert_eq!(topology.regions.len(), 3);
    println!("   ✓ Topology has 3 regions");

    for region in topology.regions.values() {
        println!("     - {} ({})", region.id, region.name);
        assert!(!region.availability_zones.is_empty());
    }

    Ok(())
}

/// Test 3: Region status reporting
#[tokio::test]
async fn test_region_status_reporting() -> anyhow::Result<()> {
    println!("🌍 Testing region status reporting");

    let regions = vec![create_test_region("us-west-2")];

    let mut node =
        create_multi_region_node(3, 10002, "us-west-2", "us-west-2-a", regions.clone()).await?;

    node.start().await?;

    // Get cluster status
    let status = node.get_status().await;

    // Verify region status is included
    assert!(status.region_status.is_some());

    if let Some(region_status) = status.region_status {
        println!("   Region ID: {}", region_status.region_id);
        println!("   AZ ID: {}", region_status.availability_zone_id);
        println!("   Total Regions: {}", region_status.total_regions);
        println!("   Monitoring Active: {}", region_status.monitoring_active);

        assert_eq!(region_status.region_id, "us-west-2");
        assert_eq!(region_status.availability_zone_id, "us-west-2-a");
        assert_eq!(region_status.total_regions, 1);
        assert!(region_status.monitoring_active);

        println!("   ✓ Region status correct");
    }

    node.stop().await?;

    Ok(())
}

/// Test 4: Multiple regions with different AZs
#[tokio::test]
async fn test_multiple_availability_zones() -> anyhow::Result<()> {
    println!("🌍 Testing multiple availability zones");

    let regions = vec![create_test_region("eu-central-1")];

    // Create nodes in different AZs
    let node_a =
        create_multi_region_node(10, 10010, "eu-central-1", "eu-central-1-a", regions.clone())
            .await?;

    let node_b =
        create_multi_region_node(11, 10011, "eu-central-1", "eu-central-1-b", regions.clone())
            .await?;

    // Verify different AZs
    assert_eq!(
        node_a.get_availability_zone_id(),
        Some("eu-central-1-a".to_string())
    );
    assert_eq!(
        node_b.get_availability_zone_id(),
        Some("eu-central-1-b".to_string())
    );

    // But same region
    assert_eq!(node_a.get_region_id(), node_b.get_region_id());

    println!("   ✓ Node A in eu-central-1-a");
    println!("   ✓ Node B in eu-central-1-b");
    println!("   ✓ Both in same region: eu-central-1");

    Ok(())
}

/// Test 5: Region topology validation
#[tokio::test]
async fn test_region_topology_validation() -> anyhow::Result<()> {
    println!("🌍 Testing region topology validation");

    let regions = vec![
        create_test_region("us-east-1"),
        create_test_region("us-west-1"),
        create_test_region("eu-west-1"),
    ];

    let node =
        create_multi_region_node(20, 10020, "us-east-1", "us-east-1-a", regions.clone()).await?;

    let topology = node.get_region_topology().await?;

    println!("   Total regions: {}", topology.regions.len());
    assert_eq!(topology.regions.len(), 3);

    // Verify each region has AZs
    for region in topology.regions.values() {
        assert!(
            !region.availability_zones.is_empty(),
            "Region {} should have availability zones",
            region.id
        );

        println!(
            "   Region {}: {} AZs",
            region.id,
            region.availability_zones.len()
        );

        for az in &region.availability_zones {
            assert_eq!(az.region_id, region.id);
        }
    }

    println!("   ✓ All regions have valid topology");

    Ok(())
}

/// Test 6: Region configuration defaults
#[tokio::test]
async fn test_region_configuration_defaults() -> anyhow::Result<()> {
    println!("🌍 Testing region configuration defaults");

    let config = RegionConfig::default();

    println!(
        "   Local replication factor: {}",
        config.local_replication_factor
    );
    println!(
        "   Cross-region replication: {}",
        config.cross_region_replication_factor
    );
    println!("   Max latency: {}ms", config.max_regional_latency_ms);
    println!("   Prefer local leader: {}", config.prefer_local_leader);
    println!(
        "   Cross-region backup: {}",
        config.enable_cross_region_backup
    );

    assert_eq!(config.local_replication_factor, 3);
    assert_eq!(config.cross_region_replication_factor, 1);
    assert_eq!(config.max_regional_latency_ms, 100);
    assert!(config.prefer_local_leader);
    assert!(config.enable_cross_region_backup);

    println!("   ✓ All defaults correct");

    Ok(())
}

/// Test 7: Consensus strategy configuration
#[tokio::test]
async fn test_consensus_strategy() {
    println!("🌍 Testing consensus strategy configuration");

    let _global_raft = ConsensusStrategy::GlobalRaft;
    println!("   Strategy: GlobalRaft");

    let _regional_raft = ConsensusStrategy::RegionalRaft {
        primary_region: "us-east-1".to_string(),
        backup_regions: vec!["us-west-1".to_string()],
    };
    println!("   Strategy: RegionalRaft");

    let _byzantine = ConsensusStrategy::ByzantineConsensus {
        byzantine_quorum: 4,
    };
    println!("   Strategy: ByzantineConsensus");

    println!("   ✓ All consensus strategies available");
}

/// Test 8: Replication strategy configuration
#[tokio::test]
async fn test_replication_strategy() {
    println!("🌍 Testing replication strategy configuration");

    // Intra-region strategies
    let _intra_sync = IntraRegionStrategy::Synchronous { min_replicas: 2 };
    let _intra_async = IntraRegionStrategy::Asynchronous {
        batch_size: 100,
        batch_timeout_ms: 1000,
    };
    let _intra_quorum = IntraRegionStrategy::Quorum { quorum_size: 3 };

    println!("   Intra-region: Synchronous");
    println!("   Intra-region: Asynchronous");
    println!("   Intra-region: Quorum");

    // Cross-region strategies
    let _cross_async = CrossRegionStrategy::AsyncAll;
    let _cross_selective = CrossRegionStrategy::SelectiveSync {
        target_regions: vec!["us-east-1".to_string(), "eu-west-1".to_string()],
    };
    let _cross_eventual = CrossRegionStrategy::EventualConsistency {
        reconciliation_interval_ms: 5000,
    };

    println!("   Cross-region: AsyncAll");
    println!("   Cross-region: SelectiveSync");
    println!("   Cross-region: EventualConsistency");

    println!("   ✓ All replication strategies available");
}

/// Test 9: Conflict resolution strategies
#[tokio::test]
async fn test_conflict_resolution_strategies() {
    println!("🌍 Testing conflict resolution strategies");

    let lww = MRConflictStrategy::LastWriterWins;
    let vc = MRConflictStrategy::VectorClock;
    let custom = MRConflictStrategy::Custom {
        resolution_function: "custom_resolver".to_string(),
    };
    let manual = MRConflictStrategy::Manual;

    println!("   LastWriterWins: {:?}", lww);
    println!("   VectorClock: {:?}", vc);
    println!("   Custom: {:?}", custom);
    println!("   Manual: {:?}", manual);

    println!("   ✓ All conflict resolution strategies available");
}

/// Test 10: Multi-region node lifecycle
#[tokio::test]
async fn test_multi_region_node_lifecycle() -> anyhow::Result<()> {
    println!("🌍 Testing multi-region node lifecycle");

    let regions = vec![create_test_region("ap-northeast-1")];

    let mut node =
        create_multi_region_node(30, 10030, "ap-northeast-1", "ap-northeast-1-a", regions).await?;

    println!("   ✓ Node created");

    // Start node
    node.start().await?;
    println!("   ✓ Node started");

    // Verify multi-region is active
    assert!(node.is_multi_region_enabled());

    // Get status
    let status = node.get_status().await;
    assert!(status.region_status.is_some());
    println!("   ✓ Multi-region status available");

    // Stop node
    node.stop().await?;
    println!("   ✓ Node stopped");

    Ok(())
}
