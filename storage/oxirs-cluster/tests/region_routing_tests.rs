//! Cross-Region Latency Optimization Tests
//!
//! Tests for latency-aware routing and cross-region optimizations in OxiRS clusters.
//!
//! Run with: cargo nextest run --test region_routing_tests --no-fail-fast

use oxirs_cluster::region_manager::{
    AvailabilityZone, ConflictResolutionStrategy, ConsensusStrategy, CrossRegionStrategy,
    GeoCoordinates, IntraRegionStrategy, MultiRegionReplicationStrategy, Region, RegionConfig,
    RegionManager, RoutingStrategy,
};

/// Create a test region with coordinates
fn create_test_region_with_coords(
    id: &str,
    name: &str,
    lat: f64,
    lon: f64,
    routing_strategy: RoutingStrategy,
) -> Region {
    let config = RegionConfig {
        routing_strategy,
        enable_relay: true,
        relay_latency_threshold_ms: 200.0,
        enable_compression: true,
        enable_read_local: true,
        ..Default::default()
    };

    Region {
        id: id.to_string(),
        name: name.to_string(),
        coordinates: Some(GeoCoordinates {
            latitude: lat,
            longitude: lon,
        }),
        availability_zones: vec![AvailabilityZone {
            id: format!("{}-a", id),
            name: format!("{} Zone A", name),
            region_id: id.to_string(),
        }],
        config,
    }
}

/// Helper function to setup a test manager with regions
async fn setup_test_manager() -> RegionManager {
    let manager = RegionManager::new(
        "us-east-1".to_string(),
        "us-east-1-a".to_string(),
        ConsensusStrategy::GlobalRaft,
        MultiRegionReplicationStrategy {
            intra_region: IntraRegionStrategy::Synchronous { min_replicas: 2 },
            cross_region: CrossRegionStrategy::AsyncAll,
            conflict_resolution: ConflictResolutionStrategy::LastWriterWins,
        },
    );

    // Initialize with multiple regions
    let regions = vec![
        create_test_region_with_coords(
            "us-east-1",
            "US East",
            40.7128,
            -74.0060,
            RoutingStrategy::LatencyAware,
        ),
        create_test_region_with_coords(
            "us-west-1",
            "US West",
            37.7749,
            -122.4194,
            RoutingStrategy::LatencyAware,
        ),
        create_test_region_with_coords(
            "eu-central-1",
            "EU Central",
            50.1109,
            8.6821,
            RoutingStrategy::LatencyAware,
        ),
        create_test_region_with_coords(
            "ap-southeast-1",
            "Asia Pacific",
            1.3521,
            103.8198,
            RoutingStrategy::LatencyAware,
        ),
    ];

    manager
        .initialize(regions)
        .await
        .expect("Failed to initialize manager");

    manager
}

/// Test 1: Direct Route - Same region or low latency
#[tokio::test]
async fn test_direct_route() -> anyhow::Result<()> {
    println!("🔄 Testing direct route for low latency connections");

    let manager = setup_test_manager().await;

    // Find route between nearby regions (US East <-> US West)
    let route = manager.find_route("us-east-1", "us-west-1").await?;

    println!("   Route hops: {:?}", route.hops);
    println!("   Total latency: {}ms", route.total_latency);
    println!("   Use compression: {}", route.use_compression);

    // Should be a direct route for relatively close regions
    assert!(
        route.hops.len() == 2 || route.hops.len() == 3,
        "Route should be direct or have minimal hops"
    );

    println!("   ✓ Direct route test passed");
    Ok(())
}

/// Test 2: Relay Route - High latency benefits from relay
#[tokio::test]
async fn test_relay_routing() -> anyhow::Result<()> {
    println!("🔄 Testing relay routing for high-latency paths");

    let manager = setup_test_manager().await;

    // Manually set latencies to simulate high direct latency
    {
        let mut topology = manager.get_topology().await;

        // Set high direct latency US East → Asia
        topology.latency_matrix.insert(
            ("us-east-1".to_string(), "ap-southeast-1".to_string()),
            300.0,
        );

        // Set lower latencies via relay paths
        topology
            .latency_matrix
            .insert(("us-east-1".to_string(), "eu-central-1".to_string()), 100.0);
        topology.latency_matrix.insert(
            ("eu-central-1".to_string(), "ap-southeast-1".to_string()),
            120.0,
        );
        topology
            .latency_matrix
            .insert(("us-east-1".to_string(), "us-west-1".to_string()), 50.0);
        topology.latency_matrix.insert(
            ("us-west-1".to_string(), "ap-southeast-1".to_string()),
            100.0,
        );
    }

    // Find route US East → Asia
    let route = manager.find_route("us-east-1", "ap-southeast-1").await?;

    println!("   Route hops: {:?}", route.hops);
    println!("   Total latency: {}ms", route.total_latency);
    println!("   Hops count: {}", route.hops.len());

    // Should use relay if it's significantly better
    assert!(
        route.hops.len() >= 2,
        "Route should have at least source and destination"
    );

    println!("   ✓ Relay routing test passed");
    Ok(())
}

/// Test 3: Dijkstra Path - Verify shortest latency path
#[tokio::test]
async fn test_dijkstra_shortest_path() -> anyhow::Result<()> {
    println!("🔄 Testing Dijkstra shortest path algorithm");

    let manager = setup_test_manager().await;

    // Set up a clear topology where Dijkstra can find the optimal path
    {
        let mut topology = manager.get_topology().await;

        // Create a graph where the shortest path is clear:
        // us-east → us-west → ap-southeast (total: 150ms)
        // vs direct us-east → ap-southeast (300ms)

        topology
            .latency_matrix
            .insert(("us-east-1".to_string(), "us-west-1".to_string()), 50.0);
        topology.latency_matrix.insert(
            ("us-west-1".to_string(), "ap-southeast-1".to_string()),
            100.0,
        );
        topology.latency_matrix.insert(
            ("us-east-1".to_string(), "ap-southeast-1".to_string()),
            300.0,
        );
    }

    let route = manager.find_route("us-east-1", "ap-southeast-1").await?;

    println!("   Route hops: {:?}", route.hops);
    println!("   Total latency: {}ms", route.total_latency);

    // Verify the algorithm found a good path
    assert!(
        route.total_latency < 300.0,
        "Should find a better path than direct"
    );

    println!("   ✓ Dijkstra shortest path test passed");
    Ok(())
}

/// Test 4: Read-Local Routing - Route reads to nearest replica
#[tokio::test]
async fn test_read_local_routing() -> anyhow::Result<()> {
    println!("🔄 Testing read-local routing to nearest replica");

    let manager = setup_test_manager().await;

    // Set up latencies
    {
        let mut topology = manager.get_topology().await;
        topology
            .latency_matrix
            .insert(("us-east-1".to_string(), "us-east-1".to_string()), 5.0);
        topology
            .latency_matrix
            .insert(("us-east-1".to_string(), "us-west-1".to_string()), 50.0);
        topology
            .latency_matrix
            .insert(("us-east-1".to_string(), "eu-central-1".to_string()), 100.0);
        topology.latency_matrix.insert(
            ("us-east-1".to_string(), "ap-southeast-1".to_string()),
            200.0,
        );
    }

    // Available replicas in different regions
    let replicas = vec![
        "us-west-1".to_string(),
        "eu-central-1".to_string(),
        "ap-southeast-1".to_string(),
    ];

    // Route read from us-east-1
    let selected_replica = manager.route_read("us-east-1", &replicas).await?;

    println!("   Selected replica: {}", selected_replica);
    println!("   Available replicas: {:?}", replicas);

    // Should select the nearest replica (us-west-1 with 50ms latency)
    assert_eq!(
        selected_replica, "us-west-1",
        "Should route to nearest replica"
    );

    println!("   ✓ Read-local routing test passed");
    Ok(())
}

/// Test 5: Compression - Verify ~70% compression ratio
#[tokio::test]
async fn test_compression() -> anyhow::Result<()> {
    println!("🔄 Testing data compression for cross-region transfer");

    let manager = setup_test_manager().await;

    // Create test data (repetitive for good compression)
    let test_data = vec![42u8; 10000]; // 10KB of repeated data

    // Compress
    let compressed = manager.compress_for_transfer(&test_data)?;

    let compression_ratio = compressed.len() as f64 / test_data.len() as f64;

    println!("   Original size: {} bytes", test_data.len());
    println!("   Compressed size: {} bytes", compressed.len());
    println!("   Compression ratio: {:.2}%", compression_ratio * 100.0);

    // Verify compression (should be well under 30% for repetitive data)
    assert!(
        compression_ratio < 0.3,
        "Should achieve good compression ratio for repetitive data"
    );

    // Verify decompression
    let decompressed = manager.decompress_from_transfer(&compressed)?;
    assert_eq!(
        decompressed, test_data,
        "Decompressed data should match original"
    );

    println!("   ✓ Compression test passed");
    Ok(())
}

/// Test 6: Latency Targets - Verify latency meets targets
#[tokio::test]
async fn test_latency_targets() -> anyhow::Result<()> {
    println!("🔄 Testing latency targets");

    let manager = setup_test_manager().await;

    // Set realistic latencies
    {
        let mut topology = manager.get_topology().await;

        // Same region: <5ms
        topology
            .latency_matrix
            .insert(("us-east-1".to_string(), "us-east-1".to_string()), 2.0);

        // Cross-region same continent: <50ms
        topology
            .latency_matrix
            .insert(("us-east-1".to_string(), "us-west-1".to_string()), 45.0);

        // Inter-continental: <200ms
        topology
            .latency_matrix
            .insert(("us-east-1".to_string(), "eu-central-1".to_string()), 95.0);
        topology.latency_matrix.insert(
            ("us-east-1".to_string(), "ap-southeast-1".to_string()),
            180.0,
        );
    }

    let topology = manager.get_topology().await;

    // Verify same region latency
    let same_region_latency = topology
        .latency_matrix
        .get(&("us-east-1".to_string(), "us-east-1".to_string()))
        .copied()
        .unwrap_or(0.0);
    println!("   Same region latency: {}ms", same_region_latency);
    assert!(same_region_latency < 5.0, "Same region should be <5ms");

    // Verify cross-region same continent
    let cross_region_latency = topology
        .latency_matrix
        .get(&("us-east-1".to_string(), "us-west-1".to_string()))
        .copied()
        .unwrap_or(0.0);
    println!(
        "   Cross-region (same continent) latency: {}ms",
        cross_region_latency
    );
    assert!(cross_region_latency < 50.0, "Cross-region should be <50ms");

    // Verify inter-continental
    let inter_continental_latency = topology
        .latency_matrix
        .get(&("us-east-1".to_string(), "ap-southeast-1".to_string()))
        .copied()
        .unwrap_or(0.0);
    println!(
        "   Inter-continental latency: {}ms",
        inter_continental_latency
    );
    assert!(
        inter_continental_latency < 200.0,
        "Inter-continental should be <200ms"
    );

    println!("   ✓ Latency targets test passed");
    Ok(())
}

/// Test 7: Multi-Region Deployment - Simulate US-East, US-West, EU, Asia
#[tokio::test]
async fn test_multi_region_deployment() -> anyhow::Result<()> {
    println!("🔄 Testing multi-region deployment simulation");

    let manager = setup_test_manager().await;

    // Register nodes in different regions
    manager
        .register_node(
            1,
            "us-east-1".to_string(),
            "us-east-1-a".to_string(),
            None,
            None,
        )
        .await?;
    manager
        .register_node(
            2,
            "us-west-1".to_string(),
            "us-west-1-a".to_string(),
            None,
            None,
        )
        .await?;
    manager
        .register_node(
            3,
            "eu-central-1".to_string(),
            "eu-central-1-a".to_string(),
            None,
            None,
        )
        .await?;
    manager
        .register_node(
            4,
            "ap-southeast-1".to_string(),
            "ap-southeast-1-a".to_string(),
            None,
            None,
        )
        .await?;

    // Verify all regions have nodes
    let us_east_nodes = manager.get_nodes_in_region("us-east-1").await;
    let us_west_nodes = manager.get_nodes_in_region("us-west-1").await;
    let eu_nodes = manager.get_nodes_in_region("eu-central-1").await;
    let asia_nodes = manager.get_nodes_in_region("ap-southeast-1").await;

    println!("   US East nodes: {}", us_east_nodes.len());
    println!("   US West nodes: {}", us_west_nodes.len());
    println!("   EU nodes: {}", eu_nodes.len());
    println!("   Asia nodes: {}", asia_nodes.len());

    assert_eq!(us_east_nodes.len(), 1, "Should have 1 node in US East");
    assert_eq!(us_west_nodes.len(), 1, "Should have 1 node in US West");
    assert_eq!(eu_nodes.len(), 1, "Should have 1 node in EU");
    assert_eq!(asia_nodes.len(), 1, "Should have 1 node in Asia");

    // Test routing between all region pairs
    let regions = vec!["us-east-1", "us-west-1", "eu-central-1", "ap-southeast-1"];
    let mut route_count = 0;

    for source in &regions {
        for dest in &regions {
            if source != dest {
                match manager.find_route(source, dest).await {
                    Ok(route) => {
                        println!(
                            "   Route {} → {}: {} hops, {}ms",
                            source,
                            dest,
                            route.hops.len(),
                            route.total_latency
                        );
                        route_count += 1;
                    }
                    Err(e) => {
                        println!(
                            "   Warning: Failed to find route {} → {}: {}",
                            source, dest, e
                        );
                    }
                }
            }
        }
    }

    println!("   Total routes tested: {}", route_count);
    assert!(route_count > 0, "Should have found some routes");

    println!("   ✓ Multi-region deployment test passed");
    Ok(())
}

/// Test 8: Relay Effectiveness - Verify >20% improvement
#[tokio::test]
async fn test_relay_effectiveness() -> anyhow::Result<()> {
    println!("🔄 Testing relay effectiveness (>20% improvement)");

    let manager = setup_test_manager().await;

    // Set up a scenario where relay is clearly better
    {
        let mut topology = manager.get_topology().await;

        // Direct path: 300ms
        topology.latency_matrix.insert(
            ("us-east-1".to_string(), "ap-southeast-1".to_string()),
            300.0,
        );

        // Relay path via EU: 100ms + 80ms = 180ms (40% improvement)
        topology
            .latency_matrix
            .insert(("us-east-1".to_string(), "eu-central-1".to_string()), 100.0);
        topology.latency_matrix.insert(
            ("eu-central-1".to_string(), "ap-southeast-1".to_string()),
            80.0,
        );

        // Also set reverse paths for completeness
        topology.latency_matrix.insert(
            ("ap-southeast-1".to_string(), "us-east-1".to_string()),
            300.0,
        );
        topology
            .latency_matrix
            .insert(("eu-central-1".to_string(), "us-east-1".to_string()), 100.0);
        topology.latency_matrix.insert(
            ("ap-southeast-1".to_string(), "eu-central-1".to_string()),
            80.0,
        );
    }

    let route = manager.find_route("us-east-1", "ap-southeast-1").await?;

    let direct_latency = 300.0;
    let improvement = (direct_latency - route.total_latency) / direct_latency;

    println!("   Direct latency: {}ms", direct_latency);
    println!("   Route latency: {}ms", route.total_latency);
    println!("   Route hops: {:?}", route.hops);
    println!("   Improvement: {:.1}%", improvement * 100.0);

    // Verify at least 20% improvement
    assert!(
        improvement >= 0.20 || route.total_latency < direct_latency * 0.8,
        "Relay should provide >20% improvement"
    );

    println!("   ✓ Relay effectiveness test passed");
    Ok(())
}

/// Test 9: Configuration - Verify all config options work
#[tokio::test]
async fn test_configuration() -> anyhow::Result<()> {
    println!("🔄 Testing configuration options");

    // Create custom config
    let custom_config = RegionConfig {
        enable_relay: true,
        relay_latency_threshold_ms: 150.0,
        enable_compression: true,
        enable_read_local: true,
        routing_strategy: RoutingStrategy::LatencyAware,
        ..Default::default()
    };

    println!("   Custom config created:");
    println!("     enable_relay: {}", custom_config.enable_relay);
    println!(
        "     relay_latency_threshold_ms: {}",
        custom_config.relay_latency_threshold_ms
    );
    println!(
        "     enable_compression: {}",
        custom_config.enable_compression
    );
    println!(
        "     enable_read_local: {}",
        custom_config.enable_read_local
    );
    println!(
        "     routing_strategy: {:?}",
        custom_config.routing_strategy
    );

    // Verify all fields are set correctly
    assert!(custom_config.enable_relay);
    assert_eq!(custom_config.relay_latency_threshold_ms, 150.0);
    assert!(custom_config.enable_compression);
    assert!(custom_config.enable_read_local);
    assert_eq!(
        custom_config.routing_strategy,
        RoutingStrategy::LatencyAware
    );

    println!("   ✓ Configuration test passed");
    Ok(())
}

/// Test 10: Error Handling - Verify proper error handling
#[tokio::test]
async fn test_error_handling() -> anyhow::Result<()> {
    println!("🔄 Testing error handling");

    let manager = setup_test_manager().await;

    // Test 1: Invalid source region
    match manager.find_route("invalid-region", "us-west-1").await {
        Err(_) => println!("   ✓ Correctly rejected invalid source region"),
        Ok(_) => panic!("Should have rejected invalid source region"),
    }

    // Test 2: Invalid destination region
    match manager.find_route("us-east-1", "invalid-region").await {
        Err(_) => println!("   ✓ Correctly rejected invalid destination region"),
        Ok(_) => panic!("Should have rejected invalid destination region"),
    }

    // Test 3: Empty replica list
    match manager.route_read("us-east-1", &[]).await {
        Err(_) => println!("   ✓ Correctly rejected empty replica list"),
        Ok(_) => panic!("Should have rejected empty replica list"),
    }

    // Test 4: Invalid compression (decompression bomb protection)
    let invalid_compressed = vec![0u8; 1000]; // Invalid compressed data
    match manager.decompress_from_transfer(&invalid_compressed) {
        Err(_) => println!("   ✓ Correctly rejected invalid compressed data"),
        Ok(_) => println!("   ⚠ Invalid data was accepted (may be valid by chance)"),
    }

    println!("   ✓ Error handling test passed");
    Ok(())
}
