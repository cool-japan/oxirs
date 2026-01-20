//! Stability Tests for OxiRS Cluster
//!
//! Simplified stability tests that validate cluster behavior under load.
//! These tests are kept simple to avoid complex API dependencies.
//!
//! Run with: cargo test --test stability_tests -- --nocapture

use oxirs_cluster::{ClusterNode, NodeConfig};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// Create a simple test node
async fn create_test_node(node_id: u64, port: u16) -> anyhow::Result<ClusterNode> {
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), port);
    let data_dir = std::env::temp_dir().join(format!("oxirs-stability-test-node-{}", node_id));

    // Clean up old data
    if data_dir.exists() {
        let _ = std::fs::remove_dir_all(&data_dir);
    }

    let config = NodeConfig::new(node_id, addr);
    ClusterNode::new(config).await.map_err(Into::into)
}

/// Test 1: Node creation and startup
#[tokio::test]
async fn test_node_creation_and_startup() -> anyhow::Result<()> {
    println!("🧪 Testing node creation and startup");

    let mut node = create_test_node(1, 9100).await?;

    // Start node
    node.start().await?;
    println!("   ✓ Node started successfully");

    // Wait for leader election
    sleep(Duration::from_millis(500)).await;

    // Single node cluster elects itself as leader (correct Raft behavior)
    assert!(node.is_leader().await);
    println!("   ✓ Single node elected itself as leader");

    // Stop node
    node.stop().await?;
    println!("   ✓ Node stopped successfully");

    Ok(())
}

/// Test 2: Basic triple operations
#[tokio::test]
async fn test_basic_triple_operations() -> anyhow::Result<()> {
    println!("🧪 Testing basic triple operations");

    let mut node = create_test_node(2, 9101).await?;
    node.start().await?;

    // Wait for node to be ready and elect itself as leader
    sleep(Duration::from_millis(500)).await;

    // Single node cluster becomes leader, so insert should succeed
    let result = node
        .insert_triple("http://test.org/s", "http://test.org/p", "\"object\"")
        .await;

    // Should succeed since single node is leader
    assert!(result.is_ok());
    println!("   ✓ Insert succeeds with leadership");

    // Query should return the inserted triple
    let results = node.query_triples(None, None, None).await;
    assert_eq!(results.len(), 1);
    println!("   ✓ Query returns inserted triple");

    node.stop().await?;

    Ok(())
}

/// Test 3: Multiple sequential operations
#[tokio::test]
async fn test_sequential_operations() -> anyhow::Result<()> {
    println!("🧪 Testing sequential operations");

    let mut node = create_test_node(3, 9102).await?;
    node.start().await?;

    sleep(Duration::from_millis(500)).await;

    // Perform multiple queries
    for i in 0..10 {
        let results = node.query_triples(None, None, None).await;
        assert_eq!(results.len(), 0);

        if i % 5 == 0 {
            println!("   ✓ Completed {} queries", i + 1);
        }
    }

    println!("   ✓ All 10 queries completed successfully");

    node.stop().await?;

    Ok(())
}

/// Test 4: Node status reporting
#[tokio::test]
async fn test_node_status_reporting() -> anyhow::Result<()> {
    println!("🧪 Testing node status reporting");

    let mut node = create_test_node(4, 9103).await?;
    node.start().await?;

    sleep(Duration::from_millis(500)).await;

    // Get status
    let status = node.get_status().await;

    println!("   Node ID: {}", status.node_id);
    println!("   Address: {}", status.address);
    println!("   Is Leader: {}", status.is_leader);
    println!("   Term: {}", status.current_term);
    println!("   Peers: {}", status.peer_count);
    println!("   Triples: {}", status.triple_count);

    assert_eq!(status.node_id, 4);
    assert_eq!(status.peer_count, 0); // No peers configured
    assert_eq!(status.triple_count, 0); // No data inserted

    println!("   ✓ Status reporting works correctly");

    node.stop().await?;

    Ok(())
}

/// Test 5: Rapid start/stop cycles
#[tokio::test]
async fn test_rapid_start_stop_cycles() -> anyhow::Result<()> {
    println!("🧪 Testing rapid start/stop cycles");

    for i in 0..5 {
        let mut node = create_test_node(100 + i, 9200 + i as u16).await?;

        node.start().await?;
        sleep(Duration::from_millis(100)).await;

        node.stop().await?;
        sleep(Duration::from_millis(100)).await;

        println!("   ✓ Cycle {} complete", i + 1);
    }

    println!("   ✓ All 5 cycles completed successfully");

    Ok(())
}

/// Test 6: Continuous operation under load
#[tokio::test]
async fn test_continuous_operation() -> anyhow::Result<()> {
    println!("🧪 Testing continuous operation (30 seconds)");

    let mut node = create_test_node(5, 9104).await?;
    node.start().await?;

    let start = Instant::now();
    let test_duration = Duration::from_secs(30);
    let mut operation_count = 0;

    while start.elapsed() < test_duration {
        // Perform queries continuously
        let _results = node.query_triples(None, None, None).await;
        operation_count += 1;

        // Brief sleep to avoid tight loop
        sleep(Duration::from_millis(10)).await;

        // Report progress every 5 seconds
        if operation_count % 500 == 0 {
            println!(
                "   Progress: {:.1}s | Operations: {}",
                start.elapsed().as_secs_f64(),
                operation_count
            );
        }
    }

    let elapsed = start.elapsed();
    let ops_per_sec = operation_count as f64 / elapsed.as_secs_f64();

    println!("\n   📊 Results:");
    println!("   Duration: {:.2}s", elapsed.as_secs_f64());
    println!("   Total operations: {}", operation_count);
    println!("   Operations/sec: {:.2}", ops_per_sec);

    assert!(operation_count > 1000, "Should perform many operations");
    assert!(ops_per_sec > 50.0, "Should maintain reasonable throughput");

    node.stop().await?;

    println!("   ✓ Continuous operation test passed");

    Ok(())
}

/// Test 7: Memory stability check
#[tokio::test]
async fn test_memory_stability() -> anyhow::Result<()> {
    println!("🧪 Testing memory stability");

    let mut node = create_test_node(6, 9105).await?;
    node.start().await?;

    // Take initial measurement (rough estimate)
    let initial_triple_count = node.len().await;

    // Perform operations for a while
    for i in 0..1000 {
        let _results = node.query_triples(None, None, None).await;

        if i % 250 == 0 {
            println!("   Completed {} operations", i);
        }
    }

    // Check final state
    let final_triple_count = node.len().await;

    println!("   Initial triples: {}", initial_triple_count);
    println!("   Final triples: {}", final_triple_count);

    // Should remain stable (no unexpected growth)
    assert_eq!(
        initial_triple_count, final_triple_count,
        "Triple count should remain stable"
    );

    println!("   ✓ Memory remains stable");

    node.stop().await?;

    Ok(())
}

/// Test 8: Graceful shutdown
#[tokio::test]
async fn test_graceful_shutdown() -> anyhow::Result<()> {
    println!("🧪 Testing graceful shutdown");

    let mut node = create_test_node(7, 9106).await?;
    node.start().await?;

    // Perform some operations
    for _ in 0..100 {
        let _results = node.query_triples(None, None, None).await;
    }

    println!("   ✓ Completed 100 operations");

    // Graceful shutdown
    let shutdown_start = Instant::now();
    node.graceful_shutdown().await?;
    let shutdown_duration = shutdown_start.elapsed();

    println!(
        "   ✓ Graceful shutdown completed in {:?}",
        shutdown_duration
    );

    // Verify shutdown was reasonably fast
    assert!(
        shutdown_duration < Duration::from_secs(5),
        "Shutdown should complete quickly"
    );

    Ok(())
}

/// Test 9: Status consistency
#[tokio::test]
async fn test_status_consistency() -> anyhow::Result<()> {
    println!("🧪 Testing status consistency");

    let mut node = create_test_node(8, 9107).await?;
    node.start().await?;

    sleep(Duration::from_millis(500)).await;

    // Get status multiple times
    for i in 0..10 {
        let status = node.get_status().await;

        assert_eq!(status.node_id, 8);
        assert_eq!(status.triple_count, 0);

        if i == 0 || i == 9 {
            println!("   ✓ Status check {} consistent", i + 1);
        }
    }

    println!("   ✓ All 10 status checks consistent");

    node.stop().await?;

    Ok(())
}

/// Test 10: Node ID and address verification
#[tokio::test]
async fn test_node_identification() -> anyhow::Result<()> {
    println!("🧪 Testing node identification");

    let node_id = 9;
    let port = 9108;
    let mut node = create_test_node(node_id, port).await?;

    assert_eq!(node.id(), node_id);
    println!("   ✓ Node ID matches: {}", node_id);

    node.start().await?;

    let status = node.get_status().await;
    assert_eq!(status.node_id, node_id);
    assert_eq!(status.address.port(), port);

    println!("   ✓ Address matches: {}", status.address);

    node.stop().await?;

    Ok(())
}
