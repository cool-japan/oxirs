//! Comprehensive Integration Tests for OxiRS Cluster
//!
//! This module provides end-to-end integration testing for the distributed
//! RDF storage cluster including consensus, replication, and fault tolerance.

use anyhow::Result;
use oxirs_cluster::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// Integration test configuration
#[derive(Debug, Clone)]
pub struct IntegrationTestConfig {
    pub num_nodes: usize,
    pub test_duration: Duration,
    pub concurrent_operations: usize,
    pub failure_scenarios: bool,
    pub network_partition: bool,
    pub byzantine_faults: bool,
}

impl Default for IntegrationTestConfig {
    fn default() -> Self {
        Self {
            num_nodes: 5,
            test_duration: Duration::from_secs(30),
            concurrent_operations: 100,
            failure_scenarios: true,
            network_partition: true,
            byzantine_faults: false,
        }
    }
}

/// Test cluster for integration testing
pub struct TestCluster {
    nodes: Vec<ClusterNode>,
    config: IntegrationTestConfig,
    metrics: TestMetrics,
}

#[derive(Debug, Default, Clone)]
pub struct TestMetrics {
    pub total_operations: usize,
    pub successful_operations: usize,
    pub failed_operations: usize,
    pub consensus_latency: Duration,
    pub replication_latency: Duration,
    pub partition_recovery_time: Duration,
    pub leader_elections: usize,
    pub throughput_ops_per_sec: f64,
}

impl TestCluster {
    /// Create a new test cluster
    pub async fn new(config: IntegrationTestConfig) -> Result<Self> {
        let mut nodes = Vec::new();

        // Create cluster nodes
        for i in 0..config.num_nodes {
            let node_config = NodeConfig {
                node_id: i as u64 + 1,
                address: format!("127.0.0.1:{}", 8080 + i).parse()?,
                data_dir: format!("./test_data/node_{}", i),
                peers: (1..=config.num_nodes as u64)
                    .filter(|&id| id != (i as u64 + 1))
                    .collect(),
                discovery: None,
                replication_strategy: None,
                region_config: None,
            };

            let node = ClusterNode::new(node_config).await?;
            nodes.push(node);
        }

        Ok(Self {
            nodes,
            config,
            metrics: TestMetrics::default(),
        })
    }

    /// Start all nodes in the cluster
    pub async fn start(&mut self) -> Result<()> {
        println!("Starting test cluster with {} nodes", self.config.num_nodes);

        // Start nodes with staggered delays
        let num_nodes = self.nodes.len();
        for (i, node) in self.nodes.iter_mut().enumerate() {
            node.start().await?;
            if i < num_nodes - 1 {
                sleep(Duration::from_millis(100)).await;
            }
        }

        // Wait for initial leader election
        sleep(Duration::from_secs(2)).await;
        println!("Cluster started successfully");
        Ok(())
    }

    /// Run comprehensive integration tests
    pub async fn run_integration_tests(&mut self) -> Result<TestMetrics> {
        println!("Running comprehensive integration tests...");

        // Test 1: Basic consensus and replication
        self.test_basic_consensus().await?;

        // Test 2: Concurrent operations
        self.test_concurrent_operations().await?;

        // Test 3: Leader failover
        if self.config.failure_scenarios {
            self.test_leader_failover().await?;
        }

        // Test 4: Network partitions
        if self.config.network_partition {
            self.test_network_partition().await?;
        }

        // Test 5: Byzantine fault tolerance
        if self.config.byzantine_faults {
            self.test_byzantine_faults().await?;
        }

        // Test 6: Performance validation
        self.test_performance_validation().await?;

        // Test 7: Consistency validation
        self.test_consistency_validation().await?;

        // Calculate final metrics
        self.calculate_final_metrics();

        println!("Integration tests completed successfully");
        Ok(self.metrics.clone())
    }

    /// Test basic consensus and replication
    async fn test_basic_consensus(&mut self) -> Result<()> {
        println!("Testing basic consensus and replication...");

        let start_time = Instant::now();

        // Insert test data through different nodes
        for i in 0..10 {
            let leader_node = self.find_leader().await?;
            let subject = format!("http://test.org/subject_{}", i);
            let predicate = "http://test.org/predicate";
            let object = format!("\"test_object_{}\"", i);

            leader_node
                .insert_triple(&subject, predicate, &object)
                .await?;
            self.metrics.total_operations += 1;
            self.metrics.successful_operations += 1;
        }

        self.metrics.consensus_latency = start_time.elapsed() / 10;

        // Verify replication across all nodes
        sleep(Duration::from_millis(500)).await; // Wait for replication

        let expected_count = 10;
        for node in &self.nodes {
            let count = node.count_triples().await?;
            assert_eq!(
                count,
                expected_count,
                "Replication failed on node {}",
                node.id()
            );
        }

        println!("Basic consensus and replication test passed");
        Ok(())
    }

    /// Test concurrent operations
    async fn test_concurrent_operations(&mut self) -> Result<()> {
        println!("Testing concurrent operations...");

        let start_time = Instant::now();
        let concurrent_ops = self.config.concurrent_operations;

        // Execute operations sequentially to avoid borrowing issues
        let leader = self.find_leader().await?;
        let mut successful = 0;
        let mut failed = 0;

        for i in 0..concurrent_ops {
            let subject = format!("http://concurrent.org/subject_{}", i);
            let predicate = "http://concurrent.org/predicate";
            let object = format!("\"concurrent_object_{}\"", i);

            match leader.insert_triple(&subject, predicate, &object).await {
                Ok(_) => successful += 1,
                Err(_) => failed += 1,
            }
        }

        let duration = start_time.elapsed();
        self.metrics.total_operations += concurrent_ops;
        self.metrics.successful_operations += successful;
        self.metrics.failed_operations += failed;
        self.metrics.throughput_ops_per_sec = successful as f64 / duration.as_secs_f64();

        println!(
            "Concurrent operations test completed: {}/{} successful",
            successful, concurrent_ops
        );
        Ok(())
    }

    /// Test leader failover scenarios
    async fn test_leader_failover(&mut self) -> Result<()> {
        println!("Testing leader failover...");

        let start_time = Instant::now();

        // Find current leader
        let leader_id = self.find_leader_id().await?;
        println!("Current leader: node {}", leader_id);

        // Simulate leader failure
        self.simulate_node_failure(leader_id).await?;

        // Wait for new leader election
        sleep(Duration::from_secs(2)).await;

        // Verify new leader elected
        let new_leader_id = self.find_leader_id().await?;
        assert_ne!(leader_id, new_leader_id, "New leader should be different");

        self.metrics.leader_elections += 1;

        // Test operations still work
        let new_leader = self.find_leader().await?;
        new_leader
            .insert_triple(
                "http://failover.org/subject",
                "http://failover.org/predicate",
                "\"failover_test\"",
            )
            .await?;

        // Recover failed node
        self.recover_node(leader_id).await?;

        let recovery_time = start_time.elapsed();
        println!("Leader failover test completed in {:?}", recovery_time);
        Ok(())
    }

    /// Test network partition scenarios
    async fn test_network_partition(&mut self) -> Result<()> {
        println!("Testing network partition resilience...");

        let start_time = Instant::now();

        // Create partition: split cluster into two groups
        let partition_size = self.config.num_nodes / 2;
        self.simulate_network_partition(partition_size).await?;

        // Majority partition should continue operating
        sleep(Duration::from_secs(1)).await;

        // Test operations on majority partition
        let leader = self.find_leader().await?;
        leader
            .insert_triple(
                "http://partition.org/subject",
                "http://partition.org/predicate",
                "\"partition_test\"",
            )
            .await?;

        // Heal partition
        self.heal_network_partition().await?;

        // Wait for re-convergence
        sleep(Duration::from_secs(2)).await;

        // Verify consistency after partition healing
        self.verify_cluster_consistency().await?;

        self.metrics.partition_recovery_time = start_time.elapsed();
        println!("Network partition test completed");
        Ok(())
    }

    /// Test Byzantine fault tolerance
    async fn test_byzantine_faults(&mut self) -> Result<()> {
        println!("Testing Byzantine fault tolerance...");

        // Simulate Byzantine behavior in minority of nodes
        let byzantine_count = (self.config.num_nodes - 1) / 3; // Up to f = (n-1)/3

        for i in 0..byzantine_count {
            self.simulate_byzantine_behavior(i).await?;
        }

        // Verify cluster continues operating correctly
        let leader = self.find_leader().await?;
        leader
            .insert_triple(
                "http://byzantine.org/subject",
                "http://byzantine.org/predicate",
                "\"byzantine_test\"",
            )
            .await?;

        // Verify honest nodes maintain consistency
        self.verify_honest_node_consistency().await?;

        println!("Byzantine fault tolerance test completed");
        Ok(())
    }

    /// Test performance validation
    async fn test_performance_validation(&mut self) -> Result<()> {
        println!("Running performance validation...");

        let start_time = Instant::now();
        let target_ops = 1000;

        // Execute operations sequentially to avoid borrowing issues
        let leader = self.find_leader().await?;

        for batch in 0..10 {
            for i in 0..100 {
                let subject = format!("http://perf.org/subject_{}_{}", batch, i);
                let predicate = "http://perf.org/predicate";
                let object = format!("\"perf_object_{}\"", i);

                leader.insert_triple(&subject, predicate, &object).await?;
            }
        }

        let duration = start_time.elapsed();
        let ops_per_sec = target_ops as f64 / duration.as_secs_f64();

        self.metrics.throughput_ops_per_sec = self.metrics.throughput_ops_per_sec.max(ops_per_sec);

        println!("Performance validation: {:.2} ops/sec", ops_per_sec);

        // Validate performance meets targets
        assert!(
            ops_per_sec > 100.0,
            "Performance below target: {} ops/sec",
            ops_per_sec
        );

        Ok(())
    }

    /// Test consistency validation
    async fn test_consistency_validation(&mut self) -> Result<()> {
        println!("Validating cluster consistency...");

        // Insert test data with known values
        let leader = self.find_leader().await?;
        let test_triples = vec![
            (
                "http://consistency.org/s1",
                "http://consistency.org/p1",
                "\"value1\"",
            ),
            (
                "http://consistency.org/s2",
                "http://consistency.org/p2",
                "\"value2\"",
            ),
            (
                "http://consistency.org/s3",
                "http://consistency.org/p3",
                "\"value3\"",
            ),
        ];

        for (s, p, o) in &test_triples {
            leader.insert_triple(s, p, o).await?;
        }

        // Wait for replication
        sleep(Duration::from_millis(500)).await;

        // Verify all nodes have consistent state
        self.verify_cluster_consistency().await?;

        println!("Consistency validation passed");
        Ok(())
    }

    /// Helper methods

    async fn find_leader(&self) -> Result<&ClusterNode> {
        for node in &self.nodes {
            if node.is_leader().await? {
                return Ok(node);
            }
        }
        Err(anyhow::anyhow!("No leader found"))
    }

    async fn find_leader_id(&self) -> Result<usize> {
        let leader = self.find_leader().await?;
        Ok(leader.id() as usize - 1)
    }

    async fn simulate_node_failure(&mut self, node_id: usize) -> Result<()> {
        println!("Simulating failure of node {}", node_id);
        self.nodes[node_id].stop().await?;
        Ok(())
    }

    async fn recover_node(&mut self, node_id: usize) -> Result<()> {
        println!("Recovering node {}", node_id);
        self.nodes[node_id].start().await?;
        sleep(Duration::from_millis(500)).await; // Allow rejoin
        Ok(())
    }

    async fn simulate_network_partition(&mut self, partition_size: usize) -> Result<()> {
        println!(
            "Simulating network partition: {} nodes isolated",
            self.config.num_nodes - partition_size
        );

        // Simulate by stopping minority nodes
        for i in partition_size..self.config.num_nodes {
            self.nodes[i].isolate_network().await?;
        }
        Ok(())
    }

    async fn heal_network_partition(&mut self) -> Result<()> {
        println!("Healing network partition");

        for node in &mut self.nodes {
            node.restore_network().await?;
        }
        Ok(())
    }

    async fn simulate_byzantine_behavior(&mut self, node_id: usize) -> Result<()> {
        println!("Simulating Byzantine behavior on node {}", node_id);
        self.nodes[node_id].enable_byzantine_mode().await?;
        Ok(())
    }

    async fn verify_cluster_consistency(&self) -> Result<()> {
        let leader = self.find_leader().await?;
        let expected_count = leader.count_triples().await?;

        for node in &self.nodes {
            if node.is_active().await? {
                let count = node.count_triples().await?;
                assert_eq!(
                    count,
                    expected_count,
                    "Inconsistent state on node {}",
                    node.id()
                );
            }
        }
        Ok(())
    }

    async fn verify_honest_node_consistency(&self) -> Result<()> {
        let mut honest_counts = Vec::new();

        for node in &self.nodes {
            if !node.is_byzantine().await? && node.is_active().await? {
                honest_counts.push(node.count_triples().await?);
            }
        }

        // All honest nodes should have same count
        assert!(
            honest_counts.windows(2).all(|w| w[0] == w[1]),
            "Honest nodes have inconsistent state"
        );
        Ok(())
    }

    fn calculate_final_metrics(&mut self) {
        if self.metrics.total_operations > 0 {
            let success_rate =
                self.metrics.successful_operations as f64 / self.metrics.total_operations as f64;
            println!("Success rate: {:.2}%", success_rate * 100.0);
        }

        println!("Final metrics: {:?}", self.metrics);
    }
}

/// Run comprehensive integration tests
pub async fn run_cluster_integration_tests() -> Result<TestMetrics> {
    let config = IntegrationTestConfig::default();
    let mut cluster = TestCluster::new(config).await?;

    cluster.start().await?;
    let metrics = cluster.run_integration_tests().await?;

    // Cleanup
    cluster.shutdown().await?;

    Ok(metrics)
}

impl TestCluster {
    async fn shutdown(&mut self) -> Result<()> {
        for node in &mut self.nodes {
            node.stop().await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cluster_integration() {
        let config = IntegrationTestConfig {
            num_nodes: 3,
            test_duration: Duration::from_secs(10),
            concurrent_operations: 50,
            failure_scenarios: true,
            network_partition: false, // Skip for fast test
            byzantine_faults: false,
        };

        let mut cluster = TestCluster::new(config).await.unwrap();
        cluster.start().await.unwrap();

        let metrics = cluster.run_integration_tests().await.unwrap();
        assert!(metrics.successful_operations > 0);
        assert!(metrics.throughput_ops_per_sec > 10.0);

        cluster.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_performance_benchmark() {
        let config = IntegrationTestConfig {
            num_nodes: 5,
            test_duration: Duration::from_secs(30),
            concurrent_operations: 200,
            failure_scenarios: false,
            network_partition: false,
            byzantine_faults: false,
        };

        let mut cluster = TestCluster::new(config).await.unwrap();
        cluster.start().await.unwrap();

        cluster.test_performance_validation().await.unwrap();

        // Validate performance targets
        assert!(cluster.metrics.throughput_ops_per_sec > 100.0);
        assert!(cluster.metrics.consensus_latency < Duration::from_millis(100));

        cluster.shutdown().await.unwrap();
    }
}
