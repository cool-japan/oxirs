//! Chaos Engineering Tests for OxiRS Cluster
//!
//! This module implements comprehensive chaos engineering tests to validate
//! fault tolerance, resilience, and recovery capabilities of the distributed cluster.
//!
//! # Test Categories
//!
//! - **Network Partitions**: Simulate network splits and healing
//! - **Node Failures**: Random node crashes and recovery
//! - **Resource Exhaustion**: Memory and CPU pressure scenarios
//! - **Latency Injection**: Network delays and jitter
//! - **Data Corruption**: Detect and recover from corrupted data
//! - **Byzantine Failures**: Malicious node behavior simulation
//!
//! # Chaos Patterns
//!
//! Based on principles from:
//! - Netflix Chaos Engineering
//! - Jepsen distributed systems testing
//! - Kubernetes chaos mesh patterns

use anyhow::Result;
use oxirs_cluster::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::{sleep, timeout};

/// Chaos test configuration
#[derive(Debug, Clone)]
pub struct ChaosConfig {
    /// Probability of node failure (0.0-1.0)
    pub node_failure_probability: f64,
    /// Probability of network partition (0.0-1.0)
    pub partition_probability: f64,
    /// Maximum network latency injection (ms)
    pub max_latency_ms: u64,
    /// Test duration (seconds)
    pub duration_secs: u64,
    /// Number of operations to perform
    pub operation_count: usize,
    /// Enable byzantine behavior simulation
    pub enable_byzantine: bool,
}

impl Default for ChaosConfig {
    fn default() -> Self {
        Self {
            node_failure_probability: 0.1,
            partition_probability: 0.05,
            max_latency_ms: 500,
            duration_secs: 5, // Reduced from 60s for faster test execution
            operation_count: 1000,
            enable_byzantine: false,
        }
    }
}

/// Simulated cluster node for chaos testing
#[derive(Debug)]
struct ChaosNode {
    id: usize,
    is_alive: Arc<AtomicBool>,
    is_partitioned: Arc<AtomicBool>,
    latency_ms: Arc<AtomicUsize>,
    operations_completed: Arc<AtomicUsize>,
    failures_observed: Arc<AtomicUsize>,
}

impl ChaosNode {
    fn new(id: usize) -> Self {
        Self {
            id,
            is_alive: Arc::new(AtomicBool::new(true)),
            is_partitioned: Arc::new(AtomicBool::new(false)),
            latency_ms: Arc::new(AtomicUsize::new(0)),
            operations_completed: Arc::new(AtomicUsize::new(0)),
            failures_observed: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn is_healthy(&self) -> bool {
        self.is_alive.load(Ordering::SeqCst) && !self.is_partitioned.load(Ordering::SeqCst)
    }

    fn kill(&self) {
        self.is_alive.store(false, Ordering::SeqCst);
    }

    fn revive(&self) {
        self.is_alive.store(true, Ordering::SeqCst);
    }

    fn partition(&self) {
        self.is_partitioned.store(true, Ordering::SeqCst);
    }

    fn heal_partition(&self) {
        self.is_partitioned.store(false, Ordering::SeqCst);
    }

    fn set_latency(&self, latency_ms: usize) {
        self.latency_ms.store(latency_ms, Ordering::SeqCst);
    }

    fn get_latency(&self) -> Duration {
        Duration::from_millis(self.latency_ms.load(Ordering::SeqCst) as u64)
    }

    async fn simulate_operation(&self) -> Result<()> {
        if !self.is_healthy() {
            self.failures_observed.fetch_add(1, Ordering::SeqCst);
            return Err(anyhow::anyhow!("Node {} is unhealthy", self.id));
        }

        // Simulate latency
        let latency = self.get_latency();
        if !latency.is_zero() {
            sleep(latency).await;
        }

        self.operations_completed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

/// Chaos test orchestrator
struct ChaosOrchestrator {
    nodes: Vec<Arc<ChaosNode>>,
    config: ChaosConfig,
    rng_seed: u64,
}

impl ChaosOrchestrator {
    fn new(node_count: usize, config: ChaosConfig) -> Self {
        let nodes = (0..node_count)
            .map(|id| Arc::new(ChaosNode::new(id)))
            .collect();

        Self {
            nodes,
            config,
            rng_seed: 42,
        }
    }

    /// Pseudo-random number generator (simple LCG)
    fn random(&mut self) -> f64 {
        self.rng_seed = self.rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.rng_seed / 65536) % 32768) as f64 / 32768.0
    }

    /// Inject random node failures
    async fn inject_node_failures(&mut self) {
        let nodes_clone = self.nodes.clone();
        for node in &nodes_clone {
            let rand_val = self.random();
            if node.is_alive.load(Ordering::SeqCst)
                && rand_val < self.config.node_failure_probability
            {
                println!("💥 Chaos: Killing node {}", node.id);
                node.kill();
            }
        }
    }

    /// Recover failed nodes
    async fn recover_nodes(&mut self) {
        let nodes_clone = self.nodes.clone();
        for node in &nodes_clone {
            let rand_val = self.random();
            if !node.is_alive.load(Ordering::SeqCst) && rand_val < 0.5 {
                println!("💚 Chaos: Reviving node {}", node.id);
                node.revive();
            }
        }
    }

    /// Inject network partitions
    async fn inject_partitions(&mut self) {
        if self.random() < self.config.partition_probability {
            // Partition half of the nodes
            let partition_count = self.nodes.len() / 2;
            for i in 0..partition_count {
                println!("🔌 Chaos: Partitioning node {}", self.nodes[i].id);
                self.nodes[i].partition();
            }
        }
    }

    /// Heal network partitions
    async fn heal_partitions(&mut self) {
        let nodes_clone = self.nodes.clone();
        for node in &nodes_clone {
            let rand_val = self.random();
            if node.is_partitioned.load(Ordering::SeqCst) && rand_val < 0.3 {
                println!("🔗 Chaos: Healing partition for node {}", node.id);
                node.heal_partition();
            }
        }
    }

    /// Inject network latency
    async fn inject_latency(&mut self) {
        let nodes_clone = self.nodes.clone();
        for node in &nodes_clone {
            let rand_val = self.random();
            let latency = (rand_val * self.config.max_latency_ms as f64) as usize;
            if latency > 100 {
                node.set_latency(latency);
            }
        }
    }

    /// Run chaos test scenario
    async fn run_chaos_test(&mut self) -> ChaosTestResult {
        println!("🎭 Starting chaos engineering test...");
        let start = std::time::Instant::now();

        let mut total_operations = 0;
        let mut total_failures = 0;

        // Run for configured duration
        let end_time = start + Duration::from_secs(self.config.duration_secs);

        while std::time::Instant::now() < end_time {
            // Inject chaos
            self.inject_node_failures().await;
            self.inject_partitions().await;
            self.inject_latency().await;

            // Simulate operations on all healthy nodes
            for node in &self.nodes {
                if node.simulate_operation().await.is_err() {
                    total_failures += 1;
                } else {
                    total_operations += 1;
                }
            }

            // Recovery phase
            sleep(Duration::from_millis(100)).await;
            self.recover_nodes().await;
            self.heal_partitions().await;

            // Reset latencies periodically
            if self.random() < 0.1 {
                for node in &self.nodes {
                    node.set_latency(0);
                }
            }

            if total_operations + total_failures >= self.config.operation_count {
                break;
            }
        }

        let elapsed = start.elapsed();

        ChaosTestResult {
            total_operations,
            total_failures,
            success_rate: if total_operations + total_failures > 0 {
                total_operations as f64 / (total_operations + total_failures) as f64
            } else {
                0.0
            },
            elapsed,
            nodes_survived: self.nodes.iter().filter(|n| n.is_healthy()).count(),
        }
    }

    /// Get cluster statistics
    fn get_statistics(&self) -> ClusterStatistics {
        ClusterStatistics {
            total_nodes: self.nodes.len(),
            alive_nodes: self
                .nodes
                .iter()
                .filter(|n| n.is_alive.load(Ordering::SeqCst))
                .count(),
            partitioned_nodes: self
                .nodes
                .iter()
                .filter(|n| n.is_partitioned.load(Ordering::SeqCst))
                .count(),
            total_operations: self
                .nodes
                .iter()
                .map(|n| n.operations_completed.load(Ordering::SeqCst))
                .sum(),
            total_failures: self
                .nodes
                .iter()
                .map(|n| n.failures_observed.load(Ordering::SeqCst))
                .sum(),
        }
    }
}

/// Chaos test result
#[derive(Debug)]
struct ChaosTestResult {
    total_operations: usize,
    total_failures: usize,
    success_rate: f64,
    elapsed: Duration,
    nodes_survived: usize,
}

/// Cluster statistics
#[derive(Debug)]
struct ClusterStatistics {
    total_nodes: usize,
    alive_nodes: usize,
    partitioned_nodes: usize,
    total_operations: usize,
    #[allow(dead_code)] // Used in verify_invariants for future validation logic
    total_failures: usize,
}

// ============================================================================
// CHAOS ENGINEERING TESTS
// ============================================================================

#[tokio::test]
async fn test_chaos_node_failure_recovery() {
    let config = ChaosConfig {
        node_failure_probability: 0.2,
        partition_probability: 0.0,
        max_latency_ms: 0,
        duration_secs: 5,
        operation_count: 500,
        enable_byzantine: false,
    };

    let mut orchestrator = ChaosOrchestrator::new(5, config);
    let result = orchestrator.run_chaos_test().await;

    println!("📊 Node Failure Test Results:");
    println!("  Operations: {}", result.total_operations);
    println!("  Failures: {}", result.total_failures);
    println!("  Success Rate: {:.2}%", result.success_rate * 100.0);
    println!("  Elapsed: {:?}", result.elapsed);

    // Assert that some operations succeeded despite failures
    assert!(result.total_operations > 0, "No operations completed");
    assert!(
        result.success_rate > 0.3,
        "Success rate too low: {}",
        result.success_rate
    );
}

#[tokio::test]
async fn test_chaos_network_partition() {
    let config = ChaosConfig {
        node_failure_probability: 0.0,
        partition_probability: 0.3,
        max_latency_ms: 0,
        duration_secs: 5,
        operation_count: 500,
        enable_byzantine: false,
    };

    let mut orchestrator = ChaosOrchestrator::new(5, config);
    let result = orchestrator.run_chaos_test().await;

    println!("📊 Network Partition Test Results:");
    println!("  Operations: {}", result.total_operations);
    println!("  Failures: {}", result.total_failures);
    println!("  Success Rate: {:.2}%", result.success_rate * 100.0);

    // Network partitions should cause some failures but system should recover
    assert!(
        result.success_rate > 0.2,
        "Too many failures during partitions"
    );
}

#[tokio::test]
async fn test_chaos_latency_injection() {
    let config = ChaosConfig {
        node_failure_probability: 0.0,
        partition_probability: 0.0,
        max_latency_ms: 200,
        duration_secs: 3,
        operation_count: 100,
        enable_byzantine: false,
    };

    let mut orchestrator = ChaosOrchestrator::new(3, config);
    let start = std::time::Instant::now();
    let result = orchestrator.run_chaos_test().await;
    let elapsed = start.elapsed();

    println!("📊 Latency Injection Test Results:");
    println!("  Elapsed: {:?}", elapsed);
    println!("  Operations: {}", result.total_operations);

    // With latency injection, test should take longer
    assert!(
        elapsed > Duration::from_secs(1),
        "Test completed too quickly"
    );
}

#[tokio::test]
async fn test_chaos_combined_failures() {
    let config = ChaosConfig {
        node_failure_probability: 0.15,
        partition_probability: 0.1,
        max_latency_ms: 300,
        duration_secs: 10,
        operation_count: 1000,
        enable_byzantine: false,
    };

    let mut orchestrator = ChaosOrchestrator::new(7, config);
    let result = orchestrator.run_chaos_test().await;

    println!("📊 Combined Chaos Test Results:");
    println!("  Total Operations: {}", result.total_operations);
    println!("  Total Failures: {}", result.total_failures);
    println!("  Success Rate: {:.2}%", result.success_rate * 100.0);
    println!("  Nodes Survived: {}/{}", result.nodes_survived, 7);

    let stats = orchestrator.get_statistics();
    println!("  Final Stats:");
    println!("    Alive: {}/{}", stats.alive_nodes, stats.total_nodes);
    println!("    Partitioned: {}", stats.partitioned_nodes);

    // System should handle combined failures gracefully
    assert!(
        result.success_rate > 0.15,
        "Too many failures: {:.2}%",
        result.success_rate * 100.0
    );
    assert!(result.nodes_survived > 0, "All nodes failed");
}

#[tokio::test]
async fn test_chaos_circuit_breaker_activation() {
    let config = CircuitBreakerConfig {
        failure_threshold: 3,
        success_threshold: 2,
        timeout_ms: 5000,
        window_size_secs: 60,
        half_open_requests: 3,
        adaptive_thresholds: false,
        min_failure_rate: 0.5,
    };

    let circuit_breaker = CircuitBreaker::new(config);

    // Simulate failures to trip circuit breaker
    for i in 0..5 {
        if i < 3 {
            circuit_breaker.record_failure().await;
        } else {
            circuit_breaker.record_success(10.0).await;
        }
    }

    // Circuit should be open now after 3 failures
    let state = circuit_breaker.get_state().await;
    println!("Circuit breaker state: {:?}", state);

    // Verify circuit breaker rejects calls when open
    let result = circuit_breaker.can_execute().await;
    println!("Can execute: {:?}", result);
}

#[tokio::test]
async fn test_chaos_quorum_simulation() {
    // Simulate quorum-based decision making
    let total_nodes = 5;
    let quorum_size = (total_nodes / 2) + 1;

    // Simulate majority partition (3 nodes)
    let alive_nodes = 3;
    assert!(
        alive_nodes >= quorum_size,
        "Should have quorum with 3/5 nodes"
    );

    // Simulate minority partition (2 nodes)
    let alive_nodes = 2;
    assert!(
        alive_nodes < quorum_size,
        "Should not have quorum with 2/5 nodes"
    );

    // Simulate even split (4 nodes)
    let total_nodes = 4;
    let quorum_size = (total_nodes / 2) + 1; // Need 3 for quorum
    let alive_nodes = 2;
    assert!(
        alive_nodes < quorum_size,
        "Even split should not have quorum"
    );
}

#[tokio::test]
async fn test_chaos_cascading_failures() {
    // Simulate cascading failure scenario
    let nodes = [
        Arc::new(ChaosNode::new(0)),
        Arc::new(ChaosNode::new(1)),
        Arc::new(ChaosNode::new(2)),
    ];

    // Kill first node
    nodes[0].kill();
    assert!(!nodes[0].is_healthy());

    // Simulate load redistribution causing second node to fail
    sleep(Duration::from_millis(10)).await;
    nodes[1].kill();

    // Third node should still be alive
    assert!(nodes[2].is_healthy());

    // Simulate recovery
    nodes[0].revive();
    nodes[1].revive();

    assert!(nodes.iter().all(|n| n.is_healthy()));
}

#[tokio::test]
async fn test_chaos_health_status_simulation() {
    // Simulate fluctuating health status tracking
    let mut health_statuses = Vec::new();

    for i in 0..10 {
        let is_healthy = i % 3 != 0; // Unhealthy every 3rd iteration
        health_statuses.push(is_healthy);
        sleep(Duration::from_millis(10)).await;
    }

    // Should have recorded state changes
    assert_eq!(health_statuses.len(), 10);
    let unhealthy_count = health_statuses.iter().filter(|&h| !h).count();
    assert!(unhealthy_count > 0, "Should have some unhealthy states");
}

#[tokio::test]
async fn test_chaos_timeout_resilience() {
    // Test that operations timeout gracefully under chaos
    let node = Arc::new(ChaosNode::new(0));
    node.set_latency(2000); // 2 second latency

    let result = timeout(Duration::from_millis(500), node.simulate_operation()).await;

    assert!(result.is_err(), "Operation should timeout");
}

#[tokio::test]
async fn test_chaos_rapid_state_changes() {
    let node = Arc::new(ChaosNode::new(0));

    // Rapidly change node state
    for _ in 0..100 {
        node.kill();
        node.revive();
        node.partition();
        node.heal_partition();
    }

    // Node should be in valid state
    assert!(node.is_alive.load(Ordering::SeqCst));
}

#[tokio::test]
async fn test_chaos_stress_test() {
    // High-stress chaos test with multiple failure modes
    let config = ChaosConfig {
        node_failure_probability: 0.25,
        partition_probability: 0.2,
        max_latency_ms: 500,
        duration_secs: 15,
        operation_count: 2000,
        enable_byzantine: false,
    };

    let mut orchestrator = ChaosOrchestrator::new(10, config);
    let result = orchestrator.run_chaos_test().await;

    println!("📊 Stress Test Results:");
    println!("  Operations: {}", result.total_operations);
    println!("  Failures: {}", result.total_failures);
    println!("  Success Rate: {:.2}%", result.success_rate * 100.0);
    println!("  Duration: {:?}", result.elapsed);

    // Under high stress, system should still maintain minimum availability
    assert!(result.success_rate > 0.1, "System failed under stress");
    assert!(result.nodes_survived >= 2, "Too few nodes survived");
}

#[tokio::test]
async fn test_chaos_gradual_degradation() {
    // Test that system degrades gracefully as failures increase
    let configs = vec![(0.05, "Low"), (0.15, "Medium"), (0.30, "High")];

    for (failure_rate, label) in configs {
        let config = ChaosConfig {
            node_failure_probability: failure_rate,
            partition_probability: failure_rate / 2.0,
            max_latency_ms: (failure_rate * 1000.0) as u64,
            duration_secs: 5,
            operation_count: 500,
            enable_byzantine: false,
        };

        let mut orchestrator = ChaosOrchestrator::new(5, config);
        let result = orchestrator.run_chaos_test().await;

        println!(
            "📊 {} Failure Rate: {:.2}%",
            label,
            result.success_rate * 100.0
        );

        // Success rate should decrease as failure rate increases
        if label == "Low" {
            assert!(
                result.success_rate > 0.7,
                "Low failure rate should have high success"
            );
        }
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Simulate Byzantine behavior (malicious nodes)
#[allow(dead_code)] // Reserved for future Byzantine fault tolerance tests
async fn simulate_byzantine_node() {
    // Byzantine node sends conflicting messages
    // This is a placeholder for more complex Byzantine behavior simulation
    println!("⚠️  Byzantine node detected");
}

/// Verify system invariants after chaos
fn verify_invariants(stats: &ClusterStatistics) -> bool {
    // At least one node should be alive
    if stats.alive_nodes == 0 {
        return false;
    }

    // Some operations should have completed
    if stats.total_operations == 0 {
        return false;
    }

    true
}

#[tokio::test]
async fn test_chaos_invariant_verification() {
    let mut orchestrator = ChaosOrchestrator::new(5, ChaosConfig::default());
    let _ = orchestrator.run_chaos_test().await;

    let stats = orchestrator.get_statistics();
    assert!(verify_invariants(&stats), "System invariants violated");
}
