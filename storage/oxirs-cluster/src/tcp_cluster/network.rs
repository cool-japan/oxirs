//! Manages a mesh of [`TcpClusterNode`] instances for test scenarios.
//!
//! `TcpClusterNetwork::spawn` creates `n` nodes bound to `127.0.0.1:0`,
//! reads each node's OS-assigned address via `TcpClusterNode::addr()`, and
//! wires every node as a peer of every other.  This full-mesh wiring ensures
//! convergence even with `Bounded(1)` or `Sqrt` fanout policies.
//!
//! After construction, tests call `set_on`, `get_on`, and `wait_converged`
//! to drive and observe the gossip protocol.

use std::time::{Duration, Instant};

use tokio::time::sleep;

use crate::gossip::fanout::GossipFanout;

use super::node::{TcpClusterNode, TcpNodeConfig, TcpNodeError};

// ─────────────────────────────────────────────────────────────────────────────
// NetworkStats
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome of a [`TcpClusterNetwork::wait_converged`] call.
#[derive(Debug, Clone)]
pub struct NetworkStats {
    /// Total number of nodes in the network.
    pub node_count: usize,
    /// Whether all nodes had the expected value before the deadline.
    pub converged: bool,
    /// How many 10 ms polling rounds elapsed before convergence (or deadline).
    pub rounds_to_converge: usize,
    /// Wall-clock time spent waiting (milliseconds).
    pub total_time_ms: u64,
    /// Number of distinct keys in node-0's state after the wait.
    pub final_state_entries: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// TcpClusterNetwork
// ─────────────────────────────────────────────────────────────────────────────

/// An N-node full-mesh network of real-socket cluster nodes.
///
/// Suitable for integration tests that need to exercise the gossip and
/// replication paths over actual TCP.
pub struct TcpClusterNetwork {
    nodes: Vec<TcpClusterNode>,
}

impl TcpClusterNetwork {
    /// Spawn `n` nodes and wire them into a full peer mesh.
    ///
    /// Each node binds to `127.0.0.1:0`; the OS assigns a free ephemeral
    /// port.  After all nodes are started their actual addresses are
    /// collected and registered as peers with every other node.
    ///
    /// # Errors
    ///
    /// Returns an error if any node fails to bind its listener.
    pub async fn spawn(
        n: usize,
        _base_port: u16,
        fanout: GossipFanout,
        gossip_interval_ms: u64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // _base_port is accepted for API compatibility but ignored; we use
        // port 0 (OS-assigned) throughout to guarantee no port conflicts.
        let mut nodes = Vec::with_capacity(n);
        for i in 0..n {
            let cfg = TcpNodeConfig {
                node_id: format!("node-{i}"),
                bind_addr: "127.0.0.1:0".parse()?,
                fanout,
                gossip_interval_ms,
            };
            let node = TcpClusterNode::start(cfg)
                .await
                .map_err(|e: TcpNodeError| Box::new(e) as Box<dyn std::error::Error>)?;
            nodes.push(node);
        }

        // Collect all bound addresses.
        let addrs: Vec<_> = nodes.iter().map(|n| n.addr()).collect();

        // Wire full-mesh peers: each node gets every other node as a peer.
        for (i, node) in nodes.iter().enumerate() {
            for (j, &addr) in addrs.iter().enumerate() {
                if i != j {
                    node.add_peer(addr);
                }
            }
        }

        Ok(Self { nodes })
    }

    /// Write `key = value` on the node at position `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= node_count()`.
    pub fn set_on(&self, idx: usize, key: &str, value: u64) {
        self.nodes[idx].set(key, value);
    }

    /// Write `key = value` with an explicit version on the node at `idx`.
    ///
    /// Useful for deterministic LWW tests where exact version ordering must
    /// be enforced regardless of wall-clock timing.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= node_count()`.
    pub fn set_with_version_on(&self, idx: usize, key: &str, value: u64, version: u64) {
        self.nodes[idx].set_with_version(key, value, version);
    }

    /// Read `key` from the node at position `idx`.
    ///
    /// Returns `None` if the key has not been seen on that node yet.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= node_count()`.
    pub fn get_on(&self, idx: usize, key: &str) -> Option<u64> {
        self.nodes[idx].get(key)
    }

    /// Poll every 10 ms until all nodes report `key == expected_value` or
    /// `max_wait` elapses.
    ///
    /// Returns [`NetworkStats`] regardless of whether convergence was
    /// achieved; check [`NetworkStats::converged`] to distinguish the two
    /// outcomes.
    pub async fn wait_converged(
        &self,
        key: &str,
        expected_value: u64,
        max_wait: Duration,
    ) -> NetworkStats {
        let start = Instant::now();
        let mut rounds = 0usize;

        loop {
            let all_match = self
                .nodes
                .iter()
                .all(|n| n.get(key) == Some(expected_value));

            if all_match {
                let elapsed = start.elapsed();
                return NetworkStats {
                    node_count: self.nodes.len(),
                    converged: true,
                    rounds_to_converge: rounds,
                    total_time_ms: elapsed.as_millis() as u64,
                    final_state_entries: self.nodes.first().map_or(0, |n| n.state_len()),
                };
            }

            if start.elapsed() >= max_wait {
                let elapsed = start.elapsed();
                return NetworkStats {
                    node_count: self.nodes.len(),
                    converged: false,
                    rounds_to_converge: rounds,
                    total_time_ms: elapsed.as_millis() as u64,
                    final_state_entries: self.nodes.first().map_or(0, |n| n.state_len()),
                };
            }

            sleep(Duration::from_millis(10)).await;
            rounds += 1;
        }
    }

    /// Shut down all nodes.
    ///
    /// Cancels each node's background tasks.  Does not block waiting for
    /// task completion.
    pub fn shutdown_all(&self) {
        for node in &self.nodes {
            node.shutdown();
        }
    }

    /// Number of nodes in this network.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_spawn_and_shutdown() {
        let net = TcpClusterNetwork::spawn(3, 0, GossipFanout::Unbounded, 50)
            .await
            .expect("spawn");
        assert_eq!(net.node_count(), 3);
        net.shutdown_all();
    }

    #[tokio::test]
    async fn test_set_and_get_on() {
        let net = TcpClusterNetwork::spawn(2, 0, GossipFanout::Unbounded, 50)
            .await
            .expect("spawn");
        net.set_on(0, "x", 77);
        assert_eq!(net.get_on(0, "x"), Some(77));
        net.shutdown_all();
    }
}
