//! In-memory cluster simulation for scalability testing.
//!
//! `SimCluster` runs N virtual nodes in a single tokio runtime with bounded
//! mailbox channels.  No real network sockets are created; all inter-node
//! communication is through `tokio::sync::mpsc` queues.
//!
//! This module is available under `#[cfg(any(test, feature = "simulation"))]`
//! to avoid adding compile-time weight to production builds.
//!
//! # Example
//!
//! ```rust
//! # #[cfg(feature = "simulation")]
//! # {
//! use oxirs_cluster::simulation::SimCluster;
//!
//! let cluster = SimCluster::new(10);
//! assert_eq!(cluster.size(), 10);
//! # }
//! ```

pub mod scaling_bench;

use std::collections::HashMap;
use tokio::sync::mpsc;

/// Messages exchanged between simulated nodes.
#[derive(Debug, Clone)]
pub enum SimMessage {
    /// A gossip payload from one node to another.
    Gossip {
        /// Sender node ID.
        from: u64,
        /// Arbitrary payload bytes.
        data: Vec<u8>,
    },
    /// Heartbeat ping from `from` to its recipient.
    Ping(u64),
    /// Heartbeat pong reply.
    Pong(u64),
}

/// A single simulated cluster node with inbox and named outboxes.
pub struct SimNode {
    /// Unique identifier for this node.
    pub id: u64,
    /// Receiving end of this node's mailbox.
    pub inbox: mpsc::Receiver<SimMessage>,
    /// Sending ends to all *other* nodes (including self, for simplicity).
    pub outboxes: HashMap<u64, mpsc::Sender<SimMessage>>,
}

/// An in-memory cluster of N simulated nodes with bounded mailboxes.
///
/// Construction is O(N) time and O(N²) channel handles (each node holds a
/// sender to every other node, which is acceptable for simulation use up to
/// a few thousand nodes).
pub struct SimCluster {
    /// All simulated nodes.
    nodes: Vec<SimNode>,
    /// Shared sending handles keyed by node ID (for `gossip` helper).
    senders: HashMap<u64, mpsc::Sender<SimMessage>>,
}

impl SimCluster {
    /// Mailbox capacity per node.
    const MAILBOX_CAPACITY: usize = 256;

    /// Create a cluster of `n` simulated nodes with bounded mailboxes.
    ///
    /// Each node receives its own `mpsc::Receiver` and a clone of every
    /// sender, so any node can message any other without coordination.
    pub fn new(n: usize) -> Self {
        // Build sender+receiver pairs for every node.
        let mut senders: HashMap<u64, mpsc::Sender<SimMessage>> = HashMap::with_capacity(n);
        let mut receivers: HashMap<u64, mpsc::Receiver<SimMessage>> = HashMap::with_capacity(n);

        for i in 0..n as u64 {
            let (tx, rx) = mpsc::channel::<SimMessage>(Self::MAILBOX_CAPACITY);
            senders.insert(i, tx);
            receivers.insert(i, rx);
        }

        // Build SimNode instances; each node clones the full sender map.
        // `receivers.remove` is infallible here because we just inserted each key.
        let nodes: Vec<SimNode> = (0..n as u64)
            .map(|i| {
                let outboxes = senders.clone();
                let inbox = receivers
                    .remove(&i)
                    .unwrap_or_else(|| unreachable!("receiver for node {i} was just inserted"));
                SimNode {
                    id: i,
                    inbox,
                    outboxes,
                }
            })
            .collect();

        SimCluster { nodes, senders }
    }

    /// Number of nodes in the cluster.
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Send a gossip message from `from` to `to`.
    ///
    /// Returns `Err` if `to` is unknown or the mailbox is full.
    pub async fn gossip(&self, from: u64, to: u64, data: Vec<u8>) -> Result<(), String> {
        let sender = self
            .senders
            .get(&to)
            .ok_or_else(|| format!("no node with id {to}"))?;
        sender
            .send(SimMessage::Gossip { from, data })
            .await
            .map_err(|e| e.to_string())
    }

    /// Send a ping from `from` to `to`.
    ///
    /// Returns `Err` if `to` is unknown or the mailbox is full.
    pub async fn ping(&self, from: u64, to: u64) -> Result<(), String> {
        let sender = self
            .senders
            .get(&to)
            .ok_or_else(|| format!("no node with id {to}"))?;
        sender
            .send(SimMessage::Ping(from))
            .await
            .map_err(|e| e.to_string())
    }

    /// Immutable borrow of the node list.
    pub fn nodes(&self) -> &[SimNode] {
        &self.nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sim_cluster_empty() {
        let c = SimCluster::new(0);
        assert_eq!(c.size(), 0);
    }

    #[test]
    fn test_sim_cluster_single_node() {
        let c = SimCluster::new(1);
        assert_eq!(c.size(), 1);
    }

    #[test]
    fn test_sim_cluster_10_nodes() {
        let c = SimCluster::new(10);
        assert_eq!(c.size(), 10);
    }

    #[test]
    fn test_sim_cluster_1000_nodes() {
        let c = SimCluster::new(1000);
        assert_eq!(c.size(), 1000);
    }

    #[tokio::test]
    async fn test_gossip_delivery() {
        let cluster = SimCluster::new(3);
        let result = cluster.gossip(0, 1, b"hello".to_vec()).await;
        assert!(result.is_ok(), "gossip delivery should succeed");
    }

    #[tokio::test]
    async fn test_gossip_unknown_target() {
        let cluster = SimCluster::new(3);
        let result = cluster.gossip(0, 999, b"hello".to_vec()).await;
        assert!(result.is_err(), "gossip to unknown node should fail");
    }

    #[tokio::test]
    async fn test_ping_delivery() {
        let cluster = SimCluster::new(2);
        let result = cluster.ping(0, 1).await;
        assert!(result.is_ok());
    }
}
