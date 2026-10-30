//! # RegionManager - accessors Methods
//!
//! This module contains method implementations for `RegionManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result as ClusterResult;
use crate::raft::OxirsNodeId;
use anyhow::Result;
use std::net::SocketAddr;

use super::regionmanager_type::RegionManager;

impl RegionManager {
    /// Record the network address of a node, used to measure the latency to
    /// its region. `ClusterNode` registers its own address and those of its
    /// configured peers.
    pub async fn register_node_address(&self, node_id: OxirsNodeId, address: SocketAddr) {
        self.node_addresses.write().await.insert(node_id, address);
    }

    /// Get the registered network address of a node, if any
    pub(crate) async fn get_node_address(
        &self,
        node_id: OxirsNodeId,
    ) -> Result<Option<SocketAddr>> {
        Ok(self.node_addresses.read().await.get(&node_id).copied())
    }

    /// Send data to a specific node
    ///
    /// There is no cross-region data transport: no node runs a receiver for
    /// replicated data, so nothing is sent and an error is returned. Earlier
    /// releases wrote the data in plaintext to a made-up local port and
    /// counted any accepted connection as delivered.
    pub(super) async fn send_data_to_node(
        &self,
        data: &[u8],
        node_id: OxirsNodeId,
    ) -> Result<Option<Vec<u8>>> {
        Err(anyhow::anyhow!(
            "cross-region data transport is not implemented: no receiver exists, so {} \
             byte(s) were not sent to node {node_id}",
            data.len()
        ))
    }
    /// Ping a node by ID
    pub(super) async fn ping_node_by_id(&self, node_id: OxirsNodeId) -> ClusterResult<()> {
        let addr = self
            .get_node_address(node_id)
            .await
            .map_err(|e| crate::error::ClusterError::Network(e.to_string()))?
            .ok_or_else(|| {
                crate::error::ClusterError::Network(format!("No address for node {}", node_id))
            })?;
        self.ping_node(addr)
            .await
            .map_err(|e| crate::error::ClusterError::Network(e.to_string()))
    }
}
