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
    /// Get the network address for a specific node
    pub(crate) async fn get_node_address(
        &self,
        node_id: OxirsNodeId,
    ) -> Result<Option<SocketAddr>> {
        let base_port = 8080;
        let addr = format!("127.0.0.1:{}", base_port + node_id as u16)
            .parse::<SocketAddr>()
            .ok();
        Ok(addr)
    }
    /// Send data to a specific node
    pub(super) async fn send_data_to_node(
        &self,
        data: &[u8],
        node_id: OxirsNodeId,
    ) -> Result<Option<Vec<u8>>> {
        let node_addr = self
            .get_node_address(node_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("No address found for node {}", node_id))?;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpStream;
        let mut stream = TcpStream::connect(node_addr).await?;
        stream.write_u32(data.len() as u32).await?;
        stream.write_all(data).await?;
        stream.flush().await?;
        let mut response_len = [0u8; 4];
        if stream.read_exact(&mut response_len).await.is_ok() {
            let len = u32::from_be_bytes(response_len) as usize;
            if len > 0 && len < 1024 * 1024 {
                let mut response = vec![0u8; len];
                stream.read_exact(&mut response).await?;
                return Ok(Some(response));
            }
        }
        Ok(None)
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
