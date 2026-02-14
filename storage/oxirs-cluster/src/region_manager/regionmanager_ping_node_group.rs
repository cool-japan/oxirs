//! # RegionManager - ping_node_group Methods
//!
//! This module contains method implementations for `RegionManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use std::net::SocketAddr;

use super::regionmanager_type::RegionManager;

impl RegionManager {
    /// Ping a node to measure network latency
    pub(crate) async fn ping_node(&self, addr: SocketAddr) -> Result<()> {
        use tokio::net::TcpStream;
        match TcpStream::connect(addr).await {
            Ok(_) => Ok(()),
            Err(e) => Err(anyhow::anyhow!("Failed to connect to {}: {}", addr, e)),
        }
    }
}
