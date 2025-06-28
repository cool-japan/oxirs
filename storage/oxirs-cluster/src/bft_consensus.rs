//! Byzantine fault-tolerant consensus integration
//!
//! This module integrates BFT consensus with the OxiRS cluster,
//! providing a Byzantine-tolerant alternative to Raft consensus.

#[cfg(feature = "bft")]
use crate::bft::{BftConfig, BftConsensus, BftMessage};
#[cfg(feature = "bft")]
use crate::bft_network::BftNetworkService;
use crate::consensus::{ConsensusManager, ConsensusStatus};
use crate::network::{NetworkConfig, NetworkService};
use crate::raft::{OxirsNodeId, RdfCommand, RdfResponse};
use crate::storage::StorageBackend;
use crate::{ClusterError, Result};
use ed25519_dalek::{Keypair, PublicKey};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// BFT consensus manager for Byzantine-tolerant clusters
#[cfg(feature = "bft")]
pub struct BftConsensusManager {
    /// Node identifier
    node_id: String,
    /// BFT consensus engine
    consensus: Arc<BftConsensus>,
    /// BFT network service
    network: Arc<BftNetworkService>,
    /// Storage backend
    storage: Arc<dyn StorageBackend>,
    /// Node status
    status: Arc<RwLock<ConsensusStatus>>,
    /// Peer information
    peers: Arc<RwLock<HashMap<String, PeerInfo>>>,
}

/// Peer information for BFT consensus
#[derive(Debug, Clone)]
struct PeerInfo {
    pub node_id: String,
    pub public_key: PublicKey,
    pub address: String,
    pub is_active: bool,
}

#[cfg(feature = "bft")]
impl BftConsensusManager {
    /// Create a new BFT consensus manager
    pub async fn new(
        node_id: String,
        peers: Vec<String>,
        storage: Arc<dyn StorageBackend>,
        network_config: NetworkConfig,
    ) -> Result<Self> {
        // Create BFT configuration based on cluster size
        let num_nodes = peers.len() + 1; // Include self
        let bft_config = BftConfig::new(num_nodes);

        // Create BFT consensus engine
        let consensus = Arc::new(BftConsensus::new(node_id.clone(), bft_config)?);

        // Create network service
        let network_service = Arc::new(NetworkService::new(
            node_id
                .parse()
                .map_err(|_| ClusterError::Config("Invalid node ID".to_string()))?,
            network_config,
        ));

        // Create BFT network service
        let bft_network = Arc::new(BftNetworkService::new(
            node_id.clone(),
            consensus.clone(),
            network_service,
        ));

        Ok(BftConsensusManager {
            node_id,
            consensus,
            network: bft_network,
            storage,
            status: Arc::new(RwLock::new(ConsensusStatus::Follower)),
            peers: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Start the BFT consensus manager
    pub async fn start(&self) -> Result<()> {
        info!("Starting BFT consensus manager for node {}", self.node_id);

        // Start network service
        self.network.clone().start().await?;

        // Start consensus timers
        self.consensus.start_view_timer()?;

        // Update status
        let mut status = self.status.write().await;
        *status = ConsensusStatus::Running;

        Ok(())
    }

    /// Stop the BFT consensus manager
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping BFT consensus manager for node {}", self.node_id);

        // Update status
        let mut status = self.status.write().await;
        *status = ConsensusStatus::Stopped;

        Ok(())
    }

    /// Register a peer node
    pub async fn register_peer(
        &self,
        node_id: String,
        public_key: PublicKey,
        address: String,
    ) -> Result<()> {
        // Register with consensus engine
        self.consensus.register_node(node_id.clone(), public_key)?;

        // Register with network service
        self.network
            .register_peer(node_id.clone(), public_key)
            .await?;

        // Store peer info
        let mut peers = self.peers.write().await;
        peers.insert(
            node_id.clone(),
            PeerInfo {
                node_id,
                public_key,
                address,
                is_active: true,
            },
        );

        Ok(())
    }

    /// Process a client request through BFT consensus
    pub async fn process_request(&self, command: RdfCommand) -> Result<RdfResponse> {
        // Serialize command
        let operation =
            serde_json::to_vec(&command).map_err(|e| ClusterError::Serialize(e.to_string()))?;

        // Create BFT request
        let request = BftMessage::Request {
            client_id: self.node_id.clone(),
            operation,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            signature: None,
        };

        // Process through consensus
        self.consensus.process_request(request)?;

        // TODO: Wait for consensus and return result
        // For now, return success
        Ok(RdfResponse::Success)
    }

    /// Get current consensus status
    pub async fn get_status(&self) -> ConsensusStatus {
        let status = self.status.read().await;
        *status
    }

    /// Check if this node is the primary
    pub fn is_primary(&self) -> Result<bool> {
        self.consensus.is_primary()
    }

    /// Get current view number
    pub fn current_view(&self) -> Result<u64> {
        self.consensus.current_view()
    }

    /// Get peer information
    pub async fn get_peers(&self) -> Vec<PeerInfo> {
        let peers = self.peers.read().await;
        peers.values().cloned().collect()
    }
}

/// Consensus manager trait implementation for BFT
#[cfg(feature = "bft")]
impl ConsensusManager for BftConsensusManager {
    fn start(&mut self) -> Result<()> {
        tokio::runtime::Runtime::new()
            .map_err(|e| ClusterError::Runtime(e.to_string()))?
            .block_on(self.start())
    }

    fn stop(&mut self) -> Result<()> {
        tokio::runtime::Runtime::new()
            .map_err(|e| ClusterError::Runtime(e.to_string()))?
            .block_on(self.stop())
    }

    fn is_leader(&self) -> bool {
        self.is_primary().unwrap_or(false)
    }

    fn get_status(&self) -> ConsensusStatus {
        tokio::runtime::Runtime::new()
            .map(|rt| rt.block_on(self.get_status()))
            .unwrap_or(ConsensusStatus::Unknown)
    }

    fn process_command(&mut self, command: RdfCommand) -> Result<RdfResponse> {
        tokio::runtime::Runtime::new()
            .map_err(|e| ClusterError::Runtime(e.to_string()))?
            .block_on(self.process_request(command))
    }
}

/// Factory function to create appropriate consensus manager
pub fn create_consensus_manager(
    node_id: OxirsNodeId,
    peers: Vec<OxirsNodeId>,
    storage: Arc<dyn StorageBackend>,
    use_bft: bool,
) -> Result<Box<dyn ConsensusManager>> {
    if use_bft {
        #[cfg(feature = "bft")]
        {
            let bft_manager = tokio::runtime::Runtime::new()
                .map_err(|e| ClusterError::Runtime(e.to_string()))?
                .block_on(BftConsensusManager::new(
                    node_id.to_string(),
                    peers.iter().map(|p| p.to_string()).collect(),
                    storage,
                    NetworkConfig::default(),
                ))?;
            Ok(Box::new(bft_manager))
        }
        #[cfg(not(feature = "bft"))]
        {
            Err(ClusterError::Config("BFT feature not enabled".to_string()))
        }
    } else {
        // Create standard Raft consensus manager
        Ok(Box::new(crate::consensus::ConsensusManager::new(
            node_id, peers,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peer_info() {
        use rand::rngs::OsRng;
        let mut csprng = OsRng {};
        let keypair = Keypair::generate(&mut csprng);

        let peer = PeerInfo {
            node_id: "node1".to_string(),
            public_key: keypair.public,
            address: "127.0.0.1:8080".to_string(),
            is_active: true,
        };

        assert_eq!(peer.node_id, "node1");
        assert_eq!(peer.address, "127.0.0.1:8080");
        assert!(peer.is_active);
    }
}
