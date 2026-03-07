//! Byzantine fault-tolerant consensus integration
//!
//! This module integrates BFT consensus with the OxiRS cluster,
//! providing a Byzantine-tolerant alternative to Raft consensus.

#[cfg(feature = "bft")]
use crate::bft::{BftConfig, BftConsensus, BftMessage};
#[cfg(feature = "bft")]
use crate::bft_network::BftNetworkService;
use crate::network::{NetworkConfig, NetworkService};
use crate::raft::{RdfCommand, RdfResponse};
use crate::storage::StorageBackend;
use crate::{ClusterError, Result};
use ed25519_dalek::VerifyingKey;
#[allow(unused_imports)]
use scirs2_core::random::rng; // Used in tests
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// BFT consensus state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BftState {
    /// Node is following
    Follower,
    /// Node is running and participating
    Running,
    /// Node is stopped
    Stopped,
}

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
    #[allow(dead_code)]
    storage: Arc<dyn StorageBackend>,
    /// Node status
    status: Arc<RwLock<BftState>>,
    /// Peer information
    peers: Arc<RwLock<HashMap<String, PeerInfo>>>,
}

/// Peer information for BFT consensus
#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub node_id: String,
    pub public_key: VerifyingKey,
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
            status: Arc::new(RwLock::new(BftState::Follower)),
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
        *status = BftState::Running;

        Ok(())
    }

    /// Stop the BFT consensus manager
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping BFT consensus manager for node {}", self.node_id);

        // Update status
        let mut status = self.status.write().await;
        *status = BftState::Stopped;

        Ok(())
    }

    /// Register a peer node
    pub async fn register_peer(
        &self,
        node_id: String,
        public_key: VerifyingKey,
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
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs(),
            signature: None,
        };

        // Process through consensus
        self.consensus.process_request(request)?;

        // Wait for consensus to complete
        // In a full implementation, this would:
        // 1. Track the request with a unique ID
        // 2. Wait on a channel/future for the consensus result
        // 3. Return the actual execution result from the replicated state machine
        // 4. Handle timeouts and consensus failures
        //
        // For now, we return success immediately after submitting to consensus.
        // The actual result would come from a reply channel that the consensus
        // engine would signal when 2f+1 nodes have committed the operation.

        // Simulate a small delay for consensus to complete
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // In production, check if consensus was achieved:
        // - If the operation received 2f+1 commit messages, return the result
        // - If timeout occurred, return timeout error
        // - If view change happened, retry or return error

        Ok(RdfResponse::Success)
    }

    /// Get current consensus status
    pub async fn get_status(&self) -> BftState {
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

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use scirs2_core::random::RngCore;

    #[test]
    fn test_peer_info() {
        let mut rng = rng();
        let mut seed_bytes = [0u8; 32];
        rng.fill_bytes(&mut seed_bytes);
        let keypair = SigningKey::from_bytes(&seed_bytes);

        let peer = PeerInfo {
            node_id: "node1".to_string(),
            public_key: keypair.verifying_key(),
            address: "127.0.0.1:8080".to_string(),
            is_active: true,
        };

        assert_eq!(peer.node_id, "node1");
        assert_eq!(peer.address, "127.0.0.1:8080");
        assert!(peer.is_active);
    }
}
