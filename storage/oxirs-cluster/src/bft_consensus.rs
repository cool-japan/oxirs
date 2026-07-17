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
use std::time::Duration;
use tokio::sync::{oneshot, RwLock};
use tracing::info;

/// Maximum time to wait for a BFT request to reach a 2f+1 commit quorum before
/// surfacing a timeout error to the caller.
const BFT_COMMIT_TIMEOUT: Duration = Duration::from_secs(10);

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
    /// In-flight client requests awaiting a 2f+1 commit quorum, keyed by request
    /// id. The completion channel is fired by [`notify_request_committed`] only
    /// when the BFT engine has observed a real matching-commit quorum.
    pending: Arc<RwLock<HashMap<String, oneshot::Sender<RdfResponse>>>>,
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
            pending: Arc::new(RwLock::new(HashMap::new())),
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

    /// Process a client request through BFT consensus.
    ///
    /// The request is submitted to the consensus engine and then this call
    /// blocks until the engine confirms a real 2f+1 matching-commit quorum
    /// (signaled via [`BftConsensusManager::notify_request_committed`]) or the
    /// commit timeout elapses. It never returns a fabricated success after a
    /// fixed sleep: without a confirmed quorum the caller receives an explicit
    /// timeout error.
    pub async fn process_request(&self, command: RdfCommand) -> Result<RdfResponse> {
        // Serialize command
        let operation =
            serde_json::to_vec(&command).map_err(|e| ClusterError::Serialize(e.to_string()))?;

        // Assign a unique id and register a completion waiter *before* submitting
        // so we cannot miss an early commit signal.
        let request_id = uuid::Uuid::new_v4().to_string();
        let (tx, rx) = oneshot::channel::<RdfResponse>();
        {
            let mut pending = self.pending.write().await;
            pending.insert(request_id.clone(), tx);
        }

        // Create BFT request
        let request = BftMessage::Request {
            client_id: self.node_id.clone(),
            operation,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| ClusterError::Runtime(format!("system clock before epoch: {e}")))?
                .as_secs(),
            signature: None,
        };

        // Submit to the consensus engine; roll back the waiter on failure.
        if let Err(e) = self.consensus.process_request(request) {
            self.pending.write().await.remove(&request_id);
            return Err(e);
        }

        // Wait for a real 2f+1 commit signal, or fail loudly on timeout. The
        // signal is fired by notify_request_committed() only after the engine
        // has observed a matching-commit quorum for this request.
        match tokio::time::timeout(BFT_COMMIT_TIMEOUT, rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_canceled)) => {
                self.pending.write().await.remove(&request_id);
                Err(ClusterError::Consensus(format!(
                    "BFT completion channel for request {request_id} was dropped before a \
                     2f+1 commit quorum was reached"
                )))
            }
            Err(_elapsed) => {
                self.pending.write().await.remove(&request_id);
                Err(ClusterError::Consensus(format!(
                    "BFT consensus did not reach a 2f+1 commit quorum for request \
                     {request_id} within {BFT_COMMIT_TIMEOUT:?}"
                )))
            }
        }
    }

    /// Signal that the BFT engine has observed a real 2f+1 matching-commit
    /// quorum for `request_id`, delivering the replicated execution result to
    /// the waiting [`process_request`] call.
    ///
    /// Returns `true` if a waiter was found and notified, `false` otherwise
    /// (e.g. the request already timed out). This is the hook the BFT
    /// commit/reply path calls once `is_committed` holds for the request.
    pub async fn notify_request_committed(&self, request_id: &str, response: RdfResponse) -> bool {
        let sender = {
            let mut pending = self.pending.write().await;
            pending.remove(request_id)
        };
        match sender {
            Some(tx) => tx.send(response).is_ok(),
            None => false,
        }
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
    use scirs2_core::RngExt;

    #[cfg(feature = "bft")]
    #[tokio::test]
    async fn test_notify_unknown_request_returns_false() {
        use crate::network::NetworkConfig;
        use crate::storage::mock::MockStorageBackend;

        let storage = Arc::new(MockStorageBackend::new());
        let mgr =
            BftConsensusManager::new("1".to_string(), vec![], storage, NetworkConfig::default())
                .await
                .expect("failed to build BFT consensus manager");

        // No request with this id is pending, so the completion hook must report
        // that nothing was notified (it never fabricates a commit).
        assert!(
            !mgr.notify_request_committed("no-such-request", RdfResponse::Success)
                .await
        );
    }

    #[test]
    fn test_peer_info() {
        let mut rng = rng();
        let seed_bytes: [u8; 32] = std::array::from_fn(|_| rng.random_range(0..256) as u8);
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
