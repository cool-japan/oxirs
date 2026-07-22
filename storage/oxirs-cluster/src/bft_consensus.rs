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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{oneshot, RwLock};
use tracing::info;

/// Default time to wait for a BFT request to reach a 2f+1 commit quorum before
/// surfacing a timeout error to the caller. Overridable per manager via
/// [`BftConsensusManager::with_commit_timeout`] so tests can shorten it.
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
    /// Node status
    status: Arc<RwLock<BftState>>,
    /// Peer information
    peers: Arc<RwLock<HashMap<String, PeerInfo>>>,
    /// In-flight client requests awaiting a 2f+1 commit quorum, keyed by the
    /// PBFT request identity `(client_id, timestamp)` that the engine echoes
    /// back through its commit callback. A std `Mutex` (not a tokio lock) is
    /// used because the callback resolves waiters from the synchronous PBFT
    /// commit path, where awaiting a tokio lock is not possible.
    pending: Arc<Mutex<HashMap<(String, u64), oneshot::Sender<RdfResponse>>>>,
    /// Monotonic per-request identity source. Combined with `node_id` as the
    /// client id it yields a unique `(client_id, timestamp)` key per request,
    /// so requests issued within the same wall-clock second never collide.
    request_seq: Arc<AtomicU64>,
    /// Maximum time [`BftConsensusManager::process_request`] waits for a 2f+1
    /// commit quorum before returning a timeout error.
    commit_timeout: Duration,
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

        // Create BFT consensus engine bound to the real state-machine backend so
        // committed operations are actually applied (never fabricated).
        let consensus = Arc::new(BftConsensus::with_storage(
            node_id.clone(),
            bft_config,
            storage,
        )?);

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

        let pending: Arc<Mutex<HashMap<(String, u64), oneshot::Sender<RdfResponse>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        // Register the commit callback: the engine fires it from the PBFT commit
        // path once a real 2f+1 matching-commit quorum is observed, carrying the
        // original request's `(client_id, timestamp)` and the serialized
        // `RdfResponse`. Resolve the matching pending waiter; a redelivery with
        // no waiter (already resolved/timed out) is silently ignored.
        {
            let pending_cb = Arc::clone(&pending);
            consensus.set_commit_callback(move |client_id, timestamp, result| {
                let response = match serde_json::from_slice::<RdfResponse>(&result) {
                    Ok(response) => response,
                    Err(e) => RdfResponse::Error(format!(
                        "BFT commit result could not be decoded as an RdfResponse: {e}"
                    )),
                };
                if let Ok(mut guard) = pending_cb.lock() {
                    if let Some(tx) = guard.remove(&(client_id, timestamp)) {
                        // Receiver may already be gone if the caller timed out;
                        // that is fine and requires no action.
                        let _ = tx.send(response);
                    }
                }
            });
        }

        // Wire the broadcast bridge: a sync-callable unbounded sender feeds a
        // spawned task that performs the authenticated network broadcast. The
        // post-construction setter resolves the consensus<->network construction
        // cycle (the network service already borrowed the consensus Arc).
        {
            let (bcast_tx, mut bcast_rx) = tokio::sync::mpsc::unbounded_channel::<BftMessage>();
            consensus.set_broadcaster(bcast_tx);
            let net = bft_network.clone();
            tokio::spawn(async move {
                while let Some(message) = bcast_rx.recv().await {
                    if let Err(e) = net.broadcast(message).await {
                        tracing::warn!("BFT authenticated broadcast failed: {e}");
                    }
                }
            });
        }

        // Wire the client-reply bridge: committed `Reply` messages are drained
        // and delivered to the requesting client over the authenticated BFT
        // network transport. This is what lets a *remote* BFT client actually
        // receive replies (previously the reply never left the node, so
        // cross-node clients timed out). A reply addressed to this node's own
        // id was already completed in-process by the commit callback above, so
        // it is not re-sent over the network.
        {
            let (reply_tx, mut reply_rx) = tokio::sync::mpsc::unbounded_channel::<BftMessage>();
            consensus.set_reply_sink(reply_tx);
            let net = bft_network.clone();
            let self_id = node_id.clone();
            tokio::spawn(async move {
                while let Some(message) = reply_rx.recv().await {
                    if let BftMessage::Reply { client_id, .. } = &message {
                        if client_id == &self_id {
                            continue;
                        }
                        if let Err(e) = net.send_to(client_id, message.clone()).await {
                            tracing::warn!("BFT reply delivery to client {client_id} failed: {e}");
                        }
                    }
                }
            });
        }

        // Seed the request-identity counter from the current time (nanoseconds)
        // so identities are unlikely to collide across manager restarts.
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| ClusterError::Runtime(format!("system clock before epoch: {e}")))?
            .as_nanos() as u64;

        Ok(BftConsensusManager {
            node_id,
            consensus,
            network: bft_network,
            status: Arc::new(RwLock::new(BftState::Follower)),
            peers: Arc::new(RwLock::new(HashMap::new())),
            pending,
            request_seq: Arc::new(AtomicU64::new(seed)),
            commit_timeout: BFT_COMMIT_TIMEOUT,
        })
    }

    /// Override the commit-quorum wait timeout (default 10s).
    ///
    /// Primarily used by tests to shorten the wait so a failure to reach quorum
    /// surfaces quickly instead of after the full production timeout.
    pub fn with_commit_timeout(mut self, timeout: Duration) -> Self {
        self.commit_timeout = timeout;
        self
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
    /// (the registered commit callback resolves the matching waiter) or the
    /// commit timeout elapses. It never returns a fabricated success after a
    /// fixed sleep: without a confirmed quorum the caller receives an explicit
    /// timeout error.
    pub async fn process_request(&self, command: RdfCommand) -> Result<RdfResponse> {
        // Serialize command
        let operation =
            serde_json::to_vec(&command).map_err(|e| ClusterError::Serialize(e.to_string()))?;

        // Derive the PBFT request identity. The client id is this node's id; the
        // timestamp is a monotonic counter so `(client_id, timestamp)` is unique
        // and matches exactly what the engine echoes back through the commit
        // callback. Register the completion waiter *before* submitting so an
        // early commit signal cannot be missed.
        let client_id = self.node_id.clone();
        let timestamp = self.request_seq.fetch_add(1, Ordering::SeqCst);
        let key = (client_id.clone(), timestamp);

        let (tx, rx) = oneshot::channel::<RdfResponse>();
        {
            let mut pending = self
                .pending
                .lock()
                .map_err(|e| ClusterError::Lock(e.to_string()))?;
            pending.insert(key.clone(), tx);
        }

        // Create BFT request
        let request = BftMessage::Request {
            client_id,
            operation,
            timestamp,
            signature: None,
        };

        // Submit to the consensus engine; roll back the waiter on failure.
        if let Err(e) = self.consensus.process_request(request) {
            self.remove_pending(&key);
            return Err(e);
        }

        // Wait for a real 2f+1 commit signal, or fail loudly on timeout. The
        // waiter is resolved by the registered commit callback only after the
        // engine has observed a matching-commit quorum for this request.
        match tokio::time::timeout(self.commit_timeout, rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_canceled)) => {
                self.remove_pending(&key);
                Err(ClusterError::Consensus(format!(
                    "BFT completion channel for request {key:?} was dropped before a \
                     2f+1 commit quorum was reached"
                )))
            }
            Err(_elapsed) => {
                self.remove_pending(&key);
                Err(ClusterError::Consensus(format!(
                    "BFT consensus did not reach a 2f+1 commit quorum for request \
                     {key:?} within {:?}",
                    self.commit_timeout
                )))
            }
        }
    }

    /// Remove a pending waiter, ignoring a poisoned lock (nothing to wait on if
    /// the map is unusable).
    fn remove_pending(&self, key: &(String, u64)) {
        if let Ok(mut pending) = self.pending.lock() {
            pending.remove(key);
        }
    }

    /// Manually resolve the waiter for a PBFT request identity `(client_id,
    /// timestamp)` with `response`.
    ///
    /// The registered commit callback already resolves waiters automatically on
    /// quorum; this remains available for out-of-band resolution (e.g. a reply
    /// redelivered over the network). Returns `true` if a waiter was found and
    /// notified, `false` otherwise (no such waiter / already resolved).
    pub fn notify_request_committed(
        &self,
        client_id: &str,
        timestamp: u64,
        response: RdfResponse,
    ) -> bool {
        let sender = match self.pending.lock() {
            Ok(mut pending) => pending.remove(&(client_id.to_string(), timestamp)),
            Err(_) => None,
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

        // No request with this identity is pending, so the manual hook must
        // report that nothing was notified (it never fabricates a commit).
        assert!(!mgr.notify_request_committed("no-such-client", 0, RdfResponse::Success));
    }

    /// The decisive closed-loop test: a manager backed by a real (mock) state
    /// machine must complete `process_request` as soon as a genuine 2f+1 commit
    /// quorum is driven through the engine — well before the timeout — and the
    /// committed operation must be visibly applied to the backend.
    #[cfg(feature = "bft")]
    #[tokio::test]
    async fn test_process_request_completes_on_commit_quorum_and_applies() {
        use crate::bft::{BftConsensus, BftMessage};
        use crate::network::NetworkConfig;
        use crate::storage::mock::MockStorageBackend;
        use crate::storage::StorageBackend;
        use ed25519_dalek::Signer;

        // Node "1" plus three peers → f=1, 2f+1 = 3 commits for quorum.
        let mock = Arc::new(MockStorageBackend::new());
        let storage: Arc<dyn StorageBackend> = mock.clone();
        let mgr = Arc::new(
            BftConsensusManager::new(
                "1".to_string(),
                vec!["2".to_string(), "3".to_string(), "4".to_string()],
                storage,
                NetworkConfig::default(),
            )
            .await
            .expect("failed to build BFT consensus manager")
            // Keep the timeout comfortably above the driving latency but far
            // below the 10s default so a regression fails fast.
            .with_commit_timeout(Duration::from_secs(5)),
        );

        // Register the three peers with the keys whose commit votes the primary
        // (node "1", primary at view 0) will trust.
        let mut remotes: Vec<(String, SigningKey)> = Vec::new();
        for id in ["2", "3", "4"] {
            let seed: [u8; 32] = rand::random();
            let signing_key = SigningKey::from_bytes(&seed);
            mgr.register_peer(
                id.to_string(),
                signing_key.verifying_key(),
                "127.0.0.1:0".to_string(),
            )
            .await
            .expect("register peer");
            remotes.push((id.to_string(), signing_key));
        }

        // Spawn the client request. It registers a waiter, submits the request
        // (storing a local pre-prepare at view=0, sequence=1) and then awaits a
        // real quorum.
        let command = RdfCommand::Insert {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "http://example.org/o".to_string(),
        };
        let operation = serde_json::to_vec(&command).expect("serialize command");
        let mgr_task = mgr.clone();
        let handle = tokio::spawn(async move { mgr_task.process_request(command).await });

        // Wait until the primary has recorded the pre-prepare before delivering
        // commit votes, so `handle_commit` can recover the request.
        let mut stored = false;
        for _ in 0..2000 {
            if mgr.consensus.has_pre_prepare(0, 1) {
                stored = true;
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(stored, "primary did not store the pre-prepare in time");

        // Drive the required 2f+1 = 3 commit votes through the engine.
        let digest = BftConsensus::create_digest(&operation);
        for (node_id, signing_key) in &remotes {
            let commit = BftMessage::Commit {
                view: 0,
                sequence: 1,
                digest: digest.clone(),
                node_id: node_id.clone(),
                signature: signing_key.sign(&digest).to_bytes().to_vec(),
            };
            mgr.consensus
                .handle_message(commit, node_id)
                .expect("handle commit");
        }

        // process_request must now complete Ok(RdfResponse::Success) well before
        // the 5s timeout.
        let response = tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("process_request did not finish before test deadline")
            .expect("process_request task panicked")
            .expect("process_request returned an error");
        assert_eq!(response, RdfResponse::Success);

        // And the operation is visibly applied to the backend.
        let triples = mock
            .get_shard_triples(0)
            .await
            .expect("query backend shard");
        assert_eq!(triples.len(), 1, "insert must be applied to the backend");
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
