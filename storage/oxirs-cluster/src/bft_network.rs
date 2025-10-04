//! Byzantine fault-tolerant network layer
//!
//! This module provides secure network communication for BFT consensus,
//! including message authentication, ordering, and Byzantine node detection.

use crate::bft::{BftConfig, BftConsensus, BftMessage};
use crate::network::{NetworkService, RpcMessage};
use crate::{ClusterError, Result};
use ed25519_dalek::{SigningKey, VerifyingKey, Signature};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio::time::{interval, Duration};
use tracing::{debug, error, info, warn};

/// BFT network message wrapper with authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticatedMessage {
    /// The actual BFT message
    pub message: BftMessage,
    /// Sender node ID
    pub sender: String,
    /// Message sequence number for ordering
    pub sequence: u64,
    /// Cryptographic signature
    pub signature: Vec<u8>,
    /// Timestamp for freshness
    pub timestamp: u64,
}

/// BFT network service for secure communication
pub struct BftNetworkService {
    /// Node identifier
    node_id: String,
    /// BFT consensus engine
    consensus: Arc<BftConsensus>,
    /// Network service for transport
    network: Arc<NetworkService>,
    /// Message sequence counter
    sequence_counter: Arc<RwLock<u64>>,
    /// Message cache for duplicate detection
    message_cache: Arc<RwLock<MessageCache>>,
    /// Peer public keys
    peer_keys: Arc<RwLock<HashMap<String, VerifyingKey>>>,
    /// Message channel sender
    tx: mpsc::Sender<AuthenticatedMessage>,
    /// Message channel receiver
    rx: Arc<RwLock<mpsc::Receiver<AuthenticatedMessage>>>,
    /// Node's Ed25519 keypair for signing
    keypair: SigningKey,
}

/// Message cache for duplicate detection and ordering
struct MessageCache {
    /// Received messages by sender and sequence
    messages: HashMap<(String, u64), AuthenticatedMessage>,
    /// Highest sequence number per sender
    highest_seq: HashMap<String, u64>,
    /// Cache size limit
    max_size: usize,
}

impl MessageCache {
    fn new(max_size: usize) -> Self {
        MessageCache {
            messages: HashMap::new(),
            highest_seq: HashMap::new(),
            max_size,
        }
    }

    /// Check if message is duplicate or out of order
    fn is_duplicate_or_old(&self, sender: &str, sequence: u64) -> bool {
        if let Some(&highest) = self.highest_seq.get(sender) {
            sequence <= highest
        } else {
            false
        }
    }

    /// Add message to cache
    fn add_message(&mut self, msg: AuthenticatedMessage) {
        let key = (msg.sender.clone(), msg.sequence);
        self.messages.insert(key, msg.clone());

        // Update highest sequence
        self.highest_seq
            .entry(msg.sender.clone())
            .and_modify(|seq| *seq = (*seq).max(msg.sequence))
            .or_insert(msg.sequence);

        // Evict old messages if cache is full
        if self.messages.len() > self.max_size {
            self.evict_oldest();
        }
    }

    /// Evict oldest messages from cache
    fn evict_oldest(&mut self) {
        let to_remove = self.messages.len() - self.max_size;
        let mut entries: Vec<_> = self
            .messages
            .iter()
            .map(|(k, v)| (k.clone(), v.timestamp))
            .collect();
        entries.sort_by_key(|(_, ts)| *ts);

        for (key, _) in entries.iter().take(to_remove) {
            self.messages.remove(key);
        }
    }
}

impl BftNetworkService {
    /// Create a new BFT network service with generated keypair
    pub fn new(
        node_id: String,
        consensus: Arc<BftConsensus>,
        network: Arc<NetworkService>,
    ) -> Self {
        let (tx, rx) = mpsc::channel(1000);

        // Generate a new Ed25519 keypair for this node
        let mut csprng = rand::rngs::OsRng {};
        let keypair = SigningKey::generate(&mut csprng);

        BftNetworkService {
            node_id,
            consensus,
            network,
            sequence_counter: Arc::new(RwLock::new(0)),
            message_cache: Arc::new(RwLock::new(MessageCache::new(10000))),
            peer_keys: Arc::new(RwLock::new(HashMap::new())),
            tx,
            rx: Arc::new(RwLock::new(rx)),
            keypair,
        }
    }

    /// Create a new BFT network service with provided keypair
    pub fn with_keypair(
        node_id: String,
        consensus: Arc<BftConsensus>,
        network: Arc<NetworkService>,
        keypair: Keypair,
    ) -> Self {
        let (tx, rx) = mpsc::channel(1000);

        BftNetworkService {
            node_id,
            consensus,
            network,
            sequence_counter: Arc::new(RwLock::new(0)),
            message_cache: Arc::new(RwLock::new(MessageCache::new(10000))),
            peer_keys: Arc::new(RwLock::new(HashMap::new())),
            tx,
            rx: Arc::new(RwLock::new(rx)),
            keypair,
        }
    }

    /// Get the node's public key
    pub fn public_key(&self) -> VerifyingKey {
        self.keypair.verifying_key()
    }

    /// Get the node's public key as bytes
    pub fn public_key_bytes(&self) -> [u8; 32] {
        self.keypair.verifying_key().to_bytes()
    }

    /// Register a peer's public key
    pub async fn register_peer(&self, peer_id: String, public_key: VerifyingKey) -> Result<()> {
        let mut keys = self.peer_keys.write().await;
        keys.insert(peer_id.clone(), public_key);

        // Also register with consensus engine
        self.consensus.register_node(peer_id, public_key)?;

        Ok(())
    }

    /// Start the BFT network service
    pub async fn start(self: Arc<Self>) -> Result<()> {
        // Start message processor
        let processor = self.clone();
        tokio::spawn(async move {
            processor.process_messages().await;
        });

        // Start heartbeat sender
        let heartbeat = self.clone();
        tokio::spawn(async move {
            heartbeat.send_heartbeats().await;
        });

        // Start view change monitor
        let monitor = self.clone();
        tokio::spawn(async move {
            monitor.monitor_view_changes().await;
        });

        Ok(())
    }

    /// Process incoming messages
    async fn process_messages(self: Arc<Self>) {
        let mut rx = self.rx.write().await;

        while let Some(auth_msg) = rx.recv().await {
            match self.handle_authenticated_message(auth_msg).await {
                Ok(_) => {}
                Err(e) => error!("Failed to handle message: {}", e),
            }
        }
    }

    /// Handle an authenticated message
    async fn handle_authenticated_message(&self, auth_msg: AuthenticatedMessage) -> Result<()> {
        // Check message freshness (5 minute window)
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        if current_time > auth_msg.timestamp + 300 {
            return Err(ClusterError::Network("Message too old".to_string()));
        }

        // Check for duplicates
        let mut cache = self.message_cache.write().await;
        if cache.is_duplicate_or_old(&auth_msg.sender, auth_msg.sequence) {
            debug!("Duplicate or old message from {}", auth_msg.sender);
            return Ok(());
        }

        // Verify signature
        if !self.verify_message_signature(&auth_msg).await? {
            warn!("Invalid signature from {}", auth_msg.sender);
            return Err(ClusterError::Network("Invalid signature".to_string()));
        }

        // Add to cache
        cache.add_message(auth_msg.clone());
        drop(cache);

        // Pass to consensus engine
        self.consensus
            .handle_message(auth_msg.message, &auth_msg.sender)?;

        Ok(())
    }

    /// Send a BFT message to all peers
    pub async fn broadcast(&self, message: BftMessage) -> Result<()> {
        let auth_msg = self.create_authenticated_message(message).await?;

        // Serialize the message
        let data = serde_json::to_vec(&auth_msg)
            .map_err(|e| ClusterError::Network(format!("Serialization error: {e}")))?;

        // Broadcast through network service
        self.network.broadcast(RpcMessage::Bft { data }).await?;

        Ok(())
    }

    /// Send a BFT message to a specific peer
    pub async fn send_to(&self, peer_id: &str, message: BftMessage) -> Result<()> {
        let auth_msg = self.create_authenticated_message(message).await?;

        // Serialize the message
        let data = serde_json::to_vec(&auth_msg)
            .map_err(|e| ClusterError::Network(format!("Serialization error: {e}")))?;

        // Send through network service
        self.network
            .send_to(peer_id, RpcMessage::Bft { data })
            .await?;

        Ok(())
    }

    /// Create an authenticated message
    async fn create_authenticated_message(
        &self,
        message: BftMessage,
    ) -> Result<AuthenticatedMessage> {
        // Increment sequence counter
        let mut seq = self.sequence_counter.write().await;
        *seq += 1;
        let sequence = *seq;

        // Get current timestamp
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Create message without signature
        let mut auth_msg = AuthenticatedMessage {
            message,
            sender: self.node_id.clone(),
            sequence,
            signature: vec![],
            timestamp,
        };

        // Sign the message
        let msg_bytes = serde_json::to_vec(&auth_msg)
            .map_err(|e| ClusterError::Network(format!("Serialization error: {e}")))?;

        // Sign the message with node's private key
        let signature = self.keypair.sign(&msg_bytes);
        auth_msg.signature = signature.to_bytes().to_vec();

        Ok(auth_msg)
    }

    /// Verify message signature
    async fn verify_message_signature(&self, auth_msg: &AuthenticatedMessage) -> Result<bool> {
        // Get sender's public key
        let peer_keys = self.peer_keys.read().await;
        let public_key = match peer_keys.get(&auth_msg.sender) {
            Some(key) => key,
            None => {
                warn!("No public key found for peer: {}", auth_msg.sender);
                return Ok(false);
            }
        };

        // Create message without signature for verification
        let mut msg_for_verification = auth_msg.clone();
        msg_for_verification.signature = vec![];

        // Serialize message for verification
        let msg_bytes = match serde_json::to_vec(&msg_for_verification) {
            Ok(bytes) => bytes,
            Err(e) => {
                error!("Failed to serialize message for verification: {}", e);
                return Ok(false);
            }
        };

        // Convert signature bytes to signature
        let signature = match ed25519_dalek::Signature::from_bytes(&auth_msg.signature) {
            Ok(sig) => sig,
            Err(e) => {
                warn!("Invalid signature format from {}: {}", auth_msg.sender, e);
                return Ok(false);
            }
        };

        // Verify the signature
        match public_key.verify(&msg_bytes, &signature) {
            Ok(_) => {
                debug!("Signature verification successful for {}", auth_msg.sender);
                Ok(true)
            }
            Err(e) => {
                warn!(
                    "Signature verification failed for {}: {}",
                    auth_msg.sender, e
                );
                Ok(false)
            }
        }
    }

    /// Send periodic heartbeats
    async fn send_heartbeats(&self) {
        let mut interval = interval(Duration::from_secs(1));

        loop {
            interval.tick().await;

            // Only send heartbeats if we're the primary
            match self.consensus.is_primary() {
                Ok(true) => {
                    let heartbeat = BftMessage::Request {
                        client_id: format!("{}-heartbeat", self.node_id),
                        operation: b"HEARTBEAT".to_vec(),
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        signature: None,
                    };

                    if let Err(e) = self.broadcast(heartbeat).await {
                        warn!("Failed to send heartbeat: {}", e);
                    }
                }
                _ => {}
            }
        }
    }

    /// Monitor for view changes
    async fn monitor_view_changes(&self) {
        let mut interval = interval(Duration::from_secs(5));

        loop {
            interval.tick().await;

            // Check if view change is needed
            match self.consensus.check_view_timeout() {
                Ok(true) => {
                    info!("View change timeout detected");

                    // Initiate view change
                    if let Err(e) = self.initiate_view_change().await {
                        error!("Failed to initiate view change: {}", e);
                    }
                }
                _ => {}
            }
        }
    }

    /// Initiate a view change
    async fn initiate_view_change(&self) -> Result<()> {
        let current_view = self.consensus.current_view()?;
        let new_view = current_view + 1;

        info!("Initiating view change to view {}", new_view);

        // Create view change message
        let view_change = BftMessage::ViewChange {
            new_view,
            node_id: self.node_id.clone(),
            prepared_messages: vec![], // TODO: Collect prepared messages
            signature: vec![],
        };

        // Broadcast view change
        self.broadcast(view_change).await?;

        Ok(())
    }

    /// Handle incoming network messages
    pub async fn handle_network_message(&self, data: Vec<u8>) -> Result<()> {
        // Deserialize the authenticated message
        let auth_msg: AuthenticatedMessage = serde_json::from_slice(&data)
            .map_err(|e| ClusterError::Network(format!("Deserialization error: {e}")))?;

        // Send to processing channel
        self.tx
            .send(auth_msg)
            .await
            .map_err(|e| ClusterError::Network(format!("Channel send error: {e}")))?;

        Ok(())
    }

    /// Remove a peer's public key (e.g., when a node is detected as Byzantine)
    pub async fn remove_peer(&self, peer_id: &str) -> Result<()> {
        let mut keys = self.peer_keys.write().await;
        keys.remove(peer_id);

        info!("Removed public key for peer: {}", peer_id);
        Ok(())
    }

    /// Get list of trusted peers
    pub async fn get_trusted_peers(&self) -> Vec<String> {
        let keys = self.peer_keys.read().await;
        keys.keys().cloned().collect()
    }

    /// Verify a standalone signature (for external verification)
    pub fn verify_signature(
        &self,
        message: &[u8],
        signature: &[u8],
        public_key: &VerifyingKey,
    ) -> Result<bool> {
        let signature = match Signature::from_bytes(signature) {
            Ok(sig) => sig,
            Err(_) => return Ok(false),
        };

        match public_key.verify(message, &signature) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Sign a message with the node's private key
    pub fn sign_message(&self, message: &[u8]) -> Vec<u8> {
        let signature = self.keypair.sign(message);
        signature.to_bytes().to_vec()
    }

    /// Check if a peer is trusted (has a registered public key)
    pub async fn is_peer_trusted(&self, peer_id: &str) -> bool {
        let keys = self.peer_keys.read().await;
        keys.contains_key(peer_id)
    }
}

/// BFT metrics for monitoring
#[derive(Debug, Clone, Default)]
pub struct BftMetrics {
    /// Total messages sent
    pub messages_sent: u64,
    /// Total messages received
    pub messages_received: u64,
    /// Invalid signatures detected
    pub invalid_signatures: u64,
    /// Byzantine nodes detected
    pub byzantine_nodes_detected: u64,
    /// View changes initiated
    pub view_changes: u64,
    /// Successful consensus rounds
    pub consensus_rounds: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_cache() {
        let mut cache = MessageCache::new(100);

        // Test duplicate detection
        let msg = AuthenticatedMessage {
            message: BftMessage::Request {
                client_id: "test".to_string(),
                operation: vec![1, 2, 3],
                timestamp: 1000,
                signature: None,
            },
            sender: "node1".to_string(),
            sequence: 1,
            signature: vec![],
            timestamp: 1000,
        };

        assert!(!cache.is_duplicate_or_old("node1", 1));
        cache.add_message(msg.clone());
        assert!(cache.is_duplicate_or_old("node1", 1));
        assert!(!cache.is_duplicate_or_old("node1", 2));
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = MessageCache::new(2);

        // Add messages to fill cache
        for i in 0..3 {
            let msg = AuthenticatedMessage {
                message: BftMessage::Request {
                    client_id: format!("test{}", i),
                    operation: vec![i as u8],
                    timestamp: i as u64,
                    signature: None,
                },
                sender: format!("node{}", i),
                sequence: 1,
                signature: vec![],
                timestamp: i as u64,
            };
            cache.add_message(msg);
        }

        // Cache should only have 2 messages
        assert_eq!(cache.messages.len(), 2);

        // Oldest message should be evicted
        assert!(!cache.messages.contains_key(&("node0".to_string(), 1)));
    }
}
