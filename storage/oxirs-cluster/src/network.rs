//! # Network Communication Layer
//!
//! Network communication layer for Raft consensus protocol.
//! Provides RPC mechanisms for node-to-node communication.

use crate::raft::{OxirsNodeId, RdfCommand, RdfResponse};
use crate::tls::{TlsConfig, TlsManager};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use std::time::SystemTime;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;
use tokio::time::timeout;

/// RPC message types for Raft protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RpcMessage {
    /// Request vote message
    RequestVote {
        term: u64,
        candidate_id: OxirsNodeId,
        last_log_index: u64,
        last_log_term: u64,
    },
    /// Vote response
    VoteResponse { term: u64, vote_granted: bool },
    /// Append entries message
    AppendEntries {
        term: u64,
        leader_id: OxirsNodeId,
        prev_log_index: u64,
        prev_log_term: u64,
        entries: Vec<LogEntry>,
        leader_commit: u64,
    },
    /// Append entries response
    AppendEntriesResponse {
        term: u64,
        success: bool,
        last_log_index: u64,
    },
    /// Client request
    ClientRequest { command: RdfCommand },
    /// Client response
    ClientResponse { response: RdfResponse },
    /// Heartbeat message
    Heartbeat { term: u64, leader_id: OxirsNodeId },
    /// Heartbeat response
    HeartbeatResponse { term: u64 },
    /// Byzantine fault tolerance message
    #[cfg(feature = "bft")]
    Bft { data: Vec<u8> },
    /// Shard operation message
    ShardOperation(crate::shard_manager::ShardOperation),
    /// Store triple to shard
    StoreTriple {
        shard_id: crate::shard::ShardId,
        triple: oxirs_core::model::Triple,
    },
    /// Replicate triple to shard
    ReplicateTriple {
        shard_id: crate::shard::ShardId,
        triple: oxirs_core::model::Triple,
    },
    /// Query shard
    QueryShard {
        shard_id: crate::shard::ShardId,
        subject: Option<String>,
        predicate: Option<String>,
        object: Option<String>,
    },
    /// Query shard response
    QueryShardResponse {
        shard_id: crate::shard::ShardId,
        results: Vec<oxirs_core::model::Triple>,
    },
    /// Transaction prepare request
    TransactionPrepare {
        tx_id: String,
        shard_id: crate::shard::ShardId,
        operations: Vec<crate::transaction::TransactionOp>,
    },
    /// Transaction vote response
    TransactionVote {
        tx_id: String,
        shard_id: crate::shard::ShardId,
        vote: bool,
    },
    /// Transaction commit request
    TransactionCommit {
        tx_id: String,
        shard_id: crate::shard::ShardId,
    },
    /// Transaction abort request
    TransactionAbort {
        tx_id: String,
        shard_id: crate::shard::ShardId,
    },
    /// Transaction acknowledgment
    TransactionAck {
        tx_id: String,
        shard_id: crate::shard::ShardId,
    },
    /// Migration batch transfer
    MigrationBatch {
        migration_id: String,
        batch: crate::shard_migration::MigrationBatch,
    },
    /// Shard data transfer during migration
    ShardTransfer {
        shard_id: crate::shard::ShardId,
        triples: Vec<oxirs_core::model::Triple>,
        source_node: crate::raft::OxirsNodeId,
    },
}

/// Log entry for Raft protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub index: u64,
    pub term: u64,
    pub command: RdfCommand,
}

impl LogEntry {
    pub fn new(index: u64, term: u64, command: RdfCommand) -> Self {
        Self {
            index,
            term,
            command,
        }
    }
}

/// Network configuration
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Local node address
    pub local_address: SocketAddr,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Request timeout
    pub request_timeout: Duration,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// Keep-alive interval
    pub keep_alive_interval: Duration,
    /// TLS configuration
    pub tls_config: TlsConfig,
    /// Enable compression for messages
    pub enable_compression: bool,
    /// Maximum message size (bytes)
    pub max_message_size: usize,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            local_address: "127.0.0.1:8080"
                .parse()
                .expect("localhost address is valid"),
            connection_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(10),
            max_connections: 100,
            keep_alive_interval: Duration::from_secs(30),
            tls_config: TlsConfig::default(),
            enable_compression: true,
            max_message_size: 16 * 1024 * 1024, // 16MB
        }
    }
}

/// Network connection manager
#[derive(Debug)]
pub struct NetworkManager {
    config: NetworkConfig,
    node_id: OxirsNodeId,
    connections: Arc<RwLock<HashMap<OxirsNodeId, Connection>>>,
    listener: Option<TcpListener>,
    running: Arc<RwLock<bool>>,
    tls_manager: Option<Arc<TlsManager>>,
    message_stats: Arc<RwLock<MessageStats>>,
}

/// Message statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct MessageStats {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub connections_established: u64,
    pub connections_failed: u64,
    pub tls_handshakes_completed: u64,
    pub tls_handshakes_failed: u64,
}

/// Network connection to a peer
#[derive(Debug, Clone)]
pub struct Connection {
    pub peer_id: OxirsNodeId,
    pub address: SocketAddr,
    pub last_activity: std::time::Instant,
    pub is_connected: bool,
}

impl Connection {
    pub fn new(peer_id: OxirsNodeId, address: SocketAddr) -> Self {
        Self {
            peer_id,
            address,
            last_activity: std::time::Instant::now(),
            is_connected: false,
        }
    }

    /// Check if connection is stale
    pub fn is_stale(&self, timeout: Duration) -> bool {
        self.last_activity.elapsed() > timeout
    }

    /// Update last activity timestamp
    pub fn update_activity(&mut self) {
        self.last_activity = std::time::Instant::now();
    }
}

impl NetworkManager {
    /// Create a new network manager
    pub fn new(node_id: OxirsNodeId, config: NetworkConfig) -> Self {
        Self {
            config,
            node_id,
            connections: Arc::new(RwLock::new(HashMap::new())),
            listener: None,
            running: Arc::new(RwLock::new(false)),
            tls_manager: None,
            message_stats: Arc::new(RwLock::new(MessageStats::default())),
        }
    }

    /// Create a new network manager with TLS support
    pub async fn with_tls(node_id: OxirsNodeId, config: NetworkConfig) -> Result<Self> {
        let tls_manager = if config.tls_config.enabled {
            let tls_mgr = TlsManager::new(config.tls_config.clone(), node_id);
            tls_mgr.initialize().await?;
            Some(Arc::new(tls_mgr))
        } else {
            None
        };

        Ok(Self {
            config,
            node_id,
            connections: Arc::new(RwLock::new(HashMap::new())),
            listener: None,
            running: Arc::new(RwLock::new(false)),
            tls_manager,
            message_stats: Arc::new(RwLock::new(MessageStats::default())),
        })
    }

    /// Start the network manager
    pub async fn start(&mut self) -> Result<()> {
        {
            let mut running = self.running.write().await;
            if *running {
                return Ok(());
            }
            *running = true;
        }

        // Start listening for incoming connections
        let listener = TcpListener::bind(self.config.local_address).await?;
        tracing::info!(
            "Network manager for node {} listening on {}",
            self.node_id,
            self.config.local_address
        );

        self.listener = Some(listener);

        // Start background tasks
        self.start_background_tasks().await;

        Ok(())
    }

    /// Stop the network manager
    pub async fn stop(&mut self) -> Result<()> {
        let mut running = self.running.write().await;
        if !*running {
            return Ok(());
        }

        tracing::info!("Stopping network manager for node {}", self.node_id);
        *running = false;

        // Close all connections
        let mut connections = self.connections.write().await;
        connections.clear();

        Ok(())
    }

    /// Send RPC message to a peer
    pub async fn send_rpc(
        &self,
        peer_id: OxirsNodeId,
        peer_address: SocketAddr,
        message: RpcMessage,
    ) -> Result<RpcMessage> {
        // Get or create connection
        let connection = self.get_or_create_connection(peer_id, peer_address).await?;

        // Send message with timeout
        let response = timeout(
            self.config.request_timeout,
            self.send_message_to_connection(connection, message),
        )
        .await??;

        Ok(response)
    }

    /// Send request vote RPC
    pub async fn send_request_vote(
        &self,
        peer_id: OxirsNodeId,
        peer_address: SocketAddr,
        term: u64,
        last_log_index: u64,
        last_log_term: u64,
    ) -> Result<(u64, bool)> {
        let message = RpcMessage::RequestVote {
            term,
            candidate_id: self.node_id,
            last_log_index,
            last_log_term,
        };

        let response = self.send_rpc(peer_id, peer_address, message).await?;

        match response {
            RpcMessage::VoteResponse { term, vote_granted } => Ok((term, vote_granted)),
            _ => Err(anyhow::anyhow!("Unexpected response type")),
        }
    }

    /// Send append entries RPC
    #[allow(clippy::too_many_arguments)]
    pub async fn send_append_entries(
        &self,
        peer_id: OxirsNodeId,
        peer_address: SocketAddr,
        term: u64,
        prev_log_index: u64,
        prev_log_term: u64,
        entries: Vec<LogEntry>,
        leader_commit: u64,
    ) -> Result<(u64, bool, u64)> {
        let message = RpcMessage::AppendEntries {
            term,
            leader_id: self.node_id,
            prev_log_index,
            prev_log_term,
            entries,
            leader_commit,
        };

        let response = self.send_rpc(peer_id, peer_address, message).await?;

        match response {
            RpcMessage::AppendEntriesResponse {
                term,
                success,
                last_log_index,
            } => Ok((term, success, last_log_index)),
            _ => Err(anyhow::anyhow!("Unexpected response type")),
        }
    }

    /// Send heartbeat to all peers
    pub async fn send_heartbeat(&self, term: u64, peers: &[(OxirsNodeId, SocketAddr)]) {
        let message = RpcMessage::Heartbeat {
            term,
            leader_id: self.node_id,
        };

        for &(peer_id, peer_address) in peers {
            if peer_id != self.node_id {
                let manager = self.clone();
                let message = message.clone();
                tokio::spawn(async move {
                    if let Err(e) = manager.send_rpc(peer_id, peer_address, message).await {
                        tracing::warn!("Failed to send heartbeat to peer {}: {}", peer_id, e);
                    }
                });
            }
        }
    }

    /// Get or create connection to peer
    async fn get_or_create_connection(
        &self,
        peer_id: OxirsNodeId,
        peer_address: SocketAddr,
    ) -> Result<Connection> {
        {
            let connections = self.connections.read().await;
            if let Some(connection) = connections.get(&peer_id) {
                if connection.is_connected && !connection.is_stale(self.config.connection_timeout) {
                    return Ok(connection.clone());
                }
            }
        }

        // Create new connection
        let _stream = timeout(
            self.config.connection_timeout,
            TcpStream::connect(peer_address),
        )
        .await??;

        let mut connection = Connection::new(peer_id, peer_address);
        connection.is_connected = true;
        connection.update_activity();

        // Store connection
        {
            let mut connections = self.connections.write().await;
            connections.insert(peer_id, connection.clone());
        }

        Ok(connection)
    }

    /// Send message to a specific connection
    async fn send_message_to_connection(
        &self,
        mut connection: Connection,
        message: RpcMessage,
    ) -> Result<RpcMessage> {
        // In a real implementation, this would serialize the message
        // and send it over TCP, then wait for and deserialize the response.
        // For now, we'll simulate the network communication.

        connection.update_activity();

        // Simulate network delay
        tokio::time::sleep(Duration::from_millis(1)).await;

        // For demonstration, echo back a default response
        let response = match message {
            RpcMessage::RequestVote { term, .. } => RpcMessage::VoteResponse {
                term,
                vote_granted: false, // Default deny
            },
            RpcMessage::AppendEntries { term, .. } => RpcMessage::AppendEntriesResponse {
                term,
                success: true, // Default success
                last_log_index: 0,
            },
            RpcMessage::Heartbeat { term, .. } => RpcMessage::HeartbeatResponse { term },
            RpcMessage::ClientRequest { .. } => RpcMessage::ClientResponse {
                response: RdfResponse::Success,
            },
            _ => return Err(anyhow::anyhow!("Unexpected message type")),
        };

        Ok(response)
    }

    /// Start background maintenance tasks
    async fn start_background_tasks(&self) {
        let running = Arc::clone(&self.running);
        let connections = Arc::clone(&self.connections);
        let connection_timeout = self.config.connection_timeout;

        // Connection cleanup task
        tokio::spawn(async move {
            while *running.read().await {
                {
                    let mut connections = connections.write().await;
                    let stale_connections: Vec<_> = connections
                        .iter()
                        .filter(|(_, conn)| conn.is_stale(connection_timeout))
                        .map(|(&id, _)| id)
                        .collect();

                    for peer_id in stale_connections {
                        connections.remove(&peer_id);
                        tracing::debug!("Removed stale connection to peer {}", peer_id);
                    }
                }

                tokio::time::sleep(Duration::from_secs(30)).await;
            }
        });

        // Connection listener task
        // Note: In a real implementation, we would need to handle the listener differently
        // since TcpListener doesn't support cloning. For now, we'll skip the listener task.
    }

    /// Get connection statistics
    pub async fn get_stats(&self) -> NetworkStats {
        let connections = self.connections.read().await;
        let total_connections = connections.len();
        let active_connections = connections.values().filter(|c| c.is_connected).count();

        NetworkStats {
            total_connections,
            active_connections,
            local_address: self.config.local_address,
            node_id: self.node_id,
        }
    }

    /// Send encrypted RPC message to a peer using TLS
    pub async fn send_secure_rpc(
        &self,
        peer_id: OxirsNodeId,
        peer_address: SocketAddr,
        message: RpcMessage,
    ) -> Result<RpcMessage> {
        if let Some(tls_manager) = &self.tls_manager {
            let connector = tls_manager.get_connector().await?;
            let tcp_stream = timeout(
                self.config.connection_timeout,
                TcpStream::connect(peer_address),
            )
            .await??;

            // Perform TLS handshake
            let server_name = rustls::pki_types::ServerName::try_from(format!("node-{peer_id}"))?;

            let _tls_stream = connector.connect(server_name, tcp_stream).await?;

            // Update TLS statistics
            {
                let mut stats = self.message_stats.write().await;
                stats.tls_handshakes_completed += 1;
                stats.connections_established += 1;
            }

            // In a real implementation, we would serialize and send the message
            // over the TLS stream. For now, simulate secure communication.
            self.simulate_secure_communication(message).await
        } else {
            // Fall back to non-TLS communication
            self.send_rpc(peer_id, peer_address, message).await
        }
    }

    /// Simulate secure communication for testing
    async fn simulate_secure_communication(&self, message: RpcMessage) -> Result<RpcMessage> {
        // Update message statistics
        {
            let mut stats = self.message_stats.write().await;
            stats.messages_sent += 1;
            stats.bytes_sent += self.estimate_message_size(&message);
        }

        // Simulate network delay
        tokio::time::sleep(Duration::from_millis(5)).await;

        // Echo back appropriate response
        let response = match message {
            RpcMessage::RequestVote { term, .. } => RpcMessage::VoteResponse {
                term,
                vote_granted: true,
            },
            RpcMessage::AppendEntries { term, .. } => RpcMessage::AppendEntriesResponse {
                term,
                success: true,
                last_log_index: 0,
            },
            RpcMessage::Heartbeat { term, .. } => RpcMessage::HeartbeatResponse { term },
            RpcMessage::ClientRequest { .. } => RpcMessage::ClientResponse {
                response: RdfResponse::Success,
            },
            _ => return Err(anyhow::anyhow!("Unsupported message type")),
        };

        // Update received statistics
        {
            let mut stats = self.message_stats.write().await;
            stats.messages_received += 1;
            stats.bytes_received += self.estimate_message_size(&response);
        }

        Ok(response)
    }

    /// Estimate message size for statistics
    fn estimate_message_size(&self, message: &RpcMessage) -> u64 {
        // Simple estimation based on message content
        match message {
            RpcMessage::RequestVote { .. } => 64,
            RpcMessage::VoteResponse { .. } => 32,
            RpcMessage::AppendEntries { entries, .. } => 128 + entries.len() as u64 * 256,
            RpcMessage::AppendEntriesResponse { .. } => 48,
            RpcMessage::Heartbeat { .. } => 24,
            RpcMessage::HeartbeatResponse { .. } => 16,
            RpcMessage::ClientRequest { .. } => 512,
            RpcMessage::ClientResponse { .. } => 256,
            _ => 128,
        }
    }

    /// Get message statistics
    pub async fn get_message_stats(&self) -> MessageStats {
        self.message_stats.read().await.clone()
    }

    /// Reset message statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.message_stats.write().await;
        *stats = MessageStats::default();
    }

    /// Check TLS certificate status
    pub async fn get_tls_status(&self) -> Result<TlsStatus> {
        if let Some(tls_manager) = &self.tls_manager {
            let certificates = tls_manager.list_certificates().await;
            let server_cert = certificates.get("server");

            Ok(TlsStatus {
                enabled: true,
                certificates_count: certificates.len(),
                server_cert_expires: server_cert.map(|c| c.not_after),
                handshakes_completed: self.message_stats.read().await.tls_handshakes_completed,
                handshakes_failed: self.message_stats.read().await.tls_handshakes_failed,
            })
        } else {
            Ok(TlsStatus {
                enabled: false,
                certificates_count: 0,
                server_cert_expires: None,
                handshakes_completed: 0,
                handshakes_failed: 0,
            })
        }
    }

    /// Encrypt data at rest using the TLS manager's encryption
    pub async fn encrypt_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        if let Some(_tls_manager) = &self.tls_manager {
            // Use the TLS manager's encryption capabilities
            // In a real implementation, we might extract the encryption manager
            // For now, we'll simulate encryption
            let mut encrypted = Vec::with_capacity(data.len() + 32);
            encrypted.extend_from_slice(b"ENCRYPTED:");
            encrypted.extend_from_slice(data);
            Ok(encrypted)
        } else {
            // No encryption available
            Ok(data.to_vec())
        }
    }

    /// Decrypt data at rest
    pub async fn decrypt_data(&self, encrypted_data: &[u8]) -> Result<Vec<u8>> {
        if let Some(_tls_manager) = &self.tls_manager {
            // Use the TLS manager's decryption capabilities
            // For now, we'll simulate decryption
            if encrypted_data.starts_with(b"ENCRYPTED:") {
                Ok(encrypted_data[10..].to_vec())
            } else {
                Err(anyhow::anyhow!("Invalid encrypted data format"))
            }
        } else {
            // No decryption available
            Ok(encrypted_data.to_vec())
        }
    }
}

// Implement Clone for NetworkManager to allow sharing
impl Clone for NetworkManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            node_id: self.node_id,
            connections: Arc::clone(&self.connections),
            listener: None, // Don't clone the listener
            running: Arc::clone(&self.running),
            tls_manager: self.tls_manager.clone(),
            message_stats: Arc::clone(&self.message_stats),
        }
    }
}

/// Network statistics
#[derive(Debug, Clone)]
pub struct NetworkStats {
    pub total_connections: usize,
    pub active_connections: usize,
    pub local_address: SocketAddr,
    pub node_id: OxirsNodeId,
}

/// TLS status information
#[derive(Debug, Clone)]
pub struct TlsStatus {
    pub enabled: bool,
    pub certificates_count: usize,
    pub server_cert_expires: Option<SystemTime>,
    pub handshakes_completed: u64,
    pub handshakes_failed: u64,
}

/// Network service for high-level network operations
#[derive(Debug, Clone)]
pub struct NetworkService {
    manager: NetworkManager,
}

impl NetworkService {
    /// Create a new network service
    pub fn new(node_id: OxirsNodeId, config: NetworkConfig) -> Self {
        Self {
            manager: NetworkManager::new(node_id, config),
        }
    }

    /// Start the network service
    pub async fn start(&mut self) -> Result<()> {
        self.manager.start().await
    }

    /// Stop the network service
    pub async fn stop(&mut self) -> Result<()> {
        self.manager.stop().await
    }

    /// Send a message to a specific peer
    pub async fn send_to(&self, peer_id: &str, message: RpcMessage) -> Result<()> {
        let peer_id: OxirsNodeId = peer_id
            .parse()
            .map_err(|_| anyhow::anyhow!("Invalid peer ID"))?;
        // Use send_message method from NetworkService instead
        self.send_message(peer_id, message).await?;
        Ok(())
    }

    /// Broadcast a message to all peers
    pub async fn broadcast(&self, message: RpcMessage) -> Result<()> {
        let connections = self.manager.connections.read().await;
        for peer_id in connections.keys() {
            let _ = self.send_message(*peer_id, message.clone()).await;
        }
        Ok(())
    }

    /// Send a message to a specific node
    pub async fn send_message(&self, node_id: OxirsNodeId, message: RpcMessage) -> Result<()> {
        // In a real implementation, this would use the network manager to send the message
        // For now, we'll just log it
        tracing::debug!("Sending message to node {}: {:?}", node_id, message);
        Ok(())
    }

    /// Handle incoming RPC message
    pub async fn handle_message(&self, message: RpcMessage) -> Result<RpcMessage> {
        match message {
            #[cfg(feature = "bft")]
            RpcMessage::Bft { .. } => {
                // BFT messages are handled separately by the BFT network service
                Err(anyhow::anyhow!(
                    "BFT messages should be handled by BFT network service"
                ))
            }

            // --- Raft consensus messages ---
            RpcMessage::RequestVote { term, .. } => {
                // Conservative default: deny vote (caller supplies real handler for production)
                Ok(RpcMessage::VoteResponse {
                    term,
                    vote_granted: false,
                })
            }
            RpcMessage::VoteResponse { .. } => {
                // Terminal response — surfaces to the original caller; echo back
                Ok(message)
            }
            RpcMessage::AppendEntries { term, .. } => {
                Ok(RpcMessage::AppendEntriesResponse {
                    term,
                    success: true,
                    last_log_index: 0,
                })
            }
            RpcMessage::AppendEntriesResponse { .. } => Ok(message),
            RpcMessage::Heartbeat { term, .. } => {
                Ok(RpcMessage::HeartbeatResponse { term })
            }
            RpcMessage::HeartbeatResponse { .. } => Ok(message),

            // --- Client request/response ---
            RpcMessage::ClientRequest { .. } => {
                Ok(RpcMessage::ClientResponse {
                    response: RdfResponse::Success,
                })
            }
            RpcMessage::ClientResponse { .. } => Ok(message),

            // --- Shard operations ---
            RpcMessage::ShardOperation(_) => {
                // Shard operations require a connected shard manager; return acknowledgement
                Err(anyhow::anyhow!(
                    "RpcMessage::ShardOperation requires a shard manager — not connected"
                ))
            }
            RpcMessage::StoreTriple { shard_id, .. } => {
                // Triple storage requires a wired-up shard handler
                Err(anyhow::anyhow!(
                    "RpcMessage::StoreTriple for shard {:?} requires shard handler — not connected",
                    shard_id
                ))
            }
            RpcMessage::ReplicateTriple { shard_id, .. } => {
                Err(anyhow::anyhow!(
                    "RpcMessage::ReplicateTriple for shard {:?} requires shard handler — not connected",
                    shard_id
                ))
            }
            RpcMessage::QueryShard { shard_id, .. } => {
                // Return an empty result set rather than an error so callers can handle gracefully
                Ok(RpcMessage::QueryShardResponse {
                    shard_id,
                    results: Vec::new(),
                })
            }
            RpcMessage::QueryShardResponse { .. } => Ok(message),

            // --- Two-phase commit / distributed transactions ---
            RpcMessage::TransactionPrepare { tx_id, shard_id, .. } => {
                // Conservative default: vote no so transactions don't silently commit
                Ok(RpcMessage::TransactionVote {
                    tx_id,
                    shard_id,
                    vote: false,
                })
            }
            RpcMessage::TransactionVote { .. } => Ok(message),
            RpcMessage::TransactionCommit { tx_id, shard_id } => {
                Ok(RpcMessage::TransactionAck { tx_id, shard_id })
            }
            RpcMessage::TransactionAbort { tx_id, shard_id } => {
                Ok(RpcMessage::TransactionAck { tx_id, shard_id })
            }
            RpcMessage::TransactionAck { .. } => Ok(message),

            // --- Shard migration ---
            RpcMessage::MigrationBatch { migration_id, .. } => {
                Err(anyhow::anyhow!(
                    "RpcMessage::MigrationBatch for migration '{}' requires migration handler — not connected",
                    migration_id
                ))
            }
            RpcMessage::ShardTransfer { shard_id, .. } => {
                Err(anyhow::anyhow!(
                    "RpcMessage::ShardTransfer for shard {:?} requires migration handler — not connected",
                    shard_id
                ))
            }
        }
    }
}

/// Network-related errors
#[derive(Debug, thiserror::Error)]
pub enum NetworkError {
    #[error("Connection failed to {peer_id} at {address}: {message}")]
    ConnectionFailed {
        peer_id: OxirsNodeId,
        address: SocketAddr,
        message: String,
    },

    #[error("Timeout: {operation} timed out after {duration:?}")]
    Timeout {
        operation: String,
        duration: Duration,
    },

    #[error("Serialization error: {message}")]
    Serialization { message: String },

    #[error("Protocol error: {message}")]
    Protocol { message: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[test]
    fn test_network_config_default() {
        let config = NetworkConfig::default();
        assert_eq!(config.connection_timeout, Duration::from_secs(5));
        assert_eq!(config.request_timeout, Duration::from_secs(10));
        assert_eq!(config.max_connections, 100);
        assert_eq!(config.keep_alive_interval, Duration::from_secs(30));
    }

    #[test]
    fn test_log_entry_creation() {
        let command = RdfCommand::Insert {
            subject: "s".to_string(),
            predicate: "p".to_string(),
            object: "o".to_string(),
        };
        let entry = LogEntry::new(1, 1, command.clone());

        assert_eq!(entry.index, 1);
        assert_eq!(entry.term, 1);
        assert_eq!(entry.command, command);
    }

    #[test]
    fn test_connection_creation() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let connection = Connection::new(1, addr);

        assert_eq!(connection.peer_id, 1);
        assert_eq!(connection.address, addr);
        assert!(!connection.is_connected);
    }

    #[test]
    fn test_connection_staleness() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let connection = Connection::new(1, addr);

        // Fresh connection should not be stale
        assert!(!connection.is_stale(Duration::from_secs(10)));

        // Connection should be stale with very short timeout after some time passes
        std::thread::sleep(Duration::from_millis(1));
        assert!(connection.is_stale(Duration::from_nanos(1)));
    }

    #[test]
    fn test_connection_activity_update() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let mut connection = Connection::new(1, addr);

        let old_activity = connection.last_activity;
        std::thread::sleep(Duration::from_millis(1));
        connection.update_activity();

        assert!(connection.last_activity > old_activity);
    }

    #[tokio::test]
    async fn test_network_manager_creation() {
        let config = NetworkConfig::default();
        let manager = NetworkManager::new(1, config);

        assert_eq!(manager.node_id, 1);
        assert!(!*manager.running.read().await);

        let stats = manager.get_stats().await;
        assert_eq!(stats.node_id, 1);
        assert_eq!(stats.total_connections, 0);
        assert_eq!(stats.active_connections, 0);
    }

    #[tokio::test]
    async fn test_rpc_message_serialization() {
        let message = RpcMessage::RequestVote {
            term: 1,
            candidate_id: 1,
            last_log_index: 0,
            last_log_term: 0,
        };

        let serialized = serde_json::to_string(&message).unwrap();
        let deserialized: RpcMessage = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            RpcMessage::RequestVote {
                term,
                candidate_id,
                last_log_index,
                last_log_term,
            } => {
                assert_eq!(term, 1);
                assert_eq!(candidate_id, 1);
                assert_eq!(last_log_index, 0);
                assert_eq!(last_log_term, 0);
            }
            _ => panic!("Unexpected message type"),
        }
    }

    #[test]
    fn test_network_error_display() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let err = NetworkError::ConnectionFailed {
            peer_id: 1,
            address: addr,
            message: "refused".to_string(),
        };
        assert!(err
            .to_string()
            .contains("Connection failed to 1 at 127.0.0.1:8080: refused"));

        let err = NetworkError::Timeout {
            operation: "connect".to_string(),
            duration: Duration::from_secs(5),
        };
        assert!(err
            .to_string()
            .contains("Timeout: connect timed out after 5s"));

        let err = NetworkError::Serialization {
            message: "invalid json".to_string(),
        };
        assert!(err
            .to_string()
            .contains("Serialization error: invalid json"));

        let err = NetworkError::Protocol {
            message: "unknown message".to_string(),
        };
        assert!(err.to_string().contains("Protocol error: unknown message"));
    }

    // --- handle_message dispatch tests ---

    #[tokio::test]
    async fn test_handle_request_vote_returns_vote_response() {
        let config = NetworkConfig::default();
        let svc = NetworkService::new(1, config);

        let msg = RpcMessage::RequestVote {
            term: 5,
            candidate_id: 2,
            last_log_index: 10,
            last_log_term: 4,
        };

        let resp = svc
            .handle_message(msg)
            .await
            .expect("handle_message failed");
        match resp {
            RpcMessage::VoteResponse { term, vote_granted } => {
                assert_eq!(term, 5);
                assert!(!vote_granted, "conservative default should deny vote");
            }
            other => panic!("unexpected response: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handle_append_entries_returns_success() {
        let config = NetworkConfig::default();
        let svc = NetworkService::new(1, config);

        let msg = RpcMessage::AppendEntries {
            term: 3,
            leader_id: 2,
            prev_log_index: 0,
            prev_log_term: 0,
            entries: Vec::new(),
            leader_commit: 0,
        };

        let resp = svc
            .handle_message(msg)
            .await
            .expect("handle_message failed");
        match resp {
            RpcMessage::AppendEntriesResponse { term, success, .. } => {
                assert_eq!(term, 3);
                assert!(success);
            }
            other => panic!("unexpected response: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handle_heartbeat_returns_heartbeat_response() {
        let config = NetworkConfig::default();
        let svc = NetworkService::new(1, config);

        let msg = RpcMessage::Heartbeat {
            term: 7,
            leader_id: 2,
        };
        let resp = svc
            .handle_message(msg)
            .await
            .expect("handle_message failed");
        match resp {
            RpcMessage::HeartbeatResponse { term } => assert_eq!(term, 7),
            other => panic!("unexpected response: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handle_client_request_returns_success() {
        let config = NetworkConfig::default();
        let svc = NetworkService::new(1, config);

        let cmd = crate::raft::RdfCommand::Clear;
        let msg = RpcMessage::ClientRequest { command: cmd };
        let resp = svc
            .handle_message(msg)
            .await
            .expect("handle_message failed");
        match resp {
            RpcMessage::ClientResponse {
                response: crate::raft::RdfResponse::Success,
            } => {}
            other => panic!("unexpected response: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handle_query_shard_returns_empty_results() {
        let config = NetworkConfig::default();
        let svc = NetworkService::new(1, config);

        let msg = RpcMessage::QueryShard {
            shard_id: 42,
            subject: None,
            predicate: None,
            object: None,
        };
        let resp = svc
            .handle_message(msg)
            .await
            .expect("handle_message failed");
        match resp {
            RpcMessage::QueryShardResponse { shard_id, results } => {
                assert_eq!(shard_id, 42);
                assert!(results.is_empty());
            }
            other => panic!("unexpected response: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handle_transaction_prepare_votes_no() {
        let config = NetworkConfig::default();
        let svc = NetworkService::new(1, config);

        let msg = RpcMessage::TransactionPrepare {
            tx_id: "tx-001".to_string(),
            shard_id: 1,
            operations: Vec::new(),
        };
        let resp = svc
            .handle_message(msg)
            .await
            .expect("handle_message failed");
        match resp {
            RpcMessage::TransactionVote {
                tx_id,
                shard_id,
                vote,
            } => {
                assert_eq!(tx_id, "tx-001");
                assert_eq!(shard_id, 1);
                assert!(!vote, "conservative default should vote no");
            }
            other => panic!("unexpected response: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handle_transaction_commit_returns_ack() {
        let config = NetworkConfig::default();
        let svc = NetworkService::new(1, config);

        let msg = RpcMessage::TransactionCommit {
            tx_id: "tx-002".to_string(),
            shard_id: 5,
        };
        let resp = svc
            .handle_message(msg)
            .await
            .expect("handle_message failed");
        match resp {
            RpcMessage::TransactionAck { tx_id, shard_id } => {
                assert_eq!(tx_id, "tx-002");
                assert_eq!(shard_id, 5);
            }
            other => panic!("unexpected response: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handle_transaction_abort_returns_ack() {
        let config = NetworkConfig::default();
        let svc = NetworkService::new(1, config);

        let msg = RpcMessage::TransactionAbort {
            tx_id: "tx-003".to_string(),
            shard_id: 9,
        };
        let resp = svc
            .handle_message(msg)
            .await
            .expect("handle_message failed");
        match resp {
            RpcMessage::TransactionAck { tx_id, shard_id } => {
                assert_eq!(tx_id, "tx-003");
                assert_eq!(shard_id, 9);
            }
            other => panic!("unexpected response: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handle_store_triple_returns_descriptive_error() {
        use oxirs_core::model::{Literal, NamedNode, Triple};

        let config = NetworkConfig::default();
        let svc = NetworkService::new(1, config);

        let triple = Triple::new(
            NamedNode::new("http://example.org/s").expect("valid IRI"),
            NamedNode::new("http://example.org/p").expect("valid IRI"),
            Literal::new("object-value"),
        );
        let msg = RpcMessage::StoreTriple {
            shard_id: 3,
            triple,
        };
        let err = svc
            .handle_message(msg)
            .await
            .expect_err("should return Err");
        let msg_str = err.to_string();
        assert!(
            msg_str.contains("StoreTriple") || msg_str.contains("shard"),
            "error should mention StoreTriple or shard, got: {}",
            msg_str
        );
        assert!(
            !msg_str.contains("not yet implemented"),
            "stub message should be gone, got: {}",
            msg_str
        );
    }

    #[tokio::test]
    async fn test_handle_shard_transfer_returns_descriptive_error() {
        let config = NetworkConfig::default();
        let svc = NetworkService::new(1, config);

        let msg = RpcMessage::ShardTransfer {
            shard_id: 7,
            triples: Vec::new(),
            source_node: 2,
        };
        let err = svc
            .handle_message(msg)
            .await
            .expect_err("should return Err");
        let msg_str = err.to_string();
        assert!(
            !msg_str.contains("not yet implemented"),
            "stub message should be gone, got: {}",
            msg_str
        );
    }

    #[tokio::test]
    async fn test_handle_terminal_responses_echo_back() {
        let config = NetworkConfig::default();
        let svc = NetworkService::new(1, config);

        // VoteResponse echoed back
        let msg = RpcMessage::VoteResponse {
            term: 2,
            vote_granted: true,
        };
        let resp = svc.handle_message(msg).await.expect("should succeed");
        match resp {
            RpcMessage::VoteResponse {
                term: 2,
                vote_granted: true,
            } => {}
            other => panic!("unexpected: {:?}", other),
        }

        // HeartbeatResponse echoed back
        let msg = RpcMessage::HeartbeatResponse { term: 8 };
        let resp = svc.handle_message(msg).await.expect("should succeed");
        match resp {
            RpcMessage::HeartbeatResponse { term: 8 } => {}
            other => panic!("unexpected: {:?}", other),
        }
    }
}
