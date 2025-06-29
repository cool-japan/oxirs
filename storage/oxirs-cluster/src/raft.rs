//! # Raft Consensus Implementation
//!
//! Raft consensus algorithm implementation for distributed RDF storage.
//! Uses openraft for production-ready consensus.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;
use std::sync::Arc;
use tokio::sync::RwLock;

#[cfg(feature = "raft")]
use openraft::{
    BasicNode, Config, Entry, EntryPayload, LogId, Membership, Node, NodeId, Raft, RaftMetrics,
    SnapshotMeta, StorageError,
};

/// Node ID type for Raft
pub type OxirsNodeId = u64;

/// Raft request ID type
pub type OxirsRequestId = u64;

/// RDF command types that can be replicated
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RdfCommand {
    /// Insert a triple
    Insert {
        subject: String,
        predicate: String,
        object: String,
    },
    /// Delete a triple  
    Delete {
        subject: String,
        predicate: String,
        object: String,
    },
    /// Clear all triples
    Clear,
    /// Begin transaction
    BeginTransaction { tx_id: String },
    /// Commit transaction
    CommitTransaction { tx_id: String },
    /// Rollback transaction
    RollbackTransaction { tx_id: String },
    /// Add a new node to the cluster
    AddNode {
        node_id: OxirsNodeId,
        address: String,
    },
    /// Remove a node from the cluster
    RemoveNode { node_id: OxirsNodeId },
    /// Transfer leadership to another node
    TransferLeadership { target_node: OxirsNodeId },
    /// Force evict a non-responsive node
    ForceEvictNode { node_id: OxirsNodeId },
}

/// RDF response from command execution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RdfResponse {
    /// Operation successful
    Success,
    /// Operation failed
    Error(String),
    /// Transaction started
    TransactionStarted { tx_id: String },
    /// Transaction committed
    TransactionCommitted { tx_id: String },
    /// Transaction rolled back
    TransactionRolledBack { tx_id: String },
}

/// Raft application data
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RdfApp {
    /// In-memory triple store for demonstration
    /// In production, this would interface with oxirs-tdb
    pub triples: BTreeSet<(String, String, String)>,
    /// Active transactions
    pub transactions: BTreeMap<String, BTreeSet<(String, String, String)>>,
}

impl Default for RdfApp {
    fn default() -> Self {
        Self {
            triples: BTreeSet::new(),
            transactions: BTreeMap::new(),
        }
    }
}

impl RdfApp {
    /// Apply a command to the state machine
    pub fn apply_command(&mut self, cmd: &RdfCommand) -> RdfResponse {
        match cmd {
            RdfCommand::Insert {
                subject,
                predicate,
                object,
            } => {
                self.triples
                    .insert((subject.clone(), predicate.clone(), object.clone()));
                RdfResponse::Success
            }
            RdfCommand::Delete {
                subject,
                predicate,
                object,
            } => {
                self.triples
                    .remove(&(subject.clone(), predicate.clone(), object.clone()));
                RdfResponse::Success
            }
            RdfCommand::Clear => {
                self.triples.clear();
                RdfResponse::Success
            }
            RdfCommand::BeginTransaction { tx_id } => {
                self.transactions.insert(tx_id.clone(), BTreeSet::new());
                RdfResponse::TransactionStarted {
                    tx_id: tx_id.clone(),
                }
            }
            RdfCommand::CommitTransaction { tx_id } => {
                if let Some(tx_triples) = self.transactions.remove(tx_id) {
                    self.triples.extend(tx_triples);
                    RdfResponse::TransactionCommitted {
                        tx_id: tx_id.clone(),
                    }
                } else {
                    RdfResponse::Error(format!("Transaction {} not found", tx_id))
                }
            }
            RdfCommand::RollbackTransaction { tx_id } => {
                if self.transactions.remove(tx_id).is_some() {
                    RdfResponse::TransactionRolledBack {
                        tx_id: tx_id.clone(),
                    }
                } else {
                    RdfResponse::Error(format!("Transaction {} not found", tx_id))
                }
            }
            RdfCommand::AddNode { node_id, address } => {
                // Log the configuration change
                tracing::info!("Adding node {} at address {} to cluster", node_id, address);
                RdfResponse::Success
            }
            RdfCommand::RemoveNode { node_id } => {
                // Log the configuration change
                tracing::info!("Removing node {} from cluster", node_id);
                RdfResponse::Success
            }
            RdfCommand::TransferLeadership { target_node } => {
                // Log the leadership transfer
                tracing::info!("Transferring leadership to node {}", target_node);
                RdfResponse::Success
            }
            RdfCommand::ForceEvictNode { node_id } => {
                // Log the forced eviction
                tracing::warn!("Force evicting node {} from cluster", node_id);
                RdfResponse::Success
            }
        }
    }

    /// Get number of triples
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Check if store is empty
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Query triples by pattern (simplified)
    pub fn query(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Vec<(String, String, String)> {
        self.triples
            .iter()
            .filter(|(s, p, o)| {
                subject.map_or(true, |subj| s == subj)
                    && predicate.map_or(true, |pred| p == pred)
                    && object.map_or(true, |obj| o == obj)
            })
            .cloned()
            .collect()
    }
}

#[cfg(feature = "raft")]
mod raft_impl {
    use super::*;
    use openraft::{
        error::{AppendEntriesError, InstallSnapshotError, VoteError},
        raft::{
            AppendEntriesRequest, AppendEntriesResponse, InstallSnapshotRequest,
            InstallSnapshotResponse, VoteRequest, VoteResponse,
        },
        storage::{LogState, Snapshot},
        AppData, AppDataResponse, RaftLogReader, RaftSnapshotBuilder, RaftStorage,
    };
    use std::io::Cursor;

    /// Raft type configuration for OxiRS
    #[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Ord, PartialOrd)]
    pub struct OxirsTypeConfig;

    impl openraft::RaftTypeConfig for OxirsTypeConfig {
        type D = RdfCommand;
        type R = RdfResponse;
        type NodeId = OxirsNodeId;
        type Node = BasicNode;
        type Entry = Entry<Self>;
        type SnapshotData = Cursor<Vec<u8>>;
        type AsyncRuntime = openraft::TokioRuntime;
    }

    /// Raft storage implementation
    pub struct OxirsStorage {
        /// Current Raft state
        pub state: Arc<RwLock<RdfApp>>,
        /// Persistent log entries
        pub log: Arc<RwLock<Vec<Entry<OxirsTypeConfig>>>>,
        /// Hard state (term, vote, committed)
        pub hard_state: Arc<RwLock<(u64, Option<OxirsNodeId>, Option<LogId<OxirsNodeId>>)>>,
        /// Last applied log index
        pub last_applied: Arc<RwLock<Option<LogId<OxirsNodeId>>>>,
        /// Current snapshot
        pub snapshot: Arc<RwLock<Option<Snapshot<OxirsTypeConfig>>>>,
    }

    impl OxirsStorage {
        pub fn new() -> Self {
            Self {
                state: Arc::new(RwLock::new(RdfApp::default())),
                log: Arc::new(RwLock::new(Vec::new())),
                hard_state: Arc::new(RwLock::new((0, None, None))),
                last_applied: Arc::new(RwLock::new(None)),
                snapshot: Arc::new(RwLock::new(None)),
            }
        }
    }

    #[async_trait::async_trait]
    impl RaftLogReader<OxirsTypeConfig> for OxirsStorage {
        async fn try_get_log_entries<RB: openraft::RaftLogReaderExt<OxirsTypeConfig> + Send>(
            &mut self,
            range: std::ops::Range<u64>,
        ) -> Result<Vec<Entry<OxirsTypeConfig>>, StorageError<OxirsNodeId>> {
            let log = self.log.read().await;
            let entries = log
                .iter()
                .filter(|entry| range.contains(&entry.log_id.index))
                .cloned()
                .collect();
            Ok(entries)
        }
    }

    #[async_trait::async_trait]
    impl RaftSnapshotBuilder<OxirsTypeConfig> for OxirsStorage {
        async fn build_snapshot(
            &mut self,
        ) -> Result<Snapshot<OxirsTypeConfig>, StorageError<OxirsNodeId>> {
            let state = self.state.read().await;
            let last_applied = *self.last_applied.read().await;

            let data =
                serde_json::to_vec(&*state).map_err(|e| StorageError::read_state_machine(&e))?;

            let snapshot = Snapshot {
                meta: SnapshotMeta {
                    last_log_id: last_applied,
                    last_membership: Membership::new(vec![BTreeSet::new()], None),
                    snapshot_id: format!("snapshot-{}", last_applied.map_or(0, |id| id.index)),
                },
                snapshot: Box::new(Cursor::new(data)),
            };

            *self.snapshot.write().await = Some(snapshot.clone());
            Ok(snapshot)
        }
    }

    #[async_trait::async_trait]
    impl RaftStorage<OxirsTypeConfig> for OxirsStorage {
        type LogReader = Self;
        type SnapshotBuilder = Self;

        async fn save_committed(
            &mut self,
            committed: Option<LogId<OxirsNodeId>>,
        ) -> Result<(), StorageError<OxirsNodeId>> {
            self.hard_state.write().await.2 = committed;
            Ok(())
        }

        async fn read_committed(
            &mut self,
        ) -> Result<Option<LogId<OxirsNodeId>>, StorageError<OxirsNodeId>> {
            Ok(self.hard_state.read().await.2)
        }

        async fn save_hard_state(
            &mut self,
            hs: &openraft::storage::HardState<OxirsNodeId>,
        ) -> Result<(), StorageError<OxirsNodeId>> {
            let mut hard_state = self.hard_state.write().await;
            hard_state.0 = hs.current_term;
            hard_state.1 = hs.voted_for;
            Ok(())
        }

        async fn read_hard_state(
            &mut self,
        ) -> Result<Option<openraft::storage::HardState<OxirsNodeId>>, StorageError<OxirsNodeId>>
        {
            let hard_state = self.hard_state.read().await;
            Ok(Some(openraft::storage::HardState {
                current_term: hard_state.0,
                voted_for: hard_state.1,
            }))
        }

        async fn get_log_reader(&mut self) -> Self::LogReader {
            self.clone()
        }

        async fn append_to_log<I>(&mut self, entries: I) -> Result<(), StorageError<OxirsNodeId>>
        where
            I: IntoIterator<Item = Entry<OxirsTypeConfig>> + Send,
        {
            let mut log = self.log.write().await;
            log.extend(entries);
            Ok(())
        }

        async fn delete_conflict_logs_since(
            &mut self,
            log_id: LogId<OxirsNodeId>,
        ) -> Result<(), StorageError<OxirsNodeId>> {
            let mut log = self.log.write().await;
            log.retain(|entry| entry.log_id.index < log_id.index);
            Ok(())
        }

        async fn purge_logs_upto(
            &mut self,
            log_id: LogId<OxirsNodeId>,
        ) -> Result<(), StorageError<OxirsNodeId>> {
            let mut log = self.log.write().await;
            log.retain(|entry| entry.log_id.index > log_id.index);
            Ok(())
        }

        async fn last_applied_state(
            &mut self,
        ) -> Result<
            (
                Option<LogId<OxirsNodeId>>,
                openraft::storage::StoredMembership<OxirsNodeId, BasicNode>,
            ),
            StorageError<OxirsNodeId>,
        > {
            let last_applied = *self.last_applied.read().await;
            let membership = openraft::storage::StoredMembership::new(
                None,
                Membership::new(vec![BTreeSet::new()], None),
            );
            Ok((last_applied, membership))
        }

        async fn apply_to_state_machine(
            &mut self,
            entries: &[Entry<OxirsTypeConfig>],
        ) -> Result<Vec<RdfResponse>, StorageError<OxirsNodeId>> {
            let mut responses = Vec::new();
            let mut state = self.state.write().await;

            for entry in entries {
                if let EntryPayload::Normal(cmd) = &entry.payload {
                    let response = state.apply_command(cmd);
                    responses.push(response);
                }
                *self.last_applied.write().await = Some(entry.log_id);
            }

            Ok(responses)
        }

        async fn get_snapshot_builder(&mut self) -> Self::SnapshotBuilder {
            self.clone()
        }

        async fn begin_receiving_snapshot(
            &mut self,
        ) -> Result<Box<Cursor<Vec<u8>>>, StorageError<OxirsNodeId>> {
            Ok(Box::new(Cursor::new(Vec::new())))
        }

        async fn install_snapshot(
            &mut self,
            meta: &SnapshotMeta<OxirsNodeId, BasicNode>,
            snapshot: Box<Cursor<Vec<u8>>>,
        ) -> Result<(), StorageError<OxirsNodeId>> {
            let data = snapshot.get_ref();
            let new_state: RdfApp =
                serde_json::from_slice(data).map_err(|e| StorageError::read_state_machine(&e))?;

            *self.state.write().await = new_state;
            *self.last_applied.write().await = meta.last_log_id;

            Ok(())
        }

        async fn get_current_snapshot(
            &mut self,
        ) -> Result<Option<Snapshot<OxirsTypeConfig>>, StorageError<OxirsNodeId>> {
            Ok(self.snapshot.read().await.clone())
        }
    }

    impl Clone for OxirsStorage {
        fn clone(&self) -> Self {
            Self {
                state: Arc::clone(&self.state),
                log: Arc::clone(&self.log),
                hard_state: Arc::clone(&self.hard_state),
                last_applied: Arc::clone(&self.last_applied),
                snapshot: Arc::clone(&self.snapshot),
            }
        }
    }
}

#[cfg(feature = "raft")]
pub use raft_impl::*;

/// Raft node implementation
#[derive(Debug)]
pub struct RaftNode {
    node_id: OxirsNodeId,
    #[cfg(feature = "raft")]
    raft: Option<Raft<OxirsTypeConfig>>,
    storage: Arc<RwLock<RdfApp>>,
}

impl RaftNode {
    pub fn new(node_id: OxirsNodeId) -> Self {
        Self {
            node_id,
            #[cfg(feature = "raft")]
            raft: None,
            storage: Arc::new(RwLock::new(RdfApp::default())),
        }
    }

    /// Initialize Raft with storage
    #[cfg(feature = "raft")]
    pub async fn init_raft(&mut self, peers: BTreeSet<OxirsNodeId>) -> Result<()> {
        let config = Config::default();
        let storage = OxirsStorage::new();

        let raft = Raft::new(self.node_id, config, storage, BasicNode::default())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create Raft: {}", e))?;

        // Initialize cluster membership
        if !peers.is_empty() {
            let mut nodes = BTreeMap::new();
            for peer in &peers {
                nodes.insert(*peer, BasicNode::default());
            }
            nodes.insert(self.node_id, BasicNode::default());

            let membership = Membership::new(vec![peers], None);
            raft.initialize(membership)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to initialize Raft: {}", e))?;
        }

        self.raft = Some(raft);
        Ok(())
    }

    /// Check if this node is the leader
    pub async fn is_leader(&self) -> bool {
        #[cfg(feature = "raft")]
        {
            if let Some(ref raft) = self.raft {
                match raft.metrics().borrow().current_leader {
                    Some(leader) => leader == self.node_id,
                    None => false,
                }
            } else {
                false
            }
        }
        #[cfg(not(feature = "raft"))]
        {
            // Fallback for non-raft mode
            true
        }
    }

    /// Get current term
    pub async fn current_term(&self) -> u64 {
        #[cfg(feature = "raft")]
        {
            if let Some(ref raft) = self.raft {
                raft.metrics().borrow().current_term
            } else {
                0
            }
        }
        #[cfg(not(feature = "raft"))]
        {
            0
        }
    }

    /// Submit a command for replication
    pub async fn submit_command(&self, cmd: RdfCommand) -> Result<RdfResponse> {
        #[cfg(feature = "raft")]
        {
            if let Some(ref raft) = self.raft {
                let response = raft
                    .client_write(cmd)
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to submit command: {}", e))?;
                Ok(response.data)
            } else {
                // Fallback to local execution
                let mut state = self.storage.write().await;
                Ok(state.apply_command(&cmd))
            }
        }
        #[cfg(not(feature = "raft"))]
        {
            // Fallback to local execution
            let mut state = self.storage.write().await;
            Ok(state.apply_command(&cmd))
        }
    }

    /// Get metrics
    #[cfg(feature = "raft")]
    pub async fn get_metrics(&self) -> Option<RaftMetrics<OxirsNodeId, BasicNode>> {
        self.raft
            .as_ref()
            .map(|raft| raft.metrics().borrow().clone())
    }

    /// Query the local state machine
    pub async fn query(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Vec<(String, String, String)> {
        let state = self.storage.read().await;
        state.query(subject, predicate, object)
    }

    /// Get number of triples
    pub async fn len(&self) -> usize {
        let state = self.storage.read().await;
        state.len()
    }

    /// Check if store is empty
    pub async fn is_empty(&self) -> bool {
        let state = self.storage.read().await;
        state.is_empty()
    }

    /// Shutdown the raft node gracefully
    pub async fn shutdown(&mut self) -> Result<()> {
        tracing::info!("Shutting down raft node {}", self.node_id);

        #[cfg(feature = "raft")]
        {
            if let Some(raft) = self.raft.take() {
                // Shutdown the raft instance
                raft.shutdown()
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to shutdown raft: {}", e))?;
                tracing::info!("Raft instance shutdown completed");
            }
        }

        // Clear storage reference
        {
            let mut storage = self.storage.write().await;
            *storage = RdfApp::default();
        }

        tracing::info!("Raft node {} shutdown completed", self.node_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rdf_command_serialization() {
        let cmd = RdfCommand::Insert {
            subject: "s".to_string(),
            predicate: "p".to_string(),
            object: "o".to_string(),
        };

        let serialized = serde_json::to_string(&cmd).unwrap();
        let deserialized: RdfCommand = serde_json::from_str(&serialized).unwrap();

        assert_eq!(cmd, deserialized);
    }

    #[test]
    fn test_rdf_response_serialization() {
        let response = RdfResponse::Success;

        let serialized = serde_json::to_string(&response).unwrap();
        let deserialized: RdfResponse = serde_json::from_str(&serialized).unwrap();

        assert_eq!(response, deserialized);
    }

    #[test]
    fn test_rdf_app_apply_insert() {
        let mut app = RdfApp::default();

        let cmd = RdfCommand::Insert {
            subject: "s".to_string(),
            predicate: "p".to_string(),
            object: "o".to_string(),
        };

        let response = app.apply_command(&cmd);
        assert_eq!(response, RdfResponse::Success);
        assert_eq!(app.len(), 1);
        assert!(!app.is_empty());
    }

    #[test]
    fn test_rdf_app_apply_delete() {
        let mut app = RdfApp::default();

        // Insert first
        let insert_cmd = RdfCommand::Insert {
            subject: "s".to_string(),
            predicate: "p".to_string(),
            object: "o".to_string(),
        };
        app.apply_command(&insert_cmd);
        assert_eq!(app.len(), 1);

        // Then delete
        let delete_cmd = RdfCommand::Delete {
            subject: "s".to_string(),
            predicate: "p".to_string(),
            object: "o".to_string(),
        };
        let response = app.apply_command(&delete_cmd);
        assert_eq!(response, RdfResponse::Success);
        assert_eq!(app.len(), 0);
        assert!(app.is_empty());
    }

    #[test]
    fn test_rdf_app_apply_clear() {
        let mut app = RdfApp::default();

        // Insert some data
        app.apply_command(&RdfCommand::Insert {
            subject: "s1".to_string(),
            predicate: "p1".to_string(),
            object: "o1".to_string(),
        });
        app.apply_command(&RdfCommand::Insert {
            subject: "s2".to_string(),
            predicate: "p2".to_string(),
            object: "o2".to_string(),
        });
        assert_eq!(app.len(), 2);

        // Clear all
        let response = app.apply_command(&RdfCommand::Clear);
        assert_eq!(response, RdfResponse::Success);
        assert_eq!(app.len(), 0);
        assert!(app.is_empty());
    }

    #[test]
    fn test_rdf_app_transactions() {
        let mut app = RdfApp::default();
        let tx_id = "tx1".to_string();

        // Begin transaction
        let response = app.apply_command(&RdfCommand::BeginTransaction {
            tx_id: tx_id.clone(),
        });
        assert_eq!(
            response,
            RdfResponse::TransactionStarted {
                tx_id: tx_id.clone()
            }
        );
        assert!(app.transactions.contains_key(&tx_id));

        // Commit transaction
        let response = app.apply_command(&RdfCommand::CommitTransaction {
            tx_id: tx_id.clone(),
        });
        assert_eq!(
            response,
            RdfResponse::TransactionCommitted {
                tx_id: tx_id.clone()
            }
        );
        assert!(!app.transactions.contains_key(&tx_id));
    }

    #[test]
    fn test_rdf_app_transaction_rollback() {
        let mut app = RdfApp::default();
        let tx_id = "tx1".to_string();

        // Begin transaction
        app.apply_command(&RdfCommand::BeginTransaction {
            tx_id: tx_id.clone(),
        });
        assert!(app.transactions.contains_key(&tx_id));

        // Rollback transaction
        let response = app.apply_command(&RdfCommand::RollbackTransaction {
            tx_id: tx_id.clone(),
        });
        assert_eq!(
            response,
            RdfResponse::TransactionRolledBack {
                tx_id: tx_id.clone()
            }
        );
        assert!(!app.transactions.contains_key(&tx_id));
    }

    #[test]
    fn test_rdf_app_query() {
        let mut app = RdfApp::default();

        // Insert test data
        app.apply_command(&RdfCommand::Insert {
            subject: "s1".to_string(),
            predicate: "p1".to_string(),
            object: "o1".to_string(),
        });
        app.apply_command(&RdfCommand::Insert {
            subject: "s1".to_string(),
            predicate: "p2".to_string(),
            object: "o2".to_string(),
        });
        app.apply_command(&RdfCommand::Insert {
            subject: "s2".to_string(),
            predicate: "p1".to_string(),
            object: "o3".to_string(),
        });

        // Query all triples
        let results = app.query(None, None, None);
        assert_eq!(results.len(), 3);

        // Query by subject
        let results = app.query(Some("s1"), None, None);
        assert_eq!(results.len(), 2);

        // Query by predicate
        let results = app.query(None, Some("p1"), None);
        assert_eq!(results.len(), 2);

        // Query by object
        let results = app.query(None, None, Some("o1"));
        assert_eq!(results.len(), 1);

        // Query specific triple
        let results = app.query(Some("s1"), Some("p1"), Some("o1"));
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0],
            ("s1".to_string(), "p1".to_string(), "o1".to_string())
        );
    }

    #[tokio::test]
    async fn test_raft_node_creation() {
        let node = RaftNode::new(1);
        assert_eq!(node.node_id, 1);
        // In non-raft mode (default for tests), node always returns true for is_leader
        #[cfg(not(feature = "raft"))]
        assert!(node.is_leader().await);
        #[cfg(feature = "raft")]
        assert!(!node.is_leader().await);
        assert_eq!(node.current_term().await, 0);
        assert_eq!(node.len().await, 0);
        assert!(node.is_empty().await);
    }

    #[tokio::test]
    async fn test_raft_node_local_operations() {
        let node = RaftNode::new(1);

        // Test insert command without Raft
        let cmd = RdfCommand::Insert {
            subject: "s".to_string(),
            predicate: "p".to_string(),
            object: "o".to_string(),
        };

        let response = node.submit_command(cmd).await.unwrap();
        assert_eq!(response, RdfResponse::Success);
        assert_eq!(node.len().await, 1);
        assert!(!node.is_empty().await);

        // Test query
        let results = node.query(Some("s"), Some("p"), Some("o")).await;
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0],
            ("s".to_string(), "p".to_string(), "o".to_string())
        );
    }
}
