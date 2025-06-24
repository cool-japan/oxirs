//! # Consensus Protocol
//!
//! Consensus protocol implementation for distributed agreement.

use anyhow::Result;
use crate::raft::{RaftNode, LogEntry};

/// Consensus manager
pub struct ConsensusManager {
    raft_node: RaftNode,
    peers: Vec<u64>,
}

impl ConsensusManager {
    pub fn new(node_id: u64, peers: Vec<u64>) -> Self {
        Self {
            raft_node: RaftNode::new(node_id),
            peers,
        }
    }
    
    pub fn is_leader(&self) -> bool {
        self.raft_node.is_leader()
    }
    
    pub async fn propose(&mut self, _data: Vec<u8>) -> Result<()> {
        if !self.is_leader() {
            return Err(anyhow::anyhow!("Not the leader"));
        }
        // TODO: Implement proposal mechanism
        Ok(())
    }
    
    pub async fn apply_entry(&mut self, _entry: LogEntry) -> Result<()> {
        // TODO: Implement entry application
        Ok(())
    }
    
    pub fn get_peers(&self) -> &[u64] {
        &self.peers
    }
    
    pub fn add_peer(&mut self, peer_id: u64) {
        if !self.peers.contains(&peer_id) {
            self.peers.push(peer_id);
        }
    }
    
    pub fn remove_peer(&mut self, peer_id: u64) {
        self.peers.retain(|&id| id != peer_id);
    }
}