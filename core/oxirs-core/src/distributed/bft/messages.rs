//! BFT message types and protocol definitions

use super::types::*;
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// BFT message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BftMessage {
    /// Client request
    Request {
        client_id: String,
        operation: RdfOperation,
        timestamp: SystemTime,
        signature: Option<Vec<u8>>,
    },

    /// Pre-prepare message from primary
    PrePrepare {
        view: ViewNumber,
        sequence: SequenceNumber,
        digest: Vec<u8>,
        request: Box<BftMessage>,
    },

    /// Prepare message from backups
    Prepare {
        view: ViewNumber,
        sequence: SequenceNumber,
        digest: Vec<u8>,
        node_id: NodeId,
    },

    /// Commit message
    Commit {
        view: ViewNumber,
        sequence: SequenceNumber,
        digest: Vec<u8>,
        node_id: NodeId,
    },

    /// Reply to client
    Reply {
        view: ViewNumber,
        sequence: SequenceNumber,
        client_id: String,
        result: OperationResult,
        timestamp: SystemTime,
    },

    /// Checkpoint message
    Checkpoint {
        sequence: SequenceNumber,
        state_digest: Vec<u8>,
        node_id: NodeId,
    },

    /// View change message
    ViewChange {
        new_view: ViewNumber,
        node_id: NodeId,
        last_sequence: SequenceNumber,
        checkpoints: Vec<CheckpointProof>,
        prepared_messages: Vec<PreparedProof>,
    },

    /// New view message from new primary
    NewView {
        view: ViewNumber,
        view_changes: Vec<BftMessage>,
        pre_prepares: Vec<BftMessage>,
    },
}

impl BftMessage {
    /// Get the digest of a message for verification
    pub fn digest(&self) -> Vec<u8> {
        use sha2::{Digest, Sha256};
        let serialized =
            bincode::serde::encode_to_vec(self, bincode::config::standard()).unwrap_or_default();
        Sha256::digest(&serialized).to_vec()
    }

    /// Extract view number from message (if applicable)
    pub fn view(&self) -> Option<ViewNumber> {
        match self {
            BftMessage::PrePrepare { view, .. }
            | BftMessage::Prepare { view, .. }
            | BftMessage::Commit { view, .. }
            | BftMessage::Reply { view, .. }
            | BftMessage::NewView { view, .. } => Some(*view),
            BftMessage::ViewChange { new_view, .. } => Some(*new_view),
            _ => None,
        }
    }

    /// Extract sequence number from message (if applicable)
    pub fn sequence(&self) -> Option<SequenceNumber> {
        match self {
            BftMessage::PrePrepare { sequence, .. }
            | BftMessage::Prepare { sequence, .. }
            | BftMessage::Commit { sequence, .. }
            | BftMessage::Reply { sequence, .. } => Some(*sequence),
            BftMessage::Checkpoint { sequence, .. } => Some(*sequence),
            BftMessage::ViewChange { last_sequence, .. } => Some(*last_sequence),
            _ => None,
        }
    }

    /// Extract node ID from message (if applicable)
    pub fn node_id(&self) -> Option<NodeId> {
        match self {
            BftMessage::Prepare { node_id, .. }
            | BftMessage::Commit { node_id, .. }
            | BftMessage::Checkpoint { node_id, .. }
            | BftMessage::ViewChange { node_id, .. } => Some(*node_id),
            _ => None,
        }
    }

    /// Check if message is a consensus message
    pub fn is_consensus_message(&self) -> bool {
        matches!(
            self,
            BftMessage::PrePrepare { .. } | BftMessage::Prepare { .. } | BftMessage::Commit { .. }
        )
    }

    /// Check if message is a view change related message
    pub fn is_view_change_message(&self) -> bool {
        matches!(
            self,
            BftMessage::ViewChange { .. } | BftMessage::NewView { .. }
        )
    }
}
