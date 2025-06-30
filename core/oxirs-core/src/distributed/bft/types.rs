//! Core types and configuration for Byzantine Fault Tolerance

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Node identifier in the BFT cluster
pub type NodeId = u64;

/// View number (incremented on view changes)
pub type ViewNumber = u64;

/// Sequence number for operations
pub type SequenceNumber = u64;

/// Byzantine fault tolerance configuration
#[derive(Debug, Clone)]
pub struct BftConfig {
    /// Number of tolerated Byzantine failures (f)
    /// System can tolerate f Byzantine nodes out of 3f+1 total nodes
    pub fault_tolerance: usize,

    /// View change timeout
    pub view_change_timeout: Duration,

    /// Message timeout for consensus rounds
    pub message_timeout: Duration,

    /// Checkpoint interval (number of operations)
    pub checkpoint_interval: u64,

    /// Maximum log size before compaction
    pub max_log_size: usize,

    /// Enable cryptographic signatures
    pub enable_signatures: bool,
}

impl Default for BftConfig {
    fn default() -> Self {
        Self {
            fault_tolerance: 1, // Tolerate 1 Byzantine node (requires 4 total nodes)
            view_change_timeout: Duration::from_secs(10),
            message_timeout: Duration::from_secs(5),
            checkpoint_interval: 100,
            max_log_size: 10_000,
            enable_signatures: true,
        }
    }
}

/// BFT consensus phases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    /// Initial phase
    Idle,
    /// Pre-prepare phase (primary broadcasts)
    PrePrepare,
    /// Prepare phase (backup nodes agree)
    Prepare,
    /// Commit phase (nodes commit)
    Commit,
    /// View change in progress
    ViewChange,
}

/// RDF operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RdfOperation {
    /// Insert a triple
    Insert(SerializableTriple),
    /// Remove a triple
    Remove(SerializableTriple),
    /// Batch insert
    BatchInsert(Vec<SerializableTriple>),
    /// Batch remove
    BatchRemove(Vec<SerializableTriple>),
    /// Read query (doesn't change state)
    Query(String),
}

/// Serializable triple for network transmission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub object_type: ObjectType,
}

/// Object type for serializable triples
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ObjectType {
    NamedNode,
    BlankNode,
    Literal {
        datatype: Option<String>,
        language: Option<String>,
    },
}

/// Operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationResult {
    Success,
    Failure(String),
    QueryResult(Vec<SerializableTriple>),
}

/// Checkpoint proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointProof {
    pub sequence: SequenceNumber,
    pub state_digest: Vec<u8>,
    pub signatures: HashMap<NodeId, Vec<u8>>,
}

/// Prepared proof for view changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedProof {
    pub view: ViewNumber,
    pub sequence: SequenceNumber,
    pub digest: Vec<u8>,
    pub pre_prepare: Box<super::messages::BftMessage>,
    pub prepares: Vec<super::messages::BftMessage>,
}

/// Node information
#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub id: NodeId,
    pub address: String,
    pub public_key: Option<Vec<u8>>,
}

/// Threat level assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreatLevel {
    Low,
    Medium,
    High,
    Critical,
}