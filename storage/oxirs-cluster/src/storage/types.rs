//! Storage type definitions

use crate::network::LogEntry;
use crate::raft::{OxirsNodeId, RdfApp};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

/// Persistent state required by Raft
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RaftState {
    /// Current term
    pub current_term: u64,
    /// Candidate ID that received vote in current term
    pub voted_for: Option<OxirsNodeId>,
    /// Log entries
    pub log: Vec<LogEntry>,
    /// Index of highest log entry known to be committed
    pub commit_index: u64,
    /// Index of highest log entry applied to state machine
    pub last_applied: u64,
}

/// Snapshot metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    /// Last included index
    pub last_included_index: u64,
    /// Last included term
    pub last_included_term: u64,
    /// Cluster configuration at the time of snapshot
    pub configuration: Vec<OxirsNodeId>,
    /// Timestamp when snapshot was created
    pub timestamp: u64,
    /// Size of the snapshot data in bytes
    pub size: u64,
    /// Checksum for corruption detection
    pub checksum: String,
}

/// Write-Ahead Log entry for atomic operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEntry {
    /// Sequence number
    pub sequence: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Operation type
    pub operation: WalOperation,
    /// Checksum of the operation data
    pub checksum: String,
}

/// WAL operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalOperation {
    /// Write Raft state
    WriteRaftState(RaftState),
    /// Write application state
    WriteAppState(RdfApp),
    /// Create snapshot
    CreateSnapshot(SnapshotMetadata),
    /// Truncate log
    TruncateLog(u64),
    /// Commit operation (mark previous operations as durable)
    Commit(u64),
}

/// Data file with corruption detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChecksummedData<T> {
    /// The actual data
    pub data: T,
    /// SHA-256 checksum
    pub checksum: String,
    /// Timestamp when written
    pub timestamp: u64,
}

impl<T> ChecksummedData<T>
where
    T: Serialize,
{
    pub fn new(data: T) -> Result<Self> {
        let data_bytes = oxicode::serde::encode_to_vec(&data, oxicode::config::standard())?;
        let mut hasher = Sha256::new();
        hasher.update(&data_bytes);
        let checksum = format!("{:x}", hasher.finalize());

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs();

        Ok(Self {
            data,
            checksum,
            timestamp,
        })
    }

    pub fn verify(&self) -> Result<bool> {
        let data_bytes = oxicode::serde::encode_to_vec(&self.data, oxicode::config::standard())?;
        let mut hasher = Sha256::new();
        hasher.update(&data_bytes);
        let computed_checksum = format!("{:x}", hasher.finalize());
        Ok(computed_checksum == self.checksum)
    }
}
