//! Storage statistics

use crate::raft::OxirsNodeId;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub node_id: OxirsNodeId,
    pub data_dir: PathBuf,
    pub log_entries: usize,
    pub current_term: u64,
    pub commit_index: u64,
    pub last_applied: u64,
    pub triple_count: usize,
    pub disk_usage_bytes: u64,
}
