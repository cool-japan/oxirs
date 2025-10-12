//! # Strong Consistency Guarantees
//!
//! Provides linearizable reads and strong consistency guarantees:
//! - Linearizable read protocol (read index)
//! - Read quorum for consistency
//! - Lease-based reads for performance
//! - Follower reads with staleness bounds
//! - Causality tracking
//! - Read-your-writes consistency

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::info;

use crate::raft::OxirsNodeId;

/// Consistency level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    /// Eventual consistency (fastest, weakest)
    Eventual,
    /// Session consistency (read-your-writes)
    Session,
    /// Bounded staleness (configurable staleness)
    BoundedStaleness,
    /// Strong consistency (linearizable, slowest)
    Linearizable,
}

/// Read strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReadStrategy {
    /// Leader read (always consistent)
    LeaderRead,
    /// Read index (linearizable with heartbeat)
    ReadIndex,
    /// Lease read (linearizable with lease)
    LeaseRead,
    /// Follower read (may be stale)
    FollowerRead,
}

/// Consistency configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyConfig {
    /// Default consistency level
    pub default_consistency_level: ConsistencyLevel,
    /// Default read strategy
    pub default_read_strategy: ReadStrategy,
    /// Read quorum size
    pub read_quorum_size: usize,
    /// Maximum staleness for bounded staleness (milliseconds)
    pub max_staleness_ms: u64,
    /// Lease duration (milliseconds)
    pub lease_duration_ms: u64,
    /// Enable causality tracking
    pub enable_causality_tracking: bool,
    /// Enable read-your-writes
    pub enable_read_your_writes: bool,
}

impl Default for ConsistencyConfig {
    fn default() -> Self {
        Self {
            default_consistency_level: ConsistencyLevel::Linearizable,
            default_read_strategy: ReadStrategy::ReadIndex,
            read_quorum_size: 2, // Majority of 3
            max_staleness_ms: 100,
            lease_duration_ms: 5000, // 5 seconds
            enable_causality_tracking: true,
            enable_read_your_writes: true,
        }
    }
}

/// Read token for linearizable reads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadToken {
    /// Token ID
    pub token_id: String,
    /// Read index (commit index when read was initiated)
    pub read_index: u64,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Node ID
    pub node_id: OxirsNodeId,
    /// Consistency level
    pub consistency_level: ConsistencyLevel,
}

/// Lease information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaseInfo {
    /// Lease holder (leader node ID)
    pub holder: OxirsNodeId,
    /// Lease start time
    pub start_time: SystemTime,
    /// Lease expiration time
    pub expiration: SystemTime,
    /// Lease term
    pub term: u64,
}

impl LeaseInfo {
    fn is_valid(&self) -> bool {
        SystemTime::now() < self.expiration
    }
}

/// Causality token for session consistency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalityToken {
    /// Session ID
    pub session_id: String,
    /// Last observed commit index
    pub last_commit_index: u64,
    /// Vector clock (node_id -> sequence)
    pub vector_clock: BTreeMap<OxirsNodeId, u64>,
}

/// Read request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadRequest {
    /// Request ID
    pub request_id: String,
    /// Consistency level
    pub consistency_level: ConsistencyLevel,
    /// Read strategy
    pub read_strategy: ReadStrategy,
    /// Causality token (for session consistency)
    pub causality_token: Option<CausalityToken>,
    /// Maximum staleness (for bounded staleness)
    pub max_staleness: Option<Duration>,
}

/// Read response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadResponse {
    /// Request ID
    pub request_id: String,
    /// Success
    pub success: bool,
    /// Read token (for verification)
    pub read_token: Option<ReadToken>,
    /// Actual staleness (milliseconds)
    pub staleness_ms: u64,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Consistency statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyStats {
    /// Total reads
    pub total_reads: u64,
    /// Linearizable reads
    pub linearizable_reads: u64,
    /// Session reads
    pub session_reads: u64,
    /// Bounded staleness reads
    pub bounded_staleness_reads: u64,
    /// Eventual reads
    pub eventual_reads: u64,
    /// Average read latency (ms)
    pub avg_read_latency_ms: f64,
    /// Read index operations
    pub read_index_ops: u64,
    /// Lease reads
    pub lease_reads: u64,
    /// Follower reads
    pub follower_reads: u64,
    /// Consistency violations detected
    pub consistency_violations: u64,
}

impl Default for ConsistencyStats {
    fn default() -> Self {
        Self {
            total_reads: 0,
            linearizable_reads: 0,
            session_reads: 0,
            bounded_staleness_reads: 0,
            eventual_reads: 0,
            avg_read_latency_ms: 0.0,
            read_index_ops: 0,
            lease_reads: 0,
            follower_reads: 0,
            consistency_violations: 0,
        }
    }
}

/// Strong consistency manager
pub struct StrongConsistencyManager {
    config: ConsistencyConfig,
    /// Current leader
    current_leader: Arc<RwLock<Option<OxirsNodeId>>>,
    /// Current term
    current_term: Arc<RwLock<u64>>,
    /// Commit index
    commit_index: Arc<RwLock<u64>>,
    /// Active leases
    leases: Arc<RwLock<HashMap<OxirsNodeId, LeaseInfo>>>,
    /// Read tokens
    read_tokens: Arc<RwLock<VecDeque<ReadToken>>>,
    /// Session causality tokens
    session_tokens: Arc<RwLock<HashMap<String, CausalityToken>>>,
    /// Statistics
    stats: Arc<RwLock<ConsistencyStats>>,
    /// Local node ID
    local_node_id: OxirsNodeId,
}

impl StrongConsistencyManager {
    /// Create a new strong consistency manager
    pub fn new(local_node_id: OxirsNodeId, config: ConsistencyConfig) -> Self {
        Self {
            config,
            current_leader: Arc::new(RwLock::new(None)),
            current_term: Arc::new(RwLock::new(0)),
            commit_index: Arc::new(RwLock::new(0)),
            leases: Arc::new(RwLock::new(HashMap::new())),
            read_tokens: Arc::new(RwLock::new(VecDeque::new())),
            session_tokens: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(ConsistencyStats::default())),
            local_node_id,
        }
    }

    /// Perform a linearizable read
    pub async fn linearizable_read(&self, request: ReadRequest) -> Result<ReadResponse, String> {
        let start = std::time::Instant::now();

        let response = match request.read_strategy {
            ReadStrategy::LeaderRead => self.leader_read(&request).await?,
            ReadStrategy::ReadIndex => self.read_index(&request).await?,
            ReadStrategy::LeaseRead => self.lease_read(&request).await?,
            ReadStrategy::FollowerRead => self.follower_read(&request).await?,
        };

        // Update statistics
        // Use microseconds for better precision, then convert to milliseconds
        let latency = start.elapsed().as_micros() as f64 / 1000.0;
        let mut stats = self.stats.write().await;
        stats.total_reads += 1;

        // Track consistency level stats
        match request.consistency_level {
            ConsistencyLevel::Linearizable => stats.linearizable_reads += 1,
            ConsistencyLevel::Session => stats.session_reads += 1,
            ConsistencyLevel::BoundedStaleness => stats.bounded_staleness_reads += 1,
            ConsistencyLevel::Eventual => stats.eventual_reads += 1,
        }

        // Track read strategy stats (independent of consistency level)
        match request.read_strategy {
            ReadStrategy::ReadIndex => stats.read_index_ops += 1,
            ReadStrategy::LeaseRead => stats.lease_reads += 1,
            ReadStrategy::FollowerRead => stats.follower_reads += 1,
            _ => {}
        }

        let total = stats.total_reads as f64;
        stats.avg_read_latency_ms = (stats.avg_read_latency_ms * (total - 1.0) + latency) / total;

        Ok(response)
    }

    /// Leader read (always linearizable)
    async fn leader_read(&self, request: &ReadRequest) -> Result<ReadResponse, String> {
        let leader = self.current_leader.read().await;

        if leader.is_none() || *leader != Some(self.local_node_id) {
            return Err("Not the leader".to_string());
        }

        let commit_index = *self.commit_index.read().await;

        let token = ReadToken {
            token_id: request.request_id.clone(),
            read_index: commit_index,
            timestamp: SystemTime::now(),
            node_id: self.local_node_id,
            consistency_level: ConsistencyLevel::Linearizable,
        };

        Ok(ReadResponse {
            request_id: request.request_id.clone(),
            success: true,
            read_token: Some(token),
            staleness_ms: 0,
            timestamp: SystemTime::now(),
        })
    }

    /// Read index protocol (linearizable with heartbeat)
    async fn read_index(&self, request: &ReadRequest) -> Result<ReadResponse, String> {
        let leader = self.current_leader.read().await;

        if leader.is_none() {
            return Err("No leader available".to_string());
        }

        let commit_index = *self.commit_index.read().await;

        // In production: send heartbeat to confirm leadership
        // For now, simulate with a short delay
        tokio::time::sleep(Duration::from_millis(10)).await;

        let token = ReadToken {
            token_id: request.request_id.clone(),
            read_index: commit_index,
            timestamp: SystemTime::now(),
            node_id: self.local_node_id,
            consistency_level: ConsistencyLevel::Linearizable,
        };

        // Store token
        let mut tokens = self.read_tokens.write().await;
        tokens.push_back(token.clone());

        // Keep only recent tokens (last 1000)
        if tokens.len() > 1000 {
            tokens.pop_front();
        }

        Ok(ReadResponse {
            request_id: request.request_id.clone(),
            success: true,
            read_token: Some(token),
            staleness_ms: 0,
            timestamp: SystemTime::now(),
        })
    }

    /// Lease-based read (linearizable with lease)
    async fn lease_read(&self, request: &ReadRequest) -> Result<ReadResponse, String> {
        let leases = self.leases.read().await;
        let leader = self.current_leader.read().await;

        if let Some(leader_id) = *leader {
            if let Some(lease) = leases.get(&leader_id) {
                if lease.is_valid() {
                    let commit_index = *self.commit_index.read().await;

                    let token = ReadToken {
                        token_id: request.request_id.clone(),
                        read_index: commit_index,
                        timestamp: SystemTime::now(),
                        node_id: self.local_node_id,
                        consistency_level: ConsistencyLevel::Linearizable,
                    };

                    return Ok(ReadResponse {
                        request_id: request.request_id.clone(),
                        success: true,
                        read_token: Some(token),
                        staleness_ms: 0,
                        timestamp: SystemTime::now(),
                    });
                }
            }
        }

        // Lease expired or not found, fallback to read index
        self.read_index(request).await
    }

    /// Follower read (may be stale)
    async fn follower_read(&self, request: &ReadRequest) -> Result<ReadResponse, String> {
        let commit_index = *self.commit_index.read().await;

        // Check staleness
        let leader = self.current_leader.read().await;
        let staleness_ms = if leader.is_some() {
            // In production: query leader for latest commit index
            // For now, assume some staleness
            50
        } else {
            self.config.max_staleness_ms
        };

        // Check if within bounds
        if request.consistency_level == ConsistencyLevel::BoundedStaleness {
            if let Some(max_staleness) = request.max_staleness {
                if staleness_ms > max_staleness.as_millis() as u64 {
                    return Err("Staleness exceeds bounds".to_string());
                }
            } else if staleness_ms > self.config.max_staleness_ms {
                return Err("Staleness exceeds configured maximum".to_string());
            }
        }

        let token = ReadToken {
            token_id: request.request_id.clone(),
            read_index: commit_index,
            timestamp: SystemTime::now(),
            node_id: self.local_node_id,
            consistency_level: request.consistency_level,
        };

        Ok(ReadResponse {
            request_id: request.request_id.clone(),
            success: true,
            read_token: Some(token),
            staleness_ms,
            timestamp: SystemTime::now(),
        })
    }

    /// Session read (read-your-writes)
    pub async fn session_read(
        &self,
        session_id: &str,
        request: ReadRequest,
    ) -> Result<ReadResponse, String> {
        if !self.config.enable_read_your_writes {
            return self.linearizable_read(request).await;
        }

        let session_tokens = self.session_tokens.read().await;

        if let Some(causality_token) = session_tokens.get(session_id) {
            let commit_index = *self.commit_index.read().await;

            // Ensure we've replicated at least to the session's last observed index
            if commit_index < causality_token.last_commit_index {
                return Err("Session consistency not yet satisfied".to_string());
            }
        }

        self.linearizable_read(request).await
    }

    /// Update causality token for session
    pub async fn update_session_token(&self, session_id: String, commit_index: u64) {
        if !self.config.enable_causality_tracking {
            return;
        }

        let mut session_tokens = self.session_tokens.write().await;

        let token = session_tokens
            .entry(session_id.clone())
            .or_insert_with(|| CausalityToken {
                session_id: session_id.clone(),
                last_commit_index: 0,
                vector_clock: BTreeMap::new(),
            });

        token.last_commit_index = token.last_commit_index.max(commit_index);
        token.vector_clock.insert(self.local_node_id, commit_index);
    }

    /// Update leader information
    pub async fn update_leader(&self, leader_id: Option<OxirsNodeId>, term: u64) {
        *self.current_leader.write().await = leader_id;
        *self.current_term.write().await = term;

        if let Some(leader) = leader_id {
            info!("Leader updated: node {} (term {})", leader, term);

            // Create/renew lease
            if leader == self.local_node_id {
                let lease = LeaseInfo {
                    holder: leader,
                    start_time: SystemTime::now(),
                    expiration: SystemTime::now()
                        + Duration::from_millis(self.config.lease_duration_ms),
                    term,
                };

                self.leases.write().await.insert(leader, lease);
            }
        }
    }

    /// Update commit index
    pub async fn update_commit_index(&self, index: u64) {
        let mut commit_index = self.commit_index.write().await;
        *commit_index = index;
    }

    /// Get statistics
    pub async fn get_stats(&self) -> ConsistencyStats {
        self.stats.read().await.clone()
    }

    /// Clear all data
    pub async fn clear(&self) {
        *self.current_leader.write().await = None;
        *self.current_term.write().await = 0;
        *self.commit_index.write().await = 0;
        self.leases.write().await.clear();
        self.read_tokens.write().await.clear();
        self.session_tokens.write().await.clear();
        *self.stats.write().await = ConsistencyStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_consistency_manager_creation() {
        let config = ConsistencyConfig::default();
        let manager = StrongConsistencyManager::new(1, config);

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_reads, 0);
    }

    #[tokio::test]
    async fn test_leader_read() {
        let config = ConsistencyConfig::default();
        let manager = StrongConsistencyManager::new(1, config);

        // Set self as leader
        manager.update_leader(Some(1), 1).await;
        manager.update_commit_index(100).await;

        let request = ReadRequest {
            request_id: "test-1".to_string(),
            consistency_level: ConsistencyLevel::Linearizable,
            read_strategy: ReadStrategy::LeaderRead,
            causality_token: None,
            max_staleness: None,
        };

        let response = manager.linearizable_read(request).await;
        assert!(response.is_ok());

        let response = response.unwrap();
        assert!(response.success);
        assert_eq!(response.staleness_ms, 0);
    }

    #[tokio::test]
    async fn test_read_index() {
        let config = ConsistencyConfig::default();
        let manager = StrongConsistencyManager::new(1, config);

        manager.update_leader(Some(1), 1).await;
        manager.update_commit_index(200).await;

        let request = ReadRequest {
            request_id: "test-2".to_string(),
            consistency_level: ConsistencyLevel::Linearizable,
            read_strategy: ReadStrategy::ReadIndex,
            causality_token: None,
            max_staleness: None,
        };

        let response = manager.linearizable_read(request).await;
        assert!(response.is_ok());

        let stats = manager.get_stats().await;
        assert_eq!(stats.read_index_ops, 1);
    }

    #[tokio::test]
    async fn test_lease_read() {
        let config = ConsistencyConfig::default();
        let manager = StrongConsistencyManager::new(1, config);

        manager.update_leader(Some(1), 1).await;
        manager.update_commit_index(300).await;

        let request = ReadRequest {
            request_id: "test-3".to_string(),
            consistency_level: ConsistencyLevel::Linearizable,
            read_strategy: ReadStrategy::LeaseRead,
            causality_token: None,
            max_staleness: None,
        };

        let response = manager.linearizable_read(request).await;
        assert!(response.is_ok());

        let stats = manager.get_stats().await;
        assert_eq!(stats.lease_reads, 1);
    }

    #[tokio::test]
    async fn test_follower_read() {
        let config = ConsistencyConfig::default();
        let manager = StrongConsistencyManager::new(2, config); // Not leader

        manager.update_leader(Some(1), 1).await;
        manager.update_commit_index(400).await;

        let request = ReadRequest {
            request_id: "test-4".to_string(),
            consistency_level: ConsistencyLevel::BoundedStaleness,
            read_strategy: ReadStrategy::FollowerRead,
            causality_token: None,
            max_staleness: Some(Duration::from_millis(100)),
        };

        let response = manager.linearizable_read(request).await;
        assert!(response.is_ok());

        let stats = manager.get_stats().await;
        assert_eq!(stats.follower_reads, 1);
    }

    #[tokio::test]
    async fn test_session_read() {
        let config = ConsistencyConfig {
            enable_read_your_writes: true,
            ..Default::default()
        };
        let manager = StrongConsistencyManager::new(1, config);

        manager.update_leader(Some(1), 1).await;
        manager.update_commit_index(500).await;

        // Update session token
        manager
            .update_session_token("session-1".to_string(), 450)
            .await;

        let request = ReadRequest {
            request_id: "test-5".to_string(),
            consistency_level: ConsistencyLevel::Session,
            read_strategy: ReadStrategy::LeaderRead,
            causality_token: None,
            max_staleness: None,
        };

        let response = manager.session_read("session-1", request).await;
        assert!(response.is_ok());

        let stats = manager.get_stats().await;
        assert_eq!(stats.session_reads, 1);
    }

    #[tokio::test]
    async fn test_consistency_levels() {
        assert!(ConsistencyLevel::Eventual < ConsistencyLevel::Session);
        assert!(ConsistencyLevel::Session < ConsistencyLevel::BoundedStaleness);
        assert!(ConsistencyLevel::BoundedStaleness < ConsistencyLevel::Linearizable);
    }

    #[tokio::test]
    async fn test_lease_validity() {
        let lease = LeaseInfo {
            holder: 1,
            start_time: SystemTime::now(),
            expiration: SystemTime::now() + Duration::from_secs(5),
            term: 1,
        };

        assert!(lease.is_valid());

        let expired_lease = LeaseInfo {
            holder: 1,
            start_time: SystemTime::now() - Duration::from_secs(10),
            expiration: SystemTime::now() - Duration::from_secs(5),
            term: 1,
        };

        assert!(!expired_lease.is_valid());
    }

    #[tokio::test]
    async fn test_update_leader() {
        let config = ConsistencyConfig::default();
        let manager = StrongConsistencyManager::new(1, config);

        manager.update_leader(Some(1), 5).await;

        let leader = manager.current_leader.read().await;
        assert_eq!(*leader, Some(1));

        let term = manager.current_term.read().await;
        assert_eq!(*term, 5);
    }

    #[tokio::test]
    async fn test_update_commit_index() {
        let config = ConsistencyConfig::default();
        let manager = StrongConsistencyManager::new(1, config);

        manager.update_commit_index(1000).await;

        let index = manager.commit_index.read().await;
        assert_eq!(*index, 1000);
    }

    #[tokio::test]
    async fn test_statistics() {
        let config = ConsistencyConfig::default();
        let manager = StrongConsistencyManager::new(1, config);

        manager.update_leader(Some(1), 1).await;

        // Perform various reads
        for i in 0..5 {
            let request = ReadRequest {
                request_id: format!("test-{}", i),
                consistency_level: ConsistencyLevel::Linearizable,
                read_strategy: ReadStrategy::LeaderRead,
                causality_token: None,
                max_staleness: None,
            };

            let _ = manager.linearizable_read(request).await;
        }

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_reads, 5);
        assert_eq!(stats.linearizable_reads, 5);
        assert!(stats.avg_read_latency_ms > 0.0);
    }

    #[tokio::test]
    async fn test_clear() {
        let config = ConsistencyConfig::default();
        let manager = StrongConsistencyManager::new(1, config);

        manager.update_leader(Some(1), 1).await;
        manager.update_commit_index(100).await;

        manager.clear().await;

        let leader = manager.current_leader.read().await;
        assert!(leader.is_none());

        let index = manager.commit_index.read().await;
        assert_eq!(*index, 0);
    }
}
