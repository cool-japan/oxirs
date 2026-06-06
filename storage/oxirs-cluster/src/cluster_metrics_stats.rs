//! Operation taxonomy and enhanced latency statistics for cluster metrics.
//!
//! This module defines:
//! - [`ClusterOperation`]: the enumeration of tracked cluster operation types
//! - [`EnhancedLatencyStats`]: a rolling-window latency accumulator with rich
//!   distribution analysis (percentiles, variance, skewness, kurtosis, trend, EMA)

use serde::{Deserialize, Serialize};

/// Cluster operation types for metrics tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClusterOperation {
    /// Raft append entries operation
    AppendEntries,
    /// Raft request vote operation
    RequestVote,
    /// Snapshot creation
    SnapshotCreate,
    /// Snapshot restoration
    SnapshotRestore,
    /// Log compaction
    LogCompaction,
    /// Network round-trip
    NetworkRoundTrip,
    /// Query execution
    QueryExecution,
    /// Batch processing
    BatchProcessing,
    /// Data replication
    DataReplication,
    /// Node discovery
    NodeDiscovery,
    /// Leadership election
    LeaderElection,
    /// Transaction commit
    TransactionCommit,
    /// Transaction rollback
    TransactionRollback,
    /// Shard migration
    ShardMigration,
    /// Merkle tree verification
    MerkleVerification,
    /// Conflict resolution
    ConflictResolution,
    /// Backup creation
    BackupCreate,
    /// Restore operation
    RestoreOperation,
    /// Auto-scaling decision
    AutoScaling,
    /// Read replica sync
    ReadReplicaSync,
    /// Circuit breaker state change
    CircuitBreakerChange,
    /// Region failover
    RegionFailover,
}

impl ClusterOperation {
    /// Get operation name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::AppendEntries => "append_entries",
            Self::RequestVote => "request_vote",
            Self::SnapshotCreate => "snapshot_create",
            Self::SnapshotRestore => "snapshot_restore",
            Self::LogCompaction => "log_compaction",
            Self::NetworkRoundTrip => "network_roundtrip",
            Self::QueryExecution => "query_execution",
            Self::BatchProcessing => "batch_processing",
            Self::DataReplication => "data_replication",
            Self::NodeDiscovery => "node_discovery",
            Self::LeaderElection => "leader_election",
            Self::TransactionCommit => "transaction_commit",
            Self::TransactionRollback => "transaction_rollback",
            Self::ShardMigration => "shard_migration",
            Self::MerkleVerification => "merkle_verification",
            Self::ConflictResolution => "conflict_resolution",
            Self::BackupCreate => "backup_create",
            Self::RestoreOperation => "restore_operation",
            Self::AutoScaling => "auto_scaling",
            Self::ReadReplicaSync => "read_replica_sync",
            Self::CircuitBreakerChange => "circuit_breaker_change",
            Self::RegionFailover => "region_failover",
        }
    }

    /// Get all operation types
    pub fn all() -> Vec<Self> {
        vec![
            Self::AppendEntries,
            Self::RequestVote,
            Self::SnapshotCreate,
            Self::SnapshotRestore,
            Self::LogCompaction,
            Self::NetworkRoundTrip,
            Self::QueryExecution,
            Self::BatchProcessing,
            Self::DataReplication,
            Self::NodeDiscovery,
            Self::LeaderElection,
            Self::TransactionCommit,
            Self::TransactionRollback,
            Self::ShardMigration,
            Self::MerkleVerification,
            Self::ConflictResolution,
            Self::BackupCreate,
            Self::RestoreOperation,
            Self::AutoScaling,
            Self::ReadReplicaSync,
            Self::CircuitBreakerChange,
            Self::RegionFailover,
        ]
    }
}

/// Enhanced latency statistics with detailed distribution analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnhancedLatencyStats {
    /// Raw latency values in microseconds
    pub(crate) values: Vec<f64>,
    /// Total count
    pub(crate) count: u64,
    /// Sum of values
    pub(crate) sum: f64,
    /// Sum of squared values (for variance calculation)
    pub(crate) sum_squared: f64,
    /// Minimum value
    pub(crate) min: f64,
    /// Maximum value
    pub(crate) max: f64,
    /// Exponentially decayed mean (alpha = 0.1)
    pub(crate) ema: f64,
    /// Rolling window size
    pub(crate) window_size: usize,
}

impl EnhancedLatencyStats {
    /// Create new enhanced latency stats
    pub fn new(window_size: usize) -> Self {
        Self {
            values: Vec::with_capacity(window_size),
            count: 0,
            sum: 0.0,
            sum_squared: 0.0,
            min: f64::MAX,
            max: 0.0,
            ema: 0.0,
            window_size,
        }
    }

    /// Record a new latency value
    pub fn record(&mut self, micros: f64) {
        // Update rolling window
        if self.values.len() >= self.window_size {
            // Remove oldest value from sum
            let oldest = self.values.remove(0);
            self.sum -= oldest;
            self.sum_squared -= oldest * oldest;
        }
        self.values.push(micros);

        // Update statistics
        self.count += 1;
        self.sum += micros;
        self.sum_squared += micros * micros;
        self.min = self.min.min(micros);
        self.max = self.max.max(micros);

        // Update exponential moving average
        let alpha = 0.1;
        if self.count == 1 {
            self.ema = micros;
        } else {
            self.ema = alpha * micros + (1.0 - alpha) * self.ema;
        }
    }

    /// Get mean latency
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.sum / self.values.len() as f64
        }
    }

    /// Get variance
    pub fn variance(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }
        let n = self.values.len() as f64;
        let mean = self.mean();
        (self.sum_squared / n) - (mean * mean)
    }

    /// Get standard deviation
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get coefficient of variation (CV)
    pub fn coefficient_of_variation(&self) -> f64 {
        let mean = self.mean();
        if mean == 0.0 {
            0.0
        } else {
            self.std_dev() / mean
        }
    }

    /// Get percentile value
    pub fn percentile(&self, p: f64) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }

        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    /// Get interquartile range (IQR)
    pub fn iqr(&self) -> f64 {
        self.percentile(75.0) - self.percentile(25.0)
    }

    /// Get skewness (measure of asymmetry)
    pub fn skewness(&self) -> f64 {
        if self.values.len() < 3 || self.std_dev() == 0.0 {
            return 0.0;
        }

        let mean = self.mean();
        let std = self.std_dev();
        let n = self.values.len() as f64;

        let sum_cubed: f64 = self
            .values
            .iter()
            .map(|&x| ((x - mean) / std).powi(3))
            .sum();

        sum_cubed / n
    }

    /// Get kurtosis (measure of tailedness)
    pub fn kurtosis(&self) -> f64 {
        if self.values.len() < 4 || self.std_dev() == 0.0 {
            return 0.0;
        }

        let mean = self.mean();
        let std = self.std_dev();
        let n = self.values.len() as f64;

        let sum_fourth: f64 = self
            .values
            .iter()
            .map(|&x| ((x - mean) / std).powi(4))
            .sum();

        (sum_fourth / n) - 3.0 // Excess kurtosis
    }

    /// Get trend (slope of recent values using linear regression)
    pub fn trend(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }

        let n = self.values.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;

        for (i, &y) in self.values.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator == 0.0 {
            0.0
        } else {
            (n * sum_xy - sum_x * sum_y) / denominator
        }
    }

    /// Get exponentially weighted moving average
    pub fn ema(&self) -> f64 {
        self.ema
    }

    /// Get rate (operations per second based on recent window)
    pub fn rate(&self) -> f64 {
        if self.mean() > 0.0 {
            1_000_000.0 / self.mean()
        } else {
            0.0
        }
    }
}
