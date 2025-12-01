//! Core types for hot/warm/cold tiering system

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

/// Storage tier classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StorageTier {
    /// In-memory storage (fastest, most expensive)
    Hot,
    /// Memory-mapped or SSD storage (moderate speed and cost)
    Warm,
    /// Compressed disk storage (slowest, cheapest)
    Cold,
}

impl StorageTier {
    /// Get the typical latency for this tier
    pub fn typical_latency(&self) -> Duration {
        match self {
            StorageTier::Hot => Duration::from_micros(100), // Sub-millisecond
            StorageTier::Warm => Duration::from_millis(5),  // Few milliseconds
            StorageTier::Cold => Duration::from_millis(100), // ~100ms for decompression + load
        }
    }

    /// Get the relative cost factor for this tier
    pub fn cost_factor(&self) -> f64 {
        match self {
            StorageTier::Hot => 10.0, // Most expensive (RAM)
            StorageTier::Warm => 2.0, // Moderate (SSD)
            StorageTier::Cold => 1.0, // Cheapest (HDD/compressed)
        }
    }

    /// Get the compression ratio for this tier
    pub fn compression_ratio(&self) -> f64 {
        match self {
            StorageTier::Hot => 1.0,  // No compression
            StorageTier::Warm => 1.2, // Light compression
            StorageTier::Cold => 4.0, // Heavy compression
        }
    }
}

/// Metadata about an indexed dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    /// Unique identifier for this index
    pub index_id: String,
    /// Current storage tier
    pub current_tier: StorageTier,
    /// Size in bytes (uncompressed)
    pub size_bytes: u64,
    /// Size in bytes (compressed, if applicable)
    pub compressed_size_bytes: u64,
    /// Number of vectors in this index
    pub vector_count: usize,
    /// Vector dimensionality
    pub dimension: usize,
    /// Index type (HNSW, IVF, etc.)
    pub index_type: IndexType,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last access timestamp
    pub last_accessed: SystemTime,
    /// Last modification timestamp
    pub last_modified: SystemTime,
    /// Access statistics
    pub access_stats: AccessStatistics,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Storage location
    pub storage_path: Option<PathBuf>,
    /// Custom metadata
    pub custom_metadata: HashMap<String, String>,
}

/// Index type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexType {
    Hnsw,
    Ivf,
    Pq,
    Sq,
    Lsh,
    Nsg,
    Flat,
}

/// Access statistics for an index
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AccessStatistics {
    /// Total number of queries to this index
    pub total_queries: u64,
    /// Number of queries in the last hour
    pub queries_last_hour: u64,
    /// Number of queries in the last day
    pub queries_last_day: u64,
    /// Number of queries in the last week
    pub queries_last_week: u64,
    /// Average queries per second (moving average)
    pub avg_qps: f64,
    /// Peak queries per second
    pub peak_qps: f64,
    /// Last access timestamp
    pub last_access_time: Option<SystemTime>,
    /// Access pattern (hot, warm, cold)
    pub access_pattern: AccessPattern,
    /// Query latencies (p50, p95, p99 in microseconds)
    pub query_latencies: LatencyPercentiles,
}

/// Access pattern classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessPattern {
    /// Frequently accessed, high QPS
    Hot,
    /// Moderately accessed
    Warm,
    /// Rarely accessed
    Cold,
    /// Bursty access pattern
    Bursty,
    /// Seasonal pattern
    Seasonal,
    /// Unknown pattern (insufficient data)
    Unknown,
}

impl Default for AccessPattern {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Latency percentiles
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    pub p50: u64,
    pub p95: u64,
    pub p99: u64,
    pub max: u64,
}

/// Performance metrics for an index
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average query latency in microseconds
    pub avg_query_latency_us: u64,
    /// Average load time in milliseconds
    pub avg_load_time_ms: u64,
    /// Cache hit rate (0.0 - 1.0)
    pub cache_hit_rate: f64,
    /// Memory footprint in bytes
    pub memory_footprint_bytes: u64,
    /// CPU utilization (0.0 - 1.0)
    pub cpu_utilization: f64,
    /// IO operations per second
    pub iops: f64,
    /// Throughput in MB/s
    pub throughput_mbps: f64,
}

/// Statistics for a storage tier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierStatistics {
    /// Total capacity in bytes
    pub capacity_bytes: u64,
    /// Used capacity in bytes
    pub used_bytes: u64,
    /// Number of indices in this tier
    pub index_count: usize,
    /// Total number of queries to this tier
    pub total_queries: u64,
    /// Average query latency in microseconds
    pub avg_query_latency_us: u64,
    /// Hit rate for this tier
    pub hit_rate: f64,
    /// Number of promotions from this tier
    pub promotions: u64,
    /// Number of demotions to this tier
    pub demotions: u64,
    /// Total bytes read
    pub bytes_read: u64,
    /// Total bytes written
    pub bytes_written: u64,
    /// Last update timestamp
    pub last_updated: SystemTime,
}

impl Default for TierStatistics {
    fn default() -> Self {
        Self {
            capacity_bytes: 0,
            used_bytes: 0,
            index_count: 0,
            total_queries: 0,
            avg_query_latency_us: 0,
            hit_rate: 0.0,
            promotions: 0,
            demotions: 0,
            bytes_read: 0,
            bytes_written: 0,
            last_updated: SystemTime::now(),
        }
    }
}

impl TierStatistics {
    /// Calculate utilization ratio (0.0 - 1.0)
    pub fn utilization(&self) -> f64 {
        if self.capacity_bytes == 0 {
            0.0
        } else {
            self.used_bytes as f64 / self.capacity_bytes as f64
        }
    }

    /// Calculate available bytes
    pub fn available_bytes(&self) -> u64 {
        self.capacity_bytes.saturating_sub(self.used_bytes)
    }

    /// Check if tier is near capacity
    pub fn is_near_capacity(&self, threshold: f64) -> bool {
        self.utilization() >= threshold
    }
}

/// Tier transition event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierTransition {
    /// Index identifier
    pub index_id: String,
    /// Source tier
    pub from_tier: StorageTier,
    /// Destination tier
    pub to_tier: StorageTier,
    /// Reason for transition
    pub reason: String,
    /// Timestamp of transition
    pub timestamp: SystemTime,
    /// Transition duration
    pub duration: Duration,
    /// Success flag
    pub success: bool,
    /// Error message (if failed)
    pub error: Option<String>,
}

/// Configuration for gradual tier transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradualTransitionConfig {
    /// Enable gradual transitions
    pub enabled: bool,
    /// Number of stages for gradual transition
    pub stages: usize,
    /// Delay between stages
    pub stage_delay: Duration,
    /// Monitor performance during transition
    pub monitor_performance: bool,
    /// Rollback on performance degradation
    pub rollback_on_degradation: bool,
    /// Performance degradation threshold (0.0 - 1.0)
    pub degradation_threshold: f64,
}

impl Default for GradualTransitionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            stages: 3,
            stage_delay: Duration::from_secs(60),
            monitor_performance: true,
            rollback_on_degradation: true,
            degradation_threshold: 0.2, // 20% degradation triggers rollback
        }
    }
}

/// Cost model for tier placement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierCostModel {
    /// Cost per GB per hour for hot tier
    pub hot_cost_per_gb_hour: f64,
    /// Cost per GB per hour for warm tier
    pub warm_cost_per_gb_hour: f64,
    /// Cost per GB per hour for cold tier
    pub cold_cost_per_gb_hour: f64,
    /// Cost per query for hot tier
    pub hot_query_cost: f64,
    /// Cost per query for warm tier
    pub warm_query_cost: f64,
    /// Cost per query for cold tier
    pub cold_query_cost: f64,
    /// Cost per tier transition
    pub transition_cost: f64,
}

impl Default for TierCostModel {
    fn default() -> Self {
        Self {
            hot_cost_per_gb_hour: 0.10,   // High cost for RAM
            warm_cost_per_gb_hour: 0.02,  // Moderate cost for SSD
            cold_cost_per_gb_hour: 0.005, // Low cost for HDD
            hot_query_cost: 0.0001,       // Low per-query cost (fast)
            warm_query_cost: 0.0005,      // Higher per-query cost (slower)
            cold_query_cost: 0.002,       // Highest per-query cost (slowest)
            transition_cost: 0.01,        // Cost of moving data between tiers
        }
    }
}

/// Resource allocation for multi-tenancy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantResourceAllocation {
    /// Tenant identifier
    pub tenant_id: String,
    /// Allocated capacity in hot tier (bytes)
    pub hot_tier_allocation_bytes: u64,
    /// Allocated capacity in warm tier (bytes)
    pub warm_tier_allocation_bytes: u64,
    /// Allocated capacity in cold tier (bytes)
    pub cold_tier_allocation_bytes: u64,
    /// QPS quota
    pub qps_quota: f64,
    /// Priority (higher values = higher priority)
    pub priority: u32,
    /// Custom policies
    pub custom_policies: HashMap<String, String>,
}
