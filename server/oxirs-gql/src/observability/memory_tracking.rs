//! Memory Allocation Tracking
//!
//! Tracks memory allocations per GraphQL query to identify memory leaks,
//! optimize memory usage, and provide detailed memory profiling.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use sysinfo::System;
use tokio::sync::RwLock;

/// Memory allocation tracker
pub struct MemoryTracker {
    /// Query memory snapshots
    snapshots: Arc<RwLock<HashMap<String, MemorySnapshot>>>,
    /// Completed allocations
    completed: Arc<RwLock<Vec<MemoryAllocation>>>,
    /// System info for memory stats
    system: Arc<RwLock<System>>,
    /// Configuration
    config: MemoryTrackingConfig,
}

/// Memory tracking configuration
#[derive(Debug, Clone)]
pub struct MemoryTrackingConfig {
    /// Enable detailed tracking
    pub enable_detailed_tracking: bool,
    /// Track allocation backtraces
    pub track_backtraces: bool,
    /// Maximum allocations to store
    pub max_allocations: usize,
    /// Retention period for completed allocations
    pub retention_period: Duration,
    /// Memory threshold for warnings (bytes)
    pub warning_threshold_bytes: u64,
}

impl Default for MemoryTrackingConfig {
    fn default() -> Self {
        Self {
            enable_detailed_tracking: true,
            track_backtraces: false,
            max_allocations: 1000,
            retention_period: Duration::from_secs(3600), // 1 hour
            warning_threshold_bytes: 100 * 1024 * 1024,  // 100 MB
        }
    }
}

/// Memory snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    /// Query ID
    pub query_id: String,
    /// Operation name
    pub operation_name: Option<String>,
    /// Start time (not serialized)
    #[serde(skip, default = "Instant::now")]
    pub start_time: Instant,
    /// Start timestamp (Unix epoch)
    pub start_timestamp: u64,
    /// Initial memory usage (bytes)
    pub initial_memory: u64,
    /// Peak memory usage (bytes)
    pub peak_memory: u64,
    /// Current memory usage (bytes)
    pub current_memory: u64,
    /// Number of allocations
    pub allocation_count: u64,
    /// Number of deallocations
    pub deallocation_count: u64,
    /// Total bytes allocated
    pub total_allocated: u64,
    /// Total bytes deallocated
    pub total_deallocated: u64,
}

impl MemorySnapshot {
    /// Create a new memory snapshot
    pub fn new(query_id: String, operation_name: Option<String>, initial_memory: u64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs();

        Self {
            query_id,
            operation_name,
            start_time: Instant::now(),
            start_timestamp: now,
            initial_memory,
            peak_memory: initial_memory,
            current_memory: initial_memory,
            allocation_count: 0,
            deallocation_count: 0,
            total_allocated: 0,
            total_deallocated: 0,
        }
    }

    /// Record an allocation
    pub fn record_allocation(&mut self, bytes: u64) {
        self.allocation_count += 1;
        self.total_allocated += bytes;
        self.current_memory += bytes;
        self.peak_memory = self.peak_memory.max(self.current_memory);
    }

    /// Record a deallocation
    pub fn record_deallocation(&mut self, bytes: u64) {
        self.deallocation_count += 1;
        self.total_deallocated += bytes;
        self.current_memory = self.current_memory.saturating_sub(bytes);
    }

    /// Get net memory change
    pub fn net_memory_change(&self) -> i64 {
        self.total_allocated as i64 - self.total_deallocated as i64
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Completed memory allocation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAllocation {
    /// Query ID
    pub query_id: String,
    /// Operation name
    pub operation_name: Option<String>,
    /// Start time (Unix epoch)
    pub start_time: u64,
    /// End time (Unix epoch)
    pub end_time: u64,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Initial memory (bytes)
    pub initial_memory: u64,
    /// Peak memory (bytes)
    pub peak_memory: u64,
    /// Final memory (bytes)
    pub final_memory: u64,
    /// Total allocated (bytes)
    pub total_allocated: u64,
    /// Total deallocated (bytes)
    pub total_deallocated: u64,
    /// Net memory change (bytes)
    pub net_change: i64,
    /// Allocation count
    pub allocation_count: u64,
    /// Deallocation count
    pub deallocation_count: u64,
    /// Memory leaked (bytes)
    pub leaked_bytes: u64,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl MemoryAllocation {
    /// Create from snapshot
    pub fn from_snapshot(snapshot: MemorySnapshot) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs();

        let net_change = snapshot.net_memory_change();
        let leaked_bytes = if net_change > 0 { net_change as u64 } else { 0 };

        let duration_ms = snapshot.elapsed().as_millis() as u64;

        Self {
            query_id: snapshot.query_id,
            operation_name: snapshot.operation_name.clone(),
            start_time: snapshot.start_timestamp,
            end_time: now,
            duration_ms,
            initial_memory: snapshot.initial_memory,
            peak_memory: snapshot.peak_memory,
            final_memory: snapshot.current_memory,
            total_allocated: snapshot.total_allocated,
            total_deallocated: snapshot.total_deallocated,
            net_change,
            allocation_count: snapshot.allocation_count,
            deallocation_count: snapshot.deallocation_count,
            leaked_bytes,
            metadata: HashMap::new(),
        }
    }

    /// Check if memory was leaked
    pub fn has_leak(&self) -> bool {
        self.leaked_bytes > 0
    }

    /// Get memory efficiency (deallocated / allocated)
    pub fn efficiency(&self) -> f64 {
        if self.total_allocated == 0 {
            1.0
        } else {
            self.total_deallocated as f64 / self.total_allocated as f64
        }
    }
}

impl MemoryTracker {
    /// Create a new memory tracker
    pub fn new(config: MemoryTrackingConfig) -> Self {
        Self {
            snapshots: Arc::new(RwLock::new(HashMap::new())),
            completed: Arc::new(RwLock::new(Vec::new())),
            system: Arc::new(RwLock::new(System::new_all())),
            config,
        }
    }

    /// Start tracking a query
    pub async fn start_tracking(
        &self,
        query_id: String,
        operation_name: Option<String>,
    ) -> Result<()> {
        // Refresh system info
        let mut system = self.system.write().await;
        system.refresh_memory();

        // Get current process memory (estimate)
        let current_memory = system.used_memory();

        drop(system);

        let snapshot = MemorySnapshot::new(query_id.clone(), operation_name, current_memory);

        let mut snapshots = self.snapshots.write().await;
        snapshots.insert(query_id, snapshot);

        Ok(())
    }

    /// Record an allocation
    pub async fn record_allocation(&self, query_id: &str, bytes: u64) -> Result<()> {
        let mut snapshots = self.snapshots.write().await;

        if let Some(snapshot) = snapshots.get_mut(query_id) {
            snapshot.record_allocation(bytes);

            // Check warning threshold
            if snapshot.current_memory > self.config.warning_threshold_bytes {
                tracing::warn!(
                    query_id = %query_id,
                    current_memory = snapshot.current_memory,
                    threshold = self.config.warning_threshold_bytes,
                    "Query exceeded memory warning threshold"
                );
            }
        }

        Ok(())
    }

    /// Record a deallocation
    pub async fn record_deallocation(&self, query_id: &str, bytes: u64) -> Result<()> {
        let mut snapshots = self.snapshots.write().await;

        if let Some(snapshot) = snapshots.get_mut(query_id) {
            snapshot.record_deallocation(bytes);
        }

        Ok(())
    }

    /// Stop tracking and get allocation record
    pub async fn stop_tracking(&self, query_id: &str) -> Result<MemoryAllocation> {
        let mut snapshots = self.snapshots.write().await;

        let snapshot = snapshots
            .remove(query_id)
            .ok_or_else(|| anyhow::anyhow!("Query not found: {}", query_id))?;

        let allocation = MemoryAllocation::from_snapshot(snapshot);

        // Store in completed
        let mut completed = self.completed.write().await;
        completed.push(allocation.clone());

        // Cleanup old allocations
        self.cleanup_old_allocations(&mut completed).await;

        Ok(allocation)
    }

    /// Get current snapshot
    pub async fn get_snapshot(&self, query_id: &str) -> Option<MemorySnapshot> {
        let snapshots = self.snapshots.read().await;
        snapshots.get(query_id).cloned()
    }

    /// Get completed allocations
    pub async fn get_completed_allocations(&self) -> Vec<MemoryAllocation> {
        let completed = self.completed.read().await;
        completed.clone()
    }

    /// Get allocations by operation
    pub async fn get_allocations_by_operation(
        &self,
        operation_name: &str,
    ) -> Vec<MemoryAllocation> {
        let completed = self.completed.read().await;
        completed
            .iter()
            .filter(|a| {
                a.operation_name
                    .as_ref()
                    .map(|n| n == operation_name)
                    .unwrap_or(false)
            })
            .cloned()
            .collect()
    }

    /// Get memory leak statistics
    pub async fn get_leak_statistics(&self) -> MemoryLeakStatistics {
        let completed = self.completed.read().await;

        let mut stats = MemoryLeakStatistics::default();

        for allocation in completed.iter() {
            stats.total_queries += 1;
            stats.total_allocated += allocation.total_allocated;
            stats.total_deallocated += allocation.total_deallocated;

            if allocation.has_leak() {
                stats.queries_with_leaks += 1;
                stats.total_leaked += allocation.leaked_bytes;
            }
        }

        if stats.total_queries > 0 {
            stats.avg_allocation_per_query = stats.total_allocated / stats.total_queries;
            stats.leak_rate = stats.queries_with_leaks as f64 / stats.total_queries as f64;
        }

        stats
    }

    /// Get top memory consumers sorted by total_allocated (the query's own allocations)
    pub async fn get_top_consumers(&self, limit: usize) -> Vec<MemoryAllocation> {
        let completed = self.completed.read().await;

        let mut sorted = completed.clone();
        // Sort by total_allocated (query's own allocations) rather than peak_memory
        // which includes varying system memory baseline
        sorted.sort_by(|a, b| b.total_allocated.cmp(&a.total_allocated));
        sorted.truncate(limit);
        sorted
    }

    /// Get memory statistics
    pub async fn get_statistics(&self) -> MemoryStatistics {
        let snapshots = self.snapshots.read().await;
        let completed = self.completed.read().await;

        let active_queries = snapshots.len();
        let completed_queries = completed.len();

        let mut total_allocated = 0;
        let mut total_deallocated = 0;
        let mut total_leaked = 0;

        for allocation in completed.iter() {
            total_allocated += allocation.total_allocated;
            total_deallocated += allocation.total_deallocated;
            total_leaked += allocation.leaked_bytes;
        }

        MemoryStatistics {
            active_queries,
            completed_queries,
            total_allocated,
            total_deallocated,
            total_leaked,
        }
    }

    /// Cleanup old allocations
    async fn cleanup_old_allocations(&self, completed: &mut Vec<MemoryAllocation>) {
        let retention_secs = self.config.retention_period.as_secs();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs();

        completed.retain(|a| now - a.end_time < retention_secs);

        // Also limit by count
        if completed.len() > self.config.max_allocations {
            let excess = completed.len() - self.config.max_allocations;
            completed.drain(0..excess);
        }
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new(MemoryTrackingConfig::default())
    }
}

/// Memory leak statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryLeakStatistics {
    /// Total queries tracked
    pub total_queries: u64,
    /// Queries with memory leaks
    pub queries_with_leaks: u64,
    /// Total bytes allocated
    pub total_allocated: u64,
    /// Total bytes deallocated
    pub total_deallocated: u64,
    /// Total bytes leaked
    pub total_leaked: u64,
    /// Average allocation per query
    pub avg_allocation_per_query: u64,
    /// Leak rate (0.0 to 1.0)
    pub leak_rate: f64,
}

impl MemoryLeakStatistics {
    /// Get efficiency (deallocated / allocated)
    pub fn efficiency(&self) -> f64 {
        if self.total_allocated == 0 {
            1.0
        } else {
            self.total_deallocated as f64 / self.total_allocated as f64
        }
    }
}

/// Memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatistics {
    /// Active queries being tracked
    pub active_queries: usize,
    /// Completed queries
    pub completed_queries: usize,
    /// Total bytes allocated
    pub total_allocated: u64,
    /// Total bytes deallocated
    pub total_deallocated: u64,
    /// Total bytes leaked
    pub total_leaked: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_snapshot_creation() {
        let snapshot =
            MemorySnapshot::new("query-1".to_string(), Some("GetUser".to_string()), 1000);

        assert_eq!(snapshot.query_id, "query-1");
        assert_eq!(snapshot.operation_name, Some("GetUser".to_string()));
        assert_eq!(snapshot.initial_memory, 1000);
        assert_eq!(snapshot.peak_memory, 1000);
        assert_eq!(snapshot.current_memory, 1000);
        assert_eq!(snapshot.allocation_count, 0);
    }

    #[test]
    fn test_snapshot_record_allocation() {
        let mut snapshot = MemorySnapshot::new("query-1".to_string(), None, 1000);

        snapshot.record_allocation(500);

        assert_eq!(snapshot.allocation_count, 1);
        assert_eq!(snapshot.total_allocated, 500);
        assert_eq!(snapshot.current_memory, 1500);
        assert_eq!(snapshot.peak_memory, 1500);
    }

    #[test]
    fn test_snapshot_record_deallocation() {
        let mut snapshot = MemorySnapshot::new("query-1".to_string(), None, 1000);

        snapshot.record_allocation(500);
        snapshot.record_deallocation(300);

        assert_eq!(snapshot.deallocation_count, 1);
        assert_eq!(snapshot.total_deallocated, 300);
        assert_eq!(snapshot.current_memory, 1200);
        assert_eq!(snapshot.peak_memory, 1500);
    }

    #[test]
    fn test_snapshot_net_memory_change() {
        let mut snapshot = MemorySnapshot::new("query-1".to_string(), None, 1000);

        snapshot.record_allocation(500);
        snapshot.record_deallocation(300);

        assert_eq!(snapshot.net_memory_change(), 200);
    }

    #[test]
    fn test_memory_allocation_from_snapshot() {
        let mut snapshot = MemorySnapshot::new("query-1".to_string(), None, 1000);

        snapshot.record_allocation(500);
        snapshot.record_deallocation(300);

        let allocation = MemoryAllocation::from_snapshot(snapshot);

        assert_eq!(allocation.query_id, "query-1");
        assert_eq!(allocation.total_allocated, 500);
        assert_eq!(allocation.total_deallocated, 300);
        assert_eq!(allocation.net_change, 200);
        assert_eq!(allocation.leaked_bytes, 200);
        assert!(allocation.has_leak());
    }

    #[test]
    fn test_memory_allocation_no_leak() {
        let mut snapshot = MemorySnapshot::new("query-1".to_string(), None, 1000);

        snapshot.record_allocation(500);
        snapshot.record_deallocation(500);

        let allocation = MemoryAllocation::from_snapshot(snapshot);

        assert_eq!(allocation.net_change, 0);
        assert_eq!(allocation.leaked_bytes, 0);
        assert!(!allocation.has_leak());
    }

    #[test]
    fn test_memory_allocation_efficiency() {
        let mut snapshot = MemorySnapshot::new("query-1".to_string(), None, 1000);

        snapshot.record_allocation(1000);
        snapshot.record_deallocation(800);

        let allocation = MemoryAllocation::from_snapshot(snapshot);

        assert_eq!(allocation.efficiency(), 0.8);
    }

    #[tokio::test]
    async fn test_memory_tracker_start_stop() {
        let tracker = MemoryTracker::default();

        tracker
            .start_tracking("query-1".to_string(), Some("GetUser".to_string()))
            .await
            .unwrap();

        let snapshot = tracker.get_snapshot("query-1").await;
        assert!(snapshot.is_some());

        let allocation = tracker.stop_tracking("query-1").await.unwrap();
        assert_eq!(allocation.query_id, "query-1");

        let snapshot = tracker.get_snapshot("query-1").await;
        assert!(snapshot.is_none());
    }

    #[tokio::test]
    async fn test_memory_tracker_record_allocations() {
        let tracker = MemoryTracker::default();

        tracker
            .start_tracking("query-1".to_string(), None)
            .await
            .unwrap();

        tracker.record_allocation("query-1", 500).await.unwrap();
        tracker.record_allocation("query-1", 300).await.unwrap();

        let snapshot = tracker.get_snapshot("query-1").await.unwrap();
        assert_eq!(snapshot.allocation_count, 2);
        assert_eq!(snapshot.total_allocated, 800);
    }

    #[tokio::test]
    async fn test_memory_tracker_record_deallocations() {
        let tracker = MemoryTracker::default();

        tracker
            .start_tracking("query-1".to_string(), None)
            .await
            .unwrap();

        tracker.record_allocation("query-1", 500).await.unwrap();
        tracker.record_deallocation("query-1", 200).await.unwrap();

        let snapshot = tracker.get_snapshot("query-1").await.unwrap();
        assert_eq!(snapshot.deallocation_count, 1);
        assert_eq!(snapshot.total_deallocated, 200);
    }

    #[tokio::test]
    async fn test_memory_tracker_completed_allocations() {
        let tracker = MemoryTracker::default();

        tracker
            .start_tracking("query-1".to_string(), None)
            .await
            .unwrap();
        tracker.record_allocation("query-1", 500).await.unwrap();
        tracker.stop_tracking("query-1").await.unwrap();

        let completed = tracker.get_completed_allocations().await;
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].query_id, "query-1");
    }

    #[tokio::test]
    async fn test_memory_tracker_allocations_by_operation() {
        let tracker = MemoryTracker::default();

        tracker
            .start_tracking("query-1".to_string(), Some("GetUser".to_string()))
            .await
            .unwrap();
        tracker.record_allocation("query-1", 500).await.unwrap();
        tracker.stop_tracking("query-1").await.unwrap();

        tracker
            .start_tracking("query-2".to_string(), Some("GetPosts".to_string()))
            .await
            .unwrap();
        tracker.record_allocation("query-2", 300).await.unwrap();
        tracker.stop_tracking("query-2").await.unwrap();

        let allocations = tracker.get_allocations_by_operation("GetUser").await;
        assert_eq!(allocations.len(), 1);
        assert_eq!(allocations[0].operation_name, Some("GetUser".to_string()));
    }

    #[tokio::test]
    async fn test_memory_tracker_leak_statistics() {
        let tracker = MemoryTracker::default();

        // Query with leak
        tracker
            .start_tracking("query-1".to_string(), None)
            .await
            .unwrap();
        tracker.record_allocation("query-1", 1000).await.unwrap();
        tracker.record_deallocation("query-1", 800).await.unwrap();
        tracker.stop_tracking("query-1").await.unwrap();

        // Query without leak
        tracker
            .start_tracking("query-2".to_string(), None)
            .await
            .unwrap();
        tracker.record_allocation("query-2", 500).await.unwrap();
        tracker.record_deallocation("query-2", 500).await.unwrap();
        tracker.stop_tracking("query-2").await.unwrap();

        let stats = tracker.get_leak_statistics().await;
        assert_eq!(stats.total_queries, 2);
        assert_eq!(stats.queries_with_leaks, 1);
        assert_eq!(stats.total_leaked, 200);
        assert_eq!(stats.leak_rate, 0.5);
    }

    #[tokio::test]
    async fn test_memory_tracker_top_consumers() {
        let tracker = MemoryTracker::default();

        tracker
            .start_tracking("query-1".to_string(), None)
            .await
            .unwrap();
        tracker.record_allocation("query-1", 1000).await.unwrap();
        tracker.stop_tracking("query-1").await.unwrap();

        tracker
            .start_tracking("query-2".to_string(), None)
            .await
            .unwrap();
        tracker.record_allocation("query-2", 2000).await.unwrap();
        tracker.stop_tracking("query-2").await.unwrap();

        tracker
            .start_tracking("query-3".to_string(), None)
            .await
            .unwrap();
        tracker.record_allocation("query-3", 500).await.unwrap();
        tracker.stop_tracking("query-3").await.unwrap();

        let top = tracker.get_top_consumers(2).await;
        assert_eq!(top.len(), 2);
        // Verify ordering by total_allocated (highest first)
        assert!(
            top[0].total_allocated >= top[1].total_allocated,
            "Top consumers should be sorted by total_allocated descending: {:?}",
            top.iter()
                .map(|t| (&t.query_id, t.total_allocated))
                .collect::<Vec<_>>()
        );
        // Verify the highest consumer has the expected total_allocated (query-2 with 2000)
        assert_eq!(
            top[0].total_allocated, 2000,
            "Highest consumer should have 2000 bytes allocated"
        );
        assert_eq!(
            top[0].query_id, "query-2",
            "Highest consumer should be query-2"
        );
        // Second highest is query-1 with 1000
        assert_eq!(
            top[1].total_allocated, 1000,
            "Second highest consumer should have 1000 bytes allocated"
        );
        assert_eq!(
            top[1].query_id, "query-1",
            "Second highest consumer should be query-1"
        );
    }

    #[tokio::test]
    async fn test_memory_tracker_statistics() {
        let tracker = MemoryTracker::default();

        tracker
            .start_tracking("query-1".to_string(), None)
            .await
            .unwrap();
        tracker
            .start_tracking("query-2".to_string(), None)
            .await
            .unwrap();

        tracker.record_allocation("query-1", 500).await.unwrap();
        tracker.stop_tracking("query-1").await.unwrap();

        let stats = tracker.get_statistics().await;
        assert_eq!(stats.active_queries, 1); // query-2 still active
        assert_eq!(stats.completed_queries, 1);
    }

    #[test]
    fn test_memory_tracking_config() {
        let config = MemoryTrackingConfig::default();

        assert!(config.enable_detailed_tracking);
        assert_eq!(config.max_allocations, 1000);
        assert_eq!(config.warning_threshold_bytes, 100 * 1024 * 1024);
    }

    #[test]
    fn test_memory_leak_statistics_efficiency() {
        let stats = MemoryLeakStatistics {
            total_allocated: 1000,
            total_deallocated: 800,
            ..Default::default()
        };

        assert_eq!(stats.efficiency(), 0.8);
    }

    #[tokio::test]
    async fn test_memory_tracker_not_found() {
        let tracker = MemoryTracker::default();

        let result = tracker.stop_tracking("nonexistent").await;
        assert!(result.is_err());

        let snapshot = tracker.get_snapshot("nonexistent").await;
        assert!(snapshot.is_none());
    }

    #[test]
    fn test_snapshot_peak_memory() {
        let mut snapshot = MemorySnapshot::new("query-1".to_string(), None, 1000);

        snapshot.record_allocation(500);
        assert_eq!(snapshot.peak_memory, 1500);

        snapshot.record_allocation(300);
        assert_eq!(snapshot.peak_memory, 1800);

        snapshot.record_deallocation(500);
        assert_eq!(snapshot.peak_memory, 1800); // Peak doesn't decrease
        assert_eq!(snapshot.current_memory, 1300);
    }

    #[test]
    fn test_memory_allocation_metadata() {
        let snapshot = MemorySnapshot::new("query-1".to_string(), None, 1000);
        let mut allocation = MemoryAllocation::from_snapshot(snapshot);

        allocation
            .metadata
            .insert("user_id".to_string(), "123".to_string());
        allocation
            .metadata
            .insert("query_type".to_string(), "mutation".to_string());

        assert_eq!(allocation.metadata.get("user_id"), Some(&"123".to_string()));
        assert_eq!(
            allocation.metadata.get("query_type"),
            Some(&"mutation".to_string())
        );
    }
}
