//! Tests for the materialized view manager / storage / rewriter / scheduler.

#![cfg(test)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime};

use crate::algebra::Algebra;
use crate::cost_model::{CostModel, CostModelConfig};
use crate::materialized_views::{
    MaintenanceScheduler, MaterializedViewConfig, MaterializedViewManager, QueryRewriter,
    ResourceLimits, SchedulerConfig, ViewData, ViewStorage,
};
use crate::statistics_collector::StatisticsCollector;

#[test]
fn test_materialized_view_manager_creation() {
    let config = MaterializedViewConfig::default();
    let cost_model = Arc::new(Mutex::new(CostModel::new(CostModelConfig::default())));
    let stats_collector = Arc::new(StatisticsCollector::new());

    let manager = MaterializedViewManager::new(config, cost_model, stats_collector);
    assert!(manager.is_ok());
}

#[test]
fn test_view_storage_memory() {
    let mut storage = ViewStorage::new(1024 * 1024); // 1MB

    let data = ViewData {
        results: vec![],
        size_bytes: 1000,
        row_count: 10,
        materialized_at: SystemTime::now(),
        checksum: 12345,
    };

    let result = storage.store_view_data("test_view".to_string(), data.clone());
    assert!(result.is_ok());

    // Verify round-trip through memory tier.
    let retrieved = storage.get_view_data("test_view");
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.row_count, data.row_count);
    assert_eq!(retrieved.checksum, data.checksum);
}

#[test]
fn test_view_storage_disk_spillover() {
    // Give a budget of only 1 byte so everything spills to disk.
    let mut storage = ViewStorage::new(1);

    let data = ViewData {
        results: vec![],
        size_bytes: 1000,
        row_count: 42,
        materialized_at: SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000),
        checksum: 99999,
    };

    let result = storage.store_view_data("disk_spill_view".to_string(), data.clone());
    assert!(
        result.is_ok(),
        "disk spillover should succeed: {:?}",
        result
    );

    // Verify round-trip through disk tier.
    let retrieved = storage.get_view_data("disk_spill_view");
    assert!(
        retrieved.is_some(),
        "should retrieve spilled view from disk"
    );
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.row_count, data.row_count);
    assert_eq!(retrieved.checksum, data.checksum);
    assert_eq!(retrieved.size_bytes, data.size_bytes);
}

#[test]
fn test_query_rewriter() {
    let rewriter = QueryRewriter::new().unwrap();
    let query = Algebra::Bgp(vec![]);
    let views = Arc::new(RwLock::new(HashMap::new()));
    let cost_model = Arc::new(Mutex::new(CostModel::new(CostModelConfig::default())));

    let result = rewriter.rewrite_query(&query, &views, &cost_model);
    assert!(result.is_ok());
}

#[test]
fn test_maintenance_scheduler() {
    let config = SchedulerConfig {
        max_concurrent_tasks: 4,
        default_interval: Duration::from_secs(3600),
        priority_threshold: 5,
        resource_limits: ResourceLimits {
            max_cpu_usage: 0.8,
            max_memory_usage: 1024 * 1024 * 1024,
            max_io_bandwidth: 100 * 1024 * 1024,
        },
    };

    let scheduler = MaintenanceScheduler::new(config);
    assert!(scheduler.is_ok());
}
