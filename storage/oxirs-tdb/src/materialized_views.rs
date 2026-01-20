//! Materialized Views for Query Acceleration
//!
//! This module provides materialized views (pre-computed query results) to accelerate
//! common SPARQL queries. Views can be maintained incrementally or refreshed on-demand.
//!
//! Features:
//! - View definition and storage
//! - Automatic view maintenance on data changes
//! - Query rewriting to use materialized views
//! - Multiple refresh strategies (immediate, deferred, on-demand)
//! - View invalidation tracking
//! - View statistics and monitoring

use crate::error::{Result, TdbError};
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Materialized view configuration
#[derive(Debug, Clone)]
pub struct MaterializedViewConfig {
    /// Maximum number of views
    pub max_views: usize,
    /// Default refresh strategy
    pub default_refresh_strategy: RefreshStrategy,
    /// Enable automatic view selection for queries
    pub enable_auto_selection: bool,
    /// Maximum view size (number of results)
    pub max_view_size: usize,
    /// View expiration time (0 = never expire)
    pub view_expiration: Duration,
}

impl Default for MaterializedViewConfig {
    fn default() -> Self {
        Self {
            max_views: 100,
            default_refresh_strategy: RefreshStrategy::Deferred,
            enable_auto_selection: true,
            max_view_size: 1_000_000,
            view_expiration: Duration::from_secs(3600), // 1 hour
        }
    }
}

/// Refresh strategy for materialized views
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefreshStrategy {
    /// Refresh immediately on data changes
    Immediate,
    /// Refresh periodically or on demand
    Deferred,
    /// Only refresh when explicitly requested
    Manual,
}

/// Materialized view definition
#[derive(Debug, Clone)]
pub struct MaterializedView {
    /// Unique view ID
    pub id: u64,
    /// View name
    pub name: String,
    /// Query pattern (simplified representation)
    pub query_pattern: String,
    /// Refresh strategy
    pub refresh_strategy: RefreshStrategy,
    /// Whether view is currently valid
    pub is_valid: bool,
    /// Creation time
    pub created_at: Instant,
    /// Last refresh time
    pub last_refreshed: Instant,
    /// Number of results in view
    pub result_count: usize,
    /// View data (in real implementation, this would be stored on disk)
    /// For now, we store as opaque byte arrays representing serialized results
    pub data: Vec<Vec<u8>>,
}

impl MaterializedView {
    /// Create a new materialized view
    pub fn new(id: u64, name: String, query_pattern: String, strategy: RefreshStrategy) -> Self {
        let now = Instant::now();
        Self {
            id,
            name,
            query_pattern,
            refresh_strategy: strategy,
            is_valid: false,
            created_at: now,
            last_refreshed: now,
            result_count: 0,
            data: Vec::new(),
        }
    }

    /// Check if view needs refresh
    pub fn needs_refresh(&self, max_age: Duration) -> bool {
        !self.is_valid || self.last_refreshed.elapsed() > max_age
    }

    /// Mark view as invalid (needs refresh)
    pub fn invalidate(&mut self) {
        self.is_valid = false;
    }

    /// Refresh view with new data
    pub fn refresh(&mut self, data: Vec<Vec<u8>>) {
        self.data = data;
        self.result_count = self.data.len();
        self.last_refreshed = Instant::now();
        self.is_valid = true;
    }

    /// Get view age
    pub fn age(&self) -> Duration {
        self.last_refreshed.elapsed()
    }
}

/// Materialized view manager
pub struct MaterializedViewManager {
    /// Configuration
    config: MaterializedViewConfig,
    /// Active views
    views: RwLock<HashMap<u64, Arc<RwLock<MaterializedView>>>>,
    /// View name to ID mapping
    name_to_id: RwLock<HashMap<String, u64>>,
    /// Query pattern to view IDs mapping (for query rewriting)
    pattern_index: RwLock<HashMap<String, HashSet<u64>>>,
    /// Next view ID
    next_view_id: AtomicU64,
    /// Statistics
    stats: MaterializedViewStats,
}

impl MaterializedViewManager {
    /// Create a new materialized view manager
    pub fn new(config: MaterializedViewConfig) -> Self {
        Self {
            config,
            views: RwLock::new(HashMap::new()),
            name_to_id: RwLock::new(HashMap::new()),
            pattern_index: RwLock::new(HashMap::new()),
            next_view_id: AtomicU64::new(1),
            stats: MaterializedViewStats::default(),
        }
    }

    /// Create a new materialized view
    pub fn create_view(
        &self,
        name: String,
        query_pattern: String,
        strategy: RefreshStrategy,
    ) -> Result<u64> {
        // Check if name already exists
        if self.name_to_id.read().contains_key(&name) {
            return Err(TdbError::Other(format!(
                "View with name '{}' already exists",
                name
            )));
        }

        // Check view limit
        if self.views.read().len() >= self.config.max_views {
            return Err(TdbError::Other(format!(
                "Maximum number of views ({}) reached",
                self.config.max_views
            )));
        }

        let view_id = self.next_view_id.fetch_add(1, Ordering::Relaxed);
        let view = MaterializedView::new(view_id, name.clone(), query_pattern.clone(), strategy);

        // Store view
        self.views
            .write()
            .insert(view_id, Arc::new(RwLock::new(view)));

        // Update name mapping
        self.name_to_id.write().insert(name, view_id);

        // Update pattern index
        self.pattern_index
            .write()
            .entry(query_pattern)
            .or_default()
            .insert(view_id);

        self.stats
            .total_views_created
            .fetch_add(1, Ordering::Relaxed);

        Ok(view_id)
    }

    /// Drop a materialized view
    pub fn drop_view(&self, view_id: u64) -> Result<()> {
        let views = self.views.read();
        let view = views
            .get(&view_id)
            .ok_or_else(|| TdbError::Other(format!("View {} not found", view_id)))?;

        let view_data = view.read();
        let name = view_data.name.clone();
        let pattern = view_data.query_pattern.clone();
        drop(view_data);
        drop(views);

        // Remove from all indexes
        self.views.write().remove(&view_id);
        self.name_to_id.write().remove(&name);

        if let Some(view_set) = self.pattern_index.write().get_mut(&pattern) {
            view_set.remove(&view_id);
        }

        self.stats
            .total_views_dropped
            .fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Refresh a specific view
    pub fn refresh_view(&self, view_id: u64, data: Vec<Vec<u8>>) -> Result<()> {
        let views = self.views.read();
        let view = views
            .get(&view_id)
            .ok_or_else(|| TdbError::Other(format!("View {} not found", view_id)))?;

        let mut view_data = view.write();

        // Check size limit
        if data.len() > self.config.max_view_size {
            return Err(TdbError::Other(format!(
                "View size {} exceeds limit {}",
                data.len(),
                self.config.max_view_size
            )));
        }

        view_data.refresh(data);
        self.stats.total_refreshes.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Invalidate views affected by data changes
    pub fn invalidate_views(&self, affected_patterns: &[String]) {
        let pattern_index = self.pattern_index.read();

        for pattern in affected_patterns {
            if let Some(view_ids) = pattern_index.get(pattern) {
                for &view_id in view_ids {
                    if let Some(view) = self.views.read().get(&view_id) {
                        let mut view_data = view.write();
                        if view_data.refresh_strategy == RefreshStrategy::Immediate {
                            // In real implementation, trigger immediate refresh
                            view_data.invalidate();
                            self.stats
                                .total_invalidations
                                .fetch_add(1, Ordering::Relaxed);
                        } else {
                            view_data.invalidate();
                            self.stats
                                .total_invalidations
                                .fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            }
        }
    }

    /// Find views that can answer a query pattern
    pub fn find_applicable_views(&self, query_pattern: &str) -> Vec<u64> {
        let pattern_index = self.pattern_index.read();

        pattern_index
            .get(query_pattern)
            .map(|view_ids| view_ids.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Get view by ID
    pub fn get_view(&self, view_id: u64) -> Option<Arc<RwLock<MaterializedView>>> {
        self.views.read().get(&view_id).cloned()
    }

    /// Get view by name
    pub fn get_view_by_name(&self, name: &str) -> Option<Arc<RwLock<MaterializedView>>> {
        let name_to_id = self.name_to_id.read();
        let view_id = name_to_id.get(name)?;
        self.views.read().get(view_id).cloned()
    }

    /// List all views
    pub fn list_views(&self) -> Vec<ViewInfo> {
        self.views
            .read()
            .values()
            .map(|view| {
                let v = view.read();
                ViewInfo {
                    id: v.id,
                    name: v.name.clone(),
                    query_pattern: v.query_pattern.clone(),
                    refresh_strategy: v.refresh_strategy,
                    is_valid: v.is_valid,
                    result_count: v.result_count,
                    age: v.age(),
                }
            })
            .collect()
    }

    /// Cleanup expired views
    pub fn cleanup_expired_views(&self) -> usize {
        if self.config.view_expiration.is_zero() {
            return 0;
        }

        let expired_views: Vec<u64> = self
            .views
            .read()
            .values()
            .filter_map(|view| {
                let v = view.read();
                if v.age() > self.config.view_expiration {
                    Some(v.id)
                } else {
                    None
                }
            })
            .collect();

        let count = expired_views.len();
        for view_id in expired_views {
            let _ = self.drop_view(view_id);
        }

        count
    }

    /// Get manager statistics
    pub fn stats(&self) -> MaterializedViewManagerStats {
        MaterializedViewManagerStats {
            total_views: self.views.read().len(),
            total_views_created: self.stats.total_views_created.load(Ordering::Relaxed),
            total_views_dropped: self.stats.total_views_dropped.load(Ordering::Relaxed),
            total_refreshes: self.stats.total_refreshes.load(Ordering::Relaxed),
            total_invalidations: self.stats.total_invalidations.load(Ordering::Relaxed),
            total_hits: self.stats.total_hits.load(Ordering::Relaxed),
            total_misses: self.stats.total_misses.load(Ordering::Relaxed),
        }
    }

    /// Record a view hit (query answered by view)
    pub fn record_hit(&self) {
        self.stats.total_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a view miss (query not answered by view)
    pub fn record_miss(&self) {
        self.stats.total_misses.fetch_add(1, Ordering::Relaxed);
    }
}

/// View information for listing
#[derive(Debug, Clone)]
pub struct ViewInfo {
    /// View ID
    pub id: u64,
    /// View name
    pub name: String,
    /// Query pattern
    pub query_pattern: String,
    /// Refresh strategy
    pub refresh_strategy: RefreshStrategy,
    /// Whether view is valid
    pub is_valid: bool,
    /// Number of results
    pub result_count: usize,
    /// View age
    pub age: Duration,
}

/// Materialized view statistics
#[derive(Debug, Default)]
struct MaterializedViewStats {
    /// Total views created
    total_views_created: AtomicU64,
    /// Total views dropped
    total_views_dropped: AtomicU64,
    /// Total view refreshes
    total_refreshes: AtomicU64,
    /// Total view invalidations
    total_invalidations: AtomicU64,
    /// Total view hits (queries answered by views)
    total_hits: AtomicU64,
    /// Total view misses (queries not answered by views)
    total_misses: AtomicU64,
}

/// Snapshot of materialized view manager statistics
#[derive(Debug, Clone)]
pub struct MaterializedViewManagerStats {
    /// Current number of views
    pub total_views: usize,
    /// Total views created
    pub total_views_created: u64,
    /// Total views dropped
    pub total_views_dropped: u64,
    /// Total refreshes performed
    pub total_refreshes: u64,
    /// Total invalidations
    pub total_invalidations: u64,
    /// Total view hits
    pub total_hits: u64,
    /// Total view misses
    pub total_misses: u64,
}

impl MaterializedViewManagerStats {
    /// Calculate view hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_hits + self.total_misses;
        if total == 0 {
            0.0
        } else {
            (self.total_hits as f64 / total as f64) * 100.0
        }
    }

    /// Calculate average refreshes per view
    pub fn avg_refreshes_per_view(&self) -> f64 {
        if self.total_views_created == 0 {
            0.0
        } else {
            self.total_refreshes as f64 / self.total_views_created as f64
        }
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn test_materialized_view_creation() {
        let view = MaterializedView::new(
            1,
            "test_view".to_string(),
            "SELECT * WHERE { ?s ?p ?o }".to_string(),
            RefreshStrategy::Deferred,
        );

        assert_eq!(view.id, 1);
        assert_eq!(view.name, "test_view");
        assert!(!view.is_valid);
        assert_eq!(view.result_count, 0);
    }

    #[test]
    fn test_view_refresh() {
        let mut view = MaterializedView::new(
            1,
            "test".to_string(),
            "pattern".to_string(),
            RefreshStrategy::Deferred,
        );

        assert!(!view.is_valid);

        let data = vec![vec![1, 2, 3], vec![4, 5, 6]];
        view.refresh(data.clone());

        assert!(view.is_valid);
        assert_eq!(view.result_count, 2);
        assert_eq!(view.data, data);
    }

    #[test]
    fn test_view_invalidation() {
        let mut view = MaterializedView::new(
            1,
            "test".to_string(),
            "pattern".to_string(),
            RefreshStrategy::Deferred,
        );

        view.refresh(vec![vec![1, 2, 3]]);
        assert!(view.is_valid);

        view.invalidate();
        assert!(!view.is_valid);
    }

    #[test]
    fn test_view_manager_creation() {
        let config = MaterializedViewConfig::default();
        let manager = MaterializedViewManager::new(config);

        let stats = manager.stats();
        assert_eq!(stats.total_views, 0);
        assert_eq!(stats.total_views_created, 0);
    }

    #[test]
    fn test_create_and_drop_view() {
        let config = MaterializedViewConfig::default();
        let manager = MaterializedViewManager::new(config);

        // Create view
        let view_id = manager
            .create_view(
                "test_view".to_string(),
                "SELECT * WHERE { ?s ?p ?o }".to_string(),
                RefreshStrategy::Deferred,
            )
            .unwrap();

        assert_eq!(view_id, 1);

        let stats = manager.stats();
        assert_eq!(stats.total_views, 1);
        assert_eq!(stats.total_views_created, 1);

        // Drop view
        manager.drop_view(view_id).unwrap();

        let stats = manager.stats();
        assert_eq!(stats.total_views, 0);
        assert_eq!(stats.total_views_dropped, 1);
    }

    #[test]
    fn test_duplicate_view_name() {
        let config = MaterializedViewConfig::default();
        let manager = MaterializedViewManager::new(config);

        // Create first view
        manager
            .create_view(
                "dup".to_string(),
                "pattern1".to_string(),
                RefreshStrategy::Deferred,
            )
            .unwrap();

        // Try to create view with same name
        let result = manager.create_view(
            "dup".to_string(),
            "pattern2".to_string(),
            RefreshStrategy::Deferred,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_max_views_limit() {
        let mut config = MaterializedViewConfig::default();
        config.max_views = 2;

        let manager = MaterializedViewManager::new(config);

        // Create two views (at limit)
        manager
            .create_view(
                "view1".to_string(),
                "pattern1".to_string(),
                RefreshStrategy::Deferred,
            )
            .unwrap();
        manager
            .create_view(
                "view2".to_string(),
                "pattern2".to_string(),
                RefreshStrategy::Deferred,
            )
            .unwrap();

        // Third view should fail
        let result = manager.create_view(
            "view3".to_string(),
            "pattern3".to_string(),
            RefreshStrategy::Deferred,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_refresh_view() {
        let config = MaterializedViewConfig::default();
        let manager = MaterializedViewManager::new(config);

        let view_id = manager
            .create_view(
                "test".to_string(),
                "pattern".to_string(),
                RefreshStrategy::Deferred,
            )
            .unwrap();

        let data = vec![vec![1, 2, 3], vec![4, 5, 6]];
        manager.refresh_view(view_id, data.clone()).unwrap();

        let view = manager.get_view(view_id).unwrap();
        let view_data = view.read();

        assert!(view_data.is_valid);
        assert_eq!(view_data.result_count, 2);
    }

    #[test]
    fn test_view_size_limit() {
        let mut config = MaterializedViewConfig::default();
        config.max_view_size = 2;

        let manager = MaterializedViewManager::new(config);

        let view_id = manager
            .create_view(
                "test".to_string(),
                "pattern".to_string(),
                RefreshStrategy::Deferred,
            )
            .unwrap();

        // Try to refresh with too much data
        let data = vec![vec![1], vec![2], vec![3]]; // 3 results > limit of 2
        let result = manager.refresh_view(view_id, data);

        assert!(result.is_err());
    }

    #[test]
    fn test_invalidate_views() {
        let config = MaterializedViewConfig::default();
        let manager = MaterializedViewManager::new(config);

        let view_id = manager
            .create_view(
                "test".to_string(),
                "pattern1".to_string(),
                RefreshStrategy::Deferred,
            )
            .unwrap();

        // Refresh view
        manager.refresh_view(view_id, vec![vec![1, 2, 3]]).unwrap();

        let view = manager.get_view(view_id).unwrap();
        assert!(view.read().is_valid);

        // Invalidate
        manager.invalidate_views(&["pattern1".to_string()]);

        assert!(!view.read().is_valid);

        let stats = manager.stats();
        assert_eq!(stats.total_invalidations, 1);
    }

    #[test]
    fn test_find_applicable_views() {
        let config = MaterializedViewConfig::default();
        let manager = MaterializedViewManager::new(config);

        let view_id1 = manager
            .create_view(
                "view1".to_string(),
                "pattern1".to_string(),
                RefreshStrategy::Deferred,
            )
            .unwrap();
        let view_id2 = manager
            .create_view(
                "view2".to_string(),
                "pattern1".to_string(),
                RefreshStrategy::Deferred,
            )
            .unwrap();

        let applicable = manager.find_applicable_views("pattern1");

        assert_eq!(applicable.len(), 2);
        assert!(applicable.contains(&view_id1));
        assert!(applicable.contains(&view_id2));
    }

    #[test]
    fn test_get_view_by_name() {
        let config = MaterializedViewConfig::default();
        let manager = MaterializedViewManager::new(config);

        manager
            .create_view(
                "myview".to_string(),
                "pattern".to_string(),
                RefreshStrategy::Deferred,
            )
            .unwrap();

        let view = manager.get_view_by_name("myview");
        assert!(view.is_some());

        let view_arc = view.unwrap();
        let view_data = view_arc.read();
        assert_eq!(view_data.name, "myview");
    }

    #[test]
    fn test_list_views() {
        let config = MaterializedViewConfig::default();
        let manager = MaterializedViewManager::new(config);

        manager
            .create_view(
                "view1".to_string(),
                "pattern1".to_string(),
                RefreshStrategy::Deferred,
            )
            .unwrap();
        manager
            .create_view(
                "view2".to_string(),
                "pattern2".to_string(),
                RefreshStrategy::Immediate,
            )
            .unwrap();

        let views = manager.list_views();
        assert_eq!(views.len(), 2);

        let names: Vec<String> = views.iter().map(|v| v.name.clone()).collect();
        assert!(names.contains(&"view1".to_string()));
        assert!(names.contains(&"view2".to_string()));
    }

    #[test]
    fn test_hit_miss_tracking() {
        let config = MaterializedViewConfig::default();
        let manager = MaterializedViewManager::new(config);

        manager.record_hit();
        manager.record_hit();
        manager.record_miss();

        let stats = manager.stats();
        assert_eq!(stats.total_hits, 2);
        assert_eq!(stats.total_misses, 1);
        assert!((stats.hit_rate() - 66.67).abs() < 0.1);
    }

    #[test]
    fn test_stats_calculations() {
        let stats = MaterializedViewManagerStats {
            total_views: 5,
            total_views_created: 10,
            total_views_dropped: 5,
            total_refreshes: 30,
            total_invalidations: 15,
            total_hits: 80,
            total_misses: 20,
        };

        assert_eq!(stats.hit_rate(), 80.0);
        assert_eq!(stats.avg_refreshes_per_view(), 3.0);
    }

    #[test]
    fn test_view_needs_refresh() {
        let view = MaterializedView::new(
            1,
            "test".to_string(),
            "pattern".to_string(),
            RefreshStrategy::Deferred,
        );

        // Invalid view needs refresh
        assert!(view.needs_refresh(Duration::from_secs(60)));

        // Valid view might not need refresh if fresh
        let mut view2 = view.clone();
        view2.refresh(vec![vec![1, 2, 3]]);
        assert!(!view2.needs_refresh(Duration::from_secs(3600)));
    }
}
