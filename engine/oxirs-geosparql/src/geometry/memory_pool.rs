//! Memory pool for efficient geometry allocation
//!
//! This module provides a thread-safe memory pool for geometry objects to reduce
//! allocation overhead when creating many geometries in bulk operations.
//!
//! ## Features
//!
//! - Thread-safe geometry allocation and deallocation
//! - Pre-allocated geometry objects for common types
//! - Configurable pool sizes for different geometry types
//! - Automatic pool expansion when needed
//! - Memory statistics and monitoring
//!
//! ## Example
//!
//! ```rust
//! use oxirs_geosparql::geometry::memory_pool::{GeometryPool, PoolConfig};
//!
//! // Create a pool with default configuration
//! let pool = GeometryPool::new(PoolConfig::default());
//!
//! // Allocate geometries from the pool
//! let point1 = pool.alloc_point();
//! let point2 = pool.alloc_point();
//!
//! // Get pool statistics
//! let stats = pool.stats();
//! println!("Total allocations: {}", stats.total_allocations);
//! println!("Pool hits: {}", stats.pool_hits);
//! println!("Memory used: {} bytes", stats.memory_used_bytes);
//! ```

use crate::error::Result;
use geo_types::{LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon};
use parking_lot::RwLock;
use std::sync::Arc;

/// Configuration for geometry memory pool
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Initial size for Point pool
    pub point_pool_size: usize,
    /// Initial size for LineString pool
    pub linestring_pool_size: usize,
    /// Initial size for Polygon pool
    pub polygon_pool_size: usize,
    /// Initial size for MultiPoint pool
    pub multipoint_pool_size: usize,
    /// Initial size for MultiLineString pool
    pub multilinestring_pool_size: usize,
    /// Initial size for MultiPolygon pool
    pub multipolygon_pool_size: usize,
    /// Maximum pool size before stopping to cache (0 = unlimited)
    pub max_pool_size: usize,
    /// Enable pool statistics tracking
    pub enable_stats: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            point_pool_size: 1000,
            linestring_pool_size: 500,
            polygon_pool_size: 200,
            multipoint_pool_size: 100,
            multilinestring_pool_size: 100,
            multipolygon_pool_size: 50,
            max_pool_size: 10000,
            enable_stats: true,
        }
    }
}

/// Statistics for memory pool usage
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total number of allocations requested
    pub total_allocations: u64,
    /// Number of allocations served from pool (cache hits)
    pub pool_hits: u64,
    /// Number of allocations that needed new memory (cache misses)
    pub pool_misses: u64,
    /// Total number of objects returned to pool
    pub total_returns: u64,
    /// Current number of pooled Point objects
    pub points_pooled: usize,
    /// Current number of pooled LineString objects
    pub linestrings_pooled: usize,
    /// Current number of pooled Polygon objects
    pub polygons_pooled: usize,
    /// Current number of pooled MultiPoint objects
    pub multipoints_pooled: usize,
    /// Current number of pooled MultiLineString objects
    pub multilinestrings_pooled: usize,
    /// Current number of pooled MultiPolygon objects
    pub multipolygons_pooled: usize,
    /// Estimated memory used by pool in bytes
    pub memory_used_bytes: usize,
}

impl PoolStats {
    /// Calculate cache hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        if self.total_allocations == 0 {
            0.0
        } else {
            self.pool_hits as f64 / self.total_allocations as f64
        }
    }

    /// Total number of objects currently in all pools
    pub fn total_pooled(&self) -> usize {
        self.points_pooled
            + self.linestrings_pooled
            + self.polygons_pooled
            + self.multipoints_pooled
            + self.multilinestrings_pooled
            + self.multipolygons_pooled
    }
}

/// Internal pool storage
struct PoolStorage {
    points: Vec<Point<f64>>,
    linestrings: Vec<LineString<f64>>,
    polygons: Vec<Polygon<f64>>,
    multipoints: Vec<MultiPoint<f64>>,
    multilinestrings: Vec<MultiLineString<f64>>,
    multipolygons: Vec<MultiPolygon<f64>>,
    stats: PoolStats,
    config: PoolConfig,
}

impl PoolStorage {
    fn new(config: PoolConfig) -> Self {
        let mut storage = Self {
            points: Vec::with_capacity(config.point_pool_size),
            linestrings: Vec::with_capacity(config.linestring_pool_size),
            polygons: Vec::with_capacity(config.polygon_pool_size),
            multipoints: Vec::with_capacity(config.multipoint_pool_size),
            multilinestrings: Vec::with_capacity(config.multilinestring_pool_size),
            multipolygons: Vec::with_capacity(config.multipolygon_pool_size),
            stats: PoolStats::default(),
            config,
        };

        // Pre-allocate empty geometries
        storage.preallocate();
        storage
    }

    fn preallocate(&mut self) {
        // Pre-fill Point pool
        for _ in 0..self.config.point_pool_size {
            self.points.push(Point::new(0.0, 0.0));
        }

        // Pre-fill other pools with minimal empty geometries
        for _ in 0..self.config.linestring_pool_size {
            self.linestrings.push(LineString::new(vec![]));
        }

        for _ in 0..self.config.polygon_pool_size {
            self.polygons
                .push(Polygon::new(LineString::new(vec![]), vec![]));
        }

        for _ in 0..self.config.multipoint_pool_size {
            self.multipoints.push(MultiPoint::new(vec![]));
        }

        for _ in 0..self.config.multilinestring_pool_size {
            self.multilinestrings.push(MultiLineString::new(vec![]));
        }

        for _ in 0..self.config.multipolygon_pool_size {
            self.multipolygons.push(MultiPolygon::new(vec![]));
        }

        self.update_memory_stats();
    }

    fn update_memory_stats(&mut self) {
        if !self.config.enable_stats {
            return;
        }

        // Estimate memory usage (approximate)
        let point_size = std::mem::size_of::<Point<f64>>();
        let linestring_size = std::mem::size_of::<LineString<f64>>() + 32; // estimate for Vec
        let polygon_size = std::mem::size_of::<Polygon<f64>>() + 64; // estimate for Vec
        let multipoint_size = std::mem::size_of::<MultiPoint<f64>>() + 32;
        let multilinestring_size = std::mem::size_of::<MultiLineString<f64>>() + 32;
        let multipolygon_size = std::mem::size_of::<MultiPolygon<f64>>() + 64;

        self.stats.points_pooled = self.points.len();
        self.stats.linestrings_pooled = self.linestrings.len();
        self.stats.polygons_pooled = self.polygons.len();
        self.stats.multipoints_pooled = self.multipoints.len();
        self.stats.multilinestrings_pooled = self.multilinestrings.len();
        self.stats.multipolygons_pooled = self.multipolygons.len();

        self.stats.memory_used_bytes = self.points.len() * point_size
            + self.linestrings.len() * linestring_size
            + self.polygons.len() * polygon_size
            + self.multipoints.len() * multipoint_size
            + self.multilinestrings.len() * multilinestring_size
            + self.multipolygons.len() * multipolygon_size;
    }
}

/// Thread-safe geometry memory pool
///
/// This pool manages pre-allocated geometry objects to reduce allocation overhead
/// during bulk geometry operations. The pool is thread-safe and can be shared
/// across multiple threads.
#[derive(Clone)]
pub struct GeometryPool {
    storage: Arc<RwLock<PoolStorage>>,
}

impl GeometryPool {
    /// Create a new geometry pool with the given configuration
    pub fn new(config: PoolConfig) -> Self {
        Self {
            storage: Arc::new(RwLock::new(PoolStorage::new(config))),
        }
    }

    /// Create a geometry pool with default configuration
    pub fn default_pool() -> Self {
        Self::new(PoolConfig::default())
    }

    /// Allocate a Point from the pool
    ///
    /// If the pool has available Point objects, one will be returned (pool hit).
    /// Otherwise, a new Point will be created (pool miss).
    pub fn alloc_point(&self) -> Point<f64> {
        let mut storage = self.storage.write();

        if storage.config.enable_stats {
            storage.stats.total_allocations += 1;
        }

        if let Some(point) = storage.points.pop() {
            if storage.config.enable_stats {
                storage.stats.pool_hits += 1;
            }
            point
        } else {
            if storage.config.enable_stats {
                storage.stats.pool_misses += 1;
            }
            Point::new(0.0, 0.0)
        }
    }

    /// Allocate a LineString from the pool
    pub fn alloc_linestring(&self) -> LineString<f64> {
        let mut storage = self.storage.write();

        if storage.config.enable_stats {
            storage.stats.total_allocations += 1;
        }

        if let Some(linestring) = storage.linestrings.pop() {
            if storage.config.enable_stats {
                storage.stats.pool_hits += 1;
            }
            linestring
        } else {
            if storage.config.enable_stats {
                storage.stats.pool_misses += 1;
            }
            LineString::new(vec![])
        }
    }

    /// Allocate a Polygon from the pool
    pub fn alloc_polygon(&self) -> Polygon<f64> {
        let mut storage = self.storage.write();

        if storage.config.enable_stats {
            storage.stats.total_allocations += 1;
        }

        if let Some(polygon) = storage.polygons.pop() {
            if storage.config.enable_stats {
                storage.stats.pool_hits += 1;
            }
            polygon
        } else {
            if storage.config.enable_stats {
                storage.stats.pool_misses += 1;
            }
            Polygon::new(LineString::new(vec![]), vec![])
        }
    }

    /// Return a Point to the pool for reuse
    ///
    /// The point will be cleared and returned to the pool if there's space.
    /// If the pool is at max capacity, the point will be dropped.
    pub fn return_point(&self, _point: Point<f64>) {
        let mut storage = self.storage.write();

        if storage.config.enable_stats {
            storage.stats.total_returns += 1;
        }

        // Check if we should add to pool
        let max_size = storage.config.max_pool_size;
        if max_size == 0 || storage.points.len() < max_size {
            // Reset point to default state before returning to pool
            let reset_point = Point::new(0.0, 0.0);
            storage.points.push(reset_point);
            storage.update_memory_stats();
        }
    }

    /// Return a LineString to the pool for reuse
    pub fn return_linestring(&self, _linestring: LineString<f64>) {
        let mut storage = self.storage.write();

        if storage.config.enable_stats {
            storage.stats.total_returns += 1;
        }

        let max_size = storage.config.max_pool_size;
        if max_size == 0 || storage.linestrings.len() < max_size {
            // Create a fresh empty LineString
            storage.linestrings.push(LineString::new(vec![]));
            storage.update_memory_stats();
        }
    }

    /// Return a Polygon to the pool for reuse
    pub fn return_polygon(&self, _polygon: Polygon<f64>) {
        let mut storage = self.storage.write();

        if storage.config.enable_stats {
            storage.stats.total_returns += 1;
        }

        let max_size = storage.config.max_pool_size;
        if max_size == 0 || storage.polygons.len() < max_size {
            // Create a fresh empty Polygon
            storage
                .polygons
                .push(Polygon::new(LineString::new(vec![]), vec![]));
            storage.update_memory_stats();
        }
    }

    /// Get current pool statistics
    pub fn stats(&self) -> PoolStats {
        let mut storage = self.storage.write();
        storage.update_memory_stats();
        storage.stats.clone()
    }

    /// Clear all pooled geometries and reset statistics
    pub fn clear(&self) {
        let mut storage = self.storage.write();
        storage.points.clear();
        storage.linestrings.clear();
        storage.polygons.clear();
        storage.multipoints.clear();
        storage.multilinestrings.clear();
        storage.multipolygons.clear();
        storage.stats = PoolStats::default();
    }

    /// Shrink the pool to fit current usage
    ///
    /// This will free unused memory while keeping currently pooled objects
    pub fn shrink_to_fit(&self) {
        let mut storage = self.storage.write();
        storage.points.shrink_to_fit();
        storage.linestrings.shrink_to_fit();
        storage.polygons.shrink_to_fit();
        storage.multipoints.shrink_to_fit();
        storage.multilinestrings.shrink_to_fit();
        storage.multipolygons.shrink_to_fit();
        storage.update_memory_stats();
    }

    /// Preallocate additional capacity in the pool
    ///
    /// This is useful when you know you'll need many geometries soon
    pub fn reserve(&self, geometry_type: GeometryType, additional: usize) -> Result<()> {
        let mut storage = self.storage.write();

        match geometry_type {
            GeometryType::Point => {
                storage.points.reserve(additional);
                for _ in 0..additional {
                    storage.points.push(Point::new(0.0, 0.0));
                }
            }
            GeometryType::LineString => {
                storage.linestrings.reserve(additional);
                for _ in 0..additional {
                    storage.linestrings.push(LineString::new(vec![]));
                }
            }
            GeometryType::Polygon => {
                storage.polygons.reserve(additional);
                for _ in 0..additional {
                    storage
                        .polygons
                        .push(Polygon::new(LineString::new(vec![]), vec![]));
                }
            }
            GeometryType::MultiPoint => {
                storage.multipoints.reserve(additional);
                for _ in 0..additional {
                    storage.multipoints.push(MultiPoint::new(vec![]));
                }
            }
            GeometryType::MultiLineString => {
                storage.multilinestrings.reserve(additional);
                for _ in 0..additional {
                    storage.multilinestrings.push(MultiLineString::new(vec![]));
                }
            }
            GeometryType::MultiPolygon => {
                storage.multipolygons.reserve(additional);
                for _ in 0..additional {
                    storage.multipolygons.push(MultiPolygon::new(vec![]));
                }
            }
        }

        storage.update_memory_stats();
        Ok(())
    }
}

/// Enum representing different geometry types for pool operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryType {
    /// Point geometry
    Point,
    /// LineString geometry
    LineString,
    /// Polygon geometry
    Polygon,
    /// MultiPoint geometry
    MultiPoint,
    /// MultiLineString geometry
    MultiLineString,
    /// MultiPolygon geometry
    MultiPolygon,
}

impl Default for GeometryPool {
    fn default() -> Self {
        Self::default_pool()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let pool = GeometryPool::new(PoolConfig::default());
        let stats = pool.stats();

        // Should have pre-allocated geometries
        assert!(stats.points_pooled > 0);
        assert!(stats.memory_used_bytes > 0);
    }

    #[test]
    fn test_point_allocation_and_return() {
        let config = PoolConfig {
            point_pool_size: 10,
            enable_stats: true,
            ..Default::default()
        };
        let pool = GeometryPool::new(config);

        // Allocate a point (should be pool hit)
        let point = pool.alloc_point();
        let stats_after_alloc = pool.stats();
        assert_eq!(stats_after_alloc.total_allocations, 1);
        assert_eq!(stats_after_alloc.pool_hits, 1);
        assert_eq!(stats_after_alloc.points_pooled, 9); // 10 - 1

        // Return the point
        pool.return_point(point);
        let stats_after_return = pool.stats();
        assert_eq!(stats_after_return.total_returns, 1);
        assert_eq!(stats_after_return.points_pooled, 10); // Back to 10
    }

    #[test]
    fn test_pool_miss() {
        let config = PoolConfig {
            point_pool_size: 2,
            enable_stats: true,
            ..Default::default()
        };
        let pool = GeometryPool::new(config);

        // Allocate all pre-allocated points
        let _p1 = pool.alloc_point();
        let _p2 = pool.alloc_point();

        // This should be a pool miss
        let _p3 = pool.alloc_point();

        let stats = pool.stats();
        assert_eq!(stats.total_allocations, 3);
        assert_eq!(stats.pool_hits, 2);
        assert_eq!(stats.pool_misses, 1);
    }

    #[test]
    fn test_hit_rate_calculation() {
        let pool = GeometryPool::new(PoolConfig::default());

        // Allocate some points
        let _p1 = pool.alloc_point(); // hit
        let _p2 = pool.alloc_point(); // hit

        let stats = pool.stats();
        assert_eq!(stats.hit_rate(), 1.0); // 2/2 = 100%
    }

    #[test]
    fn test_max_pool_size() {
        let config = PoolConfig {
            point_pool_size: 2,
            max_pool_size: 2,
            enable_stats: true,
            ..Default::default()
        };
        let pool = GeometryPool::new(config);

        // Empty the pool
        let _p1 = pool.alloc_point();
        let _p2 = pool.alloc_point();

        // Return 3 points - only 2 should be kept
        pool.return_point(Point::new(1.0, 1.0));
        pool.return_point(Point::new(2.0, 2.0));
        pool.return_point(Point::new(3.0, 3.0));

        let stats = pool.stats();
        assert_eq!(stats.points_pooled, 2); // Max is 2
    }

    #[test]
    fn test_clear_pool() {
        let pool = GeometryPool::new(PoolConfig::default());

        // Allocate some geometries
        let _p1 = pool.alloc_point();
        let _p2 = pool.alloc_point();

        // Clear the pool
        pool.clear();

        let stats = pool.stats();
        assert_eq!(stats.points_pooled, 0);
        assert_eq!(stats.total_allocations, 0);
        assert_eq!(stats.pool_hits, 0);
    }

    #[test]
    fn test_reserve() {
        let config = PoolConfig {
            point_pool_size: 10,
            ..Default::default()
        };
        let pool = GeometryPool::new(config);

        // Reserve additional space
        pool.reserve(GeometryType::Point, 20).unwrap();

        let stats = pool.stats();
        assert_eq!(stats.points_pooled, 30); // 10 initial + 20 reserved
    }

    #[test]
    fn test_linestring_allocation() {
        let pool = GeometryPool::new(PoolConfig::default());

        let linestring = pool.alloc_linestring();
        assert_eq!(linestring.0.len(), 0); // Should be empty

        pool.return_linestring(linestring);
        let stats = pool.stats();
        assert!(stats.linestrings_pooled > 0);
    }

    #[test]
    fn test_polygon_allocation() {
        let pool = GeometryPool::new(PoolConfig::default());

        let polygon = pool.alloc_polygon();
        assert!(polygon.exterior().0.is_empty()); // Should be empty

        pool.return_polygon(polygon);
        let stats = pool.stats();
        assert!(stats.polygons_pooled > 0);
    }

    #[test]
    fn test_shrink_to_fit() {
        let config = PoolConfig {
            point_pool_size: 50,
            ..Default::default()
        };
        let pool = GeometryPool::new(config);

        // Allocate many points
        let mut points = Vec::new();
        for _ in 0..50 {
            points.push(pool.alloc_point());
        }

        // Return only a few
        for point in points.into_iter().take(5) {
            pool.return_point(point);
        }

        // Shrink to fit
        pool.shrink_to_fit();

        let stats = pool.stats();
        // After allocating 50 and returning 5, we should have 5 in the pool
        assert_eq!(stats.points_pooled, 5);
    }

    #[test]
    fn test_thread_safety() {
        use std::thread;

        let pool = Arc::new(GeometryPool::new(PoolConfig::default()));
        let mut handles = vec![];

        // Spawn multiple threads that allocate and return points
        for _ in 0..4 {
            let pool_clone = Arc::clone(&pool);
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    let point = pool_clone.alloc_point();
                    pool_clone.return_point(point);
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        let stats = pool.stats();
        assert_eq!(stats.total_allocations, 400); // 4 threads * 100 iterations
        assert_eq!(stats.total_returns, 400);
    }
}
