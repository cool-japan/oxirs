//! Tests for the TDB store.

#[cfg(test)]
mod tests {
    use crate::store::store_impl::TdbStore;
    use crate::store::store_types::{IndexMetrics, StorageMetrics, TdbConfig};
    use std::env;

    #[test]
    fn test_tdb_store_open() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_store_open");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let store = TdbStore::open(&temp_dir).unwrap();
        assert_eq!(store.count(), 0);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_tdb_store_insert_count() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_store_insert");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut store = TdbStore::open(&temp_dir).unwrap();

        store
            .insert(
                "http://example.org/alice",
                "http://example.org/knows",
                "http://example.org/bob",
            )
            .unwrap();

        assert_eq!(store.count(), 1);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_tdb_store_contains() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_store_contains");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut store = TdbStore::open(&temp_dir).unwrap();

        store
            .insert(
                "http://example.org/alice",
                "http://example.org/knows",
                "http://example.org/bob",
            )
            .unwrap();

        assert!(store
            .contains(
                "http://example.org/alice",
                "http://example.org/knows",
                "http://example.org/bob"
            )
            .unwrap());

        assert!(!store
            .contains(
                "http://example.org/alice",
                "http://example.org/knows",
                "http://example.org/charlie"
            )
            .unwrap_or(false));

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_tdb_store_delete() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_store_delete");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut store = TdbStore::open(&temp_dir).unwrap();

        store
            .insert(
                "http://example.org/alice",
                "http://example.org/knows",
                "http://example.org/bob",
            )
            .unwrap();

        assert_eq!(store.count(), 1);

        let deleted = store
            .delete(
                "http://example.org/alice",
                "http://example.org/knows",
                "http://example.org/bob",
            )
            .unwrap();

        assert!(deleted);
        assert_eq!(store.count(), 0);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_tdb_store_multiple_inserts() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_store_multiple");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut store = TdbStore::open(&temp_dir).unwrap();

        store
            .insert(
                "http://example.org/alice",
                "http://example.org/knows",
                "http://example.org/bob",
            )
            .unwrap();
        store
            .insert(
                "http://example.org/alice",
                "http://example.org/knows",
                "http://example.org/charlie",
            )
            .unwrap();
        store
            .insert(
                "http://example.org/bob",
                "http://example.org/likes",
                "http://example.org/pizza",
            )
            .unwrap();

        assert_eq!(store.count(), 3);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_tdb_store_config() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_store_config");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let config = TdbConfig::new(&temp_dir)
            .with_buffer_pool_size(2000)
            .with_compression(false)
            .with_bloom_filters(false);

        let store = TdbStore::open_with_config(config).unwrap();

        assert_eq!(store.config().buffer_pool_size, 2000);
        assert!(!store.config().enable_compression);
        assert!(!store.config().enable_bloom_filters);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_tdb_store_stats() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_store_stats");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut store = TdbStore::open(&temp_dir).unwrap();

        store
            .insert(
                "http://example.org/alice",
                "http://example.org/knows",
                "http://example.org/bob",
            )
            .unwrap();

        let stats = store.stats();
        assert_eq!(stats.triple_count, 1);
        assert!(stats.dictionary_size > 0);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_tdb_store_enhanced_stats() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_store_enhanced_stats");
        // Clean up any leftover data from previous runs
        std::fs::remove_dir_all(&temp_dir).ok();
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut store = TdbStore::open(&temp_dir).unwrap();

        // Insert some triples
        for i in 0..10 {
            store
                .insert(
                    &format!("http://example.org/s{}", i),
                    "http://example.org/knows",
                    &format!("http://example.org/o{}", i),
                )
                .unwrap();
        }

        // Get enhanced statistics
        let stats = store.enhanced_stats();

        // Verify basic stats
        assert_eq!(stats.basic.triple_count, 10);
        assert!(stats.basic.dictionary_size > 0);

        // Verify buffer pool stats
        assert!(
            stats
                .buffer_pool
                .total_fetches
                .load(std::sync::atomic::Ordering::Relaxed)
                > 0
        );
        assert!(stats.buffer_pool.hit_rate() >= 0.0);
        assert!(stats.buffer_pool.hit_rate() <= 1.0);

        // Verify storage metrics
        assert!(stats.storage.page_size > 0);
        assert!(stats.storage.memory_usage_bytes > 0);
        assert!(stats.storage.pages_allocated > 0);
        assert!(stats.storage.total_size_bytes > 0);

        // Verify storage efficiency calculations
        let efficiency = stats.storage.efficiency();
        assert!(efficiency >= 0.0);
        assert!(efficiency <= 1.0);

        let fragmentation = stats.storage.fragmentation();
        assert!(fragmentation >= 0.0);
        assert!(fragmentation <= 100.0);

        // Verify transaction metrics
        assert_eq!(stats.transaction.active_transactions, 0);
        assert!(stats.transaction.wal_enabled);

        // Verify index metrics
        assert_eq!(stats.index.spo_entries, 10);
        assert_eq!(stats.index.pos_entries, 10);
        assert_eq!(stats.index.osp_entries, 10);
        assert!(stats.index.indexes_consistent);
        assert_eq!(stats.index.total_entries(), 30);
        assert_eq!(stats.index.avg_entries_per_index(), 10.0);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_storage_metrics_calculations() {
        let metrics = StorageMetrics {
            total_size_bytes: 1000,
            pages_allocated: 10,
            page_size: 200,
            memory_usage_bytes: 2000,
        };

        // Efficiency = 1000 / (10 * 200) = 1000 / 2000 = 0.5
        assert_eq!(metrics.efficiency(), 0.5);

        // Fragmentation = (1.0 - 0.5) * 100 = 50%
        assert_eq!(metrics.fragmentation(), 50.0);
    }

    #[test]
    fn test_index_metrics_calculations() {
        let metrics = IndexMetrics {
            spo_entries: 100,
            pos_entries: 100,
            osp_entries: 100,
            indexes_consistent: true,
        };

        assert_eq!(metrics.total_entries(), 300);
        assert_eq!(metrics.avg_entries_per_index(), 100.0);
    }

    // ==================== Spatial Indexing Tests ====================

    #[test]
    fn test_spatial_indexing_enabled() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_spatial_enabled");
        std::fs::remove_dir_all(&temp_dir).ok();
        std::fs::create_dir_all(&temp_dir).unwrap();

        let store = TdbStore::open(&temp_dir).unwrap();
        assert!(store.is_spatial_indexing_enabled());

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_spatial_indexing_disabled() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_spatial_disabled");
        std::fs::remove_dir_all(&temp_dir).ok();
        std::fs::create_dir_all(&temp_dir).unwrap();

        let config = TdbConfig::new(&temp_dir).with_spatial_indexing(false);
        let store = TdbStore::open_with_config(config).unwrap();
        assert!(!store.is_spatial_indexing_enabled());

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_insert_point_geometry() {
        use crate::index::spatial::{Geometry, Point};

        let temp_dir = env::temp_dir().join("oxirs_tdb_insert_point");
        std::fs::remove_dir_all(&temp_dir).ok();
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut store = TdbStore::open(&temp_dir).unwrap();

        // Insert a point for New York City
        let point = Point::new(40.7128, -74.0060);
        store
            .insert_geometry("http://example.org/nyc", point.into())
            .unwrap();

        // Verify statistics
        let stats = store.spatial_statistics().unwrap();
        assert_eq!(stats.geometry_count, 1);
        assert_eq!(stats.points_count, 1);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_insert_multiple_geometries() {
        use crate::index::spatial::{Geometry, Point};

        let temp_dir = env::temp_dir().join("oxirs_tdb_multiple_geometries");
        std::fs::remove_dir_all(&temp_dir).ok();
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut store = TdbStore::open(&temp_dir).unwrap();

        // Insert multiple cities
        let cities = vec![
            ("http://example.org/nyc", Point::new(40.7128, -74.0060)),
            ("http://example.org/london", Point::new(51.5074, -0.1278)),
            ("http://example.org/tokyo", Point::new(35.6762, 139.6503)),
        ];

        for (uri, point) in cities {
            store.insert_geometry(uri, point.into()).unwrap();
        }

        // Verify statistics
        let stats = store.spatial_statistics().unwrap();
        assert_eq!(stats.geometry_count, 3);
        assert_eq!(stats.points_count, 3);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_spatial_query_within_distance() {
        use crate::index::spatial::{Geometry, Point, SpatialQuery};

        let temp_dir = env::temp_dir().join("oxirs_tdb_spatial_within_distance");
        std::fs::remove_dir_all(&temp_dir).ok();
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut store = TdbStore::open(&temp_dir).unwrap();

        // Insert Times Square and Central Park (both in NYC)
        store
            .insert_geometry(
                "http://example.org/times_square",
                Point::new(40.7589, -73.9851).into(),
            )
            .unwrap();
        store
            .insert_geometry(
                "http://example.org/central_park",
                Point::new(40.7829, -73.9654).into(),
            )
            .unwrap();

        // Query for points within 5km of Times Square
        let query = SpatialQuery::WithinDistance {
            center: Point::new(40.7589, -73.9851),
            distance: 5000.0,
        };

        let results = store.spatial_query(&query).unwrap();
        assert!(!results.is_empty());

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_spatial_query_intersects_bbox() {
        use crate::index::spatial::{BoundingBox, Geometry, Point, SpatialQuery};

        let temp_dir = env::temp_dir().join("oxirs_tdb_spatial_intersects");
        std::fs::remove_dir_all(&temp_dir).ok();
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut store = TdbStore::open(&temp_dir).unwrap();

        // Insert points
        store
            .insert_geometry("http://example.org/p1", Point::new(40.0, -74.0).into())
            .unwrap();
        store
            .insert_geometry("http://example.org/p2", Point::new(41.0, -73.0).into())
            .unwrap();
        store
            .insert_geometry("http://example.org/p3", Point::new(50.0, 0.0).into())
            .unwrap();

        // Query for points in a bounding box covering NYC area
        let query = SpatialQuery::IntersectsBBox {
            bbox: BoundingBox::new(39.0, -75.0, 42.0, -72.0),
        };

        let results = store.spatial_query(&query).unwrap();
        assert_eq!(results.len(), 2); // p1 and p2 should match, p3 should not

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_spatial_query_knn() {
        use crate::index::spatial::{Geometry, Point, SpatialQuery};

        let temp_dir = env::temp_dir().join("oxirs_tdb_spatial_knn");
        std::fs::remove_dir_all(&temp_dir).ok();
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut store = TdbStore::open(&temp_dir).unwrap();

        // Insert multiple points
        let points = vec![
            ("http://example.org/p1", Point::new(40.0, -74.0)),
            ("http://example.org/p2", Point::new(40.5, -74.0)),
            ("http://example.org/p3", Point::new(41.0, -74.0)),
            ("http://example.org/p4", Point::new(41.5, -74.0)),
            ("http://example.org/p5", Point::new(42.0, -74.0)),
        ];

        for (uri, point) in points {
            store.insert_geometry(uri, point.into()).unwrap();
        }

        // Query for 3 nearest neighbors to (40.0, -74.0)
        let query = SpatialQuery::KNearestNeighbors {
            point: Point::new(40.0, -74.0),
            k: 3,
        };

        let results = store.spatial_query(&query).unwrap();
        assert_eq!(results.len(), 3);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_remove_geometry() {
        use crate::index::spatial::{Geometry, Point};

        let temp_dir = env::temp_dir().join("oxirs_tdb_remove_geometry");
        std::fs::remove_dir_all(&temp_dir).ok();
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut store = TdbStore::open(&temp_dir).unwrap();

        // Insert a point
        store
            .insert_geometry(
                "http://example.org/nyc",
                Point::new(40.7128, -74.0060).into(),
            )
            .unwrap();

        // Verify it's there
        let stats = store.spatial_statistics().unwrap();
        assert_eq!(stats.geometry_count, 1);

        // Remove the geometry
        let removed = store.remove_geometry("http://example.org/nyc").unwrap();
        assert!(removed);

        // Verify it's gone
        let stats = store.spatial_statistics().unwrap();
        assert_eq!(stats.geometry_count, 0);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_remove_nonexistent_geometry() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_remove_nonexistent");
        std::fs::remove_dir_all(&temp_dir).ok();
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut store = TdbStore::open(&temp_dir).unwrap();

        // Try to remove a geometry that doesn't exist
        let removed = store
            .remove_geometry("http://example.org/nonexistent")
            .unwrap();
        assert!(!removed);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_spatial_operations_when_disabled() {
        use crate::index::spatial::{Geometry, Point};

        let temp_dir = env::temp_dir().join("oxirs_tdb_spatial_disabled_ops");
        std::fs::remove_dir_all(&temp_dir).ok();
        std::fs::create_dir_all(&temp_dir).unwrap();

        let config = TdbConfig::new(&temp_dir).with_spatial_indexing(false);
        let mut store = TdbStore::open_with_config(config).unwrap();

        // Try to insert geometry - should fail
        let result = store.insert_geometry(
            "http://example.org/nyc",
            Point::new(40.7128, -74.0060).into(),
        );
        assert!(result.is_err());

        // Try to query - should fail
        let query = crate::index::spatial::SpatialQuery::WithinDistance {
            center: Point::new(40.7589, -73.9851),
            distance: 5000.0,
        };
        let result = store.spatial_query(&query);
        assert!(result.is_err());

        // Try to get statistics - should fail
        let result = store.spatial_statistics();
        assert!(result.is_err());

        std::fs::remove_dir_all(&temp_dir).ok();
    }
}
