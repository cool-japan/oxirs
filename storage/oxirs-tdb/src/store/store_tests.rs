//! Tests for the TDB store.

#[cfg(test)]
mod tests {
    use crate::store::store_impl::TdbStore;
    use crate::store::store_types::{IndexMetrics, StorageMetrics, TdbConfig};
    use std::env;

    #[test]
    fn test_tdb_store_open() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_store_open");
        // Start from a clean slate: the store is now persistent, so leftover
        // data from a previous run would otherwise be reloaded.
        std::fs::remove_dir_all(&temp_dir).ok();
        std::fs::create_dir_all(&temp_dir).unwrap();

        let store = TdbStore::open(&temp_dir).unwrap();
        assert_eq!(store.count(), 0);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_tdb_store_insert_count() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_store_insert");
        std::fs::remove_dir_all(&temp_dir).ok();
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
        std::fs::remove_dir_all(&temp_dir).ok();
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
        std::fs::remove_dir_all(&temp_dir).ok();
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
        std::fs::remove_dir_all(&temp_dir).ok();
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
        std::fs::remove_dir_all(&temp_dir).ok();
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
        std::fs::remove_dir_all(&temp_dir).ok();
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
        use crate::index::spatial::Point;

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
        use crate::index::spatial::Point;

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
        use crate::index::spatial::{Point, SpatialQuery};

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
        use crate::index::spatial::{BoundingBox, Point, SpatialQuery};

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
        use crate::index::spatial::{Point, SpatialQuery};

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
        use crate::index::spatial::Point;

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
        use crate::index::spatial::Point;

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

    // ==================== Durability / Reopen Round-Trip Tests ====================

    use crate::dictionary::Term;
    use crate::store::{GraphName, GraphTarget, QuadResult};
    use std::collections::HashSet;

    /// Unique, isolated data directory for a durability test.
    fn unique_dir(prefix: &str) -> std::path::PathBuf {
        env::temp_dir().join(format!("{prefix}_{}", uuid::Uuid::new_v4()))
    }

    /// open -> insert mixed terms -> sync -> drop -> reopen returns exactly them.
    #[test]
    fn test_persistence_round_trip_small_mixed_terms() {
        let dir = unique_dir("oxirs_tdb_rt_small");

        let s1 = Term::iri("http://example.org/alice");
        let p1 = Term::iri("http://example.org/name");
        let o1 = Term::literal("Alice");
        let p2 = Term::iri("http://example.org/label");
        let o2 = Term::literal_with_lang("Alice", "en");
        let p3 = Term::iri("http://example.org/age");
        let o3 = Term::literal_with_datatype("42", "http://www.w3.org/2001/XMLSchema#integer");
        let sb = Term::blank_node("b0");
        let pb = Term::iri("http://example.org/knows");
        let ob = Term::iri("http://example.org/bob");

        {
            let mut store = TdbStore::open(&dir).unwrap();
            store.insert_triple(&s1, &p1, &o1).unwrap();
            store.insert_triple(&s1, &p2, &o2).unwrap();
            store.insert_triple(&s1, &p3, &o3).unwrap();
            store.insert_triple(&sb, &pb, &ob).unwrap();
            assert_eq!(store.count(), 4);
            store.sync().unwrap();
        } // dropped

        // Reopen: everything must come back, with term variants intact.
        let store = TdbStore::open(&dir).unwrap();
        assert_eq!(store.count(), 4, "triple count must survive reopen");

        // Literal-with-language must not have been collapsed to a plain literal
        // or an IRI.
        let lang_hits = store
            .query_triples(Some(&s1), Some(&p2), Some(&o2))
            .unwrap();
        assert_eq!(lang_hits.len(), 1);
        assert_eq!(lang_hits[0].2, o2);

        // Typed literal preserved.
        let typed_hits = store
            .query_triples(Some(&s1), Some(&p3), Some(&o3))
            .unwrap();
        assert_eq!(typed_hits.len(), 1);
        assert_eq!(typed_hits[0].2, o3);

        // Blank-node subject preserved.
        let blank_hits = store.query_triples(Some(&sb), None, None).unwrap();
        assert_eq!(blank_hits.len(), 1);
        assert_eq!(blank_hits[0].0, sb);

        // A never-inserted pattern returns nothing.
        let miss = store
            .query_triples(Some(&Term::iri("http://example.org/nobody")), None, None)
            .unwrap();
        assert!(miss.is_empty());

        drop(store);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Reopen after deletes: only the surviving triples come back.
    #[test]
    fn test_persistence_round_trip_after_deletes() {
        let dir = unique_dir("oxirs_tdb_rt_del");
        {
            let mut store = TdbStore::open(&dir).unwrap();
            for i in 0..5 {
                store
                    .insert(
                        &format!("http://example.org/s{i}"),
                        "http://example.org/p",
                        &format!("http://example.org/o{i}"),
                    )
                    .unwrap();
            }
            assert_eq!(store.count(), 5);

            assert!(store
                .delete(
                    "http://example.org/s1",
                    "http://example.org/p",
                    "http://example.org/o1",
                )
                .unwrap());
            assert!(store
                .delete(
                    "http://example.org/s3",
                    "http://example.org/p",
                    "http://example.org/o3",
                )
                .unwrap());
            assert_eq!(store.count(), 3);
            store.sync().unwrap();
        }

        let store = TdbStore::open(&dir).unwrap();
        assert_eq!(store.count(), 3, "deletions must persist across reopen");
        assert!(store
            .contains(
                "http://example.org/s0",
                "http://example.org/p",
                "http://example.org/o0",
            )
            .unwrap());
        assert!(!store
            .contains(
                "http://example.org/s1",
                "http://example.org/p",
                "http://example.org/o1",
            )
            .unwrap_or(true));

        drop(store);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 10k triples survive a reopen with an accurate count and spot-checks.
    #[test]
    fn test_persistence_round_trip_10k() {
        let dir = unique_dir("oxirs_tdb_rt_10k");
        let n = 10_000;
        {
            // A pool large enough to keep the whole working set resident so the
            // load stays in memory until the single sync().
            let config = TdbConfig::new(&dir).with_buffer_pool_size(8192);
            let mut store = TdbStore::open_with_config(config).unwrap();
            for i in 0..n {
                store
                    .insert(
                        &format!("http://example.org/s{i}"),
                        "http://example.org/p",
                        &format!("http://example.org/o{i}"),
                    )
                    .unwrap();
            }
            assert_eq!(store.count(), n);
            store.sync().unwrap();
        }

        let config = TdbConfig::new(&dir).with_buffer_pool_size(8192);
        let store = TdbStore::open_with_config(config).unwrap();
        assert_eq!(store.count(), n, "10k triples must survive reopen");
        // Spot-check first, middle, last.
        for i in [0usize, n / 2, n - 1] {
            assert!(
                store
                    .contains(
                        &format!("http://example.org/s{i}"),
                        "http://example.org/p",
                        &format!("http://example.org/o{i}"),
                    )
                    .unwrap(),
                "triple {i} missing after reopen"
            );
        }

        drop(store);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Dropping the store without an explicit sync still persists (Drop syncs).
    #[test]
    fn test_drop_persists_without_explicit_sync() {
        let dir = unique_dir("oxirs_tdb_drop_sync");
        {
            let mut store = TdbStore::open(&dir).unwrap();
            for i in 0..4 {
                store
                    .insert_triple(
                        &Term::iri(format!("http://example.org/s{i}")),
                        &Term::iri("http://example.org/p"),
                        &Term::iri(format!("http://example.org/o{i}")),
                    )
                    .unwrap();
            }
            // No explicit sync(): rely on Drop.
        }

        let store = TdbStore::open(&dir).unwrap();
        assert_eq!(store.count(), 4, "Drop must persist pending state");

        drop(store);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Crash simulation (F3, WAL default-on): committed writes made after the
    /// last checkpoint survive a crash (drop with sync-on-drop disabled) via WAL
    /// replay on reopen. This is the new durability contract — the pre-F3
    /// behaviour lost everything written since the last `sync()`.
    #[test]
    fn test_crash_without_sync_replays_committed_writes() {
        let dir = unique_dir("oxirs_tdb_crash");

        // Phase 1: insert 5 and sync durably (checkpoint; WAL truncated).
        {
            let mut store = TdbStore::open(&dir).unwrap();
            for i in 0..5 {
                store
                    .insert(
                        &format!("http://example.org/s{i}"),
                        "http://example.org/p",
                        &format!("http://example.org/o{i}"),
                    )
                    .unwrap();
            }
            store.sync().unwrap();
        }

        // Phase 2: add 3 more (committed to the WAL) but simulate a crash: drop
        // WITHOUT a sync, so no checkpoint is written and the WAL keeps the 3
        // committed operations.
        {
            let mut store = TdbStore::open(&dir).unwrap();
            assert_eq!(store.count(), 5);
            for i in 5..8 {
                store
                    .insert(
                        &format!("http://example.org/s{i}"),
                        "http://example.org/p",
                        &format!("http://example.org/o{i}"),
                    )
                    .unwrap();
            }
            assert_eq!(store.count(), 8);
            // Simulate power loss before the next checkpoint.
            store.set_sync_on_drop(false);
        }

        // Reopen: all 8 committed triples survive — the 5 from the checkpoint
        // plus the 3 replayed from the WAL.
        let store = TdbStore::open(&dir).unwrap();
        assert_eq!(
            store.count(),
            8,
            "committed writes must survive a crash via WAL replay"
        );
        assert!(store
            .contains(
                "http://example.org/s0",
                "http://example.org/p",
                "http://example.org/o0",
            )
            .unwrap());
        assert!(
            store
                .contains(
                    "http://example.org/s7",
                    "http://example.org/p",
                    "http://example.org/o7",
                )
                .unwrap(),
            "post-checkpoint committed write must be replayed from the WAL"
        );

        drop(store);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// F3: committed data inserted without any `sync()` survives a reopen purely
    /// via WAL replay (no checkpoint ever taken for this data), across triples
    /// and named-graph quads.
    #[test]
    fn test_wal_replay_recovers_uncheckpointed_commits() {
        let dir = unique_dir("oxirs_tdb_wal_replay");

        let g = Term::iri("http://example.org/g");
        let s = Term::iri("http://example.org/s");
        let p = Term::iri("http://example.org/p");
        let o = Term::literal_with_lang("hello", "en");

        // Insert committed triples + a named-graph quad, then simulate a crash
        // (drop without sync). Nothing is checkpointed.
        {
            let mut store = TdbStore::open(&dir).unwrap();
            for i in 0..20 {
                store
                    .insert_triple(
                        &Term::iri(format!("http://example.org/s{i}")),
                        &p,
                        &Term::iri(format!("http://example.org/o{i}")),
                    )
                    .unwrap();
            }
            store.insert_quad(Some(&g), &s, &p, &o).unwrap();
            assert_eq!(store.count(), 20);
            assert_eq!(store.quad_count(), 1);
            store.set_sync_on_drop(false); // crash before any checkpoint
        }

        // Reopen: everything is recovered from the WAL, with term variants intact.
        let store = TdbStore::open(&dir).unwrap();
        assert_eq!(store.count(), 20, "triples must be recovered from the WAL");
        assert_eq!(store.quad_count(), 1, "quad must be recovered from the WAL");
        assert!(store.contains_quad(Some(&g), &s, &p, &o).unwrap());
        // Language-tagged object preserved through the logical redo record.
        let hits = store.query_triples(None, None, Some(&o)).unwrap();
        assert!(
            hits.is_empty(),
            "the lang literal is only in the named graph"
        );
        assert!(store
            .contains(
                "http://example.org/s19",
                "http://example.org/p",
                "http://example.org/o19",
            )
            .unwrap());

        drop(store);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// F3: a torn/uncommitted transaction — `DataOp` records with no matching
    /// `Commit` — must NOT be replayed, while a committed transaction in the same
    /// WAL is. Proves the redo path keys off the `Commit` record, not merely the
    /// presence of a `DataOp`.
    #[test]
    fn test_wal_torn_uncommitted_writes_are_not_replayed() {
        use crate::store::store_wal::{encode_store_op, StoreOp};
        use crate::transaction::wal::{LogRecord, TxnId, WriteAheadLog};

        let dir = unique_dir("oxirs_tdb_wal_torn");

        let committed = Term::iri("http://example.org/committed");
        let torn = Term::iri("http://example.org/torn");
        let p = Term::iri("http://example.org/p");
        let o = Term::iri("http://example.org/o");

        // Phase A: a store commits one triple, then "crashes" (drop, no sync), so
        // the WAL holds a fully committed Begin/DataOp/Commit for it.
        {
            let mut store = TdbStore::open(&dir).unwrap();
            store.insert_triple(&committed, &p, &o).unwrap();
            store.set_sync_on_drop(false);
        }

        // Phase B: append a torn transaction directly to the WAL — Begin + a
        // *valid* DataOp that would insert `torn`, but NO Commit record.
        {
            let wal = WriteAheadLog::open(&dir).unwrap();
            let txn_id = TxnId::new(9_999);
            wal.append(LogRecord::Begin { txn_id }).unwrap();
            let payload = encode_store_op(&StoreOp::InsertTriple {
                subject: torn.clone(),
                predicate: p.clone(),
                object: o.clone(),
            })
            .unwrap();
            wal.append(LogRecord::DataOp { txn_id, payload }).unwrap();
            // Deliberately NO Commit: this transaction is torn.
            wal.flush().unwrap();
        }

        // Reopen: the committed triple is replayed; the torn one is skipped.
        let store = TdbStore::open(&dir).unwrap();
        assert_eq!(
            store.count(),
            1,
            "only the committed transaction is replayed"
        );
        assert!(
            store
                .contains(
                    "http://example.org/committed",
                    "http://example.org/p",
                    "http://example.org/o",
                )
                .unwrap(),
            "committed write must be replayed"
        );
        assert!(
            !store
                .contains(
                    "http://example.org/torn",
                    "http://example.org/p",
                    "http://example.org/o",
                )
                .unwrap_or(false),
            "torn (uncommitted) write must NOT be replayed"
        );

        drop(store);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// F3: with WAL disabled, durability is checkpoint-only — a crash before
    /// `sync()` loses the un-checkpointed writes (the pre-F3 contract), proving
    /// the `enable_wal` flag actually gates the recovery path.
    #[test]
    fn test_wal_disabled_is_checkpoint_only() {
        let dir = unique_dir("oxirs_tdb_wal_disabled");

        let config = TdbConfig::new(&dir).with_wal(false);
        {
            let mut store = TdbStore::open_with_config(config.clone()).unwrap();
            for i in 0..4 {
                store
                    .insert(
                        &format!("http://example.org/s{i}"),
                        "http://example.org/p",
                        &format!("http://example.org/o{i}"),
                    )
                    .unwrap();
            }
            assert_eq!(store.count(), 4);
            store.set_sync_on_drop(false); // crash before checkpoint
        }

        // No WAL and no checkpoint -> the writes are gone on reopen.
        let store = TdbStore::open_with_config(config).unwrap();
        assert_eq!(
            store.count(),
            0,
            "with WAL disabled, un-checkpointed writes must not survive"
        );

        drop(store);
        std::fs::remove_dir_all(&dir).ok();
    }

    // ==================== Named-Graph (Quad) Tests (F4) ====================

    /// Build the [`QuadResult`] a scan is expected to return for a quad.
    fn expected_quad(graph: Option<&Term>, s: &Term, p: &Term, o: &Term) -> QuadResult {
        QuadResult {
            graph: match graph {
                None => GraphName::DefaultGraph,
                Some(g) => GraphName::Named(g.clone()),
            },
            subject: s.clone(),
            predicate: p.clone(),
            object: o.clone(),
        }
    }

    /// open -> insert quads across the default graph and multiple named graphs
    /// (mixed term kinds) -> sync -> drop -> reopen -> a full scan returns
    /// exactly the inserted quads.
    #[test]
    fn test_quad_reopen_round_trip_multigraph() {
        let dir = unique_dir("oxirs_tdb_quad_rt_multi");

        let g1 = Term::iri("http://example.org/graphs/g1");
        let g2 = Term::iri("http://example.org/graphs/g2");

        let alice = Term::iri("http://example.org/alice");
        let bob = Term::iri("http://example.org/bob");
        let carol = Term::blank_node("carol");
        let name = Term::iri("http://example.org/name");
        let knows = Term::iri("http://example.org/knows");
        let age = Term::iri("http://example.org/age");
        let alice_lit = Term::literal("Alice");
        let bob_lit = Term::literal_with_lang("Bob", "en");
        let age_lit = Term::literal_with_datatype("30", "http://www.w3.org/2001/XMLSchema#integer");
        let carol_lit = Term::literal("Carol");

        // (graph, s, p, o): default graph + two named graphs, mixed terms.
        let quads: Vec<(Option<&Term>, &Term, &Term, &Term)> = vec![
            (None, &alice, &name, &alice_lit),
            (None, &alice, &knows, &bob),
            (Some(&g1), &bob, &name, &bob_lit),
            (Some(&g1), &bob, &age, &age_lit),
            (Some(&g2), &carol, &name, &carol_lit),
            (Some(&g2), &carol, &knows, &alice),
        ];

        {
            let mut store = TdbStore::open(&dir).unwrap();
            for (g, s, p, o) in &quads {
                assert!(store.insert_quad(*g, s, p, o).unwrap());
            }
            // Duplicate insert is a no-op (returns false, count unchanged).
            assert!(!store.insert_quad(None, &alice, &name, &alice_lit).unwrap());

            assert_eq!(store.count(), 2, "two default-graph triples");
            assert_eq!(store.quad_count(), 4, "four named-graph quads");
            assert_eq!(store.dataset_len(), 6);
            store.sync().unwrap();
        }

        let expected: HashSet<QuadResult> = quads
            .iter()
            .map(|(g, s, p, o)| expected_quad(*g, s, p, o))
            .collect();

        let store = TdbStore::open(&dir).unwrap();
        assert_eq!(store.count(), 2, "default-graph count survives reopen");
        assert_eq!(store.quad_count(), 4, "named-graph count survives reopen");

        // Full dataset scan returns exactly the inserted quads.
        let all: HashSet<QuadResult> = store
            .scan_quads(GraphTarget::AnyGraph, None, None, None)
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(
            all, expected,
            "full scan must return exactly the inserted quads"
        );

        // Default-graph-only scan.
        let default_only: HashSet<QuadResult> = store
            .scan_quads(GraphTarget::DefaultGraph, None, None, None)
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(default_only.len(), 2);
        assert!(default_only
            .iter()
            .all(|q| q.graph == GraphName::DefaultGraph));

        // Named-graph-only scan (g1).
        let g1_only: HashSet<QuadResult> = store
            .scan_quads(GraphTarget::Named(&g1), None, None, None)
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(g1_only.len(), 2);
        assert!(g1_only
            .iter()
            .all(|q| q.graph == GraphName::Named(g1.clone())));

        // contains_quad across graphs.
        assert!(store
            .contains_quad(None, &alice, &name, &alice_lit)
            .unwrap());
        assert!(store
            .contains_quad(Some(&g1), &bob, &age, &age_lit)
            .unwrap());
        // Same s/p/o is NOT in a different graph.
        assert!(!store
            .contains_quad(Some(&g2), &bob, &age, &age_lit)
            .unwrap());

        drop(store);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Pattern scans on every bound-column combination for a named graph.
    #[test]
    fn test_quad_pattern_scans_each_bound_column() {
        let dir = unique_dir("oxirs_tdb_quad_patterns");
        let g = Term::iri("http://example.org/g");

        // A small, fully-enumerated dataset in one named graph.
        let s1 = Term::iri("http://example.org/s1");
        let s2 = Term::iri("http://example.org/s2");
        let p1 = Term::iri("http://example.org/p1");
        let p2 = Term::iri("http://example.org/p2");
        let o1 = Term::iri("http://example.org/o1");
        let o2 = Term::iri("http://example.org/o2");

        let data: Vec<(&Term, &Term, &Term)> = vec![
            (&s1, &p1, &o1),
            (&s1, &p1, &o2),
            (&s1, &p2, &o1),
            (&s2, &p1, &o1),
            (&s2, &p2, &o2),
        ];

        let mut store = TdbStore::open(&dir).unwrap();
        for (s, p, o) in &data {
            store.insert_quad(Some(&g), s, p, o).unwrap();
        }

        // Reference filter over the known dataset.
        let reference =
            |sf: Option<&Term>, pf: Option<&Term>, of: Option<&Term>| -> HashSet<QuadResult> {
                let mut set = HashSet::new();
                for &(s, p, o) in &data {
                    if sf.map_or(true, |x| s == x)
                        && pf.map_or(true, |x| p == x)
                        && of.map_or(true, |x| o == x)
                    {
                        set.insert(expected_quad(Some(&g), s, p, o));
                    }
                }
                set
            };

        // Every combination of bound/unbound (subject, predicate, object).
        let subjects = [None, Some(&s1), Some(&s2)];
        let predicates = [None, Some(&p1), Some(&p2)];
        let objects = [None, Some(&o1), Some(&o2)];
        for sf in subjects {
            for pf in predicates {
                for of in objects {
                    let got: HashSet<QuadResult> = store
                        .scan_quads(GraphTarget::Named(&g), sf, pf, of)
                        .unwrap()
                        .into_iter()
                        .collect();
                    let want = reference(sf, pf, of);
                    assert_eq!(got, want, "pattern mismatch for {sf:?} {pf:?} {of:?}");
                }
            }
        }

        drop(store);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Default and named graphs holding the same triple stay isolated.
    #[test]
    fn test_default_vs_named_graph_isolation() {
        let dir = unique_dir("oxirs_tdb_quad_isolation");
        let g1 = Term::iri("http://example.org/g1");
        let g2 = Term::iri("http://example.org/g2");
        let s = Term::iri("http://example.org/s");
        let p = Term::iri("http://example.org/p");
        let o = Term::iri("http://example.org/o");

        let mut store = TdbStore::open(&dir).unwrap();
        store.insert_quad(None, &s, &p, &o).unwrap();
        store.insert_quad(Some(&g1), &s, &p, &o).unwrap();

        // Present in the default graph and g1, absent from g2.
        assert!(store.contains_quad(None, &s, &p, &o).unwrap());
        assert!(store.contains_quad(Some(&g1), &s, &p, &o).unwrap());
        assert!(!store.contains_quad(Some(&g2), &s, &p, &o).unwrap());

        // The default-graph triple is still visible via the triple API.
        assert!(store
            .contains(
                "http://example.org/s",
                "http://example.org/p",
                "http://example.org/o"
            )
            .unwrap());

        // Scans do not leak across graphs.
        let default_scan = store
            .scan_quads(GraphTarget::DefaultGraph, None, None, None)
            .unwrap();
        assert_eq!(default_scan.len(), 1);
        assert_eq!(default_scan[0].graph, GraphName::DefaultGraph);

        let g1_scan = store
            .scan_quads(GraphTarget::Named(&g1), None, None, None)
            .unwrap();
        assert_eq!(g1_scan.len(), 1);
        assert_eq!(g1_scan[0].graph, GraphName::Named(g1.clone()));

        // AnyGraph sees both copies.
        let any = store
            .scan_quads(GraphTarget::AnyGraph, None, None, None)
            .unwrap();
        assert_eq!(any.len(), 2);

        // Deleting from the default graph leaves the named-graph copy intact.
        assert!(store.delete_quad(None, &s, &p, &o).unwrap());
        assert!(!store.contains_quad(None, &s, &p, &o).unwrap());
        assert!(store.contains_quad(Some(&g1), &s, &p, &o).unwrap());
        assert_eq!(store.count(), 0);
        assert_eq!(store.quad_count(), 1);

        drop(store);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The streaming quad iterator yields the same set as a materialized scan,
    /// and can be advanced one item at a time (never buffering all results).
    #[test]
    fn test_streaming_quad_iter_matches_scan() {
        let dir = unique_dir("oxirs_tdb_quad_stream");
        let g1 = Term::iri("http://example.org/g1");

        let mut store = TdbStore::open(&dir).unwrap();
        for i in 0..200 {
            let s = Term::iri(format!("http://example.org/s{i}"));
            let p = Term::iri("http://example.org/p");
            let o = Term::iri(format!("http://example.org/o{i}"));
            // Half in the default graph, half in a named graph.
            let graph = if i % 2 == 0 { None } else { Some(&g1) };
            store.insert_quad(graph, &s, &p, &o).unwrap();
        }

        // Lazy: pulling the first item must not require draining the iterator.
        {
            let mut iter = store
                .quad_iter(GraphTarget::AnyGraph, None, None, None)
                .unwrap();
            let first = iter.next();
            assert!(first.is_some(), "streaming iterator must yield lazily");
        }

        // The streamed set equals the materialized scan set.
        let streamed: HashSet<QuadResult> = store
            .quad_iter(GraphTarget::AnyGraph, None, None, None)
            .unwrap()
            .collect::<Result<HashSet<_>, _>>()
            .unwrap();
        let materialized: HashSet<QuadResult> = store
            .scan_quads(GraphTarget::AnyGraph, None, None, None)
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(streamed.len(), 200);
        assert_eq!(streamed, materialized);

        drop(store);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The streaming triple iterator yields the same set as query_triples.
    #[test]
    fn test_stream_triples_matches_query() {
        let dir = unique_dir("oxirs_tdb_triple_stream");
        let p = Term::iri("http://example.org/p");

        let mut store = TdbStore::open(&dir).unwrap();
        for i in 0..150 {
            let s = Term::iri(format!("http://example.org/s{i}"));
            let o = Term::iri(format!("http://example.org/o{i}"));
            store.insert_triple(&s, &p, &o).unwrap();
        }

        // Full scan: streaming vs materialized query.
        let streamed: HashSet<(Term, Term, Term)> = store
            .stream_triples(None, None, None)
            .unwrap()
            .collect::<Result<HashSet<_>, _>>()
            .unwrap();
        let queried: HashSet<(Term, Term, Term)> = store
            .query_triples(None, None, None)
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(streamed.len(), 150);
        assert_eq!(streamed, queried);

        // Bound predicate pattern also matches.
        let streamed_p: HashSet<(Term, Term, Term)> = store
            .stream_triples(None, Some(&p), None)
            .unwrap()
            .collect::<Result<HashSet<_>, _>>()
            .unwrap();
        assert_eq!(streamed_p.len(), 150);

        drop(store);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 10k quads across the default graph and several named graphs survive a
    /// reopen with accurate counts and correct spot-checks.
    #[test]
    fn test_quad_reopen_round_trip_10k() {
        let dir = unique_dir("oxirs_tdb_quad_rt_10k");
        let n = 10_000usize;

        let graphs = [
            Term::iri("http://example.org/graphs/g1"),
            Term::iri("http://example.org/graphs/g2"),
            Term::iri("http://example.org/graphs/g3"),
        ];
        let predicate = Term::iri("http://example.org/p");

        // i % 4 == 0 -> default graph; otherwise named graph (i % 4) - 1.
        let graph_for = |i: usize| -> Option<&Term> {
            match i % 4 {
                0 => None,
                k => Some(&graphs[k - 1]),
            }
        };

        {
            let config = TdbConfig::new(&dir).with_buffer_pool_size(16384);
            let mut store = TdbStore::open_with_config(config).unwrap();
            for i in 0..n {
                let s = Term::iri(format!("http://example.org/s{i}"));
                let o = Term::iri(format!("http://example.org/o{i}"));
                store.insert_quad(graph_for(i), &s, &predicate, &o).unwrap();
            }
            let default_count = (0..n).filter(|i| i % 4 == 0).count();
            assert_eq!(store.count(), default_count);
            assert_eq!(store.quad_count(), n - default_count);
            assert_eq!(store.dataset_len(), n);
            store.sync().unwrap();
        }

        let config = TdbConfig::new(&dir).with_buffer_pool_size(16384);
        let store = TdbStore::open_with_config(config).unwrap();
        assert_eq!(store.dataset_len(), n, "10k quads must survive reopen");

        // Spot-check first/middle/last across default and named graphs.
        for i in [0usize, 1, 2, 3, n / 2, n - 1] {
            let s = Term::iri(format!("http://example.org/s{i}"));
            let o = Term::iri(format!("http://example.org/o{i}"));
            assert!(
                store
                    .contains_quad(graph_for(i), &s, &predicate, &o)
                    .unwrap(),
                "quad {i} missing after reopen"
            );
        }

        drop(store);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A store whose on-disk superblock carries an older format version must be
    /// rejected on open with a clear version-mismatch error, not silently
    /// mis-read (F4 format bump migration path).
    #[test]
    fn test_open_rejects_old_superblock_version() {
        use crate::storage::file_manager::FileManager;
        use crate::storage::superblock::{Superblock, SUPERBLOCK_FORMAT_VERSION};

        let dir = unique_dir("oxirs_tdb_old_version");

        // Create and persist a normal (current-version) store.
        {
            let mut store = TdbStore::open(&dir).unwrap();
            store
                .insert(
                    "http://example.org/s",
                    "http://example.org/p",
                    "http://example.org/o",
                )
                .unwrap();
            store.sync().unwrap();
        }

        // Rewrite page 0 with an older format version, simulating a store
        // created by a previous oxirs-tdb release.
        {
            let data_file = dir.join("data.tdb");
            let fm = FileManager::open(&data_file, false).unwrap();
            let mut sb = Superblock::read(&fm).unwrap().unwrap();
            sb.format_version = SUPERBLOCK_FORMAT_VERSION - 1;
            sb.write(&fm).unwrap();
        }

        // Reopen must fail loudly with a version-mismatch error.
        let reopened = TdbStore::open(&dir);
        assert!(
            reopened.is_err(),
            "opening an old-format store must fail rather than silently mis-read"
        );
        let msg = reopened.err().unwrap().to_string();
        assert!(
            msg.contains("format version"),
            "expected a version-mismatch error, got: {msg}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
