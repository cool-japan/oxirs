//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::discovery::NodeMetadata;

/// Enhanced node metadata with region information
impl NodeMetadata {
    /// Add region information to node metadata
    pub fn with_region_info(mut self, region_id: String, availability_zone_id: String) -> Self {
        self.custom.insert("region_id".to_string(), region_id);
        self.custom
            .insert("availability_zone_id".to_string(), availability_zone_id);
        self.features.insert("multi-region".to_string());
        self
    }
    /// Get region ID from metadata
    pub fn region_id(&self) -> Option<&String> {
        self.custom.get("region_id")
    }
    /// Get availability zone ID from metadata
    pub fn availability_zone_id(&self) -> Option<&String> {
        self.custom.get("availability_zone_id")
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        region_manager::{
            AvailabilityZone, ConflictResolutionStrategy, ConsensusStrategy, CrossRegionStrategy,
            GeoCoordinates, IntraRegionStrategy, RegionConfig,
        },
        MultiRegionReplicationStrategy, Region, RegionManager,
    };
    fn create_test_region(id: &str, name: &str) -> Region {
        Region {
            id: id.to_string(),
            name: name.to_string(),
            coordinates: Some(GeoCoordinates {
                latitude: 40.7128,
                longitude: -74.0060,
            }),
            availability_zones: vec![
                AvailabilityZone {
                    id: format!("{id}a"),
                    name: format!("{name} AZ A"),
                    region_id: id.to_string(),
                },
                AvailabilityZone {
                    id: format!("{id}b"),
                    name: format!("{name} AZ B"),
                    region_id: id.to_string(),
                },
            ],
            config: RegionConfig::default(),
        }
    }
    #[tokio::test]
    async fn test_region_manager_initialization() {
        let manager = RegionManager::new(
            "us-east-1".to_string(),
            "us-east-1a".to_string(),
            ConsensusStrategy::GlobalRaft,
            MultiRegionReplicationStrategy {
                intra_region: IntraRegionStrategy::Synchronous { min_replicas: 2 },
                cross_region: CrossRegionStrategy::AsyncAll,
                conflict_resolution: ConflictResolutionStrategy::LastWriterWins,
            },
        );
        let regions = vec![
            create_test_region("us-east-1", "US East 1"),
            create_test_region("eu-west-1", "EU West 1"),
        ];
        assert!(manager.initialize(regions).await.is_ok());
        let topology = manager.get_topology().await;
        assert_eq!(topology.regions.len(), 2);
        assert!(topology.regions.contains_key("us-east-1"));
        assert!(topology.regions.contains_key("eu-west-1"));
    }
    #[tokio::test]
    async fn test_node_registration() {
        let manager = RegionManager::new(
            "us-east-1".to_string(),
            "us-east-1a".to_string(),
            ConsensusStrategy::GlobalRaft,
            MultiRegionReplicationStrategy {
                intra_region: IntraRegionStrategy::Synchronous { min_replicas: 2 },
                cross_region: CrossRegionStrategy::AsyncAll,
                conflict_resolution: ConflictResolutionStrategy::LastWriterWins,
            },
        );
        let regions = vec![create_test_region("us-east-1", "US East 1")];
        manager.initialize(regions).await.unwrap();
        assert!(manager
            .register_node(
                1,
                "us-east-1".to_string(),
                "us-east-1a".to_string(),
                None,
                None
            )
            .await
            .is_ok());
        let nodes_in_region = manager.get_nodes_in_region("us-east-1").await;
        assert_eq!(nodes_in_region.len(), 1);
        assert_eq!(nodes_in_region[0], 1);
        assert!(manager
            .register_node(
                2,
                "unknown-region".to_string(),
                "unknown-az".to_string(),
                None,
                None
            )
            .await
            .is_err());
    }
    #[tokio::test]
    async fn test_leader_candidates() {
        let manager = RegionManager::new(
            "us-east-1".to_string(),
            "us-east-1a".to_string(),
            ConsensusStrategy::GlobalRaft,
            MultiRegionReplicationStrategy {
                intra_region: IntraRegionStrategy::Synchronous { min_replicas: 2 },
                cross_region: CrossRegionStrategy::AsyncAll,
                conflict_resolution: ConflictResolutionStrategy::LastWriterWins,
            },
        );
        let regions = vec![
            create_test_region("us-east-1", "US East 1"),
            create_test_region("eu-west-1", "EU West 1"),
        ];
        manager.initialize(regions).await.unwrap();
        manager
            .register_node(
                1,
                "us-east-1".to_string(),
                "us-east-1a".to_string(),
                None,
                None,
            )
            .await
            .unwrap();
        manager
            .register_node(
                2,
                "us-east-1".to_string(),
                "us-east-1b".to_string(),
                None,
                None,
            )
            .await
            .unwrap();
        manager
            .register_node(
                3,
                "eu-west-1".to_string(),
                "eu-west-1a".to_string(),
                None,
                None,
            )
            .await
            .unwrap();
        let candidates = manager.get_leader_candidates("us-east-1").await;
        assert_eq!(candidates.len(), 2);
        assert!(candidates.contains(&1));
        assert!(candidates.contains(&2));
    }
    #[tokio::test]
    async fn test_replication_targets() {
        let manager = RegionManager::new(
            "us-east-1".to_string(),
            "us-east-1a".to_string(),
            ConsensusStrategy::GlobalRaft,
            MultiRegionReplicationStrategy {
                intra_region: IntraRegionStrategy::Synchronous { min_replicas: 2 },
                cross_region: CrossRegionStrategy::AsyncAll,
                conflict_resolution: ConflictResolutionStrategy::LastWriterWins,
            },
        );
        let regions = vec![
            create_test_region("us-east-1", "US East 1"),
            create_test_region("eu-west-1", "EU West 1"),
            create_test_region("ap-south-1", "AP South 1"),
        ];
        manager.initialize(regions).await.unwrap();
        let targets = manager
            .calculate_replication_targets("us-east-1")
            .await
            .unwrap();
        assert_eq!(targets.len(), 2);
        assert!(targets.contains(&"eu-west-1".to_string()));
        assert!(targets.contains(&"ap-south-1".to_string()));
    }
    #[test]
    fn test_distance_calculation() {
        let manager = RegionManager::new(
            "us-east-1".to_string(),
            "us-east-1a".to_string(),
            ConsensusStrategy::GlobalRaft,
            MultiRegionReplicationStrategy {
                intra_region: IntraRegionStrategy::Synchronous { min_replicas: 2 },
                cross_region: CrossRegionStrategy::AsyncAll,
                conflict_resolution: ConflictResolutionStrategy::LastWriterWins,
            },
        );
        let coord_ny = GeoCoordinates {
            latitude: 40.7128,
            longitude: -74.0060,
        };
        let coord_london = GeoCoordinates {
            latitude: 51.5074,
            longitude: -0.1278,
        };
        let distance = manager.calculate_distance(&coord_ny, &coord_london);
        assert!((distance - 5585.0).abs() < 100.0);
    }
    #[test]
    fn test_node_metadata_with_region_info() {
        let metadata = NodeMetadata::default()
            .with_region_info("us-east-1".to_string(), "us-east-1a".to_string());
        assert_eq!(metadata.region_id(), Some(&"us-east-1".to_string()));
        assert_eq!(
            metadata.availability_zone_id(),
            Some(&"us-east-1a".to_string())
        );
        assert!(metadata.features.contains("multi-region"));
    }
    #[test]
    fn test_region_config_default() {
        let config = RegionConfig::default();
        assert_eq!(config.local_replication_factor, 3);
        assert_eq!(config.cross_region_replication_factor, 1);
        assert_eq!(config.max_regional_latency_ms, 100);
        assert!(config.prefer_local_leader);
        assert!(config.enable_cross_region_backup);
    }

    fn two_region_manager(cross_region: CrossRegionStrategy) -> RegionManager {
        RegionManager::new(
            "us-east-1".to_string(),
            "us-east-1a".to_string(),
            ConsensusStrategy::GlobalRaft,
            MultiRegionReplicationStrategy {
                intra_region: IntraRegionStrategy::Synchronous { min_replicas: 1 },
                cross_region,
                conflict_resolution: ConflictResolutionStrategy::LastWriterWins,
            },
        )
    }

    async fn with_regions(manager: &RegionManager, nodes: &[(u64, &str)]) {
        manager
            .initialize(vec![
                create_test_region("us-east-1", "US East 1"),
                create_test_region("eu-west-1", "EU West 1"),
            ])
            .await
            .expect("initialize");
        for &(node_id, region) in nodes {
            manager
                .register_node(
                    node_id,
                    region.to_string(),
                    format!("{region}a"),
                    None,
                    None,
                )
                .await
                .expect("register node");
        }
    }

    /// Node addresses used to be made up as `127.0.0.1:{8080 + node_id}`.
    #[tokio::test]
    async fn test_unregistered_node_has_no_address() {
        let manager = two_region_manager(CrossRegionStrategy::AsyncAll);
        assert_eq!(manager.get_node_address(5).await.expect("lookup"), None);
    }

    /// Latency monitoring held the topology read lock while the measurement
    /// took the write lock, so every successful measurement hung it.
    #[tokio::test]
    async fn test_latency_monitoring_finishes_and_records_real_samples() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                drop(stream);
            }
        });
        let remote = u64::from(addr.port()) - 8080;
        let manager = two_region_manager(CrossRegionStrategy::AsyncAll);
        with_regions(&manager, &[(1, "us-east-1"), (remote, "eu-west-1")]).await;
        manager.register_node_address(remote, addr).await;

        tokio::time::timeout(
            std::time::Duration::from_secs(20),
            manager.monitor_latencies(),
        )
        .await
        .expect("latency monitoring hung")
        .expect("latency monitoring");
        let metrics = manager.performance_metrics.read().await;
        let measured = metrics
            .inter_region_latencies
            .get(&("us-east-1".to_string(), "eu-west-1".to_string()))
            .expect("the reachable region was measured");
        assert_eq!(measured.sample_count, 1);
        // The other direction cannot be measured from this node, and no
        // estimate is recorded as if it had been.
        assert!(!metrics
            .inter_region_latencies
            .contains_key(&("eu-west-1".to_string(), "us-east-1".to_string())));
    }

    /// Cross-region replication reported success although nothing reached
    /// the other region.
    #[tokio::test]
    async fn test_cross_region_replication_reports_failure() {
        for strategy in [
            CrossRegionStrategy::AsyncAll,
            CrossRegionStrategy::EventualConsistency {
                reconciliation_interval_ms: 1000,
            },
        ] {
            let manager = two_region_manager(strategy.clone());
            with_regions(&manager, &[(1, "us-east-1"), (2, "eu-west-1")]).await;
            let replication = MultiRegionReplicationStrategy {
                intra_region: IntraRegionStrategy::Synchronous { min_replicas: 1 },
                cross_region: strategy.clone(),
                conflict_resolution: ConflictResolutionStrategy::LastWriterWins,
            };
            let outcome = manager
                .replicate_cross_region(b"data", "us-east-1", &replication)
                .await;
            assert!(
                outcome.is_err(),
                "{strategy:?} reported replication that did not happen"
            );
        }
    }
}
