//! Integration tests for sharding functionality

#[cfg(test)]
mod shard_integration_tests {
    use oxirs_cluster::network::{NetworkConfig, NetworkService};
    use oxirs_cluster::shard::{
        ConceptCluster, ConceptSimilarity, DefaultConceptSimilarity, ShardRouter, ShardingStrategy,
    };
    use oxirs_cluster::shard_manager::{ShardManager, ShardManagerConfig};
    use oxirs_cluster::shard_routing::{QueryOptimizationHints, QueryRouter};
    use oxirs_cluster::storage::mock::MockStorageBackend;
    use oxirs_core::model::{NamedNode, Triple};
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_end_to_end_sharding() {
        // Create sharding strategy
        let strategy = ShardingStrategy::Hash { num_shards: 4 };
        let router = Arc::new(ShardRouter::new(strategy.clone()));
        router.init_shards(4, 3).await.unwrap();

        // Create shard manager
        let storage = Arc::new(MockStorageBackend::new());
        let network = Arc::new(NetworkService::new(1, NetworkConfig::default()));
        let config = ShardManagerConfig::default();
        let manager = Arc::new(ShardManager::new(
            1,
            router.clone(),
            config,
            storage,
            network,
        ));

        // Initialize shards
        let nodes = vec![1, 2, 3, 4, 5, 6];
        manager.initialize_shards(&strategy, nodes).await.unwrap();

        // Store some triples
        let triples = vec![
            Triple::new(
                NamedNode::new("http://example.org/alice").unwrap(),
                NamedNode::new("http://example.org/knows").unwrap(),
                NamedNode::new("http://example.org/bob").unwrap(),
            ),
            Triple::new(
                NamedNode::new("http://example.org/bob").unwrap(),
                NamedNode::new("http://example.org/knows").unwrap(),
                NamedNode::new("http://example.org/charlie").unwrap(),
            ),
            Triple::new(
                NamedNode::new("http://example.org/charlie").unwrap(),
                NamedNode::new("http://example.org/age").unwrap(),
                NamedNode::new("25").unwrap(),
            ),
        ];

        for triple in &triples {
            manager.store_triple(triple.clone()).await.unwrap();
        }

        // Query triples
        let results = manager
            .query_triples(None, Some("http://example.org/knows"), None)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_namespace_sharding() {
        let mut namespace_mapping = HashMap::new();
        namespace_mapping.insert("http://example.org/".to_string(), 0);
        namespace_mapping.insert("http://schema.org/".to_string(), 1);
        namespace_mapping.insert("http://xmlns.com/foaf/0.1/".to_string(), 2);

        let strategy = ShardingStrategy::Namespace {
            namespace_mapping: namespace_mapping.clone(),
        };
        let router = ShardRouter::new(strategy);

        // Test routing different namespaces
        let test_cases = vec![
            ("http://example.org/person/123", 0),
            ("http://schema.org/Person", 1),
            ("http://xmlns.com/foaf/0.1/name", 2),
        ];

        for (iri, expected_shard) in test_cases {
            let triple = Triple::new(
                NamedNode::new(iri).unwrap(),
                NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap(),
                NamedNode::new("http://schema.org/Thing").unwrap(),
            );

            let shard_id = router.route_triple(&triple).await.unwrap();
            assert_eq!(
                shard_id, expected_shard,
                "IRI {} should route to shard {}",
                iri, expected_shard
            );
        }
    }

    #[tokio::test]
    async fn test_semantic_sharding() {
        let clusters = vec![
            ConceptCluster {
                cluster_id: 0,
                core_concepts: vec![
                    "http://schema.org/Person".to_string(),
                    "http://schema.org/Organization".to_string(),
                ]
                .into_iter()
                .collect(),
                predicates: vec![
                    "http://schema.org/name".to_string(),
                    "http://schema.org/email".to_string(),
                ]
                .into_iter()
                .collect(),
                namespace_patterns: vec!["http://schema.org/".to_string()],
                weight: 1.0,
            },
            ConceptCluster {
                cluster_id: 1,
                core_concepts: vec![
                    "http://example.org/Document".to_string(),
                    "http://example.org/Article".to_string(),
                ]
                .into_iter()
                .collect(),
                predicates: vec![
                    "http://example.org/title".to_string(),
                    "http://example.org/author".to_string(),
                ]
                .into_iter()
                .collect(),
                namespace_patterns: vec!["http://example.org/doc/".to_string()],
                weight: 1.0,
            },
        ];

        let strategy = ShardingStrategy::Semantic {
            concept_clusters: clusters,
            similarity_threshold: 0.5,
        };

        let router = ShardRouter::new(strategy)
            .with_similarity_calculator(Arc::new(DefaultConceptSimilarity));

        // Test semantic routing
        let person_triple = Triple::new(
            NamedNode::new("http://schema.org/PersonInstance").unwrap(),
            NamedNode::new("http://schema.org/name").unwrap(),
            NamedNode::new("John Doe").unwrap(),
        );

        let doc_triple = Triple::new(
            NamedNode::new("http://example.org/doc/123").unwrap(),
            NamedNode::new("http://example.org/title").unwrap(),
            NamedNode::new("My Document").unwrap(),
        );

        assert_eq!(router.route_triple(&person_triple).await.unwrap(), 0);
        assert_eq!(router.route_triple(&doc_triple).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_query_routing() {
        let strategy = ShardingStrategy::Subject { num_shards: 4 };
        let shard_router = Arc::new(ShardRouter::new(strategy));
        shard_router.init_shards(4, 3).await.unwrap();

        let storage = Arc::new(MockStorageBackend::new());
        let network = Arc::new(NetworkService::new(1, NetworkConfig::default()));
        let shard_manager = Arc::new(ShardManager::new(
            1,
            shard_router.clone(),
            ShardManagerConfig::default(),
            storage,
            network,
        ));

        let query_router = QueryRouter::new(shard_router, shard_manager);

        // Test query planning
        let hints = QueryOptimizationHints {
            parallel_execution: true,
            enable_cache: true,
            limit: Some(100),
            ..Default::default()
        };

        // Specific subject query should route to single shard
        let plan1 = query_router
            .plan_query(Some("http://example.org/alice"), None, None, hints.clone())
            .await
            .unwrap();
        assert_eq!(plan1.shard_targets.len(), 1);

        // Query without subject should route to all shards
        let plan2 = query_router
            .plan_query(None, Some("http://example.org/knows"), None, hints)
            .await
            .unwrap();
        assert_eq!(plan2.shard_targets.len(), 4);
    }

    #[tokio::test]
    async fn test_shard_statistics() {
        let strategy = ShardingStrategy::Hash { num_shards: 3 };
        let router = Arc::new(ShardRouter::new(strategy));
        router.init_shards(3, 2).await.unwrap();

        // Update shard metadata
        for i in 0..3 {
            let mut metadata = router.get_shard_metadata(i).await.unwrap();
            metadata.triple_count = ((i + 1) * 1000) as usize;
            metadata.size_bytes = ((i + 1) * 1_000_000) as u64;
            router.update_shard_metadata(metadata).await.unwrap();
        }

        let stats = router.get_statistics().await;
        assert_eq!(stats.total_shards, 3);
        assert_eq!(stats.active_shards, 3);
        assert_eq!(stats.total_triples, 6000); // 1000 + 2000 + 3000
        assert_eq!(stats.total_size, 6_000_000); // 1M + 2M + 3M

        // Check distribution
        assert_eq!(stats.distribution.len(), 3);
        assert!(stats.distribution[0].load_factor < stats.distribution[2].load_factor);
    }

    #[test]
    fn test_concept_similarity_calculation() {
        let calc = DefaultConceptSimilarity;

        // Identical concepts
        assert_eq!(
            calc.similarity("http://schema.org/Person", "http://schema.org/Person"),
            1.0
        );

        // Similar concepts (same namespace)
        let sim1 = calc.similarity("http://schema.org/Person", "http://schema.org/Place");
        assert!(sim1 > 0.5);

        // Different namespaces
        let sim2 = calc.similarity("http://schema.org/Person", "http://example.org/Person");
        assert!(sim2 < 0.5);

        // Completely different
        let sim3 = calc.similarity("http://schema.org/Person", "http://other.org/Thing");
        assert!(sim3 < 0.3);
    }

    #[tokio::test]
    async fn test_hybrid_sharding_strategy() {
        // Create a hybrid strategy: namespace primary, hash secondary
        let mut namespace_mapping = HashMap::new();
        namespace_mapping.insert("http://important.org/".to_string(), 0);

        let primary = Box::new(ShardingStrategy::Namespace { namespace_mapping });
        let secondary = Box::new(ShardingStrategy::Hash { num_shards: 4 });

        let strategy = ShardingStrategy::Hybrid { primary, secondary };
        let router = ShardRouter::new(strategy);

        // Important namespace should go to shard 0
        let triple1 = Triple::new(
            NamedNode::new("http://important.org/data").unwrap(),
            NamedNode::new("http://example.org/type").unwrap(),
            NamedNode::new("Important").unwrap(),
        );
        assert_eq!(router.route_triple(&triple1).await.unwrap(), 0);

        // Other namespaces should use hash routing
        let triple2 = Triple::new(
            NamedNode::new("http://other.org/data").unwrap(),
            NamedNode::new("http://example.org/type").unwrap(),
            NamedNode::new("Other").unwrap(),
        );
        let shard_id = router.route_triple(&triple2).await.unwrap();
        assert!(shard_id < 4);
    }
}
