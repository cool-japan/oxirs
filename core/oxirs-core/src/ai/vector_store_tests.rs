//! Tests for the vector_store module hierarchy.

#[cfg(test)]
mod tests {
    use crate::ai::vector_store::{
        compute_similarities_batch, compute_similarity, IVFIndex, InMemoryVectorStore, IndexType,
        LSHIndex, PQIndexLocal, SimilarityMetric, VectorData, VectorIndex, VectorQuery,
        VectorStore, VectorStoreConfig,
    };
    use dashmap::DashMap;

    #[tokio::test]
    async fn test_vector_store_creation() {
        let config = VectorStoreConfig::default();
        let store = InMemoryVectorStore::new(config);
        assert_eq!(store.size(), 0);
    }

    #[tokio::test]
    async fn test_vector_insertion_and_retrieval() {
        let config = VectorStoreConfig {
            dimension: 3,
            ..Default::default()
        };
        let store = InMemoryVectorStore::new(config);

        let vector = vec![1.0, 2.0, 3.0];
        let metadata = Some(
            [("type".to_string(), "test".to_string())]
                .iter()
                .cloned()
                .collect(),
        );

        store
            .insert("test1".to_string(), vector.clone(), metadata.clone())
            .await
            .expect("operation should succeed");

        let retrieved = store
            .get("test1")
            .await
            .expect("async operation should succeed")
            .expect("operation should succeed");
        assert_eq!(retrieved.vector, vector);
        assert_eq!(retrieved.metadata, metadata);
    }

    #[tokio::test]
    async fn test_similarity_search() {
        let config = VectorStoreConfig {
            dimension: 3,
            ..Default::default()
        };
        let store = InMemoryVectorStore::new(config);

        store
            .insert("vec1".to_string(), vec![1.0, 0.0, 0.0], None)
            .await
            .expect("operation should succeed");
        store
            .insert("vec2".to_string(), vec![0.9, 0.1, 0.0], None)
            .await
            .expect("operation should succeed");
        store
            .insert("vec3".to_string(), vec![0.0, 1.0, 0.0], None)
            .await
            .expect("operation should succeed");

        let query = VectorQuery {
            vector: vec![1.0, 0.0, 0.0],
            k: 2,
            metric: Some(SimilarityMetric::Cosine),
            include_metadata: false,
            filters: None,
            min_similarity: None,
        };

        let results = store
            .search(&query)
            .await
            .expect("async operation should succeed");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "vec1");
    }

    #[test]
    fn test_similarity_metrics() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 4.0, 6.0];

        let cosine = compute_similarity(&a, &b, SimilarityMetric::Cosine)
            .expect("similarity computation should succeed");
        assert!((cosine - 1.0).abs() < 1e-6);

        let dot_product = compute_similarity(&a, &b, SimilarityMetric::DotProduct)
            .expect("similarity computation should succeed");
        assert_eq!(dot_product, 28.0);
    }

    #[tokio::test]
    async fn test_index_building() {
        let config = VectorStoreConfig {
            dimension: 3,
            index_type: IndexType::Flat,
            ..Default::default()
        };
        let store = InMemoryVectorStore::new(config);

        store
            .insert("vec1".to_string(), vec![1.0, 0.0, 0.0], None)
            .await
            .expect("operation should succeed");
        store
            .insert("vec2".to_string(), vec![0.0, 1.0, 0.0], None)
            .await
            .expect("operation should succeed");

        store
            .build_index()
            .await
            .expect("async operation should succeed");

        let stats = store
            .get_statistics()
            .await
            .expect("async operation should succeed");
        assert_eq!(stats.total_vectors, 2);
    }

    #[tokio::test]
    async fn test_hnsw_index_building() {
        let config = VectorStoreConfig {
            dimension: 3,
            index_type: IndexType::HNSW {
                max_connections: 16,
                ef_construction: 100,
                ef_search: 50,
            },
            ..Default::default()
        };
        let store = InMemoryVectorStore::new(config);

        for i in 0..10 {
            let angle = (i as f32) * std::f32::consts::PI / 5.0;
            store
                .insert(format!("vec{i}"), vec![angle.cos(), angle.sin(), 0.0], None)
                .await
                .expect("operation should succeed");
        }

        store
            .build_index()
            .await
            .expect("async operation should succeed");

        let stats = store
            .get_statistics()
            .await
            .expect("async operation should succeed");
        assert_eq!(stats.total_vectors, 10);
        assert!(stats.index_type.contains("HNSW"));
    }

    #[tokio::test]
    async fn test_hnsw_search() {
        let config = VectorStoreConfig {
            dimension: 3,
            index_type: IndexType::HNSW {
                max_connections: 16,
                ef_construction: 100,
                ef_search: 50,
            },
            ..Default::default()
        };
        let store = InMemoryVectorStore::new(config);

        for i in 0..20 {
            let angle = (i as f32) * std::f32::consts::PI * 2.0 / 20.0;
            store
                .insert(format!("vec{i}"), vec![angle.cos(), angle.sin(), 0.0], None)
                .await
                .expect("operation should succeed");
        }

        store
            .build_index()
            .await
            .expect("async operation should succeed");

        let query = VectorQuery {
            vector: vec![1.0, 0.0, 0.0],
            k: 3,
            metric: Some(SimilarityMetric::Cosine),
            include_metadata: false,
            filters: None,
            min_similarity: None,
        };

        let results = store
            .search(&query)
            .await
            .expect("async operation should succeed");

        assert!(!results.is_empty());
        assert_eq!(results[0].0, "vec0");
        assert!((results[0].1 - 1.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_hnsw_large_dataset() {
        let config = VectorStoreConfig {
            dimension: 10,
            index_type: IndexType::HNSW {
                max_connections: 16,
                ef_construction: 100,
                ef_search: 50,
            },
            ..Default::default()
        };
        let store = InMemoryVectorStore::new(config);

        for i in 0..100 {
            let vec: Vec<f32> = (0..10)
                .map(|j| ((i * 7 + j * 13) % 100) as f32 / 100.0)
                .collect();
            store
                .insert(format!("vec{i}"), vec, None)
                .await
                .expect("async operation should succeed");
        }

        store
            .build_index()
            .await
            .expect("async operation should succeed");

        let query_vec = vec![0.5f32; 10];
        let query = VectorQuery {
            vector: query_vec,
            k: 10,
            metric: Some(SimilarityMetric::Cosine),
            include_metadata: false,
            filters: None,
            min_similarity: None,
        };

        let results = store
            .search(&query)
            .await
            .expect("async operation should succeed");

        assert!(!results.is_empty());
        assert!(results.len() <= 10);

        for i in 1..results.len() {
            assert!(results[i - 1].1 >= results[i].1);
        }
    }

    #[test]
    fn test_batch_similarity_computation() {
        let query = vec![1.0, 0.0, 0.0];
        let candidates: Vec<&[f32]> =
            vec![&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.707, 0.707, 0.0]];

        let similarities =
            compute_similarities_batch(&query, &candidates, SimilarityMetric::Cosine)
                .expect("batch similarity computation should succeed");

        assert_eq!(similarities.len(), 3);
        assert!((similarities[0] - 1.0).abs() < 0.01);
        assert!(similarities[1].abs() < 0.01);
        assert!((similarities[2] - 0.707).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_ivf_index_build_and_search() {
        let vectors: DashMap<String, VectorData> = DashMap::new();
        for i in 0..20usize {
            let vec = vec![i as f32, (i * 2) as f32, (i * 3) as f32];
            vectors.insert(
                format!("v{}", i),
                VectorData {
                    id: format!("v{}", i),
                    vector: vec,
                    metadata: None,
                    timestamp: std::time::SystemTime::now(),
                },
            );
        }

        let mut index = IVFIndex::new(4, 2);
        index
            .build(&vectors)
            .await
            .expect("IVF build should succeed");
        assert_eq!(index.get_stats().index_type, "IVF");
        assert_eq!(index.get_stats().num_vectors, 20);

        let query = vec![5.0, 10.0, 15.0];
        let results = index
            .search(&query, 3, SimilarityMetric::Cosine)
            .await
            .expect("IVF search should succeed");
        assert!(!results.is_empty(), "IVF should return results");
    }

    #[tokio::test]
    async fn test_lsh_index_build_and_search() {
        let vectors: DashMap<String, VectorData> = DashMap::new();
        for i in 0..12usize {
            let vec = vec![i as f32, (20 - i) as f32, 1.0];
            vectors.insert(
                format!("lsh_{}", i),
                VectorData {
                    id: format!("lsh_{}", i),
                    vector: vec,
                    metadata: None,
                    timestamp: std::time::SystemTime::now(),
                },
            );
        }

        let mut index = LSHIndex::new(4, 8);
        index
            .build(&vectors)
            .await
            .expect("LSH build should succeed");
        assert_eq!(index.get_stats().index_type, "LSH");
        assert_eq!(index.get_stats().num_vectors, 12);

        let query = vec![3.0, 17.0, 1.0];
        let results = index
            .search(&query, 3, SimilarityMetric::Cosine)
            .await
            .expect("LSH search should succeed");
        let _ = results;
    }

    #[tokio::test]
    async fn test_pq_index_build_and_search() {
        let vectors: DashMap<String, VectorData> = DashMap::new();
        for i in 0..16usize {
            let vec = vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32];
            vectors.insert(
                format!("pq_{}", i),
                VectorData {
                    id: format!("pq_{}", i),
                    vector: vec,
                    metadata: None,
                    timestamp: std::time::SystemTime::now(),
                },
            );
        }

        let mut index = PQIndexLocal::new(2, 2);
        index
            .build(&vectors)
            .await
            .expect("PQ build should succeed");
        assert_eq!(index.get_stats().index_type, "PQ");
        assert_eq!(index.get_stats().num_vectors, 16);

        let query = vec![7.0, 8.0, 9.0, 10.0];
        let results = index
            .search(&query, 3, SimilarityMetric::Cosine)
            .await
            .expect("PQ search should succeed");
        assert!(!results.is_empty(), "PQ should return results");
        assert!(results.len() <= 3);
    }

    #[tokio::test]
    async fn test_build_index_ivf_variant() {
        let config = VectorStoreConfig {
            dimension: 3,
            index_type: IndexType::IVF {
                num_clusters: 2,
                num_probes: 1,
            },
            ..Default::default()
        };
        let store = InMemoryVectorStore::new(config);

        for i in 0..6usize {
            store
                .insert(format!("v{}", i), vec![i as f32, (i * 2) as f32, 1.0], None)
                .await
                .expect("insert should succeed");
        }
        store
            .build_index()
            .await
            .expect("IVF index build should succeed via store");
    }

    #[tokio::test]
    async fn test_build_index_lsh_variant() {
        let config = VectorStoreConfig {
            dimension: 3,
            index_type: IndexType::LSH {
                num_tables: 2,
                hash_length: 4,
            },
            ..Default::default()
        };
        let store = InMemoryVectorStore::new(config);

        for i in 0..6usize {
            store
                .insert(format!("v{}", i), vec![i as f32, 1.0, (i + 1) as f32], None)
                .await
                .expect("insert should succeed");
        }
        store
            .build_index()
            .await
            .expect("LSH index build should succeed via store");
    }

    #[tokio::test]
    async fn test_build_index_pq_variant() {
        let config = VectorStoreConfig {
            dimension: 4,
            index_type: IndexType::PQ {
                num_subquantizers: 2,
                bits_per_subquantizer: 2,
            },
            ..Default::default()
        };
        let store = InMemoryVectorStore::new(config);

        for i in 0..8usize {
            store
                .insert(
                    format!("v{}", i),
                    vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32],
                    None,
                )
                .await
                .expect("insert should succeed");
        }
        store
            .build_index()
            .await
            .expect("PQ index build should succeed via store");
    }
}
