//! Integration tests for hybrid search

#[cfg(test)]
mod integration_tests {
    use crate::hybrid_search::{
        HybridQuery, HybridSearchConfig, HybridSearchManager, KeywordAlgorithm, RankFusionStrategy,
        SearchMode, SearchWeights,
    };
    use std::collections::HashMap;

    #[test]
    fn test_end_to_end_hybrid_search() {
        let config = HybridSearchConfig {
            mode: SearchMode::Hybrid,
            fusion_strategy: RankFusionStrategy::ReciprocalRankFusion,
            ..Default::default()
        };

        let manager = HybridSearchManager::new(config).unwrap();

        // Add documents
        manager
            .add_document(
                "doc1",
                "machine learning and artificial intelligence",
                vec![0.9, 0.1, 0.0, 0.0],
                HashMap::from([("title".to_string(), "ML Intro".to_string())]),
            )
            .unwrap();

        manager
            .add_document(
                "doc2",
                "deep learning neural networks",
                vec![0.8, 0.2, 0.0, 0.0],
                HashMap::from([("title".to_string(), "DL Guide".to_string())]),
            )
            .unwrap();

        manager
            .add_document(
                "doc3",
                "natural language processing",
                vec![0.1, 0.9, 0.0, 0.0],
                HashMap::from([("title".to_string(), "NLP Basics".to_string())]),
            )
            .unwrap();

        // Search
        let query = HybridQuery {
            query_text: "machine learning".to_string(),
            query_vector: Some(vec![0.85, 0.15, 0.0, 0.0]),
            top_k: 10,
            weights: SearchWeights {
                keyword_weight: 0.4,
                semantic_weight: 0.6,
                recency_weight: 0.0,
            },
            filters: HashMap::new(),
        };

        let results = manager.search(query).unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, "doc1");
        assert!(results[0].score > 0.0);
        assert!(results[0].score_breakdown.keyword_score > 0.0);
        assert!(results[0].score_breakdown.semantic_score > 0.0);
    }

    #[test]
    fn test_different_fusion_strategies() {
        let strategies = vec![
            RankFusionStrategy::WeightedSum,
            RankFusionStrategy::ReciprocalRankFusion,
            RankFusionStrategy::Cascade,
            RankFusionStrategy::Interleave,
        ];

        for strategy in strategies {
            let config = HybridSearchConfig {
                fusion_strategy: strategy,
                ..Default::default()
            };

            let manager = HybridSearchManager::new(config).unwrap();

            manager
                .add_document("doc1", "test document", vec![0.1; 4], HashMap::new())
                .unwrap();

            let query = HybridQuery {
                query_text: "test".to_string(),
                query_vector: Some(vec![0.1; 4]),
                top_k: 10,
                weights: SearchWeights::default(),
                filters: HashMap::new(),
            };

            let results = manager.search(query);
            assert!(results.is_ok(), "Strategy {:?} failed", strategy);
        }
    }

    #[test]
    fn test_different_keyword_algorithms() {
        let algorithms = vec![
            KeywordAlgorithm::Bm25,
            KeywordAlgorithm::Tfidf,
            KeywordAlgorithm::Combined,
        ];

        for algo in algorithms {
            let config = HybridSearchConfig {
                keyword_algorithm: algo,
                mode: SearchMode::KeywordOnly,
                ..Default::default()
            };

            let manager = HybridSearchManager::new(config).unwrap();

            manager
                .add_document("doc1", "machine learning", vec![0.1; 4], HashMap::new())
                .unwrap();
            manager
                .add_document("doc2", "deep learning", vec![0.2; 4], HashMap::new())
                .unwrap();

            let query = HybridQuery {
                query_text: "machine learning".to_string(),
                query_vector: None,
                top_k: 10,
                weights: SearchWeights::default(),
                filters: HashMap::new(),
            };

            let results = manager.search(query);
            assert!(results.is_ok(), "Algorithm {:?} failed", algo);
        }
    }

    #[test]
    fn test_query_expansion() {
        let config = HybridSearchConfig {
            mode: SearchMode::KeywordOnly,
            enable_query_expansion: true,
            max_expanded_terms: 5,
            ..Default::default()
        };

        let manager = HybridSearchManager::new(config).unwrap();

        manager
            .add_document("doc1", "fast search engine", vec![0.1; 4], HashMap::new())
            .unwrap();
        manager
            .add_document("doc2", "quick lookup system", vec![0.2; 4], HashMap::new())
            .unwrap();

        let query = HybridQuery {
            query_text: "fast search".to_string(),
            query_vector: None,
            top_k: 10,
            weights: SearchWeights::default(),
            filters: HashMap::new(),
        };

        let results = manager.search(query).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_adaptive_mode() {
        let config = HybridSearchConfig {
            mode: SearchMode::Adaptive,
            ..Default::default()
        };

        let manager = HybridSearchManager::new(config).unwrap();

        manager
            .add_document("doc1", "test document", vec![0.1; 4], HashMap::new())
            .unwrap();

        // Short query without vector -> should use keyword search
        let query1 = HybridQuery {
            query_text: "test".to_string(),
            query_vector: None,
            top_k: 10,
            weights: SearchWeights::default(),
            filters: HashMap::new(),
        };

        let results1 = manager.search(query1).unwrap();
        assert!(!results1.is_empty());

        // Query with vector -> should use hybrid search
        let query2 = HybridQuery {
            query_text: "test document".to_string(),
            query_vector: Some(vec![0.1; 4]),
            top_k: 10,
            weights: SearchWeights::default(),
            filters: HashMap::new(),
        };

        let results2 = manager.search(query2).unwrap();
        assert!(!results2.is_empty());
    }

    #[test]
    fn test_metadata_preservation() {
        let config = HybridSearchConfig::default();
        let manager = HybridSearchManager::new(config).unwrap();

        let metadata = HashMap::from([
            ("author".to_string(), "John Doe".to_string()),
            ("date".to_string(), "2024-01-01".to_string()),
        ]);

        manager
            .add_document("doc1", "test content", vec![0.1; 4], metadata.clone())
            .unwrap();

        let query = HybridQuery {
            query_text: "test".to_string(),
            query_vector: Some(vec![0.1; 4]),
            top_k: 10,
            weights: SearchWeights::default(),
            filters: HashMap::new(),
        };

        let results = manager.search(query).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].metadata.get("author").unwrap(), "John Doe");
        assert_eq!(results[0].metadata.get("date").unwrap(), "2024-01-01");
    }

    #[test]
    fn test_score_thresholds() {
        let config = HybridSearchConfig {
            min_keyword_score: 0.5,
            min_semantic_score: 0.7,
            ..Default::default()
        };

        let manager = HybridSearchManager::new(config).unwrap();

        manager
            .add_document("doc1", "test", vec![1.0, 0.0, 0.0, 0.0], HashMap::new())
            .unwrap();
        manager
            .add_document("doc2", "other", vec![0.0, 1.0, 0.0, 0.0], HashMap::new())
            .unwrap();

        let query = HybridQuery {
            query_text: "test".to_string(),
            query_vector: Some(vec![0.5, 0.5, 0.0, 0.0]),
            top_k: 10,
            weights: SearchWeights::default(),
            filters: HashMap::new(),
        };

        let results = manager.search(query).unwrap();
        // Results should be filtered by thresholds
        assert!(!results.is_empty());
    }

    #[test]
    fn test_empty_results() {
        let config = HybridSearchConfig::default();
        let manager = HybridSearchManager::new(config).unwrap();

        manager
            .add_document("doc1", "foo bar", vec![0.1; 4], HashMap::new())
            .unwrap();

        let query = HybridQuery {
            query_text: "xyz".to_string(),
            query_vector: Some(vec![0.9; 4]),
            top_k: 10,
            weights: SearchWeights::default(),
            filters: HashMap::new(),
        };

        let results = manager.search(query).unwrap();
        // Might be empty or have low scores depending on semantic similarity
        assert!(results.is_empty() || results[0].score > 0.0);
    }
}
