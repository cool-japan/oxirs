//! Tests for Knowledge Graph Embedding models (TransE, DistMult, ComplEx)

#[cfg(test)]
mod transe_tests {
    use crate::kg_embeddings::{EmbeddingConfig, EmbeddingModel, TransE, Vocabulary};
    use crate::{model::NamedNode, StarTerm, StarTriple};

    fn create_test_triples() -> Vec<StarTriple> {
        vec![
            StarTriple {
                subject: StarTerm::NamedNode(NamedNode {
                    iri: "Alice".to_string(),
                }),
                predicate: StarTerm::NamedNode(NamedNode {
                    iri: "knows".to_string(),
                }),
                object: StarTerm::NamedNode(NamedNode {
                    iri: "Bob".to_string(),
                }),
            },
            StarTriple {
                subject: StarTerm::NamedNode(NamedNode {
                    iri: "Bob".to_string(),
                }),
                predicate: StarTerm::NamedNode(NamedNode {
                    iri: "knows".to_string(),
                }),
                object: StarTerm::NamedNode(NamedNode {
                    iri: "Charlie".to_string(),
                }),
            },
            StarTriple {
                subject: StarTerm::NamedNode(NamedNode {
                    iri: "Alice".to_string(),
                }),
                predicate: StarTerm::NamedNode(NamedNode {
                    iri: "likes".to_string(),
                }),
                object: StarTerm::NamedNode(NamedNode {
                    iri: "Coffee".to_string(),
                }),
            },
        ]
    }

    #[test]
    fn test_vocabulary_creation() {
        let triples = create_test_triples();
        let vocab = Vocabulary::from_triples(&triples);

        assert_eq!(vocab.num_entities(), 4); // Alice, Bob, Charlie, Coffee
        assert_eq!(vocab.num_relations(), 2); // knows, likes
    }

    #[test]
    fn test_transe_initialization() {
        let config = EmbeddingConfig {
            embedding_dim: 64,
            ..Default::default()
        };
        let model = TransE::new(config);
        assert_eq!(model.config.embedding_dim, 64);
    }

    #[test]
    fn test_transe_training() {
        let config = EmbeddingConfig {
            embedding_dim: 32,
            learning_rate: 0.01,
            batch_size: 2,
            num_negative_samples: 3,
            ..Default::default()
        };

        let mut model = TransE::new(config);
        let triples = create_test_triples();

        let stats = model.train(&triples, 10).unwrap();

        assert_eq!(stats.total_epochs, 10);
        assert!(stats.final_loss >= 0.0);
        assert_eq!(stats.losses_per_epoch.len(), 10);
    }

    #[test]
    fn test_get_embedding() {
        let config = EmbeddingConfig::default();
        let mut model = TransE::new(config);
        let triples = create_test_triples();

        model.train(&triples, 5).unwrap();

        let emb = model.get_embedding("Alice");
        assert!(emb.is_some());
        assert_eq!(emb.unwrap().len(), 128); // Default dimension
    }

    #[test]
    fn test_similarity() {
        let config = EmbeddingConfig {
            embedding_dim: 32,
            ..Default::default()
        };
        let mut model = TransE::new(config);
        let triples = create_test_triples();

        model.train(&triples, 20).unwrap();

        let sim = model.similarity("Alice", "Bob").unwrap();
        assert!((-1.0..=1.0).contains(&sim));
    }

    #[test]
    fn test_predict_tail() {
        let config = EmbeddingConfig {
            embedding_dim: 32,
            ..Default::default()
        };
        let mut model = TransE::new(config);
        let triples = create_test_triples();

        model.train(&triples, 10).unwrap();

        let predictions = model.predict_tail("Alice", "knows", 3).unwrap();
        assert_eq!(predictions.len(), 3);
        assert!(predictions[0].1 >= predictions[1].1); // Sorted by score
    }

    #[test]
    fn test_embedding_normalization() {
        let config = EmbeddingConfig {
            embedding_dim: 32,
            ..Default::default()
        };
        let mut model = TransE::new(config);
        let triples = create_test_triples();

        model.train(&triples, 5).unwrap();

        // Check that entity embeddings are normalized
        let emb = model.get_embedding("Alice").unwrap();
        let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 0.01); // Should be close to 1
    }

    #[test]
    fn test_transe_save_load_roundtrip() {
        let config = EmbeddingConfig {
            embedding_dim: 32,
            learning_rate: 0.01,
            batch_size: 2,
            num_negative_samples: 3,
            ..Default::default()
        };

        let mut model = TransE::new(config);
        let triples = create_test_triples();
        model.train(&triples, 10).unwrap();

        // Capture embedding before save
        let emb_before = model.get_embedding("Alice").unwrap();

        // Save to temp file
        let mut path = std::env::temp_dir();
        path.push(format!(
            "oxirs_transe_test_{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let path_str = path.to_string_lossy().to_string();

        model.save(&path_str).expect("save should succeed");

        // Load into a fresh model and verify embeddings match
        let mut model2 = TransE::new(EmbeddingConfig::default());
        model2.load(&path_str).expect("load should succeed");

        let emb_after = model2
            .get_embedding("Alice")
            .expect("Alice should exist after load");
        assert_eq!(emb_before.len(), emb_after.len());
        for (a, b) in emb_before.iter().zip(emb_after.iter()) {
            assert!(
                (a - b).abs() < 1e-12,
                "embedding values must match after round-trip"
            );
        }

        let _ = std::fs::remove_file(&path_str);
    }

    #[test]
    fn test_transe_save_load_predictions_match() {
        let config = EmbeddingConfig {
            embedding_dim: 16,
            batch_size: 2,
            num_negative_samples: 2,
            ..Default::default()
        };

        let mut model = TransE::new(config);
        let triples = create_test_triples();
        model.train(&triples, 5).unwrap();

        let preds_before = model.predict_tail("Alice", "knows", 2).unwrap();

        let mut path = std::env::temp_dir();
        path.push(format!(
            "oxirs_transe_pred_test_{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let path_str = path.to_string_lossy().to_string();

        model.save(&path_str).expect("save should succeed");

        let mut model2 = TransE::new(EmbeddingConfig::default());
        model2.load(&path_str).expect("load should succeed");

        let preds_after = model2.predict_tail("Alice", "knows", 2).unwrap();
        assert_eq!(preds_before.len(), preds_after.len());
        for ((name_a, score_a), (name_b, score_b)) in preds_before.iter().zip(preds_after.iter()) {
            assert_eq!(name_a, name_b, "prediction order must be preserved");
            assert!(
                (score_a - score_b).abs() < 1e-12,
                "prediction scores must match after round-trip"
            );
        }

        let _ = std::fs::remove_file(&path_str);
    }
}

#[cfg(test)]
mod advanced_model_tests {
    use crate::kg_embeddings::{ComplEx, DistMult, EmbeddingConfig, EmbeddingModel, TransE};
    use crate::{model::NamedNode, StarTerm, StarTriple};

    fn create_test_triples() -> Vec<StarTriple> {
        vec![
            StarTriple {
                subject: StarTerm::NamedNode(NamedNode {
                    iri: "Alice".to_string(),
                }),
                predicate: StarTerm::NamedNode(NamedNode {
                    iri: "knows".to_string(),
                }),
                object: StarTerm::NamedNode(NamedNode {
                    iri: "Bob".to_string(),
                }),
            },
            StarTriple {
                subject: StarTerm::NamedNode(NamedNode {
                    iri: "Bob".to_string(),
                }),
                predicate: StarTerm::NamedNode(NamedNode {
                    iri: "knows".to_string(),
                }),
                object: StarTerm::NamedNode(NamedNode {
                    iri: "Charlie".to_string(),
                }),
            },
            StarTriple {
                subject: StarTerm::NamedNode(NamedNode {
                    iri: "Alice".to_string(),
                }),
                predicate: StarTerm::NamedNode(NamedNode {
                    iri: "likes".to_string(),
                }),
                object: StarTerm::NamedNode(NamedNode {
                    iri: "Coffee".to_string(),
                }),
            },
        ]
    }

    // DistMult Tests
    #[test]
    fn test_distmult_initialization() {
        let config = EmbeddingConfig {
            embedding_dim: 64,
            ..Default::default()
        };
        let model = DistMult::new(config);
        assert_eq!(model.config.embedding_dim, 64);
    }

    #[test]
    fn test_distmult_training() {
        let config = EmbeddingConfig {
            embedding_dim: 32,
            learning_rate: 0.01,
            batch_size: 2,
            num_negative_samples: 3,
            ..Default::default()
        };

        let mut model = DistMult::new(config);
        let triples = create_test_triples();

        let stats = model.train(&triples, 10).unwrap();

        assert_eq!(stats.total_epochs, 10);
        assert!(stats.final_loss >= 0.0);
        assert_eq!(stats.losses_per_epoch.len(), 10);
    }

    #[test]
    fn test_distmult_get_embedding() {
        let config = EmbeddingConfig::default();
        let mut model = DistMult::new(config);
        let triples = create_test_triples();

        model.train(&triples, 5).unwrap();

        let emb = model.get_embedding("Alice");
        assert!(emb.is_some());
        assert_eq!(emb.unwrap().len(), 128);
    }

    #[test]
    fn test_distmult_similarity() {
        let config = EmbeddingConfig {
            embedding_dim: 32,
            ..Default::default()
        };
        let mut model = DistMult::new(config);
        let triples = create_test_triples();

        model.train(&triples, 20).unwrap();

        let sim = model.similarity("Alice", "Bob").unwrap();
        assert!((-1.0..=1.0).contains(&sim));
    }

    #[test]
    fn test_distmult_predict_tail() {
        let config = EmbeddingConfig {
            embedding_dim: 32,
            ..Default::default()
        };
        let mut model = DistMult::new(config);
        let triples = create_test_triples();

        model.train(&triples, 10).unwrap();

        let predictions = model.predict_tail("Alice", "knows", 3).unwrap();
        assert_eq!(predictions.len(), 3);
        assert!(predictions[0].1 >= predictions[1].1); // Sorted by score
    }

    // ComplEx Tests
    #[test]
    fn test_complex_initialization() {
        let config = EmbeddingConfig {
            embedding_dim: 64,
            ..Default::default()
        };
        let model = ComplEx::new(config);
        assert_eq!(model.config.embedding_dim, 64);
    }

    #[test]
    fn test_complex_training() {
        let config = EmbeddingConfig {
            embedding_dim: 32,
            learning_rate: 0.01,
            batch_size: 2,
            num_negative_samples: 3,
            ..Default::default()
        };

        let mut model = ComplEx::new(config);
        let triples = create_test_triples();

        let stats = model.train(&triples, 10).unwrap();

        assert_eq!(stats.total_epochs, 10);
        assert!(stats.final_loss >= 0.0);
        assert_eq!(stats.losses_per_epoch.len(), 10);
    }

    #[test]
    fn test_complex_get_embedding() {
        let config = EmbeddingConfig::default();
        let mut model = ComplEx::new(config);
        let triples = create_test_triples();

        model.train(&triples, 5).unwrap();

        let emb = model.get_embedding("Alice");
        assert!(emb.is_some());
        // ComplEx returns concatenated real+imag
        assert_eq!(emb.unwrap().len(), 128 * 2);
    }

    #[test]
    fn test_complex_similarity() {
        let config = EmbeddingConfig {
            embedding_dim: 32,
            ..Default::default()
        };
        let mut model = ComplEx::new(config);
        let triples = create_test_triples();

        model.train(&triples, 20).unwrap();

        let sim = model.similarity("Alice", "Bob").unwrap();
        assert!((-1.0..=1.0).contains(&sim));
    }

    #[test]
    fn test_complex_predict_tail() {
        let config = EmbeddingConfig {
            embedding_dim: 32,
            ..Default::default()
        };
        let mut model = ComplEx::new(config);
        let triples = create_test_triples();

        model.train(&triples, 10).unwrap();

        let predictions = model.predict_tail("Alice", "knows", 3).unwrap();
        assert_eq!(predictions.len(), 3);
        assert!(predictions[0].1 >= predictions[1].1); // Sorted by score
    }

    // Model Comparison Test
    #[test]
    fn test_model_comparison() {
        let config = EmbeddingConfig {
            embedding_dim: 32,
            learning_rate: 0.01,
            ..Default::default()
        };

        let triples = create_test_triples();

        // Train all three models
        let mut transe = TransE::new(config.clone());
        let mut distmult = DistMult::new(config.clone());
        let mut complex = ComplEx::new(config);

        let stats_transe = transe.train(&triples, 20).unwrap();
        let stats_distmult = distmult.train(&triples, 20).unwrap();
        let stats_complex = complex.train(&triples, 20).unwrap();

        // All should converge (loss decreases)
        assert!(stats_transe.final_loss < stats_transe.losses_per_epoch[0]);
        assert!(stats_distmult.final_loss < stats_distmult.losses_per_epoch[0]);
        assert!(stats_complex.final_loss < stats_complex.losses_per_epoch[0]);

        // All should be able to make predictions
        let pred_transe = transe.predict_tail("Alice", "knows", 1).unwrap();
        let pred_distmult = distmult.predict_tail("Alice", "knows", 1).unwrap();
        let pred_complex = complex.predict_tail("Alice", "knows", 1).unwrap();

        assert_eq!(pred_transe.len(), 1);
        assert_eq!(pred_distmult.len(), 1);
        assert_eq!(pred_complex.len(), 1);
    }

    #[test]
    fn test_distmult_save_load_roundtrip() {
        let config = EmbeddingConfig {
            embedding_dim: 32,
            learning_rate: 0.01,
            batch_size: 2,
            num_negative_samples: 3,
            ..Default::default()
        };

        let mut model = DistMult::new(config);
        let triples = create_test_triples();
        model.train(&triples, 10).unwrap();

        let emb_before = model.get_embedding("Bob").unwrap();

        let mut path = std::env::temp_dir();
        path.push(format!(
            "oxirs_distmult_test_{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let path_str = path.to_string_lossy().to_string();

        model.save(&path_str).expect("DistMult save should succeed");

        let mut model2 = DistMult::new(EmbeddingConfig::default());
        model2
            .load(&path_str)
            .expect("DistMult load should succeed");

        let emb_after = model2
            .get_embedding("Bob")
            .expect("Bob should exist after load");
        assert_eq!(emb_before.len(), emb_after.len());
        for (a, b) in emb_before.iter().zip(emb_after.iter()) {
            assert!(
                (a - b).abs() < 1e-12,
                "DistMult embedding values must match after round-trip"
            );
        }

        let _ = std::fs::remove_file(&path_str);
    }

    #[test]
    fn test_distmult_save_load_similarity_preserved() {
        let config = EmbeddingConfig {
            embedding_dim: 16,
            batch_size: 2,
            num_negative_samples: 2,
            ..Default::default()
        };

        let mut model = DistMult::new(config);
        let triples = create_test_triples();
        model.train(&triples, 5).unwrap();

        let sim_before = model.similarity("Alice", "Charlie").unwrap();

        let mut path = std::env::temp_dir();
        path.push(format!(
            "oxirs_distmult_sim_test_{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let path_str = path.to_string_lossy().to_string();

        model.save(&path_str).expect("DistMult save should succeed");

        let mut model2 = DistMult::new(EmbeddingConfig::default());
        model2
            .load(&path_str)
            .expect("DistMult load should succeed");

        let sim_after = model2.similarity("Alice", "Charlie").unwrap();
        assert!(
            (sim_before - sim_after).abs() < 1e-12,
            "similarity must be preserved after round-trip"
        );

        let _ = std::fs::remove_file(&path_str);
    }

    #[test]
    fn test_complex_save_load_roundtrip() {
        let config = EmbeddingConfig {
            embedding_dim: 32,
            learning_rate: 0.01,
            batch_size: 2,
            num_negative_samples: 3,
            ..Default::default()
        };

        let mut model = ComplEx::new(config);
        let triples = create_test_triples();
        model.train(&triples, 10).unwrap();

        // ComplEx returns concatenated real+imag embeddings
        let emb_before = model.get_embedding("Alice").unwrap();

        let mut path = std::env::temp_dir();
        path.push(format!(
            "oxirs_complex_test_{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let path_str = path.to_string_lossy().to_string();

        model.save(&path_str).expect("ComplEx save should succeed");

        let mut model2 = ComplEx::new(EmbeddingConfig::default());
        model2.load(&path_str).expect("ComplEx load should succeed");

        let emb_after = model2
            .get_embedding("Alice")
            .expect("Alice should exist after ComplEx load");
        assert_eq!(emb_before.len(), emb_after.len());
        for (a, b) in emb_before.iter().zip(emb_after.iter()) {
            assert!(
                (a - b).abs() < 1e-12,
                "ComplEx embedding values must match after round-trip"
            );
        }

        let _ = std::fs::remove_file(&path_str);
    }

    #[test]
    fn test_complex_save_load_predictions_preserved() {
        let config = EmbeddingConfig {
            embedding_dim: 16,
            batch_size: 2,
            num_negative_samples: 2,
            ..Default::default()
        };

        let mut model = ComplEx::new(config);
        let triples = create_test_triples();
        model.train(&triples, 5).unwrap();

        let preds_before = model.predict_tail("Bob", "knows", 2).unwrap();

        let mut path = std::env::temp_dir();
        path.push(format!(
            "oxirs_complex_pred_test_{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let path_str = path.to_string_lossy().to_string();

        model.save(&path_str).expect("ComplEx save should succeed");

        let mut model2 = ComplEx::new(EmbeddingConfig::default());
        model2.load(&path_str).expect("ComplEx load should succeed");

        let preds_after = model2.predict_tail("Bob", "knows", 2).unwrap();
        assert_eq!(preds_before.len(), preds_after.len());
        for ((name_a, score_a), (name_b, score_b)) in preds_before.iter().zip(preds_after.iter()) {
            assert_eq!(name_a, name_b, "ComplEx prediction order must be preserved");
            assert!(
                (score_a - score_b).abs() < 1e-12,
                "ComplEx prediction scores must match after round-trip"
            );
        }

        let _ = std::fs::remove_file(&path_str);
    }
}
