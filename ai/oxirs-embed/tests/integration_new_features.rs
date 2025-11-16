//! Integration Tests for New Features
//!
//! Tests for link prediction, clustering, community detection,
//! visualization, interpretability modules, and new embedding models (HolE, ConvE).

use oxirs_embed::{
    clustering::{ClusteringAlgorithm, ClusteringConfig, EntityClustering},
    community_detection::{CommunityAlgorithm, CommunityConfig, CommunityDetector},
    interpretability::{InterpretabilityAnalyzer, InterpretabilityConfig, InterpretationMethod},
    link_prediction::{LinkPredictionConfig, LinkPredictor},
    mixed_precision::{MixedPrecisionConfig, MixedPrecisionTrainer},
    models::hole::{HoLE, HoLEConfig},
    quantization::{BitWidth, ModelQuantizer, QuantizationConfig, QuantizationScheme},
    visualization::{EmbeddingVisualizer, ReductionMethod, VisualizationConfig},
    EmbeddingModel, ModelConfig, NamedNode, TransE, Triple,
};

#[cfg(feature = "conve")]
use oxirs_embed::models::conve::{ConvE, ConvEConfig};

use scirs2_core::ndarray_ext::{array, Array1};
use std::collections::HashMap;

/// Helper function to create sample embeddings
fn create_sample_embeddings() -> HashMap<String, scirs2_core::ndarray_ext::Array1<f32>> {
    let mut embeddings = HashMap::new();

    // Create clusters for testing
    // Cluster 1: around (1, 0, 0)
    embeddings.insert("e1".to_string(), array![1.0, 0.1, 0.0]);
    embeddings.insert("e2".to_string(), array![0.9, 0.0, 0.1]);
    embeddings.insert("e3".to_string(), array![1.1, 0.1, 0.0]);

    // Cluster 2: around (0, 1, 0)
    embeddings.insert("e4".to_string(), array![0.0, 1.0, 0.1]);
    embeddings.insert("e5".to_string(), array![0.1, 0.9, 0.0]);
    embeddings.insert("e6".to_string(), array![0.0, 1.1, 0.1]);

    // Cluster 3: around (0, 0, 1)
    embeddings.insert("e7".to_string(), array![0.1, 0.0, 1.0]);
    embeddings.insert("e8".to_string(), array![0.0, 0.1, 0.9]);

    embeddings
}

#[tokio::test]
async fn test_link_prediction_integration() {
    // Create and train a simple model
    let config = ModelConfig {
        dimensions: 50,
        learning_rate: 0.01,
        max_epochs: 30,
        ..Default::default()
    };

    let mut model = TransE::new(config);

    // Add training data
    model
        .add_triple(Triple::new(
            NamedNode::new("alice").unwrap(),
            NamedNode::new("knows").unwrap(),
            NamedNode::new("bob").unwrap(),
        ))
        .unwrap();

    model
        .add_triple(Triple::new(
            NamedNode::new("bob").unwrap(),
            NamedNode::new("knows").unwrap(),
            NamedNode::new("charlie").unwrap(),
        ))
        .unwrap();

    model
        .add_triple(Triple::new(
            NamedNode::new("alice").unwrap(),
            NamedNode::new("likes").unwrap(),
            NamedNode::new("dave").unwrap(),
        ))
        .unwrap();

    // Train
    let stats = model.train(Some(30)).await.unwrap();
    assert!(stats.final_loss >= 0.0);

    // Test link prediction
    let pred_config = LinkPredictionConfig {
        top_k: 3,
        filter_known_triples: false,
        ..Default::default()
    };

    let predictor = LinkPredictor::new(pred_config, model);

    let candidates = vec!["bob".to_string(), "charlie".to_string(), "dave".to_string()];

    let predictions = predictor
        .predict_tail("alice", "knows", &candidates)
        .unwrap();

    assert!(!predictions.is_empty());
    assert!(predictions.len() <= 3);

    // Verify predictions are sorted by score
    for i in 0..predictions.len() - 1 {
        assert!(predictions[i].score >= predictions[i + 1].score);
    }
}

#[test]
fn test_clustering_integration() {
    let embeddings = create_sample_embeddings();

    // Test K-Means clustering
    let config = ClusteringConfig {
        algorithm: ClusteringAlgorithm::KMeans,
        num_clusters: 3,
        max_iterations: 50,
        ..Default::default()
    };

    let mut clustering = EntityClustering::new(config);
    let result = clustering.cluster(&embeddings).unwrap();

    assert_eq!(result.assignments.len(), 8);
    assert_eq!(result.centroids.len(), 3);
    assert!(result.silhouette_score >= -1.0 && result.silhouette_score <= 1.0);

    // Verify that similar entities are in same cluster
    let cluster_e1 = result.assignments["e1"];
    let cluster_e2 = result.assignments["e2"];
    assert_eq!(
        cluster_e1, cluster_e2,
        "e1 and e2 should be in same cluster"
    );
}

#[test]
fn test_hierarchical_clustering() {
    let embeddings = create_sample_embeddings();

    let config = ClusteringConfig {
        algorithm: ClusteringAlgorithm::Hierarchical,
        num_clusters: 3,
        ..Default::default()
    };

    let mut clustering = EntityClustering::new(config);
    let result = clustering.cluster(&embeddings).unwrap();

    assert_eq!(result.assignments.len(), 8);
    assert_eq!(result.num_iterations, 5); // 8 - 3 merges
}

#[test]
fn test_dbscan_clustering() {
    let embeddings = create_sample_embeddings();

    let config = ClusteringConfig {
        algorithm: ClusteringAlgorithm::DBSCAN,
        epsilon: 0.5,
        min_points: 2,
        ..Default::default()
    };

    let mut clustering = EntityClustering::new(config);
    let result = clustering.cluster(&embeddings).unwrap();

    assert!(!result.centroids.is_empty());
}

#[test]
fn test_community_detection_integration() {
    // Create triples representing a graph
    let triples = vec![
        Triple::new(
            NamedNode::new("a").unwrap(),
            NamedNode::new("r").unwrap(),
            NamedNode::new("b").unwrap(),
        ),
        Triple::new(
            NamedNode::new("b").unwrap(),
            NamedNode::new("r").unwrap(),
            NamedNode::new("c").unwrap(),
        ),
        Triple::new(
            NamedNode::new("a").unwrap(),
            NamedNode::new("r").unwrap(),
            NamedNode::new("c").unwrap(),
        ),
        Triple::new(
            NamedNode::new("d").unwrap(),
            NamedNode::new("r").unwrap(),
            NamedNode::new("e").unwrap(),
        ),
        Triple::new(
            NamedNode::new("e").unwrap(),
            NamedNode::new("r").unwrap(),
            NamedNode::new("f").unwrap(),
        ),
    ];

    // Test Louvain algorithm
    let config = CommunityConfig {
        algorithm: CommunityAlgorithm::Louvain,
        max_iterations: 10,
        ..Default::default()
    };

    let mut detector = CommunityDetector::new(config);
    let result = detector.detect_from_triples(&triples).unwrap();

    assert!(result.num_communities > 0);
    assert_eq!(result.assignments.len(), 6); // a, b, c, d, e, f
    assert!(result.modularity >= 0.0);
}

#[test]
fn test_label_propagation() {
    let triples = vec![
        Triple::new(
            NamedNode::new("a").unwrap(),
            NamedNode::new("r").unwrap(),
            NamedNode::new("b").unwrap(),
        ),
        Triple::new(
            NamedNode::new("b").unwrap(),
            NamedNode::new("r").unwrap(),
            NamedNode::new("c").unwrap(),
        ),
    ];

    let config = CommunityConfig {
        algorithm: CommunityAlgorithm::LabelPropagation,
        max_iterations: 20,
        ..Default::default()
    };

    let mut detector = CommunityDetector::new(config);
    let result = detector.detect_from_triples(&triples).unwrap();

    assert!(result.num_communities > 0);
}

#[test]
fn test_embedding_based_community_detection() {
    let embeddings = create_sample_embeddings();

    let config = CommunityConfig {
        algorithm: CommunityAlgorithm::EmbeddingBased,
        similarity_threshold: 0.8,
        min_community_size: 2,
        ..Default::default()
    };

    let mut detector = CommunityDetector::new(config);
    let result = detector.detect_from_embeddings(&embeddings).unwrap();

    assert!(result.num_communities > 0);
}

#[test]
fn test_pca_visualization() {
    let embeddings = create_sample_embeddings();

    let config = VisualizationConfig {
        method: ReductionMethod::PCA,
        target_dims: 2,
        ..Default::default()
    };

    let mut visualizer = EmbeddingVisualizer::new(config);
    let result = visualizer.visualize(&embeddings).unwrap();

    assert_eq!(result.coordinates.len(), 8);
    assert_eq!(result.dimensions, 2);
    assert!(result.explained_variance.is_some());

    // Verify all coordinates are 2D
    for coords in result.coordinates.values() {
        assert_eq!(coords.len(), 2);
    }
}

#[test]
fn test_tsne_visualization() {
    let embeddings = create_sample_embeddings();

    let config = VisualizationConfig {
        method: ReductionMethod::TSNE,
        target_dims: 2,
        max_iterations: 100,
        ..Default::default()
    };

    let mut visualizer = EmbeddingVisualizer::new(config);
    let result = visualizer.visualize(&embeddings).unwrap();

    assert_eq!(result.coordinates.len(), 8);
    assert_eq!(result.dimensions, 2);
    assert!(result.final_loss.is_some());
}

#[test]
fn test_random_projection_visualization() {
    let embeddings = create_sample_embeddings();

    let config = VisualizationConfig {
        method: ReductionMethod::RandomProjection,
        target_dims: 2,
        ..Default::default()
    };

    let mut visualizer = EmbeddingVisualizer::new(config);
    let result = visualizer.visualize(&embeddings).unwrap();

    assert_eq!(result.coordinates.len(), 8);
    assert_eq!(result.dimensions, 2);
}

#[test]
fn test_visualization_export() {
    let embeddings = create_sample_embeddings();

    let config = VisualizationConfig::default();
    let mut visualizer = EmbeddingVisualizer::new(config);
    let result = visualizer.visualize(&embeddings).unwrap();

    // Test JSON export
    let json = visualizer.export_json(&result).unwrap();
    assert!(json.contains("coordinates"));

    // Test CSV export
    let csv = visualizer.export_csv(&result).unwrap();
    assert!(csv.contains("entity,dim1,dim2"));
}

#[test]
fn test_similarity_analysis() {
    let embeddings = create_sample_embeddings();

    let config = InterpretabilityConfig {
        method: InterpretationMethod::SimilarityAnalysis,
        top_k: 3,
        ..Default::default()
    };

    let analyzer = InterpretabilityAnalyzer::new(config);
    let analysis = analyzer.similarity_analysis("e1", &embeddings).unwrap();

    assert_eq!(analysis.entity, "e1");
    assert_eq!(analysis.similar_entities.len(), 3);
    assert_eq!(analysis.dissimilar_entities.len(), 3);
    assert!(analysis.avg_similarity >= 0.0 && analysis.avg_similarity <= 1.0);

    // e2 and e3 should be most similar to e1
    assert!(analysis.similar_entities[0].0 == "e2" || analysis.similar_entities[0].0 == "e3");
}

#[test]
fn test_feature_importance() {
    let embeddings = create_sample_embeddings();

    let config = InterpretabilityConfig {
        method: InterpretationMethod::FeatureImportance,
        top_k: 3,
        ..Default::default()
    };

    let analyzer = InterpretabilityAnalyzer::new(config);
    let importance = analyzer.feature_importance("e1", &embeddings).unwrap();

    assert_eq!(importance.entity, "e1");
    assert_eq!(importance.important_features.len(), 3);
    assert_eq!(importance.feature_stats.mean.len(), 3);

    // Verify features are sorted by importance
    for i in 0..importance.important_features.len() - 1 {
        assert!(importance.important_features[i].1 >= importance.important_features[i + 1].1);
    }
}

#[test]
fn test_counterfactual_explanation() {
    let embeddings = create_sample_embeddings();

    let config = InterpretabilityConfig::default();
    let analyzer = InterpretabilityAnalyzer::new(config);

    let cf = analyzer
        .counterfactual_explanation("e1", "e4", &embeddings)
        .unwrap();

    assert_eq!(cf.original, "e1");
    assert_eq!(cf.target, "e4");
    assert!(!cf.required_changes.is_empty());
    assert!(cf.difficulty >= 0.0 && cf.difficulty <= 1.0);
}

#[test]
fn test_nearest_neighbors_analysis() {
    let embeddings = create_sample_embeddings();

    let config = InterpretabilityConfig {
        method: InterpretationMethod::NearestNeighbors,
        top_k: 3,
        detailed: true,
        ..Default::default()
    };

    let analyzer = InterpretabilityAnalyzer::new(config);
    let nn = analyzer
        .nearest_neighbors_analysis("e1", &embeddings)
        .unwrap();

    assert_eq!(nn.entity, "e1");
    assert_eq!(nn.neighbors.len(), 3);

    // Verify neighbors are sorted by distance
    for i in 0..nn.neighbors.len() - 1 {
        assert!(nn.neighbors[i].1 <= nn.neighbors[i + 1].1);
    }
}

#[test]
fn test_interpretability_report() {
    let embeddings = create_sample_embeddings();

    let config = InterpretabilityConfig::default();
    let analyzer = InterpretabilityAnalyzer::new(config);

    let report = analyzer.generate_report("e1", &embeddings).unwrap();

    assert!(report.contains("Interpretability Report"));
    assert!(report.contains("Similarity Analysis"));
    assert!(report.contains("Feature Importance"));
    assert!(report.contains("Nearest Neighbors"));
}

#[test]
fn test_mixed_precision_training() {
    let config = MixedPrecisionConfig {
        enabled: true,
        init_scale: 1024.0,
        dynamic_loss_scale: true,
        grad_clip_threshold: 10.0, // Set high to avoid clipping in this test
        ..Default::default()
    };

    let mut trainer = MixedPrecisionTrainer::new(config);

    // Test loss scaling
    let loss = 0.5;
    let scaled_loss = trainer.scale_loss(loss);
    assert_eq!(scaled_loss, 512.0);

    // Test gradient unscaling
    let scaled_grads = array![1024.0, 2048.0, 512.0];
    let unscaled = trainer.unscale_gradients(&scaled_grads).unwrap();

    assert!((unscaled[0] - 1.0).abs() < 1e-5);
    assert!((unscaled[1] - 2.0).abs() < 1e-5);
    assert!((unscaled[2] - 0.5).abs() < 1e-5);

    // Test parameter update
    let mut params = array![1.0, 2.0, 3.0];
    // In mixed precision training, gradients are scaled by init_scale (1024.0)
    let scaled_grads = array![0.1 * 1024.0, 0.2 * 1024.0, 0.3 * 1024.0];
    trainer
        .update_parameters(&mut params, &scaled_grads, 0.1)
        .unwrap();

    // Verify update was applied (should be params - lr * unscaled_grads)
    // unscaled_grads = scaled_grads / 1024.0 = [0.1, 0.2, 0.3]
    // params[0] = 1.0 - 0.1 * 0.1 = 0.99
    assert!((params[0] - 0.99).abs() < 1e-5);
}

#[test]
fn test_quantization_integration() {
    let mut embeddings = HashMap::new();
    // Use larger embeddings (128 dimensions) to ensure compression ratio > 1.0
    // With small embeddings, the quantization params overhead dominates
    for i in 0..10 {
        let emb = Array1::from_vec(vec![i as f32; 128]);
        embeddings.insert(format!("e{}", i), emb);
    }

    // Test Int8 quantization
    let config = QuantizationConfig {
        scheme: QuantizationScheme::Symmetric,
        bit_width: BitWidth::Int8,
        ..Default::default()
    };

    let mut quantizer = ModelQuantizer::new(config);
    let quantized = quantizer.quantize_embeddings(&embeddings).unwrap();

    assert_eq!(quantized.len(), 10);
    // With 128-dim embeddings, compression ratio should be ~3.8x
    assert!(quantizer.get_stats().compression_ratio > 1.0);

    // Test roundtrip
    let dequantized = quantizer.dequantize_embeddings(&quantized);
    assert_eq!(dequantized.len(), 10);

    // Verify values are approximately preserved
    for (entity, original) in &embeddings {
        let recovered = &dequantized[entity];
        for i in 0..original.len() {
            let error = (original[i] - recovered[i]).abs();
            assert!(error < 2.0, "Quantization error too large: {}", error);
        }
    }
}

#[test]
fn test_quantization_compression_ratio() {
    let mut embeddings = HashMap::new();
    for i in 0..100 {
        let emb = scirs2_core::ndarray_ext::Array1::from_vec(vec![i as f32; 128]);
        embeddings.insert(format!("e{}", i), emb);
    }

    let config = QuantizationConfig::default();
    let mut quantizer = ModelQuantizer::new(config);

    quantizer.quantize_embeddings(&embeddings).unwrap();

    let stats = quantizer.get_stats();
    assert!(stats.compression_ratio > 3.0);
    assert!(stats.compression_ratio < 5.0);
}

#[test]
fn test_end_to_end_pipeline() {
    // Create embeddings with sufficient dimensions for meaningful quantization
    let mut embeddings = HashMap::new();
    for i in 0..8 {
        let mut values = vec![0.0; 128];
        for (j, value) in values.iter_mut().enumerate().take(128) {
            *value = ((i + j) as f32 * 0.1).sin();
        }
        embeddings.insert(format!("e{}", i + 1), Array1::from_vec(values));
    }

    // 1. Cluster embeddings
    let cluster_config = ClusteringConfig {
        algorithm: ClusteringAlgorithm::KMeans,
        num_clusters: 3,
        ..Default::default()
    };
    let mut clustering = EntityClustering::new(cluster_config);
    let cluster_result = clustering.cluster(&embeddings).unwrap();

    assert!(cluster_result.centroids.len() == 3);

    // 2. Visualize embeddings
    let vis_config = VisualizationConfig {
        method: ReductionMethod::PCA,
        target_dims: 2,
        ..Default::default()
    };
    let mut visualizer = EmbeddingVisualizer::new(vis_config);
    let vis_result = visualizer.visualize(&embeddings).unwrap();

    assert_eq!(vis_result.dimensions, 2);

    // 3. Interpret embeddings
    let interp_config = InterpretabilityConfig::default();
    let analyzer = InterpretabilityAnalyzer::new(interp_config);
    let analysis = analyzer.similarity_analysis("e1", &embeddings).unwrap();

    assert!(!analysis.similar_entities.is_empty());

    // 4. Quantize embeddings
    let quant_config = QuantizationConfig::default();
    let mut quantizer = ModelQuantizer::new(quant_config);
    let _quantized = quantizer.quantize_embeddings(&embeddings).unwrap();

    assert!(quantizer.get_stats().compression_ratio > 1.0);

    println!("End-to-end pipeline completed successfully!");
}

// ============================================================================
// HolE (Holographic Embeddings) Integration Tests
// ============================================================================

#[tokio::test]
async fn test_hole_basic_training() {
    let config = HoLEConfig {
        base: ModelConfig {
            dimensions: 50,
            learning_rate: 0.01,
            max_epochs: 40,
            ..Default::default()
        },
        regularization: 0.0001,
        margin: 1.0,
        num_negatives: 5,
        use_sigmoid: true,
    };

    let mut model = HoLE::new(config);

    // Add training triples
    model
        .add_triple(Triple::new(
            NamedNode::new("paris").unwrap(),
            NamedNode::new("capital_of").unwrap(),
            NamedNode::new("france").unwrap(),
        ))
        .unwrap();

    model
        .add_triple(Triple::new(
            NamedNode::new("london").unwrap(),
            NamedNode::new("capital_of").unwrap(),
            NamedNode::new("uk").unwrap(),
        ))
        .unwrap();

    model
        .add_triple(Triple::new(
            NamedNode::new("france").unwrap(),
            NamedNode::new("in_continent").unwrap(),
            NamedNode::new("europe").unwrap(),
        ))
        .unwrap();

    model
        .add_triple(Triple::new(
            NamedNode::new("uk").unwrap(),
            NamedNode::new("in_continent").unwrap(),
            NamedNode::new("europe").unwrap(),
        ))
        .unwrap();

    // Train the model
    let stats = model.train(Some(40)).await.unwrap();

    assert_eq!(stats.epochs_completed, 40);
    assert!(stats.final_loss >= 0.0);
    assert!(stats.training_time_seconds > 0.0);
    assert_eq!(stats.loss_history.len(), 40);

    // Verify model statistics
    let model_stats = model.get_stats();
    assert_eq!(model_stats.num_entities, 5);
    assert_eq!(model_stats.num_relations, 2);
    assert_eq!(model_stats.num_triples, 4);
    assert_eq!(model_stats.dimensions, 50);
    assert!(model_stats.is_trained);
    assert_eq!(model_stats.model_type, "HoLE");
}

#[tokio::test]
async fn test_hole_link_prediction() {
    let config = HoLEConfig {
        base: ModelConfig {
            dimensions: 50,
            max_epochs: 30,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut model = HoLE::new(config);

    // Add training data
    for i in 0..5 {
        model
            .add_triple(Triple::new(
                NamedNode::new(&format!("entity_{}", i)).unwrap(),
                NamedNode::new("knows").unwrap(),
                NamedNode::new(&format!("entity_{}", (i + 1) % 5)).unwrap(),
            ))
            .unwrap();
    }

    model.train(Some(30)).await.unwrap();

    // Test tail prediction
    let predictions = model.predict_objects("entity_0", "knows", 3).unwrap();
    assert!(!predictions.is_empty());
    assert!(predictions.len() <= 3);

    // Test head prediction
    let predictions = model.predict_subjects("knows", "entity_1", 3).unwrap();
    assert!(!predictions.is_empty());

    // Test relation prediction
    let predictions = model.predict_relations("entity_0", "entity_1", 2).unwrap();
    assert!(!predictions.is_empty());
}

#[tokio::test]
async fn test_hole_with_link_predictor() {
    let config = HoLEConfig {
        base: ModelConfig {
            dimensions: 64,
            learning_rate: 0.01,
            max_epochs: 50,
            ..Default::default()
        },
        num_negatives: 10,
        ..Default::default()
    };

    let mut model = HoLE::new(config);

    // Add comprehensive training data
    let triples = vec![
        ("alice", "knows", "bob"),
        ("bob", "knows", "charlie"),
        ("charlie", "knows", "dave"),
        ("alice", "likes", "music"),
        ("bob", "likes", "sports"),
        ("charlie", "likes", "music"),
    ];

    for (s, p, o) in triples {
        model
            .add_triple(Triple::new(
                NamedNode::new(s).unwrap(),
                NamedNode::new(p).unwrap(),
                NamedNode::new(o).unwrap(),
            ))
            .unwrap();
    }

    model.train(Some(50)).await.unwrap();

    // Use LinkPredictor for advanced prediction
    let pred_config = LinkPredictionConfig {
        top_k: 3,
        filter_known_triples: false,
        min_confidence: 0.0,
        ..Default::default()
    };

    let predictor = LinkPredictor::new(pred_config, model);
    let candidates = vec![
        "bob".to_string(),
        "charlie".to_string(),
        "dave".to_string(),
        "music".to_string(),
    ];

    let predictions = predictor
        .predict_tail("alice", "knows", &candidates)
        .unwrap();
    assert!(!predictions.is_empty());
    assert!(predictions.len() <= 3);

    // Verify predictions are ranked properly
    for i in 0..predictions.len() - 1 {
        assert!(predictions[i].rank <= predictions[i + 1].rank);
    }
}

#[tokio::test]
async fn test_hole_embedding_quality() {
    let config = HoLEConfig {
        base: ModelConfig {
            dimensions: 100,
            max_epochs: 60,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut model = HoLE::new(config);

    // Create symmetric relationships
    model
        .add_triple(Triple::new(
            NamedNode::new("a").unwrap(),
            NamedNode::new("similar_to").unwrap(),
            NamedNode::new("b").unwrap(),
        ))
        .unwrap();

    model
        .add_triple(Triple::new(
            NamedNode::new("b").unwrap(),
            NamedNode::new("similar_to").unwrap(),
            NamedNode::new("a").unwrap(),
        ))
        .unwrap();

    model.train(Some(60)).await.unwrap();

    // Test embedding retrieval
    let emb_a = model.get_entity_embedding("a").unwrap();
    let emb_b = model.get_entity_embedding("b").unwrap();
    let rel_emb = model.get_relation_embedding("similar_to").unwrap();

    assert_eq!(emb_a.dimensions, 100);
    assert_eq!(emb_b.dimensions, 100);
    assert_eq!(rel_emb.dimensions, 100);

    // Test triple scoring
    let score_ab = model.score_triple("a", "similar_to", "b").unwrap();
    let score_ba = model.score_triple("b", "similar_to", "a").unwrap();

    // Scores should be positive and relatively similar for symmetric relation
    assert!(score_ab > 0.0);
    assert!(score_ba > 0.0);
}

// ============================================================================
// ConvE (Convolutional Embeddings) Integration Tests
// ============================================================================

#[cfg(feature = "conve")]
#[tokio::test]
async fn test_conve_basic_training() {
    let config = ConvEConfig {
        base: ModelConfig {
            dimensions: 100,
            learning_rate: 0.001,
            max_epochs: 30,
            ..Default::default()
        },
        reshape_width: 10,
        num_filters: 16,
        kernel_size: 3,
        dropout_rate: 0.3,
        ..Default::default()
    };

    let mut model = ConvE::new(config);

    // Add training triples
    model
        .add_triple(Triple::new(
            NamedNode::new("dog").unwrap(),
            NamedNode::new("is_a").unwrap(),
            NamedNode::new("animal").unwrap(),
        ))
        .unwrap();

    model
        .add_triple(Triple::new(
            NamedNode::new("cat").unwrap(),
            NamedNode::new("is_a").unwrap(),
            NamedNode::new("animal").unwrap(),
        ))
        .unwrap();

    model
        .add_triple(Triple::new(
            NamedNode::new("dog").unwrap(),
            NamedNode::new("has_trait").unwrap(),
            NamedNode::new("loyal").unwrap(),
        ))
        .unwrap();

    model
        .add_triple(Triple::new(
            NamedNode::new("cat").unwrap(),
            NamedNode::new("has_trait").unwrap(),
            NamedNode::new("independent").unwrap(),
        ))
        .unwrap();

    // Train the model
    let stats = model.train(Some(30)).await.unwrap();

    assert_eq!(stats.epochs_completed, 30);
    assert!(stats.final_loss >= 0.0);
    assert!(stats.training_time_seconds > 0.0);

    // Verify model statistics
    let model_stats = model.get_stats();
    assert_eq!(model_stats.num_entities, 5);
    assert_eq!(model_stats.num_relations, 2);
    assert_eq!(model_stats.num_triples, 4);
    assert_eq!(model_stats.dimensions, 100);
    assert!(model_stats.is_trained);
    assert_eq!(model_stats.model_type, "ConvE");
}

#[cfg(feature = "conve")]
#[tokio::test]
async fn test_conve_link_prediction() {
    let config = ConvEConfig {
        base: ModelConfig {
            dimensions: 100,
            max_epochs: 25,
            ..Default::default()
        },
        reshape_width: 10,
        num_filters: 16,
        ..Default::default()
    };

    let mut model = ConvE::new(config);

    // Add circular relationships
    for i in 0..4 {
        model
            .add_triple(Triple::new(
                NamedNode::new(&format!("node_{}", i)).unwrap(),
                NamedNode::new("connected_to").unwrap(),
                NamedNode::new(&format!("node_{}", (i + 1) % 4)).unwrap(),
            ))
            .unwrap();
    }

    model.train(Some(25)).await.unwrap();

    // Test predictions
    let predictions = model.predict_objects("node_0", "connected_to", 3).unwrap();
    assert!(!predictions.is_empty());

    let predictions = model.predict_subjects("connected_to", "node_1", 3).unwrap();
    assert!(!predictions.is_empty());
}

#[cfg(feature = "conve")]
#[tokio::test]
async fn test_conve_with_different_configurations() {
    // Test with smaller dimensions and filters
    let config = ConvEConfig {
        base: ModelConfig {
            dimensions: 50,
            max_epochs: 20,
            ..Default::default()
        },
        reshape_width: 10,
        num_filters: 8,
        kernel_size: 2,
        ..Default::default()
    };

    let mut model = ConvE::new(config);

    model
        .add_triple(Triple::new(
            NamedNode::new("x").unwrap(),
            NamedNode::new("r").unwrap(),
            NamedNode::new("y").unwrap(),
        ))
        .unwrap();

    let stats = model.train(Some(20)).await.unwrap();
    assert!(stats.final_loss >= 0.0);
}

#[cfg(feature = "conve")]
#[tokio::test]
async fn test_conve_embedding_dimensions() {
    let config = ConvEConfig {
        base: ModelConfig {
            dimensions: 200,
            max_epochs: 15,
            ..Default::default()
        },
        reshape_width: 20,
        num_filters: 32,
        ..Default::default()
    };

    let mut model = ConvE::new(config);

    model
        .add_triple(Triple::new(
            NamedNode::new("entity1").unwrap(),
            NamedNode::new("relation1").unwrap(),
            NamedNode::new("entity2").unwrap(),
        ))
        .unwrap();

    model.train(Some(15)).await.unwrap();

    // Verify embedding dimensions
    let emb = model.get_entity_embedding("entity1").unwrap();
    assert_eq!(emb.dimensions, 200);

    let rel_emb = model.get_relation_embedding("relation1").unwrap();
    assert_eq!(rel_emb.dimensions, 200);
}

// ============================================================================
// Comparative Tests: HolE vs ConvE vs TransE
// ============================================================================

#[cfg(feature = "conve")]
#[tokio::test]
async fn test_model_comparison_on_same_data() {
    // Shared training data
    let triples = vec![
        ("a", "rel1", "b"),
        ("b", "rel2", "c"),
        ("c", "rel1", "d"),
        ("a", "rel2", "d"),
    ];

    // TransE model
    let mut transe = TransE::new(ModelConfig {
        dimensions: 50,
        max_epochs: 30,
        ..Default::default()
    });

    for (s, p, o) in &triples {
        transe
            .add_triple(Triple::new(
                NamedNode::new(s).unwrap(),
                NamedNode::new(p).unwrap(),
                NamedNode::new(o).unwrap(),
            ))
            .unwrap();
    }

    let transe_stats = transe.train(Some(30)).await.unwrap();

    // HolE model
    let mut hole = HoLE::new(HoLEConfig {
        base: ModelConfig {
            dimensions: 50,
            max_epochs: 30,
            ..Default::default()
        },
        ..Default::default()
    });

    for (s, p, o) in &triples {
        hole.add_triple(Triple::new(
            NamedNode::new(s).unwrap(),
            NamedNode::new(p).unwrap(),
            NamedNode::new(o).unwrap(),
        ))
        .unwrap();
    }

    let hole_stats = hole.train(Some(30)).await.unwrap();

    // ConvE model
    let mut conve = ConvE::new(ConvEConfig {
        base: ModelConfig {
            dimensions: 50,
            max_epochs: 30,
            ..Default::default()
        },
        reshape_width: 10,
        num_filters: 8,
        ..Default::default()
    });

    for (s, p, o) in &triples {
        conve
            .add_triple(Triple::new(
                NamedNode::new(s).unwrap(),
                NamedNode::new(p).unwrap(),
                NamedNode::new(o).unwrap(),
            ))
            .unwrap();
    }

    let conve_stats = conve.train(Some(30)).await.unwrap();

    // All models should have trained successfully
    assert!(transe_stats.final_loss >= 0.0);
    assert!(hole_stats.final_loss >= 0.0);
    assert!(conve_stats.final_loss >= 0.0);

    // All models should produce valid predictions
    let transe_pred = transe.predict_objects("a", "rel1", 2).unwrap();
    let hole_pred = hole.predict_objects("a", "rel1", 2).unwrap();
    let conve_pred = conve.predict_objects("a", "rel1", 2).unwrap();

    assert!(!transe_pred.is_empty());
    assert!(!hole_pred.is_empty());
    assert!(!conve_pred.is_empty());

    println!("TransE final loss: {}", transe_stats.final_loss);
    println!("HolE final loss: {}", hole_stats.final_loss);
    println!("ConvE final loss: {}", conve_stats.final_loss);
}

#[tokio::test]
async fn test_hole_conve_with_clustering() {
    // Train HolE model
    let mut hole = HoLE::new(HoLEConfig {
        base: ModelConfig {
            dimensions: 30,
            max_epochs: 20,
            ..Default::default()
        },
        ..Default::default()
    });

    for i in 0..6 {
        hole.add_triple(Triple::new(
            NamedNode::new(&format!("e{}", i)).unwrap(),
            NamedNode::new("connects").unwrap(),
            NamedNode::new(&format!("e{}", (i + 1) % 6)).unwrap(),
        ))
        .unwrap();
    }

    hole.train(Some(20)).await.unwrap();

    // Extract embeddings
    let mut embeddings = HashMap::new();
    for i in 0..6 {
        let entity = format!("e{}", i);
        if let Ok(emb_vec) = hole.get_entity_embedding(&entity) {
            let array_emb = scirs2_core::ndarray_ext::Array1::from_vec(emb_vec.values.clone());
            embeddings.insert(entity, array_emb);
        }
    }

    // Cluster the embeddings
    let cluster_config = ClusteringConfig {
        algorithm: ClusteringAlgorithm::KMeans,
        num_clusters: 2,
        ..Default::default()
    };

    let mut clustering = EntityClustering::new(cluster_config);
    let result = clustering.cluster(&embeddings).unwrap();

    assert_eq!(result.centroids.len(), 2);
    assert_eq!(result.assignments.len(), 6);
}
