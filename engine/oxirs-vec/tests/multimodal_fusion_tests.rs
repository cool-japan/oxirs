//! Comprehensive tests for multimodal search fusion
//!
//! This test suite covers all fusion strategies, normalization methods,
//! and real-world scenarios for multimodal search.

use oxirs_vec::hybrid_search::multimodal_fusion::{
    FusedResult, FusionConfig, FusionStrategy, Modality, MultimodalFusion, NormalizationMethod,
};
use oxirs_vec::hybrid_search::types::DocumentScore;

/// Helper function to create test results for conference venues
fn create_conference_venue_results() -> (
    Vec<DocumentScore>,
    Vec<DocumentScore>,
    Vec<DocumentScore>,
) {
    // Text search results: Based on keyword matching (BM25-style scores)
    let text = vec![
        DocumentScore {
            doc_id: "neurips2025".to_string(),
            score: 15.2, // High BM25 score for "machine learning conference"
            rank: 0,
        },
        DocumentScore {
            doc_id: "icml2025".to_string(),
            score: 14.8,
            rank: 1,
        },
        DocumentScore {
            doc_id: "cvpr2025".to_string(),
            score: 10.5, // Lower for computer vision
            rank: 2,
        },
        DocumentScore {
            doc_id: "iclr2025".to_string(),
            score: 9.2,
            rank: 3,
        },
        DocumentScore {
            doc_id: "aaai2025".to_string(),
            score: 8.0,
            rank: 4,
        },
    ];

    // Vector search results: Based on semantic similarity (cosine similarity)
    let vector = vec![
        DocumentScore {
            doc_id: "neurips2025".to_string(),
            score: 0.95, // Very similar to "premier ML conference"
            rank: 0,
        },
        DocumentScore {
            doc_id: "iclr2025".to_string(),
            score: 0.93, // Also ML-focused
            rank: 1,
        },
        DocumentScore {
            doc_id: "icml2025".to_string(),
            score: 0.91,
            rank: 2,
        },
        DocumentScore {
            doc_id: "emnlp2025".to_string(),
            score: 0.85, // NLP venue, semantically related
            rank: 3,
        },
        DocumentScore {
            doc_id: "cvpr2025".to_string(),
            score: 0.72, // Less semantically similar
            rank: 4,
        },
    ];

    // Spatial search results: Based on GPS proximity (distance-based scores)
    let spatial = vec![
        DocumentScore {
            doc_id: "icml2025".to_string(),
            score: 0.99, // Very close to query location
            rank: 0,
        },
        DocumentScore {
            doc_id: "neurips2025".to_string(),
            score: 0.88, // Moderately close
            rank: 1,
        },
        DocumentScore {
            doc_id: "aaai2025".to_string(),
            score: 0.82,
            rank: 2,
        },
        DocumentScore {
            doc_id: "cvpr2025".to_string(),
            score: 0.75,
            rank: 3,
        },
        DocumentScore {
            doc_id: "kdd2025".to_string(),
            score: 0.65, // Farther away
            rank: 4,
        },
    ];

    (text, vector, spatial)
}

#[test]
fn test_weighted_fusion_balanced() {
    let (text, vector, spatial) = create_conference_venue_results();
    let fusion = MultimodalFusion::new(FusionConfig::default());

    // Balanced weights
    let weights = vec![0.33, 0.33, 0.34];
    let strategy = FusionStrategy::Weighted { weights };

    let results = fusion.fuse(&text, &vector, &spatial, Some(strategy)).unwrap();

    assert!(!results.is_empty());
    assert!(results.len() <= 8); // Union of all results

    // NeurIPS should rank highly (appears in all three with good scores)
    let neurips = results.iter().find(|r| r.uri == "neurips2025").unwrap();
    assert!(neurips.total_score > 0.5);
    assert_eq!(neurips.scores.len(), 3); // All three modalities

    // ICML should also rank highly
    let icml = results.iter().find(|r| r.uri == "icml2025").unwrap();
    assert!(icml.total_score > 0.5);
}

#[test]
fn test_weighted_fusion_text_heavy() {
    let (text, vector, spatial) = create_conference_venue_results();
    let fusion = MultimodalFusion::new(FusionConfig::default());

    // Text-heavy weights
    let weights = vec![0.7, 0.2, 0.1];
    let strategy = FusionStrategy::Weighted { weights };

    let results = fusion.fuse(&text, &vector, &spatial, Some(strategy)).unwrap();

    // Top result should be text-focused
    assert!(!results.is_empty());
    // NeurIPS or ICML should be at top (both have high text scores)
    let top_result = &results[0];
    assert!(
        top_result.uri == "neurips2025" || top_result.uri == "icml2025",
        "Top result should be text-focused"
    );
}

#[test]
fn test_sequential_fusion_text_then_vector() {
    let (text, vector, spatial) = create_conference_venue_results();
    let fusion = MultimodalFusion::new(FusionConfig::default());

    let order = vec![Modality::Text, Modality::Vector];
    let strategy = FusionStrategy::Sequential { order };

    let results = fusion.fuse(&text, &vector, &spatial, Some(strategy)).unwrap();

    assert!(!results.is_empty());

    // All results should have passed text filter
    let text_uris: Vec<_> = text.iter().map(|r| r.doc_id.as_str()).collect();
    for result in &results {
        assert!(
            text_uris.contains(&result.uri.as_str()),
            "Result {} should have passed text filter",
            result.uri
        );
    }

    // Results should be ranked by vector scores
    // NeurIPS has best vector score among text results
    let neurips_rank = results
        .iter()
        .position(|r| r.uri == "neurips2025")
        .unwrap();
    assert!(
        neurips_rank <= 2,
        "NeurIPS should rank highly in sequential fusion"
    );
}

#[test]
fn test_cascade_fusion_progressive_filtering() {
    let (text, vector, spatial) = create_conference_venue_results();
    let fusion = MultimodalFusion::new(FusionConfig::default());

    // Strict thresholds: 0.5, 0.7, 0.8
    let thresholds = vec![0.5, 0.7, 0.8];
    let strategy = FusionStrategy::Cascade { thresholds };

    let results = fusion.fuse(&text, &vector, &spatial, Some(strategy)).unwrap();

    // Should have fewer results due to progressive filtering
    assert!(results.len() <= 5);

    // All results should have high scores in all modalities
    for result in &results {
        assert!(
            result.total_score > 1.5,
            "Cascade results should have high combined scores"
        );
    }
}

#[test]
fn test_cascade_fusion_lenient_thresholds() {
    let (text, vector, spatial) = create_conference_venue_results();
    let fusion = MultimodalFusion::new(FusionConfig::default());

    // Lenient thresholds
    let thresholds = vec![0.0, 0.0, 0.0];
    let strategy = FusionStrategy::Cascade { thresholds };

    let results = fusion.fuse(&text, &vector, &spatial, Some(strategy)).unwrap();

    // Should have more results with lenient thresholds
    assert!(!results.is_empty());
}

#[test]
fn test_rank_fusion_rrf() {
    let (text, vector, spatial) = create_conference_venue_results();
    let fusion = MultimodalFusion::new(FusionConfig::default());

    let strategy = FusionStrategy::RankFusion;
    let results = fusion.fuse(&text, &vector, &spatial, Some(strategy)).unwrap();

    assert!(!results.is_empty());

    // NeurIPS appears at good positions in all three lists
    // Should have high RRF score
    let neurips = results.iter().find(|r| r.uri == "neurips2025").unwrap();

    // EMNLP appears only in vector list
    let emnlp = results
        .iter()
        .find(|r| r.uri == "emnlp2025")
        .unwrap_or_else(|| panic!("EMNLP should be in results"));

    // NeurIPS should rank higher due to multiple appearances
    let neurips_rank = results
        .iter()
        .position(|r| r.uri == "neurips2025")
        .unwrap();
    let emnlp_rank = results.iter().position(|r| r.uri == "emnlp2025").unwrap();

    assert!(
        neurips_rank < emnlp_rank,
        "NeurIPS (appears in all lists) should rank higher than EMNLP (appears in one)"
    );
}

#[test]
fn test_normalization_minmax() {
    let config = FusionConfig {
        default_strategy: FusionStrategy::RankFusion,
        score_normalization: NormalizationMethod::MinMax,
    };
    let fusion = MultimodalFusion::new(config);

    let scores = vec![
        DocumentScore {
            doc_id: "doc1".to_string(),
            score: 100.0,
            rank: 0,
        },
        DocumentScore {
            doc_id: "doc2".to_string(),
            score: 50.0,
            rank: 1,
        },
        DocumentScore {
            doc_id: "doc3".to_string(),
            score: 0.0,
            rank: 2,
        },
    ];

    let normalized = fusion.normalize_scores(&scores).unwrap();

    // Should be normalized to [0, 1]
    assert!((normalized[0] - 1.0).abs() < 1e-6, "Max should be 1.0");
    assert!((normalized[1] - 0.5).abs() < 1e-6, "Mid should be 0.5");
    assert!((normalized[2] - 0.0).abs() < 1e-6, "Min should be 0.0");
}

#[test]
fn test_normalization_zscore() {
    let config = FusionConfig {
        default_strategy: FusionStrategy::RankFusion,
        score_normalization: NormalizationMethod::ZScore,
    };
    let fusion = MultimodalFusion::new(config);

    let scores = vec![
        DocumentScore {
            doc_id: "doc1".to_string(),
            score: 10.0,
            rank: 0,
        },
        DocumentScore {
            doc_id: "doc2".to_string(),
            score: 5.0,
            rank: 1,
        },
        DocumentScore {
            doc_id: "doc3".to_string(),
            score: 0.0,
            rank: 2,
        },
    ];

    let normalized = fusion.normalize_scores(&scores).unwrap();

    // Z-scores should have mean ≈ 0
    let mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
    assert!(mean.abs() < 1e-6, "Z-score mean should be ~0");

    // Standard deviation should be ≈ 1
    let variance: f64 = normalized
        .iter()
        .map(|&s| (s - mean).powi(2))
        .sum::<f64>()
        / normalized.len() as f64;
    let std = variance.sqrt();
    assert!((std - 1.0).abs() < 1e-6, "Z-score std should be ~1");
}

#[test]
fn test_normalization_sigmoid() {
    let config = FusionConfig {
        default_strategy: FusionStrategy::RankFusion,
        score_normalization: NormalizationMethod::Sigmoid,
    };
    let fusion = MultimodalFusion::new(config);

    let scores = vec![
        DocumentScore {
            doc_id: "doc1".to_string(),
            score: 5.0,
            rank: 0,
        },
        DocumentScore {
            doc_id: "doc2".to_string(),
            score: 0.0,
            rank: 1,
        },
        DocumentScore {
            doc_id: "doc3".to_string(),
            score: -5.0,
            rank: 2,
        },
    ];

    let normalized = fusion.normalize_scores(&scores).unwrap();

    // All values should be in (0, 1)
    for &score in &normalized {
        assert!(score > 0.0 && score < 1.0, "Sigmoid should be in (0, 1)");
    }

    // Sigmoid of 0 should be 0.5
    assert!((normalized[1] - 0.5).abs() < 1e-6, "Sigmoid(0) should be 0.5");

    // Sigmoid should be monotonic
    assert!(normalized[0] > normalized[1]);
    assert!(normalized[1] > normalized[2]);
}

#[test]
fn test_precision_at_k() {
    let (text, vector, spatial) = create_conference_venue_results();
    let fusion = MultimodalFusion::new(FusionConfig::default());

    let strategy = FusionStrategy::RankFusion;
    let results = fusion.fuse(&text, &vector, &spatial, Some(strategy)).unwrap();

    // Take top 3 results
    let top_3: Vec<_> = results.iter().take(3).collect();

    // Expected top venues for ML conferences near a location
    let expected = ["neurips2025", "icml2025", "iclr2025"];

    // Count how many of the expected are in top 3
    let mut correct = 0;
    for result in &top_3 {
        if expected.contains(&result.uri.as_str()) {
            correct += 1;
        }
    }

    let precision_at_3 = correct as f64 / 3.0;

    // Should have high precision (at least 2 out of 3)
    assert!(
        precision_at_3 >= 0.66,
        "Precision@3 should be >= 0.66, got {}",
        precision_at_3
    );
}

#[test]
fn test_performance_latency() {
    use std::time::Instant;

    let (text, vector, spatial) = create_conference_venue_results();
    let fusion = MultimodalFusion::new(FusionConfig::default());

    let start = Instant::now();

    let strategy = FusionStrategy::RankFusion;
    let _results = fusion.fuse(&text, &vector, &spatial, Some(strategy)).unwrap();

    let duration = start.elapsed();

    // Should complete in less than 100ms
    assert!(
        duration.as_millis() < 100,
        "Fusion should complete in <100ms, took {}ms",
        duration.as_millis()
    );
}

#[test]
fn test_empty_modality_handling() {
    let (text, vector, _spatial) = create_conference_venue_results();
    let empty: Vec<DocumentScore> = Vec::new();
    let fusion = MultimodalFusion::new(FusionConfig::default());

    // Test with empty spatial results
    let strategy = FusionStrategy::RankFusion;
    let results = fusion.fuse(&text, &vector, &empty, Some(strategy)).unwrap();

    assert!(!results.is_empty(), "Should handle empty modality");
}

#[test]
fn test_all_empty_modalities() {
    let empty: Vec<DocumentScore> = Vec::new();
    let fusion = MultimodalFusion::new(FusionConfig::default());

    let strategy = FusionStrategy::RankFusion;
    let results = fusion
        .fuse(&empty, &empty, &empty, Some(strategy))
        .unwrap();

    assert!(results.is_empty(), "All empty should return empty");
}

#[test]
fn test_single_modality_only() {
    let (text, _vector, _spatial) = create_conference_venue_results();
    let empty: Vec<DocumentScore> = Vec::new();
    let fusion = MultimodalFusion::new(FusionConfig::default());

    let strategy = FusionStrategy::RankFusion;
    let results = fusion.fuse(&text, &empty, &empty, Some(strategy)).unwrap();

    assert!(!results.is_empty(), "Single modality should work");
    assert_eq!(
        results.len(),
        text.len(),
        "Should return all text results"
    );
}

#[test]
fn test_fused_result_operations() {
    let mut result = FusedResult::new("test_venue".to_string());

    result.add_score(Modality::Text, 0.5);
    result.add_score(Modality::Vector, 0.3);
    result.add_score(Modality::Spatial, 0.2);

    assert_eq!(result.get_score(Modality::Text), Some(0.5));
    assert_eq!(result.get_score(Modality::Vector), Some(0.3));
    assert_eq!(result.get_score(Modality::Spatial), Some(0.2));

    result.calculate_total();
    assert!((result.total_score - 1.0).abs() < 1e-6);
}

#[test]
fn test_config_update() {
    let mut fusion = MultimodalFusion::new(FusionConfig::default());

    let new_config = FusionConfig {
        default_strategy: FusionStrategy::Weighted {
            weights: vec![0.5, 0.3, 0.2],
        },
        score_normalization: NormalizationMethod::ZScore,
    };

    fusion.set_config(new_config);

    let config = fusion.config();
    assert!(matches!(
        config.default_strategy,
        FusionStrategy::Weighted { .. }
    ));
    assert!(matches!(
        config.score_normalization,
        NormalizationMethod::ZScore
    ));
}

#[test]
fn test_real_world_scenario_ml_conference_search() {
    // Scenario: User searches for "machine learning conference near San Francisco"
    let (text, vector, spatial) = create_conference_venue_results();
    let fusion = MultimodalFusion::new(FusionConfig::default());

    // Use weighted fusion with balanced weights
    let weights = vec![0.35, 0.35, 0.30]; // Text, Vector, Spatial
    let strategy = FusionStrategy::Weighted { weights };

    let results = fusion.fuse(&text, &vector, &spatial, Some(strategy)).unwrap();

    // Top result should be NeurIPS or ICML (good in all modalities)
    assert!(!results.is_empty());
    let top_result = &results[0];
    assert!(
        top_result.uri == "neurips2025" || top_result.uri == "icml2025",
        "Top result should be NeurIPS or ICML"
    );

    // Should have high combined score
    assert!(
        top_result.total_score > 0.8,
        "Top result should have high combined score"
    );
}

#[test]
fn test_score_consistency() {
    let (text, vector, spatial) = create_conference_venue_results();
    let fusion = MultimodalFusion::new(FusionConfig::default());

    let strategy = FusionStrategy::RankFusion;

    // Run multiple times - should get consistent results
    let results1 = fusion.fuse(&text, &vector, &spatial, Some(strategy.clone())).unwrap();
    let results2 = fusion.fuse(&text, &vector, &spatial, Some(strategy)).unwrap();

    assert_eq!(results1.len(), results2.len());
    for (r1, r2) in results1.iter().zip(results2.iter()) {
        assert_eq!(r1.uri, r2.uri);
        assert!((r1.total_score - r2.total_score).abs() < 1e-6);
    }
}
