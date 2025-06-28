//! Tests for pattern analysis functionality

use oxirs_core::model::{Literal, NamedNode, Triple};
use oxirs_core::Store;
use oxirs_shacl_ai::patterns::*;

#[test]
fn test_pattern_analyzer_creation() {
    let analyzer = PatternAnalyzer::new();
    assert!(analyzer.config().enable_pattern_recognition);
    assert_eq!(analyzer.config().min_support_threshold, 0.1);
    assert_eq!(analyzer.config().max_pattern_complexity, 5);
}

#[test]
fn test_custom_pattern_config() {
    let config = PatternConfig {
        enable_pattern_recognition: true,
        enable_structural_analysis: true,
        enable_usage_analysis: false,
        enable_temporal_analysis: true,
        min_support_threshold: 0.2,
        min_confidence_threshold: 0.8,
        max_pattern_complexity: 3,
        algorithms: PatternAlgorithms::default(),
        enable_training: false,
        cache_settings: PatternCacheSettings::default(),
    };

    let analyzer = PatternAnalyzer::with_config(config);
    assert_eq!(analyzer.config().min_support_threshold, 0.2);
    assert_eq!(analyzer.config().min_confidence_threshold, 0.8);
    assert_eq!(analyzer.config().max_pattern_complexity, 3);
    assert!(!analyzer.config().enable_usage_analysis);
    assert!(analyzer.config().enable_temporal_analysis);
}

#[test]
fn test_pattern_algorithms_config() {
    let algorithms = PatternAlgorithms {
        enable_frequent_itemsets: true,
        enable_association_rules: false,
        enable_graph_patterns: true,
        enable_clustering: false,
        enable_anomaly_detection: true,
        enable_sequential_patterns: false,
    };

    assert!(algorithms.enable_frequent_itemsets);
    assert!(!algorithms.enable_association_rules);
    assert!(algorithms.enable_graph_patterns);
    assert!(!algorithms.enable_clustering);
    assert!(algorithms.enable_anomaly_detection);
    assert!(!algorithms.enable_sequential_patterns);
}

#[test]
fn test_pattern_cache_settings() {
    let cache_settings = PatternCacheSettings {
        enable_caching: true,
        max_cache_size: 500,
        cache_ttl_seconds: 1800, // 30 minutes in seconds
        enable_similarity_cache: true,
    };

    assert!(cache_settings.enable_caching);
    assert_eq!(cache_settings.max_cache_size, 500);
    assert_eq!(cache_settings.cache_ttl_seconds, 1800);
    assert!(cache_settings.enable_similarity_cache);
}

#[test]
fn test_pattern_types() {
    // Test ClassUsage pattern
    let class_pattern = Pattern::ClassUsage {
        class: NamedNode::new("http://example.org/Person").unwrap(),
        instance_count: 100,
        support: 0.8,
        confidence: 0.9,
        pattern_type: PatternType::Structural,
    };

    assert_eq!(class_pattern.support(), 0.8);
    assert_eq!(class_pattern.confidence(), 0.9);
    assert_eq!(class_pattern.pattern_type(), &PatternType::Structural);

    // Test PropertyUsage pattern
    let property_pattern = Pattern::PropertyUsage {
        property: NamedNode::new("http://example.org/name").unwrap(),
        usage_count: 95,
        support: 0.95,
        confidence: 1.0,
        pattern_type: PatternType::Usage,
    };

    assert_eq!(property_pattern.support(), 0.95);
    assert_eq!(property_pattern.confidence(), 1.0);
    assert_eq!(property_pattern.pattern_type(), &PatternType::Usage);
}

#[test]
fn test_cardinality_pattern() {
    let cardinality_pattern = Pattern::Cardinality {
        property: NamedNode::new("http://example.org/name").unwrap(),
        cardinality_type: CardinalityType::Functional,
        min_count: Some(1),
        max_count: Some(1),
        avg_count: 1.0,
        support: 0.95,
        confidence: 1.0,
        pattern_type: PatternType::Usage,
    };

    assert_eq!(cardinality_pattern.support(), 0.95);
    assert_eq!(cardinality_pattern.confidence(), 1.0);

    match &cardinality_pattern {
        Pattern::Cardinality {
            cardinality_type,
            min_count,
            max_count,
            ..
        } => {
            assert_eq!(*cardinality_type, CardinalityType::Functional);
            assert_eq!(*min_count, Some(1));
            assert_eq!(*max_count, Some(1));
        }
        _ => panic!("Expected Cardinality pattern"),
    }
}

#[test]
fn test_hierarchy_pattern() {
    let hierarchy_pattern = Pattern::Hierarchy {
        subclass: NamedNode::new("http://example.org/Student").unwrap(),
        superclass: NamedNode::new("http://example.org/Person").unwrap(),
        relationship_type: HierarchyType::SubClassOf,
        depth: 1,
        support: 0.2,
        confidence: 1.0,
        pattern_type: PatternType::Structural,
    };

    assert_eq!(hierarchy_pattern.support(), 0.2);
    assert_eq!(hierarchy_pattern.confidence(), 1.0);

    match &hierarchy_pattern {
        Pattern::Hierarchy {
            relationship_type,
            depth,
            ..
        } => {
            assert_eq!(*relationship_type, HierarchyType::SubClassOf);
            assert_eq!(*depth, 1);
        }
        _ => panic!("Expected Hierarchy pattern"),
    }
}

#[test]
fn test_datatype_pattern() {
    let datatype_pattern = Pattern::Datatype {
        property: NamedNode::new("http://example.org/age").unwrap(),
        datatype: NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
        usage_count: 98,
        support: 0.95,
        confidence: 0.98,
        pattern_type: PatternType::Structural,
    };

    assert_eq!(datatype_pattern.support(), 0.95);
    assert_eq!(datatype_pattern.confidence(), 0.98);

    match &datatype_pattern {
        Pattern::Datatype { usage_count, .. } => {
            assert_eq!(*usage_count, 98);
        }
        _ => panic!("Expected Datatype pattern"),
    }
}

#[test]
fn test_constraint_usage_pattern() {
    let constraint_pattern = Pattern::ConstraintUsage {
        constraint_type: "MinCountConstraint".to_string(),
        usage_count: 45,
        support: 0.9,
        confidence: 0.95,
        pattern_type: PatternType::Usage,
    };

    assert_eq!(constraint_pattern.support(), 0.9);
    assert_eq!(constraint_pattern.confidence(), 0.95);

    match &constraint_pattern {
        Pattern::ConstraintUsage {
            constraint_type,
            usage_count,
            ..
        } => {
            assert_eq!(*constraint_type, "MinCountConstraint");
            assert_eq!(*usage_count, 45);
        }
        _ => panic!("Expected ConstraintUsage pattern"),
    }
}

#[test]
fn test_target_usage_pattern() {
    let target_pattern = Pattern::TargetUsage {
        target_type: "ClassTarget".to_string(),
        usage_count: 25,
        support: 0.9,
        confidence: 0.95,
        pattern_type: PatternType::Usage,
    };

    assert_eq!(target_pattern.support(), 0.9);
    assert_eq!(target_pattern.confidence(), 0.95);

    match &target_pattern {
        Pattern::TargetUsage {
            target_type,
            usage_count,
            ..
        } => {
            assert_eq!(*target_type, "ClassTarget");
            assert_eq!(*usage_count, 25);
        }
        _ => panic!("Expected TargetUsage pattern"),
    }
}

#[test]
fn test_cardinality_types() {
    assert_eq!(CardinalityType::Functional, CardinalityType::Functional);
    assert_ne!(
        CardinalityType::Functional,
        CardinalityType::InverseFunctional
    );
    assert_ne!(CardinalityType::Optional, CardinalityType::Required);
    assert_ne!(
        CardinalityType::Functional,
        CardinalityType::InverseFunctional
    );
}

#[test]
fn test_hierarchy_types() {
    assert_eq!(HierarchyType::SubClassOf, HierarchyType::SubClassOf);
    assert_ne!(HierarchyType::SubClassOf, HierarchyType::SubPropertyOf);
    assert_ne!(HierarchyType::EquivalentClass, HierarchyType::DisjointWith);
    assert_ne!(HierarchyType::SameAs, HierarchyType::DifferentFrom);
}

#[test]
fn test_pattern_type_enum() {
    assert_eq!(PatternType::Structural, PatternType::Structural);
    assert_ne!(PatternType::Structural, PatternType::Usage);
    assert_ne!(PatternType::Usage, PatternType::Temporal);
    assert_ne!(PatternType::Temporal, PatternType::Structural);
}

#[test]
fn test_pattern_analysis_statistics() {
    let stats = PatternStatistics {
        total_patterns_discovered: 50,
        structural_patterns: 20,
        usage_patterns: 25,
        temporal_patterns: 5,
        high_confidence_patterns: 30,
        analysis_duration_ms: 1500,
        cache_hits: 10,
        cache_misses: 40,
    };

    assert_eq!(stats.total_patterns_discovered, 50);
    assert_eq!(stats.structural_patterns, 20);
    assert_eq!(stats.usage_patterns, 25);
    assert_eq!(stats.temporal_patterns, 5);
    assert_eq!(stats.high_confidence_patterns, 30);
    assert_eq!(stats.analysis_duration_ms, 1500);
    assert_eq!(stats.cache_hits, 10);
    assert_eq!(stats.cache_misses, 40);
}

#[test]
fn test_cached_pattern_result() {
    let patterns = vec![Pattern::ClassUsage {
        class: NamedNode::new("http://example.org/Test").unwrap(),
        instance_count: 10,
        support: 0.5,
        confidence: 0.8,
        pattern_type: PatternType::Structural,
    }];

    let cached = CachedPatternResult {
        patterns: patterns.clone(),
        timestamp: chrono::Utc::now(),
        ttl: std::time::Duration::from_hours(1),
    };

    assert_eq!(cached.patterns.len(), 1);
    assert!(!cached.is_expired()); // Should not be expired immediately

    // Test expired cache
    let expired_cached = CachedPatternResult {
        patterns,
        timestamp: chrono::Utc::now() - chrono::Duration::hours(2),
        ttl: std::time::Duration::from_hours(1),
    };

    assert!(expired_cached.is_expired());
}

#[test]
fn test_empty_store_pattern_analysis() {
    let analyzer = PatternAnalyzer::new();
    let empty_store = Store::new();

    let result = analyzer.discover_patterns(&empty_store, None);

    // Should handle empty store gracefully
    match result {
        Ok(patterns) => assert!(patterns.is_empty()),
        Err(_) => {} // Error is acceptable for empty store
    }
}

#[test]
fn test_pattern_analysis_with_graph_name() {
    let analyzer = PatternAnalyzer::new();
    let store = Store::new();

    let result = analyzer.discover_patterns(&store, Some("http://example.org/graph"));

    // Should handle graph-specific analysis
    match result {
        Ok(patterns) => assert!(patterns.is_empty()), // Empty store should produce no patterns
        Err(_) => {} // Error is acceptable for empty store with graph name
    }
}
