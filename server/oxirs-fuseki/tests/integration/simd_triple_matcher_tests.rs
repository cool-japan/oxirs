//! Integration tests for SIMD-accelerated triple pattern matching
//!
//! These tests verify the correctness and performance of the SIMD triple matcher
//! with realistic workloads and edge cases.

use oxirs_fuseki::simd_triple_matcher::{SimdTripleMatcher, Triple, TriplePattern};

#[test]
fn test_empty_matcher() {
    let matcher = SimdTripleMatcher::new();
    let stats = matcher.get_statistics();

    assert_eq!(stats.total_triples, 0);
    assert_eq!(stats.total_matches, 0);
}

#[test]
fn test_single_triple_exact_match() {
    let mut matcher = SimdTripleMatcher::new();

    let triple = Triple::new(
        "http://example.org/Alice".to_string(),
        "http://xmlns.com/foaf/0.1/knows".to_string(),
        "http://example.org/Bob".to_string(),
    );

    matcher.add_triple(triple.clone());

    let pattern = TriplePattern {
        subject: Some("http://example.org/Alice".to_string()),
        predicate: Some("http://xmlns.com/foaf/0.1/knows".to_string()),
        object: Some("http://example.org/Bob".to_string()),
    };

    let results = matcher.match_pattern(&pattern).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].subject, "http://example.org/Alice");
}

#[test]
fn test_wildcard_subject() {
    let mut matcher = SimdTripleMatcher::new();

    matcher.add_triple(Triple::new(
        "http://example.org/Alice".to_string(),
        "http://xmlns.com/foaf/0.1/knows".to_string(),
        "http://example.org/Bob".to_string(),
    ));

    matcher.add_triple(Triple::new(
        "http://example.org/Charlie".to_string(),
        "http://xmlns.com/foaf/0.1/knows".to_string(),
        "http://example.org/Bob".to_string(),
    ));

    let pattern = TriplePattern {
        subject: None, // Wildcard
        predicate: Some("http://xmlns.com/foaf/0.1/knows".to_string()),
        object: Some("http://example.org/Bob".to_string()),
    };

    let results = matcher.match_pattern(&pattern).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn test_wildcard_predicate() {
    let mut matcher = SimdTripleMatcher::new();

    matcher.add_triple(Triple::new(
        "http://example.org/Alice".to_string(),
        "http://xmlns.com/foaf/0.1/knows".to_string(),
        "http://example.org/Bob".to_string(),
    ));

    matcher.add_triple(Triple::new(
        "http://example.org/Alice".to_string(),
        "http://xmlns.com/foaf/0.1/name".to_string(),
        "Alice".to_string(),
    ));

    let pattern = TriplePattern {
        subject: Some("http://example.org/Alice".to_string()),
        predicate: None, // Wildcard
        object: None,
    };

    let results = matcher.match_pattern(&pattern).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn test_wildcard_object() {
    let mut matcher = SimdTripleMatcher::new();

    matcher.add_triple(Triple::new(
        "http://example.org/Alice".to_string(),
        "http://xmlns.com/foaf/0.1/knows".to_string(),
        "http://example.org/Bob".to_string(),
    ));

    matcher.add_triple(Triple::new(
        "http://example.org/Alice".to_string(),
        "http://xmlns.com/foaf/0.1/knows".to_string(),
        "http://example.org/Charlie".to_string(),
    ));

    let pattern = TriplePattern {
        subject: Some("http://example.org/Alice".to_string()),
        predicate: Some("http://xmlns.com/foaf/0.1/knows".to_string()),
        object: None, // Wildcard
    };

    let results = matcher.match_pattern(&pattern).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn test_all_wildcards() {
    let mut matcher = SimdTripleMatcher::new();

    for i in 0..10 {
        matcher.add_triple(Triple::new(
            format!("http://example.org/s{}", i),
            format!("http://example.org/p{}", i),
            format!("http://example.org/o{}", i),
        ));
    }

    let pattern = TriplePattern {
        subject: None,
        predicate: None,
        object: None,
    };

    let results = matcher.match_pattern(&pattern).unwrap();
    assert_eq!(results.len(), 10);
}

#[test]
fn test_no_match() {
    let mut matcher = SimdTripleMatcher::new();

    matcher.add_triple(Triple::new(
        "http://example.org/Alice".to_string(),
        "http://xmlns.com/foaf/0.1/knows".to_string(),
        "http://example.org/Bob".to_string(),
    ));

    let pattern = TriplePattern {
        subject: Some("http://example.org/NonExistent".to_string()),
        predicate: None,
        object: None,
    };

    let results = matcher.match_pattern(&pattern).unwrap();
    assert_eq!(results.len(), 0);
}

#[test]
fn test_large_dataset_fallback() {
    // Test with <32 triples (should use fallback matching)
    let mut matcher = SimdTripleMatcher::new();

    for i in 0..20 {
        matcher.add_triple(Triple::new(
            format!("http://example.org/s{}", i),
            "http://example.org/common_predicate".to_string(),
            format!("http://example.org/o{}", i),
        ));
    }

    let pattern = TriplePattern {
        subject: None,
        predicate: Some("http://example.org/common_predicate".to_string()),
        object: None,
    };

    let results = matcher.match_pattern(&pattern).unwrap();
    assert_eq!(results.len(), 20);

    let stats = matcher.get_statistics();
    assert_eq!(stats.fallback_matches, 1);
    assert_eq!(stats.simd_accelerated_matches, 0);
}

#[test]
fn test_large_dataset_simd() {
    // Test with >=32 triples (should use SIMD acceleration)
    let mut matcher = SimdTripleMatcher::new();

    for i in 0..100 {
        matcher.add_triple(Triple::new(
            format!("http://example.org/s{}", i),
            "http://example.org/common_predicate".to_string(),
            format!("http://example.org/o{}", i),
        ));
    }

    let pattern = TriplePattern {
        subject: None,
        predicate: Some("http://example.org/common_predicate".to_string()),
        object: None,
    };

    let results = matcher.match_pattern(&pattern).unwrap();
    assert_eq!(results.len(), 100);

    let stats = matcher.get_statistics();
    assert_eq!(stats.simd_accelerated_matches, 1);
}

#[test]
fn test_batch_add_triples() {
    let mut matcher = SimdTripleMatcher::new();

    let triples = (0..50)
        .map(|i| {
            Triple::new(
                format!("s{}", i),
                format!("p{}", i % 5),
                format!("o{}", i),
            )
        })
        .collect();

    matcher.add_triples(triples);

    let stats = matcher.get_statistics();
    assert_eq!(stats.total_triples, 50);
}

#[test]
fn test_index_building() {
    let mut matcher = SimdTripleMatcher::new();

    for i in 0..100 {
        matcher.add_triple(Triple::new(
            format!("s{}", i % 10),
            format!("p{}", i % 5),
            format!("o{}", i),
        ));
    }

    let stats = matcher.get_statistics();
    // With 100 triples distributed across 10 subjects and 5 predicates
    assert!(stats.index_sizes.subject_index_size <= 10);
    assert!(stats.index_sizes.predicate_index_size <= 5);
    assert_eq!(stats.index_sizes.object_index_size, 100); // All unique objects
}

#[test]
fn test_matcher_clear() {
    let mut matcher = SimdTripleMatcher::new();

    for i in 0..50 {
        matcher.add_triple(Triple::new(
            format!("s{}", i),
            "p".to_string(),
            format!("o{}", i),
        ));
    }

    assert_eq!(matcher.get_statistics().total_triples, 50);

    matcher.clear();

    let stats = matcher.get_statistics();
    assert_eq!(stats.total_triples, 0);
    assert_eq!(stats.total_matches, 0);
    assert_eq!(stats.index_sizes.subject_index_size, 0);
}

#[test]
fn test_hash_collision_handling() {
    // Test that hash collisions are handled correctly by string comparison
    let mut matcher = SimdTripleMatcher::new();

    // Add triples that might have hash collisions
    matcher.add_triple(Triple::new("AA".to_string(), "p".to_string(), "o1".to_string()));
    matcher.add_triple(Triple::new("BB".to_string(), "p".to_string(), "o2".to_string()));

    // Match specific subject
    let pattern = TriplePattern {
        subject: Some("AA".to_string()),
        predicate: Some("p".to_string()),
        object: None,
    };

    let results = matcher.match_pattern(&pattern).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].subject, "AA");
}

#[test]
fn test_performance_metrics_tracking() {
    let mut matcher = SimdTripleMatcher::new();

    for i in 0..100 {
        matcher.add_triple(Triple::new(
            format!("s{}", i),
            "p".to_string(),
            format!("o{}", i),
        ));
    }

    // Perform multiple matches
    for _ in 0..5 {
        let pattern = TriplePattern {
            subject: None,
            predicate: Some("p".to_string()),
            object: None,
        };
        matcher.match_pattern(&pattern).unwrap();
    }

    let stats = matcher.get_statistics();
    assert_eq!(stats.total_matches, 5);
}

#[test]
fn test_realistic_foaf_dataset() {
    let mut matcher = SimdTripleMatcher::new();

    // Create a realistic FOAF social network graph
    let people = vec!["Alice", "Bob", "Charlie", "David", "Eve"];
    let foaf_ns = "http://xmlns.com/foaf/0.1/";

    // Add name triples
    for person in &people {
        matcher.add_triple(Triple::new(
            format!("http://example.org/{}", person),
            format!("{}name", foaf_ns),
            person.to_string(),
        ));
    }

    // Add knows relationships
    matcher.add_triple(Triple::new(
        "http://example.org/Alice".to_string(),
        format!("{}knows", foaf_ns),
        "http://example.org/Bob".to_string(),
    ));

    matcher.add_triple(Triple::new(
        "http://example.org/Alice".to_string(),
        format!("{}knows", foaf_ns),
        "http://example.org/Charlie".to_string(),
    ));

    matcher.add_triple(Triple::new(
        "http://example.org/Bob".to_string(),
        format!("{}knows", foaf_ns),
        "http://example.org/David".to_string(),
    ));

    // Query: Find all people Alice knows
    let pattern = TriplePattern {
        subject: Some("http://example.org/Alice".to_string()),
        predicate: Some(format!("{}knows", foaf_ns)),
        object: None,
    };

    let results = matcher.match_pattern(&pattern).unwrap();
    assert_eq!(results.len(), 2);

    // Query: Find all names
    let pattern = TriplePattern {
        subject: None,
        predicate: Some(format!("{}name", foaf_ns)),
        object: None,
    };

    let results = matcher.match_pattern(&pattern).unwrap();
    assert_eq!(results.len(), 5);
}

#[test]
fn test_unicode_support() {
    let mut matcher = SimdTripleMatcher::new();

    matcher.add_triple(Triple::new(
        "http://example.org/東京".to_string(),
        "http://www.w3.org/2000/01/rdf-schema#label".to_string(),
        "東京".to_string(),
    ));

    matcher.add_triple(Triple::new(
        "http://example.org/パリ".to_string(),
        "http://www.w3.org/2000/01/rdf-schema#label".to_string(),
        "パリ".to_string(),
    ));

    let pattern = TriplePattern {
        subject: Some("http://example.org/東京".to_string()),
        predicate: None,
        object: None,
    };

    let results = matcher.match_pattern(&pattern).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].object, "東京");
}

#[test]
fn test_statistics_accuracy() {
    let mut matcher = SimdTripleMatcher::new();

    // Add 50 triples
    for i in 0..50 {
        matcher.add_triple(Triple::new(
            format!("s{}", i % 10),
            format!("p{}", i % 5),
            format!("o{}", i),
        ));
    }

    let stats = matcher.get_statistics();

    assert_eq!(stats.total_triples, 50);
    assert!(stats.index_sizes.subject_index_size <= 10);
    assert!(stats.index_sizes.predicate_index_size <= 5);
    assert!(stats.index_sizes.object_index_size <= 50);
}
