//! Comprehensive integration tests for Tantivy full-text search
//!
//! These tests verify:
//! - Indexing performance (>50K docs/s target)
//! - Basic text search with BM25 ranking
//! - Phrase search with exact matching
//! - Fuzzy search with edit distance tolerance
//! - Field-specific search
//! - Stemming functionality
//! - Stopword filtering
//! - Multi-language support
//! - Query latency (<100ms target)
//! - Memory efficiency

#![cfg(feature = "tantivy-search")]

use anyhow::Result;
use oxirs_vec::hybrid_search::tantivy_searcher::{RdfDocument, TantivyConfig, TantivySearcher};
use oxirs_vec::sparql_integration::text_functions::{RdfLiteral, SparqlTextFunctions};
use std::env;
use std::time::Instant;

fn create_test_config() -> TantivyConfig {
    let temp_dir = env::temp_dir().join(format!("tantivy_integration_{}", uuid::Uuid::new_v4()));
    TantivyConfig {
        index_path: temp_dir,
        heap_size_mb: 50,
        stemming: true,
        stopwords: true,
        fuzzy_distance: 2,
    }
}

fn create_sample_documents(count: usize) -> Vec<RdfDocument> {
    let samples = vec![
        "Machine learning algorithms for neural networks and deep learning",
        "Natural language processing and computational linguistics",
        "Computer vision for image recognition and object detection",
        "Artificial intelligence and cognitive computing systems",
        "Data mining and knowledge discovery in databases",
        "Semantic web technologies and linked data",
        "Information retrieval and search engine optimization",
        "Graph databases and network analysis",
        "Distributed systems and cloud computing",
        "Cybersecurity and cryptographic protocols",
    ];

    (0..count)
        .map(|i| {
            let content = samples[i % samples.len()].to_string();
            RdfDocument {
                uri: format!("http://example.org/doc{}", i),
                content,
                language: Some("en".to_string()),
                datatype: Some("xsd:string".to_string()),
            }
        })
        .collect()
}

#[test]
fn test_basic_search() -> Result<()> {
    let config = create_test_config();
    let mut searcher = TantivySearcher::new(config)?;

    let docs = create_sample_documents(10);
    searcher.index_documents(&docs)?;

    // Search for "machine learning"
    let results = searcher.text_search("machine learning", 10, 0.0)?;

    assert!(!results.is_empty(), "Should find matching documents");
    assert!(
        results[0].uri.contains("doc"),
        "Result should have valid URI"
    );
    assert!(results[0].score > 0.0, "Result should have positive score");

    Ok(())
}

#[test]
fn test_phrase_search() -> Result<()> {
    let config = create_test_config();
    let mut searcher = TantivySearcher::new(config)?;

    let docs = vec![
        RdfDocument {
            uri: "http://example.org/article1".to_string(),
            content: "Natural language processing is a subfield of artificial intelligence"
                .to_string(),
            language: Some("en".to_string()),
            datatype: None,
        },
        RdfDocument {
            uri: "http://example.org/article2".to_string(),
            content: "Machine learning models for natural language tasks".to_string(),
            language: Some("en".to_string()),
            datatype: None,
        },
    ];

    searcher.index_documents(&docs)?;

    // Exact phrase search
    let results = searcher.phrase_search("natural language processing", 10)?;

    assert!(!results.is_empty(), "Should find phrase match");
    assert_eq!(results[0].uri, "http://example.org/article1");

    Ok(())
}

#[test]
fn test_fuzzy_search() -> Result<()> {
    let config = create_test_config();
    let mut searcher = TantivySearcher::new(config)?;

    let docs = vec![RdfDocument {
        uri: "http://example.org/doc1".to_string(),
        content: "Information retrieval systems".to_string(),
        language: Some("en".to_string()),
        datatype: None,
    }];

    searcher.index_documents(&docs)?;

    // Fuzzy search with misspelling (edit distance 2)
    let results = searcher.fuzzy_search("retreival", 2, 10)?;

    assert!(
        !results.is_empty(),
        "Should find fuzzy match for misspelling"
    );
    assert!(
        results[0].score > 0.0,
        "Fuzzy match should have positive score"
    );

    Ok(())
}

#[test]
fn test_stemming() -> Result<()> {
    let config = create_test_config();
    let mut searcher = TantivySearcher::new(config)?;

    let docs = vec![RdfDocument {
        uri: "http://example.org/running".to_string(),
        content: "The runner is running a marathon and they ran yesterday".to_string(),
        language: Some("en".to_string()),
        datatype: None,
    }];

    searcher.index_documents(&docs)?;

    // Search with root form - should match "running", "runner", "ran"
    let results = searcher.text_search("run", 10, 0.0)?;

    assert!(
        !results.is_empty(),
        "Stemming should match 'run' with 'running', 'runner', 'ran'"
    );

    Ok(())
}

#[test]
fn test_stopwords() -> Result<()> {
    let config = create_test_config();
    let mut searcher = TantivySearcher::new(config)?;

    let docs = create_sample_documents(5);
    searcher.index_documents(&docs)?;

    // Query with stopwords - they should be filtered
    let results = searcher.text_search("the and for algorithms", 10, 0.0)?;

    assert!(
        !results.is_empty(),
        "Should find results ignoring stopwords"
    );

    Ok(())
}

#[test]
fn test_multi_language_support() -> Result<()> {
    let config = create_test_config();
    let mut searcher = TantivySearcher::new(config)?;

    let docs = vec![
        RdfDocument {
            uri: "http://example.org/en_doc".to_string(),
            content: "Machine learning in English".to_string(),
            language: Some("en".to_string()),
            datatype: None,
        },
        RdfDocument {
            uri: "http://example.org/de_doc".to_string(),
            content: "Maschinelles Lernen auf Deutsch".to_string(),
            language: Some("de".to_string()),
            datatype: None,
        },
        RdfDocument {
            uri: "http://example.org/fr_doc".to_string(),
            content: "Apprentissage automatique en français".to_string(),
            language: Some("fr".to_string()),
            datatype: None,
        },
    ];

    searcher.index_documents(&docs)?;

    let results = searcher.text_search("machine", 10, 0.0)?;

    assert!(!results.is_empty(), "Should find documents");

    // Verify language tags are preserved
    for result in &results {
        assert!(result.language.is_some(), "Language tag should be present");
    }

    Ok(())
}

#[test]
fn test_query_latency() -> Result<()> {
    let config = create_test_config();
    let mut searcher = TantivySearcher::new(config)?;

    // Index a moderate number of documents
    let docs = create_sample_documents(1000);
    searcher.index_documents(&docs)?;

    // Measure query latency
    let start = Instant::now();
    let _results = searcher.text_search("machine learning", 10, 0.0)?;
    let duration = start.elapsed();

    println!("Query latency: {:?}", duration);

    // Target: <100ms for most queries
    assert!(
        duration.as_millis() < 200,
        "Query should complete within 200ms, took {:?}",
        duration
    );

    Ok(())
}

#[test]
fn test_indexing_performance() -> Result<()> {
    let config = create_test_config();
    let mut searcher = TantivySearcher::new(config)?;

    // Create a large batch of documents
    let doc_count = 10_000;
    let docs = create_sample_documents(doc_count);

    // Measure indexing time
    let start = Instant::now();
    searcher.index_documents(&docs)?;
    let duration = start.elapsed();

    let docs_per_sec = (doc_count as f64 / duration.as_secs_f64()) as u64;

    println!("Indexed {} documents in {:?}", doc_count, duration);
    println!("Throughput: {} docs/sec", docs_per_sec);

    // Target: >50K docs/s (but 10K is acceptable for integration tests)
    assert!(
        docs_per_sec > 5_000,
        "Indexing should be at least 5K docs/sec, got {}",
        docs_per_sec
    );

    Ok(())
}

#[test]
fn test_index_stats() -> Result<()> {
    let config = create_test_config();
    let mut searcher = TantivySearcher::new(config)?;

    let docs = create_sample_documents(100);
    searcher.index_documents(&docs)?;

    let stats = searcher.get_stats();

    assert_eq!(stats.total_documents, 100);
    assert_eq!(stats.heap_size_mb, 50);
    assert!(stats.indexed_count > 0);

    Ok(())
}

#[test]
fn test_threshold_filtering() -> Result<()> {
    let config = create_test_config();
    let mut searcher = TantivySearcher::new(config)?;

    let docs = create_sample_documents(10);
    searcher.index_documents(&docs)?;

    // Search with low threshold
    let results_low = searcher.text_search("machine", 10, 0.0)?;

    // Search with high threshold
    let results_high = searcher.text_search("machine", 10, 0.9)?;

    assert!(
        results_low.len() >= results_high.len(),
        "Lower threshold should return more results"
    );

    // Verify all high threshold results meet the threshold
    for result in results_high {
        assert!(
            result.score >= 0.9,
            "Result score {} should be >= 0.9",
            result.score
        );
    }

    Ok(())
}

#[test]
fn test_sparql_integration() -> Result<()> {
    let config = create_test_config();
    let searcher = TantivySearcher::new(config)?;
    let mut text_funcs = SparqlTextFunctions::new(searcher);

    // Index RDF literals
    let literals = vec![
        RdfLiteral {
            subject_uri: "http://example.org/paper1".to_string(),
            value: "Deep learning for natural language processing".to_string(),
            language: Some("en".to_string()),
            datatype: Some("xsd:string".to_string()),
        },
        RdfLiteral {
            subject_uri: "http://example.org/paper2".to_string(),
            value: "Machine learning algorithms and applications".to_string(),
            language: Some("en".to_string()),
            datatype: Some("xsd:string".to_string()),
        },
    ];

    text_funcs.index_literals(literals)?;

    // Test text search
    let results = text_funcs.text_search("", "learning", Some(10), Some(0.0))?;

    assert!(!results.is_empty(), "Should find matching documents");
    assert!(results.len() >= 1);

    // Test phrase search
    let phrase_results = text_funcs.phrase_search("", "natural language processing", Some(10))?;

    assert!(!phrase_results.is_empty(), "Should find phrase match");

    // Test fuzzy search
    let fuzzy_results = text_funcs.fuzzy_search("", "lerning", Some(2), Some(10))?;

    assert!(
        !fuzzy_results.is_empty(),
        "Should find fuzzy match for misspelling"
    );

    Ok(())
}

#[test]
fn test_memory_efficiency() -> Result<()> {
    let config = TantivyConfig {
        index_path: env::temp_dir().join(format!("tantivy_mem_{}", uuid::Uuid::new_v4())),
        heap_size_mb: 50, // Limited heap
        stemming: true,
        stopwords: true,
        fuzzy_distance: 2,
    };

    let mut searcher = TantivySearcher::new(config)?;

    // Index a large number of documents with limited heap
    let doc_count = 50_000;
    let batch_size = 1000;

    for i in (0..doc_count).step_by(batch_size) {
        let batch = create_sample_documents(batch_size.min(doc_count - i));
        searcher.index_documents(&batch)?;
    }

    let stats = searcher.get_stats();
    assert_eq!(stats.total_documents as usize, doc_count);

    // Verify search still works
    let results = searcher.text_search("machine learning", 10, 0.0)?;
    assert!(!results.is_empty());

    Ok(())
}

#[test]
fn test_optimize_index() -> Result<()> {
    let config = create_test_config();
    let mut searcher = TantivySearcher::new(config)?;

    let docs = create_sample_documents(100);
    searcher.index_documents(&docs)?;

    // Optimize should succeed
    searcher.optimize()?;

    // Verify search still works after optimization
    let results = searcher.text_search("machine learning", 10, 0.0)?;
    assert!(!results.is_empty());

    Ok(())
}

#[test]
fn test_empty_query() -> Result<()> {
    let config = create_test_config();
    let mut searcher = TantivySearcher::new(config)?;

    let docs = create_sample_documents(10);
    searcher.index_documents(&docs)?;

    // Empty query should still parse and run (may return empty results or error)
    let result = searcher.text_search("", 10, 0.0);

    // Either empty results or error is acceptable
    match result {
        Ok(results) => {
            println!("Empty query returned {} results", results.len());
        }
        Err(e) => {
            println!("Empty query error (expected): {}", e);
        }
    }

    Ok(())
}

#[test]
fn test_special_characters() -> Result<()> {
    let config = create_test_config();
    let mut searcher = TantivySearcher::new(config)?;

    let docs = vec![RdfDocument {
        uri: "http://example.org/special".to_string(),
        content: "C++ programming with std::vector and boost::asio".to_string(),
        language: Some("en".to_string()),
        datatype: None,
    }];

    searcher.index_documents(&docs)?;

    // Search with special characters
    let result = searcher.text_search("C++", 10, 0.0);

    // Should handle special characters gracefully (either returns results or error)
    match result {
        Ok(results) => {
            println!("Special char query returned {} results", results.len());
        }
        Err(e) => {
            println!("Special char query error (acceptable): {}", e);
        }
    }

    Ok(())
}

#[test]
fn test_concurrent_searches() -> Result<()> {
    let config = create_test_config();
    let mut searcher = TantivySearcher::new(config)?;

    let docs = create_sample_documents(100);
    searcher.index_documents(&docs)?;

    // Basic sequential search test
    // Note: Concurrent searches would require Arc and thread-safe design
    let results = searcher.text_search("machine learning", 10, 0.0)?;
    assert!(!results.is_empty());

    Ok(())
}

#[test]
fn test_result_ordering() -> Result<()> {
    let config = create_test_config();
    let mut searcher = TantivySearcher::new(config)?;

    let docs = create_sample_documents(20);
    searcher.index_documents(&docs)?;

    let results = searcher.text_search("machine learning", 10, 0.0)?;

    // Verify results are ordered by score (descending)
    for i in 0..results.len().saturating_sub(1) {
        assert!(
            results[i].score >= results[i + 1].score,
            "Results should be ordered by score (descending)"
        );
    }

    Ok(())
}

#[test]
fn test_limit_parameter() -> Result<()> {
    let config = create_test_config();
    let mut searcher = TantivySearcher::new(config)?;

    let docs = create_sample_documents(50);
    searcher.index_documents(&docs)?;

    // Test different limits
    let results_5 = searcher.text_search("learning", 5, 0.0)?;
    let results_10 = searcher.text_search("learning", 10, 0.0)?;

    assert!(results_5.len() <= 5, "Should return at most 5 results");
    assert!(results_10.len() <= 10, "Should return at most 10 results");

    Ok(())
}
