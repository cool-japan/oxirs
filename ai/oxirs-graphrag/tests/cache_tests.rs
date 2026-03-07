//! Tests for adaptive TTL caching

use oxirs_graphrag::{
    CacheConfiguration, EmbeddingModelTrait, GraphRAGConfig, GraphRAGEngine, GraphRAGResult,
    LlmClientTrait, SparqlEngineTrait, Triple, VectorIndexTrait,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

// Mock implementations for testing

struct MockVectorIndex {
    results: Vec<(String, f32)>,
}

#[async_trait::async_trait]
impl VectorIndexTrait for MockVectorIndex {
    async fn search_knn(
        &self,
        _query_vector: &[f32],
        _k: usize,
    ) -> GraphRAGResult<Vec<(String, f32)>> {
        Ok(self.results.clone())
    }

    async fn search_threshold(
        &self,
        _query_vector: &[f32],
        _threshold: f32,
    ) -> GraphRAGResult<Vec<(String, f32)>> {
        Ok(self.results.clone())
    }
}

struct MockEmbeddingModel;

#[async_trait::async_trait]
impl EmbeddingModelTrait for MockEmbeddingModel {
    async fn embed(&self, _text: &str) -> GraphRAGResult<Vec<f32>> {
        Ok(vec![0.1; 384])
    }

    async fn embed_batch(&self, _texts: &[&str]) -> GraphRAGResult<Vec<Vec<f32>>> {
        Ok(vec![vec![0.1; 384]])
    }
}

struct MockSparqlEngine {
    triples: Vec<Triple>,
}

#[async_trait::async_trait]
impl SparqlEngineTrait for MockSparqlEngine {
    async fn select(&self, _query: &str) -> GraphRAGResult<Vec<HashMap<String, String>>> {
        Ok(vec![])
    }

    async fn ask(&self, _query: &str) -> GraphRAGResult<bool> {
        Ok(false)
    }

    async fn construct(&self, _query: &str) -> GraphRAGResult<Vec<Triple>> {
        Ok(self.triples.clone())
    }
}

struct MockLlmClient;

#[async_trait::async_trait]
impl LlmClientTrait for MockLlmClient {
    async fn generate(&self, _context: &str, _query: &str) -> GraphRAGResult<String> {
        Ok("Mock answer".to_string())
    }

    async fn generate_stream(
        &self,
        _context: &str,
        _query: &str,
        _callback: Box<dyn Fn(&str) + Send + Sync>,
    ) -> GraphRAGResult<String> {
        Ok("Mock answer".to_string())
    }
}

fn create_mock_engine(
) -> GraphRAGEngine<MockVectorIndex, MockEmbeddingModel, MockSparqlEngine, MockLlmClient> {
    let vec_index = Arc::new(MockVectorIndex {
        results: vec![
            ("http://entity1".to_string(), 0.9),
            ("http://entity2".to_string(), 0.8),
        ],
    });

    let embedding_model = Arc::new(MockEmbeddingModel);

    let sparql_engine = Arc::new(MockSparqlEngine {
        triples: vec![
            Triple::new("http://entity1", "http://rel", "http://entity2"),
            Triple::new("http://entity2", "http://rel", "http://entity3"),
        ],
    });

    let llm_client = Arc::new(MockLlmClient);

    let config = GraphRAGConfig {
        cache_size: Some(100),
        enable_communities: false,
        ..Default::default()
    };

    GraphRAGEngine::new(
        vec_index,
        embedding_model,
        sparql_engine,
        llm_client,
        config,
    )
}

/// Test adaptive TTL calculation
#[tokio::test]
async fn test_adaptive_ttl() {
    let engine = create_mock_engine();

    // Simulate low update rate
    let result1 = engine.query("test query").await;
    assert!(result1.is_ok());

    // Record many updates
    for _ in 0..150 {
        engine.record_graph_update();
    }

    // Cache should now have shorter TTL due to high update rate
    let result2 = engine.query("test query 2").await;
    assert!(result2.is_ok());
}

/// Test cache hit rate
#[tokio::test]
async fn test_cache_hit_rate() {
    let engine = create_mock_engine();

    // First query - cache miss
    let result1 = engine.query("test query").await;
    assert!(result1.is_ok());

    // Second same query - should be cache hit
    let result2 = engine.query("test query").await;
    assert!(result2.is_ok());

    // Check cache stats
    let (used, capacity) = engine.get_cache_stats().await;
    assert_eq!(used, 1); // One unique query cached
    assert_eq!(capacity, 100);
}

/// Test cache with different queries
#[tokio::test]
async fn test_cache_multiple_queries() {
    let engine = create_mock_engine();

    // Query multiple different queries
    for i in 0..10 {
        let query = format!("test query {}", i);
        let result = engine.query(&query).await;
        assert!(result.is_ok());
    }

    // Check cache stats
    let (used, capacity) = engine.get_cache_stats().await;
    assert_eq!(used, 10); // Ten unique queries cached
    assert_eq!(capacity, 100);
}

/// Test cache eviction (LRU)
#[tokio::test]
async fn test_cache_eviction() {
    let vec_index = Arc::new(MockVectorIndex {
        results: vec![("http://entity1".to_string(), 0.9)],
    });
    let embedding_model = Arc::new(MockEmbeddingModel);
    let sparql_engine = Arc::new(MockSparqlEngine {
        triples: vec![Triple::new("http://e1", "http://r", "http://e2")],
    });
    let llm_client = Arc::new(MockLlmClient);

    let config = GraphRAGConfig {
        cache_size: Some(5), // Small cache
        enable_communities: false,
        ..Default::default()
    };

    let engine = GraphRAGEngine::new(
        vec_index,
        embedding_model,
        sparql_engine,
        llm_client,
        config,
    );

    // Fill cache beyond capacity
    for i in 0..10 {
        let query = format!("query {}", i);
        let result = engine.query(&query).await;
        assert!(result.is_ok());
    }

    // Cache should be at capacity
    let (used, capacity) = engine.get_cache_stats().await;
    assert_eq!(used, 5);
    assert_eq!(capacity, 5);
}

/// Test cache configuration from config
#[test]
fn test_cache_configuration() {
    let config = CacheConfiguration {
        base_ttl_seconds: 1800,
        min_ttl_seconds: 600,
        max_ttl_seconds: 7200,
        adaptive: true,
    };

    assert_eq!(config.base_ttl_seconds, 1800);
    assert_eq!(config.min_ttl_seconds, 600);
    assert_eq!(config.max_ttl_seconds, 7200);
    assert!(config.adaptive);
}

/// Test default cache configuration
#[test]
fn test_default_cache_configuration() {
    let config = CacheConfiguration::default();

    assert_eq!(config.base_ttl_seconds, 3600); // 1 hour
    assert_eq!(config.min_ttl_seconds, 300); // 5 minutes
    assert_eq!(config.max_ttl_seconds, 86400); // 24 hours
    assert!(config.adaptive);
}
