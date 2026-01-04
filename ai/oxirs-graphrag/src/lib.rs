//! # OxiRS GraphRAG
//!
//! GraphRAG (Graph Retrieval-Augmented Generation) combines vector similarity search
//! with graph topology traversal for enhanced knowledge retrieval.
//!
//! ## Architecture
//!
//! ```text
//! Query → Embed → Vector KNN + Keyword Search → Fusion → Graph Expansion → LLM Answer
//! ```
//!
//! ## Key Features
//!
//! - **Hybrid Retrieval**: Vector similarity + BM25 keyword search
//! - **Graph Expansion**: SPARQL-based N-hop neighbor traversal
//! - **Community Detection**: Louvain algorithm for hierarchical summarization
//! - **Context Building**: Intelligent subgraph extraction for LLM context
//!
//! ## Example
//!
//! ```rust,ignore
//! use oxirs_graphrag::{GraphRAGEngine, GraphRAGConfig};
//!
//! let engine = GraphRAGEngine::new(config).await?;
//! let result = engine.query("What safety issues affect battery cells?").await?;
//! println!("Answer: {}", result.answer);
//! ```

pub mod config;
pub mod generation;
pub mod graph;
pub mod query;
pub mod retrieval;
pub mod sparql;

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock;

// Re-exports
pub use config::GraphRAGConfig;
pub use graph::community::CommunityDetector;
pub use graph::traversal::GraphTraversal;
pub use query::planner::QueryPlanner;
pub use retrieval::fusion::FusionStrategy;

/// GraphRAG error types
#[derive(Error, Debug)]
pub enum GraphRAGError {
    #[error("Vector search failed: {0}")]
    VectorSearchError(String),

    #[error("Graph traversal failed: {0}")]
    GraphTraversalError(String),

    #[error("Community detection failed: {0}")]
    CommunityDetectionError(String),

    #[error("LLM generation failed: {0}")]
    GenerationError(String),

    #[error("Embedding failed: {0}")]
    EmbeddingError(String),

    #[error("SPARQL query failed: {0}")]
    SparqlError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Internal error: {0}")]
    InternalError(String),
}

pub type GraphRAGResult<T> = Result<T, GraphRAGError>;

/// Triple representation for RDF data
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl Triple {
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }
}

/// Entity with relevance score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredEntity {
    /// Entity URI
    pub uri: String,
    /// Relevance score (0.0 - 1.0)
    pub score: f64,
    /// Source of the score (vector, keyword, or fused)
    pub source: ScoreSource,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Source of entity score
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ScoreSource {
    /// Score from vector similarity search
    Vector,
    /// Score from keyword/BM25 search
    Keyword,
    /// Fused score from multiple sources
    Fused,
    /// Score from graph traversal (path-based)
    Graph,
}

/// Community summary for hierarchical retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunitySummary {
    /// Community identifier
    pub id: String,
    /// Human-readable summary of the community
    pub summary: String,
    /// Member entities in this community
    pub entities: Vec<String>,
    /// Representative triples from this community
    pub representative_triples: Vec<Triple>,
    /// Community level in hierarchy (0 = leaf, higher = more abstract)
    pub level: u32,
    /// Modularity score
    pub modularity: f64,
}

/// Query provenance for attribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryProvenance {
    /// Query timestamp
    pub timestamp: DateTime<Utc>,
    /// Original query text
    pub original_query: String,
    /// Expanded query (if any)
    pub expanded_query: Option<String>,
    /// Seed entities used
    pub seed_entities: Vec<String>,
    /// Triples contributing to the answer
    pub source_triples: Vec<Triple>,
    /// Community summaries used (if hierarchical)
    pub community_sources: Vec<String>,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

/// GraphRAG query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRAGResult2 {
    /// Natural language answer
    pub answer: String,
    /// Source subgraph (RDF triples)
    pub subgraph: Vec<Triple>,
    /// Seed entities with scores
    pub seeds: Vec<ScoredEntity>,
    /// Community summaries (if enabled)
    pub communities: Vec<CommunitySummary>,
    /// Provenance information
    pub provenance: QueryProvenance,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
}

/// Trait for vector index operations
#[async_trait]
pub trait VectorIndexTrait: Send + Sync {
    /// Search for k nearest neighbors
    async fn search_knn(
        &self,
        query_vector: &[f32],
        k: usize,
    ) -> GraphRAGResult<Vec<(String, f32)>>;

    /// Search with similarity threshold
    async fn search_threshold(
        &self,
        query_vector: &[f32],
        threshold: f32,
    ) -> GraphRAGResult<Vec<(String, f32)>>;
}

/// Trait for embedding model operations
#[async_trait]
pub trait EmbeddingModelTrait: Send + Sync {
    /// Embed text into vector
    async fn embed(&self, text: &str) -> GraphRAGResult<Vec<f32>>;

    /// Embed multiple texts in batch
    async fn embed_batch(&self, texts: &[&str]) -> GraphRAGResult<Vec<Vec<f32>>>;
}

/// Trait for SPARQL engine operations
#[async_trait]
pub trait SparqlEngineTrait: Send + Sync {
    /// Execute SELECT query
    async fn select(&self, query: &str) -> GraphRAGResult<Vec<HashMap<String, String>>>;

    /// Execute ASK query
    async fn ask(&self, query: &str) -> GraphRAGResult<bool>;

    /// Execute CONSTRUCT query
    async fn construct(&self, query: &str) -> GraphRAGResult<Vec<Triple>>;
}

/// Trait for LLM client operations
#[async_trait]
pub trait LlmClientTrait: Send + Sync {
    /// Generate response from context and query
    async fn generate(&self, context: &str, query: &str) -> GraphRAGResult<String>;

    /// Generate with streaming response
    async fn generate_stream(
        &self,
        context: &str,
        query: &str,
        callback: Box<dyn Fn(&str) + Send + Sync>,
    ) -> GraphRAGResult<String>;
}

/// Main GraphRAG engine
pub struct GraphRAGEngine<V, E, S, L>
where
    V: VectorIndexTrait,
    E: EmbeddingModelTrait,
    S: SparqlEngineTrait,
    L: LlmClientTrait,
{
    /// Vector index for similarity search
    vec_index: Arc<V>,
    /// Embedding model for query vectorization
    embedding_model: Arc<E>,
    /// SPARQL engine for graph traversal
    sparql_engine: Arc<S>,
    /// LLM client for answer generation
    llm_client: Arc<L>,
    /// Configuration
    config: GraphRAGConfig,
    /// Query result cache
    cache: Arc<RwLock<lru::LruCache<String, GraphRAGResult2>>>,
    /// Community detector (lazy initialized)
    community_detector: Option<Arc<CommunityDetector>>,
}

impl<V, E, S, L> GraphRAGEngine<V, E, S, L>
where
    V: VectorIndexTrait,
    E: EmbeddingModelTrait,
    S: SparqlEngineTrait,
    L: LlmClientTrait,
{
    /// Create a new GraphRAG engine
    pub fn new(
        vec_index: Arc<V>,
        embedding_model: Arc<E>,
        sparql_engine: Arc<S>,
        llm_client: Arc<L>,
        config: GraphRAGConfig,
    ) -> Self {
        const DEFAULT_CACHE_SIZE: std::num::NonZeroUsize = match std::num::NonZeroUsize::new(1000) {
            Some(size) => size,
            None => panic!("constant is non-zero"),
        };

        let cache_size = config
            .cache_size
            .and_then(std::num::NonZeroUsize::new)
            .unwrap_or(DEFAULT_CACHE_SIZE);

        Self {
            vec_index,
            embedding_model,
            sparql_engine,
            llm_client,
            config,
            cache: Arc::new(RwLock::new(lru::LruCache::new(cache_size))),
            community_detector: None,
        }
    }

    /// Execute a GraphRAG query
    pub async fn query(&self, query: &str) -> GraphRAGResult<GraphRAGResult2> {
        let start_time = std::time::Instant::now();

        // Check cache
        if let Some(cached) = self.cache.read().await.peek(&query.to_string()) {
            return Ok(cached.clone());
        }

        // 1. Embed query
        let query_vec = self.embedding_model.embed(query).await?;

        // 2. Vector retrieval (Top-K)
        let vector_results = self
            .vec_index
            .search_knn(&query_vec, self.config.top_k)
            .await?;

        // 3. Keyword retrieval (BM25) - simplified for now
        let keyword_results = self.keyword_search(query).await?;

        // 4. Fusion (RRF)
        let seeds = self.fuse_results(&vector_results, &keyword_results)?;

        // 5. Graph expansion (SPARQL)
        let subgraph = self.expand_graph(&seeds).await?;

        // 6. Community detection (optional)
        let communities = if self.config.enable_communities {
            self.detect_communities(&subgraph)?
        } else {
            vec![]
        };

        // 7. Build context
        let context = self.build_context(&subgraph, &communities, query)?;

        // 8. Generate answer
        let answer = self.llm_client.generate(&context, query).await?;

        // Calculate confidence based on seed scores and graph coverage
        let confidence = self.calculate_confidence(&seeds, &subgraph);

        let result = GraphRAGResult2 {
            answer,
            subgraph: subgraph.clone(),
            seeds: seeds.clone(),
            communities,
            provenance: QueryProvenance {
                timestamp: Utc::now(),
                original_query: query.to_string(),
                expanded_query: None,
                seed_entities: seeds.iter().map(|s| s.uri.clone()).collect(),
                source_triples: subgraph,
                community_sources: vec![],
                processing_time_ms: start_time.elapsed().as_millis() as u64,
            },
            confidence,
        };

        // Update cache
        self.cache
            .write()
            .await
            .put(query.to_string(), result.clone());

        Ok(result)
    }

    /// Keyword search using BM25 (simplified)
    async fn keyword_search(&self, query: &str) -> GraphRAGResult<Vec<(String, f32)>> {
        // Build SPARQL query with text matching
        let terms: Vec<&str> = query.split_whitespace().collect();
        if terms.is_empty() {
            return Ok(vec![]);
        }

        // Create SPARQL FILTER with regex for each term
        let filters: Vec<String> = terms
            .iter()
            .map(|term| format!("REGEX(STR(?label), \"{}\", \"i\")", term))
            .collect();

        let sparql = format!(
            r#"
            SELECT DISTINCT ?entity (COUNT(*) AS ?score) WHERE {{
                ?entity rdfs:label|schema:name|dc:title ?label .
                FILTER({})
            }}
            GROUP BY ?entity
            ORDER BY DESC(?score)
            LIMIT {}
            "#,
            filters.join(" || "),
            self.config.top_k
        );

        let results = self.sparql_engine.select(&sparql).await?;

        Ok(results
            .into_iter()
            .filter_map(|row| {
                let entity = row.get("entity")?.clone();
                let score = row.get("score")?.parse::<f32>().ok()?;
                Some((entity, score))
            })
            .collect())
    }

    /// Fuse vector and keyword results using Reciprocal Rank Fusion
    fn fuse_results(
        &self,
        vector_results: &[(String, f32)],
        keyword_results: &[(String, f32)],
    ) -> GraphRAGResult<Vec<ScoredEntity>> {
        let k = 60.0; // RRF constant

        let mut scores: HashMap<String, (f64, ScoreSource)> = HashMap::new();

        // Add vector scores
        for (rank, (uri, score)) in vector_results.iter().enumerate() {
            let rrf_score = 1.0 / (k + rank as f64 + 1.0);
            scores.insert(
                uri.clone(),
                (
                    rrf_score * self.config.vector_weight as f64,
                    ScoreSource::Vector,
                ),
            );
        }

        // Add keyword scores
        for (rank, (uri, _score)) in keyword_results.iter().enumerate() {
            let rrf_score = 1.0 / (k + rank as f64 + 1.0);
            let keyword_contribution = rrf_score * self.config.keyword_weight as f64;

            match scores.get(uri).cloned() {
                Some((existing_score, _)) => {
                    let new_score = existing_score + keyword_contribution;
                    scores.insert(uri.clone(), (new_score, ScoreSource::Fused));
                }
                None => {
                    scores.insert(uri.clone(), (keyword_contribution, ScoreSource::Keyword));
                }
            }
        }

        // Sort by score and take top results
        let mut entities: Vec<ScoredEntity> = scores
            .into_iter()
            .map(|(uri, (score, source))| ScoredEntity {
                uri,
                score,
                source,
                metadata: HashMap::new(),
            })
            .collect();

        entities.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        entities.truncate(self.config.max_seeds);

        Ok(entities)
    }

    /// Expand graph from seed entities using SPARQL
    async fn expand_graph(&self, seeds: &[ScoredEntity]) -> GraphRAGResult<Vec<Triple>> {
        if seeds.is_empty() {
            return Ok(vec![]);
        }

        let seed_uris: Vec<String> = seeds.iter().map(|s| format!("<{}>", s.uri)).collect();
        let values = seed_uris.join(" ");

        // N-hop neighbor expansion
        let hops = self.config.expansion_hops;
        let path_pattern = if hops == 1 {
            "?seed ?p ?neighbor".to_string()
        } else {
            format!("?seed (:|!:){{1,{}}} ?neighbor", hops)
        };

        let sparql = format!(
            r#"
            CONSTRUCT {{
                ?seed ?p ?o .
                ?s ?p2 ?seed .
                ?neighbor ?p3 ?o2 .
            }}
            WHERE {{
                VALUES ?seed {{ {} }}
                {{
                    ?seed ?p ?o .
                }} UNION {{
                    ?s ?p2 ?seed .
                }} UNION {{
                    {}
                    ?neighbor ?p3 ?o2 .
                }}
            }}
            LIMIT {}
            "#,
            values, path_pattern, self.config.max_subgraph_size
        );

        self.sparql_engine.construct(&sparql).await
    }

    /// Detect communities in the subgraph using Louvain algorithm
    fn detect_communities(&self, subgraph: &[Triple]) -> GraphRAGResult<Vec<CommunitySummary>> {
        use petgraph::graph::UnGraph;

        if subgraph.is_empty() {
            return Ok(vec![]);
        }

        // Build undirected graph
        let mut graph: UnGraph<String, ()> = UnGraph::new_undirected();
        let mut node_indices: HashMap<String, petgraph::graph::NodeIndex> = HashMap::new();

        for triple in subgraph {
            let subj_idx = *node_indices
                .entry(triple.subject.clone())
                .or_insert_with(|| graph.add_node(triple.subject.clone()));
            let obj_idx = *node_indices
                .entry(triple.object.clone())
                .or_insert_with(|| graph.add_node(triple.object.clone()));

            if subj_idx != obj_idx {
                graph.add_edge(subj_idx, obj_idx, ());
            }
        }

        // Simple community detection based on connected components
        // (Full Louvain implementation would be more complex)
        let components = petgraph::algo::kosaraju_scc(&graph);

        let communities: Vec<CommunitySummary> = components
            .into_iter()
            .enumerate()
            .filter(|(_, component)| component.len() >= 2)
            .map(|(idx, component)| {
                let entities: Vec<String> = component
                    .iter()
                    .filter_map(|&node_idx| graph.node_weight(node_idx).cloned())
                    .collect();

                let representative_triples: Vec<Triple> = subgraph
                    .iter()
                    .filter(|t| entities.contains(&t.subject) || entities.contains(&t.object))
                    .take(5)
                    .cloned()
                    .collect();

                CommunitySummary {
                    id: format!("community_{}", idx),
                    summary: format!("Community with {} entities", entities.len()),
                    entities,
                    representative_triples,
                    level: 0,
                    modularity: 0.0,
                }
            })
            .collect();

        Ok(communities)
    }

    /// Build context string for LLM from subgraph and communities
    fn build_context(
        &self,
        subgraph: &[Triple],
        communities: &[CommunitySummary],
        _query: &str,
    ) -> GraphRAGResult<String> {
        let mut context = String::new();

        // Add community summaries if available
        if !communities.is_empty() {
            context.push_str("## Community Context\n\n");
            for community in communities {
                context.push_str(&format!("### {}\n", community.id));
                context.push_str(&format!("{}\n", community.summary));
                context.push_str(&format!("Entities: {}\n\n", community.entities.join(", ")));
            }
        }

        // Add relevant triples
        context.push_str("## Knowledge Graph Facts\n\n");
        for triple in subgraph.iter().take(self.config.max_context_triples) {
            context.push_str(&format!(
                "- {} → {} → {}\n",
                triple.subject, triple.predicate, triple.object
            ));
        }

        Ok(context)
    }

    /// Calculate confidence score based on retrieval quality
    fn calculate_confidence(&self, seeds: &[ScoredEntity], subgraph: &[Triple]) -> f64 {
        if seeds.is_empty() {
            return 0.0;
        }

        // Average seed score
        let avg_seed_score: f64 = seeds.iter().map(|s| s.score).sum::<f64>() / seeds.len() as f64;

        // Graph coverage (how many seeds appear in subgraph)
        let seed_uris: std::collections::HashSet<_> = seeds.iter().map(|s| &s.uri).collect();
        let covered: usize = subgraph
            .iter()
            .filter(|t| seed_uris.contains(&t.subject) || seed_uris.contains(&t.object))
            .count();
        let coverage = if subgraph.is_empty() {
            0.0
        } else {
            (covered as f64 / subgraph.len() as f64).min(1.0)
        };

        // Combined confidence
        (avg_seed_score * 0.6 + coverage * 0.4).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triple_creation() {
        let triple = Triple::new(
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o",
        );
        assert_eq!(triple.subject, "http://example.org/s");
        assert_eq!(triple.predicate, "http://example.org/p");
        assert_eq!(triple.object, "http://example.org/o");
    }

    #[test]
    fn test_scored_entity() {
        let entity = ScoredEntity {
            uri: "http://example.org/entity".to_string(),
            score: 0.85,
            source: ScoreSource::Fused,
            metadata: HashMap::new(),
        };
        assert_eq!(entity.score, 0.85);
        assert_eq!(entity.source, ScoreSource::Fused);
    }
}
