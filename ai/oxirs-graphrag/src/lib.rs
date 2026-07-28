//! # OxiRS GraphRAG
//!
//! **GraphRAG** (Graph Retrieval-Augmented Generation) is a production-ready
//! Rust library that combines **knowledge-graph topology traversal** with
//! **vector similarity search** to deliver context-rich answers for LLM
//! pipelines — without any network dependencies at query time.
//!
//! It is the JVM-free, pure-Rust counterpart of Microsoft's GraphRAG and
//! LangChain's knowledge-graph QA stack, integrated directly with the OxiRS
//! semantic-web engine.
//!
//! ## Data-flow overview
//!
//! ```text
//! Natural-Language Query
//!         │
//!         ▼
//! ┌───────────────────┐
//! │  Query Embedding  │  (oxirs-embed / Node2Vec / TransE)
//! └────────┬──────────┘
//!          │
//!   ┌──────┴──────┐
//!   │             │
//!   ▼             ▼
//! Vector        Keyword
//! KNN           BM25
//! Search        Search
//!   │             │
//!   └──────┬──────┘
//!          │
//!          ▼
//!  ┌───────────────┐
//!  │  RRF Fusion   │  Reciprocal Rank Fusion → Seed Entities
//!  └───────┬───────┘
//!          │
//!          ▼
//!  ┌────────────────────────┐
//!  │  SPARQL N-hop Expansion│  Graph traversal (up to 500 triples)
//!  └────────────┬───────────┘
//!               │
//!               ▼
//!  ┌────────────────────────┐
//!  │  Community Detection   │  Louvain / Leiden clustering
//!  └────────────┬───────────┘
//!               │
//!               ▼
//!  ┌────────────────────────┐
//!  │  Context Building      │  Subgraph → natural-language context
//!  └────────────┬───────────┘
//!               │
//!               ▼
//!  ┌────────────────────────┐
//!  │  LLM Generation        │  Answer + citations
//!  └────────────────────────┘
//! ```
//!
//! ## Key modules
//!
//! | Module | Purpose |
//! |--------|---------|
//! [`triple_extractor`] | Rule-based NLP → RDF triple extraction |
//! [`community_detector`] | Greedy label-propagation community detection |
//! [`path_finder`] | BFS / DFS shortest-path retrieval in KGs |
//! [`graph_embedder`] | Node2Vec-style random-walk structural embeddings |
//! [`summarizer`] | Cluster-based subgraph summarization for LLM context |
//! [`path_ranker`] | Predicate-weighted path ranking |
//! [`context_builder`] | N-hop subgraph extraction and truncation |
//! [`knowledge_fusion`] | Multi-source KG fusion with provenance |
//! [`graph_summarization`] | PageRank-style community summary generation |
//! [`entity_linking`] | Entity linking and disambiguation |
//! [`explainability`] | Attention weights, path explanation, provenance |
//! [`feedback`] | Session-scoped user-feedback weight adaptation |
//! [`graph`] | Core community detection and graph traversal primitives |
//! [`retrieval`] | Hybrid vector + keyword retrieval with RRF fusion |
//! [`generation`] | Prompt templates and LLM context building |
//! [`temporal`] | Temporal knowledge graph retrieval |
//!
//! ## Quickstart — standalone pipeline (no network, no LLM)
//!
//! The example below runs an end-to-end mini-pipeline entirely in memory on a
//! synthetic 8-node knowledge graph: extract triples from text, detect
//! communities, find paths, and summarize the result.
//!
//! ```rust
//! use oxirs_graphrag::triple_extractor::{ExtractionConfig, TripleExtractor};
//! use oxirs_graphrag::community_detector::{CommunityGraph, CommunityDetector};
//! use oxirs_graphrag::path_finder::{KnowledgeEdge, PathFinder, PathFinderConfig};
//! use oxirs_graphrag::summarizer::{KgEdge, KgNode, KgSubgraph, SubgraphSummarizer};
//!
//! // ── Step 1: Extract triples from natural language ─────────────────────────
//! let corpus = [
//!     "Alice is a data scientist.",
//!     "Bob works at ACME.",
//!     "Carol is a software engineer.",
//!     "Dave is part of the AI team.",
//!     "ACME has a research division.",
//! ];
//! let extractor = TripleExtractor::with_defaults(ExtractionConfig::default());
//! let all_triples: Vec<_> = corpus
//!     .iter()
//!     .flat_map(|sentence| extractor.extract(sentence))
//!     .collect();
//! assert!(!all_triples.is_empty(), "at least one triple extracted");
//!
//! // ── Step 2: Build community graph and detect clusters ─────────────────────
//! let mut cg = CommunityGraph::new();
//! // 8 synthetic nodes
//! for (id, label) in [
//!     (1u64, "Alice"), (2, "Bob"), (3, "Carol"), (4, "Dave"),
//!     (5, "ACME"),    (6, "AI-Team"), (7, "Research"), (8, "Berlin"),
//! ] {
//!     cg.add_node(id, label);
//! }
//! for (a, b) in [(1,5),(2,5),(3,6),(4,6),(5,7),(6,7),(7,8),(1,2)] {
//!     cg.add_edge(a, b, 1.0);
//! }
//! let detector = CommunityDetector::new(2, 50);
//! let detection = detector.detect(&mut cg);
//! assert!(!detection.communities.is_empty(), "at least one community");
//!
//! // ── Step 3: Graph path retrieval ──────────────────────────────────────────
//! let edges = vec![
//!     KnowledgeEdge::new("Alice",    "works_at",    "ACME"),
//!     KnowledgeEdge::new("ACME",     "located_in",  "Berlin"),
//!     KnowledgeEdge::new("Bob",      "knows",       "Alice"),
//!     KnowledgeEdge::new("Alice",    "member_of",   "AI-Team"),
//!     KnowledgeEdge::new("AI-Team",  "part_of",     "ACME"),
//!     KnowledgeEdge::new("Carol",    "works_at",    "ACME"),
//!     KnowledgeEdge::new("Dave",     "leads",       "AI-Team"),
//!     KnowledgeEdge::new("Research", "division_of", "ACME"),
//! ];
//! let finder = PathFinder::new(edges, PathFinderConfig::default());
//! let paths = finder.bfs_paths("Bob", "Berlin", 4);
//! assert!(!paths.is_empty(), "path Bob→Berlin found");
//!
//! // ── Step 4: Summarize subgraph for LLM context ────────────────────────────
//! let mut subgraph = KgSubgraph::new();
//! for (id, label, ty) in [
//!     ("alice",    "Alice",    "Person"),
//!     ("bob",      "Bob",      "Person"),
//!     ("carol",    "Carol",    "Person"),
//!     ("acme",     "ACME",     "Organization"),
//!     ("berlin",   "Berlin",   "Place"),
//!     ("ai_team",  "AI-Team",  "Team"),
//!     ("research", "Research", "Department"),
//!     ("dave",     "Dave",     "Person"),
//! ] {
//!     subgraph.add_node(KgNode::simple(id, label, ty));
//! }
//! subgraph.add_edge(KgEdge::unweighted("alice", "acme",  "works_at"));
//! subgraph.add_edge(KgEdge::unweighted("acme",  "berlin","located_in"));
//!
//! let summarizer = SubgraphSummarizer::new();
//! let clusters = summarizer.summarize(&subgraph, 10);
//! assert!(!clusters.is_empty(), "at least one cluster");
//! let text_summary = summarizer.generate_text_summary(&clusters);
//! assert!(!text_summary.is_empty(), "non-empty summary text");
//! ```
//!
//! ## Full engine usage (async, requires trait impls)
//!
//! For production usage with a real vector index, embedding model, SPARQL engine,
//! and LLM client:
//!
//! ```rust,ignore
//! use oxirs_graphrag::{GraphRAGEngine, GraphRAGConfig};
//! use std::sync::Arc;
//!
//! let config = GraphRAGConfig {
//!     top_k: 20,
//!     expansion_hops: 2,
//!     enable_communities: true,
//!     ..Default::default()
//! };
//!
//! // Provide your own implementations of VectorIndexTrait, EmbeddingModelTrait,
//! // SparqlEngineTrait, and LlmClientTrait:
//! let engine = GraphRAGEngine::new(
//!     Arc::new(my_vec_index),
//!     Arc::new(my_embedder),
//!     Arc::new(my_sparql),
//!     Arc::new(my_llm),
//!     config,
//! );
//!
//! let result = engine.query("What safety issues affect battery cells?").await?;
//! println!("Answer: {}", result.answer);
//! println!("Confidence: {:.2}", result.confidence);
//! ```
//!
//! See [`docs/tutorial.md`](https://github.com/cool-japan/oxirs/blob/master/ai/oxirs-graphrag/docs/tutorial.md)
//! for a step-by-step walkthrough.

pub mod cache;
pub mod config;
pub mod distributed;
// v1.1.0: Graph summarization for RAG
pub mod embeddings;
pub mod federation;
pub mod fusion;
pub mod generation;
pub mod graph;
pub mod graph_summarization;
pub mod query;
pub mod reasoning;
pub mod retrieval;
pub mod sparql;
pub mod streaming;
pub mod temporal;

// v1.1.0 TransE knowledge graph embedding model
pub mod transe_model;

// v1.1.0: Entity linking and disambiguation for knowledge graphs
pub mod entity_linking;

// v1.1.0 round 5: Community detection (Louvain-inspired greedy label propagation)
pub mod community_detector;

// v1.1.0 round 6: Knowledge graph path ranking (DFS + Dijkstra + scoring)
pub mod path_ranker;

// v1.1.0 round 7: String-to-RDF entity linking (mention detection + candidate ranking)
pub mod entity_linker;

// v1.1.0 round 11: Node2Vec-inspired graph embedding and structural node representations
pub mod graph_embedder;

// v1.1.0 round 12: Graph partitioning using greedy / label-propagation / bisection methods
pub mod graph_partitioner;

// v1.1.0 round 13: Rule-based knowledge triple extraction from natural language text
pub mod triple_extractor;

// v1.1.0 round 11: Multi-source knowledge fusion with provenance tracking
pub mod knowledge_fusion;

// v1.1.0 round 12: Context building for graph-based RAG (N-hop, ranking, truncation, formatting)
pub mod context_builder;

// v1.1.0 round 13: Graph path finding for RAG (BFS/DFS, shortest path, predicate filtering, scoring)
pub mod path_finder;

// v1.1.0 round 14: KG subgraph summarization via cluster-based abstraction
pub mod summarizer;

// v1.1.0 round 15: Entity type classification for knowledge graph nodes
pub mod entity_classifier;

// v1.1.0 round 16: Explainability — attention weights, path explanation, provenance
pub mod explainability;

// v1.1.0 round 17: Interactive refinement with user feedback
pub mod feedback;

// v0.4.0: Re-export new GraphSummarizer + GraphSummary types
pub use summarizer::{GraphSummarizer, GraphSummary};
// v0.4.0: Re-export new TripleRelevanceFeedback + Relevance types
pub use feedback::{Relevance, TripleId, TripleRelevanceFeedback};

// v0.3.0 / block-5: GNN encoder — phase a: GraphSAGE over the knowledge graph
pub mod gnn_encoder;

// v0.3.1: GNN encoder new components
pub use gnn_encoder::{
    AdjacencyGraph, EdgeList, GnnEncoder, GnnEncoderConfig, ScaledDotProductAttention,
};

// v0.3.0 / block-6: Hybrid GNN+LLM — phase b/c: LLM head with frozen GNN soft-prompt
pub mod hybrid;

// v0.3.0 / block-8: Hybrid GNN+LLM phase d — GGUF model loader + LoRA adapter
#[cfg(feature = "gguf-loader")]
pub mod model_loader;

// v0.3.0 / block-7: Neuro-symbolic fusion — PINN-driven physics-informed entity scoring
pub mod neuro_symbolic;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock;

// Re-exports
pub use cache::query_cache::{CacheEntry, CacheStats, QueryCache, QueryCacheConfig};
pub use config::{CacheConfiguration, GraphRAGConfig};
pub use embeddings::node2vec::{
    Node2VecConfig, Node2VecEmbedder, Node2VecEmbeddings, Node2VecWalkConfig,
};
pub use graph::community::{CommunityAlgorithm, CommunityConfig, CommunityDetector};
pub use graph::embeddings::{CommunityAwareEmbeddings, CommunityStructure, EmbeddingConfig};
pub use graph::traversal::GraphTraversal;
pub use hybrid::lora::{LoraAdapter, LoraTrainer};
pub use query::planner::QueryPlanner;
pub use retrieval::fusion::FusionStrategy;

// Feature-gated re-exports for GGUF model loader.
#[cfg(feature = "gguf-loader")]
pub use model_loader::{
    GgufMetadata, GgufModelArch, GgufParseError, GgufParser, GgufTensorInfo, GgufValue,
    ModelHandle, ModelInfo, ModelRegistry, RegistryError,
};

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

/// Cached result with metadata
#[derive(Debug, Clone)]
struct CachedResult {
    result: GraphRAGResult2,
    timestamp: SystemTime,
    ttl: Duration,
}

impl CachedResult {
    /// Check if the cached result is still fresh
    fn is_fresh(&self) -> bool {
        self.timestamp
            .elapsed()
            .map(|elapsed| elapsed < self.ttl)
            .unwrap_or(false)
    }
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Base TTL in seconds (default: 3600 = 1 hour)
    pub base_ttl_seconds: u64,
    /// Minimum TTL in seconds (default: 300 = 5 minutes)
    pub min_ttl_seconds: u64,
    /// Maximum TTL in seconds (default: 86400 = 24 hours)
    pub max_ttl_seconds: u64,
    /// Enable adaptive TTL based on update frequency
    pub adaptive: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            base_ttl_seconds: 3600,
            min_ttl_seconds: 300,
            max_ttl_seconds: 86400,
            adaptive: true,
        }
    }
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
    /// Query result cache with adaptive TTL
    cache: Arc<RwLock<lru::LruCache<String, CachedResult>>>,
    /// Cache configuration
    cache_config: CacheConfig,
    /// Graph update counter for adaptive TTL
    graph_update_count: Arc<AtomicU64>,
    /// Community detector, built from `config.community_algorithm` at
    /// construction time and used by [`Self::detect_communities`]. Always
    /// `Some` once the engine is constructed via `new` / `with_cache_config`
    /// — `Option` only because `detect_communities` needs a borrow-checker
    /// friendly way to report the (unreachable in practice) uninitialized
    /// case via `GraphRAGResult` rather than panicking.
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
        let cache_config = CacheConfig {
            base_ttl_seconds: config.cache_config.base_ttl_seconds,
            min_ttl_seconds: config.cache_config.min_ttl_seconds,
            max_ttl_seconds: config.cache_config.max_ttl_seconds,
            adaptive: config.cache_config.adaptive,
        };

        Self::with_cache_config(
            vec_index,
            embedding_model,
            sparql_engine,
            llm_client,
            config,
            cache_config,
        )
    }

    /// Create a new GraphRAG engine with custom cache configuration
    pub fn with_cache_config(
        vec_index: Arc<V>,
        embedding_model: Arc<E>,
        sparql_engine: Arc<S>,
        llm_client: Arc<L>,
        config: GraphRAGConfig,
        cache_config: CacheConfig,
    ) -> Self {
        const DEFAULT_CACHE_SIZE: std::num::NonZeroUsize = match std::num::NonZeroUsize::new(1000) {
            Some(size) => size,
            None => panic!("constant is non-zero"),
        };

        let cache_size = config
            .cache_size
            .and_then(std::num::NonZeroUsize::new)
            .unwrap_or(DEFAULT_CACHE_SIZE);

        // Build the real community detector up front from
        // `config.community_algorithm` so `detect_communities` always has a
        // genuine Louvain/Leiden/label-propagation/connected-components
        // implementation to delegate to (see that method's doc comment).
        // `min_community_size: 2` (rather than the detector's own default of
        // 3) matches this engine's historical "communities of >= 2 entities"
        // behavior for the typically-small subgraphs a single query expands.
        let community_config = CommunityConfig {
            algorithm: map_community_algorithm(config.community_algorithm),
            min_community_size: 2,
            ..CommunityConfig::default()
        };
        let community_detector = Some(Arc::new(CommunityDetector::new(community_config)));

        Self {
            vec_index,
            embedding_model,
            sparql_engine,
            llm_client,
            config,
            cache: Arc::new(RwLock::new(lru::LruCache::new(cache_size))),
            cache_config,
            graph_update_count: Arc::new(AtomicU64::new(0)),
            community_detector,
        }
    }

    /// Calculate adaptive TTL based on graph update frequency
    fn calculate_ttl(&self) -> Duration {
        if !self.cache_config.adaptive {
            return Duration::from_secs(self.cache_config.base_ttl_seconds);
        }

        let updates_per_hour = self.graph_update_count.load(Ordering::Relaxed) as f64;

        // More updates = shorter TTL
        let ttl_secs = if updates_per_hour > 100.0 {
            self.cache_config.min_ttl_seconds // High update rate: 5 min TTL
        } else if updates_per_hour > 10.0 {
            self.cache_config.base_ttl_seconds / 2 // Medium: 30 min TTL
        } else {
            self.cache_config.max_ttl_seconds // Low update rate: 24 hour TTL
        };

        Duration::from_secs(ttl_secs)
    }

    /// Record graph update for adaptive TTL calculation
    pub fn record_graph_update(&self) {
        self.graph_update_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current cache hit rate for monitoring
    pub async fn get_cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.read().await;
        (cache.len(), cache.cap().get())
    }

    /// Execute a GraphRAG query
    pub async fn query(&self, query: &str) -> GraphRAGResult<GraphRAGResult2> {
        let start_time = std::time::Instant::now();

        // Check cache with freshness validation
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.peek(&query.to_string()) {
                if cached.is_fresh() {
                    return Ok(cached.result.clone());
                }
            }
        }

        // 0. Consult the query planner. `QueryPlanner::plan` computes stage
        // ordering, per-stage dependencies, and an estimated cost; here we
        // actually act on it rather than letting it sit unused: `parallel`
        // (true whenever vector search and keyword search have no
        // dependency on each other, which is always for this fixed
        // pipeline) determines whether steps 2 and 3 below run concurrently
        // via `tokio::join!` or sequentially.
        let parsed_query = query::parser::QueryParser::new().parse(query)?;
        let plan = query::planner::QueryPlanner::new(self.config.clone()).plan(&parsed_query)?;
        tracing::debug!(
            estimated_cost = plan.estimated_cost,
            parallel = plan.parallel,
            stages = plan.stages.len(),
            "GraphRAG query execution plan"
        );

        // 1. Embed query
        let query_vec = self.embedding_model.embed(query).await?;

        // 2 + 3. Vector retrieval (Top-K) and keyword retrieval (BM25).
        // These two stages have no dependency on each other in the plan, so
        // when `plan.parallel` is set they genuinely run concurrently
        // instead of one `.await` after another.
        let (vector_results, keyword_results) = if plan.parallel {
            let vector_fut = self.vec_index.search_knn(&query_vec, self.config.top_k);
            let keyword_fut = self.keyword_search(query);
            let (vector_results, keyword_results) = tokio::join!(vector_fut, keyword_fut);
            (vector_results?, keyword_results?)
        } else {
            let vector_results = self
                .vec_index
                .search_knn(&query_vec, self.config.top_k)
                .await?;
            let keyword_results = self.keyword_search(query).await?;
            (vector_results, keyword_results)
        };

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

        // Update cache with adaptive TTL
        let ttl = self.calculate_ttl();
        let cached = CachedResult {
            result: result.clone(),
            timestamp: SystemTime::now(),
            ttl,
        };
        self.cache.write().await.put(query.to_string(), cached);

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
        let sparql = build_expand_graph_query(
            &seed_uris,
            self.config.expansion_hops,
            self.config.max_subgraph_size,
        );

        self.sparql_engine.construct(&sparql).await
    }

    /// Detect communities in the subgraph using the configured algorithm
    /// (`self.config.community_algorithm`: Louvain / Leiden / Label
    /// Propagation / Connected Components), delegating to the real
    /// [`graph::community::CommunityDetector`] so callers get genuine,
    /// resolution-parameterised community structure and an honest
    /// Newman-Girvan modularity score instead of a fixed `0.0`.
    fn detect_communities(&self, subgraph: &[Triple]) -> GraphRAGResult<Vec<CommunitySummary>> {
        if subgraph.is_empty() {
            return Ok(vec![]);
        }

        let detector = self.community_detector.as_ref().ok_or_else(|| {
            GraphRAGError::CommunityDetectionError(
                "community detector was not initialized".to_string(),
            )
        })?;

        detector.detect(subgraph)
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

/// Build the CONSTRUCT query used by [`GraphRAGEngine::expand_graph`] for
/// N-hop neighbor expansion from a set of seed IRI terms (already
/// `<...>`-wrapped).
///
/// Pulled out as a free function so it can be exercised (and, in tests,
/// round-tripped through the real `oxirs-arq` parser) without needing a
/// live `SparqlEngineTrait` implementation.
///
/// Two formatting quirks below are load-bearing, not stylistic (both
/// verified against the real `oxirs-arq` parser while writing this query
/// builder — see the round-trip regression tests):
///
/// - `CONSTRUCT { ... }` is kept on a single physical line, immediately
///   followed by `WHERE {` on that *same* line: the parser does not skip a
///   bare newline between the template's closing `}` and the `WHERE`
///   keyword, and fails with "Expected LeftBrace, found Some(Newline)" if
///   one is present.
/// - The `WHERE` clause's closing `}` is immediately followed by `LIMIT` on
///   the same line, for the same reason (a newline there instead yields
///   "Unexpected trailing tokens after query: Some(Limit)" — the modifier
///   parser doesn't skip a leading newline before checking for `LIMIT`).
///
/// The `WHERE` clause body itself has no such restriction and is formatted
/// multi-line for readability.
fn build_expand_graph_query(seed_uris: &[String], hops: usize, max_subgraph_size: usize) -> String {
    let values = seed_uris.join(" ");

    // N-hop neighbor expansion. See `sparql::hop_pattern` for why this is an
    // explicit UNION of path-free BGP chains rather than a SPARQL property
    // path: the previous `(:|!:){1,hops}` referenced an undeclared empty
    // prefix and made every real `query()` call fail at this step for the
    // (default) multi-hop case.
    let hop_pattern = crate::sparql::hop_pattern::build_forward_hop_pattern(
        "?seed",
        "?neighbor",
        hops,
        "hp",
        "hn",
    );

    format!(
        r#"
        CONSTRUCT {{ ?seed ?p ?o . ?s ?p2 ?seed . ?neighbor ?p3 ?o2 . }} WHERE {{
            VALUES ?seed {{ {values} }}
            {{
                ?seed ?p ?o .
            }} UNION {{
                ?s ?p2 ?seed .
            }} UNION {{
                {hop_pattern}
                ?neighbor ?p3 ?o2 .
            }}
        }} LIMIT {max_subgraph_size}
        "#
    )
}

/// Map the crate's public [`config::CommunityAlgorithm`] configuration enum
/// onto the [`graph::community::CommunityAlgorithm`] the real detector
/// implementation understands. Kept as an explicit mapping (rather than
/// reusing one enum for both) because `config::CommunityAlgorithm` is a
/// stable, `serde`-versioned user-facing config surface while
/// `graph::community::CommunityAlgorithm` also has an internal-only
/// `Hierarchical` variant that is not (yet) exposed as a top-level engine
/// config choice.
fn map_community_algorithm(algorithm: config::CommunityAlgorithm) -> CommunityAlgorithm {
    match algorithm {
        config::CommunityAlgorithm::Louvain => CommunityAlgorithm::Louvain,
        config::CommunityAlgorithm::Leiden => CommunityAlgorithm::Leiden,
        config::CommunityAlgorithm::LabelPropagation => CommunityAlgorithm::LabelPropagation,
        config::CommunityAlgorithm::ConnectedComponents => CommunityAlgorithm::ConnectedComponents,
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

    // ── Regression: expand_graph SPARQL must actually parse (P0) ───────────

    #[test]
    fn regression_expand_graph_query_never_emits_empty_prefix_hack() {
        for hops in [1usize, 2, 3, 5] {
            let seed_uris = vec!["<http://example.org/e>".to_string()];
            let sparql = build_expand_graph_query(&seed_uris, hops, 500);
            assert!(
                !sparql.contains(":|!:"),
                "hops={hops}: must not reference the undeclared empty prefix `:`/`!:`\n{sparql}"
            );
            assert!(
                !sparql.contains("!()"),
                "hops={hops}: must not use the unsupported empty negated property set\n{sparql}"
            );
        }
    }

    #[test]
    fn regression_expand_graph_query_round_trips_through_real_arq_parser() {
        // The actual bug: the previous `(:|!:){1,hops}` property path failed
        // to parse against the real oxirs-arq engine for any hops > 1 (the
        // crate's own default config), so every real `GraphRAGEngine::query`
        // call failed at the graph-expansion step. Assert the generated
        // query is genuinely valid SPARQL per the workspace's own parser,
        // not just "doesn't contain a known-bad substring".
        for hops in [1usize, 2, 3, 5, 10] {
            let seed_uris = vec![
                "<http://example.org/seed1>".to_string(),
                "<http://example.org/seed2>".to_string(),
            ];
            let sparql = build_expand_graph_query(&seed_uris, hops, 500);
            let mut parser = oxirs_arq::query::QueryParser::new();
            parser
                .parse(&sparql)
                .unwrap_or_else(|e| panic!("hops={hops} query failed to parse: {e}\n{sparql}"));
        }
    }

    // ── Shared test mocks for GraphRAGEngine ────────────────────────────────

    struct MockVectorIndex;

    #[async_trait]
    impl VectorIndexTrait for MockVectorIndex {
        async fn search_knn(
            &self,
            _query_vector: &[f32],
            _k: usize,
        ) -> GraphRAGResult<Vec<(String, f32)>> {
            Ok(vec![])
        }

        async fn search_threshold(
            &self,
            _query_vector: &[f32],
            _threshold: f32,
        ) -> GraphRAGResult<Vec<(String, f32)>> {
            Ok(vec![])
        }
    }

    struct MockEmbeddingModel;

    #[async_trait]
    impl EmbeddingModelTrait for MockEmbeddingModel {
        async fn embed(&self, _text: &str) -> GraphRAGResult<Vec<f32>> {
            Ok(vec![0.0; 4])
        }

        async fn embed_batch(&self, texts: &[&str]) -> GraphRAGResult<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.0; 4]).collect())
        }
    }

    struct MockSparqlEngine;

    #[async_trait]
    impl SparqlEngineTrait for MockSparqlEngine {
        async fn select(&self, _query: &str) -> GraphRAGResult<Vec<HashMap<String, String>>> {
            Ok(vec![])
        }

        async fn ask(&self, _query: &str) -> GraphRAGResult<bool> {
            Ok(false)
        }

        async fn construct(&self, _query: &str) -> GraphRAGResult<Vec<Triple>> {
            Ok(vec![])
        }
    }

    struct MockLlmClient;

    #[async_trait]
    impl LlmClientTrait for MockLlmClient {
        async fn generate(&self, _context: &str, _query: &str) -> GraphRAGResult<String> {
            Ok("mock answer".to_string())
        }

        async fn generate_stream(
            &self,
            _context: &str,
            _query: &str,
            _callback: Box<dyn Fn(&str) + Send + Sync>,
        ) -> GraphRAGResult<String> {
            Ok("mock answer".to_string())
        }
    }

    fn make_test_engine(
        algorithm: config::CommunityAlgorithm,
    ) -> GraphRAGEngine<MockVectorIndex, MockEmbeddingModel, MockSparqlEngine, MockLlmClient> {
        let config = GraphRAGConfig {
            community_algorithm: algorithm,
            ..GraphRAGConfig::default()
        };
        GraphRAGEngine::new(
            Arc::new(MockVectorIndex),
            Arc::new(MockEmbeddingModel),
            Arc::new(MockSparqlEngine),
            Arc::new(MockLlmClient),
            config,
        )
    }

    // ── Regression: detect_communities uses real modularity (P1) ───────────

    #[tokio::test]
    async fn regression_detect_communities_computes_real_modularity_not_hardcoded_zero() {
        let engine = make_test_engine(config::CommunityAlgorithm::ConnectedComponents);

        // Two disconnected triangles: strong, unambiguous community
        // structure, so the true partition's modularity must be well above
        // zero (previously this was hardcoded to `0.0` for every community
        // regardless of actual graph structure).
        let subgraph = vec![
            Triple::new("http://a", "http://rel", "http://b"),
            Triple::new("http://b", "http://rel", "http://c"),
            Triple::new("http://c", "http://rel", "http://a"),
            Triple::new("http://x", "http://rel", "http://y"),
            Triple::new("http://y", "http://rel", "http://z"),
            Triple::new("http://z", "http://rel", "http://x"),
        ];

        let communities = engine
            .detect_communities(&subgraph)
            .expect("community detection should succeed");

        assert_eq!(
            communities.len(),
            2,
            "two disconnected triangles should form two communities"
        );
        for community in &communities {
            assert!(
                community.modularity > 0.4,
                "modularity must be genuinely computed (expected ~0.5 for two \
                 disconnected triangles), got {}",
                community.modularity
            );
        }
    }

    #[tokio::test]
    async fn regression_detect_communities_respects_configured_algorithm() {
        // Distinct algorithms are wired through to the real detector: this
        // would previously always run plain connected-components no matter
        // what `config.community_algorithm` said.
        let subgraph = vec![
            Triple::new("http://a", "http://rel", "http://b"),
            Triple::new("http://b", "http://rel", "http://c"),
            Triple::new("http://c", "http://rel", "http://a"),
            Triple::new("http://x", "http://rel", "http://y"),
            Triple::new("http://y", "http://rel", "http://z"),
            Triple::new("http://z", "http://rel", "http://x"),
        ];

        for algorithm in [
            config::CommunityAlgorithm::Louvain,
            config::CommunityAlgorithm::Leiden,
            config::CommunityAlgorithm::LabelPropagation,
            config::CommunityAlgorithm::ConnectedComponents,
        ] {
            let engine = make_test_engine(algorithm);
            let communities = engine
                .detect_communities(&subgraph)
                .unwrap_or_else(|e| panic!("{algorithm:?} community detection failed: {e}"));
            assert!(
                !communities.is_empty(),
                "{algorithm:?} should find the obvious two-triangle community structure"
            );
        }
    }

    // ── Regression: QueryPlanner is actually consulted by query() (P2) ─────

    #[test]
    fn regression_query_planner_marks_vector_and_keyword_search_independent() {
        // `GraphRAGEngine::query` uses `plan.parallel` to decide whether to
        // run vector and keyword search concurrently via `tokio::join!`.
        // That decision is only honest if the planner's dependency graph
        // genuinely has no edge between the two stages.
        let config = GraphRAGConfig::default();
        let planner = query::planner::QueryPlanner::new(config);
        let parsed = query::parser::QueryParser::new()
            .parse("What are the battery safety issues?")
            .expect("should parse");
        let plan = planner.plan(&parsed).expect("should plan");

        assert!(plan.parallel);

        let vector_stage = plan
            .stages
            .iter()
            .position(|s| s.stage_type == query::planner::StageType::VectorSearch)
            .expect("vector search stage present");
        let keyword_stage = plan
            .stages
            .iter()
            .position(|s| s.stage_type == query::planner::StageType::KeywordSearch)
            .expect("keyword search stage present");

        assert!(
            !plan.stages[keyword_stage]
                .depends_on
                .contains(&vector_stage),
            "keyword search must not depend on vector search"
        );
        assert!(
            !plan.stages[vector_stage]
                .depends_on
                .contains(&keyword_stage),
            "vector search must not depend on keyword search"
        );
    }

    #[tokio::test]
    async fn regression_query_actually_consults_planner_and_completes() {
        // End-to-end: `query()` must build a plan (not just leave
        // `QueryPlanner` as unreferenced dead code) and still produce a
        // valid result via the parallel vector+keyword path.
        let engine = make_test_engine(config::CommunityAlgorithm::ConnectedComponents);
        let result = engine
            .query("What are the safety issues?")
            .await
            .expect("query should succeed end-to-end with mock backends");
        assert_eq!(result.answer, "mock answer");
    }
}
