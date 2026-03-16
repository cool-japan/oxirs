//! Streaming SPARQL subgraph retrieval with chunked processing and backpressure
//!
//! This module provides memory-safe streaming access to large knowledge graph
//! subgraphs by decomposing SPARQL queries into paginated chunks and applying
//! configurable backpressure to prevent OOM conditions.

use crate::{GraphRAGError, GraphRAGResult, ScoredEntity, SparqlEngineTrait, Triple};
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::Semaphore;
use tracing::{debug, warn};

/// Configuration for streaming SPARQL retrieval
#[derive(Debug, Clone)]
pub struct StreamingSparqlConfig {
    /// Number of triples fetched per SPARQL page
    pub page_size: usize,
    /// Maximum total triples to stream (0 = unlimited)
    pub max_total_triples: usize,
    /// Capacity of the internal mpsc channel (backpressure buffer)
    pub channel_capacity: usize,
    /// Maximum concurrent SPARQL page requests
    pub max_concurrency: usize,
    /// Whether to deduplicate triples across pages
    pub deduplicate: bool,
    /// Minimum score threshold for seed filtering
    pub min_seed_score: f64,
    /// Number of graph hops to expand
    pub expansion_hops: usize,
}

impl Default for StreamingSparqlConfig {
    fn default() -> Self {
        Self {
            page_size: 1_000,
            max_total_triples: 50_000,
            channel_capacity: 512,
            max_concurrency: 4,
            deduplicate: true,
            min_seed_score: 0.0,
            expansion_hops: 2,
        }
    }
}

/// A single page of streaming results
#[derive(Debug, Clone)]
pub struct TriplePage {
    /// The triples in this page
    pub triples: Vec<Triple>,
    /// Page index (0-based)
    pub page_index: usize,
    /// Cumulative triples delivered so far (inclusive of this page)
    pub cumulative_count: usize,
    /// Whether this is the last page of a seed's subgraph
    pub is_last_page: bool,
    /// The seed URI that produced this page
    pub seed_uri: String,
}

/// Stream state returned to the caller
pub struct SubgraphStream {
    receiver: mpsc::Receiver<GraphRAGResult<TriplePage>>,
}

impl SubgraphStream {
    /// Receive the next page of triples, or `None` when the stream is exhausted.
    pub async fn next_page(&mut self) -> Option<GraphRAGResult<TriplePage>> {
        self.receiver.recv().await
    }

    /// Collect all pages into a single `Vec<Triple>`, propagating the first error.
    pub async fn collect_all(mut self) -> GraphRAGResult<Vec<Triple>> {
        let mut result: Vec<Triple> = Vec::new();
        while let Some(page_result) = self.next_page().await {
            match page_result {
                Ok(page) => result.extend(page.triples),
                Err(e) => return Err(e),
            }
        }
        Ok(result)
    }
}

/// Streaming SPARQL retriever that fetches large subgraphs in chunks
pub struct StreamingSparqlRetriever<S: SparqlEngineTrait> {
    engine: Arc<S>,
    config: StreamingSparqlConfig,
}

impl<S: SparqlEngineTrait + 'static> StreamingSparqlRetriever<S> {
    /// Create a new streaming retriever
    pub fn new(engine: Arc<S>, config: StreamingSparqlConfig) -> Self {
        Self { engine, config }
    }

    /// Create with default config
    pub fn with_defaults(engine: Arc<S>) -> Self {
        Self::new(engine, StreamingSparqlConfig::default())
    }

    /// Begin streaming a subgraph for the given seed entities.
    ///
    /// Returns a [`SubgraphStream`] that yields pages of triples as they arrive.
    /// Backpressure is applied via the mpsc channel: the producer pauses when
    /// the consumer is slow (channel full).
    pub fn stream_subgraph(&self, seeds: Vec<ScoredEntity>) -> SubgraphStream {
        let (tx, rx) = mpsc::channel(self.config.channel_capacity);

        let engine = Arc::clone(&self.engine);
        let config = self.config.clone();

        tokio::spawn(async move {
            let semaphore = Arc::new(Semaphore::new(config.max_concurrency));
            let filtered_seeds: Vec<ScoredEntity> = seeds
                .into_iter()
                .filter(|s| s.score >= config.min_seed_score)
                .collect();

            for seed in filtered_seeds {
                let seed_uri = seed.uri.clone();
                let mut offset = 0usize;
                let mut cumulative = 0usize;
                let mut seen: HashSet<(String, String, String)> = HashSet::new();

                loop {
                    // Acquire semaphore slot for concurrency control
                    let permit = match semaphore.clone().acquire_owned().await {
                        Ok(p) => p,
                        Err(e) => {
                            let _ = tx
                                .send(Err(GraphRAGError::InternalError(format!(
                                    "Semaphore acquire failed: {e}"
                                ))))
                                .await;
                            return;
                        }
                    };

                    let sparql = build_expansion_query(
                        &seed_uri,
                        config.expansion_hops,
                        config.page_size,
                        offset,
                    );

                    let raw_triples = engine.construct(&sparql).await;
                    drop(permit); // Release concurrency slot

                    let raw_triples = match raw_triples {
                        Ok(t) => t,
                        Err(e) => {
                            warn!("SPARQL page fetch failed for {seed_uri}: {e}");
                            let _ = tx
                                .send(Err(GraphRAGError::SparqlError(format!(
                                    "Page fetch error for {seed_uri}: {e}"
                                ))))
                                .await;
                            break;
                        }
                    };

                    let page_len = raw_triples.len();
                    debug!(
                        seed = %seed_uri,
                        offset,
                        fetched = page_len,
                        "Fetched SPARQL page"
                    );

                    // Empty page means we have exhausted the result set (e.g. the
                    // previous page was exactly page_size triples). Do not emit an
                    // empty TriplePage – just stop.
                    if page_len == 0 {
                        break;
                    }

                    // Deduplication
                    let triples: Vec<Triple> = if config.deduplicate {
                        raw_triples
                            .into_iter()
                            .filter(|t| {
                                let key =
                                    (t.subject.clone(), t.predicate.clone(), t.object.clone());
                                seen.insert(key)
                            })
                            .collect()
                    } else {
                        raw_triples
                    };

                    cumulative += triples.len();

                    // Apply max_total_triples ceiling
                    let (triples, is_last) =
                        if config.max_total_triples > 0 && cumulative >= config.max_total_triples {
                            let excess = cumulative - config.max_total_triples;
                            let capped_len = triples.len().saturating_sub(excess);
                            let mut t = triples;
                            t.truncate(capped_len);
                            (t, true)
                        } else {
                            let exhausted = page_len < config.page_size;
                            (triples, exhausted)
                        };

                    let page = TriplePage {
                        triples,
                        page_index: offset / config.page_size,
                        cumulative_count: cumulative,
                        is_last_page: is_last,
                        seed_uri: seed_uri.clone(),
                    };

                    // Send page (blocks if channel is full – backpressure)
                    if tx.send(Ok(page)).await.is_err() {
                        // Receiver dropped – stop streaming
                        return;
                    }

                    if is_last {
                        break;
                    }

                    offset += config.page_size;
                }
            }
        });

        SubgraphStream { receiver: rx }
    }

    /// Convenience method: stream and collect into a Vec, returning error on failure.
    pub async fn collect_subgraph(&self, seeds: Vec<ScoredEntity>) -> GraphRAGResult<Vec<Triple>> {
        self.stream_subgraph(seeds).collect_all().await
    }
}

/// Build a paginated SPARQL CONSTRUCT query for a single seed entity
fn build_expansion_query(seed_uri: &str, hops: usize, limit: usize, offset: usize) -> String {
    // We produce a conservative CONSTRUCT with outgoing and incoming triples
    // for up to `hops` levels via SPARQL property paths.
    let path = if hops <= 1 {
        "?p".to_string()
    } else {
        // SPARQL 1.1 property path: 1 to `hops` arbitrary steps
        format!("(:|!:){{1,{}}}", hops)
    };

    format!(
        r#"
CONSTRUCT {{
    <{seed}> ?p1 ?o1 .
    ?s1 ?p2 <{seed}> .
    ?o1 {path} ?o2 .
}}
WHERE {{
    {{
        <{seed}> ?p1 ?o1 .
    }} UNION {{
        ?s1 ?p2 <{seed}> .
    }} UNION {{
        <{seed}> ?p_mid ?mid .
        ?mid {path} ?o2 .
    }}
}}
LIMIT {limit}
OFFSET {offset}
"#,
        seed = seed_uri,
        path = path,
        limit = limit,
        offset = offset,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphRAGResult, ScoreSource, SparqlEngineTrait, Triple};
    use async_trait::async_trait;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Mock SPARQL engine that returns deterministic triple pages
    struct MockSparql {
        triples: Vec<Triple>,
        page_call_count: Arc<AtomicUsize>,
    }

    impl MockSparql {
        fn new(triples: Vec<Triple>) -> Self {
            Self {
                triples,
                page_call_count: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    #[async_trait]
    impl SparqlEngineTrait for MockSparql {
        async fn select(&self, _query: &str) -> GraphRAGResult<Vec<HashMap<String, String>>> {
            Ok(vec![])
        }

        async fn ask(&self, _query: &str) -> GraphRAGResult<bool> {
            Ok(false)
        }

        async fn construct(&self, query: &str) -> GraphRAGResult<Vec<Triple>> {
            self.page_call_count.fetch_add(1, Ordering::Relaxed);

            // Parse OFFSET from the query to simulate pagination
            let offset: usize = query
                .lines()
                .find(|l| l.trim_start().starts_with("OFFSET"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);

            let limit: usize = query
                .lines()
                .find(|l| l.trim_start().starts_with("LIMIT"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|s| s.parse().ok())
                .unwrap_or(1000);

            let slice: Vec<Triple> = self
                .triples
                .iter()
                .skip(offset)
                .take(limit)
                .cloned()
                .collect();

            Ok(slice)
        }
    }

    fn make_seed(uri: &str) -> ScoredEntity {
        ScoredEntity {
            uri: uri.to_string(),
            score: 0.9,
            source: ScoreSource::Vector,
            metadata: HashMap::new(),
        }
    }

    fn make_triples(n: usize) -> Vec<Triple> {
        (0..n)
            .map(|i| {
                Triple::new(
                    format!("http://s/{i}"),
                    "http://p/rel",
                    format!("http://o/{i}"),
                )
            })
            .collect()
    }

    #[tokio::test]
    async fn test_stream_collects_all_triples() {
        let triples = make_triples(50);
        let engine = Arc::new(MockSparql::new(triples.clone()));
        let config = StreamingSparqlConfig {
            page_size: 20,
            max_total_triples: 100,
            channel_capacity: 8,
            max_concurrency: 2,
            deduplicate: false,
            min_seed_score: 0.0,
            expansion_hops: 1,
        };
        let retriever = StreamingSparqlRetriever::new(engine, config);
        let seeds = vec![make_seed("http://example.org/seed1")];
        let collected = retriever
            .collect_subgraph(seeds)
            .await
            .expect("should succeed");
        assert_eq!(collected.len(), 50);
    }

    #[tokio::test]
    async fn test_stream_respects_max_total_triples() {
        let triples = make_triples(3_000);
        let engine = Arc::new(MockSparql::new(triples));
        let config = StreamingSparqlConfig {
            page_size: 1_000,
            max_total_triples: 2_500,
            channel_capacity: 16,
            max_concurrency: 1,
            deduplicate: false,
            min_seed_score: 0.0,
            expansion_hops: 1,
        };
        let retriever = StreamingSparqlRetriever::new(engine, config);
        let seeds = vec![make_seed("http://example.org/seed1")];
        let collected = retriever
            .collect_subgraph(seeds)
            .await
            .expect("should succeed");
        assert!(
            collected.len() <= 2_500,
            "Expected at most 2500 triples, got {}",
            collected.len()
        );
    }

    #[tokio::test]
    async fn test_stream_deduplicates() {
        // All triples are identical – deduplication should keep only 1
        let triple = Triple::new("http://s", "http://p", "http://o");
        let triples: Vec<Triple> = vec![triple.clone(); 100];
        let engine = Arc::new(MockSparql::new(triples));
        let config = StreamingSparqlConfig {
            page_size: 50,
            max_total_triples: 200,
            channel_capacity: 8,
            max_concurrency: 1,
            deduplicate: true,
            min_seed_score: 0.0,
            expansion_hops: 1,
        };
        let retriever = StreamingSparqlRetriever::new(engine, config);
        let seeds = vec![make_seed("http://example.org/seed1")];
        let collected = retriever
            .collect_subgraph(seeds)
            .await
            .expect("should succeed");
        assert_eq!(collected.len(), 1);
    }

    #[tokio::test]
    async fn test_stream_empty_seeds() {
        let engine = Arc::new(MockSparql::new(vec![]));
        let retriever = StreamingSparqlRetriever::with_defaults(engine);
        let collected = retriever
            .collect_subgraph(vec![])
            .await
            .expect("should succeed");
        assert!(collected.is_empty());
    }

    #[tokio::test]
    async fn test_stream_filters_low_score_seeds() {
        let triples = make_triples(10);
        let engine = Arc::new(MockSparql::new(triples));
        let config = StreamingSparqlConfig {
            min_seed_score: 0.8,
            ..Default::default()
        };
        let retriever = StreamingSparqlRetriever::new(engine, config);
        // Seed with score 0.5 should be filtered out
        let seeds = vec![ScoredEntity {
            uri: "http://seed".to_string(),
            score: 0.5,
            source: ScoreSource::Vector,
            metadata: HashMap::new(),
        }];
        let collected = retriever
            .collect_subgraph(seeds)
            .await
            .expect("should succeed");
        assert!(collected.is_empty());
    }

    #[tokio::test]
    async fn test_stream_multiple_seeds() {
        // Two seeds, each returning 5 distinct triples
        let seed1_triples: Vec<Triple> = (0..5)
            .map(|i| {
                Triple::new(
                    format!("http://s1/{i}"),
                    "http://p",
                    format!("http://o1/{i}"),
                )
            })
            .collect();
        let seed2_triples: Vec<Triple> = (0..5)
            .map(|i| {
                Triple::new(
                    format!("http://s2/{i}"),
                    "http://p",
                    format!("http://o2/{i}"),
                )
            })
            .collect();
        let mut all_triples = seed1_triples;
        all_triples.extend(seed2_triples);

        let engine = Arc::new(MockSparql::new(all_triples));
        let config = StreamingSparqlConfig {
            page_size: 100,
            max_total_triples: 100,
            channel_capacity: 8,
            max_concurrency: 2,
            deduplicate: true,
            min_seed_score: 0.0,
            expansion_hops: 1,
        };
        let retriever = StreamingSparqlRetriever::new(engine, config);
        let seeds = vec![
            make_seed("http://example.org/s1"),
            make_seed("http://example.org/s2"),
        ];
        let collected = retriever
            .collect_subgraph(seeds)
            .await
            .expect("should succeed");
        // Each seed gets its own deduplicated set of 10 triples
        assert!(!collected.is_empty());
    }

    #[tokio::test]
    async fn test_stream_pagination_calls() {
        let triples = make_triples(250);
        let call_count = Arc::new(AtomicUsize::new(0));
        let engine = MockSparql {
            triples: triples.clone(),
            page_call_count: Arc::clone(&call_count),
        };
        let engine = Arc::new(engine);
        let local_count = Arc::clone(&engine.page_call_count);

        let config = StreamingSparqlConfig {
            page_size: 100,
            max_total_triples: 0, // no cap
            channel_capacity: 16,
            max_concurrency: 1,
            deduplicate: false,
            min_seed_score: 0.0,
            expansion_hops: 1,
        };
        let retriever = StreamingSparqlRetriever::new(engine, config);
        let seeds = vec![make_seed("http://example.org/seed")];
        let collected = retriever
            .collect_subgraph(seeds)
            .await
            .expect("should succeed");
        assert_eq!(collected.len(), 250);
        // 250 triples with page_size=100 → 3 pages
        assert_eq!(local_count.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn test_next_page_interface() {
        let triples = make_triples(30);
        let engine = Arc::new(MockSparql::new(triples));
        let config = StreamingSparqlConfig {
            page_size: 10,
            max_total_triples: 0,
            channel_capacity: 8,
            max_concurrency: 1,
            deduplicate: false,
            min_seed_score: 0.0,
            expansion_hops: 1,
        };
        let retriever = StreamingSparqlRetriever::new(engine, config);
        let seeds = vec![make_seed("http://example.org/seed")];
        let mut stream = retriever.stream_subgraph(seeds);

        let mut pages_received = 0usize;
        let mut total_triples = 0usize;
        while let Some(page_result) = stream.next_page().await {
            let page = page_result.expect("should succeed");
            total_triples += page.triples.len();
            pages_received += 1;
            assert_eq!(page.page_index, pages_received - 1);
        }
        assert_eq!(total_triples, 30);
        assert_eq!(pages_received, 3);
    }

    #[test]
    fn test_build_expansion_query_single_hop() {
        let q = build_expansion_query("http://example.org/e", 1, 100, 0);
        assert!(q.contains("<http://example.org/e>"));
        assert!(q.contains("LIMIT 100"));
        assert!(q.contains("OFFSET 0"));
    }

    #[test]
    fn test_build_expansion_query_multi_hop() {
        let q = build_expansion_query("http://example.org/e", 3, 500, 1000);
        assert!(q.contains("LIMIT 500"));
        assert!(q.contains("OFFSET 1000"));
        assert!(q.contains("{1,3}"));
    }

    #[test]
    fn test_config_defaults() {
        let cfg = StreamingSparqlConfig::default();
        assert_eq!(cfg.page_size, 1_000);
        assert_eq!(cfg.max_concurrency, 4);
        assert!(cfg.deduplicate);
    }
}

// ─── Additional tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod additional_tests {
    use super::*;
    use crate::{GraphRAGResult, ScoreSource, SparqlEngineTrait, Triple};
    use async_trait::async_trait;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct MockSparql {
        triples: Vec<Triple>,
        page_call_count: Arc<AtomicUsize>,
    }

    impl MockSparql {
        fn new(triples: Vec<Triple>) -> Self {
            Self {
                triples,
                page_call_count: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    #[async_trait]
    impl SparqlEngineTrait for MockSparql {
        async fn select(&self, _query: &str) -> GraphRAGResult<Vec<HashMap<String, String>>> {
            Ok(vec![])
        }
        async fn ask(&self, _query: &str) -> GraphRAGResult<bool> {
            Ok(false)
        }
        async fn construct(&self, query: &str) -> GraphRAGResult<Vec<Triple>> {
            self.page_call_count.fetch_add(1, Ordering::Relaxed);
            let offset: usize = query
                .lines()
                .find(|l| l.trim_start().starts_with("OFFSET"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let limit: usize = query
                .lines()
                .find(|l| l.trim_start().starts_with("LIMIT"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|s| s.parse().ok())
                .unwrap_or(1000);
            Ok(self
                .triples
                .iter()
                .skip(offset)
                .take(limit)
                .cloned()
                .collect())
        }
    }

    fn make_seed(uri: &str, score: f64) -> ScoredEntity {
        ScoredEntity {
            uri: uri.to_string(),
            score,
            source: ScoreSource::Vector,
            metadata: HashMap::new(),
        }
    }

    fn make_triples(n: usize) -> Vec<Triple> {
        (0..n)
            .map(|i| {
                Triple::new(
                    format!("http://s/{i}"),
                    "http://p/rel",
                    format!("http://o/{i}"),
                )
            })
            .collect()
    }

    // ── Config / builder tests ────────────────────────────────────────────

    #[test]
    fn test_config_channel_capacity_default() {
        let cfg = StreamingSparqlConfig::default();
        assert_eq!(cfg.channel_capacity, 512);
    }

    #[test]
    fn test_config_expansion_hops_default() {
        let cfg = StreamingSparqlConfig::default();
        assert_eq!(cfg.expansion_hops, 2);
    }

    #[test]
    fn test_config_min_seed_score_default() {
        let cfg = StreamingSparqlConfig::default();
        assert!((cfg.min_seed_score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_max_total_triples_default() {
        let cfg = StreamingSparqlConfig::default();
        assert_eq!(cfg.max_total_triples, 50_000);
    }

    // ── Build expansion query tests ───────────────────────────────────────

    #[test]
    fn test_build_expansion_query_hop2() {
        let q = build_expansion_query("http://example.org/e", 2, 200, 400);
        assert!(q.contains("LIMIT 200"));
        assert!(q.contains("OFFSET 400"));
        assert!(q.contains("{1,2}"));
    }

    #[test]
    fn test_build_expansion_query_hop1_uses_p() {
        let q = build_expansion_query("http://example.org/e", 1, 50, 0);
        // For 1 hop, path is just ?p
        assert!(q.contains("?p"));
    }

    #[test]
    fn test_build_expansion_query_contains_union() {
        let q = build_expansion_query("http://example.org/e", 2, 100, 0);
        assert!(q.contains("UNION"));
    }

    #[test]
    fn test_build_expansion_query_construct_keyword() {
        let q = build_expansion_query("http://example.org/e", 1, 100, 0);
        assert!(q.contains("CONSTRUCT"));
        assert!(q.contains("WHERE"));
    }

    // ── Streaming behaviour tests ─────────────────────────────────────────

    #[tokio::test]
    async fn test_stream_zero_triples_engine() {
        let engine = Arc::new(MockSparql::new(vec![]));
        let config = StreamingSparqlConfig {
            page_size: 10,
            max_total_triples: 100,
            channel_capacity: 4,
            max_concurrency: 1,
            deduplicate: false,
            min_seed_score: 0.0,
            expansion_hops: 1,
        };
        let retriever = StreamingSparqlRetriever::new(engine, config);
        let seeds = vec![make_seed("http://seed", 0.9)];
        let collected = retriever
            .collect_subgraph(seeds)
            .await
            .expect("should succeed");
        assert!(collected.is_empty());
    }

    #[tokio::test]
    async fn test_stream_exactly_page_size_triples() {
        // Exactly page_size triples → 1 page (page is full but next page is empty)
        let triples = make_triples(10);
        let engine = Arc::new(MockSparql::new(triples));
        let config = StreamingSparqlConfig {
            page_size: 10,
            max_total_triples: 0,
            channel_capacity: 8,
            max_concurrency: 1,
            deduplicate: false,
            min_seed_score: 0.0,
            expansion_hops: 1,
        };
        let retriever = StreamingSparqlRetriever::new(engine, config);
        let seeds = vec![make_seed("http://seed", 0.9)];
        let collected = retriever
            .collect_subgraph(seeds)
            .await
            .expect("should succeed");
        // Should collect exactly 10 (stops after empty second page)
        assert_eq!(collected.len(), 10);
    }

    #[tokio::test]
    async fn test_stream_deduplicate_across_pages() {
        // First page: triples 0..=9, second page: repeat triples 0..=4 + new 10..=14
        let base_triples = make_triples(15);
        // Interleave repeated ones
        let mut triples = base_triples[0..10].to_vec();
        triples.extend_from_slice(&base_triples[0..5]); // duplicates
        triples.extend_from_slice(&base_triples[10..15]); // new ones
        let engine = Arc::new(MockSparql::new(triples));
        let config = StreamingSparqlConfig {
            page_size: 10,
            max_total_triples: 0,
            channel_capacity: 8,
            max_concurrency: 1,
            deduplicate: true,
            min_seed_score: 0.0,
            expansion_hops: 1,
        };
        let retriever = StreamingSparqlRetriever::new(engine, config);
        let seeds = vec![make_seed("http://seed", 0.9)];
        let collected = retriever
            .collect_subgraph(seeds)
            .await
            .expect("should succeed");
        // 15 unique triples even though 5 were duplicated
        assert_eq!(collected.len(), 15);
    }

    #[tokio::test]
    async fn test_stream_no_deduplicate_counts_duplicates() {
        let triple = Triple::new("http://s", "http://p", "http://o");
        let triples = vec![triple; 30];
        let engine = Arc::new(MockSparql::new(triples));
        let config = StreamingSparqlConfig {
            page_size: 15,
            max_total_triples: 0,
            channel_capacity: 8,
            max_concurrency: 1,
            deduplicate: false,
            min_seed_score: 0.0,
            expansion_hops: 1,
        };
        let retriever = StreamingSparqlRetriever::new(engine, config);
        let seeds = vec![make_seed("http://seed", 0.9)];
        let collected = retriever
            .collect_subgraph(seeds)
            .await
            .expect("should succeed");
        assert_eq!(collected.len(), 30);
    }

    #[tokio::test]
    async fn test_stream_score_exactly_at_threshold() {
        let triples = make_triples(5);
        let engine = Arc::new(MockSparql::new(triples));
        let config = StreamingSparqlConfig {
            min_seed_score: 0.5,
            ..Default::default()
        };
        let retriever = StreamingSparqlRetriever::new(engine, config);
        // Seed exactly at threshold → should be included (>= comparison)
        let seeds = vec![make_seed("http://seed", 0.5)];
        let collected = retriever
            .collect_subgraph(seeds)
            .await
            .expect("should succeed");
        assert_eq!(collected.len(), 5);
    }

    #[tokio::test]
    async fn test_stream_score_just_below_threshold_filtered() {
        let triples = make_triples(5);
        let engine = Arc::new(MockSparql::new(triples));
        let config = StreamingSparqlConfig {
            min_seed_score: 0.5,
            ..Default::default()
        };
        let retriever = StreamingSparqlRetriever::new(engine, config);
        let seeds = vec![make_seed("http://seed", 0.499)];
        let collected = retriever
            .collect_subgraph(seeds)
            .await
            .expect("should succeed");
        assert!(collected.is_empty());
    }

    #[tokio::test]
    async fn test_stream_pages_have_correct_page_indices() {
        let triples = make_triples(30);
        let engine = Arc::new(MockSparql::new(triples));
        let config = StreamingSparqlConfig {
            page_size: 10,
            max_total_triples: 0,
            channel_capacity: 8,
            max_concurrency: 1,
            deduplicate: false,
            min_seed_score: 0.0,
            expansion_hops: 1,
        };
        let retriever = StreamingSparqlRetriever::new(engine, config);
        let seeds = vec![make_seed("http://seed", 0.9)];
        let mut stream = retriever.stream_subgraph(seeds);
        let mut expected_idx = 0usize;
        while let Some(page_result) = stream.next_page().await {
            let page = page_result.expect("should succeed");
            assert_eq!(page.page_index, expected_idx);
            expected_idx += 1;
        }
        assert_eq!(expected_idx, 3); // 30 triples / page_size=10
    }

    #[tokio::test]
    async fn test_stream_last_page_flag_set() {
        let triples = make_triples(15);
        let engine = Arc::new(MockSparql::new(triples));
        let config = StreamingSparqlConfig {
            page_size: 10,
            max_total_triples: 0,
            channel_capacity: 8,
            max_concurrency: 1,
            deduplicate: false,
            min_seed_score: 0.0,
            expansion_hops: 1,
        };
        let retriever = StreamingSparqlRetriever::new(engine, config);
        let seeds = vec![make_seed("http://seed", 0.9)];
        let mut stream = retriever.stream_subgraph(seeds);
        let mut pages = Vec::new();
        while let Some(page_result) = stream.next_page().await {
            pages.push(page_result.expect("should succeed"));
        }
        // Last page must have is_last_page = true
        assert!(pages.last().map(|p| p.is_last_page).unwrap_or(false));
    }

    #[tokio::test]
    async fn test_stream_cumulative_count_monotone() {
        let triples = make_triples(30);
        let engine = Arc::new(MockSparql::new(triples));
        let config = StreamingSparqlConfig {
            page_size: 10,
            max_total_triples: 0,
            channel_capacity: 8,
            max_concurrency: 1,
            deduplicate: false,
            min_seed_score: 0.0,
            expansion_hops: 1,
        };
        let retriever = StreamingSparqlRetriever::new(engine, config);
        let seeds = vec![make_seed("http://seed", 0.9)];
        let mut stream = retriever.stream_subgraph(seeds);
        let mut prev = 0usize;
        while let Some(page_result) = stream.next_page().await {
            let page = page_result.expect("should succeed");
            assert!(
                page.cumulative_count > prev,
                "cumulative_count should be strictly increasing"
            );
            prev = page.cumulative_count;
        }
    }

    #[tokio::test]
    async fn test_stream_seed_uri_in_page() {
        let triples = make_triples(5);
        let engine = Arc::new(MockSparql::new(triples));
        let config = StreamingSparqlConfig {
            page_size: 10,
            max_total_triples: 0,
            channel_capacity: 4,
            max_concurrency: 1,
            deduplicate: false,
            min_seed_score: 0.0,
            expansion_hops: 1,
        };
        let retriever = StreamingSparqlRetriever::new(engine, config);
        let seeds = vec![make_seed("http://my-seed/42", 0.9)];
        let mut stream = retriever.stream_subgraph(seeds);
        let page = stream
            .next_page()
            .await
            .expect("should succeed")
            .expect("should succeed");
        assert_eq!(page.seed_uri, "http://my-seed/42");
    }

    #[tokio::test]
    async fn test_stream_max_total_ceiling_exact() {
        // 200 triples, cap at exactly 100
        let triples = make_triples(200);
        let engine = Arc::new(MockSparql::new(triples));
        let config = StreamingSparqlConfig {
            page_size: 50,
            max_total_triples: 100,
            channel_capacity: 8,
            max_concurrency: 1,
            deduplicate: false,
            min_seed_score: 0.0,
            expansion_hops: 1,
        };
        let retriever = StreamingSparqlRetriever::new(engine, config);
        let seeds = vec![make_seed("http://seed", 0.9)];
        let collected = retriever
            .collect_subgraph(seeds)
            .await
            .expect("should succeed");
        assert!(collected.len() <= 100);
        assert_eq!(collected.len(), 100);
    }

    #[tokio::test]
    async fn test_with_defaults_builds_retriever() {
        let engine = Arc::new(MockSparql::new(make_triples(3)));
        let retriever = StreamingSparqlRetriever::with_defaults(engine);
        let seeds = vec![make_seed("http://seed", 0.9)];
        let collected = retriever
            .collect_subgraph(seeds)
            .await
            .expect("should succeed");
        assert_eq!(collected.len(), 3);
    }
}
