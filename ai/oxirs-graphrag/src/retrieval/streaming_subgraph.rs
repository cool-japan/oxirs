//! Incremental streaming subgraph retrieval using BFS expansion
//!
//! `StreamingSubgraphRetriever` retrieves large subgraphs batch by batch,
//! expanding outward from a SPARQL query in breadth-first order.
//! Each batch yields a `SubgraphBatch` with metadata and a `is_final` flag.

use crate::{GraphRAGError, GraphRAGResult, SparqlEngineTrait, Triple};
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for incremental subgraph streaming
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Maximum triples returned in a single batch
    pub max_triples_per_batch: usize,
    /// Timeout budget in milliseconds (0 = no limit, enforced per-batch by caller)
    pub timeout_ms: u64,
    /// Maximum BFS expansion depth
    pub max_depth: u8,
    /// Deduplicate triples across batches
    pub deduplicate: bool,
    /// Maximum total triples to deliver (0 = unlimited)
    pub max_total_triples: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            max_triples_per_batch: 500,
            timeout_ms: 30_000,
            max_depth: 3,
            deduplicate: true,
            max_total_triples: 50_000,
        }
    }
}

// ── Batch type ─────────────────────────────────────────────────────────────────

/// A single batch of triples from the streaming retriever
#[derive(Debug, Clone)]
pub struct SubgraphBatch {
    /// Triples in this batch
    pub triples: Vec<Triple>,
    /// Whether this is the final batch (no more data follows)
    pub is_final: bool,
    /// Zero-based batch sequence number
    pub batch_id: usize,
    /// Current BFS depth at which these triples were collected
    pub current_depth: u8,
}

// ── Stream handle ──────────────────────────────────────────────────────────────

/// Handle returned by `retrieve_stream`. Yields `SubgraphBatch` values synchronously.
pub struct SubgraphStream {
    batches: Vec<SubgraphBatch>,
    next_idx: usize,
}

impl SubgraphStream {
    fn new(batches: Vec<SubgraphBatch>) -> Self {
        Self {
            batches,
            next_idx: 0,
        }
    }

    /// Retrieve the next batch, or `None` when exhausted.
    pub fn next_batch(&mut self) -> Option<SubgraphBatch> {
        if self.next_idx < self.batches.len() {
            let batch = self.batches[self.next_idx].clone();
            self.next_idx += 1;
            Some(batch)
        } else {
            None
        }
    }

    /// Collect all batches into a flat `Vec<Triple>`.
    pub fn collect_all(mut self) -> Vec<Triple> {
        let mut out = Vec::new();
        while let Some(batch) = self.next_batch() {
            out.extend(batch.triples);
        }
        out
    }

    /// Total number of batches available
    pub fn batch_count(&self) -> usize {
        self.batches.len()
    }
}

// ── StreamingSubgraphRetriever ─────────────────────────────────────────────────

/// Incrementally retrieves a subgraph by running SPARQL CONSTRUCT queries
/// layer by layer (BFS expansion) and packaging results into fixed-size batches.
pub struct StreamingSubgraphRetriever<S: SparqlEngineTrait> {
    engine: Arc<S>,
    config: StreamConfig,
}

impl<S: SparqlEngineTrait + 'static> StreamingSubgraphRetriever<S> {
    /// Create a new retriever with the given config.
    pub fn new(engine: Arc<S>, config: StreamConfig) -> Self {
        Self { engine, config }
    }

    /// Create with default config.
    pub fn with_defaults(engine: Arc<S>) -> Self {
        Self::new(engine, StreamConfig::default())
    }

    /// Start streaming for the given SPARQL CONSTRUCT query.
    ///
    /// The query is used as the initial seed: its results form depth 0.
    /// Then each distinct object entity in those results is expanded for
    /// subsequent depths, up to `config.max_depth`.
    ///
    /// This is a synchronous method; it runs a Tokio `block_in_place` internally
    /// (or the caller must be in a Tokio runtime context).
    pub fn retrieve_stream(
        &self,
        query: &str,
        config: &StreamConfig,
    ) -> GraphRAGResult<SubgraphStream> {
        // We need an async executor; use a blocking thread via tokio::task::block_in_place
        // or, since tests always run inside Tokio, call a helper.
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|_| GraphRAGError::InternalError("No Tokio runtime available".to_string()))?;

        let engine = Arc::clone(&self.engine);
        let query_owned = query.to_string();
        let config_owned = config.clone();

        // Run the BFS expansion under the existing Tokio handle
        let batches = rt.block_on(run_bfs_expansion(engine, &query_owned, &config_owned))?;

        Ok(SubgraphStream::new(batches))
    }
}

// ── BFS expansion ─────────────────────────────────────────────────────────────

/// Run BFS expansion up to `config.max_depth`, returning packaged `SubgraphBatch`es.
async fn run_bfs_expansion<S: SparqlEngineTrait>(
    engine: Arc<S>,
    initial_query: &str,
    config: &StreamConfig,
) -> GraphRAGResult<Vec<SubgraphBatch>> {
    let mut batches: Vec<SubgraphBatch> = Vec::new();
    let mut seen: HashSet<(String, String, String)> = HashSet::new();
    let mut total_delivered: usize = 0;

    // Depth 0: run the initial CONSTRUCT query
    let initial_triples = engine.construct(initial_query).await?;
    let initial_triples = deduplicate_if(initial_triples, config.deduplicate, &mut seen);

    // Collect frontier entities from depth-0 results (objects that could be expanded)
    let mut frontier: VecDeque<String> = VecDeque::new();
    for t in &initial_triples {
        if t.object.starts_with("http") {
            frontier.push_back(t.object.clone());
        }
    }

    // Package depth-0 triples into batches
    let (new_batches, delivered) = package_into_batches(
        initial_triples,
        0,
        config.max_triples_per_batch,
        config.max_total_triples,
        total_delivered,
        &mut batches,
    );
    let _ = new_batches;
    total_delivered += delivered;

    if config.max_total_triples > 0 && total_delivered >= config.max_total_triples {
        mark_last_batch(&mut batches);
        return Ok(batches);
    }

    // Depths 1..max_depth: expand frontier entities
    for depth in 1..=config.max_depth {
        if frontier.is_empty() {
            break;
        }

        let current_frontier: Vec<String> = frontier.drain(..).collect();
        let mut depth_triples: Vec<Triple> = Vec::new();

        for entity_uri in &current_frontier {
            let expand_query = build_entity_expand_query(entity_uri, 1);
            let raw = engine.construct(&expand_query).await?;
            let filtered = deduplicate_if(raw, config.deduplicate, &mut seen);
            for t in &filtered {
                if t.object.starts_with("http") {
                    frontier.push_back(t.object.clone());
                }
            }
            depth_triples.extend(filtered);

            if config.max_total_triples > 0
                && total_delivered + depth_triples.len() >= config.max_total_triples
            {
                break;
            }
        }

        let (_, delivered) = package_into_batches(
            depth_triples,
            depth,
            config.max_triples_per_batch,
            config.max_total_triples,
            total_delivered,
            &mut batches,
        );
        total_delivered += delivered;

        if config.max_total_triples > 0 && total_delivered >= config.max_total_triples {
            break;
        }
    }

    mark_last_batch(&mut batches);
    Ok(batches)
}

/// Mark the last batch in the list as final.
fn mark_last_batch(batches: &mut [SubgraphBatch]) {
    if let Some(last) = batches.last_mut() {
        last.is_final = true;
    }
}

/// Deduplicate triples if enabled, updating the seen set in place.
fn deduplicate_if(
    triples: Vec<Triple>,
    dedup: bool,
    seen: &mut HashSet<(String, String, String)>,
) -> Vec<Triple> {
    if !dedup {
        return triples;
    }
    triples
        .into_iter()
        .filter(|t| seen.insert((t.subject.clone(), t.predicate.clone(), t.object.clone())))
        .collect()
}

/// Pack `triples` into batches of `batch_size`, respecting `max_total`.
/// Returns (number of batches created, number of triples delivered).
fn package_into_batches(
    triples: Vec<Triple>,
    depth: u8,
    batch_size: usize,
    max_total: usize,
    already_delivered: usize,
    out: &mut Vec<SubgraphBatch>,
) -> (usize, usize) {
    let mut remaining = triples;
    if max_total > 0 && already_delivered + remaining.len() > max_total {
        remaining.truncate(max_total - already_delivered);
    }

    let mut total_delivered = 0usize;
    let mut batches_created = 0usize;
    let mut offset = 0usize;

    while offset < remaining.len() {
        let end = (offset + batch_size).min(remaining.len());
        let chunk: Vec<Triple> = remaining[offset..end].to_vec();
        let chunk_len = chunk.len();
        let batch_id = out.len();

        out.push(SubgraphBatch {
            triples: chunk,
            is_final: false, // will be set by mark_last_batch
            batch_id,
            current_depth: depth,
        });

        total_delivered += chunk_len;
        batches_created += 1;
        offset = end;
    }

    (batches_created, total_delivered)
}

/// Build a 1-hop CONSTRUCT expansion query for a single entity.
fn build_entity_expand_query(entity_uri: &str, _hops: usize) -> String {
    format!(
        r#"CONSTRUCT {{ <{e}> ?p ?o . ?s ?p2 <{e}> . }}
WHERE {{ {{ <{e}> ?p ?o . }} UNION {{ ?s ?p2 <{e}> . }} }}"#,
        e = entity_uri,
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphRAGResult, SparqlEngineTrait, Triple};
    use async_trait::async_trait;
    use std::collections::HashMap;
    use std::sync::Arc;

    /// Minimal mock SPARQL engine
    struct MockEngine {
        triples: Vec<Triple>,
    }

    impl MockEngine {
        fn new(triples: Vec<Triple>) -> Arc<Self> {
            Arc::new(Self { triples })
        }
    }

    #[async_trait]
    impl SparqlEngineTrait for MockEngine {
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

    fn make_triples(n: usize) -> Vec<Triple> {
        (0..n)
            .map(|i| Triple::new(format!("http://s/{i}"), "http://p", format!("http://o/{i}")))
            .collect()
    }

    fn run<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Runtime::new()
            .expect("should succeed")
            .block_on(f)
    }

    // ── StreamConfig default tests ──────────────────────────────────────────

    #[test]
    fn test_stream_config_defaults() {
        let cfg = StreamConfig::default();
        assert_eq!(cfg.max_triples_per_batch, 500);
        assert_eq!(cfg.timeout_ms, 30_000);
        assert_eq!(cfg.max_depth, 3);
        assert!(cfg.deduplicate);
        assert_eq!(cfg.max_total_triples, 50_000);
    }

    // ── SubgraphBatch field tests ───────────────────────────────────────────

    #[test]
    fn test_subgraph_batch_fields() {
        let batch = SubgraphBatch {
            triples: make_triples(5),
            is_final: true,
            batch_id: 2,
            current_depth: 1,
        };
        assert_eq!(batch.triples.len(), 5);
        assert!(batch.is_final);
        assert_eq!(batch.batch_id, 2);
        assert_eq!(batch.current_depth, 1);
    }

    // ── SubgraphStream collect_all ──────────────────────────────────────────

    #[test]
    fn test_stream_collect_all() {
        let batches = vec![
            SubgraphBatch {
                triples: make_triples(3),
                is_final: false,
                batch_id: 0,
                current_depth: 0,
            },
            SubgraphBatch {
                triples: make_triples(2),
                is_final: true,
                batch_id: 1,
                current_depth: 1,
            },
        ];
        let stream = SubgraphStream::new(batches);
        let all = stream.collect_all();
        assert_eq!(all.len(), 5);
    }

    // ── SubgraphStream next_batch ───────────────────────────────────────────

    #[test]
    fn test_stream_next_batch_exhaustion() {
        let batches = vec![SubgraphBatch {
            triples: make_triples(1),
            is_final: true,
            batch_id: 0,
            current_depth: 0,
        }];
        let mut stream = SubgraphStream::new(batches);
        assert!(stream.next_batch().is_some());
        assert!(stream.next_batch().is_none());
    }

    // ── SubgraphStream batch_count ──────────────────────────────────────────

    #[test]
    fn test_stream_batch_count() {
        let batches = (0..5)
            .map(|i| SubgraphBatch {
                triples: make_triples(1),
                is_final: i == 4,
                batch_id: i,
                current_depth: 0,
            })
            .collect();
        let stream = SubgraphStream::new(batches);
        assert_eq!(stream.batch_count(), 5);
    }

    // ── BFS expansion: depth 0 ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_bfs_depth0_basic() {
        let triples = make_triples(10);
        let engine: Arc<MockEngine> = MockEngine::new(triples);
        let config = StreamConfig {
            max_triples_per_batch: 100,
            max_depth: 0,
            deduplicate: false,
            max_total_triples: 0,
            ..Default::default()
        };
        let batches = run_bfs_expansion(engine, "CONSTRUCT {}", &config)
            .await
            .expect("should succeed");
        // Only the initial query at depth 0
        let total: usize = batches.iter().map(|b| b.triples.len()).sum();
        assert_eq!(total, 10);
        assert!(batches.last().expect("should succeed").is_final);
    }

    // ── BFS expansion: max_total_triples cap ────────────────────────────────

    #[tokio::test]
    async fn test_bfs_max_total_cap() {
        let engine = MockEngine::new(make_triples(200));
        let config = StreamConfig {
            max_triples_per_batch: 100,
            max_depth: 0,
            deduplicate: false,
            max_total_triples: 50,
            ..Default::default()
        };
        let batches = run_bfs_expansion(engine, "CONSTRUCT {}", &config)
            .await
            .expect("should succeed");
        let total: usize = batches.iter().map(|b| b.triples.len()).sum();
        assert!(total <= 50);
    }

    // ── BFS expansion: deduplication ───────────────────────────────────────

    #[tokio::test]
    async fn test_bfs_deduplication() {
        let triple = Triple::new("http://s", "http://p", "http://o");
        let engine = MockEngine::new(vec![triple; 50]);
        let config = StreamConfig {
            max_triples_per_batch: 100,
            max_depth: 0,
            deduplicate: true,
            max_total_triples: 0,
            ..Default::default()
        };
        let batches = run_bfs_expansion(engine, "CONSTRUCT {}", &config)
            .await
            .expect("should succeed");
        let total: usize = batches.iter().map(|b| b.triples.len()).sum();
        assert_eq!(total, 1);
    }

    // ── BFS expansion: no deduplication ───────────────────────────────────

    #[tokio::test]
    async fn test_bfs_no_deduplication_counts_all() {
        let triple = Triple::new("http://s", "http://p", "http://o");
        let engine = MockEngine::new(vec![triple; 20]);
        let config = StreamConfig {
            max_triples_per_batch: 100,
            max_depth: 0,
            deduplicate: false,
            max_total_triples: 0,
            ..Default::default()
        };
        let batches = run_bfs_expansion(engine, "CONSTRUCT {}", &config)
            .await
            .expect("should succeed");
        let total: usize = batches.iter().map(|b| b.triples.len()).sum();
        assert_eq!(total, 20);
    }

    // ── Package into batches: batch_size splits ─────────────────────────────

    #[test]
    fn test_package_into_batches_splits_correctly() {
        let triples = make_triples(25);
        let mut out: Vec<SubgraphBatch> = Vec::new();
        let (batches_created, delivered) = package_into_batches(triples, 0, 10, 0, 0, &mut out);
        assert_eq!(batches_created, 3); // 10 + 10 + 5
        assert_eq!(delivered, 25);
        assert_eq!(out.len(), 3);
    }

    // ── Package into batches: max_total truncation ──────────────────────────

    #[test]
    fn test_package_into_batches_respects_max_total() {
        let triples = make_triples(100);
        let mut out: Vec<SubgraphBatch> = Vec::new();
        let (_, delivered) = package_into_batches(triples, 0, 50, 30, 0, &mut out);
        assert!(delivered <= 30);
        let total: usize = out.iter().map(|b| b.triples.len()).sum();
        assert!(total <= 30);
    }

    // ── mark_last_batch ─────────────────────────────────────────────────────

    #[test]
    fn test_mark_last_batch_sets_is_final() {
        let mut batches = vec![
            SubgraphBatch {
                triples: vec![],
                is_final: false,
                batch_id: 0,
                current_depth: 0,
            },
            SubgraphBatch {
                triples: vec![],
                is_final: false,
                batch_id: 1,
                current_depth: 0,
            },
        ];
        mark_last_batch(&mut batches);
        assert!(!batches[0].is_final);
        assert!(batches[1].is_final);
    }

    // ── build_entity_expand_query ───────────────────────────────────────────

    #[test]
    fn test_build_entity_expand_query_contains_uri() {
        let q = build_entity_expand_query("http://example.org/e", 1);
        assert!(q.contains("http://example.org/e"));
        assert!(q.contains("CONSTRUCT"));
    }

    // ── deduplicate_if ──────────────────────────────────────────────────────

    #[test]
    fn test_deduplicate_if_disabled() {
        let triples = vec![
            Triple::new("http://s", "http://p", "http://o"),
            Triple::new("http://s", "http://p", "http://o"),
        ];
        let mut seen = HashSet::new();
        let result = deduplicate_if(triples, false, &mut seen);
        assert_eq!(result.len(), 2);
        assert!(seen.is_empty()); // Not updated when disabled
    }

    #[test]
    fn test_deduplicate_if_enabled_removes_dupes() {
        let triples = vec![
            Triple::new("http://s", "http://p", "http://o"),
            Triple::new("http://s", "http://p", "http://o"),
            Triple::new("http://s2", "http://p", "http://o"),
        ];
        let mut seen = HashSet::new();
        let result = deduplicate_if(triples, true, &mut seen);
        assert_eq!(result.len(), 2);
    }

    // ── StreamingSubgraphRetriever with_defaults ────────────────────────────

    #[tokio::test]
    async fn test_retriever_with_defaults() {
        let engine = MockEngine::new(make_triples(5));
        let retriever = StreamingSubgraphRetriever::with_defaults(engine);
        assert_eq!(retriever.config.max_depth, 3);
    }

    // ── Empty engine returns empty stream ───────────────────────────────────

    #[tokio::test]
    async fn test_bfs_empty_engine_returns_empty() {
        let engine = MockEngine::new(vec![]);
        let config = StreamConfig {
            max_triples_per_batch: 10,
            max_depth: 0,
            deduplicate: false,
            max_total_triples: 0,
            ..Default::default()
        };
        let batches = run_bfs_expansion(engine, "CONSTRUCT {}", &config)
            .await
            .expect("should succeed");
        let total: usize = batches.iter().map(|b| b.triples.len()).sum();
        assert_eq!(total, 0);
    }

    // ── Batch IDs are sequential ────────────────────────────────────────────

    #[tokio::test]
    async fn test_bfs_batch_ids_sequential() {
        let engine = MockEngine::new(make_triples(30));
        let config = StreamConfig {
            max_triples_per_batch: 10,
            max_depth: 0,
            deduplicate: false,
            max_total_triples: 0,
            ..Default::default()
        };
        let batches = run_bfs_expansion(engine, "CONSTRUCT {}", &config)
            .await
            .expect("should succeed");
        for (expected_id, batch) in batches.iter().enumerate() {
            assert_eq!(batch.batch_id, expected_id);
        }
    }

    // ── Only last batch has is_final = true ─────────────────────────────────

    #[tokio::test]
    async fn test_bfs_only_last_batch_is_final() {
        let engine = MockEngine::new(make_triples(25));
        let config = StreamConfig {
            max_triples_per_batch: 10,
            max_depth: 0,
            deduplicate: false,
            max_total_triples: 0,
            ..Default::default()
        };
        let batches = run_bfs_expansion(engine, "CONSTRUCT {}", &config)
            .await
            .expect("should succeed");
        for (i, batch) in batches.iter().enumerate() {
            if i < batches.len() - 1 {
                assert!(!batch.is_final, "Batch {i} should not be final");
            } else {
                assert!(batch.is_final, "Last batch should be final");
            }
        }
    }

    // ── depth-0 batches have current_depth = 0 ──────────────────────────────

    #[tokio::test]
    async fn test_bfs_depth0_batches_have_depth_zero() {
        let engine = MockEngine::new(make_triples(10));
        let config = StreamConfig {
            max_triples_per_batch: 5,
            max_depth: 0,
            deduplicate: false,
            max_total_triples: 0,
            ..Default::default()
        };
        let batches = run_bfs_expansion(engine, "CONSTRUCT {}", &config)
            .await
            .expect("should succeed");
        for batch in &batches {
            assert_eq!(batch.current_depth, 0);
        }
    }

    // ── StreamingSubgraphRetriever::new sets config correctly ───────────────

    #[test]
    fn test_retriever_new_config() {
        let engine = MockEngine::new(vec![]);
        let config = StreamConfig {
            max_triples_per_batch: 42,
            ..Default::default()
        };
        let retriever = StreamingSubgraphRetriever::new(engine, config);
        assert_eq!(retriever.config.max_triples_per_batch, 42);
    }
}
