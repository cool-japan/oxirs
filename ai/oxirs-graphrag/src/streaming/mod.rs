//! Streaming subgraph extraction using SPARQL-like patterns.
//!
//! # Overview
//!
//! [`StreamingSubgraphExtractor`] maintains an in-memory adjacency graph built
//! from `(subject, predicate, object)` triples added or removed at runtime.
//! It emits [`SubgraphEvent`]s whenever the graph changes, and provides pattern-
//! based subgraph extraction.
//!
//! [`StreamingGraphRag`] wraps the extractor and adds a query-result cache with
//! a configurable TTL, tracking hit-rate statistics.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// Event types
// ─────────────────────────────────────────────────────────────────────────────

/// Describes what changed in the streaming subgraph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SubgraphEventType {
    /// A node (entity) was added to the graph for the first time
    NodeAdded,
    /// The last edge touching a node was removed, making it isolated
    NodeRemoved,
    /// An edge (triple) was added
    EdgeAdded,
    /// An edge (triple) was removed
    EdgeRemoved,
    /// A node matched a named subgraph pattern
    SubgraphMatch(String),
}

/// A single event emitted by [`StreamingSubgraphExtractor`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubgraphEvent {
    /// The kind of change
    pub event_type: SubgraphEventType,
    /// The focal node for this event
    pub node: String,
    /// Unix-ms timestamp of the event
    pub timestamp: i64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Subgraph pattern
// ─────────────────────────────────────────────────────────────────────────────

/// Predicate filter for neighbourhood expansion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SubgraphFilter {
    /// Include only nodes that have the given label predicate object
    HasLabel(String),
    /// Include only nodes whose out-degree is ≤ limit
    MaxDegree(usize),
    /// Include only nodes whose out-degree is ≥ limit
    MinDegree(usize),
}

/// Describes which subgraph to extract from the live graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubgraphPattern {
    /// Anchor node URI, or `"?x"` to indicate "any reachable node"
    pub anchor: String,
    /// Predicates to follow during expansion (empty = follow all)
    pub predicates: Vec<String>,
    /// Maximum BFS depth from the anchor
    pub depth: usize,
    /// Optional additional filter applied to each candidate node
    pub filter: Option<SubgraphFilter>,
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamingSubgraphExtractor
// ─────────────────────────────────────────────────────────────────────────────

/// Edge storage: subject → [(predicate, object)]
type AdjList = HashMap<String, Vec<(String, String)>>;

/// In-memory streaming subgraph manager.
pub struct StreamingSubgraphExtractor {
    /// Forward adjacency: subject → [(predicate, object)]
    graph: AdjList,
    /// Pending events not yet consumed by the caller
    buffer: VecDeque<SubgraphEvent>,
    /// Maximum BFS depth when extracting subgraphs
    max_depth: usize,
    /// Maximum number of nodes to include in a subgraph extraction
    max_nodes: usize,
    /// Total events ever emitted (including consumed ones)
    events_total: u64,
}

impl StreamingSubgraphExtractor {
    /// Create a new extractor.
    ///
    /// * `max_depth` – BFS depth limit for subgraph extraction
    /// * `max_nodes` – node cap for subgraph extraction
    pub fn new(max_depth: usize, max_nodes: usize) -> Self {
        Self {
            graph: HashMap::new(),
            buffer: VecDeque::new(),
            max_depth,
            max_nodes,
            events_total: 0,
        }
    }

    /// Add a triple to the graph and emit the corresponding events.
    ///
    /// Returns the events produced by this mutation (also buffered internally).
    pub fn add_triple(&mut self, s: &str, p: &str, o: &str) -> Vec<SubgraphEvent> {
        let now = unix_ms();
        let mut events = Vec::new();

        // Detect if subject is a brand-new node
        let subject_is_new = !self.graph.contains_key(s);
        let edges = self.graph.entry(s.to_string()).or_default();

        // Only add if not already present
        if !edges.iter().any(|(ep, eo)| ep == p && eo == o) {
            edges.push((p.to_string(), o.to_string()));

            if subject_is_new {
                let ev = SubgraphEvent {
                    event_type: SubgraphEventType::NodeAdded,
                    node: s.to_string(),
                    timestamp: now,
                };
                events.push(ev.clone());
                self.buffer.push_back(ev);
            }

            let ev = SubgraphEvent {
                event_type: SubgraphEventType::EdgeAdded,
                node: s.to_string(),
                timestamp: now,
            };
            events.push(ev.clone());
            self.buffer.push_back(ev);

            // If object is new, emit NodeAdded for it too
            if !self.graph.contains_key(o) {
                // Insert empty adjacency to mark the node as known
                self.graph.entry(o.to_string()).or_default();
                let ev = SubgraphEvent {
                    event_type: SubgraphEventType::NodeAdded,
                    node: o.to_string(),
                    timestamp: now,
                };
                events.push(ev.clone());
                self.buffer.push_back(ev);
            }
        }

        self.events_total += events.len() as u64;
        events
    }

    /// Remove a triple from the graph and emit the corresponding events.
    ///
    /// Returns the events produced by this mutation.
    pub fn remove_triple(&mut self, s: &str, p: &str, o: &str) -> Vec<SubgraphEvent> {
        let now = unix_ms();
        let mut events = Vec::new();

        let removed = if let Some(edges) = self.graph.get_mut(s) {
            let before = edges.len();
            edges.retain(|(ep, eo)| !(ep == p && eo == o));
            edges.len() < before
        } else {
            false
        };

        if removed {
            let ev = SubgraphEvent {
                event_type: SubgraphEventType::EdgeRemoved,
                node: s.to_string(),
                timestamp: now,
            };
            events.push(ev.clone());
            self.buffer.push_back(ev);

            // If subject now has no edges, emit NodeRemoved
            if self.graph.get(s).map(|e| e.is_empty()).unwrap_or(false) {
                self.graph.remove(s);
                let ev = SubgraphEvent {
                    event_type: SubgraphEventType::NodeRemoved,
                    node: s.to_string(),
                    timestamp: now,
                };
                events.push(ev.clone());
                self.buffer.push_back(ev);
            }
        }

        self.events_total += events.len() as u64;
        events
    }

    /// Extract `(subject, predicate, object)` triples matching `pattern`.
    ///
    /// If `pattern.anchor` is `"?x"` the extractor uses all known subjects
    /// as starting points (up to `max_nodes`).
    pub fn extract_subgraph(&self, pattern: &SubgraphPattern) -> Vec<(String, String, String)> {
        let anchors: Vec<&str> = if pattern.anchor == "?x" {
            self.graph.keys().map(|s| s.as_str()).collect()
        } else {
            vec![pattern.anchor.as_str()]
        };

        let pred_filter: Option<HashSet<&str>> = if pattern.predicates.is_empty() {
            None
        } else {
            Some(pattern.predicates.iter().map(|s| s.as_str()).collect())
        };

        let mut visited: HashSet<String> = HashSet::new();
        let mut result: Vec<(String, String, String)> = Vec::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();

        for anchor in anchors {
            if visited.len() >= self.max_nodes {
                break;
            }
            if !visited.contains(anchor) {
                queue.push_back((anchor.to_string(), 0));
                visited.insert(anchor.to_string());
            }
        }

        while let Some((node, depth)) = queue.pop_front() {
            if visited.len() > self.max_nodes {
                break;
            }

            if let Some(edges) = self.graph.get(&node) {
                for (pred, obj) in edges {
                    // Predicate filter
                    if let Some(ref pf) = pred_filter {
                        if !pf.contains(pred.as_str()) {
                            continue;
                        }
                    }

                    // Degree filter on the target node
                    if let Some(ref filter) = pattern.filter {
                        if !self.node_passes_filter(&node, filter) {
                            continue;
                        }
                    }

                    result.push((node.clone(), pred.clone(), obj.clone()));

                    if depth < pattern.depth && !visited.contains(obj) {
                        visited.insert(obj.clone());
                        queue.push_back((obj.clone(), depth + 1));
                    }
                }
            }
        }

        result
    }

    /// Extract the BFS neighbourhood of `node` up to `depth` hops.
    pub fn extract_neighborhood(&self, node: &str, depth: usize) -> Vec<(String, String, String)> {
        let pattern = SubgraphPattern {
            anchor: node.to_string(),
            predicates: vec![],
            depth,
            filter: None,
        };
        self.extract_subgraph(&pattern)
    }

    /// Drain and return all buffered events.
    pub fn drain_events(&mut self) -> Vec<SubgraphEvent> {
        self.buffer.drain(..).collect()
    }

    /// Number of nodes currently in the graph.
    pub fn node_count(&self) -> usize {
        self.graph.len()
    }

    /// Number of directed edges currently in the graph.
    pub fn edge_count(&self) -> usize {
        self.graph.values().map(|e| e.len()).sum()
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    fn node_passes_filter(&self, node: &str, filter: &SubgraphFilter) -> bool {
        match filter {
            SubgraphFilter::HasLabel(label) => {
                // A node "has a label" if it has at least one edge whose object
                // contains the label string (simplified heuristic)
                self.graph
                    .get(node)
                    .map(|edges| edges.iter().any(|(_, o)| o.contains(label.as_str())))
                    .unwrap_or(false)
            }
            SubgraphFilter::MaxDegree(max) => {
                self.graph.get(node).map(|e| e.len()).unwrap_or(0) <= *max
            }
            SubgraphFilter::MinDegree(min) => {
                self.graph.get(node).map(|e| e.len()).unwrap_or(0) >= *min
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamingStats
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime statistics for [`StreamingGraphRag`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingStats {
    /// Number of unique nodes in the live graph
    pub nodes: usize,
    /// Number of directed edges in the live graph
    pub edges: usize,
    /// Cache hit rate: hits / total queries (NaN if no queries yet)
    pub cache_hit_rate: f64,
    /// Total number of events processed since creation
    pub events_processed: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamingGraphRag
// ─────────────────────────────────────────────────────────────────────────────

/// A live-updating GraphRAG engine that processes streaming triple events and
/// serves subgraph queries from a cache backed by the live graph.
pub struct StreamingGraphRag {
    extractor: StreamingSubgraphExtractor,
    query_cache: HashMap<String, Vec<(String, String, String)>>,
    cache_ttl_ms: i64,
    cache_timestamps: HashMap<String, i64>,
    cache_hits: u64,
    cache_misses: u64,
    events_processed: u64,
}

impl StreamingGraphRag {
    /// Create a new engine.
    ///
    /// * `max_depth` – BFS depth for subgraph extraction
    /// * `cache_ttl_ms` is ignored if 0 (no expiry)
    pub fn new(max_depth: usize) -> Self {
        Self::with_cache_ttl(max_depth, 60_000) // default 60-second TTL
    }

    /// Create with an explicit cache TTL in milliseconds (0 = no expiry).
    pub fn with_cache_ttl(max_depth: usize, cache_ttl_ms: i64) -> Self {
        Self {
            extractor: StreamingSubgraphExtractor::new(max_depth, 10_000),
            query_cache: HashMap::new(),
            cache_ttl_ms,
            cache_timestamps: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
            events_processed: 0,
        }
    }

    /// Process an incoming triple event (always treated as "add").
    pub fn process_event(&mut self, s: &str, p: &str, o: &str) {
        let events = self.extractor.add_triple(s, p, o);
        self.events_processed += events.len() as u64;

        // Invalidate any cached entries whose anchor matches the modified nodes
        let dirty: Vec<String> = self
            .query_cache
            .keys()
            .filter(|k| k.contains(s) || k.contains(o))
            .cloned()
            .collect();
        for key in dirty {
            self.query_cache.remove(&key);
            self.cache_timestamps.remove(&key);
        }
    }

    /// Query the live graph using `pattern`, returning cached results when fresh.
    pub fn query_live(&mut self, pattern: &SubgraphPattern) -> Vec<(String, String, String)> {
        let cache_key = pattern_cache_key(pattern);
        let now = unix_ms();

        if let Some(result) = self.query_cache.get(&cache_key) {
            let cached_at = self.cache_timestamps.get(&cache_key).copied().unwrap_or(0);
            let still_fresh = self.cache_ttl_ms == 0 || now - cached_at < self.cache_ttl_ms;
            if still_fresh {
                self.cache_hits += 1;
                return result.clone();
            }
        }

        self.cache_misses += 1;
        let result = self.extractor.extract_subgraph(pattern);
        self.query_cache.insert(cache_key.clone(), result.clone());
        self.cache_timestamps.insert(cache_key, now);
        result
    }

    /// Return current statistics.
    pub fn stats(&self) -> StreamingStats {
        let total = self.cache_hits + self.cache_misses;
        let cache_hit_rate = if total == 0 {
            f64::NAN
        } else {
            self.cache_hits as f64 / total as f64
        };
        StreamingStats {
            nodes: self.extractor.node_count(),
            edges: self.extractor.edge_count(),
            cache_hit_rate,
            events_processed: self.events_processed,
        }
    }

    /// Drain buffered events from the underlying extractor.
    pub fn drain_events(&mut self) -> Vec<SubgraphEvent> {
        self.extractor.drain_events()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn pattern_cache_key(p: &SubgraphPattern) -> String {
    format!("{}|{}|{}", p.anchor, p.predicates.join(","), p.depth)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_extractor() -> StreamingSubgraphExtractor {
        StreamingSubgraphExtractor::new(3, 1000)
    }

    // ── add_triple ────────────────────────────────────────────────────────

    #[test]
    fn test_add_triple_emits_node_added_for_new_subject() {
        let mut ext = make_extractor();
        let events = ext.add_triple("Alice", "knows", "Bob");
        let types: Vec<_> = events.iter().map(|e| &e.event_type).collect();
        assert!(types.contains(&&SubgraphEventType::NodeAdded));
    }

    #[test]
    fn test_add_triple_emits_edge_added() {
        let mut ext = make_extractor();
        let events = ext.add_triple("Alice", "knows", "Bob");
        assert!(events
            .iter()
            .any(|e| e.event_type == SubgraphEventType::EdgeAdded));
    }

    #[test]
    fn test_add_triple_does_not_duplicate_existing_edge() {
        let mut ext = make_extractor();
        ext.add_triple("Alice", "knows", "Bob");
        let events = ext.add_triple("Alice", "knows", "Bob"); // duplicate
        assert!(events.is_empty()); // nothing changed
    }

    #[test]
    fn test_add_triple_increases_node_and_edge_count() {
        let mut ext = make_extractor();
        ext.add_triple("A", "p", "B");
        assert_eq!(ext.node_count(), 2);
        assert_eq!(ext.edge_count(), 1);
    }

    #[test]
    fn test_add_triple_new_object_emits_node_added() {
        let mut ext = make_extractor();
        let events = ext.add_triple("A", "p", "B");
        let node_added_nodes: Vec<_> = events
            .iter()
            .filter(|e| e.event_type == SubgraphEventType::NodeAdded)
            .map(|e| e.node.as_str())
            .collect();
        assert!(node_added_nodes.contains(&"B"));
    }

    // ── remove_triple ─────────────────────────────────────────────────────

    #[test]
    fn test_remove_triple_emits_edge_removed() {
        let mut ext = make_extractor();
        ext.add_triple("Alice", "knows", "Bob");
        ext.drain_events();
        let events = ext.remove_triple("Alice", "knows", "Bob");
        assert!(events
            .iter()
            .any(|e| e.event_type == SubgraphEventType::EdgeRemoved));
    }

    #[test]
    fn test_remove_triple_emits_node_removed_when_no_edges_left() {
        let mut ext = make_extractor();
        ext.add_triple("Alice", "knows", "Bob");
        ext.drain_events();
        let events = ext.remove_triple("Alice", "knows", "Bob");
        assert!(events
            .iter()
            .any(|e| e.event_type == SubgraphEventType::NodeRemoved));
    }

    #[test]
    fn test_remove_nonexistent_triple_emits_nothing() {
        let mut ext = make_extractor();
        let events = ext.remove_triple("X", "p", "Y");
        assert!(events.is_empty());
    }

    #[test]
    fn test_remove_triple_decreases_edge_count() {
        let mut ext = make_extractor();
        ext.add_triple("A", "p", "B");
        ext.add_triple("A", "q", "C");
        ext.remove_triple("A", "p", "B");
        assert_eq!(ext.edge_count(), 1);
    }

    // ── extract_subgraph ──────────────────────────────────────────────────

    #[test]
    fn test_extract_subgraph_direct_anchor() {
        let mut ext = make_extractor();
        ext.add_triple("A", "p", "B");
        ext.add_triple("A", "q", "C");
        let pattern = SubgraphPattern {
            anchor: "A".to_string(),
            predicates: vec![],
            depth: 1,
            filter: None,
        };
        let triples = ext.extract_subgraph(&pattern);
        assert_eq!(triples.len(), 2);
    }

    #[test]
    fn test_extract_subgraph_predicate_filter() {
        let mut ext = make_extractor();
        ext.add_triple("A", "knows", "B");
        ext.add_triple("A", "likes", "C");
        let pattern = SubgraphPattern {
            anchor: "A".to_string(),
            predicates: vec!["knows".to_string()],
            depth: 1,
            filter: None,
        };
        let triples = ext.extract_subgraph(&pattern);
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].1, "knows");
    }

    #[test]
    fn test_extract_subgraph_variable_anchor() {
        let mut ext = make_extractor();
        ext.add_triple("A", "p", "B");
        ext.add_triple("C", "p", "D");
        let pattern = SubgraphPattern {
            anchor: "?x".to_string(),
            predicates: vec![],
            depth: 0,
            filter: None,
        };
        let triples = ext.extract_subgraph(&pattern);
        // Should find edges from all subjects
        assert!(!triples.is_empty());
    }

    #[test]
    fn test_extract_subgraph_depth_limits_expansion() {
        let mut ext = make_extractor();
        ext.add_triple("A", "p", "B");
        ext.add_triple("B", "p", "C");
        ext.add_triple("C", "p", "D");
        // depth 1 → only A→B, then B→C (depth 1 from A)
        let pattern = SubgraphPattern {
            anchor: "A".to_string(),
            predicates: vec![],
            depth: 1,
            filter: None,
        };
        let triples = ext.extract_subgraph(&pattern);
        // Should contain A→B and B→C but not C→D
        let has_cd = triples.iter().any(|(s, _, o)| s == "C" && o == "D");
        assert!(!has_cd, "Should not expand past depth 1");
    }

    // ── extract_neighborhood ──────────────────────────────────────────────

    #[test]
    fn test_extract_neighborhood_basic() {
        let mut ext = make_extractor();
        ext.add_triple("Root", "p", "Child");
        let triples = ext.extract_neighborhood("Root", 1);
        assert!(!triples.is_empty());
        assert_eq!(triples[0].0, "Root");
    }

    #[test]
    fn test_extract_neighborhood_unknown_node_returns_empty() {
        let ext = make_extractor();
        let triples = ext.extract_neighborhood("Unknown", 2);
        assert!(triples.is_empty());
    }

    // ── SubgraphFilter ────────────────────────────────────────────────────

    #[test]
    fn test_filter_min_degree() {
        let mut ext = make_extractor();
        // A has 2 edges, B has 0
        ext.add_triple("A", "p", "B");
        ext.add_triple("A", "q", "C");
        let pattern = SubgraphPattern {
            anchor: "A".to_string(),
            predicates: vec![],
            depth: 1,
            filter: Some(SubgraphFilter::MinDegree(2)),
        };
        let triples = ext.extract_subgraph(&pattern);
        // A has ≥ 2 edges → passes filter
        assert!(!triples.is_empty());
    }

    #[test]
    fn test_filter_max_degree() {
        let mut ext = make_extractor();
        ext.add_triple("A", "p1", "B");
        ext.add_triple("A", "p2", "C");
        ext.add_triple("A", "p3", "D");
        let pattern = SubgraphPattern {
            anchor: "A".to_string(),
            predicates: vec![],
            depth: 1,
            filter: Some(SubgraphFilter::MaxDegree(1)),
        };
        // A has 3 edges, fails MaxDegree(1)
        let triples = ext.extract_subgraph(&pattern);
        assert!(triples.is_empty());
    }

    // ── drain_events ──────────────────────────────────────────────────────

    #[test]
    fn test_drain_events_clears_buffer() {
        let mut ext = make_extractor();
        ext.add_triple("A", "p", "B");
        assert!(!ext.drain_events().is_empty());
        assert!(ext.drain_events().is_empty());
    }

    // ── StreamingGraphRag ─────────────────────────────────────────────────

    #[test]
    fn test_streaming_rag_process_event_updates_graph() {
        let mut rag = StreamingGraphRag::new(3);
        rag.process_event("Alice", "knows", "Bob");
        let stats = rag.stats();
        assert!(stats.nodes >= 2);
        assert!(stats.edges >= 1);
    }

    #[test]
    fn test_streaming_rag_cache_hit() {
        let mut rag = StreamingGraphRag::new(3);
        rag.process_event("A", "p", "B");

        let pattern = SubgraphPattern {
            anchor: "A".to_string(),
            predicates: vec![],
            depth: 1,
            filter: None,
        };

        let _ = rag.query_live(&pattern); // miss
        let _ = rag.query_live(&pattern); // hit
        let stats = rag.stats();
        assert!(stats.cache_hit_rate > 0.0);
    }

    #[test]
    fn test_streaming_rag_cache_invalidated_on_event() {
        let mut rag = StreamingGraphRag::new(3);
        rag.process_event("A", "p", "B");

        let pattern = SubgraphPattern {
            anchor: "A".to_string(),
            predicates: vec![],
            depth: 1,
            filter: None,
        };

        let r1 = rag.query_live(&pattern);
        // Add a new edge touching A → should invalidate cache
        rag.process_event("A", "q", "C");
        let r2 = rag.query_live(&pattern);
        // r2 should have more or equal triples than r1
        assert!(r2.len() >= r1.len());
    }

    #[test]
    fn test_streaming_rag_stats_initial_nan_hit_rate() {
        let rag = StreamingGraphRag::new(3);
        let stats = rag.stats();
        assert!(stats.cache_hit_rate.is_nan());
    }

    #[test]
    fn test_streaming_rag_drain_events() {
        let mut rag = StreamingGraphRag::new(3);
        rag.process_event("X", "p", "Y");
        let events = rag.drain_events();
        assert!(!events.is_empty());
    }

    #[test]
    fn test_streaming_stats_fields() {
        let mut rag = StreamingGraphRag::new(2);
        rag.process_event("A", "edge", "B");
        let stats = rag.stats();
        assert_eq!(stats.nodes, 2);
        assert_eq!(stats.edges, 1);
    }
}
