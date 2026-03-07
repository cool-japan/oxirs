//! Time-aware knowledge graph with BFS path finding and entity history tracking.
//!
//! # Overview
//!
//! [`TemporalKnowledgeGraph`] stores RDF-like triples annotated with validity
//! intervals (`valid_from` / `valid_to` as Unix-ms timestamps) and a confidence
//! score.  It exposes point-in-time and range queries, entity set queries, and a
//! BFS-based temporal path finder.
//!
//! [`TemporalGraphRag`] wraps the knowledge graph and adds a simple keyword-based
//! retrieval layer that returns the most relevant temporal triples for a natural-
//! language question at a given point in time.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// Core types
// ─────────────────────────────────────────────────────────────────────────────

/// A triple annotated with a validity interval and confidence score.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TemporalTriple {
    /// Subject URI or blank node identifier
    pub subject: String,
    /// Predicate URI
    pub predicate: String,
    /// Object URI or literal
    pub object: String,
    /// Inclusive start of the validity interval (Unix-ms)
    pub valid_from: i64,
    /// Exclusive end of the validity interval (`None` means "still valid")
    pub valid_to: Option<i64>,
    /// Confidence in [0.0, 1.0]
    pub confidence: f64,
}

impl TemporalTriple {
    /// Returns `true` if this triple is valid at `timestamp` (Unix-ms).
    pub fn is_valid_at(&self, timestamp: i64) -> bool {
        if timestamp < self.valid_from {
            return false;
        }
        match self.valid_to {
            Some(to) => timestamp < to,
            None => true,
        }
    }

    /// Returns `true` if the triple's validity interval overlaps with `[from, to)`.
    pub fn overlaps_range(&self, from: i64, to: i64) -> bool {
        // Triple ends before range starts
        if let Some(t) = self.valid_to {
            if t <= from {
                return false;
            }
        }
        // Triple starts at or after range ends
        self.valid_from < to
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TemporalKnowledgeGraph
// ─────────────────────────────────────────────────────────────────────────────

/// An in-memory knowledge graph that stores triples with temporal validity.
///
/// Internally, triples are stored in a `Vec` and a secondary `BTreeMap` index
/// maps `valid_from` timestamps to the indices of triples that start at that
/// timestamp.  This allows efficient range scans over the time axis.
pub struct TemporalKnowledgeGraph {
    triples: Vec<TemporalTriple>,
    /// timestamp → indices of triples whose `valid_from` equals the key
    time_index: BTreeMap<i64, Vec<usize>>,
}

impl Default for TemporalKnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalKnowledgeGraph {
    /// Create an empty knowledge graph.
    pub fn new() -> Self {
        Self {
            triples: Vec::new(),
            time_index: BTreeMap::new(),
        }
    }

    /// Insert a temporal triple and update the time index.
    pub fn insert(&mut self, triple: TemporalTriple) {
        let idx = self.triples.len();
        let ts = triple.valid_from;
        self.triples.push(triple);
        self.time_index.entry(ts).or_default().push(idx);
    }

    /// Return all triples that are valid at `timestamp`.
    pub fn query_at(&self, timestamp: i64) -> Vec<&TemporalTriple> {
        self.triples
            .iter()
            .filter(|t| t.is_valid_at(timestamp))
            .collect()
    }

    /// Return all triples whose validity interval overlaps `[from, to)`.
    pub fn query_range(&self, from: i64, to: i64) -> Vec<&TemporalTriple> {
        self.triples
            .iter()
            .filter(|t| t.overlaps_range(from, to))
            .collect()
    }

    /// Return the set of entity identifiers (subjects and objects) that appear
    /// in at least one valid triple at `timestamp`.
    pub fn entities_at(&self, timestamp: i64) -> HashSet<String> {
        let mut set = HashSet::new();
        for t in self.query_at(timestamp) {
            set.insert(t.subject.clone());
            set.insert(t.object.clone());
        }
        set
    }

    /// Find a path from `from` to `to` using BFS over triples that are valid at
    /// `at`.  Returns the sequence of entity identifiers along the path
    /// (inclusive of both endpoints), or `None` if no path exists.
    pub fn temporal_path(&self, from: &str, to: &str, at: i64) -> Option<Vec<String>> {
        if from == to {
            return Some(vec![from.to_string()]);
        }

        // Build adjacency from triples valid at `at`
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for t in self.query_at(at) {
            adj.entry(t.subject.as_str())
                .or_default()
                .push(t.object.as_str());
        }

        // BFS
        let mut visited: HashSet<&str> = HashSet::new();
        let mut queue: VecDeque<(&str, Vec<&str>)> = VecDeque::new();
        queue.push_back((from, vec![from]));
        visited.insert(from);

        while let Some((node, path)) = queue.pop_front() {
            if let Some(neighbors) = adj.get(node) {
                for &neighbor in neighbors {
                    if neighbor == to {
                        let mut result: Vec<String> = path.iter().map(|s| s.to_string()).collect();
                        result.push(to.to_string());
                        return Some(result);
                    }
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor);
                        let mut new_path = path.clone();
                        new_path.push(neighbor);
                        queue.push_back((neighbor, new_path));
                    }
                }
            }
        }

        None
    }

    /// Total number of triples (including those no longer valid).
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Returns `true` if the knowledge graph contains no triples.
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EntityHistory
// ─────────────────────────────────────────────────────────────────────────────

/// Summary of an entity's appearances across time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityHistory {
    /// The entity identifier
    pub entity: String,
    /// All triples in which this entity appears (as subject or object)
    pub events: Vec<TemporalTriple>,
    /// Earliest `valid_from` among all events
    pub first_seen: Option<i64>,
    /// Latest `valid_from` among all events
    pub last_seen: Option<i64>,
    /// Total number of distinct predicates observed for this entity
    pub relationship_count: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// TemporalGraphRag
// ─────────────────────────────────────────────────────────────────────────────

/// A lightweight retrieval engine that wraps [`TemporalKnowledgeGraph`] and
/// scores triples by keyword overlap with a natural-language question.
pub struct TemporalGraphRag {
    kg: TemporalKnowledgeGraph,
    /// Optional pre-computed embedding cache (placeholder for future GPU-backed
    /// embedding; stored as flat `Vec<f32>` keyed by entity URI).
    embedding_cache: HashMap<String, Vec<f32>>,
}

impl Default for TemporalGraphRag {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalGraphRag {
    /// Create an empty engine.
    pub fn new() -> Self {
        Self {
            kg: TemporalKnowledgeGraph::new(),
            embedding_cache: HashMap::new(),
        }
    }

    /// Ingest a new event into the underlying knowledge graph.
    ///
    /// `timestamp` is the Unix-ms start time; the triple is treated as
    /// "still valid" (no `valid_to`).
    pub fn ingest_event(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
        timestamp: i64,
        confidence: f64,
    ) {
        self.kg.insert(TemporalTriple {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            valid_from: timestamp,
            valid_to: None,
            confidence: confidence.clamp(0.0, 1.0),
        });
    }

    /// Retrieve up to `top_k` triples valid at `timestamp` that are most
    /// relevant to `question`, ranked by keyword overlap score.
    pub fn query(&self, question: &str, timestamp: i64, top_k: usize) -> Vec<TemporalTriple> {
        let keywords: Vec<String> = question
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();

        let candidates = self.kg.query_at(timestamp);

        let mut scored: Vec<(f64, &TemporalTriple)> = candidates
            .into_iter()
            .map(|t| {
                let score = self.keyword_score(t, &keywords);
                (score, t)
            })
            .collect();

        // Sort descending by (score, confidence)
        scored.sort_by(|(sa, ta), (sb, tb)| {
            sa.partial_cmp(sb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .reverse()
                .then_with(|| {
                    ta.confidence
                        .partial_cmp(&tb.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .reverse()
                })
        });

        scored
            .into_iter()
            .take(top_k)
            .map(|(_, t)| t.clone())
            .collect()
    }

    /// Compute a keyword overlap score for a triple against a set of query keywords.
    fn keyword_score(&self, triple: &TemporalTriple, keywords: &[String]) -> f64 {
        if keywords.is_empty() {
            return 0.0;
        }

        let text = format!(
            "{} {} {}",
            triple.subject.to_lowercase(),
            triple.predicate.to_lowercase(),
            triple.object.to_lowercase()
        );

        let matched = keywords
            .iter()
            .filter(|kw| text.contains(kw.as_str()))
            .count();

        let raw_score = matched as f64 / keywords.len() as f64;
        // Blend with confidence
        raw_score * 0.7 + triple.confidence * 0.3
    }

    /// Summarise all historical appearances of `entity` in the knowledge graph.
    pub fn summarize_entity_history(&self, entity: &str) -> EntityHistory {
        let events: Vec<TemporalTriple> = self
            .kg
            .triples
            .iter()
            .filter(|t| t.subject == entity || t.object == entity)
            .cloned()
            .collect();

        let first_seen = events.iter().map(|t| t.valid_from).min();
        let last_seen = events.iter().map(|t| t.valid_from).max();

        let relationship_count: HashSet<String> =
            events.iter().map(|t| t.predicate.clone()).collect();
        let relationship_count = relationship_count.len();

        EntityHistory {
            entity: entity.to_string(),
            events,
            first_seen,
            last_seen,
            relationship_count,
        }
    }

    /// Store an embedding vector for `entity` in the cache.
    pub fn cache_embedding(&mut self, entity: &str, embedding: Vec<f32>) {
        self.embedding_cache.insert(entity.to_string(), embedding);
    }

    /// Retrieve a cached embedding, if present.
    pub fn get_embedding(&self, entity: &str) -> Option<&Vec<f32>> {
        self.embedding_cache.get(entity)
    }

    /// Number of events ingested.
    pub fn event_count(&self) -> usize {
        self.kg.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_triple(s: &str, p: &str, o: &str, from: i64, to: Option<i64>) -> TemporalTriple {
        TemporalTriple {
            subject: s.to_string(),
            predicate: p.to_string(),
            object: o.to_string(),
            valid_from: from,
            valid_to: to,
            confidence: 0.9,
        }
    }

    // ── TemporalTriple::is_valid_at ───────────────────────────────────────

    #[test]
    fn test_is_valid_at_within_interval() {
        let t = make_triple("s", "p", "o", 100, Some(200));
        assert!(t.is_valid_at(100));
        assert!(t.is_valid_at(150));
        assert!(!t.is_valid_at(99));
        assert!(!t.is_valid_at(200)); // exclusive end
    }

    #[test]
    fn test_is_valid_at_no_end() {
        let t = make_triple("s", "p", "o", 100, None);
        assert!(t.is_valid_at(100));
        assert!(t.is_valid_at(i64::MAX));
        assert!(!t.is_valid_at(99));
    }

    // ── TemporalTriple::overlaps_range ────────────────────────────────────

    #[test]
    fn test_overlaps_range_full_overlap() {
        let t = make_triple("s", "p", "o", 50, Some(150));
        assert!(t.overlaps_range(100, 200)); // starts before range end, ends after range start
    }

    #[test]
    fn test_overlaps_range_no_overlap_before() {
        let t = make_triple("s", "p", "o", 0, Some(50));
        assert!(!t.overlaps_range(100, 200));
    }

    #[test]
    fn test_overlaps_range_no_overlap_after() {
        let t = make_triple("s", "p", "o", 300, None);
        assert!(!t.overlaps_range(100, 200));
    }

    #[test]
    fn test_overlaps_range_open_end() {
        let t = make_triple("s", "p", "o", 50, None);
        assert!(t.overlaps_range(100, 200)); // always overlaps when no end
    }

    // ── TemporalKnowledgeGraph::insert and len ────────────────────────────

    #[test]
    fn test_insert_increases_len() {
        let mut kg = TemporalKnowledgeGraph::new();
        assert_eq!(kg.len(), 0);
        assert!(kg.is_empty());
        kg.insert(make_triple("s", "p", "o", 0, None));
        assert_eq!(kg.len(), 1);
        assert!(!kg.is_empty());
    }

    #[test]
    fn test_insert_updates_time_index() {
        let mut kg = TemporalKnowledgeGraph::new();
        kg.insert(make_triple("s", "p", "o", 1000, None));
        assert!(kg.time_index.contains_key(&1000));
    }

    // ── TemporalKnowledgeGraph::query_at ─────────────────────────────────

    #[test]
    fn test_query_at_returns_valid_triples() {
        let mut kg = TemporalKnowledgeGraph::new();
        kg.insert(make_triple("a", "p", "b", 0, Some(100)));
        kg.insert(make_triple("c", "p", "d", 50, None));
        kg.insert(make_triple("e", "p", "f", 200, None));

        let valid = kg.query_at(75);
        assert_eq!(valid.len(), 2);
    }

    #[test]
    fn test_query_at_empty_graph() {
        let kg = TemporalKnowledgeGraph::new();
        assert!(kg.query_at(0).is_empty());
    }

    #[test]
    fn test_query_at_excludes_expired() {
        let mut kg = TemporalKnowledgeGraph::new();
        kg.insert(make_triple("a", "p", "b", 0, Some(50)));
        let valid = kg.query_at(100);
        assert!(valid.is_empty());
    }

    // ── TemporalKnowledgeGraph::query_range ───────────────────────────────

    #[test]
    fn test_query_range_returns_overlapping() {
        let mut kg = TemporalKnowledgeGraph::new();
        kg.insert(make_triple("a", "p", "b", 0, Some(100))); // overlaps [50,150)
        kg.insert(make_triple("c", "p", "d", 80, Some(200))); // overlaps [50,150)
        kg.insert(make_triple("e", "p", "f", 200, None)); // does NOT overlap [50,150)

        let result = kg.query_range(50, 150);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_query_range_no_overlap() {
        let mut kg = TemporalKnowledgeGraph::new();
        kg.insert(make_triple("a", "p", "b", 0, Some(10)));
        let result = kg.query_range(100, 200);
        assert!(result.is_empty());
    }

    // ── TemporalKnowledgeGraph::entities_at ───────────────────────────────

    #[test]
    fn test_entities_at_collects_subjects_and_objects() {
        let mut kg = TemporalKnowledgeGraph::new();
        kg.insert(make_triple("Alice", "knows", "Bob", 0, None));
        kg.insert(make_triple("Bob", "likes", "Carol", 0, None));

        let entities = kg.entities_at(0);
        assert!(entities.contains("Alice"));
        assert!(entities.contains("Bob"));
        assert!(entities.contains("Carol"));
    }

    #[test]
    fn test_entities_at_respects_timestamp() {
        let mut kg = TemporalKnowledgeGraph::new();
        kg.insert(make_triple("Alice", "knows", "Bob", 0, Some(50)));
        kg.insert(make_triple("Carol", "knows", "Dave", 100, None));

        let entities = kg.entities_at(75);
        // "Alice"/"Bob" expired, "Carol"/"Dave" not yet started
        assert!(entities.is_empty());
    }

    // ── TemporalKnowledgeGraph::temporal_path ─────────────────────────────

    #[test]
    fn test_temporal_path_direct_edge() {
        let mut kg = TemporalKnowledgeGraph::new();
        kg.insert(make_triple("A", "p", "B", 0, None));

        let path = kg.temporal_path("A", "B", 0).unwrap();
        assert_eq!(path, vec!["A", "B"]);
    }

    #[test]
    fn test_temporal_path_multi_hop() {
        let mut kg = TemporalKnowledgeGraph::new();
        kg.insert(make_triple("A", "p", "B", 0, None));
        kg.insert(make_triple("B", "p", "C", 0, None));
        kg.insert(make_triple("C", "p", "D", 0, None));

        let path = kg.temporal_path("A", "D", 0).unwrap();
        assert_eq!(path.first().map(|s| s.as_str()), Some("A"));
        assert_eq!(path.last().map(|s| s.as_str()), Some("D"));
        assert!(path.len() >= 2);
    }

    #[test]
    fn test_temporal_path_no_path() {
        let mut kg = TemporalKnowledgeGraph::new();
        kg.insert(make_triple("A", "p", "B", 0, None));
        // C is disconnected
        assert!(kg.temporal_path("A", "C", 0).is_none());
    }

    #[test]
    fn test_temporal_path_same_node() {
        let kg = TemporalKnowledgeGraph::new();
        let path = kg.temporal_path("A", "A", 0).unwrap();
        assert_eq!(path, vec!["A"]);
    }

    #[test]
    fn test_temporal_path_ignores_future_triples() {
        let mut kg = TemporalKnowledgeGraph::new();
        kg.insert(make_triple("A", "p", "B", 1000, None)); // not valid at t=0
        assert!(kg.temporal_path("A", "B", 0).is_none());
    }

    // ── TemporalGraphRag::ingest_event ────────────────────────────────────

    #[test]
    fn test_ingest_event_stores_triple() {
        let mut rag = TemporalGraphRag::new();
        rag.ingest_event("Alice", "knows", "Bob", 1000, 0.9);
        assert_eq!(rag.event_count(), 1);
    }

    #[test]
    fn test_ingest_event_clamps_confidence() {
        let mut rag = TemporalGraphRag::new();
        rag.ingest_event("A", "p", "B", 0, 1.5); // > 1.0 → clamp to 1.0
        rag.ingest_event("C", "p", "D", 0, -0.5); // < 0.0 → clamp to 0.0
        let triples = rag.kg.query_at(0);
        for t in triples {
            assert!(t.confidence >= 0.0 && t.confidence <= 1.0);
        }
    }

    // ── TemporalGraphRag::query ───────────────────────────────────────────

    #[test]
    fn test_query_returns_relevant_triple() {
        let mut rag = TemporalGraphRag::new();
        rag.ingest_event("Apple", "releases", "iPhone", 1000, 0.9);
        rag.ingest_event("Google", "releases", "Pixel", 1000, 0.8);

        let results = rag.query("Apple iPhone", 1000, 5);
        assert!(!results.is_empty());
        // The Apple/iPhone triple should rank first
        assert_eq!(results[0].subject, "Apple");
    }

    #[test]
    fn test_query_respects_top_k() {
        let mut rag = TemporalGraphRag::new();
        for i in 0..10 {
            rag.ingest_event(&format!("S{i}"), "p", &format!("O{i}"), 0, 0.9);
        }
        let results = rag.query("any", 0, 3);
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_query_respects_timestamp() {
        let mut rag = TemporalGraphRag::new();
        rag.ingest_event("Past", "event", "X", 0, 0.9);
        // Query at timestamp before the event
        let results = rag.query("Past event X", -1, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_empty_graph_returns_empty() {
        let rag = TemporalGraphRag::new();
        let results = rag.query("anything", 0, 5);
        assert!(results.is_empty());
    }

    // ── TemporalGraphRag::summarize_entity_history ────────────────────────

    #[test]
    fn test_summarize_entity_history_basic() {
        let mut rag = TemporalGraphRag::new();
        rag.ingest_event("Alice", "knows", "Bob", 100, 0.9);
        rag.ingest_event("Alice", "likes", "Carol", 200, 0.8);
        rag.ingest_event("Dave", "knows", "Alice", 300, 0.7);

        let history = rag.summarize_entity_history("Alice");
        assert_eq!(history.entity, "Alice");
        assert_eq!(history.events.len(), 3);
        assert_eq!(history.first_seen, Some(100));
        assert_eq!(history.last_seen, Some(300));
        assert_eq!(history.relationship_count, 2); // "knows" and "likes"
    }

    #[test]
    fn test_summarize_entity_history_unknown_entity() {
        let rag = TemporalGraphRag::new();
        let history = rag.summarize_entity_history("Unknown");
        assert!(history.events.is_empty());
        assert!(history.first_seen.is_none());
        assert!(history.last_seen.is_none());
        assert_eq!(history.relationship_count, 0);
    }

    // ── TemporalGraphRag::embedding cache ─────────────────────────────────

    #[test]
    fn test_embedding_cache_roundtrip() {
        let mut rag = TemporalGraphRag::new();
        let embedding = vec![0.1_f32, 0.2, 0.3];
        rag.cache_embedding("Alice", embedding.clone());
        assert_eq!(rag.get_embedding("Alice"), Some(&embedding));
        assert!(rag.get_embedding("Bob").is_none());
    }
}
