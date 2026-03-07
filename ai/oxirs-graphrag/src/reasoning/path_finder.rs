//! Multi-hop path finder with configurable path scoring strategies
//!
//! Provides `MultiHopPathFinder` for finding paths between entities in a
//! knowledge graph, supporting Uniform, AttentionWeighted, and PathLength scoring.

use crate::Triple;
use std::collections::{HashMap, HashSet, VecDeque};

// ── PathScoring ────────────────────────────────────────────────────────────────

/// Strategy for scoring multi-hop paths
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PathScoring {
    /// All paths receive the same score of 1.0
    Uniform,
    /// Score decreases with path length: 1.0 / (1 + hop_count)
    PathLength,
    /// Simulated attention-weighted: assigns higher weight to earlier hops
    /// score = sum over i of 1/(i+1), normalised by hop_count
    #[default]
    AttentionWeighted,
}

// ── MultiHopReasoningConfig ────────────────────────────────────────────────────

/// Configuration for multi-hop path finding
#[derive(Debug, Clone)]
pub struct MultiHopReasoningConfig {
    /// Maximum number of hops to traverse per path
    pub max_hops: u8,
    /// Minimum confidence threshold for a path to be returned
    pub min_confidence: f64,
    /// Path scoring strategy
    pub path_scoring: PathScoring,
    /// Maximum number of paths to return per (start, end) pair
    pub max_paths_per_pair: usize,
    /// Maximum BFS frontier size to prevent explosion on dense graphs
    pub max_frontier: usize,
}

impl Default for MultiHopReasoningConfig {
    fn default() -> Self {
        Self {
            max_hops: 3,
            min_confidence: 0.0,
            path_scoring: PathScoring::default(),
            max_paths_per_pair: 20,
            max_frontier: 10_000,
        }
    }
}

// ── HopPath ────────────────────────────────────────────────────────────────────

/// A single path through the knowledge graph
#[derive(Debug, Clone)]
pub struct HopPath {
    /// Ordered entity URIs from start to end
    pub entities: Vec<String>,
    /// Relation predicates connecting consecutive entities
    pub relations: Vec<String>,
    /// Path score computed by the chosen `PathScoring` strategy
    pub score: f64,
}

impl HopPath {
    /// Number of hops (edges) in this path
    pub fn hop_count(&self) -> usize {
        self.relations.len()
    }
}

// ── KnowledgeGraph (adjacency helper) ──────────────────────────────────────────

/// Lightweight adjacency representation built from RDF triples.
pub struct KnowledgeGraph {
    /// Forward adjacency: subject -> Vec<(predicate, object)>
    adj: HashMap<String, Vec<(String, String)>>,
}

impl KnowledgeGraph {
    /// Build from a slice of RDF triples.
    pub fn from_triples(triples: &[Triple]) -> Self {
        let mut adj: HashMap<String, Vec<(String, String)>> = HashMap::new();
        for t in triples {
            adj.entry(t.subject.clone())
                .or_default()
                .push((t.predicate.clone(), t.object.clone()));
        }
        Self { adj }
    }

    /// Iterate over the (predicate, object) neighbours of a node.
    pub fn neighbours(&self, node: &str) -> &[(String, String)] {
        self.adj.get(node).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Number of unique subject nodes
    pub fn node_count(&self) -> usize {
        self.adj.len()
    }
}

// ── MultiHopPathFinder ─────────────────────────────────────────────────────────

/// Finds multi-hop paths between entities using BFS.
pub struct MultiHopPathFinder {
    config: MultiHopReasoningConfig,
}

impl MultiHopPathFinder {
    /// Create with the given config.
    pub fn new(config: MultiHopReasoningConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(MultiHopReasoningConfig::default())
    }

    /// Find all paths from `start` to `end` in the given graph, up to `max_hops`.
    ///
    /// Returns paths sorted descending by score, limited to `max_paths_per_pair`.
    pub fn find_paths(
        &self,
        start: &str,
        end: &str,
        max_hops: u8,
        graph: &KnowledgeGraph,
    ) -> Vec<HopPath> {
        // BFS: queue holds (current_node, entity_path, relation_path, visited)
        struct State {
            node: String,
            entities: Vec<String>,
            relations: Vec<String>,
            visited: HashSet<String>,
        }

        let mut queue: VecDeque<State> = VecDeque::new();
        queue.push_back(State {
            node: start.to_string(),
            entities: vec![start.to_string()],
            relations: vec![],
            visited: {
                let mut h = HashSet::new();
                h.insert(start.to_string());
                h
            },
        });

        let mut paths: Vec<HopPath> = Vec::new();
        let mut frontier_visited = 0usize;

        while let Some(state) = queue.pop_front() {
            if frontier_visited >= self.config.max_frontier {
                break;
            }
            frontier_visited += 1;

            let hops_so_far = state.relations.len() as u8;

            if hops_so_far >= max_hops {
                continue;
            }

            for (pred, obj) in graph.neighbours(&state.node) {
                if state.visited.contains(obj) {
                    continue;
                }
                let mut new_entities = state.entities.clone();
                new_entities.push(obj.clone());
                let mut new_relations = state.relations.clone();
                new_relations.push(pred.clone());

                if obj == end {
                    let score = self.score_path(&new_relations, &self.config.path_scoring);
                    if score >= self.config.min_confidence {
                        paths.push(HopPath {
                            entities: new_entities,
                            relations: new_relations,
                            score,
                        });
                        if paths.len() >= self.config.max_paths_per_pair {
                            paths.sort_by(|a, b| {
                                b.score
                                    .partial_cmp(&a.score)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            });
                            return paths;
                        }
                    }
                } else {
                    let mut new_visited = state.visited.clone();
                    new_visited.insert(obj.clone());
                    queue.push_back(State {
                        node: obj.clone(),
                        entities: new_entities,
                        relations: new_relations,
                        visited: new_visited,
                    });
                }
            }
        }

        paths.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        paths
    }

    /// Score a path given a list of relations and the chosen strategy.
    pub fn score_path(&self, relations: &[String], scoring: &PathScoring) -> f64 {
        let hops = relations.len();
        match scoring {
            PathScoring::Uniform => 1.0,
            PathScoring::PathLength => 1.0 / (1.0 + hops as f64),
            PathScoring::AttentionWeighted => {
                if hops == 0 {
                    return 0.0;
                }
                // sum(1/(i+1)) for i in 0..hops, divided by hops for normalisation
                let sum: f64 = (0..hops).map(|i| 1.0 / (i as f64 + 1.0)).sum();
                sum / hops as f64
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_graph() -> (Vec<Triple>, KnowledgeGraph) {
        let triples = vec![
            Triple::new("http://a", "http://rel/r1", "http://b"),
            Triple::new("http://b", "http://rel/r2", "http://c"),
            Triple::new("http://c", "http://rel/r3", "http://d"),
            Triple::new("http://a", "http://rel/direct", "http://c"),
        ];
        let graph = KnowledgeGraph::from_triples(&triples);
        (triples, graph)
    }

    // ── PathScoring ──────────────────────────────────────────────────────

    #[test]
    fn test_path_scoring_default_is_attention_weighted() {
        assert_eq!(PathScoring::default(), PathScoring::AttentionWeighted);
    }

    #[test]
    fn test_score_uniform_always_one() {
        let finder = MultiHopPathFinder::with_defaults();
        let rels: Vec<String> = vec!["r1".to_string(), "r2".to_string(), "r3".to_string()];
        let s = finder.score_path(&rels, &PathScoring::Uniform);
        assert!((s - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_score_path_length_decreases_with_hops() {
        let finder = MultiHopPathFinder::with_defaults();
        let s1 = finder.score_path(&["r".to_string()], &PathScoring::PathLength);
        let s2 = finder.score_path(
            &["r".to_string(), "r2".to_string()],
            &PathScoring::PathLength,
        );
        assert!(s1 > s2, "Longer path should score lower: {s1} vs {s2}");
    }

    #[test]
    fn test_score_attention_weighted_single_hop() {
        let finder = MultiHopPathFinder::with_defaults();
        let s = finder.score_path(&["r".to_string()], &PathScoring::AttentionWeighted);
        // 1 hop: sum=1.0, div by 1 = 1.0
        assert!((s - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_score_attention_weighted_two_hops() {
        let finder = MultiHopPathFinder::with_defaults();
        let rels: Vec<String> = vec!["r1".to_string(), "r2".to_string()];
        let s = finder.score_path(&rels, &PathScoring::AttentionWeighted);
        // sum = 1 + 0.5 = 1.5, div by 2 = 0.75
        assert!((s - 0.75).abs() < 1e-9, "Expected 0.75, got {s}");
    }

    #[test]
    fn test_score_attention_weighted_empty_is_zero() {
        let finder = MultiHopPathFinder::with_defaults();
        let s = finder.score_path(&[], &PathScoring::AttentionWeighted);
        assert!((s - 0.0).abs() < f64::EPSILON);
    }

    // ── MultiHopReasoningConfig ──────────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let cfg = MultiHopReasoningConfig::default();
        assert_eq!(cfg.max_hops, 3);
        assert!((cfg.min_confidence - 0.0).abs() < f64::EPSILON);
        assert_eq!(cfg.path_scoring, PathScoring::AttentionWeighted);
        assert_eq!(cfg.max_paths_per_pair, 20);
    }

    // ── KnowledgeGraph ───────────────────────────────────────────────────

    #[test]
    fn test_knowledge_graph_node_count() {
        let (_, graph) = simple_graph();
        assert_eq!(graph.node_count(), 3); // a, b, c (d and a-direct-c are objects only for unique subjects)
    }

    #[test]
    fn test_knowledge_graph_neighbours() {
        let triples = vec![Triple::new("http://x", "http://p", "http://y")];
        let graph = KnowledgeGraph::from_triples(&triples);
        let nb = graph.neighbours("http://x");
        assert_eq!(nb.len(), 1);
        assert_eq!(nb[0].0, "http://p");
        assert_eq!(nb[0].1, "http://y");
    }

    #[test]
    fn test_knowledge_graph_missing_node_returns_empty() {
        let graph = KnowledgeGraph::from_triples(&[]);
        assert!(graph.neighbours("http://nobody").is_empty());
    }

    // ── MultiHopPathFinder::find_paths ────────────────────────────────────

    #[test]
    fn test_find_paths_direct_one_hop() {
        let (_, graph) = simple_graph();
        let finder = MultiHopPathFinder::with_defaults();
        let paths = finder.find_paths("http://a", "http://b", 1, &graph);
        assert!(!paths.is_empty());
        assert_eq!(paths[0].hop_count(), 1);
    }

    #[test]
    fn test_find_paths_two_hop() {
        let (_, graph) = simple_graph();
        let finder = MultiHopPathFinder::with_defaults();
        let paths = finder.find_paths("http://a", "http://c", 3, &graph);
        // direct (1 hop) and via b (2 hops) — both valid
        assert!(!paths.is_empty());
        let hop_counts: Vec<usize> = paths.iter().map(|p| p.hop_count()).collect();
        assert!(
            hop_counts.contains(&1) || hop_counts.contains(&2),
            "Expected 1- or 2-hop path"
        );
    }

    #[test]
    fn test_find_paths_no_path_returns_empty() {
        let triples = vec![Triple::new("http://a", "http://p", "http://b")];
        let graph = KnowledgeGraph::from_triples(&triples);
        let finder = MultiHopPathFinder::with_defaults();
        // No path from b to a (directed)
        let paths = finder.find_paths("http://b", "http://a", 3, &graph);
        assert!(paths.is_empty());
    }

    #[test]
    fn test_find_paths_respects_max_hops() {
        let (_, graph) = simple_graph();
        let finder = MultiHopPathFinder::new(MultiHopReasoningConfig {
            max_hops: 1,
            ..Default::default()
        });
        // Only 1-hop direct path a->c should be found; 2-hop via b should NOT
        let paths = finder.find_paths("http://a", "http://c", 1, &graph);
        for p in &paths {
            assert!(
                p.hop_count() <= 1,
                "Found path with {} hops > max 1",
                p.hop_count()
            );
        }
    }

    #[test]
    fn test_find_paths_sorted_descending() {
        let (_, graph) = simple_graph();
        let finder = MultiHopPathFinder::with_defaults();
        let paths = finder.find_paths("http://a", "http://c", 3, &graph);
        for i in 1..paths.len() {
            assert!(
                paths[i - 1].score >= paths[i].score,
                "Paths not sorted: {} < {}",
                paths[i - 1].score,
                paths[i].score
            );
        }
    }

    #[test]
    fn test_find_paths_min_confidence_filters() {
        let (_, graph) = simple_graph();
        let finder = MultiHopPathFinder::new(MultiHopReasoningConfig {
            min_confidence: 10.0, // impossibly high
            ..Default::default()
        });
        let paths = finder.find_paths("http://a", "http://b", 3, &graph);
        assert!(paths.is_empty());
    }

    #[test]
    fn test_hop_path_hop_count() {
        let path = HopPath {
            entities: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            relations: vec!["r1".to_string(), "r2".to_string()],
            score: 0.75,
        };
        assert_eq!(path.hop_count(), 2);
    }

    #[test]
    fn test_find_paths_three_hop() {
        let (_, graph) = simple_graph();
        let finder = MultiHopPathFinder::with_defaults();
        let paths = finder.find_paths("http://a", "http://d", 3, &graph);
        // a->b->c->d is a 3-hop path
        assert!(!paths.is_empty());
        let three_hop = paths.iter().any(|p| p.hop_count() == 3);
        assert!(three_hop, "Expected at least one 3-hop path");
    }

    #[test]
    fn test_find_paths_uniform_scoring() {
        let (_, graph) = simple_graph();
        let finder = MultiHopPathFinder::new(MultiHopReasoningConfig {
            path_scoring: PathScoring::Uniform,
            ..Default::default()
        });
        let paths = finder.find_paths("http://a", "http://b", 3, &graph);
        for p in &paths {
            assert!((p.score - 1.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_path_scoring_path_length_formula() {
        let finder = MultiHopPathFinder::with_defaults();
        // 2 hops: 1/(1+2) = 1/3
        let rels = vec!["r1".to_string(), "r2".to_string()];
        let s = finder.score_path(&rels, &PathScoring::PathLength);
        assert!((s - 1.0 / 3.0).abs() < 1e-9, "Expected 1/3, got {s}");
    }

    #[test]
    fn test_find_paths_max_paths_per_pair() {
        // Create a star-like graph from a to many targets, all 1-hop
        let triples: Vec<Triple> = (0..50)
            .map(|i| Triple::new("http://src", "http://p", format!("http://t{i}")))
            .chain(std::iter::once(Triple::new(
                "http://src",
                "http://p",
                "http://target",
            )))
            .collect();
        let graph = KnowledgeGraph::from_triples(&triples);
        let finder = MultiHopPathFinder::new(MultiHopReasoningConfig {
            max_paths_per_pair: 3,
            ..Default::default()
        });
        let paths = finder.find_paths("http://src", "http://target", 1, &graph);
        assert!(paths.len() <= 3);
    }
}
