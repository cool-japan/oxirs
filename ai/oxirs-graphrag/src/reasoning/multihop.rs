//! Multi-hop reasoning with rule engine integration
//!
//! This module implements rule-guided multi-hop graph traversal.  Given a set
//! of seed entities and a collection of inference rules (expressed using the
//! oxirs-rule `Rule` / `RuleAtom` / `Term` API), the engine:
//!
//! 1. Loads the subgraph into the rule engine as `RuleAtom::Triple` facts.
//! 2. Runs forward chaining to materialise derived facts.
//! 3. Selects paths between seeds and goal nodes that pass through the
//!    derived facts, ranking them by a configurable path scoring function.
//! 4. Returns scored `HopPath` objects that can be fed back into a context
//!    builder or summariser.
//!
//! The module is intentionally self-contained (no external ML dependencies)
//! and operates purely on the RDF triple model from this crate.

use crate::{GraphRAGError, GraphRAGResult, ScoredEntity, Triple};
use std::collections::{HashMap, HashSet, VecDeque};

// ─── Re-export rule-engine types ────────────────────────────────────────────

// We keep a lightweight re-export to decouple from oxirs-rule's public API
// changes (only the pieces we actually use).
use oxirs_rule::{Rule, RuleAtom, RuleEngine, Term};

// ─── Types ──────────────────────────────────────────────────────────────────

/// Maps a (subject, predicate, object) triple key to the list of rule names that fired it.
pub type FiredRulesMap = HashMap<(String, String, String), Vec<String>>;

/// A directed edge in the knowledge graph
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphEdge {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    /// Whether this edge was derived by rule inference (vs. asserted)
    pub inferred: bool,
}

/// A multi-hop path through the knowledge graph
#[derive(Debug, Clone)]
pub struct HopPath {
    /// Ordered list of edges traversed
    pub edges: Vec<GraphEdge>,
    /// Starting entity
    pub start: String,
    /// Ending entity
    pub end: String,
    /// Path score (higher = more relevant)
    pub score: f64,
    /// Number of inferred edges on this path
    pub inferred_hops: usize,
    /// Rule names that fired to produce inferred edges
    pub fired_rules: Vec<String>,
}

impl HopPath {
    /// Total number of hops (edges)
    pub fn hop_count(&self) -> usize {
        self.edges.len()
    }

    /// Whether this path contains at least one inferred edge
    pub fn has_inferred_hop(&self) -> bool {
        self.inferred_hops > 0
    }
}

// ─── Configuration ──────────────────────────────────────────────────────────

/// Path scoring function variant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PathScoringFn {
    /// Score = 1 / hop_count (prefer shorter paths)
    InverseHopCount,
    /// Score = seed_score / hop_count
    #[default]
    SeedWeighted,
    /// Uniform score of 1.0 for all paths
    Uniform,
    /// Penalise inferred hops: score = (1/hop_count) * (0.8 ^ inferred_hops)
    InferencePenalised,
}

/// Configuration for multi-hop reasoning
#[derive(Debug, Clone)]
pub struct MultiHopConfig {
    /// Maximum hop count per path
    pub max_hops: usize,
    /// Maximum number of paths to return
    pub max_paths: usize,
    /// Maximum edges to process during BFS (budget guard)
    pub max_edges_budget: usize,
    /// Whether to include inferred (rule-derived) edges
    pub include_inferred: bool,
    /// Path scoring function
    pub scoring_fn: PathScoringFn,
    /// Predicates to follow (empty = all)
    pub allowed_predicates: HashSet<String>,
    /// Predicates to skip
    pub blocked_predicates: HashSet<String>,
    /// Minimum path score threshold
    pub min_path_score: f64,
}

impl Default for MultiHopConfig {
    fn default() -> Self {
        Self {
            max_hops: 3,
            max_paths: 50,
            max_edges_budget: 100_000,
            include_inferred: true,
            scoring_fn: PathScoringFn::SeedWeighted,
            allowed_predicates: HashSet::new(),
            blocked_predicates: HashSet::new(),
            min_path_score: 0.0,
        }
    }
}

// ─── Graph builder from Rule atoms ──────────────────────────────────────────

fn atoms_to_edges(atoms: &[RuleAtom], inferred: bool) -> Vec<GraphEdge> {
    atoms
        .iter()
        .filter_map(|atom| match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                let s = term_to_str(subject)?;
                let p = term_to_str(predicate)?;
                let o = term_to_str(object)?;
                Some(GraphEdge {
                    subject: s,
                    predicate: p,
                    object: o,
                    inferred,
                })
            }
            _ => None,
        })
        .collect()
}

fn term_to_str(term: &Term) -> Option<String> {
    match term {
        Term::Constant(c) | Term::Literal(c) => Some(c.clone()),
        Term::Variable(_) | Term::Function { .. } => None, // unbound variables
    }
}

fn triples_to_atoms(triples: &[Triple]) -> Vec<RuleAtom> {
    triples
        .iter()
        .map(|t| RuleAtom::Triple {
            subject: Term::Constant(t.subject.clone()),
            predicate: Term::Constant(t.predicate.clone()),
            object: Term::Constant(t.object.clone()),
        })
        .collect()
}

// ─── Multi-hop engine ────────────────────────────────────────────────────────

/// Multi-hop reasoning engine backed by the oxirs-rule RuleEngine
pub struct MultiHopEngine {
    config: MultiHopConfig,
}

impl Default for MultiHopEngine {
    fn default() -> Self {
        Self::new(MultiHopConfig::default())
    }
}

impl MultiHopEngine {
    pub fn new(config: MultiHopConfig) -> Self {
        Self { config }
    }

    /// Run multi-hop reasoning over `subgraph`, guided by `rules`.
    ///
    /// Returns all scored paths starting from `seeds`.
    pub fn reason(
        &self,
        seeds: &[ScoredEntity],
        subgraph: &[Triple],
        rules: &[Rule],
    ) -> GraphRAGResult<Vec<HopPath>> {
        if seeds.is_empty() || subgraph.is_empty() {
            return Ok(vec![]);
        }

        // 1. Materialise inferred facts with the rule engine
        let (asserted_edges, inferred_edges, fired_rule_map) = self.materialise(subgraph, rules)?;

        // 2. Build adjacency index
        let mut all_edges: Vec<GraphEdge> = asserted_edges;
        if self.config.include_inferred {
            all_edges.extend(inferred_edges);
        }

        let adj = self.build_adjacency(&all_edges);

        // 3. BFS/DFS from each seed to find paths
        let mut paths: Vec<HopPath> = Vec::new();
        let seed_map: HashMap<String, f64> =
            seeds.iter().map(|s| (s.uri.clone(), s.score)).collect();

        for seed in seeds {
            let new_paths =
                self.bfs_paths(&seed.uri, seed.score, &adj, &all_edges, &fired_rule_map);
            paths.extend(new_paths);
        }

        // 4. Score and filter
        paths.retain(|p| p.score >= self.config.min_path_score);
        paths.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        paths.truncate(self.config.max_paths);

        // Suppress unused warning for seed_map (used conceptually above)
        let _ = seed_map;

        Ok(paths)
    }

    /// Materialise inferred facts and return (asserted_edges, inferred_edges, fired_rules_by_triple)
    fn materialise(
        &self,
        subgraph: &[Triple],
        rules: &[Rule],
    ) -> GraphRAGResult<(Vec<GraphEdge>, Vec<GraphEdge>, FiredRulesMap)> {
        let asserted_edges = atoms_to_edges(&triples_to_atoms(subgraph), false);

        if rules.is_empty() {
            return Ok((asserted_edges, vec![], HashMap::new()));
        }

        let mut engine = RuleEngine::new();
        engine.add_rules(rules.to_vec());
        engine.enable_cache();

        let facts = triples_to_atoms(subgraph);

        let inferred_atoms = engine
            .forward_chain(&facts)
            .map_err(|e| GraphRAGError::InternalError(format!("Rule engine error: {e}")))?;

        // Collect inferred triples (skip those already in subgraph)
        let asserted_keys: HashSet<(String, String, String)> = subgraph
            .iter()
            .map(|t| (t.subject.clone(), t.predicate.clone(), t.object.clone()))
            .collect();

        let inferred_edges: Vec<GraphEdge> = atoms_to_edges(&inferred_atoms, true)
            .into_iter()
            .filter(|e| {
                !asserted_keys.contains(&(e.subject.clone(), e.predicate.clone(), e.object.clone()))
            })
            .collect();

        // Build a map from triple → fired rule names (approximation: one rule per triple)
        let fired_rule_map: FiredRulesMap = rules
            .iter()
            .flat_map(|rule| {
                rule.head.iter().filter_map(|atom| match atom {
                    RuleAtom::Triple {
                        subject,
                        predicate,
                        object,
                    } => {
                        let s = term_to_str(subject)?;
                        let p = term_to_str(predicate)?;
                        let o = term_to_str(object)?;
                        Some(((s, p, o), rule.name.clone()))
                    }
                    _ => None,
                })
            })
            .fold(HashMap::new(), |mut acc, (key, rule_name)| {
                acc.entry(key).or_default().push(rule_name);
                acc
            });

        Ok((asserted_edges, inferred_edges, fired_rule_map))
    }

    /// Build an adjacency list: node → list of edge indices
    fn build_adjacency(&self, edges: &[GraphEdge]) -> HashMap<String, Vec<usize>> {
        let mut adj: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, edge) in edges.iter().enumerate() {
            if self.allow_predicate(&edge.predicate) {
                adj.entry(edge.subject.clone()).or_default().push(i);
            }
        }
        adj
    }

    fn allow_predicate(&self, pred: &str) -> bool {
        if !self.config.allowed_predicates.is_empty()
            && !self.config.allowed_predicates.contains(pred)
        {
            return false;
        }
        !self.config.blocked_predicates.contains(pred)
    }

    /// BFS from a seed node; returns all valid paths
    fn bfs_paths(
        &self,
        start: &str,
        seed_score: f64,
        adj: &HashMap<String, Vec<usize>>,
        edges: &[GraphEdge],
        fired_rule_map: &HashMap<(String, String, String), Vec<String>>,
    ) -> Vec<HopPath> {
        // Queue entry: (current_node, path_so_far_edge_indices, visited_nodes)
        struct State {
            node: String,
            edge_path: Vec<usize>,
            visited: HashSet<String>,
        }

        let mut queue: VecDeque<State> = VecDeque::new();
        queue.push_back(State {
            node: start.to_string(),
            edge_path: vec![],
            visited: {
                let mut h = HashSet::new();
                h.insert(start.to_string());
                h
            },
        });

        let mut paths: Vec<HopPath> = Vec::new();
        let mut budget = self.config.max_edges_budget;

        while let Some(state) = queue.pop_front() {
            if budget == 0 {
                break;
            }
            budget -= 1;

            if state.edge_path.len() > self.config.max_hops {
                continue;
            }

            // If we have at least one hop, record as a path
            if !state.edge_path.is_empty() {
                let path_edges: Vec<GraphEdge> =
                    state.edge_path.iter().map(|&i| edges[i].clone()).collect();

                let inferred_hops = path_edges.iter().filter(|e| e.inferred).count();
                let fired_rules: Vec<String> = path_edges
                    .iter()
                    .filter(|e| e.inferred)
                    .flat_map(|e| {
                        let key = (e.subject.clone(), e.predicate.clone(), e.object.clone());
                        fired_rule_map.get(&key).cloned().unwrap_or_default()
                    })
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect();

                let score = self.score_path(state.edge_path.len(), inferred_hops, seed_score);

                paths.push(HopPath {
                    edges: path_edges,
                    start: start.to_string(),
                    end: state.node.clone(),
                    score,
                    inferred_hops,
                    fired_rules,
                });

                if paths.len() >= self.config.max_paths {
                    return paths;
                }
            }

            if state.edge_path.len() >= self.config.max_hops {
                continue;
            }

            // Expand neighbours
            if let Some(edge_indices) = adj.get(&state.node) {
                for &ei in edge_indices {
                    let edge = &edges[ei];
                    if !state.visited.contains(&edge.object) {
                        let mut new_visited = state.visited.clone();
                        new_visited.insert(edge.object.clone());
                        let mut new_path = state.edge_path.clone();
                        new_path.push(ei);
                        queue.push_back(State {
                            node: edge.object.clone(),
                            edge_path: new_path,
                            visited: new_visited,
                        });
                    }
                }
            }
        }

        paths
    }

    fn score_path(&self, hops: usize, inferred_hops: usize, seed_score: f64) -> f64 {
        let h = hops.max(1) as f64;
        match self.config.scoring_fn {
            PathScoringFn::InverseHopCount => 1.0 / h,
            PathScoringFn::SeedWeighted => seed_score / h,
            PathScoringFn::Uniform => 1.0,
            PathScoringFn::InferencePenalised => (1.0 / h) * 0.8_f64.powi(inferred_hops as i32),
        }
    }
}

// ─── Convenience: build rules from SPARQL-like property chains ──────────────

/// Build a transitivity rule for a given predicate
/// e.g.  `subClassOf(X,Y) ∧ subClassOf(Y,Z) → subClassOf(X,Z)`
pub fn transitivity_rule(predicate: &str) -> Rule {
    Rule {
        name: format!("{predicate}_transitive"),
        body: vec![
            RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant(predicate.to_string()),
                object: Term::Variable("Y".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Variable("Y".to_string()),
                predicate: Term::Constant(predicate.to_string()),
                object: Term::Variable("Z".to_string()),
            },
        ],
        head: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant(predicate.to_string()),
            object: Term::Variable("Z".to_string()),
        }],
    }
}

/// Build a property chain rule:  p1(X,Y) ∧ p2(Y,Z) → q(X,Z)
pub fn property_chain_rule(p1: &str, p2: &str, conclusion_pred: &str) -> Rule {
    Rule {
        name: format!("{p1}_{p2}_chain"),
        body: vec![
            RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant(p1.to_string()),
                object: Term::Variable("Y".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Variable("Y".to_string()),
                predicate: Term::Constant(p2.to_string()),
                object: Term::Variable("Z".to_string()),
            },
        ],
        head: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant(conclusion_pred.to_string()),
            object: Term::Variable("Z".to_string()),
        }],
    }
}

/// Build a symmetry rule:  p(X,Y) → p(Y,X)
pub fn symmetry_rule(predicate: &str) -> Rule {
    Rule {
        name: format!("{predicate}_symmetric"),
        body: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant(predicate.to_string()),
            object: Term::Variable("Y".to_string()),
        }],
        head: vec![RuleAtom::Triple {
            subject: Term::Variable("Y".to_string()),
            predicate: Term::Constant(predicate.to_string()),
            object: Term::Variable("X".to_string()),
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScoreSource;

    fn make_seed(uri: &str, score: f64) -> ScoredEntity {
        ScoredEntity {
            uri: uri.to_string(),
            score,
            source: ScoreSource::Vector,
            metadata: HashMap::new(),
        }
    }

    fn make_triple(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(s, p, o)
    }

    // ── rule helper tests ─────────────────────────────────────────────────

    #[test]
    fn test_transitivity_rule_structure() {
        let rule = transitivity_rule("subClassOf");
        assert_eq!(rule.name, "subClassOf_transitive");
        assert_eq!(rule.body.len(), 2);
        assert_eq!(rule.head.len(), 1);
    }

    #[test]
    fn test_property_chain_rule_structure() {
        let rule = property_chain_rule("partOf", "locatedIn", "indirectlyIn");
        assert_eq!(rule.name, "partOf_locatedIn_chain");
        assert_eq!(rule.body.len(), 2);
    }

    #[test]
    fn test_symmetry_rule_structure() {
        let rule = symmetry_rule("sameAs");
        assert_eq!(rule.name, "sameAs_symmetric");
        assert_eq!(rule.body.len(), 1);
        assert_eq!(rule.head.len(), 1);
    }

    // ── MultiHopEngine – basic ─────────────────────────────────────────────

    #[test]
    fn test_reason_empty_seeds() {
        let engine = MultiHopEngine::default();
        let triples = vec![make_triple("http://a", "http://rel", "http://b")];
        let result = engine.reason(&[], &triples, &[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_reason_empty_subgraph() {
        let engine = MultiHopEngine::default();
        let seeds = vec![make_seed("http://a", 0.9)];
        let result = engine.reason(&seeds, &[], &[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_reason_single_hop_no_rules() {
        let engine = MultiHopEngine::default();
        let seeds = vec![make_seed("http://a", 0.9)];
        let triples = vec![
            make_triple("http://a", "http://p/rel", "http://b"),
            make_triple("http://b", "http://p/rel", "http://c"),
            make_triple("http://x", "http://p/other", "http://y"),
        ];
        let paths = engine.reason(&seeds, &triples, &[]).unwrap();
        assert!(!paths.is_empty());
        // All paths should start from http://a
        for p in &paths {
            assert_eq!(p.start, "http://a");
        }
        // There should be no inferred hops (no rules)
        for p in &paths {
            assert_eq!(p.inferred_hops, 0);
        }
    }

    #[test]
    fn test_reason_respects_max_hops() {
        let config = MultiHopConfig {
            max_hops: 1,
            ..Default::default()
        };
        let engine = MultiHopEngine::new(config);
        let seeds = vec![make_seed("http://a", 0.9)];
        let triples = vec![
            make_triple("http://a", "http://p", "http://b"),
            make_triple("http://b", "http://p", "http://c"),
        ];
        let paths = engine.reason(&seeds, &triples, &[]).unwrap();
        for p in &paths {
            assert!(
                p.hop_count() <= 1,
                "Path hop count {} > max_hops 1",
                p.hop_count()
            );
        }
    }

    #[test]
    fn test_reason_respects_max_paths() {
        let config = MultiHopConfig {
            max_paths: 2,
            ..Default::default()
        };
        let engine = MultiHopEngine::new(config);
        let seeds = vec![make_seed("http://a", 0.9)];
        let triples: Vec<Triple> = (0..20)
            .map(|i| make_triple("http://a", "http://p", &format!("http://n{i}")))
            .collect();
        let paths = engine.reason(&seeds, &triples, &[]).unwrap();
        assert!(paths.len() <= 2);
    }

    // ── MultiHopEngine – with rules ───────────────────────────────────────

    #[test]
    fn test_reason_transitivity_rule() {
        let config = MultiHopConfig {
            max_hops: 3,
            include_inferred: true,
            ..Default::default()
        };
        let engine = MultiHopEngine::new(config);
        let seeds = vec![make_seed("http://a", 0.9)];
        let triples = vec![
            make_triple("http://a", "http://subClassOf", "http://b"),
            make_triple("http://b", "http://subClassOf", "http://c"),
        ];
        let rules = vec![transitivity_rule("http://subClassOf")];
        let paths = engine.reason(&seeds, &triples, &rules).unwrap();
        assert!(!paths.is_empty());
        // At least some paths should have inferred hops
        let has_inferred = paths.iter().any(|p| p.has_inferred_hop());
        assert!(
            has_inferred,
            "Expected at least one path with inferred hops"
        );
    }

    #[test]
    fn test_reason_no_inferred_when_disabled() {
        let config = MultiHopConfig {
            include_inferred: false,
            ..Default::default()
        };
        let engine = MultiHopEngine::new(config);
        let seeds = vec![make_seed("http://a", 0.9)];
        let triples = vec![
            make_triple("http://a", "http://subClassOf", "http://b"),
            make_triple("http://b", "http://subClassOf", "http://c"),
        ];
        let rules = vec![transitivity_rule("http://subClassOf")];
        let paths = engine.reason(&seeds, &triples, &rules).unwrap();
        for p in &paths {
            assert_eq!(
                p.inferred_hops, 0,
                "Expected no inferred hops when disabled"
            );
        }
    }

    // ── Path scoring ──────────────────────────────────────────────────────

    #[test]
    fn test_score_inverse_hop_count() {
        let config = MultiHopConfig {
            scoring_fn: PathScoringFn::InverseHopCount,
            ..Default::default()
        };
        let engine = MultiHopEngine::new(config);
        // 1-hop path: score = 1.0, 2-hop: score = 0.5
        assert!((engine.score_path(1, 0, 1.0) - 1.0).abs() < 1e-9);
        assert!((engine.score_path(2, 0, 1.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_score_seed_weighted() {
        let config = MultiHopConfig {
            scoring_fn: PathScoringFn::SeedWeighted,
            ..Default::default()
        };
        let engine = MultiHopEngine::new(config);
        let s = engine.score_path(2, 0, 0.8);
        assert!((s - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_score_uniform() {
        let config = MultiHopConfig {
            scoring_fn: PathScoringFn::Uniform,
            ..Default::default()
        };
        let engine = MultiHopEngine::new(config);
        assert_eq!(engine.score_path(5, 3, 0.5), 1.0);
    }

    #[test]
    fn test_score_inference_penalised() {
        let config = MultiHopConfig {
            scoring_fn: PathScoringFn::InferencePenalised,
            ..Default::default()
        };
        let engine = MultiHopEngine::new(config);
        let s_no_inf = engine.score_path(2, 0, 1.0);
        let s_with_inf = engine.score_path(2, 1, 1.0);
        assert!(s_no_inf > s_with_inf, "Inferred hop should reduce score");
    }

    // ── Predicate filtering ───────────────────────────────────────────────

    #[test]
    fn test_blocked_predicates_filter() {
        let mut config = MultiHopConfig::default();
        config
            .blocked_predicates
            .insert("http://p/blocked".to_string());
        let engine = MultiHopEngine::new(config);
        let seeds = vec![make_seed("http://a", 0.9)];
        let triples = vec![
            make_triple("http://a", "http://p/allowed", "http://b"),
            make_triple("http://a", "http://p/blocked", "http://c"),
        ];
        let paths = engine.reason(&seeds, &triples, &[]).unwrap();
        // No path should traverse the blocked predicate
        for p in &paths {
            for e in &p.edges {
                assert_ne!(
                    e.predicate, "http://p/blocked",
                    "Blocked predicate found in path"
                );
            }
        }
    }

    #[test]
    fn test_allowed_predicates_whitelist() {
        let mut config = MultiHopConfig::default();
        config
            .allowed_predicates
            .insert("http://p/allowed".to_string());
        let engine = MultiHopEngine::new(config);
        let seeds = vec![make_seed("http://a", 0.9)];
        let triples = vec![
            make_triple("http://a", "http://p/allowed", "http://b"),
            make_triple("http://a", "http://p/other", "http://c"),
        ];
        let paths = engine.reason(&seeds, &triples, &[]).unwrap();
        for p in &paths {
            for e in &p.edges {
                assert_eq!(e.predicate, "http://p/allowed");
            }
        }
    }

    // ── Path metadata ─────────────────────────────────────────────────────

    #[test]
    fn test_hop_path_fields() {
        let engine = MultiHopEngine::default();
        let seeds = vec![make_seed("http://a", 0.8)];
        let triples = vec![make_triple("http://a", "http://p", "http://b")];
        let paths = engine.reason(&seeds, &triples, &[]).unwrap();
        assert!(!paths.is_empty());
        let path = &paths[0];
        assert_eq!(path.start, "http://a");
        assert_eq!(path.end, "http://b");
        assert_eq!(path.hop_count(), 1);
        assert!(!path.has_inferred_hop());
    }

    #[test]
    fn test_min_path_score_threshold() {
        let config = MultiHopConfig {
            min_path_score: 99.0, // impossible
            ..Default::default()
        };
        let engine = MultiHopEngine::new(config);
        let seeds = vec![make_seed("http://a", 0.9)];
        let triples = vec![make_triple("http://a", "http://p", "http://b")];
        let paths = engine.reason(&seeds, &triples, &[]).unwrap();
        assert!(paths.is_empty());
    }

    // ── Helper functions ──────────────────────────────────────────────────

    #[test]
    fn test_triples_to_atoms_roundtrip() {
        let triples = vec![Triple::new("http://s", "http://p", "http://o")];
        let atoms = triples_to_atoms(&triples);
        assert_eq!(atoms.len(), 1);
        if let RuleAtom::Triple {
            subject,
            predicate,
            object,
        } = &atoms[0]
        {
            assert_eq!(term_to_str(subject).unwrap(), "http://s");
            assert_eq!(term_to_str(predicate).unwrap(), "http://p");
            assert_eq!(term_to_str(object).unwrap(), "http://o");
        } else {
            panic!("Expected Triple atom");
        }
    }

    #[test]
    fn test_property_chain_produces_derived_edges() {
        let config = MultiHopConfig {
            max_hops: 3,
            include_inferred: true,
            ..Default::default()
        };
        let engine = MultiHopEngine::new(config);
        let seeds = vec![make_seed("http://a", 0.9)];
        let triples = vec![
            make_triple("http://a", "http://partOf", "http://b"),
            make_triple("http://b", "http://locatedIn", "http://c"),
        ];
        let rules = vec![property_chain_rule(
            "http://partOf",
            "http://locatedIn",
            "http://indirectlyIn",
        )];
        let paths = engine.reason(&seeds, &triples, &rules).unwrap();
        assert!(!paths.is_empty());
    }
}

// ─── Additional tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod additional_tests {
    use super::*;
    use crate::ScoreSource;

    fn make_seed(uri: &str, score: f64) -> ScoredEntity {
        ScoredEntity {
            uri: uri.to_string(),
            score,
            source: ScoreSource::Vector,
            metadata: HashMap::new(),
        }
    }

    fn make_triple(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(s, p, o)
    }

    // ── Rule builder tests ────────────────────────────────────────────────

    #[test]
    fn test_symmetry_rule_head_swapped() {
        let rule = symmetry_rule("http://sameAs");
        // head should have subject=Variable(Y), object=Variable(X) — swapped from body
        assert!(matches!(&rule.head[0], RuleAtom::Triple { .. }));
        if let RuleAtom::Triple {
            subject, object, ..
        } = &rule.head[0]
        {
            // Both are Variable terms; they should be different variable names
            match (subject, object) {
                (Term::Variable(sv), Term::Variable(ov)) => {
                    assert_ne!(sv, ov, "Head subject and object variables should differ");
                }
                _ => panic!("Expected Variable terms in head"),
            }
        } else {
            panic!("Expected Triple head");
        }
    }

    #[test]
    fn test_transitivity_rule_has_shared_variable() {
        // body[0].object == body[1].subject (shared Y)
        let rule = transitivity_rule("http://subClassOf");
        if let (RuleAtom::Triple { object: obj0, .. }, RuleAtom::Triple { subject: subj1, .. }) =
            (&rule.body[0], &rule.body[1])
        {
            // Both should be a Variable("Y")
            matches!(obj0, Term::Variable(v) if v == "Y");
            matches!(subj1, Term::Variable(v) if v == "Y");
        }
    }

    #[test]
    fn test_property_chain_rule_body_predicates() {
        let rule = property_chain_rule("http://partOf", "http://locatedIn", "http://indirectlyIn");
        if let RuleAtom::Triple { predicate: p1, .. } = &rule.body[0] {
            assert_eq!(term_to_str(p1).unwrap(), "http://partOf");
        }
        if let RuleAtom::Triple { predicate: p2, .. } = &rule.body[1] {
            assert_eq!(term_to_str(p2).unwrap(), "http://locatedIn");
        }
        if let RuleAtom::Triple { predicate: ph, .. } = &rule.head[0] {
            assert_eq!(term_to_str(ph).unwrap(), "http://indirectlyIn");
        }
    }

    // ── term_to_str helper ────────────────────────────────────────────────

    #[test]
    fn test_term_to_str_constant() {
        let t = Term::Constant("http://example.org/x".to_string());
        assert_eq!(term_to_str(&t).unwrap(), "http://example.org/x");
    }

    #[test]
    fn test_term_to_str_literal() {
        let t = Term::Literal("hello world".to_string());
        assert_eq!(term_to_str(&t).unwrap(), "hello world");
    }

    #[test]
    fn test_term_to_str_variable_returns_none() {
        let t = Term::Variable("X".to_string());
        assert!(term_to_str(&t).is_none());
    }

    // ── atoms_to_edges helper ─────────────────────────────────────────────

    #[test]
    fn test_atoms_to_edges_filters_non_triple() {
        let atoms = vec![RuleAtom::Triple {
            subject: Term::Constant("http://s".to_string()),
            predicate: Term::Constant("http://p".to_string()),
            object: Term::Constant("http://o".to_string()),
        }];
        let edges = atoms_to_edges(&atoms, false);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].subject, "http://s");
        assert!(!edges[0].inferred);
    }

    #[test]
    fn test_atoms_to_edges_inferred_flag() {
        let atoms = vec![RuleAtom::Triple {
            subject: Term::Constant("http://s".to_string()),
            predicate: Term::Constant("http://p".to_string()),
            object: Term::Constant("http://o".to_string()),
        }];
        let edges = atoms_to_edges(&atoms, true);
        assert!(edges[0].inferred);
    }

    // ── Path scoring ──────────────────────────────────────────────────────

    #[test]
    fn test_score_inference_penalised_zero_inferred_equals_inverse() {
        let config = MultiHopConfig {
            scoring_fn: PathScoringFn::InferencePenalised,
            ..Default::default()
        };
        let engine = MultiHopEngine::new(config);
        // 0 inferred hops → 0.8^0 = 1.0 → same as InverseHopCount
        let s = engine.score_path(2, 0, 1.0);
        assert!((s - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_score_seed_weighted_scales_with_seed_score() {
        let config = MultiHopConfig {
            scoring_fn: PathScoringFn::SeedWeighted,
            ..Default::default()
        };
        let engine = MultiHopEngine::new(config);
        let s1 = engine.score_path(1, 0, 1.0);
        let s2 = engine.score_path(1, 0, 0.5);
        assert!((s1 - 2.0 * s2).abs() < 1e-9, "s1={s1}, s2={s2}");
    }

    // ── GraphEdge ─────────────────────────────────────────────────────────

    #[test]
    fn test_graph_edge_equality() {
        let e1 = GraphEdge {
            subject: "http://a".to_string(),
            predicate: "http://p".to_string(),
            object: "http://b".to_string(),
            inferred: false,
        };
        let e2 = e1.clone();
        assert_eq!(e1, e2);
    }

    // ── MultiHopConfig ────────────────────────────────────────────────────

    #[test]
    fn test_multihop_config_defaults() {
        let cfg = MultiHopConfig::default();
        assert_eq!(cfg.max_hops, 3);
        assert_eq!(cfg.max_paths, 50);
        assert!(cfg.include_inferred);
        assert!(cfg.allowed_predicates.is_empty());
        assert!(cfg.blocked_predicates.is_empty());
    }

    // ── Cycle detection ───────────────────────────────────────────────────

    #[test]
    fn test_reason_cycle_in_graph_does_not_loop() {
        // a → b → c → a (cycle); BFS should not loop infinitely
        let engine = MultiHopEngine::default();
        let seeds = vec![make_seed("http://a", 0.9)];
        let triples = vec![
            make_triple("http://a", "http://p", "http://b"),
            make_triple("http://b", "http://p", "http://c"),
            make_triple("http://c", "http://p", "http://a"), // back edge
        ];
        let paths = engine.reason(&seeds, &triples, &[]);
        assert!(paths.is_ok(), "Should not error on cyclic graphs");
        // Should return some paths without hanging
        let paths = paths.unwrap();
        assert!(paths.len() < 1000, "Cycle guard should bound path count");
    }

    #[test]
    fn test_hop_path_has_inferred_hop_false_for_asserted() {
        let path = HopPath {
            edges: vec![GraphEdge {
                subject: "http://s".to_string(),
                predicate: "http://p".to_string(),
                object: "http://o".to_string(),
                inferred: false,
            }],
            start: "http://s".to_string(),
            end: "http://o".to_string(),
            score: 0.8,
            inferred_hops: 0,
            fired_rules: vec![],
        };
        assert!(!path.has_inferred_hop());
        assert_eq!(path.hop_count(), 1);
    }

    #[test]
    fn test_hop_path_has_inferred_hop_true_for_inferred() {
        let path = HopPath {
            edges: vec![GraphEdge {
                subject: "http://s".to_string(),
                predicate: "http://p".to_string(),
                object: "http://o".to_string(),
                inferred: true,
            }],
            start: "http://s".to_string(),
            end: "http://o".to_string(),
            score: 0.5,
            inferred_hops: 1,
            fired_rules: vec!["rule1".to_string()],
        };
        assert!(path.has_inferred_hop());
    }

    // ── Multiple seeds produce paths from each ────────────────────────────

    #[test]
    fn test_reason_two_seeds_produce_paths_from_both() {
        let engine = MultiHopEngine::default();
        let seeds = vec![make_seed("http://a", 0.9), make_seed("http://x", 0.8)];
        let triples = vec![
            make_triple("http://a", "http://p", "http://b"),
            make_triple("http://x", "http://q", "http://y"),
        ];
        let paths = engine.reason(&seeds, &triples, &[]).unwrap();
        let from_a = paths.iter().any(|p| p.start == "http://a");
        let from_x = paths.iter().any(|p| p.start == "http://x");
        assert!(from_a, "Expected paths from http://a");
        assert!(from_x, "Expected paths from http://x");
    }

    // ── Symmetry rule produces inferred edges ─────────────────────────────

    #[test]
    fn test_reason_symmetry_rule_adds_reverse_edge() {
        let config = MultiHopConfig {
            max_hops: 2,
            include_inferred: true,
            ..Default::default()
        };
        let engine = MultiHopEngine::new(config);
        let seeds = vec![make_seed("http://b", 0.9)];
        let triples = vec![make_triple("http://a", "http://sameAs", "http://b")];
        let rules = vec![symmetry_rule("http://sameAs")];
        let paths = engine.reason(&seeds, &triples, &rules).unwrap();
        // Should find path from http://b via the inferred reverse edge
        let has_inferred = paths.iter().any(|p| p.has_inferred_hop());
        assert!(has_inferred, "Symmetry rule should create inferred edges");
    }

    // ── Budget guard ──────────────────────────────────────────────────────

    #[test]
    fn test_reason_budget_guard_limits_expansion() {
        let config = MultiHopConfig {
            max_edges_budget: 5, // very small
            max_hops: 10,
            max_paths: 1000,
            ..Default::default()
        };
        let engine = MultiHopEngine::new(config);
        let seeds = vec![make_seed("http://a", 0.9)];
        // Large star graph: http://a → http://n0..n99
        let triples: Vec<Triple> = (0..100)
            .map(|i| make_triple("http://a", "http://p", &format!("http://n{i}")))
            .collect();
        let paths = engine.reason(&seeds, &triples, &[]).unwrap();
        // Budget=5 should stop after at most a few paths
        assert!(paths.len() < 100, "Budget guard should limit path count");
    }

    // ── Scoring sorted descending ─────────────────────────────────────────

    #[test]
    fn test_reason_paths_sorted_descending() {
        let engine = MultiHopEngine::default();
        let seeds = vec![make_seed("http://a", 1.0)];
        let triples = vec![
            make_triple("http://a", "http://p", "http://b"),
            make_triple("http://b", "http://p", "http://c"),
            make_triple("http://c", "http://p", "http://d"),
        ];
        let paths = engine.reason(&seeds, &triples, &[]).unwrap();
        for i in 1..paths.len() {
            assert!(
                paths[i - 1].score >= paths[i].score,
                "Paths should be sorted descending: {} < {}",
                paths[i - 1].score,
                paths[i].score
            );
        }
    }
}
