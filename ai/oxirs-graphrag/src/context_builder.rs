//! Context building for graph-based RAG.
//!
//! Extracts N-hop subgraph neighborhoods, retrieves entity connections,
//! filters by predicate, ranks triples by relevance, truncates to a token
//! budget, formats structured context from templates, merges multi-entity
//! contexts, and deduplicates redundant triples.

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single RDF-like triple used as context.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContextTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl ContextTriple {
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

    /// Approximate token count for budget estimation.
    ///
    /// Uses a simple heuristic: split on whitespace and separators.
    pub fn estimated_tokens(&self) -> usize {
        let text = format!("{} {} {}", self.subject, self.predicate, self.object);
        // Rough tokenizer: count whitespace-delimited + URI path segments.
        text.split(|c: char| c.is_whitespace() || c == '/' || c == '#')
            .filter(|s| !s.is_empty())
            .count()
    }
}

/// A scored triple (used during ranking).
#[derive(Debug, Clone)]
pub struct ScoredTriple {
    pub triple: ContextTriple,
    pub score: f64,
}

/// Built context ready for LLM consumption.
#[derive(Debug, Clone)]
pub struct BuiltContext {
    /// The selected triples (ordered by relevance).
    pub triples: Vec<ContextTriple>,
    /// Formatted text output.
    pub text: String,
    /// Estimated total token count.
    pub estimated_tokens: usize,
    /// Number of triples before truncation.
    pub total_candidates: usize,
}

/// Configuration for the context builder.
#[derive(Debug, Clone)]
pub struct ContextBuilderConfig {
    /// Maximum number of hops for subgraph extraction.
    pub max_hops: usize,
    /// Token budget for the final context.
    pub token_budget: usize,
    /// Template string for formatting each triple.
    /// `{s}`, `{p}`, `{o}` are replaced with subject, predicate, object.
    pub triple_template: String,
    /// Separator between triples in the formatted output.
    pub separator: String,
}

impl Default for ContextBuilderConfig {
    fn default() -> Self {
        Self {
            max_hops: 2,
            token_budget: 2048,
            triple_template: "{s} -- {p} --> {o}".to_string(),
            separator: "\n".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Knowledge graph (simple in-memory adjacency)
// ---------------------------------------------------------------------------

/// A simple in-memory knowledge graph for context extraction.
///
/// Stores triples and supports N-hop traversal.
pub struct KnowledgeGraph {
    /// All triples.
    triples: Vec<ContextTriple>,
    /// subject -> list of triple indices.
    subject_index: HashMap<String, Vec<usize>>,
    /// object -> list of triple indices.
    object_index: HashMap<String, Vec<usize>>,
    /// predicate -> list of triple indices.
    predicate_index: HashMap<String, Vec<usize>>,
}

impl KnowledgeGraph {
    /// Create an empty knowledge graph.
    pub fn new() -> Self {
        Self {
            triples: Vec::new(),
            subject_index: HashMap::new(),
            object_index: HashMap::new(),
            predicate_index: HashMap::new(),
        }
    }

    /// Build a knowledge graph from a slice of triples.
    pub fn from_triples(triples: &[ContextTriple]) -> Self {
        let mut kg = Self::new();
        for t in triples {
            kg.add_triple(t.clone());
        }
        kg
    }

    /// Add a single triple.
    pub fn add_triple(&mut self, triple: ContextTriple) {
        let idx = self.triples.len();
        self.subject_index
            .entry(triple.subject.clone())
            .or_default()
            .push(idx);
        self.object_index
            .entry(triple.object.clone())
            .or_default()
            .push(idx);
        self.predicate_index
            .entry(triple.predicate.clone())
            .or_default()
            .push(idx);
        self.triples.push(triple);
    }

    /// Number of triples in the graph.
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Whether the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Get all triples where `entity` appears as subject or object.
    pub fn neighbors(&self, entity: &str) -> Vec<&ContextTriple> {
        let mut result: Vec<&ContextTriple> = Vec::new();
        if let Some(indices) = self.subject_index.get(entity) {
            for &idx in indices {
                result.push(&self.triples[idx]);
            }
        }
        if let Some(indices) = self.object_index.get(entity) {
            for &idx in indices {
                result.push(&self.triples[idx]);
            }
        }
        result
    }

    /// Get all triples with a specific predicate.
    pub fn triples_by_predicate(&self, predicate: &str) -> Vec<&ContextTriple> {
        self.predicate_index
            .get(predicate)
            .map(|indices| indices.iter().map(|&idx| &self.triples[idx]).collect())
            .unwrap_or_default()
    }

    /// Get all triples.
    pub fn all_triples(&self) -> &[ContextTriple] {
        &self.triples
    }
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ContextBuilder
// ---------------------------------------------------------------------------

/// Builds LLM context from a knowledge graph by extracting relevant subgraphs.
pub struct ContextBuilder {
    config: ContextBuilderConfig,
}

impl ContextBuilder {
    /// Create a builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: ContextBuilderConfig::default(),
        }
    }

    /// Create with a custom configuration.
    pub fn with_config(config: ContextBuilderConfig) -> Self {
        Self { config }
    }

    // ── Subgraph extraction ──────────────────────────────────────────────

    /// Extract the N-hop neighborhood around `entity`.
    pub fn extract_neighborhood(
        &self,
        kg: &KnowledgeGraph,
        entity: &str,
        max_hops: Option<usize>,
    ) -> Vec<ContextTriple> {
        let hops = max_hops.unwrap_or(self.config.max_hops);
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        let mut result_set: HashSet<ContextTriple> = HashSet::new();

        queue.push_back((entity.to_string(), 0));
        visited.insert(entity.to_string());

        while let Some((current, depth)) = queue.pop_front() {
            let neighbors = kg.neighbors(&current);
            for triple in neighbors {
                result_set.insert(triple.clone());

                if depth < hops {
                    // Expand to connected entities.
                    let other = if triple.subject == current {
                        &triple.object
                    } else {
                        &triple.subject
                    };
                    if !visited.contains(other.as_str()) {
                        visited.insert(other.clone());
                        queue.push_back((other.clone(), depth + 1));
                    }
                }
            }
        }

        result_set.into_iter().collect()
    }

    /// Retrieve direct connections of an entity (1-hop neighborhood).
    pub fn entity_neighborhood(&self, kg: &KnowledgeGraph, entity: &str) -> Vec<ContextTriple> {
        self.extract_neighborhood(kg, entity, Some(1))
    }

    /// Retrieve all triples involving a specific predicate.
    pub fn relation_context(&self, kg: &KnowledgeGraph, predicate: &str) -> Vec<ContextTriple> {
        kg.triples_by_predicate(predicate)
            .into_iter()
            .cloned()
            .collect()
    }

    // ── Ranking ──────────────────────────────────────────────────────────

    /// Rank triples by relevance to a set of seed entities.
    ///
    /// Triples mentioning more seed entities score higher.
    pub fn rank_triples(
        &self,
        triples: &[ContextTriple],
        seed_entities: &[&str],
    ) -> Vec<ScoredTriple> {
        let seeds: HashSet<&str> = seed_entities.iter().copied().collect();
        let mut scored: Vec<ScoredTriple> = triples
            .iter()
            .map(|t| {
                let mut score = 0.0;
                if seeds.contains(t.subject.as_str()) {
                    score += 1.0;
                }
                if seeds.contains(t.object.as_str()) {
                    score += 1.0;
                }
                // Boost triples connecting two seed entities.
                if seeds.contains(t.subject.as_str()) && seeds.contains(t.object.as_str()) {
                    score += 0.5;
                }
                ScoredTriple {
                    triple: t.clone(),
                    score,
                }
            })
            .collect();

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored
    }

    // ── Truncation ───────────────────────────────────────────────────────

    /// Truncate a list of triples to fit within the token budget.
    pub fn truncate_to_budget(&self, triples: &[ContextTriple]) -> Vec<ContextTriple> {
        let mut result = Vec::new();
        let mut tokens_used = 0;

        for t in triples {
            let t_tokens = t.estimated_tokens();
            if tokens_used + t_tokens > self.config.token_budget {
                break;
            }
            tokens_used += t_tokens;
            result.push(t.clone());
        }

        result
    }

    // ── Formatting ───────────────────────────────────────────────────────

    /// Format a list of triples using the configured template.
    pub fn format_triples(&self, triples: &[ContextTriple]) -> String {
        triples
            .iter()
            .map(|t| {
                self.config
                    .triple_template
                    .replace("{s}", &t.subject)
                    .replace("{p}", &t.predicate)
                    .replace("{o}", &t.object)
            })
            .collect::<Vec<_>>()
            .join(&self.config.separator)
    }

    // ── Multi-entity merging ─────────────────────────────────────────────

    /// Merge contexts extracted for multiple entities, deduplicating triples.
    pub fn merge_contexts(&self, contexts: &[Vec<ContextTriple>]) -> Vec<ContextTriple> {
        let mut seen: HashSet<ContextTriple> = HashSet::new();
        let mut merged: Vec<ContextTriple> = Vec::new();

        for ctx in contexts {
            for triple in ctx {
                if seen.insert(triple.clone()) {
                    merged.push(triple.clone());
                }
            }
        }

        merged
    }

    /// Deduplicate a list of triples (preserving order).
    pub fn deduplicate(&self, triples: &[ContextTriple]) -> Vec<ContextTriple> {
        let mut seen: HashSet<&ContextTriple> = HashSet::new();
        let mut result: Vec<ContextTriple> = Vec::new();
        for t in triples {
            if seen.insert(t) {
                result.push(t.clone());
            }
        }
        result
    }

    // ── Full build pipeline ──────────────────────────────────────────────

    /// Build context for a single entity.
    pub fn build(&self, kg: &KnowledgeGraph, entity: &str) -> BuiltContext {
        let candidates = self.extract_neighborhood(kg, entity, None);
        let total_candidates = candidates.len();
        let ranked = self.rank_triples(&candidates, &[entity]);
        let ranked_triples: Vec<ContextTriple> = ranked.into_iter().map(|st| st.triple).collect();
        let truncated = self.truncate_to_budget(&ranked_triples);
        let text = self.format_triples(&truncated);
        let estimated_tokens: usize = truncated.iter().map(|t| t.estimated_tokens()).sum();

        BuiltContext {
            triples: truncated,
            text,
            estimated_tokens,
            total_candidates,
        }
    }

    /// Build context for multiple entities, merging and deduplicating.
    pub fn build_multi(&self, kg: &KnowledgeGraph, entities: &[&str]) -> BuiltContext {
        let mut all_contexts: Vec<Vec<ContextTriple>> = Vec::new();
        for &entity in entities {
            all_contexts.push(self.extract_neighborhood(kg, entity, None));
        }
        let merged = self.merge_contexts(&all_contexts);
        let total_candidates = merged.len();
        let ranked = self.rank_triples(&merged, entities);
        let ranked_triples: Vec<ContextTriple> = ranked.into_iter().map(|st| st.triple).collect();
        let truncated = self.truncate_to_budget(&ranked_triples);
        let text = self.format_triples(&truncated);
        let estimated_tokens: usize = truncated.iter().map(|t| t.estimated_tokens()).sum();

        BuiltContext {
            triples: truncated,
            text,
            estimated_tokens,
            total_candidates,
        }
    }

    /// Access the configuration.
    pub fn config(&self) -> &ContextBuilderConfig {
        &self.config
    }
}

impl Default for ContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_kg() -> KnowledgeGraph {
        KnowledgeGraph::from_triples(&[
            ContextTriple::new("Alice", "knows", "Bob"),
            ContextTriple::new("Bob", "knows", "Charlie"),
            ContextTriple::new("Charlie", "knows", "Dave"),
            ContextTriple::new("Alice", "likes", "Music"),
            ContextTriple::new("Bob", "likes", "Sports"),
            ContextTriple::new("Dave", "likes", "Art"),
        ])
    }

    fn builder() -> ContextBuilder {
        ContextBuilder::new()
    }

    // ── ContextTriple ────────────────────────────────────────────────────

    #[test]
    fn test_context_triple_new() {
        let t = ContextTriple::new("s", "p", "o");
        assert_eq!(t.subject, "s");
        assert_eq!(t.predicate, "p");
        assert_eq!(t.object, "o");
    }

    #[test]
    fn test_context_triple_estimated_tokens() {
        let t = ContextTriple::new("Alice", "knows", "Bob");
        assert!(t.estimated_tokens() >= 3);
    }

    #[test]
    fn test_context_triple_equality() {
        let a = ContextTriple::new("s", "p", "o");
        let b = ContextTriple::new("s", "p", "o");
        assert_eq!(a, b);
    }

    #[test]
    fn test_context_triple_inequality() {
        let a = ContextTriple::new("s", "p", "o1");
        let b = ContextTriple::new("s", "p", "o2");
        assert_ne!(a, b);
    }

    // ── KnowledgeGraph ───────────────────────────────────────────────────

    #[test]
    fn test_kg_new_empty() {
        let kg = KnowledgeGraph::new();
        assert!(kg.is_empty());
        assert_eq!(kg.len(), 0);
    }

    #[test]
    fn test_kg_from_triples() {
        let kg = sample_kg();
        assert_eq!(kg.len(), 6);
        assert!(!kg.is_empty());
    }

    #[test]
    fn test_kg_add_triple() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple(ContextTriple::new("A", "r", "B"));
        assert_eq!(kg.len(), 1);
    }

    #[test]
    fn test_kg_neighbors() {
        let kg = sample_kg();
        let n = kg.neighbors("Alice");
        // Alice is subject of "knows Bob" and "likes Music"
        assert_eq!(n.len(), 2);
    }

    #[test]
    fn test_kg_neighbors_as_object() {
        let kg = sample_kg();
        let n = kg.neighbors("Bob");
        // Bob is subject in 2 triples and object in 1
        assert_eq!(n.len(), 3);
    }

    #[test]
    fn test_kg_neighbors_unknown_entity() {
        let kg = sample_kg();
        let n = kg.neighbors("Unknown");
        assert!(n.is_empty());
    }

    #[test]
    fn test_kg_triples_by_predicate() {
        let kg = sample_kg();
        let knows = kg.triples_by_predicate("knows");
        assert_eq!(knows.len(), 3);
    }

    #[test]
    fn test_kg_triples_by_predicate_unknown() {
        let kg = sample_kg();
        assert!(kg.triples_by_predicate("unknown").is_empty());
    }

    #[test]
    fn test_kg_all_triples() {
        let kg = sample_kg();
        assert_eq!(kg.all_triples().len(), 6);
    }

    #[test]
    fn test_kg_default() {
        let kg = KnowledgeGraph::default();
        assert!(kg.is_empty());
    }

    // ── Neighborhood extraction ──────────────────────────────────────────

    #[test]
    fn test_extract_neighborhood_1_hop() {
        let kg = sample_kg();
        let b = builder();
        let ctx = b.extract_neighborhood(&kg, "Alice", Some(1));
        // Direct connections only: (Alice, knows, Bob) and (Alice, likes, Music)
        assert!(ctx.len() >= 2);
    }

    #[test]
    fn test_extract_neighborhood_2_hops() {
        let kg = sample_kg();
        let b = builder();
        let ctx = b.extract_neighborhood(&kg, "Alice", Some(2));
        // Should include Bob's connections too
        assert!(ctx.len() > 2);
    }

    #[test]
    fn test_extract_neighborhood_0_hops() {
        let kg = sample_kg();
        let b = builder();
        let ctx = b.extract_neighborhood(&kg, "Alice", Some(0));
        // Only direct triples mentioning Alice
        assert!(!ctx.is_empty());
    }

    #[test]
    fn test_entity_neighborhood() {
        let kg = sample_kg();
        let b = builder();
        let ctx = b.entity_neighborhood(&kg, "Bob");
        assert!(ctx.len() >= 3); // 2 outgoing + 1 incoming
    }

    // ── Relation context ─────────────────────────────────────────────────

    #[test]
    fn test_relation_context() {
        let kg = sample_kg();
        let b = builder();
        let ctx = b.relation_context(&kg, "likes");
        assert_eq!(ctx.len(), 3);
    }

    #[test]
    fn test_relation_context_unknown() {
        let kg = sample_kg();
        let b = builder();
        let ctx = b.relation_context(&kg, "hates");
        assert!(ctx.is_empty());
    }

    // ── Ranking ──────────────────────────────────────────────────────────

    #[test]
    fn test_rank_triples_seed_first() {
        let b = builder();
        let triples = vec![
            ContextTriple::new("X", "r", "Y"),
            ContextTriple::new("Alice", "r", "Bob"),
        ];
        let ranked = b.rank_triples(&triples, &["Alice"]);
        assert_eq!(ranked[0].triple.subject, "Alice");
    }

    #[test]
    fn test_rank_triples_both_seeds_highest() {
        let b = builder();
        let triples = vec![
            ContextTriple::new("Alice", "r", "X"),
            ContextTriple::new("Alice", "r", "Bob"),
        ];
        let ranked = b.rank_triples(&triples, &["Alice", "Bob"]);
        // Alice→Bob has score 2.5 (1+1+0.5), Alice→X has score 1.0
        assert!(ranked[0].score > ranked[1].score);
    }

    #[test]
    fn test_rank_triples_empty() {
        let b = builder();
        let ranked = b.rank_triples(&[], &["Alice"]);
        assert!(ranked.is_empty());
    }

    // ── Truncation ───────────────────────────────────────────────────────

    #[test]
    fn test_truncate_to_budget() {
        let b = ContextBuilder::with_config(ContextBuilderConfig {
            token_budget: 10,
            ..ContextBuilderConfig::default()
        });
        let triples: Vec<ContextTriple> = (0..100)
            .map(|i| ContextTriple::new(format!("s{i}"), "p", format!("o{i}")))
            .collect();
        let truncated = b.truncate_to_budget(&triples);
        let total_tokens: usize = truncated.iter().map(|t| t.estimated_tokens()).sum();
        assert!(total_tokens <= 10);
    }

    #[test]
    fn test_truncate_to_budget_all_fit() {
        let b = ContextBuilder::with_config(ContextBuilderConfig {
            token_budget: 100_000,
            ..ContextBuilderConfig::default()
        });
        let triples = vec![
            ContextTriple::new("A", "r", "B"),
            ContextTriple::new("C", "r", "D"),
        ];
        let truncated = b.truncate_to_budget(&triples);
        assert_eq!(truncated.len(), 2);
    }

    // ── Formatting ───────────────────────────────────────────────────────

    #[test]
    fn test_format_triples_default_template() {
        let b = builder();
        let triples = vec![ContextTriple::new("Alice", "knows", "Bob")];
        let text = b.format_triples(&triples);
        assert!(text.contains("Alice"));
        assert!(text.contains("knows"));
        assert!(text.contains("Bob"));
    }

    #[test]
    fn test_format_triples_custom_template() {
        let b = ContextBuilder::with_config(ContextBuilderConfig {
            triple_template: "({s}, {p}, {o})".to_string(),
            separator: "; ".to_string(),
            ..ContextBuilderConfig::default()
        });
        let triples = vec![
            ContextTriple::new("A", "r", "B"),
            ContextTriple::new("C", "r", "D"),
        ];
        let text = b.format_triples(&triples);
        assert!(text.contains("(A, r, B); (C, r, D)"));
    }

    #[test]
    fn test_format_triples_empty() {
        let b = builder();
        let text = b.format_triples(&[]);
        assert!(text.is_empty());
    }

    // ── Merging and deduplication ────────────────────────────────────────

    #[test]
    fn test_merge_contexts_deduplicates() {
        let b = builder();
        let t = ContextTriple::new("A", "r", "B");
        let ctx1 = vec![t.clone()];
        let ctx2 = vec![t.clone()];
        let merged = b.merge_contexts(&[ctx1, ctx2]);
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn test_merge_contexts_combines() {
        let b = builder();
        let ctx1 = vec![ContextTriple::new("A", "r", "B")];
        let ctx2 = vec![ContextTriple::new("C", "r", "D")];
        let merged = b.merge_contexts(&[ctx1, ctx2]);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_merge_contexts_empty() {
        let b = builder();
        let merged = b.merge_contexts(&[]);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_deduplicate() {
        let b = builder();
        let t = ContextTriple::new("A", "r", "B");
        let triples = vec![t.clone(), t.clone(), t];
        let deduped = b.deduplicate(&triples);
        assert_eq!(deduped.len(), 1);
    }

    #[test]
    fn test_deduplicate_preserves_order() {
        let b = builder();
        let triples = vec![
            ContextTriple::new("C", "r", "D"),
            ContextTriple::new("A", "r", "B"),
            ContextTriple::new("C", "r", "D"),
        ];
        let deduped = b.deduplicate(&triples);
        assert_eq!(deduped.len(), 2);
        assert_eq!(deduped[0].subject, "C");
        assert_eq!(deduped[1].subject, "A");
    }

    // ── Full build pipeline ──────────────────────────────────────────────

    #[test]
    fn test_build_single_entity() {
        let kg = sample_kg();
        let b = builder();
        let ctx = b.build(&kg, "Alice");
        assert!(!ctx.triples.is_empty());
        assert!(!ctx.text.is_empty());
        assert!(ctx.estimated_tokens > 0);
    }

    #[test]
    fn test_build_unknown_entity() {
        let kg = sample_kg();
        let b = builder();
        let ctx = b.build(&kg, "Unknown");
        assert!(ctx.triples.is_empty());
    }

    #[test]
    fn test_build_multi_entity() {
        let kg = sample_kg();
        let b = builder();
        let ctx = b.build_multi(&kg, &["Alice", "Dave"]);
        assert!(!ctx.triples.is_empty());
        assert!(ctx.total_candidates >= 2);
    }

    #[test]
    fn test_build_multi_empty_entities() {
        let kg = sample_kg();
        let b = builder();
        let ctx = b.build_multi(&kg, &[]);
        assert!(ctx.triples.is_empty());
    }

    // ── Config ───────────────────────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let cfg = ContextBuilderConfig::default();
        assert_eq!(cfg.max_hops, 2);
        assert_eq!(cfg.token_budget, 2048);
    }

    #[test]
    fn test_config_access() {
        let b = builder();
        assert_eq!(b.config().max_hops, 2);
    }

    #[test]
    fn test_builder_default() {
        let b = ContextBuilder::default();
        assert_eq!(b.config().max_hops, 2);
    }

    // ── Token budget edge case ───────────────────────────────────────────

    #[test]
    fn test_zero_token_budget() {
        let b = ContextBuilder::with_config(ContextBuilderConfig {
            token_budget: 0,
            ..ContextBuilderConfig::default()
        });
        let triples = vec![ContextTriple::new("A", "r", "B")];
        let truncated = b.truncate_to_budget(&triples);
        assert!(truncated.is_empty());
    }

    // ── ScoredTriple fields ──────────────────────────────────────────────

    #[test]
    fn test_scored_triple_fields() {
        let st = ScoredTriple {
            triple: ContextTriple::new("A", "r", "B"),
            score: 0.9,
        };
        assert_eq!(st.triple.subject, "A");
        assert!((st.score - 0.9).abs() < 1e-10);
    }

    // ── BuiltContext fields ──────────────────────────────────────────────

    #[test]
    fn test_built_context_total_candidates() {
        let kg = sample_kg();
        let b = builder();
        let ctx = b.build(&kg, "Bob");
        assert!(ctx.total_candidates > 0);
        assert!(ctx.triples.len() <= ctx.total_candidates);
    }
}
