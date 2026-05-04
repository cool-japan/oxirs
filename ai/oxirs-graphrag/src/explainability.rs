//! Explainability engine for graph-based RAG — attention weights, path explanation, provenance.
//!
//! Provides human-interpretable explanations of why a set of triples was
//! retrieved for a given query.  Three explanation strategies:
//!
//! 1. **Attention weights** — normalized relevance scores per triple
//! 2. **Path explanation** — BFS shortest-path from query entity to result entity
//! 3. **Provenance** — source documents / IRIs that contributed to the answer

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// ─────────────────────────────────────────────────────────────────────────────
// Core types
// ─────────────────────────────────────────────────────────────────────────────

/// A knowledge graph triple with an associated relevance score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    /// Relevance score in [0, 1].
    pub score: f64,
    /// Optional provenance source IRI.
    pub source: Option<String>,
}

impl ScoredTriple {
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
        score: f64,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            score: score.clamp(0.0, 1.0),
            source: None,
        }
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Attention weights
// ─────────────────────────────────────────────────────────────────────────────

/// Normalized attention weights over a set of triples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionExplanation {
    /// Triples with softmax-normalized attention weights.
    pub weighted_triples: Vec<ScoredTriple>,
    /// The query that generated these weights.
    pub query: String,
    /// Entropy of the weight distribution (higher = more diffuse attention).
    pub attention_entropy: f64,
}

impl AttentionExplanation {
    /// Compute softmax-normalized attention weights for the given triples.
    ///
    /// The `raw_scores` slice must have the same length as `triples`.
    pub fn compute(query: &str, triples: &[ScoredTriple], raw_scores: &[f64]) -> Self {
        assert_eq!(triples.len(), raw_scores.len(), "lengths must match");
        let weights = softmax(raw_scores);
        let entropy = shannon_entropy(&weights);

        let weighted_triples = triples
            .iter()
            .zip(weights.iter())
            .map(|(t, &w)| {
                let mut wt = t.clone();
                wt.score = w;
                wt
            })
            .collect();

        Self {
            weighted_triples,
            query: query.to_string(),
            attention_entropy: entropy,
        }
    }

    /// Return the top-k triples by attention weight.
    pub fn top_k(&self, k: usize) -> Vec<&ScoredTriple> {
        let mut sorted: Vec<&ScoredTriple> = self.weighted_triples.iter().collect();
        sorted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(k).collect()
    }
}

fn softmax(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / scores.len() as f64; scores.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

fn shannon_entropy(probs: &[f64]) -> f64 {
    probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// Path explanation
// ─────────────────────────────────────────────────────────────────────────────

/// One hop in a graph path.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PathHop {
    pub from: String,
    pub predicate: String,
    pub to: String,
}

/// A BFS shortest-path explanation from a query entity to a result entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathExplanation {
    pub from: String,
    pub to: String,
    /// Ordered sequence of hops forming the shortest path.
    pub hops: Vec<PathHop>,
    /// Total path length (number of hops).
    pub path_length: usize,
}

impl PathExplanation {
    /// Find the BFS shortest path in a set of triples from `from` to `to`.
    ///
    /// Returns `None` if no path exists.
    pub fn find(triples: &[ScoredTriple], from: &str, to: &str) -> Option<Self> {
        if from == to {
            return Some(Self {
                from: from.to_string(),
                to: to.to_string(),
                hops: Vec::new(),
                path_length: 0,
            });
        }

        // Build adjacency list
        let mut adj: HashMap<&str, Vec<(&str, &str)>> = HashMap::new();
        for t in triples {
            adj.entry(&t.subject)
                .or_default()
                .push((&t.predicate, &t.object));
        }

        // BFS
        let mut queue: VecDeque<(&str, Vec<PathHop>)> = VecDeque::new();
        let mut visited: HashSet<&str> = HashSet::new();
        queue.push_back((from, Vec::new()));
        visited.insert(from);

        while let Some((node, path)) = queue.pop_front() {
            if let Some(neighbors) = adj.get(node) {
                for &(pred, obj) in neighbors {
                    if visited.contains(obj) {
                        continue;
                    }
                    let mut new_path = path.clone();
                    new_path.push(PathHop {
                        from: node.to_string(),
                        predicate: pred.to_string(),
                        to: obj.to_string(),
                    });
                    if obj == to {
                        let length = new_path.len();
                        return Some(Self {
                            from: from.to_string(),
                            to: to.to_string(),
                            hops: new_path,
                            path_length: length,
                        });
                    }
                    visited.insert(obj);
                    queue.push_back((obj, new_path));
                }
            }
        }
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Provenance
// ─────────────────────────────────────────────────────────────────────────────

/// Source provenance for a set of triples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceReport {
    /// Mapping from source IRI to the triples sourced from it.
    pub sources: HashMap<String, Vec<ScoredTriple>>,
    /// Number of triples with no provenance information.
    pub unknown_count: usize,
}

impl ProvenanceReport {
    /// Build a provenance report from a set of scored triples.
    pub fn from_triples(triples: &[ScoredTriple]) -> Self {
        let mut sources: HashMap<String, Vec<ScoredTriple>> = HashMap::new();
        let mut unknown_count = 0;
        for t in triples {
            match &t.source {
                Some(src) => sources.entry(src.clone()).or_default().push(t.clone()),
                None => unknown_count += 1,
            }
        }
        Self {
            sources,
            unknown_count,
        }
    }

    /// Return all distinct source IRIs.
    pub fn source_iris(&self) -> Vec<&str> {
        self.sources.keys().map(|s| s.as_str()).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ExplainabilityEngine
// ─────────────────────────────────────────────────────────────────────────────

/// Top-level explainability engine — wraps all explanation strategies.
pub struct ExplainabilityEngine;

impl ExplainabilityEngine {
    pub fn new() -> Self {
        Self
    }

    /// Compute attention-based explanation.
    pub fn explain_attention(
        &self,
        query: &str,
        triples: &[ScoredTriple],
        raw_scores: &[f64],
    ) -> AttentionExplanation {
        AttentionExplanation::compute(query, triples, raw_scores)
    }

    /// Find the shortest explanatory path from `from` to `to`.
    pub fn explain_path(
        &self,
        triples: &[ScoredTriple],
        from: &str,
        to: &str,
    ) -> Option<PathExplanation> {
        PathExplanation::find(triples, from, to)
    }

    /// Build a provenance report for the given triples.
    pub fn explain_provenance(&self, triples: &[ScoredTriple]) -> ProvenanceReport {
        ProvenanceReport::from_triples(triples)
    }
}

impl Default for ExplainabilityEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_triples() -> Vec<ScoredTriple> {
        vec![
            ScoredTriple::new("Alice", "knows", "Bob", 0.9).with_source("doc:1"),
            ScoredTriple::new("Bob", "worksAt", "Acme", 0.7).with_source("doc:2"),
            ScoredTriple::new("Alice", "livesIn", "Tokyo", 0.5).with_source("doc:1"),
        ]
    }

    #[test]
    fn test_attention_softmax_sum_to_one() {
        let triples = make_triples();
        let raw = vec![1.0, 2.0, 0.5];
        let expl = AttentionExplanation::compute("Who does Alice know?", &triples, &raw);
        let total: f64 = expl.weighted_triples.iter().map(|t| t.score).sum();
        assert!((total - 1.0).abs() < 1e-9, "weights must sum to 1");
    }

    #[test]
    fn test_attention_top_k() {
        let triples = make_triples();
        let raw = vec![3.0, 1.0, 2.0];
        let expl = AttentionExplanation::compute("q", &triples, &raw);
        let top1 = expl.top_k(1);
        assert_eq!(top1[0].subject, "Alice");
        assert_eq!(top1[0].predicate, "knows");
    }

    #[test]
    fn test_attention_entropy_uniform() {
        let triples = make_triples();
        let raw = vec![1.0, 1.0, 1.0]; // uniform → max entropy for 3 items
        let expl = AttentionExplanation::compute("q", &triples, &raw);
        // entropy of [1/3, 1/3, 1/3] = ln(3) ≈ 1.099
        assert!(
            expl.attention_entropy > 1.09,
            "uniform should have high entropy"
        );
    }

    #[test]
    fn test_path_explanation_direct_hop() {
        let triples = make_triples();
        let path = PathExplanation::find(&triples, "Alice", "Bob").unwrap();
        assert_eq!(path.path_length, 1);
        assert_eq!(path.hops[0].predicate, "knows");
    }

    #[test]
    fn test_path_explanation_two_hops() {
        let triples = make_triples();
        let path = PathExplanation::find(&triples, "Alice", "Acme").unwrap();
        assert_eq!(path.path_length, 2);
    }

    #[test]
    fn test_path_explanation_no_path() {
        let triples = make_triples();
        let path = PathExplanation::find(&triples, "Alice", "XYZ");
        assert!(path.is_none(), "no path to unknown node");
    }

    #[test]
    fn test_path_explanation_same_node() {
        let triples = make_triples();
        let path = PathExplanation::find(&triples, "Alice", "Alice").unwrap();
        assert_eq!(path.path_length, 0);
        assert!(path.hops.is_empty());
    }

    #[test]
    fn test_provenance_report() {
        let triples = make_triples();
        let report = ProvenanceReport::from_triples(&triples);
        let mut sources = report.source_iris();
        sources.sort();
        assert_eq!(sources, vec!["doc:1", "doc:2"]);
        assert_eq!(report.unknown_count, 0);
    }

    #[test]
    fn test_provenance_unknown_triples() {
        let triples = vec![
            ScoredTriple::new("A", "p", "B", 0.5), // no source
        ];
        let report = ProvenanceReport::from_triples(&triples);
        assert_eq!(report.unknown_count, 1);
        assert!(report.sources.is_empty());
    }

    #[test]
    fn test_explainability_engine_integration() {
        let engine = ExplainabilityEngine::new();
        let triples = make_triples();
        let raw = vec![0.8, 0.6, 0.4];

        let attn = engine.explain_attention("query", &triples, &raw);
        assert!(!attn.weighted_triples.is_empty());

        let path = engine.explain_path(&triples, "Alice", "Acme");
        assert!(path.is_some());

        let prov = engine.explain_provenance(&triples);
        assert!(!prov.sources.is_empty());
    }
}
