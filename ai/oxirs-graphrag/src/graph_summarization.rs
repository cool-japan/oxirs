//! # Graph Summarization for GraphRAG
//!
//! Produces concise summaries of subgraphs for efficient retrieval-augmented
//! generation. Compresses large knowledge subgraphs into textual summaries
//! preserving key entities, relationships, and structural properties.
//!
//! ## Features
//!
//! - **Entity-centric summaries**: Focus on hub entities with highest connectivity
//! - **Relationship summarization**: Aggregate edge types and multiplicities
//! - **Structural feature extraction**: Degree distribution, clustering, paths
//! - **Community detection**: Summarize communities independently
//! - **Hierarchical summarization**: Multi-level detail (brief → detailed)
//! - **Template-based output**: Customisable summary templates

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for graph summarization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSummaryConfig {
    /// Maximum entities to include in summary (default: 20).
    pub max_entities: usize,
    /// Maximum relationships to include (default: 50).
    pub max_relationships: usize,
    /// Minimum entity degree to include (default: 1).
    pub min_entity_degree: usize,
    /// Detail level: Brief, Standard, Detailed (default: Standard).
    pub detail_level: DetailLevel,
    /// Whether to include structural statistics (default: true).
    pub include_stats: bool,
    /// Whether to include community detection (default: true).
    pub include_communities: bool,
}

impl Default for GraphSummaryConfig {
    fn default() -> Self {
        Self {
            max_entities: 20,
            max_relationships: 50,
            min_entity_degree: 1,
            detail_level: DetailLevel::Standard,
            include_stats: true,
            include_communities: true,
        }
    }
}

/// Summary detail level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetailLevel {
    /// One-liner overview.
    Brief,
    /// Key entities and relationships.
    Standard,
    /// Full structural analysis.
    Detailed,
}

// ─────────────────────────────────────────────
// Graph Input
// ─────────────────────────────────────────────

/// A triple (edge) in the knowledge graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl Triple {
    pub fn new(subject: &str, predicate: &str, object: &str) -> Self {
        Self {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
        }
    }
}

/// A subgraph to be summarised.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subgraph {
    /// Triples in the subgraph.
    pub triples: Vec<Triple>,
    /// Optional label for the subgraph.
    pub label: Option<String>,
}

impl Subgraph {
    pub fn new(triples: Vec<Triple>) -> Self {
        Self {
            triples,
            label: None,
        }
    }

    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    /// Get all unique entities (subjects and objects).
    pub fn entities(&self) -> HashSet<String> {
        let mut entities = HashSet::new();
        for t in &self.triples {
            entities.insert(t.subject.clone());
            entities.insert(t.object.clone());
        }
        entities
    }

    /// Get all unique predicates.
    pub fn predicates(&self) -> HashSet<String> {
        self.triples.iter().map(|t| t.predicate.clone()).collect()
    }

    /// Compute degree for each entity.
    pub fn entity_degrees(&self) -> HashMap<String, usize> {
        let mut degrees: HashMap<String, usize> = HashMap::new();
        for t in &self.triples {
            *degrees.entry(t.subject.clone()).or_insert(0) += 1;
            *degrees.entry(t.object.clone()).or_insert(0) += 1;
        }
        degrees
    }
}

// ─────────────────────────────────────────────
// Summary Output
// ─────────────────────────────────────────────

/// Summary of a knowledge graph subgraph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSummary {
    /// Human-readable text summary.
    pub text: String,
    /// Total entities in the subgraph.
    pub entity_count: usize,
    /// Total triples in the subgraph.
    pub triple_count: usize,
    /// Unique relationship types.
    pub relationship_types: usize,
    /// Hub entities (highest degree).
    pub hub_entities: Vec<EntitySummary>,
    /// Relationship type distribution.
    pub relationship_distribution: HashMap<String, usize>,
    /// Structural statistics.
    pub structural_stats: Option<StructuralStats>,
    /// Detected communities.
    pub communities: Vec<CommunitySummary>,
    /// Detail level used.
    pub detail_level: DetailLevel,
}

/// Summary of a single important entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySummary {
    /// Entity URI.
    pub uri: String,
    /// Short name.
    pub label: String,
    /// Degree (number of triples involving this entity).
    pub degree: usize,
    /// Incoming relationship types.
    pub incoming_types: Vec<String>,
    /// Outgoing relationship types.
    pub outgoing_types: Vec<String>,
}

/// Structural statistics of the subgraph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralStats {
    /// Average entity degree.
    pub avg_degree: f64,
    /// Maximum entity degree.
    pub max_degree: usize,
    /// Graph density (edges / possible edges).
    pub density: f64,
    /// Number of connected components (approximate).
    pub connected_components: usize,
}

/// Summary of a detected community.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunitySummary {
    /// Community identifier.
    pub id: usize,
    /// Number of entities.
    pub size: usize,
    /// Hub entity of the community.
    pub hub: String,
    /// Key relationship types in the community.
    pub key_relationships: Vec<String>,
}

// ─────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────

/// Statistics for the summarizer.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SummarizerStats {
    pub graphs_summarized: u64,
    pub total_triples_processed: u64,
    pub total_entities_processed: u64,
    pub avg_compression_ratio: f64,
}

// ─────────────────────────────────────────────
// Graph Summarizer
// ─────────────────────────────────────────────

/// Summarizes knowledge graph subgraphs for RAG applications.
pub struct GraphSummarizer {
    config: GraphSummaryConfig,
    stats: SummarizerStats,
}

impl GraphSummarizer {
    /// Create a new summarizer.
    pub fn new(config: GraphSummaryConfig) -> Self {
        Self {
            config,
            stats: SummarizerStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(GraphSummaryConfig::default())
    }

    /// Summarize a subgraph.
    pub fn summarize(&mut self, subgraph: &Subgraph) -> GraphSummary {
        let entities = subgraph.entities();
        let predicates = subgraph.predicates();
        let degrees = subgraph.entity_degrees();

        // Find hub entities (top-N by degree)
        let mut sorted_entities: Vec<_> = degrees.iter().collect();
        sorted_entities.sort_by(|a, b| b.1.cmp(a.1));

        let hub_entities: Vec<EntitySummary> = sorted_entities
            .iter()
            .filter(|(_, &deg)| deg >= self.config.min_entity_degree)
            .take(self.config.max_entities)
            .map(|(uri, &degree)| {
                let outgoing: Vec<String> = subgraph
                    .triples
                    .iter()
                    .filter(|t| &t.subject == *uri)
                    .map(|t| shorten_uri(&t.predicate))
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect();

                let incoming: Vec<String> = subgraph
                    .triples
                    .iter()
                    .filter(|t| &t.object == *uri)
                    .map(|t| shorten_uri(&t.predicate))
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect();

                EntitySummary {
                    uri: uri.to_string(),
                    label: shorten_uri(uri),
                    degree,
                    incoming_types: incoming,
                    outgoing_types: outgoing,
                }
            })
            .collect();

        // Relationship distribution
        let mut rel_dist: HashMap<String, usize> = HashMap::new();
        for t in &subgraph.triples {
            *rel_dist.entry(shorten_uri(&t.predicate)).or_insert(0) += 1;
        }

        // Structural stats
        let structural_stats = if self.config.include_stats {
            let total_degree: usize = degrees.values().sum();
            let n = entities.len().max(1);
            let avg_degree = total_degree as f64 / n as f64;
            let max_degree = degrees.values().copied().max().unwrap_or(0);
            let possible_edges = n * (n.saturating_sub(1));
            let density = if possible_edges > 0 {
                subgraph.triples.len() as f64 / possible_edges as f64
            } else {
                0.0
            };

            Some(StructuralStats {
                avg_degree,
                max_degree,
                density,
                connected_components: self.estimate_components(subgraph),
            })
        } else {
            None
        };

        // Communities (simplified: group by predicate type)
        let communities = if self.config.include_communities {
            self.detect_communities(subgraph, &degrees)
        } else {
            Vec::new()
        };

        // Generate text summary
        let text = self.generate_text(subgraph, &hub_entities, &rel_dist, &structural_stats);

        self.stats.graphs_summarized += 1;
        self.stats.total_triples_processed += subgraph.triples.len() as u64;
        self.stats.total_entities_processed += entities.len() as u64;

        GraphSummary {
            text,
            entity_count: entities.len(),
            triple_count: subgraph.triples.len(),
            relationship_types: predicates.len(),
            hub_entities,
            relationship_distribution: rel_dist,
            structural_stats,
            communities,
            detail_level: self.config.detail_level,
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &SummarizerStats {
        &self.stats
    }

    /// Get configuration.
    pub fn config(&self) -> &GraphSummaryConfig {
        &self.config
    }

    // ─── Internal ────────────────────────────

    fn generate_text(
        &self,
        subgraph: &Subgraph,
        hubs: &[EntitySummary],
        rel_dist: &HashMap<String, usize>,
        stats: &Option<StructuralStats>,
    ) -> String {
        let mut parts = Vec::new();

        if let Some(ref label) = subgraph.label {
            parts.push(format!("Subgraph: {label}"));
        }

        parts.push(format!(
            "Contains {} entities and {} triples with {} relationship types.",
            subgraph.entities().len(),
            subgraph.triples.len(),
            subgraph.predicates().len()
        ));

        if self.config.detail_level != DetailLevel::Brief {
            if !hubs.is_empty() {
                let hub_names: Vec<String> = hubs.iter().take(5).map(|h| h.label.clone()).collect();
                parts.push(format!("Key entities: {}.", hub_names.join(", ")));
            }

            let mut rels: Vec<_> = rel_dist.iter().collect();
            rels.sort_by(|a, b| b.1.cmp(a.1));
            let top_rels: Vec<String> = rels
                .iter()
                .take(5)
                .map(|(r, c)| format!("{r} ({c})"))
                .collect();
            if !top_rels.is_empty() {
                parts.push(format!("Top relationships: {}.", top_rels.join(", ")));
            }
        }

        if self.config.detail_level == DetailLevel::Detailed {
            if let Some(ref s) = stats {
                parts.push(format!(
                    "Structure: avg degree {:.1}, max degree {}, density {:.4}, {} component(s).",
                    s.avg_degree, s.max_degree, s.density, s.connected_components
                ));
            }
        }

        parts.join(" ")
    }

    fn estimate_components(&self, subgraph: &Subgraph) -> usize {
        let entities = subgraph.entities();
        if entities.is_empty() {
            return 0;
        }

        let mut parent: HashMap<String, String> = HashMap::new();
        for e in &entities {
            parent.insert(e.clone(), e.clone());
        }

        for t in &subgraph.triples {
            let root_s = find_root(&parent, &t.subject);
            let root_o = find_root(&parent, &t.object);
            if root_s != root_o {
                parent.insert(root_s, root_o);
            }
        }

        let roots: HashSet<String> = entities.iter().map(|e| find_root(&parent, e)).collect();
        roots.len()
    }

    fn detect_communities(
        &self,
        subgraph: &Subgraph,
        degrees: &HashMap<String, usize>,
    ) -> Vec<CommunitySummary> {
        // Simplified: use predicate-based grouping
        let mut pred_groups: HashMap<String, HashSet<String>> = HashMap::new();
        for t in &subgraph.triples {
            let group = pred_groups.entry(shorten_uri(&t.predicate)).or_default();
            group.insert(t.subject.clone());
            group.insert(t.object.clone());
        }

        pred_groups
            .iter()
            .enumerate()
            .take(5)
            .map(|(id, (pred, members))| {
                let hub = members
                    .iter()
                    .max_by_key(|m| degrees.get(*m).unwrap_or(&0))
                    .cloned()
                    .unwrap_or_default();
                CommunitySummary {
                    id,
                    size: members.len(),
                    hub: shorten_uri(&hub),
                    key_relationships: vec![pred.clone()],
                }
            })
            .collect()
    }
}

fn find_root(parent: &HashMap<String, String>, node: &str) -> String {
    let mut current = node.to_string();
    while let Some(p) = parent.get(&current) {
        if p == &current {
            break;
        }
        current = p.clone();
    }
    current
}

fn shorten_uri(uri: &str) -> String {
    // Check for '#' first — in RDF URIs the fragment identifier always
    // follows the path, so `rfind('#')` gives the true local-name boundary
    // even when '/' appears earlier in the scheme or authority.
    if let Some(idx) = uri.rfind('#') {
        uri[idx + 1..].to_string()
    } else if let Some(idx) = uri.rfind('/') {
        uri[idx + 1..].to_string()
    } else {
        uri.to_string()
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_subgraph() -> Subgraph {
        Subgraph::new(vec![
            Triple::new(
                "http://ex.org/Alice",
                "http://ex.org/knows",
                "http://ex.org/Bob",
            ),
            Triple::new(
                "http://ex.org/Alice",
                "http://ex.org/likes",
                "http://ex.org/Charlie",
            ),
            Triple::new(
                "http://ex.org/Bob",
                "http://ex.org/knows",
                "http://ex.org/Charlie",
            ),
            Triple::new(
                "http://ex.org/Charlie",
                "http://ex.org/worksAt",
                "http://ex.org/ACME",
            ),
            Triple::new(
                "http://ex.org/Alice",
                "http://ex.org/worksAt",
                "http://ex.org/ACME",
            ),
        ])
    }

    #[test]
    fn test_default_config() {
        let config = GraphSummaryConfig::default();
        assert_eq!(config.max_entities, 20);
        assert_eq!(config.detail_level, DetailLevel::Standard);
    }

    #[test]
    fn test_basic_summarize() {
        let mut summarizer = GraphSummarizer::with_defaults();
        let summary = summarizer.summarize(&sample_subgraph());
        assert_eq!(summary.triple_count, 5);
        assert!(summary.entity_count > 0);
        assert!(!summary.text.is_empty());
    }

    #[test]
    fn test_hub_entities() {
        let mut summarizer = GraphSummarizer::with_defaults();
        let summary = summarizer.summarize(&sample_subgraph());
        assert!(!summary.hub_entities.is_empty());
        // Alice should be a hub (degree 3)
        assert!(summary.hub_entities.iter().any(|e| e.label == "Alice"));
    }

    #[test]
    fn test_relationship_distribution() {
        let mut summarizer = GraphSummarizer::with_defaults();
        let summary = summarizer.summarize(&sample_subgraph());
        assert!(summary.relationship_distribution.contains_key("knows"));
        assert_eq!(summary.relationship_distribution["knows"], 2);
    }

    #[test]
    fn test_structural_stats() {
        let mut summarizer = GraphSummarizer::with_defaults();
        let summary = summarizer.summarize(&sample_subgraph());
        let stats = summary.structural_stats.expect("should have stats");
        assert!(stats.avg_degree > 0.0);
        assert!(stats.max_degree > 0);
        assert!(stats.density > 0.0);
    }

    #[test]
    fn test_communities() {
        let mut summarizer = GraphSummarizer::with_defaults();
        let summary = summarizer.summarize(&sample_subgraph());
        assert!(!summary.communities.is_empty());
    }

    #[test]
    fn test_brief_summary() {
        let mut summarizer = GraphSummarizer::new(GraphSummaryConfig {
            detail_level: DetailLevel::Brief,
            ..Default::default()
        });
        let summary = summarizer.summarize(&sample_subgraph());
        assert!(!summary.text.is_empty());
        // Brief should be shorter
        assert!(!summary.text.contains("Key entities"));
    }

    #[test]
    fn test_detailed_summary() {
        let mut summarizer = GraphSummarizer::new(GraphSummaryConfig {
            detail_level: DetailLevel::Detailed,
            ..Default::default()
        });
        let summary = summarizer.summarize(&sample_subgraph());
        assert!(summary.text.contains("Structure"));
    }

    #[test]
    fn test_empty_subgraph() {
        let mut summarizer = GraphSummarizer::with_defaults();
        let summary = summarizer.summarize(&Subgraph::new(vec![]));
        assert_eq!(summary.triple_count, 0);
        assert_eq!(summary.entity_count, 0);
    }

    #[test]
    fn test_single_triple() {
        let mut summarizer = GraphSummarizer::with_defaults();
        let sg = Subgraph::new(vec![Triple::new("A", "knows", "B")]);
        let summary = summarizer.summarize(&sg);
        assert_eq!(summary.triple_count, 1);
        assert_eq!(summary.entity_count, 2);
    }

    #[test]
    fn test_subgraph_with_label() {
        let mut summarizer = GraphSummarizer::with_defaults();
        let sg = sample_subgraph().with_label("Social Network");
        let summary = summarizer.summarize(&sg);
        assert!(summary.text.contains("Social Network"));
    }

    #[test]
    fn test_entities_extraction() {
        let sg = sample_subgraph();
        let entities = sg.entities();
        assert_eq!(entities.len(), 4);
    }

    #[test]
    fn test_predicates_extraction() {
        let sg = sample_subgraph();
        let preds = sg.predicates();
        assert_eq!(preds.len(), 3);
    }

    #[test]
    fn test_entity_degrees() {
        let sg = sample_subgraph();
        let degrees = sg.entity_degrees();
        assert!(degrees["http://ex.org/Alice"] >= 3);
    }

    #[test]
    fn test_connected_components() {
        let mut summarizer = GraphSummarizer::with_defaults();
        let sg = sample_subgraph();
        let summary = summarizer.summarize(&sg);
        let stats = summary.structural_stats.expect("should have stats");
        assert_eq!(stats.connected_components, 1); // All connected
    }

    #[test]
    fn test_disconnected_components() {
        let mut summarizer = GraphSummarizer::with_defaults();
        let sg = Subgraph::new(vec![
            Triple::new("A", "knows", "B"),
            Triple::new("C", "knows", "D"), // Disconnected from A-B
        ]);
        let summary = summarizer.summarize(&sg);
        let stats = summary.structural_stats.expect("should have stats");
        assert_eq!(stats.connected_components, 2);
    }

    #[test]
    fn test_min_entity_degree_filter() {
        let mut summarizer = GraphSummarizer::new(GraphSummaryConfig {
            min_entity_degree: 3,
            ..Default::default()
        });
        let summary = summarizer.summarize(&sample_subgraph());
        // Only entities with degree >= 3 should be hubs
        for hub in &summary.hub_entities {
            assert!(hub.degree >= 3);
        }
    }

    #[test]
    fn test_no_stats() {
        let mut summarizer = GraphSummarizer::new(GraphSummaryConfig {
            include_stats: false,
            ..Default::default()
        });
        let summary = summarizer.summarize(&sample_subgraph());
        assert!(summary.structural_stats.is_none());
    }

    #[test]
    fn test_no_communities() {
        let mut summarizer = GraphSummarizer::new(GraphSummaryConfig {
            include_communities: false,
            ..Default::default()
        });
        let summary = summarizer.summarize(&sample_subgraph());
        assert!(summary.communities.is_empty());
    }

    #[test]
    fn test_shorten_uri_slash() {
        assert_eq!(shorten_uri("http://ex.org/Alice"), "Alice");
    }

    #[test]
    fn test_shorten_uri_hash() {
        assert_eq!(shorten_uri("http://ex.org#name"), "name");
    }

    #[test]
    fn test_stats_tracking() {
        let mut summarizer = GraphSummarizer::with_defaults();
        summarizer.summarize(&sample_subgraph());
        assert_eq!(summarizer.stats().graphs_summarized, 1);
        assert_eq!(summarizer.stats().total_triples_processed, 5);
    }

    #[test]
    fn test_config_serialization() {
        let config = GraphSummaryConfig::default();
        let json = serde_json::to_string(&config).expect("serialize failed");
        assert!(json.contains("max_entities"));
    }

    #[test]
    fn test_summary_serialization() {
        let mut summarizer = GraphSummarizer::with_defaults();
        let summary = summarizer.summarize(&sample_subgraph());
        let json = serde_json::to_string(&summary).expect("serialize failed");
        assert!(json.contains("text"));
    }

    #[test]
    fn test_triple_equality() {
        let t1 = Triple::new("A", "knows", "B");
        let t2 = Triple::new("A", "knows", "B");
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_entity_summary_outgoing() {
        let mut summarizer = GraphSummarizer::with_defaults();
        let summary = summarizer.summarize(&sample_subgraph());
        let alice = summary
            .hub_entities
            .iter()
            .find(|e| e.label == "Alice")
            .expect("Alice should be a hub");
        assert!(!alice.outgoing_types.is_empty());
    }

    #[test]
    fn test_large_subgraph() {
        let mut triples = Vec::new();
        for i in 0..100 {
            triples.push(Triple::new(
                &format!("http://ex.org/e{i}"),
                "http://ex.org/rel",
                &format!("http://ex.org/e{}", (i + 1) % 100),
            ));
        }
        let mut summarizer = GraphSummarizer::with_defaults();
        let summary = summarizer.summarize(&Subgraph::new(triples));
        assert_eq!(summary.triple_count, 100);
        assert!(summary.hub_entities.len() <= 20);
    }

    #[test]
    fn test_relationship_types_count() {
        let mut summarizer = GraphSummarizer::with_defaults();
        let summary = summarizer.summarize(&sample_subgraph());
        assert_eq!(summary.relationship_types, 3);
    }

    #[test]
    fn test_density_calculation() {
        let mut summarizer = GraphSummarizer::with_defaults();
        let sg = Subgraph::new(vec![Triple::new("A", "r", "B"), Triple::new("B", "r", "A")]);
        let summary = summarizer.summarize(&sg);
        let stats = summary.structural_stats.expect("should have stats");
        assert!(stats.density > 0.0 && stats.density <= 1.0);
    }

    #[test]
    fn test_community_hub() {
        let mut summarizer = GraphSummarizer::with_defaults();
        let summary = summarizer.summarize(&sample_subgraph());
        for community in &summary.communities {
            assert!(!community.hub.is_empty());
        }
    }
}
