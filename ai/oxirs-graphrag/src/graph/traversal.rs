//! Graph traversal for subgraph expansion

use crate::{GraphRAGResult, ScoredEntity, Triple};
use std::collections::{HashMap, HashSet};

/// Graph traversal configuration
#[derive(Debug, Clone)]
pub struct TraversalConfig {
    /// Maximum hops from seed entities
    pub max_hops: usize,
    /// Maximum edges per node
    pub max_edges_per_node: usize,
    /// Maximum total triples
    pub max_triples: usize,
    /// Predicates to follow (empty = all)
    pub follow_predicates: HashSet<String>,
    /// Predicates to exclude
    pub exclude_predicates: HashSet<String>,
    /// Whether to traverse inverse edges
    pub traverse_inverse: bool,
}

impl Default for TraversalConfig {
    fn default() -> Self {
        Self {
            max_hops: 2,
            max_edges_per_node: 50,
            max_triples: 500,
            follow_predicates: HashSet::new(),
            exclude_predicates: HashSet::new(),
            traverse_inverse: true,
        }
    }
}

/// Graph traversal engine
pub struct GraphTraversal {
    config: TraversalConfig,
}

impl Default for GraphTraversal {
    fn default() -> Self {
        Self::new(TraversalConfig::default())
    }
}

impl GraphTraversal {
    pub fn new(config: TraversalConfig) -> Self {
        Self { config }
    }

    /// Generate SPARQL query for N-hop expansion
    pub fn generate_expansion_query(&self, seeds: &[ScoredEntity]) -> String {
        if seeds.is_empty() {
            return String::new();
        }

        let seed_uris: Vec<String> = seeds.iter().map(|s| format!("<{}>", s.uri)).collect();
        let values = seed_uris.join(" ");

        // Build predicate filter
        let predicate_filter = if !self.config.exclude_predicates.is_empty() {
            let excluded: Vec<String> = self
                .config
                .exclude_predicates
                .iter()
                .map(|p| format!("<{}>", p))
                .collect();
            format!("FILTER(?p NOT IN ({}))", excluded.join(", "))
        } else {
            String::new()
        };

        // Build path pattern based on hops
        let path_pattern = match self.config.max_hops {
            1 => "?seed ?p ?o".to_string(),
            2 => "?seed ?p1 ?mid . ?mid ?p2 ?o".to_string(),
            n => format!("?seed (:|!:){{1,{}}} ?o", n),
        };

        // Build CONSTRUCT query
        format!(
            r#"
CONSTRUCT {{
    ?seed ?p ?o .
    ?s ?p2 ?seed .
}}
WHERE {{
    VALUES ?seed {{ {} }}
    {{
        ?seed ?p ?o .
        {}
    }}
    {}
}}
LIMIT {}
"#,
            values,
            predicate_filter,
            if self.config.traverse_inverse {
                "UNION { ?s ?p2 ?seed . }"
            } else {
                ""
            },
            self.config.max_triples
        )
    }

    /// Expand subgraph from triples (in-memory traversal)
    pub fn expand_local(
        &self,
        seeds: &[ScoredEntity],
        all_triples: &[Triple],
    ) -> GraphRAGResult<Vec<Triple>> {
        let seed_uris: HashSet<String> = seeds.iter().map(|s| s.uri.clone()).collect();

        // Build adjacency index
        let mut subject_index: HashMap<String, Vec<&Triple>> = HashMap::new();
        let mut object_index: HashMap<String, Vec<&Triple>> = HashMap::new();

        for triple in all_triples {
            subject_index
                .entry(triple.subject.clone())
                .or_default()
                .push(triple);
            object_index
                .entry(triple.object.clone())
                .or_default()
                .push(triple);
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut result: Vec<Triple> = Vec::new();
        let mut frontier: Vec<String> = seed_uris.iter().cloned().collect();

        for hop in 0..self.config.max_hops {
            if frontier.is_empty() || result.len() >= self.config.max_triples {
                break;
            }

            let mut next_frontier: Vec<String> = Vec::new();

            for node in &frontier {
                if visited.contains(node) {
                    continue;
                }
                visited.insert(node.clone());

                // Get outgoing edges
                if let Some(triples) = subject_index.get(node) {
                    for triple in triples.iter().take(self.config.max_edges_per_node) {
                        if self.should_follow_predicate(&triple.predicate) {
                            result.push((*triple).clone());
                            if hop < self.config.max_hops - 1 && !visited.contains(&triple.object) {
                                next_frontier.push(triple.object.clone());
                            }
                        }
                    }
                }

                // Get incoming edges (if enabled)
                if self.config.traverse_inverse {
                    if let Some(triples) = object_index.get(node) {
                        for triple in triples.iter().take(self.config.max_edges_per_node) {
                            if self.should_follow_predicate(&triple.predicate) {
                                result.push((*triple).clone());
                                if hop < self.config.max_hops - 1
                                    && !visited.contains(&triple.subject)
                                {
                                    next_frontier.push(triple.subject.clone());
                                }
                            }
                        }
                    }
                }

                if result.len() >= self.config.max_triples {
                    break;
                }
            }

            frontier = next_frontier;
        }

        // Deduplicate
        let mut seen: HashSet<(String, String, String)> = HashSet::new();
        let deduped: Vec<Triple> = result
            .into_iter()
            .filter(|t| {
                let key = (t.subject.clone(), t.predicate.clone(), t.object.clone());
                if seen.contains(&key) {
                    false
                } else {
                    seen.insert(key);
                    true
                }
            })
            .take(self.config.max_triples)
            .collect();

        Ok(deduped)
    }

    /// Check if predicate should be followed
    fn should_follow_predicate(&self, predicate: &str) -> bool {
        // If follow list is specified, predicate must be in it
        if !self.config.follow_predicates.is_empty() {
            return self.config.follow_predicates.contains(predicate);
        }

        // Otherwise, check exclude list
        !self.config.exclude_predicates.contains(predicate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_expansion_query_generation() {
        let config = TraversalConfig {
            max_hops: 2,
            ..Default::default()
        };
        let traversal = GraphTraversal::new(config);

        let seeds = vec![ScoredEntity {
            uri: "http://example.org/entity1".to_string(),
            score: 0.9,
            source: crate::ScoreSource::Vector,
            metadata: HashMap::new(),
        }];

        let query = traversal.generate_expansion_query(&seeds);
        assert!(query.contains("http://example.org/entity1"));
        assert!(query.contains("CONSTRUCT"));
    }

    #[test]
    fn test_local_expansion() {
        let traversal = GraphTraversal::default();

        let seeds = vec![ScoredEntity {
            uri: "http://a".to_string(),
            score: 0.9,
            source: crate::ScoreSource::Vector,
            metadata: HashMap::new(),
        }];

        let triples = vec![
            Triple::new("http://a", "http://rel", "http://b"),
            Triple::new("http://b", "http://rel", "http://c"),
            Triple::new("http://x", "http://rel", "http://y"),
        ];

        let result = traversal.expand_local(&seeds, &triples).unwrap();

        // Should include a->b and b->c (2 hops from a)
        assert!(result.iter().any(|t| t.subject == "http://a"));
        assert!(result.iter().any(|t| t.subject == "http://b"));
        // Should not include x->y (unconnected)
        assert!(!result.iter().any(|t| t.subject == "http://x"));
    }
}
