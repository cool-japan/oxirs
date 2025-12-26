//! Subgraph extraction for context building

use crate::{GraphRAGResult, ScoredEntity, Triple};
use std::collections::HashSet;

/// Subgraph extraction configuration
#[derive(Debug, Clone)]
pub struct SubgraphConfig {
    /// Maximum number of triples to include
    pub max_triples: usize,
    /// Include all edges between selected nodes
    pub include_internal_edges: bool,
    /// Include edges to/from external nodes
    pub include_external_edges: bool,
    /// Prioritize triples with higher-scored entities
    pub score_weighted: bool,
}

impl Default for SubgraphConfig {
    fn default() -> Self {
        Self {
            max_triples: 100,
            include_internal_edges: true,
            include_external_edges: true,
            score_weighted: true,
        }
    }
}

/// Subgraph extractor
pub struct SubgraphExtractor {
    config: SubgraphConfig,
}

impl Default for SubgraphExtractor {
    fn default() -> Self {
        Self::new(SubgraphConfig::default())
    }
}

impl SubgraphExtractor {
    pub fn new(config: SubgraphConfig) -> Self {
        Self { config }
    }

    /// Extract relevant subgraph for LLM context
    pub fn extract(
        &self,
        seeds: &[ScoredEntity],
        expanded_triples: &[Triple],
    ) -> GraphRAGResult<Vec<Triple>> {
        let seed_uris: HashSet<String> = seeds.iter().map(|s| s.uri.clone()).collect();

        // Score triples based on relevance to seeds
        let mut scored_triples: Vec<(f64, &Triple)> = expanded_triples
            .iter()
            .map(|triple| {
                let score = self.score_triple(triple, seeds, &seed_uris);
                (score, triple)
            })
            .filter(|(score, _)| *score > 0.0)
            .collect();

        // Sort by score (descending)
        scored_triples.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top triples
        let result: Vec<Triple> = scored_triples
            .into_iter()
            .take(self.config.max_triples)
            .map(|(_, t)| t.clone())
            .collect();

        Ok(result)
    }

    /// Score a triple based on relevance to seeds
    fn score_triple(
        &self,
        triple: &Triple,
        seeds: &[ScoredEntity],
        seed_uris: &HashSet<String>,
    ) -> f64 {
        let subject_is_seed = seed_uris.contains(&triple.subject);
        let object_is_seed = seed_uris.contains(&triple.object);

        // Internal edges (both endpoints are seeds)
        if subject_is_seed && object_is_seed {
            if !self.config.include_internal_edges {
                return 0.0;
            }

            if self.config.score_weighted {
                // Average score of both seed entities
                let subj_score = seeds
                    .iter()
                    .find(|s| s.uri == triple.subject)
                    .map(|s| s.score)
                    .unwrap_or(0.5);
                let obj_score = seeds
                    .iter()
                    .find(|s| s.uri == triple.object)
                    .map(|s| s.score)
                    .unwrap_or(0.5);
                return (subj_score + obj_score) / 2.0 * 1.5; // Boost internal edges
            }
            return 1.5;
        }

        // External edges (one endpoint is seed)
        if subject_is_seed || object_is_seed {
            if !self.config.include_external_edges {
                return 0.0;
            }

            if self.config.score_weighted {
                let seed_uri = if subject_is_seed {
                    &triple.subject
                } else {
                    &triple.object
                };
                return seeds
                    .iter()
                    .find(|s| &s.uri == seed_uri)
                    .map(|s| s.score)
                    .unwrap_or(0.5);
            }
            return 1.0;
        }

        // Neither endpoint is seed (context edges)
        0.1
    }

    /// Extract minimal subgraph connecting seeds
    pub fn extract_steiner(
        &self,
        seeds: &[ScoredEntity],
        all_triples: &[Triple],
    ) -> GraphRAGResult<Vec<Triple>> {
        // Build adjacency for path finding
        use std::collections::HashMap;

        let mut adjacency: HashMap<String, Vec<(String, Triple)>> = HashMap::new();
        for triple in all_triples {
            adjacency
                .entry(triple.subject.clone())
                .or_default()
                .push((triple.object.clone(), triple.clone()));
            adjacency
                .entry(triple.object.clone())
                .or_default()
                .push((triple.subject.clone(), triple.clone()));
        }

        let seed_uris: Vec<String> = seeds.iter().map(|s| s.uri.clone()).collect();
        let mut result_triples: HashSet<Triple> = HashSet::new();

        // Find shortest paths between all pairs of seeds
        for i in 0..seed_uris.len() {
            for j in (i + 1)..seed_uris.len() {
                if let Some(path) = self.bfs_path(&seed_uris[i], &seed_uris[j], &adjacency) {
                    for triple in path {
                        result_triples.insert(triple);
                    }
                }
            }
        }

        Ok(result_triples
            .into_iter()
            .take(self.config.max_triples)
            .collect())
    }

    /// BFS to find shortest path between two nodes
    fn bfs_path(
        &self,
        start: &str,
        end: &str,
        adjacency: &std::collections::HashMap<String, Vec<(String, Triple)>>,
    ) -> Option<Vec<Triple>> {
        use std::collections::VecDeque;

        if start == end {
            return Some(vec![]);
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, Vec<Triple>)> = VecDeque::new();

        queue.push_back((start.to_string(), vec![]));
        visited.insert(start.to_string());

        while let Some((current, path)) = queue.pop_front() {
            if let Some(neighbors) = adjacency.get(&current) {
                for (neighbor, triple) in neighbors {
                    if neighbor == end {
                        let mut result = path.clone();
                        result.push(triple.clone());
                        return Some(result);
                    }

                    if !visited.contains(neighbor) && path.len() < 5 {
                        // Limit path length
                        visited.insert(neighbor.clone());
                        let mut new_path = path.clone();
                        new_path.push(triple.clone());
                        queue.push_back((neighbor.clone(), new_path));
                    }
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_subgraph_extraction() {
        let extractor = SubgraphExtractor::default();

        let seeds = vec![
            ScoredEntity {
                uri: "http://a".to_string(),
                score: 0.9,
                source: crate::ScoreSource::Vector,
                metadata: HashMap::new(),
            },
            ScoredEntity {
                uri: "http://b".to_string(),
                score: 0.8,
                source: crate::ScoreSource::Vector,
                metadata: HashMap::new(),
            },
        ];

        let triples = vec![
            Triple::new("http://a", "http://rel", "http://b"),
            Triple::new("http://a", "http://rel", "http://c"),
            Triple::new("http://x", "http://rel", "http://y"),
        ];

        let result = extractor.extract(&seeds, &triples).unwrap();

        // Should prioritize a->b (internal) over a->c (external)
        assert!(!result.is_empty());
        assert!(result
            .iter()
            .any(|t| t.subject == "http://a" && t.object == "http://b"));
    }

    #[test]
    fn test_steiner_extraction() {
        let extractor = SubgraphExtractor::default();

        let seeds = vec![
            ScoredEntity {
                uri: "http://a".to_string(),
                score: 0.9,
                source: crate::ScoreSource::Vector,
                metadata: HashMap::new(),
            },
            ScoredEntity {
                uri: "http://c".to_string(),
                score: 0.8,
                source: crate::ScoreSource::Vector,
                metadata: HashMap::new(),
            },
        ];

        let triples = vec![
            Triple::new("http://a", "http://rel", "http://b"),
            Triple::new("http://b", "http://rel", "http://c"),
        ];

        let result = extractor.extract_steiner(&seeds, &triples).unwrap();

        // Should find path a->b->c
        assert_eq!(result.len(), 2);
    }
}
