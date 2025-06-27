//! Graph-aware vector search for named graph filtering and contextual search
//!
//! This module provides graph-scoped vector search capabilities that enable:
//! - Named graph filtering for vector searches  
//! - Contextual search within specific RDF graphs
//! - Hierarchical graph search patterns
//! - Cross-graph similarity analysis

use crate::{Vector, VectorStore, similarity::SimilarityMetric};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Graph-aware search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAwareConfig {
    /// Enable graph-scoped filtering
    pub enable_graph_filtering: bool,
    /// Enable hierarchical graph search
    pub enable_hierarchical_search: bool,
    /// Enable cross-graph similarity
    pub enable_cross_graph_similarity: bool,
    /// Default graph context if none specified
    pub default_graph: Option<String>,
    /// Graph hierarchy configuration
    pub graph_hierarchy: GraphHierarchy,
    /// Cache graph metadata
    pub cache_graph_metadata: bool,
}

impl Default for GraphAwareConfig {
    fn default() -> Self {
        Self {
            enable_graph_filtering: true,
            enable_hierarchical_search: false,
            enable_cross_graph_similarity: false,
            default_graph: None,
            graph_hierarchy: GraphHierarchy::default(),
            cache_graph_metadata: true,
        }
    }
}

/// Graph hierarchy for hierarchical search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphHierarchy {
    /// Parent-child relationships between graphs
    pub parent_child: HashMap<String, Vec<String>>,
    /// Graph categories/types
    pub graph_types: HashMap<String, String>,
    /// Graph priority weights for search ranking
    pub graph_weights: HashMap<String, f32>,
}

impl Default for GraphHierarchy {
    fn default() -> Self {
        Self {
            parent_child: HashMap::new(),
            graph_types: HashMap::new(),
            graph_weights: HashMap::new(),
        }
    }
}

/// Graph context for search operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphContext {
    /// Primary graph to search in
    pub primary_graph: String,
    /// Additional graphs to include (if hierarchical search enabled)
    pub additional_graphs: Vec<String>,
    /// Search scope configuration
    pub scope: GraphSearchScope,
    /// Context-specific weights
    pub context_weights: HashMap<String, f32>,
}

/// Graph search scope
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GraphSearchScope {
    /// Search only in the specified graph
    Exact,
    /// Search in specified graph and its children
    IncludeChildren,
    /// Search in specified graph and its parents
    IncludeParents,
    /// Search in entire hierarchy branch
    FullHierarchy,
    /// Search across all related graphs
    Related,
}

/// Graph-aware search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAwareSearchResult {
    /// Resource URI
    pub resource: String,
    /// Similarity score
    pub score: f32,
    /// Graph where the resource was found
    pub source_graph: String,
    /// Context relevance score
    pub context_score: f32,
    /// Combined final score
    pub final_score: f32,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Resource graph membership information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceGraphInfo {
    /// Resource URI
    pub resource: String,
    /// Graphs containing this resource
    pub graphs: HashSet<String>,
    /// Primary graph (most relevant)
    pub primary_graph: Option<String>,
    /// Last updated timestamp
    pub last_updated: std::time::SystemTime,
}

/// Graph-aware vector search engine
pub struct GraphAwareSearch {
    config: GraphAwareConfig,
    /// Resource to graph mappings
    resource_graph_map: HashMap<String, ResourceGraphInfo>,
    /// Graph metadata cache
    graph_metadata: HashMap<String, GraphMetadata>,
    /// Graph size cache for performance optimization
    graph_sizes: HashMap<String, usize>,
}

/// Graph metadata for optimization and ranking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetadata {
    /// Graph URI
    pub graph_uri: String,
    /// Number of resources in this graph
    pub resource_count: usize,
    /// Average vector similarity within graph
    pub avg_internal_similarity: f32,
    /// Graph creation/update time
    pub last_modified: std::time::SystemTime,
    /// Graph type/category
    pub graph_type: Option<String>,
    /// Graph quality score
    pub quality_score: f32,
}

impl GraphAwareSearch {
    pub fn new(config: GraphAwareConfig) -> Self {
        Self {
            config,
            resource_graph_map: HashMap::new(),
            graph_metadata: HashMap::new(),
            graph_sizes: HashMap::new(),
        }
    }

    /// Register a resource as belonging to specific graphs
    pub fn register_resource_graph(&mut self, resource: String, graphs: Vec<String>) {
        let graph_set: HashSet<String> = graphs.iter().cloned().collect();
        let primary_graph = graphs.first().cloned();

        let info = ResourceGraphInfo {
            resource: resource.clone(),
            graphs: graph_set,
            primary_graph,
            last_updated: std::time::SystemTime::now(),
        };

        self.resource_graph_map.insert(resource, info);

        // Update graph sizes
        for graph in graphs {
            *self.graph_sizes.entry(graph).or_insert(0) += 1;
        }
    }

    /// Perform graph-aware vector search
    pub fn search_in_graph(
        &self,
        vector_store: &VectorStore,
        query_text: &str,
        graph_context: &GraphContext,
        limit: usize,
    ) -> Result<Vec<GraphAwareSearchResult>> {
        // Determine which graphs to search in
        let target_graphs = self.resolve_search_graphs(graph_context)?;

        // Perform vector search across all target graphs
        let mut all_results = Vec::new();

        for graph_uri in &target_graphs {
            let graph_results = self.search_single_graph(vector_store, query_text, graph_uri, limit * 2)?;
            all_results.extend(graph_results);
        }

        // Apply graph-aware ranking and filtering
        let ranked_results = self.rank_results_by_graph_context(all_results, graph_context)?;

        // Return top results up to limit
        Ok(ranked_results.into_iter().take(limit).collect())
    }

    /// Search within a specific named graph
    pub fn search_single_graph(
        &self,
        vector_store: &VectorStore,
        query_text: &str,
        graph_uri: &str,
        limit: usize,
    ) -> Result<Vec<GraphAwareSearchResult>> {
        // Get all potential results from vector store
        let vector_results = vector_store.similarity_search(query_text, limit * 3)?; // Get more candidates

        let mut graph_results = Vec::new();

        for (resource, score) in vector_results {
            // Check if resource belongs to the target graph
            if let Some(resource_info) = self.resource_graph_map.get(&resource) {
                if resource_info.graphs.contains(graph_uri) {
                    let context_score = self.calculate_context_score(&resource, graph_uri)?;
                    let final_score = self.combine_scores(score, context_score, graph_uri);

                    graph_results.push(GraphAwareSearchResult {
                        resource,
                        score,
                        source_graph: graph_uri.to_string(),
                        context_score,
                        final_score,
                        metadata: HashMap::new(),
                    });
                }
            }
        }

        // Sort by final score
        graph_results.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap());

        Ok(graph_results.into_iter().take(limit).collect())
    }

    /// Resolve which graphs to search based on context and hierarchy
    fn resolve_search_graphs(&self, context: &GraphContext) -> Result<Vec<String>> {
        let mut target_graphs = vec![context.primary_graph.clone()];

        match context.scope {
            GraphSearchScope::Exact => {
                // Only search in the primary graph
            }
            GraphSearchScope::IncludeChildren => {
                // Add child graphs if hierarchy is configured
                if let Some(children) = self.config.graph_hierarchy.parent_child.get(&context.primary_graph) {
                    target_graphs.extend(children.clone());
                }
            }
            GraphSearchScope::IncludeParents => {
                // Find parent graphs
                for (parent, children) in &self.config.graph_hierarchy.parent_child {
                    if children.contains(&context.primary_graph) {
                        target_graphs.push(parent.clone());
                    }
                }
            }
            GraphSearchScope::FullHierarchy => {
                // Include all graphs in the hierarchy branch
                target_graphs.extend(self.get_hierarchy_branch(&context.primary_graph));
            }
            GraphSearchScope::Related => {
                // Include additional graphs specified in context
                target_graphs.extend(context.additional_graphs.clone());
            }
        }

        // Add additional graphs from context
        target_graphs.extend(context.additional_graphs.clone());

        // Remove duplicates and return
        target_graphs.sort();
        target_graphs.dedup();
        Ok(target_graphs)
    }

    /// Get all graphs in a hierarchy branch
    fn get_hierarchy_branch(&self, graph_uri: &str) -> Vec<String> {
        let mut branch_graphs = Vec::new();

        // Add children recursively
        self.add_children_recursive(graph_uri, &mut branch_graphs);

        // Add parents recursively  
        self.add_parents_recursive(graph_uri, &mut branch_graphs);

        branch_graphs
    }

    /// Recursively add child graphs
    fn add_children_recursive(&self, graph_uri: &str, result: &mut Vec<String>) {
        if let Some(children) = self.config.graph_hierarchy.parent_child.get(graph_uri) {
            for child in children {
                if !result.contains(child) {
                    result.push(child.clone());
                    self.add_children_recursive(child, result);
                }
            }
        }
    }

    /// Recursively add parent graphs
    fn add_parents_recursive(&self, graph_uri: &str, result: &mut Vec<String>) {
        for (parent, children) in &self.config.graph_hierarchy.parent_child {
            if children.contains(&graph_uri.to_string()) && !result.contains(parent) {
                result.push(parent.clone());
                self.add_parents_recursive(parent, result);
            }
        }
    }

    /// Calculate context relevance score for a resource in a graph
    fn calculate_context_score(&self, resource: &str, graph_uri: &str) -> Result<f32> {
        let mut context_score = 1.0;

        // Apply graph weight if configured
        if let Some(&weight) = self.config.graph_hierarchy.graph_weights.get(graph_uri) {
            context_score *= weight;
        }

        // Consider graph metadata quality
        if let Some(metadata) = self.graph_metadata.get(graph_uri) {
            context_score *= metadata.quality_score;
        }

        // Check if this is the primary graph for the resource
        if let Some(resource_info) = self.resource_graph_map.get(resource) {
            if resource_info.primary_graph.as_ref() == Some(&graph_uri.to_string()) {
                context_score *= 1.2; // Boost for primary graph
            }
        }

        Ok(context_score.min(1.0)) // Cap at 1.0
    }

    /// Combine vector similarity score with context score
    fn combine_scores(&self, similarity_score: f32, context_score: f32, graph_uri: &str) -> f32 {
        // Weighted combination of similarity and context scores
        let similarity_weight = 0.7;
        let context_weight = 0.3;

        // Apply graph-specific boosting
        let graph_boost = self.config.graph_hierarchy.graph_weights
            .get(graph_uri)
            .unwrap_or(&1.0);

        (similarity_score * similarity_weight + context_score * context_weight) * graph_boost
    }

    /// Rank results considering graph context and hierarchy
    fn rank_results_by_graph_context(
        &self,
        mut results: Vec<GraphAwareSearchResult>,
        context: &GraphContext,
    ) -> Result<Vec<GraphAwareSearchResult>> {
        // Apply context-specific weights
        for result in &mut results {
            if let Some(&weight) = context.context_weights.get(&result.source_graph) {
                result.final_score *= weight;
            }

            // Boost results from primary graph
            if result.source_graph == context.primary_graph {
                result.final_score *= 1.1;
            }
        }

        // Sort by final score (descending)
        results.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap());

        // Apply diversity filtering if enabled (ensure results from different graphs)
        if self.config.enable_cross_graph_similarity {
            results = self.apply_diversity_filtering(results);
        }

        Ok(results)
    }

    /// Apply diversity filtering to ensure results from multiple graphs
    fn apply_diversity_filtering(&self, results: Vec<GraphAwareSearchResult>) -> Vec<GraphAwareSearchResult> {
        let mut filtered_results = Vec::new();
        let mut graph_counts: HashMap<String, usize> = HashMap::new();
        let max_per_graph = 3; // Maximum results per graph

        for result in results {
            let count = graph_counts.entry(result.source_graph.clone()).or_insert(0);
            if *count < max_per_graph {
                filtered_results.push(result);
                *count += 1;
            }
        }

        filtered_results
    }

    /// Update graph metadata for optimization
    pub fn update_graph_metadata(&mut self, graph_uri: String, metadata: GraphMetadata) {
        self.graph_metadata.insert(graph_uri, metadata);
    }

    /// Get graph statistics
    pub fn get_graph_stats(&self, graph_uri: &str) -> Option<(usize, Option<&GraphMetadata>)> {
        let size = self.graph_sizes.get(graph_uri).cloned();
        let metadata = self.graph_metadata.get(graph_uri);
        size.map(|s| (s, metadata))
    }

    /// Clear graph caches
    pub fn clear_caches(&mut self) {
        self.resource_graph_map.clear();
        self.graph_metadata.clear();
        self.graph_sizes.clear();
    }

    /// Check if a resource exists in a specific graph
    pub fn resource_in_graph(&self, resource: &str, graph_uri: &str) -> bool {
        self.resource_graph_map
            .get(resource)
            .map(|info| info.graphs.contains(graph_uri))
            .unwrap_or(false)
    }

    /// Get all graphs containing a resource
    pub fn get_resource_graphs(&self, resource: &str) -> Option<&HashSet<String>> {
        self.resource_graph_map
            .get(resource)
            .map(|info| &info.graphs)
    }

    /// Calculate cross-graph similarity
    pub fn cross_graph_similarity(
        &self,
        vector_store: &VectorStore,
        resource1: &str,
        graph1: &str,
        resource2: &str,
        graph2: &str,
    ) -> Result<f32> {
        if !self.config.enable_cross_graph_similarity {
            return Err(anyhow!("Cross-graph similarity is disabled"));
        }

        // Verify resources exist in specified graphs
        if !self.resource_in_graph(resource1, graph1) || !self.resource_in_graph(resource2, graph2) {
            return Err(anyhow!("Resources not found in specified graphs"));
        }

        // Calculate base similarity
        let base_similarity = vector_store.calculate_similarity(resource1, resource2)?;

        // Apply cross-graph penalty/boost based on graph relationship
        let graph_relationship_factor = self.calculate_graph_relationship_factor(graph1, graph2);

        Ok(base_similarity * graph_relationship_factor)
    }

    /// Calculate relationship factor between two graphs
    fn calculate_graph_relationship_factor(&self, graph1: &str, graph2: &str) -> f32 {
        if graph1 == graph2 {
            return 1.0; // Same graph, no adjustment
        }

        // Check if graphs are in parent-child relationship
        if let Some(children) = self.config.graph_hierarchy.parent_child.get(graph1) {
            if children.contains(&graph2.to_string()) {
                return 0.9; // Parent-child relationship, slight boost
            }
        }

        if let Some(children) = self.config.graph_hierarchy.parent_child.get(graph2) {
            if children.contains(&graph1.to_string()) {
                return 0.9; // Child-parent relationship, slight boost
            }
        }

        // Check if graphs are of the same type
        if let (Some(type1), Some(type2)) = (
            self.config.graph_hierarchy.graph_types.get(graph1),
            self.config.graph_hierarchy.graph_types.get(graph2)
        ) {
            if type1 == type2 {
                return 0.8; // Same type, moderate boost
            }
        }

        0.7 // Different, unrelated graphs - apply penalty
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_context_creation() {
        let context = GraphContext {
            primary_graph: "http://example.org/graph1".to_string(),
            additional_graphs: vec!["http://example.org/graph2".to_string()],
            scope: GraphSearchScope::IncludeChildren,
            context_weights: HashMap::new(),
        };

        assert_eq!(context.primary_graph, "http://example.org/graph1");
        assert_eq!(context.scope, GraphSearchScope::IncludeChildren);
    }

    #[test]
    fn test_resource_graph_registration() {
        let mut search = GraphAwareSearch::new(GraphAwareConfig::default());

        search.register_resource_graph(
            "http://example.org/resource1".to_string(),
            vec!["http://example.org/graph1".to_string()],
        );

        assert!(search.resource_in_graph("http://example.org/resource1", "http://example.org/graph1"));
        assert!(!search.resource_in_graph("http://example.org/resource1", "http://example.org/graph2"));
    }

    #[test]
    fn test_graph_hierarchy() {
        let mut config = GraphAwareConfig::default();
        config.graph_hierarchy.parent_child.insert(
            "http://example.org/parent".to_string(),
            vec!["http://example.org/child1".to_string(), "http://example.org/child2".to_string()],
        );

        let search = GraphAwareSearch::new(config);
        let branch = search.get_hierarchy_branch("http://example.org/parent");

        assert!(branch.contains(&"http://example.org/child1".to_string()));
        assert!(branch.contains(&"http://example.org/child2".to_string()));
    }

    #[test]
    fn test_graph_search_scope() {
        let context = GraphContext {
            primary_graph: "http://example.org/main".to_string(),
            additional_graphs: vec![],
            scope: GraphSearchScope::Exact,
            context_weights: HashMap::new(),
        };

        let search = GraphAwareSearch::new(GraphAwareConfig::default());
        let graphs = search.resolve_search_graphs(&context).unwrap();

        assert_eq!(graphs.len(), 1);
        assert_eq!(graphs[0], "http://example.org/main");
    }
}