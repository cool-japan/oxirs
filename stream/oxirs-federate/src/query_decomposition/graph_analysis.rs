//! Query graph analysis and component finding
//!
//! This module contains functionality for building query graphs, analyzing their structure,
//! and finding connected components for decomposition.

use anyhow::Result;
use petgraph::graph::NodeIndex;
use std::collections::{HashMap, HashSet, VecDeque};
use tracing::debug;

use crate::planner::{TriplePattern};
use crate::planner::planning::types::QueryInfo as PlanningQueryInfo;

use super::types::*;

impl QueryDecomposer {
    /// Build a graph representation of the query
    pub fn build_query_graph(&self, query_info: &PlanningQueryInfo) -> Result<QueryGraph> {
        let mut graph = QueryGraph::new();
        let mut variable_nodes: HashMap<String, NodeIndex> = HashMap::new();
        let mut pattern_nodes: Vec<NodeIndex> = Vec::new();

        // Add nodes for each variable
        for var in &query_info.variables {
            let node_idx = graph.add_variable_node(var.clone());
            variable_nodes.insert(var.clone(), node_idx);
        }

        // Add nodes for each triple pattern and connect to variables
        for (i, pattern) in query_info.patterns.iter().enumerate() {
            let pattern_node = graph.add_pattern_node(i, pattern.clone());
            pattern_nodes.push(pattern_node);

            // Connect pattern to its variables
            if let Some(ref subject) = pattern.subject {
                if subject.starts_with('?') {
                    if let Some(&var_node) = variable_nodes.get(subject) {
                        graph.connect_pattern_to_variable(
                            pattern_node,
                            var_node,
                            VariableRole::Subject,
                        );
                    }
                }
            }
            if let Some(ref predicate) = pattern.predicate {
                if predicate.starts_with('?') {
                    if let Some(&var_node) = variable_nodes.get(predicate) {
                        graph.connect_pattern_to_variable(
                            pattern_node,
                            var_node,
                            VariableRole::Predicate,
                        );
                    }
                }
            }
            if let Some(ref object) = pattern.object {
                if object.starts_with('?') {
                    if let Some(&var_node) = variable_nodes.get(object) {
                        graph.connect_pattern_to_variable(
                            pattern_node,
                            var_node,
                            VariableRole::Object,
                        );
                    }
                }
            }
        }

        // Add filter nodes and dependencies
        for filter in &query_info.filters {
            let filter_node = graph.add_filter_node(filter.clone());

            // Connect filter to patterns that provide its variables
            for var in &filter.variables {
                if let Some(&var_node) = variable_nodes.get(var) {
                    // Find patterns that bind this variable
                    for &pattern_node in &pattern_nodes {
                        if graph.pattern_binds_variable(pattern_node, var_node) {
                            graph.connect_filter_to_pattern(filter_node, pattern_node);
                        }
                    }
                }
            }
        }

        Ok(graph)
    }

    /// Find connected components in the query graph
    pub fn find_connected_components(&self, graph: &QueryGraph) -> Vec<QueryComponent> {
        let mut components = Vec::new();
        let mut visited = HashSet::new();

        for node in graph.pattern_nodes() {
            if !visited.contains(node) {
                let component = self.explore_component(graph, *node, &mut visited);
                components.push(component);
            }
        }

        debug!("Found {} connected components", components.len());
        components
    }

    /// Explore a connected component starting from a node
    pub fn explore_component(
        &self,
        graph: &QueryGraph,
        start: NodeIndex,
        visited: &mut HashSet<NodeIndex>,
    ) -> QueryComponent {
        let mut component = QueryComponent::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            if visited.insert(node) {
                match graph.node_type(node) {
                    Some(QueryNode::Pattern(idx, pattern)) => {
                        component.patterns.push((*idx, pattern.clone()));
                    }
                    Some(QueryNode::Variable(var)) => {
                        component.variables.insert(var.clone());
                    }
                    Some(QueryNode::Filter(filter)) => {
                        component.filters.push(filter.clone());
                    }
                    None => {}
                }

                // Add connected nodes to queue
                for neighbor in graph.neighbors(node) {
                    if !visited.contains(&neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        component
    }

    /// Analyze graph structure for optimization opportunities
    pub fn analyze_graph_structure(&self, graph: &QueryGraph) -> GraphStructureAnalysis {
        let mut analysis = GraphStructureAnalysis::new();

        // Count different types of nodes
        analysis.variable_count = graph.variable_nodes.len();
        analysis.pattern_count = graph.pattern_nodes.len();
        analysis.filter_count = graph.filter_nodes.len();

        // Analyze connectivity patterns
        analysis.connectivity_analysis = self.analyze_connectivity(graph);

        // Detect common patterns
        analysis.detected_patterns = self.detect_query_patterns(graph);

        analysis
    }

    /// Analyze connectivity in the graph
    fn analyze_connectivity(&self, graph: &QueryGraph) -> ConnectivityAnalysis {
        let mut analysis = ConnectivityAnalysis::new();

        // Calculate node degrees
        for &node in graph.pattern_nodes() {
            let degree = graph.neighbors(node).count();
            analysis.node_degrees.insert(node, degree);
            analysis.max_degree = analysis.max_degree.max(degree);
        }

        // Detect hub nodes (high-degree nodes)
        let threshold = (analysis.max_degree as f64 * 0.7) as usize;
        for (&node, &degree) in &analysis.node_degrees {
            if degree >= threshold {
                analysis.hub_nodes.push(node);
            }
        }

        analysis
    }

    /// Detect common query patterns
    fn detect_query_patterns(&self, graph: &QueryGraph) -> Vec<DetectedPattern> {
        let mut patterns = Vec::new();

        // Detect star patterns
        if let Some(star) = self.detect_star_pattern(graph) {
            patterns.push(DetectedPattern::Star(star));
        }

        // Detect chain patterns
        if let Some(chain) = self.detect_chain_pattern(graph) {
            patterns.push(DetectedPattern::Chain(chain));
        }

        // Detect cyclic patterns
        if self.detect_cycles(graph) {
            patterns.push(DetectedPattern::Cycle);
        }

        patterns
    }

    /// Detect star join patterns
    fn detect_star_pattern(&self, graph: &QueryGraph) -> Option<StarPattern> {
        // Find nodes with high connectivity (potential star centers)
        for &node in graph.pattern_nodes() {
            let neighbors: Vec<_> = graph.neighbors(node).collect();
            if neighbors.len() >= 3 {
                return Some(StarPattern {
                    center_node: node,
                    spoke_nodes: neighbors,
                });
            }
        }
        None
    }

    /// Detect chain patterns
    fn detect_chain_pattern(&self, graph: &QueryGraph) -> Option<ChainPattern> {
        // Find linear sequences of connected nodes
        for &start_node in graph.pattern_nodes() {
            let neighbors: Vec<_> = graph.neighbors(start_node).collect();
            if neighbors.len() == 1 {
                // Potential start of a chain
                let chain = self.follow_chain(graph, start_node, &mut HashSet::new());
                if chain.len() >= 3 {
                    return Some(ChainPattern { nodes: chain });
                }
            }
        }
        None
    }

    /// Follow a chain of connected nodes
    fn follow_chain(
        &self,
        graph: &QueryGraph,
        current: NodeIndex,
        visited: &mut HashSet<NodeIndex>,
    ) -> Vec<NodeIndex> {
        let mut chain = vec![current];
        visited.insert(current);

        let neighbors: Vec<_> = graph
            .neighbors(current)
            .filter(|&n| !visited.contains(&n))
            .collect();

        if neighbors.len() == 1 {
            let next = neighbors[0];
            let next_neighbors: Vec<_> = graph
                .neighbors(next)
                .filter(|&n| !visited.contains(&n))
                .collect();

            if next_neighbors.len() <= 1 {
                chain.extend(self.follow_chain(graph, next, visited));
            }
        }

        chain
    }

    /// Detect cycles in the graph
    fn detect_cycles(&self, _graph: &QueryGraph) -> bool {
        // Simplified cycle detection - would use proper graph algorithms
        false
    }
}

/// Analysis of graph structure
#[derive(Debug)]
pub struct GraphStructureAnalysis {
    pub variable_count: usize,
    pub pattern_count: usize,
    pub filter_count: usize,
    pub connectivity_analysis: ConnectivityAnalysis,
    pub detected_patterns: Vec<DetectedPattern>,
}

impl GraphStructureAnalysis {
    pub fn new() -> Self {
        Self {
            variable_count: 0,
            pattern_count: 0,
            filter_count: 0,
            connectivity_analysis: ConnectivityAnalysis::new(),
            detected_patterns: Vec::new(),
        }
    }
}

/// Connectivity analysis results
#[derive(Debug)]
pub struct ConnectivityAnalysis {
    pub node_degrees: HashMap<NodeIndex, usize>,
    pub max_degree: usize,
    pub hub_nodes: Vec<NodeIndex>,
}

impl ConnectivityAnalysis {
    pub fn new() -> Self {
        Self {
            node_degrees: HashMap::new(),
            max_degree: 0,
            hub_nodes: Vec::new(),
        }
    }
}

/// Detected query patterns
#[derive(Debug)]
pub enum DetectedPattern {
    Star(StarPattern),
    Chain(ChainPattern),
    Cycle,
}

/// Star pattern detection result
#[derive(Debug)]
pub struct StarPattern {
    pub center_node: NodeIndex,
    pub spoke_nodes: Vec<NodeIndex>,
}

/// Chain pattern detection result
#[derive(Debug)]
pub struct ChainPattern {
    pub nodes: Vec<NodeIndex>,
}
