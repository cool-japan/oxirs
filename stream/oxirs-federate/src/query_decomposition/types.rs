//! Type definitions for query decomposition
//!
//! This module contains all the type definitions, configuration structs, and data structures
//! used throughout the query decomposition system.

use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Duration;

use crate::{
    planner::{ExecutionPlan, FilterExpression, QueryInfo, TriplePattern},
    FederatedService,
};

/// Advanced query decomposer with optimization algorithms
#[derive(Debug)]
pub struct QueryDecomposer {
    pub config: DecomposerConfig,
    pub cost_estimator: CostEstimator,
}

/// Configuration for the query decomposer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecomposerConfig {
    pub optimization_strategy: OptimizationStrategy,
    pub min_patterns_for_distribution: usize,
    pub max_component_size: usize,
    pub enable_join_optimization: bool,
    pub enable_bloom_filters: bool,
    pub cost_threshold: f64,
    pub parallel_threshold: usize,
}

impl Default for DecomposerConfig {
    fn default() -> Self {
        Self {
            optimization_strategy: OptimizationStrategy::Balanced,
            min_patterns_for_distribution: 3,
            max_component_size: 10,
            enable_join_optimization: true,
            enable_bloom_filters: true,
            cost_threshold: 1000.0,
            parallel_threshold: 2,
        }
    }
}

/// Optimization strategies for query decomposition
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    MinimizeCost,
    MinimizeTime,
    MinimizeTransfer,
    Balanced,
}

/// Result of query decomposition
#[derive(Debug, Clone)]
pub struct DecompositionResult {
    pub plan: ExecutionPlan,
    pub statistics: DecompositionStatistics,
}

/// Statistics about the decomposition process
#[derive(Debug, Clone)]
pub struct DecompositionStatistics {
    pub total_patterns: usize,
    pub components_found: usize,
    pub plans_evaluated: usize,
    pub selected_strategy: PlanStrategy,
    pub estimated_total_cost: f64,
    pub decomposition_time: Duration,
}

/// Graph representation of a query
#[derive(Debug)]
pub struct QueryGraph {
    pub graph: DiGraph<QueryNode, EdgeType>,
    pub variable_nodes: HashMap<String, NodeIndex>,
    pub pattern_nodes: Vec<NodeIndex>,
    pub filter_nodes: Vec<NodeIndex>,
}

impl QueryGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            variable_nodes: HashMap::new(),
            pattern_nodes: Vec::new(),
            filter_nodes: Vec::new(),
        }
    }

    pub fn add_variable_node(&mut self, variable: String) -> NodeIndex {
        let node_idx = self.graph.add_node(QueryNode::Variable(variable.clone()));
        self.variable_nodes.insert(variable, node_idx);
        node_idx
    }

    pub fn add_pattern_node(&mut self, index: usize, pattern: TriplePattern) -> NodeIndex {
        let node_idx = self.graph.add_node(QueryNode::Pattern(index, pattern));
        self.pattern_nodes.push(node_idx);
        node_idx
    }

    pub fn add_filter_node(&mut self, filter: FilterExpression) -> NodeIndex {
        let node_idx = self.graph.add_node(QueryNode::Filter(filter));
        self.filter_nodes.push(node_idx);
        node_idx
    }

    pub fn connect_pattern_to_variable(
        &mut self,
        pattern_node: NodeIndex,
        variable_node: NodeIndex,
        role: VariableRole,
    ) {
        self.graph.add_edge(pattern_node, variable_node, EdgeType::Variable(role));
    }

    pub fn connect_filter_to_pattern(&mut self, filter_node: NodeIndex, pattern_node: NodeIndex) {
        self.graph.add_edge(filter_node, pattern_node, EdgeType::Dependency);
    }

    pub fn pattern_nodes(&self) -> &[NodeIndex] {
        &self.pattern_nodes
    }

    pub fn neighbors(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        self.graph.neighbors(node)
    }

    pub fn node_type(&self, node: NodeIndex) -> Option<&QueryNode> {
        self.graph.node_weight(node)
    }

    pub fn pattern_binds_variable(&self, _pattern_node: NodeIndex, _variable_node: NodeIndex) -> bool {
        // Implementation would check if pattern binds the variable
        true
    }
}

/// Nodes in the query graph
#[derive(Debug, Clone)]
pub enum QueryNode {
    Variable(String),
    Pattern(usize, TriplePattern),
    Filter(FilterExpression),
}

/// Types of nodes in the query graph (for pattern matching)
#[derive(Debug, Clone)]
pub enum NodeType {
    Variable(String),
    Pattern(usize, TriplePattern),
    Filter(FilterExpression),
}

/// Edge types in the query graph
#[derive(Debug, Clone)]
pub enum EdgeType {
    Variable(VariableRole),
    Dependency,
}

/// Role of a variable in a triple pattern
#[derive(Debug, Clone, Copy)]
pub enum VariableRole {
    Subject,
    Predicate,
    Object,
}

/// A connected component of the query
#[derive(Debug, Clone)]
pub struct QueryComponent {
    pub patterns: Vec<(usize, TriplePattern)>,
    pub variables: HashSet<String>,
    pub filters: Vec<FilterExpression>,
}

impl QueryComponent {
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            variables: HashSet::new(),
            filters: Vec::new(),
        }
    }
}

/// A plan for executing a query component
#[derive(Debug, Clone)]
pub struct ComponentPlan {
    pub strategy: PlanStrategy,
    pub steps: Vec<PlanStep>,
    pub total_cost: f64,
    pub requires_join: bool,
}

/// Strategy used for plan generation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanStrategy {
    SingleService,
    EvenDistribution,
    MinimizeIntermediate,
    MaximizeParallel,
    SpecializedServices,
    JoinAware,
    CostBased,
    PatternBased,
    StarJoinOptimized,
    MinimizeIntermediateAdvanced,
    MaximizeParallelAdvanced,
}

/// A single step in a component plan
#[derive(Debug, Clone)]
pub struct PlanStep {
    pub service_id: String,
    pub patterns: Vec<(usize, TriplePattern)>,
    pub filters: Vec<FilterExpression>,
    pub estimated_cost: f64,
    pub estimated_results: u64,
}

/// Cost estimator for query execution
#[derive(Debug)]
pub struct CostEstimator {
    base_cost: f64,
    network_cost_factor: f64,
    join_cost_factor: f64,
}

impl CostEstimator {
    pub fn new() -> Self {
        Self {
            base_cost: 100.0,
            network_cost_factor: 10.0,
            join_cost_factor: 50.0,
        }
    }

    pub fn estimate_single_service_cost(
        &self,
        _service: &FederatedService,
        component: &QueryComponent,
    ) -> f64 {
        self.base_cost + (component.patterns.len() as f64 * 20.0)
    }

    pub fn estimate_pattern_cost(
        &self,
        _service: &FederatedService,
        patterns: &[(usize, TriplePattern)],
    ) -> f64 {
        self.base_cost + (patterns.len() as f64 * 15.0) + self.network_cost_factor
    }

    pub fn estimate_join_cost(&self, left_size: u64, right_size: u64) -> f64 {
        self.join_cost_factor * ((left_size * right_size) as f64).log10()
    }
}

/// Advanced distribution algorithms
#[derive(Debug, Clone, Copy)]
pub enum DistributionAlgorithm {
    JoinAware,
    CostBased,
    PatternBased,
    StarJoinOptimized,
    MinimizeIntermediate,
    MaximizeParallel,
}

/// Join pattern types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinPattern {
    Star,
    Chain,
    Tree,
    Complex,
}

/// Pattern selectivity analysis
#[derive(Debug, Clone)]
pub struct PatternSelectivity {
    pub pattern_index: usize,
    pub selectivity_score: f64,
    pub variable_count: usize,
    pub constant_count: usize,
}

/// Service affinity analysis
#[derive(Debug, Clone)]
pub struct ServiceAffinity {
    pub service_id: String,
    pub affinity_score: f64,
    pub predicate_matches: Vec<String>,
    pub capability_matches: Vec<String>,
}

/// Bloom filter configuration for optimization
#[derive(Debug, Clone)]
pub struct BloomFilterConfig {
    pub enabled: bool,
    pub expected_elements: u64,
    pub false_positive_rate: f64,
}

impl Default for BloomFilterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            expected_elements: 10000,
            false_positive_rate: 0.01,
        }
    }
}

/// Parallelization analysis
#[derive(Debug, Clone)]
pub struct ParallelizationAnalysis {
    pub independent_groups: Vec<Vec<usize>>,
    pub dependency_graph: HashMap<usize, Vec<usize>>,
    pub parallel_potential: f64,
}

/// Network optimization configuration
#[derive(Debug, Clone)]
pub struct NetworkOptimization {
    pub minimize_round_trips: bool,
    pub batch_size_threshold: usize,
    pub compression_enabled: bool,
    pub connection_pooling: bool,
}

impl Default for NetworkOptimization {
    fn default() -> Self {
        Self {
            minimize_round_trips: true,
            batch_size_threshold: 100,
            compression_enabled: true,
            connection_pooling: true,
        }
    }
}